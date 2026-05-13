"""
KVRHook - Injects KVR (Window + Retrieval) attention into a transformer model
via forward hooks. Lazy retrieval: only built when context > window_size.
No LLaMA-specific imports - compatible with any HuggingFace model.
"""
import torch
import torch.nn.functional as F
from .kvr_window import WindowBuffer
from .kvr_retrieval import RetrievalIndex, _apply_rotary, _reverse_rotary


class KVRHook:
    def __init__(self, model, window_size=2048, top_k=128, ret_weight=1.0,
                 device=None, **kwargs):
        self.model = model
        self.device = device or model.device
        cfg = model.config
        self.n_layers = cfg.num_hidden_layers
        self.n_kv = cfg.num_key_value_heads
        self.n_q = cfg.num_attention_heads
        self.d_head = getattr(cfg, 'head_dim', None) or (cfg.hidden_size // self.n_q)
        self.g = self.n_q // self.n_kv
        self.window_size = window_size
        self.top_k = top_k
        self._ret_weight = ret_weight

        self._context_len = 0
        self._prefill_done = False
        self._hooks = []
        self._step = 0
        self._retrieval_built = False

        self.windows = [WindowBuffer(self.n_kv, self.d_head, window_size, self.device)
                        for _ in range(self.n_layers)]
        self.retrievals = [RetrievalIndex(self.n_kv, self.d_head, top_k, device=self.device)
                           for _ in range(self.n_layers)]

        max_pos = cfg.max_position_embeddings
        first_attn = self.model.model.layers[0].self_attn
        dummy_q = torch.empty(1, max_pos, self.n_q, self.d_head, device=self.device)
        all_pos = torch.arange(max_pos, device=self.device).unsqueeze(0)
        cos_full, sin_full = first_attn.rotary_emb(dummy_q, position_ids=all_pos)
        self.cos_tbl = cos_full[0]
        self.sin_tbl = sin_full[0]

    def _ensure_retrieval_built(self, li):
        if self._retrieval_built and self.retrievals[li].n_stored > 0:
            return
        win = self.windows[li]
        if win.n < win.cap:
            return

        k_post_all = win.k[:win.n].float().reshape(win.n, self.n_kv, self.d_head)
        v_all = win.v[:win.n].float()
        pos_ids = torch.arange(win.n, device=self.device)
        c = self.cos_tbl[pos_ids].to(k_post_all.dtype)
        s = self.sin_tbl[pos_ids].to(k_post_all.dtype)
        ck = c.unsqueeze(1).expand(-1, self.n_kv, -1).reshape(-1, self.d_head)
        sk = s.unsqueeze(1).expand(-1, self.n_kv, -1).reshape(-1, self.d_head)
        k_pre_all = _reverse_rotary(k_post_all.reshape(-1, self.d_head), ck, sk)
        k_pre_all = k_pre_all.view(-1, self.n_kv, self.d_head)

        self.retrievals[li].set_rotary_tables(self.cos_tbl, self.sin_tbl)
        self.retrievals[li].calibrate(k_pre_all, v_all)
        self.retrievals[li].batch_append(k_pre_all, v_all)

    def _build_all_retrievals(self):
        if self._retrieval_built:
            return
        for li in range(self.n_layers):
            self._ensure_retrieval_built(li)
        self._retrieval_built = True

    @torch.no_grad()
    def prefill(self, input_ids):
        """Hook-based prefill: model forward with hooks to capture exact K_pre.
        Produces exact hidden states (flash attention), no output_hidden_states overhead.
        """
        n_prompt = input_ids.shape[1]

        # Register hooks to capture K_pre/V during forward
        captured_k, captured_v = {}, {}
        prefill_hooks = []
        for li in range(self.n_layers):
            attn = self.model.model.layers[li].self_attn
            def make_hook(li, ck, cv):
                def hook_fn(mod, args, kwargs, output):
                    hidden = kwargs['hidden_states']
                    kp = mod.k_proj(hidden).float().view(-1, self.n_kv, self.d_head)
                    vv = mod.v_proj(hidden).float().view(-1, self.n_kv, self.d_head)
                    ck[li] = kp; cv[li] = vv
                return hook_fn
            h = attn.register_forward_hook(make_hook(li, captured_k, captured_v), with_kwargs=True)
            prefill_hooks.append(h)

        self.model(input_ids)

        for h in prefill_hooks:
            h.remove()

        # Compute K_post for window from captured K_pre
        for li in range(self.n_layers):
            k_pre = captured_k[li]; v = captured_v[li]
            c = self.cos_tbl[:n_prompt].unsqueeze(1).expand(-1, self.n_kv, -1).reshape(-1, self.d_head)
            s = self.sin_tbl[:n_prompt].unsqueeze(1).expand(-1, self.n_kv, -1).reshape(-1, self.d_head)
            k_post = _apply_rotary(k_pre.reshape(-1, self.d_head), c, s)
            k_post = k_post.view(n_prompt, self.n_kv, self.d_head)

            self.retrievals[li].set_rotary_tables(self.cos_tbl, self.sin_tbl)
            self.windows[li].batch_append(k_post, v)

        self._context_len = n_prompt
        self._prefill_done = True

        # Build retrieval eagerly if prompt exceeds window
        if n_prompt > self.window_size:
            n_ret = n_prompt - self.window_size
            for li in range(self.n_layers):
                k_pre = captured_k[li]; v = captured_v[li]
                self.retrievals[li].calibrate(k_pre, v)
                self.retrievals[li].batch_append(k_pre[:n_ret], v[:n_ret])
            self._retrieval_built = True

    def _update_stores(self, li, k_pre, v_val, k_post):
        win = self.windows[li]
        if self._retrieval_built and win.n >= win.cap:
            slot = win.pos % win.cap
            old_k_post = win.k[slot].float()
            old_v = win.v[slot].float()
            c = self.cos_tbl[win.pos:win.pos+1].to(old_k_post.dtype)
            s = self.sin_tbl[win.pos:win.pos+1].to(old_k_post.dtype)
            old_k_pre = _reverse_rotary(old_k_post.unsqueeze(0), c, s)[0]
            self.retrievals[li].append(old_k_pre, old_v)
        elif not self._retrieval_built and self._context_len >= self.window_size - 1 and li == 0:
            self._build_all_retrievals()
        win.append(k_post, v_val)

    def register(self):
        for li in range(self.n_layers):
            attn_mod = self.model.model.layers[li].self_attn
            hook = attn_mod.register_forward_hook(
                self._make_hook(li), with_kwargs=True)
            self._hooks.append(hook)

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _make_hook(self, li):
        attn_mod = self.model.model.layers[li].self_attn

        def hook(mod, args, kwargs, output):
            hidden = kwargs['hidden_states']
            pos_ids = kwargs.get('position_ids', None)

            q_all = mod.q_proj(hidden).float().view(1, hidden.shape[1], self.n_q, self.d_head)
            k_all = mod.k_proj(hidden).float().view(1, hidden.shape[1], self.n_kv, self.d_head)
            v_all = mod.v_proj(hidden).float()

            cos, sin = mod.rotary_emb(q_all, position_ids=pos_ids)
            c = cos[0]; s = sin[0]
            cq = c.unsqueeze(1).expand(-1, self.n_q, -1).reshape(-1, self.d_head)
            sq = s.unsqueeze(1).expand(-1, self.n_q, -1).reshape(-1, self.d_head)
            ck = c.unsqueeze(1).expand(-1, self.n_kv, -1).reshape(-1, self.d_head)
            sk = s.unsqueeze(1).expand(-1, self.n_kv, -1).reshape(-1, self.d_head)
            q_rot = _apply_rotary(q_all.reshape(-1, self.d_head), cq, sq).view_as(q_all)
            k_rot = _apply_rotary(k_all.reshape(-1, self.d_head), ck, sk).view_as(k_all)

            q_last = q_rot[0, -1, :, :]
            k_post = k_rot[0, -1, :, :]
            k_pre = k_all[0, -1, :, :]
            v_last = v_all.view(1, hidden.shape[1], self.n_kv, self.d_head)[0, -1, :, :]

            if self._step > 0:
                self._update_stores(li, k_pre, v_last, k_post)

            # Window K/V (fp16, exact)
            win = self.windows[li]
            nw = win.n
            win_k = win.k[:nw].float()
            win_v = win.v[:nw].float()

            # Retrieval K/V (lazy: only if built and window is full)
            ret = self.retrievals[li]
            if self._ret_weight > 0 and self._retrieval_built and ret.n_stored > 0:
                n_stored = ret.n_stored
                exc_start = max(0, n_stored - nw)
                exc_end = n_stored
                ret_k, ret_v = ret.retrieve_topk(q_last, n_q=self.n_q,
                                                  exclude_start=exc_start, exclude_end=exc_end)
            else:
                ret_k = torch.zeros(0, self.n_kv, self.d_head, device=self.device, dtype=torch.float32)
                ret_v = torch.zeros(0, self.n_kv, self.d_head, device=self.device, dtype=torch.float32)

            k_cat = torch.cat([win_k, ret_k], dim=0)
            v_cat = torch.cat([win_v, ret_v], dim=0)
            d = self.d_head
            scale = d ** 0.5

            qg = q_last.float().view(self.n_kv, self.g, d)
            scores = torch.einsum('hgd, thd -> h g t', qg, k_cat.float())
            w = F.softmax(scores / scale, dim=-1)
            out = torch.einsum('h g t, t h d -> h g d', w, v_cat.float()).reshape(self.n_q, d)

            fwr_proj = mod.o_proj(out.half().reshape(1, -1))

            new_attn = output[0].clone()
            new_attn[:, -1, :] = fwr_proj
            extra = tuple(output[i] for i in range(1, len(output)))
            return (new_attn,) + extra if extra else (new_attn,)

        return hook

    def reset(self):
        self._context_len = 0
        self._prefill_done = False
        self._step = 0
        self._retrieval_built = False
        self.remove()
        self.windows = [WindowBuffer(self.n_kv, self.d_head, self.window_size, self.device)
                        for _ in range(self.n_layers)]
        self.retrievals = [RetrievalIndex(self.n_kv, self.d_head, self.top_k, device=self.device)
                           for _ in range(self.n_layers)]

    def __repr__(self):
        ret_state = "ready" if self._retrieval_built else "lazy"
        return (f"KVRHook(layers={self.n_layers}, window={self.window_size}, "
                f"top_k={self.top_k}, ctx={self._context_len}, ret={ret_state})")
