"""
KVRGenerator — Incremental generation with KVR attention.
Own generate loop, no model.generate() overhead.
Only O(2048+128) attention per step, O(1) MLP per step (single token).
"""
import torch
import torch.nn.functional as F
from transformers import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper
from transformers import RepetitionPenaltyLogitsProcessor
from .kvr_hook import KVRHook
from .kvr_retrieval import _apply_rotary
from .kvr_triton import run_fused_attn


class KVRGenerator:
    def __init__(self, model, window_size=2048, top_k=128, device=None):
        self.model = model
        self.device = device or model.device
        cfg = model.config
        self.n_layers = cfg.num_hidden_layers
        self.n_kv = cfg.num_key_value_heads
        self.n_q = cfg.num_attention_heads
        self.d_head = getattr(cfg, 'head_dim', None) or (cfg.hidden_size // self.n_q)
        self.hidden_size = cfg.hidden_size
        self.g = self.n_q // self.n_kv
        self.vocab_size = cfg.vocab_size
        self._step_count = 0
        self._generated_ids = []

        # KVR indices (shared)
        self.kvr = KVRHook(model, window_size=window_size, top_k=top_k, device=device)
        self.window_size = window_size
        self.top_k = top_k

        # Model components needed for incremental step
        self.embed = model.model.embed_tokens
        self.final_norm = model.model.norm
        self.lm_head = model.lm_head
        self.layers = model.model.layers

    @torch.no_grad()
    def prefill(self, input_ids):
        """Full forward to build window + retrieval indices."""
        self.kvr.prefill(input_ids)
        self._generated_ids = []
        self._step_count = 0
        self._is_first = True

    @torch.no_grad()
    def step(self, token_id=None, temperature=1.0, top_k=0, top_p=1.0,
             repetition_penalty=1.0):
        """Generate one token incrementally.

        Args:
            token_id: If None, take last generated token. For first step,
                      must pass the last prompt token's ID.
            temperature: sampling temperature.
            top_k: top-k sampling (0 = off).
            top_p: nucleus sampling (1.0 = off).
        Returns:
            next_token_id: (1,) tensor.
        """
        if token_id is None:
            if not self._generated_ids:
                raise ValueError("First step requires token_id")
            token_id = self._generated_ids[-1]

        if isinstance(token_id, (int,)):
            token_id = torch.tensor([[token_id]], device=self.device)
        elif token_id.dim() == 0:
            token_id = token_id.unsqueeze(0).unsqueeze(0)

        # Embed
        h = self.embed(token_id)[0, 0]  # (hidden_size,)

        pos_id = self.kvr._context_len

        for li in range(self.n_layers):
            layer = self.layers[li]
            attn = layer.self_attn

            # Layernorm
            h_norm = layer.input_layernorm(h)

            # Q/K/V projections
            q = attn.q_proj(h_norm).view(self.n_q, self.d_head)
            k_pre = attn.k_proj(h_norm).view(self.n_kv, self.d_head)
            v = attn.v_proj(h_norm).view(self.n_kv, self.d_head)

            # RoPE
            c = self.kvr.cos_tbl[pos_id:pos_id+1].to(h.dtype)
            s = self.kvr.sin_tbl[pos_id:pos_id+1].to(h.dtype)
            q_rot = _apply_rotary(q.unsqueeze(0), c, s)[0]
            k_rot = _apply_rotary(k_pre.unsqueeze(0), c, s)[0]

            # Store K/V for NEW tokens
            if not self._is_first:
                self.kvr._update_stores(li, k_pre.to(torch.float16), v.to(torch.float16), k_rot.to(torch.float16))

            # Query window
            win = self.kvr.windows[li]
            nw = win.n
            win_k = win.k[:nw].to(q_rot.dtype)
            win_v = win.v[:nw].to(q_rot.dtype)

            # Query retrieval (excluding window positions)
            ret = self.kvr.retrievals[li]
            n_stored = ret.n_stored
            excl_s = max(0, n_stored - nw)
            excl_e = n_stored
            ret_k, ret_v = ret.retrieve_topk(q_rot.float(), n_q=self.n_q,
                                              exclude_start=excl_s, exclude_end=excl_e)

            k_cat = torch.cat([win_k, ret_k.to(win_k.dtype)], dim=0)
            v_cat = torch.cat([win_v, ret_v.to(win_v.dtype)], dim=0)
            attn_out = run_fused_attn(q_rot, k_cat, v_cat)

            # o_proj + residual
            h = h + attn.o_proj(attn_out.reshape(-1).half())

            # Post-attention layernorm + MLP + residual
            h_norm2 = layer.post_attention_layernorm(h)
            h = h + layer.mlp(h_norm2)

        # Final norm + LM head
        h_final = self.final_norm(h)
        logits = self.lm_head(h_final).float()  # (vocab_size,)

        # Sampling
        processors = LogitsProcessorList()
        if repetition_penalty != 1.0:
            all_ids = torch.tensor(self._generated_ids, device=self.device).unsqueeze(0)
            from transformers import RepetitionPenaltyLogitsProcessor
            processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        if temperature != 1.0:
            processors.append(TemperatureLogitsWarper(temperature))
        if top_k > 0:
            processors.append(TopKLogitsWarper(top_k))
        if top_p < 1.0:
            processors.append(TopPLogitsWarper(top_p))

        if self._generated_ids:
            input_ids = torch.tensor(self._generated_ids, device=self.device).unsqueeze(0)
        else:
            input_ids = token_id

        for proc in processors:
            logits = proc(input_ids, logits.unsqueeze(0))[0]

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)

        self._generated_ids.append(next_id.item())
        self._step_count += 1
        self.kvr._context_len += 1
        self._is_first = False

        return next_id

    @torch.no_grad()
    def generate(self, max_new_tokens=100, temperature=1.0, top_k=0, top_p=1.0,
                 repetition_penalty=1.0):
        """Generate full sequence. Returns tensor of token IDs."""
        generated = []
        for _ in range(max_new_tokens):
            next_id = self.step(
                temperature=temperature, top_k=top_k, top_p=top_p,
                repetition_penalty=repetition_penalty)
            generated.append(next_id)
        return torch.cat(generated)
