"""
RetrievalIndex — int4 K_pre + W matrix + int2 V_residual.
Stores: int4 K_pre (search + V prediction via W)
        int2 V_residual (corrects W's error)
No fp16 V storage. V = W(K_pre) + dequantize(V_residual).
"""
import torch
import torch.nn.functional as F


def _rotate_half(x):
    d2 = x.shape[-1] // 2
    return torch.cat([-x[..., d2:], x[..., :d2]], dim=-1)


def _apply_rotary(x, cos, sin):
    return x * cos[:x.shape[0]] + _rotate_half(x) * sin[:x.shape[0]]


def _reverse_rotary(x, cos, sin):
    """Inverse RoPE: apply rotation with negative sin (sin(-θ) = -sin(θ))."""
    return x * cos[:x.shape[0]] - _rotate_half(x) * sin[:x.shape[0]]


class RetrievalIndex:
    def __init__(self, n_kv, d_head, top_k=128, k_bits=4, v_res_bits=2, device='cuda'):
        self.n_kv = n_kv
        self.d_head = d_head
        self.kv_dim = n_kv * d_head
        self.top_k = top_k
        self.k_bits = k_bits
        self.v_res_bits = v_res_bits
        self.k_nlv = 2 ** k_bits
        self.k_half = self.k_nlv // 2
        self.vr_nlv = 2 ** v_res_bits
        self.vr_half = self.vr_nlv // 2
        self.device = device
        self.k_scales = None
        self.k_codes = None
        self.vr_scales = None
        self.vr_codes = None
        self.n_stored = 0
        self.W = None
        self.K_mean = None
        self.V_mean = None
        self.cos = None
        self.sin = None

    def set_rotary_tables(self, cos, sin):
        self.cos = cos
        self.sin = sin

    @torch.no_grad()
    def calibrate(self, k_pre_sample, v_sample=None):
        """Fit W: K_pre → V, compute residual scales. k_pre_sample: (n, n_kv, d)."""
        n = k_pre_sample.shape[0]
        k2 = k_pre_sample.reshape(n, self.kv_dim).float()
        self.k_scales = k_pre_sample.abs().max(dim=0).values.clamp(min=1e-8)
        self.k_codes = torch.zeros(0, self.n_kv, self.d_head, device=self.device, dtype=torch.int8)
        self.vr_codes = torch.zeros(0, self.n_kv, self.d_head, device=self.device, dtype=torch.int8)

        if v_sample is not None and n > 10:
            half = n // 2
            v2 = v_sample.reshape(n, self.kv_dim).float()
            k_tr, v_tr = k2[:half], v2[:half]
            self.K_mean = k_tr.mean(0, keepdim=True)
            self.V_mean = v_tr.mean(0, keepdim=True)
            kc = k_tr - self.K_mean; vc = v_tr - self.V_mean
            self.W = (kc.T @ vc).T @ torch.linalg.pinv(kc.T @ kc)

            # Pre-compute residual scales from calibration data
            v_pred_all = (k2 - self.K_mean) @ self.W.T + self.V_mean
            v_res = v2 - v_pred_all
            vr_shaped = v_res.view(-1, self.n_kv, self.d_head)
            self.vr_scales = vr_shaped.abs().amax(dim=(0, 2)).unsqueeze(1).clamp(min=1e-8)
        else:
            self.W = torch.eye(self.kv_dim, device=self.device)
            self.K_mean = torch.zeros(1, self.kv_dim, device=self.device)
            self.V_mean = torch.zeros(1, self.kv_dim, device=self.device)
            self.vr_scales = torch.ones(self.n_kv, 1, device=self.device)
        self.n_stored = 0

    def _deq_k(self, kvh):
        """Dequantize K_pre for one KV head: codes → (n_stored, d)."""
        step = 2 * self.k_scales / self.k_nlv
        return self.k_codes[:, kvh, :].float() * step[kvh].float()

    def _rotary(self, k_pre_2d, indices):
        if self.cos is None:
            return k_pre_2d
        return _apply_rotary(k_pre_2d,
                             self.cos[indices].to(k_pre_2d.dtype),
                             self.sin[indices].to(k_pre_2d.dtype))

    def _predict_v(self, k_pre_flat):
        if self.W is None:
            return torch.zeros_like(k_pre_flat)
        return (k_pre_flat - self.K_mean) @ self.W.T + self.V_mean

    def _reconstruct_v(self, k_pre_top, top_idx):
        """V = W(K_pre) + dequantize(V_residual). k_pre_top: (k, n_kv, d)."""
        kv_flat = k_pre_top.reshape(-1, self.kv_dim)
        v_base = self._predict_v(kv_flat).reshape(-1, self.n_kv, self.d_head)
        vr_step = 2 * self.vr_scales / self.vr_nlv
        vr_dq = self.vr_codes[top_idx].float() * vr_step.unsqueeze(0)
        return (v_base + vr_dq).float()

    @torch.no_grad()
    def append(self, k_pre_rope, v=None):
        if self.k_scales is None:
            raise RuntimeError("Call calibrate() first")

        k_step = 2 * self.k_scales / self.k_nlv
        kq = torch.round(k_pre_rope.float() / k_step.float()).clamp(-self.k_half, self.k_half - 1).to(torch.int8)
        self.k_codes = torch.cat([self.k_codes, kq.unsqueeze(0)])

        if v is not None and self.W is not None:
            kf = k_pre_rope.float().reshape(-1)
            v_pred = self._predict_v(kf.unsqueeze(0)).reshape(self.n_kv, self.d_head)
            v_res = v.float() - v_pred
            vr_step = 2 * self.vr_scales / self.vr_nlv
            vrq = torch.round(v_res / vr_step).clamp(-self.vr_half, self.vr_half - 1).to(torch.int8)
            self.vr_codes = torch.cat([self.vr_codes, vrq.unsqueeze(0)])
        else:
            self.vr_codes = torch.cat([self.vr_codes, torch.zeros(1, self.n_kv, self.d_head, device=self.device, dtype=torch.int8)])

        self.n_stored += 1

    @torch.no_grad()
    def batch_append(self, k_pre_all, v_all):
        """For prefill: store all prompt tokens at once (no torch.cat)."""
        n = k_pre_all.shape[0]
        k_step = 2 * self.k_scales / self.k_nlv
        self.k_codes = torch.round(k_pre_all.float() / k_step).clamp(-self.k_half, self.k_half - 1).to(torch.int8)

        if self.W is not None:
            k_flat = k_pre_all.reshape(n, -1).float()
            v_pred = self._predict_v(k_flat).reshape(n, self.n_kv, self.d_head)
            v_res = v_all.float() - v_pred
            vr_step = 2 * self.vr_scales / self.vr_nlv
            self.vr_codes = torch.round(v_res.float() / vr_step).clamp(-self.vr_half, self.vr_half - 1).to(torch.int8)
        else:
            self.vr_codes = torch.zeros(n, self.n_kv, self.d_head, device=self.device, dtype=torch.int8)

        self.n_stored = n

    @torch.no_grad()
    def retrieve(self, q_post_rope, n_q=None):
        n_q = n_q or q_post_rope.shape[0]
        g = n_q // self.n_kv
        d = self.d_head; scale = d ** 0.5
        out = torch.zeros(n_q, d, device=q_post_rope.device, dtype=torch.float32)
        nr = self.n_stored
        if nr == 0:
            return out
        top_k = min(self.top_k, nr)
        all_idx = torch.arange(nr, device=self.device)

        for hi in range(n_q):
            kvh = hi // g; qh = q_post_rope[hi]
            k_pre = self._deq_k(kvh)
            k_post = self._rotary(k_pre, all_idx)
            sim = (qh.unsqueeze(0) @ k_post.T).squeeze(0) / scale
            tidx = sim.argsort(descending=True)[:top_k]
            k_top = self._deq_k(kvh)[tidx]
            k_top_post = self._rotary(k_top, tidx)
            k_top_pre_shape = self.k_codes[tidx].float() * (2 * self.k_scales / self.k_nlv).unsqueeze(0)
            v_top = self._reconstruct_v(k_top_pre_shape, tidx)[:, kvh, :]
            s = (qh.unsqueeze(0) @ k_top_post.T).squeeze(0) / scale
            out[hi] = F.softmax(s, dim=-1) @ v_top
        return out

    @torch.no_grad()
    def retrieve_topk(self, q_post_rope, n_q=None, exclude_start=0, exclude_end=0):
        n_q = n_q or q_post_rope.shape[0]
        g = n_q // self.n_kv
        d = self.d_head; scale = d ** 0.5
        nr = self.n_stored
        empty = (torch.zeros(0, self.n_kv, d, device=self.device, dtype=torch.float16),
                 torch.zeros(0, self.n_kv, d, device=self.device, dtype=torch.float16))
        if nr == 0 or exclude_start >= nr:
            return empty
        if exclude_start <= 0 and exclude_end >= nr:
            return empty

        top_k = min(self.top_k, nr)
        all_idx = torch.arange(nr, device=self.device)

        collected_k, collected_v = [], []
        for kvh in range(self.n_kv):
            k_pre = self._deq_k(kvh)
            k_post = self._rotary(k_pre, all_idx)

            all_sim = []
            for qg_idx in range(g):
                hi = kvh * g + qg_idx
                qh = q_post_rope[hi]
                sim = (qh.unsqueeze(0) @ k_post.T).squeeze(0) / scale
                all_sim.append(sim)
            sim = torch.stack(all_sim, dim=0).mean(dim=0)
            sim[exclude_start:exclude_end] = -float('inf')
            tidx = sim.argsort(descending=True)[:top_k]

            k_top = self._deq_k(kvh)[tidx]
            collected_k.append(self._rotary(k_top, tidx))

            k_top_pre = self.k_codes[tidx].float() * (2 * self.k_scales / self.k_nlv).unsqueeze(0)
            collected_v.append(self._reconstruct_v(k_top_pre, tidx)[:, kvh, :])

        return (torch.stack(collected_k, dim=1).contiguous(),
                torch.stack(collected_v, dim=1).contiguous().half())
