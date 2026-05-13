"""
RetrievalIndex — int4 K_pre + W matrix + int2 V_residual (bit-packed).
Stores K and V residual in packed format (int4 = 2 per byte, int2 = 4 per byte).
V = W(K_pre) + dequantize(V_residual). No fp16 V storage.
"""
import torch
import torch.nn.functional as F


def _rotate_half(x):
    d2 = x.shape[-1] // 2
    return torch.cat([-x[..., d2:], x[..., :d2]], dim=-1)


def _apply_rotary(x, cos, sin):
    return x * cos[:x.shape[0]] + _rotate_half(x) * sin[:x.shape[0]]


def _reverse_rotary(x, cos, sin):
    return x * cos[:x.shape[0]] - _rotate_half(x) * sin[:x.shape[0]]


class RetrievalIndex:
    def __init__(self, n_kv, d_head, top_k=128, device='cuda'):
        self.n_kv = n_kv
        self.d_head = d_head
        self.kv_dim = n_kv * d_head
        self.top_k = top_k
        self.device = device
        self.k_scales = None
        self.k_packed = None        # (n, n_kv, d//2) uint8, 2 int4 per byte
        self.vr_scales = None
        self.vr_packed = None       # (n, n_kv, d//4) uint8, 4 int2 per byte
        self.n_stored = 0
        self.W = None
        self.K_mean = None
        self.V_mean = None
        self.cos = None
        self.sin = None

    def set_rotary_tables(self, cos, sin):
        self.cos = cos
        self.sin = sin

    # ── Pack/Unpack ──

    def _pack_int4(self, x):
        """x: (..., d) int8 [-8,7] → (..., d//2) uint8."""
        u = (x.to(torch.uint8) + 8)
        return (u[..., 0::2] << 4) | u[..., 1::2]

    def _unpack_int4(self, p):
        """p: (..., d//2) uint8 → (..., d) int8."""
        hi = ((p >> 4).to(torch.int8) - 8)
        lo = ((p & 0x0F).to(torch.int8) - 8)
        return torch.stack([hi, lo], dim=-1).reshape(*p.shape[:-1], p.shape[-1] * 2)

    def _pack_int2(self, x):
        """x: (..., d) int8 [-2,1] → (..., d//4) uint8."""
        u = (x.to(torch.uint8) + 2)
        return (u[..., 0::4] << 6) | (u[..., 1::4] << 4) | (u[..., 2::4] << 2) | u[..., 3::4]

    def _unpack_int2(self, p):
        """p: (..., d//4) uint8 → (..., d) int8."""
        v0 = ((p >> 6) & 0x03).to(torch.int8) - 2
        v1 = ((p >> 4) & 0x03).to(torch.int8) - 2
        v2 = ((p >> 2) & 0x03).to(torch.int8) - 2
        v3 = (p & 0x03).to(torch.int8) - 2
        return torch.stack([v0, v1, v2, v3], dim=-1).reshape(*p.shape[:-1], p.shape[-1] * 4)

    # ── Core ──

    @torch.no_grad()
    def calibrate(self, k_pre_sample, v_sample=None):
        n = k_pre_sample.shape[0]
        k2 = k_pre_sample.reshape(n, self.kv_dim).float()
        self.k_scales = k_pre_sample.abs().max(dim=0).values.clamp(min=1e-8)
        self.k_codes = torch.zeros(0, self.n_kv, self.d_head, device=self.device, dtype=torch.int8)
        self.k_packed = None
        self.vr_codes = torch.zeros(0, self.n_kv, self.d_head, device=self.device, dtype=torch.int8)
        self.vr_packed = None

        if v_sample is not None and n > 10:
            half = n // 2
            v2 = v_sample.reshape(n, self.kv_dim).float()
            k_tr, v_tr = k2[:half], v2[:half]
            self.K_mean = k_tr.mean(0, keepdim=True)
            self.V_mean = v_tr.mean(0, keepdim=True)
            kc = k_tr - self.K_mean; vc = v_tr - self.V_mean
            self.W = (kc.T @ vc).T @ torch.linalg.pinv(kc.T @ kc)

            v_pred_all = (k2 - self.K_mean) @ self.W.T + self.V_mean
            v_res = v2 - v_pred_all
            self.vr_scales = v_res.view(-1, self.n_kv, self.d_head).abs().amax(dim=(0, 2)).unsqueeze(1).clamp(min=1e-8)
        else:
            self.W = torch.eye(self.kv_dim, device=self.device)
            self.K_mean = torch.zeros(1, self.kv_dim, device=self.device)
            self.V_mean = torch.zeros(1, self.kv_dim, device=self.device)
            self.vr_scales = torch.ones(self.n_kv, 1, device=self.device)
        self.n_stored = 0

    def _deq_k(self, kvh):
        """Dequantize K_pre for one KV head: read raw int8 codes."""
        step = 2 * self.k_scales / 16
        if self.k_packed is not None:
            packed = self.k_packed[:, kvh, :]
            unpacked = self._unpack_int4(packed).float()
        else:
            unpacked = self.k_codes[:, kvh, :].float()
        return unpacked * step[kvh].float()

    def _predict_v(self, k_pre_flat):
        if self.W is None:
            return torch.zeros_like(k_pre_flat)
        return (k_pre_flat - self.K_mean) @ self.W.T + self.V_mean

    def _rotary(self, k_pre_2d, indices):
        if self.cos is None:
            return k_pre_2d
        return _apply_rotary(k_pre_2d,
                             self.cos[indices].to(k_pre_2d.dtype),
                             self.sin[indices].to(k_pre_2d.dtype))

    def _reconstruct_v(self, k_pre_top, top_idx):
        kv_flat = k_pre_top.reshape(-1, self.kv_dim)
        v_base = self._predict_v(kv_flat).reshape(-1, self.n_kv, self.d_head)
        vr_step = 2 * self.vr_scales / 4
        vr_dq = (self.vr_codes[top_idx].float() if self.vr_packed is None
                 else self._unpack_int2(self.vr_packed[top_idx]).float()) * vr_step.unsqueeze(0)
        return (v_base + vr_dq).float()

    @torch.no_grad()
    def append(self, k_pre_rope, v=None):
        if self.k_scales is None:
            raise RuntimeError("Call calibrate() first")

        k_step = 2 * self.k_scales / 16
        kq = torch.round(k_pre_rope.float() / k_step.float()).clamp(-8, 7).to(torch.int8)
        self.k_codes = torch.cat([self.k_codes, kq.unsqueeze(0)])

        if v is not None and self.W is not None:
            v_pred = self._predict_v(k_pre_rope.float().reshape(-1).unsqueeze(0)).reshape(1, self.n_kv, self.d_head)
            v_res = v.float().unsqueeze(0) - v_pred
            vr_step = 2 * self.vr_scales / 4
            vrq = torch.round(v_res / vr_step).clamp(-2, 1).to(torch.int8)
            self.vr_codes = torch.cat([self.vr_codes, vrq])
        else:
            self.vr_codes = torch.cat([self.vr_codes, torch.zeros(1, self.n_kv, self.d_head, device=self.device, dtype=torch.int8)])

        self.n_stored += 1

    @torch.no_grad()
    def batch_append(self, k_pre_all, v_all):
        n = k_pre_all.shape[0]
        k_step = 2 * self.k_scales / 16
        self.k_codes = torch.round(k_pre_all.float() / k_step).clamp(-8, 7).to(torch.int8)
        self.k_packed = None

        if self.W is not None:
            k_flat = k_pre_all.reshape(n, -1).float()
            v_pred = self._predict_v(k_flat).reshape(n, self.n_kv, self.d_head)
            v_res = v_all.float() - v_pred
            vr_step = 2 * self.vr_scales / 4
            self.vr_codes = torch.round(v_res / vr_step).clamp(-2, 1).to(torch.int8)
            self.vr_packed = None
        else:
            self.vr_codes = torch.zeros(n, self.n_kv, self.d_head, device=self.device, dtype=torch.int8)
            self.vr_packed = None

        self.n_stored = n

    @torch.no_grad()
    def retrieve(self, q_post_rope, n_q=None):
        n_q = n_q or q_post_rope.shape[0]
        g = n_q // self.n_kv
        d = self.d_head; scale = d ** 0.5
        out = torch.zeros(n_q, d, device=q_post_rope.device, dtype=torch.float32)
        nr = self.n_stored
        if nr == 0: return out
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
            k_top_pre = (self.k_codes[tidx].float() if self.k_packed is None
                         else self._unpack_int4(self.k_packed[tidx]).float()) * (2 * self.k_scales / 16).unsqueeze(0)
            v_top = self._reconstruct_v(k_top_pre, tidx)[:, kvh, :]
            s = (qh.unsqueeze(0) @ k_top_post.T).squeeze(0) / scale
            out[hi] = F.softmax(s, dim=-1) @ v_top
        return out

    @torch.no_grad()
    def compute_all_scores(self, q_post_rope):
        """Compute Q·K scores for all stored tokens × KV heads.
        Uses Triton if data is bit-packed, else Python fallback.
        """
        d = self.d_head; half = d // 2
        n_kv_ret = self.n_kv; n_stored_ret = self.n_stored
        g = q_post_rope.shape[0] // n_kv_ret

        if self.k_packed is not None:
            # Triton score kernel for bit-packed data
            from .kvr_triton import score_kernel as sk
            q_avg = q_post_rope.view(n_kv_ret, g, d).mean(dim=1)
            scores = torch.empty(n_stored_ret, n_kv_ret, device=self.device)
            sk[(n_stored_ret, n_kv_ret)](
                q_avg, self.k_packed, self.k_scales,
                self.cos, self.sin, scores,
                N_KV=n_kv_ret, G=1, D=d, HALF=half, N_STORED=n_stored_ret)
            return scores
        else:
            # Python fallback for raw int8 codes
            scores = torch.empty(n_stored_ret, n_kv_ret, device=self.device)
            q_avg = q_post_rope.view(n_kv_ret, g, d).mean(dim=1)
            all_idx = torch.arange(n_stored_ret, device=self.device)
            for kvh in range(n_kv_ret):
                k_pre = self._deq_k(kvh)
                k_post = self._rotary(k_pre, all_idx)
                scores[:, kvh] = (q_avg[kvh] @ k_post.T) / (d ** 0.5)
            return scores

    @torch.no_grad()
    def retrieve_topk(self, q_post_rope, n_q=None, exclude_start=0, exclude_end=0):
        n_q = n_q or q_post_rope.shape[0]
        g = n_q // self.n_kv
        d = self.d_head; scale = d ** 0.5
        nr = self.n_stored
        empty = (torch.zeros(0, self.n_kv, d, device=self.device, dtype=torch.float16),
                 torch.zeros(0, self.n_kv, d, device=self.device, dtype=torch.float16))
        if nr == 0 or exclude_start >= nr: return empty
        if exclude_start <= 0 and exclude_end >= nr: return empty

        top_k = min(self.top_k, nr)
        all_idx = torch.arange(nr, device=self.device)

        # Triton-accelerated scores: (nr, n_kv)
        scores = self.compute_all_scores(q_post_rope)
        scores[exclude_start:exclude_end, :] = float('-inf')
        top_indices = torch.topk(scores, top_k, dim=0)[1]  # (top_k, n_kv)

        # Batch V reconstruction: one W@K matmul for all unique tokens
        all_tidx = top_indices.T.reshape(-1)  # (n_kv * top_k,)
        unique_tidx, inverse = all_tidx.unique(sorted=False, return_inverse=True)
        k_pre_u = (self.k_codes[unique_tidx].float() if self.k_packed is None
                   else self._unpack_int4(self.k_packed[unique_tidx]).float()) * (2 * self.k_scales / 16).unsqueeze(0)
        kv_flat = k_pre_u.reshape(-1, self.kv_dim)
        v_base = self._predict_v(kv_flat).reshape(-1, self.n_kv, self.d_head)
        vr_step = 2 * self.vr_scales / 4
        vr_dq = (self.vr_codes[unique_tidx].float() if self.vr_packed is None
                 else self._unpack_int2(self.vr_packed[unique_tidx]).float()) * vr_step.unsqueeze(0)
        v_all = (v_base + vr_dq)[inverse].reshape(self.n_kv, top_k, self.n_kv, self.d_head)

        collected_k, collected_v = [], []
        for kvh in range(self.n_kv):
            tidx = top_indices[:, kvh]
            k_top = self._deq_k(kvh)[tidx]
            collected_k.append(self._rotary(k_top, tidx))
            collected_v.append(v_all[kvh, :, kvh, :])

        return (torch.stack(collected_k, dim=1).contiguous(),
                torch.stack(collected_v, dim=1).contiguous().half())
