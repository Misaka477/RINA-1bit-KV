"""
WindowBuffer — Circular buffer of last N tokens' post-RoPE K/V.
Stores post-RoPE K for correct attention dot-products with the model's post-RoPE Q.
"""
import torch
import torch.nn.functional as F


class WindowBuffer:
    def __init__(self, n_kv, d_head, window_size=2048, device='cuda'):
        self.n_kv = n_kv
        self.d_head = d_head
        self.cap = window_size
        self.k = torch.zeros(window_size, n_kv, d_head, device=device, dtype=torch.float16)
        self.v = torch.zeros(window_size, n_kv, d_head, device=device, dtype=torch.float16)
        self.pos = 0
        self.n = 0

    @torch.no_grad()
    def append(self, k_post_rope, v):
        idx = self.pos % self.cap
        self.k[idx] = k_post_rope.to(torch.float16)
        self.v[idx] = v.to(torch.float16)
        self.pos += 1
        self.n = min(self.n + 1, self.cap)

    @torch.no_grad()
    def batch_append(self, k_all, v_all):
        """Append multiple tokens, evicting oldest if over capacity."""
        n = k_all.shape[0]
        remaining = n
        idx = 0
        while remaining > 0:
            slot = self.pos % self.cap
            batch = min(remaining, self.cap - slot if self.n >= self.cap else self.cap - slot)
            self.k[slot:slot+batch] = k_all[idx:idx+batch].to(torch.float16)
            self.v[slot:slot+batch] = v_all[idx:idx+batch].to(torch.float16)
            self.pos += batch
            remaining -= batch
            idx += batch
        self.n = min(self.n + n, self.cap)

    @torch.no_grad()
    def attention(self, q_post_rope, n_q=None, scale=None):
        """Exact softmax attention over window contents.

        Args:
            q_post_rope: (n_q, d_head) post-RoPE query.
            n_q: total Q heads (for GQA grouping). Defaults to q_post_rope.shape[0].
            scale: √d_head. Computed if not provided.

        Returns:
            (n_q, d_head) attention output.
        """
        n_q = n_q or q_post_rope.shape[0]
        g = n_q // self.n_kv
        d = self.d_head
        scale = scale or (d ** 0.5)

        kw = self.k[:self.n].float()  # (n_win, n_kv, d)
        vw = self.v[:self.n].float()

        out = torch.zeros(n_q, d, device=q_post_rope.device)
        for hi in range(n_q):
            kvh = hi // g
            scores = (q_post_rope[hi] @ kw[:, kvh, :].T) / scale  # (n_win,)
            w = F.softmax(scores, dim=-1)
            out[hi] = w @ vw[:, kvh, :]
        return out
