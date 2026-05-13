"""KVR Triton kernels — score search (10.5x) + fused attention (1.7x)."""
import torch
import triton
import triton.language as tl


@triton.jit
def score_kernel(
    q_ptr, k_packed_ptr, k_scales_ptr, cos_tbl_ptr, sin_tbl_ptr,
    scores_ptr,
    N_KV: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
    HALF: tl.constexpr, N_STORED: tl.constexpr,
):
    """Unpack, deq, RoPE, dot product for one (token, kv_head)."""
    tid = tl.program_id(0)
    kvh = tl.program_id(1)
    if tid >= N_STORED or kvh >= N_KV:
        return

    q0 = tl.load(q_ptr + kvh * G * D + tl.arange(0, HALF))
    q1 = tl.load(q_ptr + kvh * G * D + HALF + tl.arange(0, HALF))

    base = tid * N_KV * HALF + kvh * HALF + tl.arange(0, HALF)
    packed = tl.load(k_packed_ptr + base)
    hi = (packed >> 4).to(tl.float32) - 8.0
    lo = (packed & 0x0F).to(tl.float32) - 8.0

    s0 = tl.load(k_scales_ptr + kvh * D + 2 * tl.arange(0, HALF))
    s1 = tl.load(k_scales_ptr + kvh * D + 2 * tl.arange(0, HALF) + 1)
    k0 = hi * (2.0 * s0 / 16.0)
    k1 = lo * (2.0 * s1 / 16.0)

    c = tl.load(cos_tbl_ptr + tid * D + tl.arange(0, HALF))
    s = tl.load(sin_tbl_ptr + tid * D + tl.arange(0, HALF))
    k0r = k0 * c - k1 * s
    k1r = k0 * s + k1 * c

    sc = (tl.sum(q0 * k0r) + tl.sum(q1 * k1r)) / (D ** 0.5)
    tl.store(scores_ptr + tid * N_KV + kvh, sc)


@triton.jit
def fused_attn_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    N_Q: tl.constexpr, N_KV: tl.constexpr, G: tl.constexpr,
    D: tl.constexpr, N_TOK: tl.constexpr,
    BLOCK_TOK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= N_Q:
        return
    kvh = pid // G

    q = tl.load(q_ptr + pid * D + tl.arange(0, D))

    m = -float('inf')
    s = 0.0
    o = tl.zeros([D], dtype=tl.float32)

    for start in range(0, N_TOK, BLOCK_TOK):
        off = start + tl.arange(0, BLOCK_TOK)
        mask = off < N_TOK

        k_off = off[:, None] * N_KV * D + kvh * D + tl.arange(0, D)[None, :]
        v_off = off[:, None] * N_KV * D + kvh * D + tl.arange(0, D)[None, :]
        k = tl.load(k_ptr + k_off, mask=mask[:, None], other=0.0)
        v = tl.load(v_ptr + v_off, mask=mask[:, None], other=0.0)

        scores = tl.sum(q[None, :] * k, axis=1) / (D ** 0.5)
        scores = tl.where(mask, scores, float('-inf'))

        m_new = tl.maximum(m, tl.max(scores))
        alpha = tl.exp(m - m_new)
        exp_s = tl.exp(scores - m_new)
        o = o * alpha + tl.sum(exp_s[:, None] * v, axis=0)
        s = s * alpha + tl.sum(exp_s)
        m = m_new

    tl.store(out_ptr + pid * D + tl.arange(0, D), o / s)


def run_score_kernel(q_first, k_packed, k_scales, cos_tbl, sin_tbl):
    n_stored, n_kv = k_packed.shape[0], k_packed.shape[1]
    d = k_packed.shape[2] * 2
    half = d // 2
    g = q_first.shape[0] // n_kv
    scores = torch.empty(n_stored, n_kv, device=q_first.device)
    grid = (n_stored, n_kv)
    score_kernel[grid](q_first, k_packed, k_scales, cos_tbl, sin_tbl, scores,
                       N_KV=n_kv, G=g, D=d, HALF=half, N_STORED=n_stored)
    return scores


def run_fused_attn(q, k, v):
    n_q, d = q.shape
    n_tok, n_kv = k.shape[0], k.shape[1]
    out = torch.empty_like(q)
    grid = (n_q,)
    fused_attn_kernel[grid](
        q, k, v, out,
        N_Q=n_q, N_KV=n_kv, G=n_q // n_kv, D=d, N_TOK=n_tok,
        BLOCK_TOK=64,
    )
    return out
