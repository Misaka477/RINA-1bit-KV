"""Test KVR triton score kernel."""
import sys, torch, time
sys.path.insert(0, ".")
from modules.kvr_triton import score_kernel, run_score_kernel as rsk

torch.manual_seed(0)
n_kv, d, n_s = 8, 64, 1280
g, half = 4, 32

q = torch.randn(n_kv * g, d, device='cuda').half()
k_scales = torch.randn(n_kv, d, device='cuda') * 0.5
k_raw = torch.randn(n_s, n_kv, d, device='cuda') * k_scales.unsqueeze(0)
kq = torch.round(k_raw / (2 * k_scales.unsqueeze(0) / 16)).clamp(-8, 7).to(torch.int8)
ku = (kq + 8).to(torch.uint8)
k_packed = (ku[..., 0::2] << 4) | ku[..., 1::2]

cos_tbl = torch.cos(torch.arange(n_s, device='cuda').float()[:, None] / 10000 ** (
    torch.arange(0, d, 2, device='cuda').float() / d))
sin_tbl = torch.sin(torch.arange(n_s, device='cuda').float()[:, None] / 10000 ** (
    torch.arange(0, d, 2, device='cuda').float() / d))

# Reference: PyTorch
k_deq = kq.float() * (2 * k_scales.unsqueeze(0) / 16)
k_rot = torch.empty(n_s, n_kv, d, device='cuda')
k_rot[..., :half] = k_deq[..., :half] * cos_tbl[:, None, :] - k_deq[..., half:] * sin_tbl[:, None, :]
k_rot[..., half:] = k_deq[..., :half] * sin_tbl[:, None, :] + k_deq[..., half:] * cos_tbl[:, None, :]
q_first = q.view(n_kv, g, d)[:, 0, :]  # first Q head per KV group
s_ref = torch.einsum('hd, thd -> h t', q_first, k_rot) / (d ** 0.5)
s_ref = s_ref.T  # (n_s, n_kv)

# Triton
s_tri = rsk(q_first, k_packed, k_scales, cos_tbl, sin_tbl)

diff = (s_ref - s_tri).abs().max().item()
print(f"Score kernel diff: {diff:.6f} PASS={diff < 1e-3}")

# Speed: 100 iterations
for _ in range(10):
    rsk(q_first, k_packed, k_scales, cos_tbl, sin_tbl)
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(100):
    s_tri = rsk(q_first, k_packed, k_scales, cos_tbl, sin_tbl)
torch.cuda.synchronize()
tri_ms = (time.perf_counter() - t0) * 10

# PyTorch speed
t0 = time.perf_counter()
for _ in range(100):
    k_deq = kq.float() * (2 * k_scales.unsqueeze(0) / 16)
    k_rot2 = torch.empty(n_s, n_kv, d, device='cuda')
    k_rot2[..., :half] = k_deq[..., :half] * cos_tbl[:, None, :] - k_deq[..., half:] * sin_tbl[:, None, :]
    k_rot2[..., half:] = k_deq[..., :half] * sin_tbl[:, None, :] + k_deq[..., half:] * cos_tbl[:, None, :]
    s_ref2 = torch.einsum('hd, thd -> h t', q_first, k_rot2) / (d ** 0.5)
torch.cuda.synchronize()
th_ms = (time.perf_counter() - t0) * 10

print(f"Triton: {tri_ms:.2f}ms  PyTorch: {th_ms:.2f}ms  Speedup: {th_ms/max(tri_ms,0.1):.1f}x")
