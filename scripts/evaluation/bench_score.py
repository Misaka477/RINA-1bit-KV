"""Benchmark score kernel."""
import sys; sys.path.insert(0, ".")
import torch, time
from modules.kvr_triton import score_kernel as sk

n_kv, d, n_s, g = 8, 64, 1280, 4
hh = d // 2

q = torch.randn(n_kv * g, d, device="cuda").half()
ks = torch.randn(n_kv, d, device="cuda") * 0.5
kr = torch.randn(n_s, n_kv, d, device="cuda") * ks.unsqueeze(0)
kq = torch.round(kr / (2 * ks.unsqueeze(0) / 16)).clamp(-8, 7).to(torch.int8)
kp = ((kq.to(torch.uint8) + 8)[..., 0::2] << 4) | ((kq.to(torch.uint8) + 8)[..., 1::2])
pos = torch.arange(n_s, device="cuda").float()
fr = 10000 ** (-torch.arange(0, d, 2, device="cuda").float() / d)
ct = torch.cos(pos[:, None] * fr); st = torch.sin(pos[:, None] * fr)

st2 = torch.empty(n_s, n_kv, device="cuda")
for _ in range(5):
    sk[(n_s, n_kv)](q, kp, ks, ct, st, st2, N_KV=n_kv, G=g, D=d, HALF=hh, N_STORED=n_s)
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(100):
    sk[(n_s, n_kv)](q, kp, ks, ct, st, st2, N_KV=n_kv, G=g, D=d, HALF=hh, N_STORED=n_s)
torch.cuda.synchronize()
triton_ms = (time.perf_counter() - t0) * 10

# PyTorch
qf = q.float().view(n_kv, g, d)[:, 0, :]
t0 = time.perf_counter()
for _ in range(100):
    kdf = kq.float() * (2 * ks.unsqueeze(0) / 16)
    krot = torch.empty(n_s, n_kv, d, device="cuda")
    krot[..., :hh] = kdf[..., :hh] * ct[:, None, :] - kdf[..., hh:] * st[:, None, :]
    krot[..., hh:] = kdf[..., :hh] * st[:, None, :] + kdf[..., hh:] * ct[:, None, :]
    torch.einsum("hd, thd -> ht", qf, krot).T / (d ** 0.5)
torch.cuda.synchronize()
torch_ms = (time.perf_counter() - t0) * 10
print(f"Triton: {triton_ms:.2f}ms  PyTorch: {torch_ms:.2f}ms  Speedup: {torch_ms / max(triton_ms, 0.1):.1f}x")
