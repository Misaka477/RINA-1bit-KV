"""Quick score kernel test."""
import sys; sys.path.insert(0, ".")
import torch
from modules.kvr_triton import score_kernel

n_kv, d = 8, 64; hh = 32; n_s = 20; g = 4
q = torch.randn(n_kv * g, d, device="cuda").half()
ks = torch.randn(n_kv, d, device="cuda") * 0.5
kr = torch.randn(n_s, n_kv, d, device="cuda") * ks.unsqueeze(0)
kq = torch.round(kr / (2 * ks.unsqueeze(0) / 16)).clamp(-8, 7).to(torch.int8)
ku = (kq + 8).to(torch.uint8)
kp = (ku[..., 0::2] << 4) | ku[..., 1::2]

pos = torch.arange(n_s, device="cuda").float()
freq = 10000 ** (-torch.arange(0, d, 2, device="cuda").float() / d)
ct = torch.cos(pos[:, None] * freq); st = torch.sin(pos[:, None] * freq)

kdf = kq.float() * (2 * ks.unsqueeze(0) / 16)
krot = torch.empty(n_s, n_kv, d, device="cuda")
krot[..., :hh] = kdf[..., :hh] * ct[:, None, :] - kdf[..., hh:] * st[:, None, :]
krot[..., hh:] = kdf[..., :hh] * st[:, None, :] + kdf[..., hh:] * ct[:, None, :]
qf = q.float().view(n_kv, g, d)[:, 0, :]
sr = torch.einsum("hd, thd -> ht", qf, krot).T / (d ** 0.5)

st2 = torch.empty(n_s, n_kv, device="cuda")
score_kernel[(n_s, n_kv)](q, kp, ks, ct, st, st2,
    N_KV=n_kv, G=g, D=d, HALF=hh, N_STORED=n_s)
diff = (sr - st2).abs().max().item()
print(f"Diff: {diff:.6f} PASS={diff < 1e-3}")
print(f"sr[0,:4]: {sr[:4, 0].tolist()}")
print(f"st[0,:4]: {st2[:4, 0].tolist()}")
