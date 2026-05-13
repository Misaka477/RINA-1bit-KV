"""Step-by-step debug score kernel."""
import sys; sys.path.insert(0, ".")
import torch
from modules.kvr_triton import score_kernel

torch.manual_seed(0)
n_kv, d, n_s, g = 2, 8, 3, 2  # very small
hh = d // 2
n_q = n_kv * g

q = torch.randn(n_q, d, device="cuda").half()
ks = torch.randn(n_kv, d, device="cuda") * 0.5
kr = torch.randn(n_s, n_kv, d, device="cuda") * ks.unsqueeze(0)
kq = torch.round(kr / (2 * ks.unsqueeze(0) / 16)).clamp(-8, 7).to(torch.int8)
ku = (kq + 8).to(torch.uint8)
kp = (ku[..., 0::2] << 4) | ku[..., 1::2]

pos = torch.arange(n_s, device="cuda").float()
freq = 10000 ** (-torch.arange(0, d, 2, device="cuda").float() / d)
ct = torch.cos(pos[:, None] * freq); st = torch.sin(pos[:, None] * freq)

# Manual: token 0, kvh 0
tid, kvh = 0, 0
hi_man = ku[tid, kvh, 0::2].float() - 8.0
lo_man = ku[tid, kvh, 1::2].float() - 8.0
s0_man = ks[kvh, 0::2]; s1_man = ks[kvh, 1::2]
k0_man = hi_man * (2.0 * s0_man / 16.0)
k1_man = lo_man * (2.0 * s1_man / 16.0)
ct_man = ct[tid, :]; st_man = st[tid, :]
k0r_man = k0_man * ct_man - k1_man * st_man
k1r_man = k0_man * st_man + k1_man * ct_man
q0_man = q[kvh * g, :hh]; q1_man = q[kvh * g, hh:]
sc_man = (torch.sum(q0_man.float() * k0r_man) + torch.sum(q1_man.float() * k1r_man)) / (d ** 0.5)

# Kernel
st2 = torch.empty(n_s, n_kv, device="cuda")
score_kernel[(n_s, n_kv)](q.half(), kp, ks, ct, st, st2,
    N_KV=n_kv, G=g, D=d, HALF=hh, N_STORED=n_s)
sc_krn = st2[tid, kvh].item()

print(f"Manual score: {sc_man.item():.6f}")
print(f"Kernel score: {sc_krn:.6f}")
print(f"Diff: {abs(sc_man.item() - sc_krn):.6f}")

# Also compare q loading
# kernel loads q0 from q_ptr + kvh * G * D
q0_krn_offset = kvh * g * d  # elements
q0_krn = q.float()[kvh * g, :hh]
print(f"\nQ0 manual: {q0_krn[:4].tolist()}")
print(f"Q offset={q0_krn_offset}, element at pos 0: {q.float().flatten()[q0_krn_offset:q0_krn_offset+4].tolist()}")
