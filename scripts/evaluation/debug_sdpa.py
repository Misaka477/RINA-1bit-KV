"""Debug SDPA prefill issue."""
import sys, torch, torch.nn.functional as F
sys.path.insert(0, ".")

torch.manual_seed(0)
n_kv, d, bsz, n_past = 8, 64, 4, 6
n_q, g = 32, 4
n_tot = n_past + bsz

q = torch.randn(bsz, n_q, d, device='cuda')
k = torch.randn(n_tot, n_kv, d, device='cuda')
v = torch.randn(n_tot, n_kv, d, device='cuda')

# Reference scores: (bsz, n_q, n_tot)
s_ref = torch.zeros(bsz, n_q, n_tot, device='cuda')
for bi in range(bsz):
    for hi in range(n_q):
        kvh = hi // g
        s_ref[bi, hi, :] = q[bi, hi] @ k[:, kvh, :].T / (d ** 0.5)
for i in range(bsz):
    s_ref[i, :, n_past + i + 1:] = float('-inf')

# SDPA scores (before softmax): need to extract from SDPA or compute directly
# Direct computation with GQA expansion
q_t = q.transpose(0, 1).unsqueeze(0)  # (1, n_q, bsz, d)
k_exp = k.unsqueeze(2).expand(-1, -1, g, -1).reshape(n_tot, n_q, d)
k_t = k_exp.transpose(0, 1).unsqueeze(0)

# Manual dot: each q head with its expanded K
s_sdpa = torch.bmm(q_t[0].float(), k_t[0].float().transpose(-2, -1)) / (d ** 0.5)
s_sdpa = s_sdpa.transpose(0, 1)  # (bsz, n_q, n_tot) — but need to verify

# Compare first Q head, first batch position
print("s_ref[0,0,:10]:", s_ref[0, 0, :10].tolist())
print("s_sdpa[0,0,:10]:", s_sdpa[0, 0, :10].tolist())
print("Match:", (s_ref[0, 0] - s_sdpa[0, 0]).abs().max().item() < 1e-4)
