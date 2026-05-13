"""Direct score comparison: triton vs Python for a single token."""
import sys, torch
sys.path.insert(0, ".")
from modules.kvr_hook import KVRHook
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "D:/Software_Development/Project/models/Llama-3.2-1B"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
model.eval(); dev = model.device

HS = "The grass is green. "
ND = "The secret password is KILO42. "
Qtxt = " I just told you a secret password. The password is"
hs_ids = tok(HS, add_special_tokens=False)["input_ids"]
nd_ids = tok(ND, add_special_tokens=False)["input_ids"]
q_ids = tok(Qtxt, add_special_tokens=False)["input_ids"]
nr = max(0, int(512 * 0.5 / len(hs_ids)))
seq = []; ri = 0
while len(seq) < 512:
    seq.extend(nd_ids if ri == nr else hs_ids); ri += 1
seq = seq[:512] + q_ids
t = torch.tensor([seq], device=dev)
kvrh = KVRHook(model, window_size=64, top_k=128, device=dev)
kvrh.prefill(t)

ret = kvrh.retrievals[0]
d = 64; half = 32; nk = 8; g = 4

# Get last token Q
base = model(t, output_hidden_states=True)
h_norm = model.model.layers[0].input_layernorm(base.hidden_states[0][0].to(torch.float16))
q4 = model.model.layers[0].self_attn.q_proj(h_norm).view(-1, nk * g, d)
k4 = model.model.layers[0].self_attn.k_proj(h_norm).view(-1, nk, d)
pos_ids = torch.arange(512 + len(q_ids), device=dev).unsqueeze(0)
cos, sin = model.model.layers[0].self_attn.rotary_emb(q4.unsqueeze(0), position_ids=pos_ids)
cq = cos[0].unsqueeze(1).expand(-1, nk*g, -1).reshape(-1, d)
sq = sin[0].unsqueeze(1).expand(-1, nk*g, -1).reshape(-1, d)
ck0 = cos[0].unsqueeze(1).expand(-1, nk, -1).reshape(-1, d)
sk0 = sin[0].unsqueeze(1).expand(-1, nk, -1).reshape(-1, d)

from modules.kvr_retrieval import _apply_rotary
q_rot = _apply_rotary(q4.reshape(-1, d), cq, sq).view(-1, nk*g, d)
k_post_all = _apply_rotary(k4.reshape(-1, d), ck0, sk0).view(-1, nk, d)
q_last = q_rot[-1]  # (32, 64)

# Compare scores for token 252, kvh=0
tid = 252; kvh = 0

# Python
k_pre = ret._deq_k(kvh)[tid]  # (64,) dequantized K_pre
k_post = ret._rotary(k_pre.unsqueeze(0), torch.tensor([tid], device=dev))[0]
scores_py = 0.0
for qgi in range(g):
    hi = kvh * g + qgi
    scores_py += (q_last[hi].float() @ k_post).item()
scores_py = scores_py / g / (d ** 0.5)

# Triton equivalent
# Compute manually what the triton kernel does
import triton.language as tl
# Can't directly call tl operations from Python. Use PyTorch to simulate the triton code:
packed = ret.k_packed[tid, kvh].float()  # but packed is uint8, float() is wrong
# Actually, ret.k_packed is uint8. Loading via .float() converts 0-255 to float, but then
# (float_val >> 4) doesn't work on floats. The triton kernel uses >> on uint8, not float.
# So _deq_k properly handles this with _unpack_int4 which uses uint8.
#
# Let me just use _deq_k output and apply the same RoPE
q0 = q_last[0, :half]; q1 = q_last[0, half:]
ct = ret.cos[tid, :half]; st = ret.sin[tid, :half]
k0 = k_pre[:half]; k1 = k_pre[half:]
k0r = k0 * ct - k1 * st
k1r = k0 * st + k1 * ct
scores_tri = ((q0 * k0r).sum() + (q1 * k1r).sum()).item() / (d ** 0.5)

print(f"Python score: {scores_py:.6f}")
print(f"Triton score: {scores_tri:.6f}")
print(f"Diff: {abs(scores_py - scores_tri):.6f}")
