"""Debug: compare old Python retrieve_topk scores vs new Triton scores on a NIAH case."""
import sys, torch, time
sys.path.insert(0, ".")
from transformers import AutoModelForCausalLM, AutoTokenizer
from modules.kvr_hook import KVRHook
from modules.kvr_retrieval import _apply_rotary

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

# Build NIAH: ctx=512, depth=0.5, window=64
ctx_len = 512
nr = max(0, int(ctx_len * 0.5 / len(hs_ids)))
seq = []; ri = 0
while len(seq) < ctx_len:
    seq.extend(nd_ids if ri == nr else hs_ids); ri += 1
seq = seq[:ctx_len] + q_ids
needle_tok = nr * len(hs_ids)

t = torch.tensor([seq], device=dev)
n_prompt = t.shape[1]
print(f"Prompt: {n_prompt} tokens, needle at token ~{needle_tok}")

# --- KVRHook prefill ---
kvrh = KVRHook(model, window_size=64, top_k=128, device=dev)
kvrh.prefill(t)

ret = kvrh.retrievals[0]  # layer 0
n_stored = ret.n_stored
win_n = kvrh.windows[0].n
print(f"ret built={kvrh._retrieval_built}, n_stored={n_stored}, win_n={win_n}")
print(f"retrieval indices range: 0..{n_stored-1}, needle at {needle_tok}")
print(f"needle in retrieval: {needle_tok < n_stored}")

# --- Get last prompt token's Q (rotated) ---
with torch.no_grad():
    base_out = model(t, output_hidden_states=True)
q4d = model.model.layers[0].self_attn.q_proj(base_out.hidden_states[0][0].to(torch.float16)).float().view(1, n_prompt, 32, 64)
k4d = model.model.layers[0].self_attn.k_proj(base_out.hidden_states[0][0].to(torch.float16)).float().view(1, n_prompt, 8, 64)
cos, sin = model.model.layers[0].self_attn.rotary_emb(q4d, position_ids=torch.arange(n_prompt, device=dev).unsqueeze(0))
from modules.kvr_retrieval import _apply_rotary
cq = cos[0].unsqueeze(1).expand(-1, 32, -1).reshape(-1, 64)
sq = sin[0].unsqueeze(1).expand(-1, 32, -1).reshape(-1, 64)
q_rot = _apply_rotary(q4d[0].reshape(-1, 64), cq, sq).view(n_prompt, 32, 64)
ck = cos[0].unsqueeze(1).expand(-1, 8, -1).reshape(-1, 64)
sk = sin[0].unsqueeze(1).expand(-1, 8, -1).reshape(-1, 64)
k_post_all = _apply_rotary(k4d[0].reshape(-1, 64), ck, sk).view(n_prompt, 8, 64)
q_last = q_rot[-1]

# --- OLD scores (Python, average all g heads per KV) ---
d = 64; half = 32
n_kv = 8; g = 32 // n_kv
all_idx = torch.arange(n_prompt, device=dev)

old_scores = torch.zeros(n_prompt, n_kv, device=dev)
for kvh in range(n_kv):
    k_pre = ret._deq_k(kvh)  # (n_stored, d)
    k_post = ret._rotary(k_pre, all_idx[:n_stored])  # (n_stored, d)
    scores_sum = torch.zeros(n_stored, device=dev)
    for qg_idx in range(g):
        hi = kvh * g + qg_idx
        scores_sum += q_last[hi] @ k_post.T
    old_scores[:n_stored, kvh] = (scores_sum / g) / (d ** 0.5)

# --- NEW scores (Triton) ---
q_avg = q_last.view(n_kv, g, d).mean(dim=1)  # (8, 64)
new_scores = ret.compute_all_scores(q_avg)  # (n_stored, n_kv)

# Compare
diff = (old_scores[:n_stored] - new_scores).abs().max().item()
print(f"\nScores max diff: {diff:.6f}")

# Compare top-K for KV head 0
old_top = old_scores[:n_stored, 0].argsort(descending=True)[:10]
new_top = new_scores[:, 0].argsort(descending=True)[:10]
print(f"Old top-10 for kvh=0: {old_top.tolist()}")
print(f"New top-10 for kvh=0: {new_top.tolist()}")
print(f"Needle position {needle_tok} in old top: {needle_tok in old_top.tolist()}")
print(f"Needle position {needle_tok} in new top: {needle_tok in new_top.tolist()}")

# Check top-128
old_top128 = old_scores[:n_stored, 0].argsort(descending=True)[:128]
new_top128 = new_scores[:, 0].argsort(descending=True)[:128]
print(f"Needle in old top-128: {needle_tok in old_top128.tolist()}")
print(f"Needle in new top-128: {needle_tok in new_top128.tolist()}")
