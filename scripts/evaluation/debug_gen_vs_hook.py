"""Debug: KVRGenerator vs Hook step 0 and 1."""
import sys; sys.path.insert(0, ".")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from modules import KVRGenerator
from modules.kvr_hook import KVRHook

MODEL = "D:/Software_Development/Project/models/Llama-3.2-1B"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
model.eval(); dev = model.device

HS = "The grass is green. "
ND = "The secret password is KILO42. "
Qtxt = " I just told you a secret password. The password is"
ctx = 128

hs_ids = tok(HS, add_special_tokens=False)["input_ids"]
nd_ids = tok(ND, add_special_tokens=False)["input_ids"]
q_ids = tok(Qtxt, add_special_tokens=False)["input_ids"]
nr = int(ctx * 0.25 / len(hs_ids))
seq = []; ri = 0
while len(seq) < ctx:
    seq.extend(nd_ids if ri == nr else hs_ids); ri += 1
seq = seq[:ctx] + q_ids
t = torch.tensor([seq], device=dev)
print("ctx=", len(seq))

# Generator
print("=== GENERATOR ===")
gen = KVRGenerator(model, window_size=64, top_k=128, device=dev)
gen.prefill(t)
n0 = gen.step(token_id=t[0, -1], temperature=1.0, top_k=1)
n1 = gen.step(token_id=None, temperature=1.0, top_k=1)
t0 = tok.decode([n0.item()])
t1 = tok.decode([n1.item()])
print(f"  step0={t0}  step1={t1}")

# Hook
print("=== HOOK ===")
kvrh = KVRHook(model, window_size=64, top_k=128, device=dev)
kvrh.prefill(t)
kvrh.register()
out0 = model(t, use_cache=False)
h0 = out0.logits[:, -1, :].argmax(dim=-1, keepdim=True)
cur = torch.cat([t, h0], dim=1)
kvrh._step += 1; kvrh._context_len += 1
out1 = model(cur, use_cache=False)
h1 = out1.logits[:, -1, :].argmax(dim=-1, keepdim=True)
ht0 = tok.decode([h0[0, 0].item()])
ht1 = tok.decode([h1[0, 0].item()])
print(f"  step0={ht0}  step1={ht1}")
kvrh.remove()
