"""Test online calibration: fit (s,b) from warmup to correct KVR output."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from transformers import AutoModelForCausalLM, AutoTokenizer
from modules.kvr_hook import KVRHook

MODEL = "D:/Software_Development/Project/models/Llama-3.2-1B"
haystack_text = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_pride.txt"), "r", encoding="utf-8").read()
ND = "The secret password is KILO42. "
Q = " I just told you a secret password. The password is"
KW = "KILO42"
WS = 512; CTX = 16384

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
nd_ids = tok(ND, add_special_tokens=False)["input_ids"]
q_ids = tok(Q, add_special_tokens=False)["input_ids"]
chap_idx = haystack_text.find("CHAPTER")
haystack_ids = tok(haystack_text[chap_idx:chap_idx+200000], add_special_tokens=False)["input_ids"]

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
model.eval()

# Build context
depth = 0.5
np_pos = int(CTX * depth)
base = haystack_ids[:CTX].copy()
base[np_pos:min(np_pos+len(nd_ids),CTX)] = nd_ids[:min(len(nd_ids), CTX-np_pos)]
base.extend(q_ids)
sq = torch.tensor([base], device="cuda")

hook = KVRHook(model, window_size=WS, top_k=2048, field_weight=0.0, ret_weight=1.0, device="cuda",
               warmup_mode=True)
hook.prefill(sq)
hook.register()
print(f"Needle at pos={np_pos}")

# Warmup: 10 native steps, hooks capture (KVR, native) pairs
gen_ids = []
q_tok = torch.tensor([q_ids], device="cuda")
for step in range(10):
    cur = q_tok if step == 0 else gen_ids[-1]
    out = model(cur, use_cache=False, num_logits_to_keep=1)
    nid = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    gen_ids.append(nid)
    hook._step += 1
    hook._context_len += 1

print(f"Warmup done, fitting calibration...")
hook.fit_calibration()
hook.warmup_mode = False

# KVR with calibration
for step in range(40):
    out = model(gen_ids[-1], use_cache=False, num_logits_to_keep=1)
    nid = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    gen_ids.append(nid)
    hook._step += 1
    hook._context_len += 1

text = tok.decode(torch.cat(gen_ids, dim=1).cpu()[0], skip_special_tokens=True)
acc = 1.0 if KW.upper() in text.upper() else 0.0
print(f"NIAH: {'PASS' if acc else 'FAIL'}: {text[:80]}")
