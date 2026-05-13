"""Native NIAH baseline."""
import sys, torch
sys.path.insert(0, ".")
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "D:/Software_Development/Project/models/Llama-3.2-1B"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
model.eval(); dev = model.device

HS = "The grass is green. "
ND = "The secret password is KILO42. "
Q = " I just told you a secret password. The password is"
hs_ids = tok(HS, add_special_tokens=False)["input_ids"]
nd_ids = tok(ND, add_special_tokens=False)["input_ids"]
q_ids = tok(Q, add_special_tokens=False)["input_ids"]

passed = 0; total = 0
for ctx_len in [128, 256, 512]:
    for depth in [0.25, 0.5, 0.75]:
        nr = max(0, int(ctx_len * depth / len(hs_ids)))
        seq = []; ri = 0
        while len(seq) < ctx_len:
            seq.extend(nd_ids if ri == nr else hs_ids); ri += 1
        seq = seq[:ctx_len] + q_ids
        t = torch.tensor([seq], device=dev)
        out = model.generate(t, max_new_tokens=12, do_sample=False, pad_token_id=tok.eos_token_id)
        txt = tok.decode(out[0, t.shape[1]:].cpu(), skip_special_tokens=True)
        p = 1 if "KILO42" in txt.upper() else 0
        passed += p; total += 1
        print(f"native ctx={ctx_len} d={depth:.2f} = {p}")
print(f"\nNATIVE: {passed}/{total}")
