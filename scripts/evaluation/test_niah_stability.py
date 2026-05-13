"""Run ctx=256 depth=0.75 NIAH multiple times."""
import sys, torch
sys.path.insert(0, ".")
from transformers import AutoModelForCausalLM, AutoTokenizer
from modules.kvr_hook import KVRHook

MODEL = "D:/Software_Development/Project/models/Llama-3.2-1B"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
model.eval(); dev = model.device

HS = "The grass is green. "
ND = "The secret password is KILO42. "
Q = "I just told you a secret password. The password is"
hs_ids = tok(HS, add_special_tokens=False)["input_ids"]
nd_ids = tok(ND, add_special_tokens=False)["input_ids"]
q_ids = tok(Q, add_special_tokens=False)["input_ids"]

nr = max(0, int(256 * 0.75 / len(hs_ids)))  # depth=0.75, ctx=256
ntok = nr * len(hs_ids)
print(f"Needle at token ~{ntok}")

results = []
for run in range(5):
    torch.cuda.empty_cache()
    seq = []; ri = 0
    while len(seq) < 256:
        seq.extend(tok(ND, add_special_tokens=False)["input_ids"] if ri == nr else hs_ids)
        ri += 1
    seq = seq[:256] + q_ids
    t = torch.tensor([seq], device=dev)

    kvrh = KVRHook(model, window_size=64, top_k=128, device=dev)
    kvrh.prefill(t); kvrh.register()
    gen_ids = []
    for step in range(12):
        cur = t if step == 0 else torch.cat([t] + gen_ids, dim=1)
        out = model(cur, use_cache=False)
        nid = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        gen_ids.append(nid)
        kvrh._step += 1; kvrh._context_len += 1
    kvrh.remove()

    txt = tok.decode(torch.cat(gen_ids, dim=1)[0].cpu(), skip_special_tokens=True)
    p = "PASS" if "KILO42" in txt.upper() else "FAIL"
    results.append(p)
    print(f"  Run {run+1}: {p} — {txt[:50]}")

print(f"\n{results.count('PASS')}/5 PASS")
