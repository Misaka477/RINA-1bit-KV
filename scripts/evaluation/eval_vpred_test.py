"""AR gen with V prediction â€?longer context to exercise retrieval."""
import os, sys, torch, torch.nn.functional as F, numpy as np, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from transformers import AutoModelForCausalLM, AutoTokenizer
from modules.kvr_hook import KVRHook

MODEL = "D:/Software_Development/Project/models/Llama-3.2-1B"
PROMPT = "The capital of France is Paris, a city known for its rich history and culture. The city"
MAX_TOKENS = 60

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
model.eval(); dev = model.device
inp = tok(PROMPT, return_tensors="pt").to(dev)

def js_div(p, q):
    ps = F.softmax(p, -1).clamp(1e-12); qs = F.softmax(q, -1).clamp(1e-12); m = 0.5 * (ps + qs)
    return 0.5 * (F.kl_div(ps.log().clamp(-50, 50), m, reduction='batchmean') + F.kl_div(qs.log().clamp(-50, 50), m, reduction='batchmean'))

out = model.generate(**inp, max_new_tokens=MAX_TOKENS, do_sample=False, pad_token_id=tok.eos_token_id, return_dict_in_generate=True, output_logits=True)
native_new = tok.decode(out.sequences[0, inp["input_ids"].shape[1]:].cpu(), skip_special_tokens=True)
print(f"NATIVE: {native_new[:150]}")

for ws in [64]:
    fwr = KVRHook(model, window_size=ws, top_k=128, field_weight=0.0, ret_weight=1.0, device=dev)
    fwr.prefill(inp["input_ids"])
    fwr.register()
    gen_ids, js_vals = [], []
    try:
        for step in range(MAX_TOKENS):
            cur = inp["input_ids"] if step == 0 else torch.cat([inp["input_ids"]] + gen_ids, dim=1)
            out2 = model(cur, use_cache=False)
            logits = out2.logits[:, -1, :].float()
            nid = logits.argmax(dim=-1, keepdim=True)
            gen_ids.append(nid)
            fwr._step += 1; fwr._context_len += 1
            if step < len(out.logits):
                js_vals.append(float(js_div(logits.cpu(), out.logits[step][0].float().cpu())))
    finally:
        fwr.remove()
    txt = tok.decode(torch.cat(gen_ids, dim=1).cpu()[0], skip_special_tokens=True)
    js_mn = float(np.mean(js_vals)) if js_vals else 0
    print(f"WIN{ws}:  JS={js_mn:.4f}")
    print(f"  NATIVE: {native_new}")
    print(f"  FWR:    {txt}")
    m = sum(1 for a, b in zip(native_new.split(), txt.split()) if a == b)
    print(f"  Matched tokens: {m}/{len(native_new.split())}")


