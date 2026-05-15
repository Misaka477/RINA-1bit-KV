"""Test KVR on real long-text generation (qualitative)."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from transformers import AutoModelForCausalLM, AutoTokenizer
from modules.kvr_hook import KVRHook

MODEL = "D:/Software_Development/Project/models/Llama-3.2-1B"
haystack_text = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_pride.txt"), "r", encoding="utf-8").read()

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tok.pad_token = tok.eos_token

# Use first 8K tokens of P&P as context
chap_idx = haystack_text.find("CHAPTER")
context = haystack_text[chap_idx:chap_idx+30000]
input_ids = tok(context, add_special_tokens=False, return_tensors="pt")["input_ids"]
ctx_len = 8192
input_ids = input_ids[:, :ctx_len].to("cuda")

print(f"Context: {input_ids.shape[1]} tokens")
print(f"Context starts: {tok.decode(input_ids[0,:20])}")
print(f"Context ends: {tok.decode(input_ids[0,-20:])}")

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
model.eval()

print(f"\n=== KVR (window=512, top-K=128, no attractor) ===")
hook = KVRHook(model, window_size=512, top_k=128, field_weight=0.0, ret_weight=1.0, device="cuda")
hook.prefill(input_ids)
hook.register()
del input_ids; torch.cuda.empty_cache()

gen_ids = []
for step in range(30):
    if step == 0:
        q_tok = tok("\n\nThe story continues:", add_special_tokens=False, return_tensors="pt").to("cuda")
        cur = q_tok["input_ids"]
    else:
        cur = gen_ids[-1]
    out = model(cur, use_cache=False, num_logits_to_keep=1)
    nid = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    gen_ids.append(nid)
    hook._step += 1; hook._context_len += 1

hook.remove()
text = tok.decode(torch.cat(gen_ids, dim=1).cpu()[0], skip_special_tokens=True)
print(f"Generated: {text[:200]}")
