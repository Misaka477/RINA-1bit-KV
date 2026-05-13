"""Smoke test: KVR imports + generation after cleanup."""
import sys, torch
sys.path.insert(0, ".")
from modules import KVRGenerator, KVRHook, WindowBuffer, RetrievalIndex
print("Imports OK")

from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL = "D:/Software_Development/Project/models/Llama-3.2-1B"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
model.eval(); dev = model.device

inp = tok("The capital of France is Paris", return_tensors="pt").to(dev)
n = inp["input_ids"].shape[1]

native = model.generate(**inp, max_new_tokens=10, do_sample=False, pad_token_id=tok.eos_token_id)
native_text = tok.decode(native[0, n:n+10].cpu(), skip_special_tokens=True)
print("Native:", native_text)

gen = KVRGenerator(model, window_size=512, top_k=128, device=dev)
gen.prefill(inp["input_ids"])
ids = []
for step in range(10):
    tid = None if step > 0 else inp["input_ids"][0, -1]
    nid = gen.step(token_id=tid, temperature=1.0, top_k=1)
    ids.append(nid.item())
kvr_text = tok.decode(ids, skip_special_tokens=True)
print("KVR:", kvr_text)
print("MATCH" if native_text == kvr_text else "OK")
