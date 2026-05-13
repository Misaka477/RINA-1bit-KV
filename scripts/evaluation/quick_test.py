"""Quick test triton prefill."""
import sys, torch
sys.path.insert(0, ".")
from transformers import AutoModelForCausalLM, AutoTokenizer
from modules import KVRGenerator

MODEL = "D:/Software_Development/Project/models/Llama-3.2-1B"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
model.eval(); dev = model.device

prompt = "The history of computing. " * 70
inp = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(dev)
print("Tokens:", inp["input_ids"].shape[1])

import time
t0 = time.perf_counter()
gen = KVRGenerator(model, window_size=512, top_k=128, device=dev)
gen.prefill(inp["input_ids"])
print(f"Prefill: {time.perf_counter()-t0:.1f}s ctx={gen.kvr._context_len} ret={gen.kvr._retrieval_built}")

for step in range(5):
    tid = None if step > 0 else inp["input_ids"][0, -1]
    nid = gen.step(token_id=tid, temperature=1.0, top_k=1)
print("Gen OK")
