"""Profile KVRGenerator step with retrieval active."""
import sys, time, torch
sys.path.insert(0, ".")
from transformers import AutoModelForCausalLM, AutoTokenizer
from modules import KVRGenerator

MODEL = "D:/Software_Development/Project/models/Llama-3.2-1B"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
model.eval(); dev = model.device

prompt = "The history of computing. " * 120
inp = tok(prompt, return_tensors="pt", truncation=True, max_length=550).to(dev)
print(f"Prompt: {inp['input_ids'].shape[1]} tokens")

gen = KVRGenerator(model, window_size=512, top_k=128, device=dev)
gen.prefill(inp["input_ids"])
print(f"After prefill: ctx={gen.kvr._context_len}, ret_built={gen.kvr._retrieval_built}, win_n={gen.kvr.windows[0].n}")

step_times = []
for i in range(10):
    t0 = time.perf_counter()
    nid = gen.step(token_id=inp["input_ids"][0, -1], temperature=1.0, top_k=1)
    step_times.append(time.perf_counter() - t0)

step_str = [f"{t*1000:.0f}ms" for t in step_times]
print(f"Step times: {step_str}")
print(f"Avg: {sum(step_times)/len(step_times)*1000:.0f}ms")
