"""Quick perf: 4K/16K prefill + gen."""
import sys, time, torch
sys.path.insert(0, ".")
from transformers import AutoModelForCausalLM, AutoTokenizer
from modules import KVRGenerator

MODEL = "D:/Software_Development/Project/models/Llama-3.2-1B"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
model.eval(); dev = model.device

for ctx_target in [4096, 16384]:
    prompt = "The history of computing. " * (ctx_target // 7)
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=ctx_target).to(dev)
    n = inp["input_ids"].shape[1]
    print(f"\n=== ctx={n} ===")

    t0 = time.perf_counter()
    gen = KVRGenerator(model, window_size=512, top_k=128, device=dev)
    gen.prefill(inp["input_ids"])
    t_prefill = time.perf_counter() - t0
    print(f"Prefill: {t_prefill:.1f}s  ctx={gen.kvr._context_len} ret={gen.kvr._retrieval_built}")

    step_times = []
    for i in range(10):
        ts = time.perf_counter()
        tid = None if i > 0 else inp["input_ids"][0, -1]
        gen.step(token_id=tid, temperature=1.0, top_k=1)
        step_times.append(time.perf_counter() - ts)
    avg_step = sum(step_times[2:]) / max(len(step_times[2:]), 1) * 1000
    print(f"Gen step avg: {avg_step:.0f}ms (first step: {step_times[0]*1000:.0f}ms)")
