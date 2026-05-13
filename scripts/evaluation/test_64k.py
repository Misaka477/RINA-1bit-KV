"""64K context generation with Python group softmax + lazy retrieval."""
import sys, time, torch
sys.path.insert(0, ".")
from transformers import AutoModelForCausalLM, AutoTokenizer
from modules import KVRGenerator

MODEL = "D:/Software_Development/Project/models/Llama-3.2-1B"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
model.eval(); dev = model.device

# Build non-repetitive 64K prompt
segments = [
    ("computing", "The history of computing spans centuries. Babbage designed the Analytical Engine in 1837. Alan Turing proposed the Turing machine in 1936. ENIAC was built in 1945. Transistors replaced vacuum tubes. Integrated circuits emerged in the 1960s."),
    ("biology", "DNA's double helix was discovered by Watson and Crick in 1953. The human genome project was completed in 2003. Mitochondria generate energy through oxidative phosphorylation. Neurons communicate through synapses using neurotransmitters."),
    ("geography", "Mount Everest is the tallest mountain on Earth at 8848 meters. The Amazon river is the largest by water volume. Lake Baikal is the deepest lake. The Sahara desert covers most of North Africa."),
    ("art", "The Renaissance began in Florence Italy in the 14th century. Leonardo da Vinci painted the Mona Lisa in the early 16th century. Impressionism emerged in France in the 1870s. Picasso developed Cubism in the early 20th century."),
]

prompt = ""
seg_idx = 0
while True:
    prompt += segments[seg_idx % len(segments)][1] + "\n"
    seg_idx += 1
    if len(tok(prompt)["input_ids"]) > 64000:
        break

inp = tok(prompt, return_tensors="pt", truncation=True, max_length=64000).to(dev)
n = inp["input_ids"].shape[1]
print(f"Prompt: {n} tokens")

# Skip Native (OOM at 64K)
t0 = time.perf_counter()
gen = KVRGenerator(model, window_size=2048, top_k=128, device=dev)
gen.prefill(inp["input_ids"])
t_prefill = time.perf_counter() - t0
print(f"Prefill: {t_prefill:.1f}s  ret={gen.kvr._retrieval_built}")

step_times = []
for i in range(20):
    ts = time.perf_counter()
    tid = None if i > 0 else inp["input_ids"][0, -1]
    nid = gen.step(token_id=tid, temperature=1.0, top_k=1)
    step_times.append(time.perf_counter() - ts)

avg_step = sum(step_times[1:]) / len(step_times[1:]) * 1000
print(f"First step: {step_times[0]*1000:.0f}ms")
print(f"Avg step: {avg_step:.0f}ms")
print(f"Total gen: {sum(step_times):.1f}s")

ids = []
for step in range(20):
    tid = None if step > 0 else inp["input_ids"][0, -1]
    ids.append(gen.step(token_id=tid, temperature=1.0, top_k=1).item())
print(f"Output: {tok.decode(ids, skip_special_tokens=True)[:200]}")
