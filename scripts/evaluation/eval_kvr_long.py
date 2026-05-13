"""KVRGenerator long context test - save full output."""
import os, sys, json, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from transformers import AutoModelForCausalLM, AutoTokenizer
from modules import KVRGenerator

MODEL = "D:/Software_Development/Project/models/Llama-3.2-1B"
LONG = (
    "The history of computing spans centuries. Babbage designed the Analytical Engine in 1837. "
    "Alan Turing proposed the Turing machine in 1936. ENIAC was built in 1945. "
    "Transistors replaced vacuum tubes in the 1950s. Integrated circuits emerged in the 1960s. "
    "Intel released the 4004 CPU in 1971. The personal computer revolution began with the Altair 8800. "
    "The internet started as ARPANET in 1969. TCP/IP became standard in 1983. "
    "Tim Berners-Lee invented the World Wide Web in 1989. Mosaic browser launched in 1993. "
    "Google was founded in 1998. Wikipedia launched in 2001. Facebook in 2004. YouTube in 2005. "
) * 8

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
model.eval(); dev = model.device

inp = tok(LONG, return_tensors="pt", truncation=True, max_length=2000).to(dev)
n_prompt = inp["input_ids"].shape[1]
print(f"Prompt: {n_prompt} tokens")

# Native
native_out = model.generate(**inp, max_new_tokens=30, do_sample=False, pad_token_id=tok.eos_token_id)
native_new = tok.decode(native_out[0, n_prompt:n_prompt+30].cpu(), skip_special_tokens=True)

# KVRGenerator
gen = KVRGenerator(model, window_size=256, top_k=128, device=dev)
gen.prefill(inp["input_ids"])

gen_ids = []
for step in range(30):
    tid = None if step > 0 else inp["input_ids"][0, -1]
    nid = gen.step(token_id=tid, temperature=1.0, top_k=1)
    gen_ids.append(nid.item())

kvr_new = tok.decode(gen_ids, skip_special_tokens=True)

print(f"\n{'='*60}")
print(f"NATIVE: {native_new}")
print(f"{'='*60}")
print(f"KVR:    {kvr_new}")
print(f"{'='*60}")

json.dump({"native": native_new, "kvr": kvr_new,
           "prompt_tokens": n_prompt}, open("eval_kvr_long.json", "w"), indent=2, ensure_ascii=False)
print(f"Saved eval_kvr_long.json")
