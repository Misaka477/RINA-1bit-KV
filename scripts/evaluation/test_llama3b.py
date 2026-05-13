"""Test KVR with Llama-3.2-3B."""
import sys, torch
sys.path.insert(0, ".")
from modules import KVRGenerator
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "D:/Software_Development/Project/models/Llama-3.2-3B"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
model.eval(); dev = model.device
cfg = model.config; d = cfg.head_dim or cfg.hidden_size // cfg.num_attention_heads
print(f"{MODEL}: {cfg.num_hidden_layers}L, {cfg.num_key_value_heads}KV, {cfg.num_attention_heads}Q, d={d}")

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
print("KVR:", tok.decode(ids, skip_special_tokens=True))
print("MATCH" if native_text == tok.decode(ids, skip_special_tokens=True) else "OK")
