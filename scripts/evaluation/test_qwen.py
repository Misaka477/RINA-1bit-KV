"""Test KVR with Qwen2.5-0.5B."""
import sys, torch
sys.path.insert(0, ".")
from modules import KVRGenerator

MODEL = "D:/Software_Development/Project/models/Qwen2.5-0.5B"
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
model.eval(); dev = model.device
h = model.config.hidden_size; nq = model.config.num_attention_heads
d = model.config.head_dim if hasattr(model.config, 'head_dim') and model.config.head_dim else h // nq
print(f"Loaded {MODEL}: {model.config.num_hidden_layers}L, "
      f"{model.config.num_key_value_heads}KV, {nq}Q, d={d}")

# Native
inp = tok("The capital of France is Paris", return_tensors="pt").to(dev)
n = inp["input_ids"].shape[1]
native = model.generate(**inp, max_new_tokens=10, do_sample=False, pad_token_id=tok.eos_token_id)
native_text = tok.decode(native[0, n:n+10].cpu(), skip_special_tokens=True)
print("Native:", native_text)

# KVR
gen = KVRGenerator(model, window_size=512, top_k=128, device=dev)
gen.prefill(inp["input_ids"])
ids = []
for step in range(10):
    tid = None if step > 0 else inp["input_ids"][0, -1]
    nid = gen.step(token_id=tid, temperature=1.0, top_k=1)
    ids.append(nid.item())
print("KVR:", tok.decode(ids, skip_special_tokens=True))
print("MATCH" if native_text == tok.decode(ids, skip_special_tokens=True) else "OK")
