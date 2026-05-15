"""Debug cos table shapes."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "D:/Software_Development/Project/models/Llama-3.2-1B"
print("Loading...")
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
model.eval()

cfg = model.config
n_q = cfg.num_attention_heads
n_kv = cfg.num_key_value_heads
d_head = getattr(cfg, 'head_dim', None) or (cfg.hidden_size // n_q)
max_pos = cfg.max_position_embeddings
print(f"d_head={d_head}, n_kv={n_kv}, max_pos={max_pos}")

first_attn = model.model.layers[0].self_attn
dummy_q = torch.empty(1, max_pos, n_q, d_head, device=model.device)
cos_tbl, sin_tbl = first_attn.rotary_emb(
    dummy_q, position_ids=torch.arange(max_pos, device=model.device).unsqueeze(0))
print(f"cos_tbl raw: {cos_tbl.shape}")
cos_tbl = cos_tbl[0].float()
print(f"cos_tbl after [0]: {cos_tbl.shape}")

# Test prefill cos construction
n_prompt = 65536
c = cos_tbl[:n_prompt].unsqueeze(1).expand(-1, n_kv, -1)
print(f"After unsqueeze+expand: {c.shape}")
try:
    c2 = c.reshape(-1, d_head)
    print(f"After reshape(-1, {d_head}): {c2.shape}")
except RuntimeError as e:
    print(f"Reshape error: {e}")

# What should work: keep the d//2 dimension
c3 = cos_tbl[:n_prompt].unsqueeze(1).expand(-1, n_kv, -1)
k_pre = torch.empty(n_prompt, n_kv, d_head, device=model.device)
print(f"k_pre shape: {k_pre.shape}")
print(f"c3 shape: {c3.shape}  (n_prompt, n_kv, d//2)")

# The expected shapes for _apply_rotary
k_flat = k_pre.reshape(-1, d_head)
c_flat = c3.reshape(n_prompt * n_kv, d_head // 2)
print(f"k_flat: {k_flat.shape}, c_flat: {c_flat.shape}")
print("Mismatch: k_flat has 64 cols, c_flat has 32 cols")
