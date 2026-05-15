import json, torch
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("D:/Software_Development/Project/models/Llama-3.2-1B")
print(f"max_position_embeddings: {cfg.max_position_embeddings}")
print(f"rope_theta: {cfg.rope_theta}")
print(f"head_dim: {getattr(cfg, 'head_dim', None)}")
print(f"hidden_size: {cfg.hidden_size}")
print(f"num_heads: {cfg.num_attention_heads}")
print(f"num_kv: {cfg.num_key_value_heads}")
