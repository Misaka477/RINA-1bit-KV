"""Diagnose: needle rank in int4 retrieval vs full-precision."""
import os, sys, json, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import rotate_half, apply_rotary_pos_emb
from modules.kvr_hook import KVRHook

MODEL = "D:/Software_Development/Project/models/Llama-3.2-1B"
haystack_text = open(os.path.join(os.path.dirname(__file__), "_pride.txt"), "r", encoding="utf-8").read()
ND = "The secret password is KILO42. "
Q = " I just told you a secret password. The password is"
WS = 512; CTX = 16384

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
nd_ids = tok(ND, add_special_tokens=False)["input_ids"]
q_ids = tok(Q, add_special_tokens=False)["input_ids"]

chap_idx = haystack_text.find("CHAPTER")
haystack_ids = tok(haystack_text[chap_idx:chap_idx+200000], add_special_tokens=False)["input_ids"]

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
model.eval()

# Build context with needle at depth 0.5
depth = 0.5
needle_tok_pos = int(CTX * depth)
base = haystack_ids[:CTX].copy()
base[needle_tok_pos:needle_tok_pos + len(nd_ids)] = nd_ids
base.extend(q_ids)
seq_t = torch.tensor([base], device="cuda")

# Prefill + register
hook = KVRHook(model, window_size=WS, top_k=2048, field_weight=0.0, ret_weight=1.0, device="cuda")
hook.prefill(seq_t)
hook.register()
del seq_t; torch.cuda.empty_cache()

# Get post-RoPE Q for the query tokens (from layer 0)
q_tok = torch.tensor([q_ids], device="cuda")
with torch.no_grad():
    hidden = model.model.embed_tokens(q_tok)
    ln = model.model.layers[0].input_layernorm(hidden)
    q = model.model.layers[0].self_attn.q_proj(ln)
    n_q = model.config.num_attention_heads
    d_head = model.config.hidden_size // n_q
    q = q.view(1, -1, n_q, d_head)  # (1, n_q_tok, n_q, d)
    cos, sin = model.model.layers[0].self_attn.rotary_emb(q, torch.arange(q.shape[1], device='cuda').unsqueeze(0))
    q_rot, _ = apply_rotary_pos_emb(q, q, cos, sin, unsqueeze_dim=2)  # (1, n_q_tok, n_q, d)

# GQA mean: average over query token positions and GQA groups
n_kv = model.config.num_key_value_heads
g = n_q // n_kv
q_avg = q_rot.mean(dim=1)  # (1, n_q, d)
q_avg = q_avg.view(1, n_kv, g, d_head).mean(dim=2)  # (1, n_kv, d)

print(f"Needle at token position: {needle_tok_pos}")
for li in [0, 5, 10, 15]:
    ret = hook.retrievals[li]
    nr = ret.n_stored
    if needle_tok_pos >= nr: continue
    
    # Get Q for this specific layer
    with torch.no_grad():
        hidden = model.model.embed_tokens(q_tok)
        ln = model.model.layers[li].input_layernorm(hidden)
        q = model.model.layers[li].self_attn.q_proj(ln)
        q = q.view(1, -1, n_q, d_head)
        cos, sin = model.model.layers[li].self_attn.rotary_emb(q, torch.arange(q.shape[1], device='cuda').unsqueeze(0))
        q_rot, _ = apply_rotary_pos_emb(q, q, cos, sin, unsqueeze_dim=2)
    q_avg = q_rot.mean(dim=1)  # (1, n_q, d)
    q_avg_gqa = q_avg.view(1, n_kv, g, d_head).mean(dim=2)  # (1, n_kv, d)
    
    # === int4 retrieval score (KVR path) ===
    scores_int4 = ret.compute_all_scores(q_avg_gqa.squeeze(0).float())
    rank_int4 = scores_int4[:, 0].argsort(descending=True)
    rk_int4 = (rank_int4 == needle_tok_pos).nonzero(as_tuple=True)[0]
    rk_int4 = rk_int4[0].item() if len(rk_int4) > 0 else -1
    
    # === Full-precision scores (native-like) ===
    kp = ret._deq_k(0)
    aidx = torch.arange(kp.shape[0], device=kp.device)
    kp_rot = ret._rotary(kp, aidx)
    qv = q_avg_gqa[0, 0].float()
    scores_fp16 = (qv / (qv.norm() + 1e-8)) @ (kp_rot / (kp_rot.norm(dim=-1, keepdim=True) + 1e-8)).T
    rank_fp16 = scores_fp16.argsort(descending=True)
    rk_fp16 = (rank_fp16 == needle_tok_pos).nonzero(as_tuple=True)[0]
    rk_fp16 = rk_fp16[0].item() if len(rk_fp16) > 0 else -1
    
    # === Check: is needle score in int4 version similar to fp16? ===
    needle_s_int4 = scores_int4[needle_tok_pos, 0].item()
    needle_s_fp16 = scores_fp16[needle_tok_pos].item()
    top1_s_int4 = scores_int4[rank_int4[0], 0].item()
    top1_s_fp16 = scores_fp16[rank_fp16[0]].item()
    
    print(f"L{li:2d} n_stored={nr:5d} | int4 rank={rk_int4:5d} fp16 rank={rk_fp16:5d} | "
          f"needle_score int4={needle_s_int4:.4f} fp16={needle_s_fp16:.4f} | "
          f"top1 int4={top1_s_int4:.4f} fp16={top1_s_fp16:.4f}")
