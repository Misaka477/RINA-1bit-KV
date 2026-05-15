"""Quick rank check: is needle in retrieval top-K?"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from transformers import AutoModelForCausalLM, AutoTokenizer
from modules.kvr_hook import KVRHook

MODEL = "D:/Software_Development/Project/models/Llama-3.2-1B"
haystack_text = open(os.path.join(os.path.dirname(__file__), "_pride.txt"), "r", encoding="utf-8").read()
ND = "The secret password is KILO42. "
Q = " I just told you a secret password. The password is"

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
nd_ids = tok(ND, add_special_tokens=False)["input_ids"]
q_ids = tok(Q, add_special_tokens=False)["input_ids"]

chap_idx = haystack_text.find("CHAPTER")
haystack_ids = tok(haystack_text[chap_idx:chap_idx+200000], add_special_tokens=False)["input_ids"]

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
model.eval()

ctx_len = 16384; WS = 512; depth = 0.5
needle_tok_pos = int(ctx_len * depth)
base = haystack_ids[:ctx_len].copy()
base[needle_tok_pos:needle_tok_pos+len(nd_ids)] = nd_ids
base.extend(q_ids)
seq_t = torch.tensor([base], device="cuda")

hook = KVRHook(model, window_size=WS, top_k=2048, field_weight=0.0, ret_weight=1.0, device="cuda")
hook.prefill(seq_t)
hook.register()

# Monkey-patch: run retrieve_topk with a logged version
# But first: directly check retrieval scores using the hook's internal Q
q_tok = torch.tensor([q_ids], device="cuda")

# Run one forward to get actual Q from the hook
with torch.no_grad():
    out = model(q_tok, use_cache=False, num_logits_to_keep=1)

# Now check retrieval for each layer
for li in [0, 5, 10, 15]:
    ret = hook.retrievals[li]
    nr = ret.n_stored
    if needle_tok_pos >= nr or nr == 0:
        print(f"L{li}: n_stored={nr}, needle out of range")
        continue
    
    # Get Q directly from layer li using rotary
    with torch.no_grad():
        h = model.model.layers[li].input_layernorm(model.model.embed_tokens(q_tok)).squeeze(0)
        q = model.model.layers[li].self_attn.q_proj(h)
        n_q = model.config.num_attention_heads
        d_head = model.config.hidden_size // n_q
        q = q.view(-1, n_q, d_head)
        pos_ids = torch.arange(q.shape[0], device='cuda').unsqueeze(0)
        cos, sin = model.model.layers[li].self_attn.rotary_emb(q.unsqueeze(0), pos_ids)
        cos = cos.squeeze(0); sin = sin.squeeze(0)
        # rotate_half
        q_rot = q * cos + torch.cat([-q[..., d_head//2:], q[..., :d_head//2]], dim=-1) * sin
        n_kv = model.config.num_key_value_heads
        g = n_q // n_kv
        q_avg = q_rot.view(-1, n_kv, g, d_head).mean(dim=2)  # (n_q_tok, n_kv, d)
        q_avg = q_rot.mean(dim=0)  # (n_q, d) -> mean over query token positions
        q_avg = q_avg.view(n_kv, g, d_head).mean(dim=1)  # (n_kv, d)

    # Score
    scores = torch.zeros(nr, n_kv, device='cuda')
    for kvh in range(n_kv):
        kp = ret._deq_k(kvh)
        aidx = torch.arange(kp.shape[0], device='cuda')
        kp_r = ret._rotary(kp, aidx)
        scores[:, kvh] = q_avg[kvh] @ kp_r.T
    rank = scores[:, 0].argsort(descending=True)
    needle_rk = (rank == needle_tok_pos).nonzero(as_tuple=True)[0]
    rk = needle_rk[0].item() if len(needle_rk) > 0 else -1
    print(f"L{li}: needle rank={rk}/{nr}")
