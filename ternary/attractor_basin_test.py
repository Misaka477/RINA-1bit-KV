"""
attractor_basin_test.py — offline experiment
Check: can K-means clustering on K vectors create basins
that preserve the needle better than raw top-K?
"""
import os, sys, json, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoModelForCausalLM, AutoTokenizer
from modules.kvr_hook import KVRHook

MODEL = "D:/Software_Development/Project/models/Llama-3.2-1B"
haystack_text = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "evaluation", "_pride.txt"), "r", encoding="utf-8").read()
ND = "The secret password is KILO42. "
Q = " I just told you a secret password. The password is"
WS = 512; CTX = 16384; K_CLUSTERS = 128

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

hook = KVRHook(model, window_size=WS, top_k=2048, field_weight=0.0, ret_weight=1.0, device="cuda")
hook.prefill(seq_t)
hook.register()
del seq_t; torch.cuda.empty_cache()

# Extract K vectors from layer 10 (best layer from diagnostics)
ret = hook.retrievals[10]
nr = ret.n_stored
print(f"n_stored: {nr}, needle_pos: {needle_tok_pos}")

# Deq all K for head 0 (pre-RoPE, to match pre-RoPE Q comparison)
kp = ret._deq_k(0).cpu().numpy()  # (nr, d), pre-RoPE

print(f"K shape: {kp.shape}")
print(f"Clustering into {K_CLUSTERS} basins...")

# K-means on K vectors
from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(n_clusters=K_CLUSTERS, random_state=42, batch_size=1024)
labels = kmeans.fit_predict(kp)  # (nr,) cluster ID per token

# Check: which cluster has the needle?
needle_cluster = labels[needle_tok_pos]
cluster_counts = np.bincount(labels, minlength=K_CLUSTERS)
needle_cluster_size = cluster_counts[needle_cluster]
print(f"Needle is in cluster {needle_cluster}, size={needle_cluster_size}")

# Get query Q from the model (use pre-RoPE Q, K-means already works on pre-RoPE K)
q_tok = torch.tensor([q_ids], device="cuda")
with torch.no_grad():
    hidden = model.model.embed_tokens(q_tok)
    ln = model.model.layers[10].input_layernorm(hidden)
    q = model.model.layers[10].self_attn.q_proj(ln)  # (1, n_tok, n_q*d_head)
    n_q = model.config.num_attention_heads
    d_head = model.config.hidden_size // n_q
    n_kv = model.config.num_key_value_heads
    g = n_q // n_kv
    q = q.view(-1, n_q, d_head)  # (n_tok, n_q, d)
    q_gqa = q.view(-1, n_kv, g, d_head).mean(dim=2).mean(dim=0)  # (n_kv, d) mean over tokens and GQA
    q_vec = q_gqa[0].cpu().numpy()  # (d,)

# Score each cluster centroid against Q (pre-RoPE K, pre-RoPE Q)
centroids = kmeans.cluster_centers_  # (K_CLUSTERS, d)
centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8
centroids_n = centroids / centroid_norms
q_n = q_vec / (np.linalg.norm(q_vec) + 1e-8)
scores = centroids_n @ q_n  # (K_CLUSTERS,)
rank = np.argsort(-scores)
needle_rank_in_clusters = np.where(rank == needle_cluster)[0][0]

print(f"Needle cluster centroid rank by query: {needle_rank_in_clusters} / {K_CLUSTERS}")
print(f"Needle cluster score: {scores[needle_cluster]:.4f}, top score: {scores[rank[0]]:.4f}")

# Also: does token-level top-K do better or worse?
token_scores = kp @ q_n  # pre-RoPE
token_rank = np.argsort(-token_scores)
needle_token_rank = np.where(token_rank == needle_tok_pos)[0][0]
print(f"\nToken-level needle rank: {needle_token_rank} / {nr}")
print(f"Cluster-level needle rank: {needle_rank_in_clusters} / {K_CLUSTERS}")

# Conclusion
at_top128 = needle_rank_in_clusters < 128
at_top64 = needle_rank_in_clusters < 64
print(f"\nNeedle in top-128 clusters? {'YES' if at_top128 else 'NO'}")
print(f"Needle in top-64 clusters? {'YES' if at_top64 else 'NO'}")
