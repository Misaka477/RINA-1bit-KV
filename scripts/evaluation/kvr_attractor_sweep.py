"""
kvr_attractor_sweep.py — Parameter sweep for attractor basin KVR.
Fast path: G1+G2 offline, G4 best-combo generation.
"""
import os, sys, json, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from transformers import AutoModelForCausalLM, AutoTokenizer
from modules.kvr_hook import KVRHook
from sklearn.cluster import MiniBatchKMeans

MODEL = "D:/Software_Development/Project/models/Llama-3.2-1B"
haystack_text = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_pride.txt"), "r", encoding="utf-8").read()
ND = "The secret password is KILO42. "
Q = " I just told you a secret password. The password is"
KW = "KILO42"; WS = 512; CTX = 16384; MAX_NEW = 50

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
nd_ids = tok(ND, add_special_tokens=False)["input_ids"]
q_ids = tok(Q, add_special_tokens=False)["input_ids"]
chap_idx = haystack_text.find("CHAPTER")
haystack_ids = tok(haystack_text[chap_idx:chap_idx+200000], add_special_tokens=False)["input_ids"]

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
model.eval()

def build_ctx(depth):
    base = haystack_ids[:CTX].copy()
    nd_len = len(nd_ids)
    np_pos = int(CTX * depth)
    base[np_pos:min(np_pos+nd_len,CTX)] = nd_ids[:min(nd_len, CTX-np_pos)]
    base.extend(q_ids)
    return torch.tensor([base], device="cuda"), np_pos

def check(text):
    return 1.0 if KW.upper() in text.upper() else 0.0

results = {"groups": {}}

# ══════════════════════════════
# Prefill
# ══════════════════════════════
sq, np_pos = build_ctx(0.5)
hook = KVRHook(model, window_size=WS, top_k=2048, field_weight=0.0, ret_weight=1.0, device="cuda")
hook.prefill(sq)
del sq; torch.cuda.empty_cache()

# ══════════════════════════════
# G1: Cluster count sweep (offline)
# ══════════════════════════════
print("G1: Cluster count sweep")
ret = hook.retrievals[10]
nr = ret.n_stored
kp = ret._deq_k(0).float().cpu().numpy()

with torch.no_grad():
    h = model.model.layers[10].input_layernorm(model.model.embed_tokens(torch.tensor([q_ids], device="cuda")))
    q = model.model.layers[10].self_attn.q_proj(h)
    nq=32;dh=64;nkv=8;g=4
    qv = q.view(-1,nq,dh).view(-1,nkv,g,dh).mean(dim=2).mean(dim=0)[0].cpu().numpy()

group1 = []
for nc in [128, 256, 512, 1024]:
    km = MiniBatchKMeans(n_clusters=nc, random_state=42, batch_size=min(1024,nr), n_init=1)
    lbl = km.fit_predict(kp)
    sc = km.cluster_centers_ @ qv / (np.linalg.norm(km.cluster_centers_,axis=1)*np.linalg.norm(qv)+1e-8)
    rk = np.where(np.argsort(-sc) == lbl[np_pos])[0][0]
    avg = nr / nc
    print(f"  {nc:4d} cls: needle_rank={rk:3d}/{nc}, avg_tok={avg:.0f}, total≈{rk*avg:.0f}")
    group1.append({"n_clusters":nc,"needle_rank":int(rk),"avg_tok":round(avg,1),"total_tok":round(rk*avg,0)})
results["groups"]["1_cluster_sweep"] = group1

# ══════════════════════════════
# G2: Basin count sweep with n_clusters=256
# ══════════════════════════════
print("\nG2: Basin count (256 cls)")
km = MiniBatchKMeans(n_clusters=256, random_state=42, batch_size=min(1024,nr), n_init=1)
lbl = km.fit_predict(kp)
sc = km.cluster_centers_ @ qv / (np.linalg.norm(km.cluster_centers_,axis=1)*np.linalg.norm(qv)+1e-8)
rk = np.where(np.argsort(-sc) == lbl[np_pos])[0][0]
print(f"  needle_rank={rk}/256")

group2 = []
for topb in [10, 20, 30, 50]:
    found = rk < topb
    tok_cnt = topb * (nr / 256)
    print(f"  top-{topb:2d}: find={'YES' if found else 'NO'}, tok≈{tok_cnt:.0f}")
    group2.append({"top_basins":topb,"needle_found":bool(found),"total_tok":round(tok_cnt,0)})
results["groups"]["2_basin_count"] = group2

# ══════════════════════════════
# G4: Best config all depths (256 cls, native warmup 10)
# ══════════════════════════════
print("\nG4: Best config (256 cls, warmup=10, all depths)")
hook.remove()

group4 = []
for depth in [0.25, 0.5, 0.75]:
    sq, np2 = build_ctx(depth)
    # Full native baseline
    ntv = model.generate(sq, max_new_tokens=10, do_sample=False, pad_token_id=tok.eos_token_id)
    gen = [ntv[0, len(sq[0])+i:len(sq[0])+i+1].unsqueeze(0) for i in range(10)]
    del ntv
    
    # KVR with attractors
    hk = KVRHook(model, window_size=WS, top_k=2048, field_weight=0.0, ret_weight=1.0, device="cuda")
    for li in range(model.config.num_hidden_layers):
        hk.retrievals[li].n_attractors = 256
    hk.prefill(sq)
    hk.register()
    del sq; torch.cuda.empty_cache()
    
    for _ in range(MAX_NEW - 10):
        out = model(gen[-1], use_cache=False, num_logits_to_keep=1)
        gen.append(out.logits[:,-1,:].argmax(dim=-1,keepdim=True))
        hk._step += 1; hk._context_len += 1
    hk.remove()
    
    txt = tok.decode(torch.cat(gen,dim=1).cpu()[0], skip_special_tokens=True)
    acc = check(txt)
    print(f"  depth={depth:.2f}: {'PASS' if acc else 'FAIL'} {txt[:80]}")
    group4.append({"depth":depth,"niah_pass":bool(acc),"text":txt[:200]})
results["groups"]["4_best_combo"] = group4

json.dump(results, open("attractor_sweep_results.json","w"),indent=2)
print("\nSaved attractor_sweep_results.json")

