"""Quick 64K NIAH test — KVR only, single depth."""
import os, sys, json, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoModelForCausalLM, AutoTokenizer
from modules.kvr_hook import KVRHook

MODEL = "D:/Software_Development/Project/models/Llama-3.2-1B"
HS = "The grass is green. The sky is blue. "
ND = "The secret password is KILO42. "
Q = " I just told you a secret password. The password is"
KW = "KILO42"
CTX = 65536
WS = 2048
TOP_K = 128
MAX_NEW = 20

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
model.eval()
hs_ids = tok(HS, add_special_tokens=False)["input_ids"]
nd_ids = tok(ND, add_special_tokens=False)["input_ids"]
q_ids = tok(Q, add_special_tokens=False)["input_ids"]

results = {}
for depth in [0.25, 0.5, 0.75]:
    needle_pos = max(0, int(CTX * depth / len(hs_ids)))
    seq = []
    rep = 0
    while len(seq) < CTX:
        seq.extend(nd_ids if rep == needle_pos else hs_ids)
        rep += 1
    seq = seq[:CTX]
    seq.extend(q_ids)
    seq_t = torch.tensor([seq], device="cuda")
    n_tok = seq_t.shape[1]
    needle_tok = needle_pos * len(hs_ids)
    in_win = needle_tok >= (n_tok - len(q_ids) - WS)
    print(f"\nDepth {depth}: pos~{needle_tok} in_win={in_win}")

    torch.cuda.reset_peak_memory_stats()
    hook = KVRHook(model, window_size=WS, top_k=TOP_K,
                   field_weight=0.0, ret_weight=1.0, device="cuda")
    hook.prefill(seq_t)
    hook.register()
    # Free prefill memory
    del seq_t
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"  VRAM after prefill+gc: {mem:.2f} GB")

    gen_ids = []
    # Step 0: pass query tokens (last ~8 tokens) to generate first output
    q_tok = tok(Q, add_special_tokens=False, return_tensors="pt").to("cuda")
    for step in range(MAX_NEW):
        cur = q_tok["input_ids"] if step == 0 else gen_ids[-1]
        out = model(cur, use_cache=False, num_logits_to_keep=1)
        nid = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        gen_ids.append(nid)
        hook._step += 1
        hook._context_len += 1
    hook.remove()

    text = tok.decode(torch.cat(gen_ids, dim=1).cpu()[0],
                      skip_special_tokens=True)
    acc = KW.upper() in text.upper()
    mark = "PASS" if acc else "FAIL"
    print(f"  result: {text[:80]}")
    print(f"  {mark}")
    results[str(depth)] = {"acc": acc, "text": text, "in_window": in_win}

print(f"\n{'='*50}")
print(f"64K NIAH: {sum(1 for v in results.values() if v['acc'])}/{len(results)}")
json.dump(results, open("niah_64k_result.json", "w"), indent=2)
