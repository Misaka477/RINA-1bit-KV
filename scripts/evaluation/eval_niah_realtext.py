"""NIAH with real text haystack — Pride and Prejudice."""
import os, sys, json, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from transformers import AutoModelForCausalLM, AutoTokenizer
from modules.kvr_hook import KVRHook

MODEL = "D:/Software_Development/Project/models/Llama-3.2-1B"
ND = "The secret password is KILO42. "
Q = " I just told you a secret password. The password is"
KW = "KILO42"
WS = 512
TOP_K = 2048  # effectively no limit for 16K
MAX_NEW = 20

# Load real haystack text
haystack_path = os.path.join(os.path.dirname(__file__), "_pride.txt")
haystack_text = open(haystack_path, "r", encoding="utf-8").read()

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

nd_ids = tok(ND, add_special_tokens=False)["input_ids"]
q_ids = tok(Q, add_special_tokens=False)["input_ids"]

# Tokenize haystack and find good start point (skip Gutenberg header)
haystack_ids = tok(haystack_text[:300000], add_special_tokens=False)["input_ids"]
# Find "CHAPTER" to start from real content
chap_idx = haystack_text[:300000].find("CHAPTER")
if chap_idx >= 0:
    haystack_ids = tok(haystack_text[chap_idx:chap_idx+200000], add_special_tokens=False)["input_ids"]
print(f"Haystack tokens available: {len(haystack_ids)}")

model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
model.eval()

ctx_len = 16384
results = {}

for depth in [0.25, 0.5, 0.75]:
    nd_len = len(nd_ids)
    needle_tok_pos = int(ctx_len * depth)

    # Base: first ctx_len tokens of haystack
    base = haystack_ids[:ctx_len].copy()
    # Replace a segment at needle position with the needle
    needle_end = min(needle_tok_pos + nd_len, ctx_len)
    base[needle_tok_pos:needle_end] = nd_ids[:needle_end - needle_tok_pos]
    base.extend(q_ids)
    seq_t = torch.tensor([base], device="cuda")
    n_tok = seq_t.shape[1]

    needle_in_window = needle_tok_pos >= (n_tok - len(q_ids) - WS)
    print(f"\nContext {ctx_len}, depth={depth:.2f}, needle~tok{needle_tok_pos}, in_win={needle_in_window}")

    # === Native (no hooks, standard KV cache) ===
    native_ids = model.generate(seq_t, max_new_tokens=MAX_NEW,
                                do_sample=False, pad_token_id=tok.eos_token_id)
    native_text = tok.decode(native_ids[0, n_tok:].cpu(), skip_special_tokens=True)
    native_acc = 1.0 if KW.upper() in native_text.upper() else 0.0
    native_mark = "PASS" if native_acc else "FAIL"
    print(f"  native: {native_mark}: {native_text[:80]}")

    # === First N tokens: native (no hooks, uses KV cache internally) ===
    NATIVE_STEPS = 5
    native_out = model.generate(seq_t, max_new_tokens=NATIVE_STEPS,
                                do_sample=False, pad_token_id=tok.eos_token_id)
    gen_ids = [native_out[0, n_tok + i:n_tok + i + 1].unsqueeze(0) for i in range(NATIVE_STEPS)]
    del native_out

    # === Remaining steps: KVR (compressed) ===
    hook = KVRHook(model, window_size=WS, top_k=TOP_K, field_weight=0.0, ret_weight=1.0, device="cuda")
    hook.prefill(seq_t)
    hook.register()
    del seq_t
    torch.cuda.empty_cache()

    for step in range(MAX_NEW - NATIVE_STEPS):
        out = model(gen_ids[-1], use_cache=False, num_logits_to_keep=1)
        nid = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        gen_ids.append(nid)
        hook._step += 1
        hook._context_len += 1
    hook.remove()

    text = tok.decode(torch.cat(gen_ids, dim=1).cpu()[0], skip_special_tokens=True)
    acc = 1.0 if KW.upper() in text.upper() else 0.0
    mark = "PASS" if acc else "FAIL"
    print(f"  {mark}: {text[:80]}")
    results[str(depth)] = {"acc": acc, "native_acc": native_acc, "text": text, "native_text": native_text, "in_window": needle_in_window}

print(f"\n{'='*50}")
print(f"Real-text NIAH @ {ctx_len}: KVR={sum(1 for v in results.values() if v['acc'])}/{len(results)}, Native={sum(1 for v in results.values() if v['native_acc'])}/{len(results)}")
json.dump(results, open("niah_real_16k.json", "w"), indent=2)
print("Saved niah_real_16k.json")
