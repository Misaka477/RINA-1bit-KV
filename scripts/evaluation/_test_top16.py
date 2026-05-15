"""Test per-basin top-16 NIAH."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from transformers import AutoModelForCausalLM, AutoTokenizer
from modules.kvr_hook import KVRHook

MODEL = "D:/Software_Development/Project/models/Llama-3.2-1B"
haystack_text = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_pride.txt"), "r", encoding="utf-8").read()
ND = "The secret password is KILO42. "
Q = " I just told you a secret password. The password is"
KW = "KILO42"
WS = 512; CTX = 16384; WARMUP = 10

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tok.pad_token = tok.eos_token
nd_ids = tok(ND, add_special_tokens=False)["input_ids"]
q_ids = tok(Q, add_special_tokens=False)["input_ids"]
chap_idx = haystack_text.find("CHAPTER")
haystack_ids = tok(haystack_text[chap_idx:chap_idx+200000], add_special_tokens=False)["input_ids"]

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
model.eval()

results = {}
for depth in [0.25, 0.5, 0.75]:
    np_pos = int(CTX * depth)
    base = haystack_ids[:CTX].copy()
    base[np_pos:min(np_pos+len(nd_ids),CTX)] = nd_ids[:min(len(nd_ids), CTX-np_pos)]
    base.extend(q_ids)
    sq = torch.tensor([base], device="cuda")

    hook = KVRHook(model, window_size=WS, top_k=2048, field_weight=0.0, ret_weight=1.0, device="cuda")
    for li in range(model.config.num_hidden_layers):
        hook.retrievals[li].top_per_basin = 16
    hook.prefill(sq)
    hook.register()

    native_out = model.generate(sq, max_new_tokens=WARMUP, do_sample=False, pad_token_id=tok.eos_token_id)
    gen = [native_out[0, len(sq[0])+i:len(sq[0])+i+1].unsqueeze(0) for i in range(WARMUP)]
    del native_out; del sq; torch.cuda.empty_cache()

    for _ in range(40):
        out = model(gen[-1], use_cache=False, num_logits_to_keep=1)
        gen.append(out.logits[:,-1,:].argmax(dim=-1,keepdim=True))
        hook._step += 1; hook._context_len += 1
    hook.remove()

    txt = tok.decode(torch.cat(gen,dim=1).cpu()[0], skip_special_tokens=True)
    acc = 1.0 if KW.upper() in txt.upper() else 0.0
    print(f"depth={depth:.2f}: {'PASS' if acc else 'FAIL'} tok={len(gen)} {txt[:60]}")
    results[depth] = {"acc": acc, "text": txt[:100]}

print(f"\nTotal: {sum(1 for v in results.values() if v['acc'])}/3")
