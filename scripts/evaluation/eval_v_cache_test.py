"""
AR gen test: window=2048 + retrieval with various v_cache_bits for retrieved tokens.
Window tokens use fp16 V (exact). Retrieved tokens use quantized V (simulated cache).
"""
import os, sys, json, torch, torch.nn.functional as F, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from transformers import AutoModelForCausalLM, AutoTokenizer
from modules.kvr_hook import KVRHook

MODEL = "D:/Software_Development/Project/models/Llama-3.2-1B"
PROMPT = "The capital of France is Paris, a city known for its rich history and culture. The city"


def js_div(p, q):
    p_s = F.softmax(p, -1).clamp(1e-12)
    q_s = F.softmax(q, -1).clamp(1e-12)
    m = 0.5 * (p_s + q_s)
    return 0.5 * (F.kl_div(p_s.log().clamp(-50, 50), m, reduction='batchmean') +
                  F.kl_div(q_s.log().clamp(-50, 50), m, reduction='batchmean'))


@torch.no_grad()
def main():
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval(); dev = model.device

    inp = tok(PROMPT, return_tensors="pt").to(dev)

    native_out = model.generate(**inp, max_new_tokens=50, do_sample=False,
        pad_token_id=tok.eos_token_id, return_dict_in_generate=True, output_logits=True)
    native_text = tok.decode(native_out.sequences[0, inp["input_ids"].shape[1]:].cpu(), skip_special_tokens=True)
    print(f"Native: {native_text[:80]}\n")

    for vcb in [0, 4, 6, 8]:
        fwr = KVRHook(model, window_size=64, top_k=128,
                      field_weight=0.0, ret_weight=1.0, device=dev)
        for ri in fwr.retrievals:
            ri.v_cache_bits = vcb
        fwr.prefill(inp["input_ids"])
        fwr.register()
        js_vals, gen_ids = [], []
        try:
            for step in range(50):
                cur = inp["input_ids"] if step == 0 else torch.cat([inp["input_ids"]] + gen_ids, dim=1)
                out = model(cur, use_cache=False)
                logits = out.logits[:, -1, :].float()
                nid = logits.argmax(dim=-1, keepdim=True)
                gen_ids.append(nid)
                fwr._step += 1; fwr._context_len += 1
                if step < len(native_out.logits):
                    nl = native_out.logits[step][0].float().cpu()
                    js = float(js_div(logits.cpu(), nl))
                    js_vals.append(js)
        finally:
            fwr.remove()
        text = tok.decode(torch.cat(gen_ids, dim=1).cpu()[0], skip_special_tokens=True)
        print(f"v_cache_bits={vcb}: JS mean={np.mean(js_vals):.4f} max={max(js_vals):.4f}")
        print(f"  text: {text[:100]}\n")


if __name__ == "__main__":
    main()


