"""Comprehensive W + int2 V_residual test: NIAH + AR gen."""
import os, sys, json, torch, torch.nn.functional as F, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from transformers import AutoModelForCausalLM, AutoTokenizer
from modules.kvr_hook import KVRHook

MODEL = "D:/Software_Development/Project/models/Llama-3.2-1B"
DEV = None

@torch.no_grad()
def main():
    global DEV
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval(); DEV = model.device

    results = {"config": {"model": MODEL}}

    # ======== NIAH ========
    HS = "The grass is green. The sky is blue. "
    ND = "The secret password is KILO42. "
    QRY = " I just told you a secret password. The password is"
    KW = "KILO42"

    niah_results = {}
    for ctx_len in [128, 256, 512, 1024, 2048]:
        for depth in [0.25, 0.5, 0.75]:
            hs_ids = tok(HS, add_special_tokens=False)["input_ids"]
            nd_ids = tok(ND, add_special_tokens=False)["input_ids"]
            q_ids = tok(QRY, add_special_tokens=False)["input_ids"]
            needle_rep = max(0, int(ctx_len * depth / len(hs_ids)))
            seq, ri = [], 0
            while len(seq) < ctx_len:
                seq.extend(nd_ids if ri == needle_rep else hs_ids); ri += 1
            seq = seq[:ctx_len] + q_ids
            seq_t = torch.tensor([seq], device=DEV)

            native = tok.decode(model.generate(seq_t, max_new_tokens=12, do_sample=False, pad_token_id=tok.eos_token_id)[0, seq_t.shape[1]:].cpu(), skip_special_tokens=True)
            na = 1.0 if KW in native.upper() else 0.0

            fwr = KVRHook(model, window_size=64, top_k=128, field_weight=0.0, ret_weight=1.0, device=DEV)
            fwr.prefill(seq_t); fwr.register()
            try:
                gen = []
                for step in range(12):
                    cur = seq_t if step == 0 else torch.cat([seq_t] + gen, dim=1)
                    nid = model(cur, use_cache=False).logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    gen.append(nid); fwr._step += 1; fwr._context_len += 1
                fa_text = tok.decode(torch.cat(gen, dim=1).cpu()[0], skip_special_tokens=True)
            finally:
                fwr.remove()
            fa = 1.0 if KW in fa_text.upper() else 0.0

            needle_tok = needle_rep * len(hs_ids)
            niah_results[f"{ctx_len}_{depth}"] = {
                "ctx": ctx_len, "depth": depth,
                "needle_tok": needle_tok,
                "native_acc": int(na), "fwr_acc": int(fa),
                "native_text": native[:60], "fwr_text": fa_text[:60],
            }
            print(f"NIAH ctx={ctx_len:>4d} d={depth:.2f} needle@{needle_tok}  native={int(na)} fwr={int(fa)}  {native[:40]}")

    nat_pass = sum(1 for v in niah_results.values() if v["native_acc"])
    fwr_pass = sum(1 for v in niah_results.values() if v["fwr_acc"])
    print(f"\nNIAH: native={nat_pass}/15 FWR={fwr_pass}/15 ({fwr_pass/15*100:.0f}%)")
    results["niah"] = niah_results

    # ======== AR GEN (long context, window=64 so retrieval active) ========
    print(f"\n{'='*50}")
    print(f"AR GENERATION TEST (60 tok, win=64, w+int2res V)")
    print(f"{'='*50}")

    ar_inp = tok("The capital of France is Paris, a city known for its rich history and culture. The city", return_tensors="pt").to(DEV)
    native_out = model.generate(**ar_inp, max_new_tokens=60, do_sample=False, pad_token_id=tok.eos_token_id, return_dict_in_generate=True, output_logits=True)
    native_new = tok.decode(native_out.sequences[0, ar_inp["input_ids"].shape[1]:].cpu(), skip_special_tokens=True)
    native_full = tok.decode(native_out.sequences[0].cpu(), skip_special_tokens=True)

    fwr = KVRHook(model, window_size=64, top_k=128, field_weight=0.0, ret_weight=1.0, device=DEV)
    fwr.prefill(ar_inp["input_ids"]); fwr.register()
    gen_ids, js_vals = [], []
    try:
        for step in range(60):
            cur = ar_inp["input_ids"] if step == 0 else torch.cat([ar_inp["input_ids"]] + gen_ids, dim=1)
            out2 = model(cur, use_cache=False)
            logits = out2.logits[:, -1, :].float()
            nid = logits.argmax(dim=-1, keepdim=True)
            gen_ids.append(nid)
            fwr._step += 1; fwr._context_len += 1
            if step < len(native_out.logits):
                nl = native_out.logits[step][0].float().cpu()
                ps = F.softmax(logits.cpu(), -1).clamp(1e-12)
                qs = F.softmax(nl, -1).clamp(1e-12)
                m = 0.5 * (ps + qs)
                js_vals.append(float(0.5 * (F.kl_div(ps.log().clamp(-50, 50), m, reduction='batchmean') + F.kl_div(qs.log().clamp(-50, 50), m, reduction='batchmean'))))
    finally:
        fwr.remove()

    fwr_new = tok.decode(torch.cat(gen_ids, dim=1).cpu()[0], skip_special_tokens=True)
    fwr_full = tok.decode(torch.cat([ar_inp["input_ids"].cpu(), torch.cat(gen_ids, dim=1).cpu()], dim=1)[0], skip_special_tokens=True)

    results["ar_generation"] = {
        "native": {"full": native_full, "new": native_new},
        "fwr_win64": {"full": fwr_full, "new": fwr_new, "js_mean": float(np.mean(js_vals)) if js_vals else 0},
    }

    print(f"NATIVE: {native_new[:150]}")
    print(f"FWR:    {fwr_new[:150]}")
    print(f"JS mean={results['ar_generation']['fwr_win64']['js_mean']:.4f}")

    json.dump(results, open("eval_fwr_wres.json", "w"), indent=2, ensure_ascii=False)
    print(f"\nSaved eval_fwr_wres.json")

if __name__ == "__main__":
    main()


