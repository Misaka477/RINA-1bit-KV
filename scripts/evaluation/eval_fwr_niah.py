# -*- coding: latin-1 -*-
"""
FWR NIAH - Needle In A Haystack test."""
import os, sys, json, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from transformers import AutoModelForCausalLM, AutoTokenizer
from modules.kvr_hook import KVRHook

MODEL = "D:/Software_Development/Project/models/Llama-3.2-1B"
HAYSTACK_SENTENCE = "The grass is green. The sky is blue. "
NEEDLE_SENTENCE = "The secret password is KILO42. "
QUERY = " I just told you a secret password. The password is"
NEEDLE_KEYWORD = "KILO42"


def build_haystack(needle_pos, context_len, tok):
    hs_ids = tok(HAYSTACK_SENTENCE, add_special_tokens=False)["input_ids"]
    nd_ids = tok(NEEDLE_SENTENCE, add_special_tokens=False)["input_ids"]
    q_ids = tok(QUERY, add_special_tokens=False)["input_ids"]

    seq = []
    rep_idx = 0
    while len(seq) < context_len:
        if rep_idx == needle_pos:
            seq.extend(nd_ids)
        else:
            seq.extend(hs_ids)
        rep_idx += 1
    seq = seq[:context_len]
    seq.extend(q_ids)
    return seq, len(q_ids)


def check_answer(text, keyword):
    text_upper = text.upper()
    keyword_upper = keyword.upper()
    return 1.0 if keyword_upper in text_upper else 0.0


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL)
    p.add_argument("--context-lens", type=int, nargs="+", default=[128, 256, 512, 1024])
    p.add_argument("--needle-depths", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    p.add_argument("--window-sizes", type=int, nargs="+", default=[64, 256])
    p.add_argument("--top-k", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=15)
    p.add_argument("--json", default="eval_fwr_niah.json")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval(); dev = model.device

    nd_len = len(tok(NEEDLE_SENTENCE, add_special_tokens=False)["input_ids"])
    hs_len = len(tok(HAYSTACK_SENTENCE, add_special_tokens=False)["input_ids"])
    print(f"Model: 1B, needle_tokens={nd_len}, haystack_sent_tokens={hs_len}")
    print(f"Top-K: {args.top_k}")

    results = {}
    for ctx_len in args.context_lens:
        print(f"\n--- Context {ctx_len} tok ---")
        first_col = True
        for depth in args.needle_depths:
            needle_pos = max(0, int(ctx_len * depth / hs_len))
            needle_tok_pos = needle_pos * hs_len
            seq_ids, q_len = build_haystack(needle_pos, ctx_len, tok)
            seq_t = torch.tensor([seq_ids], device=dev)
            n_prompt = seq_t.shape[1]

            # Native
            out = model.generate(seq_t, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
            native_text = tok.decode(out[0, n_prompt:].cpu(), skip_special_tokens=True)
            native_acc = check_answer(native_text, NEEDLE_KEYWORD)

            # FWR at various window sizes
            row = {"native_acc": native_acc}
            row_answers = {"native": native_text}
            for ws in args.window_sizes:
                fwr = KVRHook(model, window_size=ws, top_k=args.top_k,
                              field_weight=0.0, ret_weight=1.0, device=dev)
                fwr.prefill(seq_t)
                fwr.register()
                try:
                    gen_ids = []
                    for step in range(args.max_new_tokens):
                        if step == 0:
                            cur = seq_t
                        else:
                            cur = torch.cat([seq_t] + gen_ids, dim=1)
                        out2 = model(cur, use_cache=False, num_logits_to_keep=1)
                        nid = out2.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        gen_ids.append(nid)
                        fwr._step += 1
                        fwr._context_len += 1
                    fwr_text = tok.decode(torch.cat(gen_ids, dim=1).cpu()[0], skip_special_tokens=True)
                    fwr_acc = check_answer(fwr_text, NEEDLE_KEYWORD)
                finally:
                    fwr.remove()

                row[f"win{ws}_acc"] = fwr_acc
                row_answers[f"win{ws}"] = fwr_text

                needle_in_win = needle_tok_pos >= (n_prompt - q_len - ws)
                row[f"win{ws}_needle_in_window"] = needle_in_win

            if first_col:
                print(f"  depth={depth:.2f} needle~tok{needle_tok_pos}  "
                      f"in_win={[row.get(f'win{ws}_needle_in_window') for ws in args.window_sizes]}")
                first_col = False
            print(f"    native={native_acc:.0f} "
                  + " ".join([f"win{ws}={row.get(f'win{ws}_acc', 0):.0f}" for ws in args.window_sizes])
                  + f"  ans: {native_text[:40]}")

            results[(ctx_len, depth)] = row

    # Summary
    print(f"\n{'='*70}")
    print("=== NIAH SUMMARY ===")
    header = f"{'ctx':>5s}  {'depth':>5s}  {'native':>7s}"
    for ws in args.window_sizes:
        header += f"  {'win'+str(ws):>7s}"
    print(header)
    print("-" * (5 + 5 + 7 + 1 + 8 * len(args.window_sizes)))
    for ctx_len in args.context_lens:
        for depth in args.needle_depths:
            r = results.get((ctx_len, depth), {})
            row = f"{ctx_len:>5d}  {depth:>5.2f}  {r.get('native_acc', 0):>7.0f}"
            for ws in args.window_sizes:
                row += f"  {r.get(f'win{ws}_acc', 0):>7.0f}"
            print(row)

    json.dump({str(k): v for k, v in results.items()}, open(args.json, "w"), indent=2)
    print(f"\nSaved {args.json}")


if __name__ == "__main__":
    main()


