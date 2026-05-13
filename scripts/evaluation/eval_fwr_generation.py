"""
FWR Generation Validation �?JS divergence vs native model at various context lengths.
Tests: window-only, window+field, window+retrieval, all-three.

Usage:
  python eval_fwr_generation.py --mode window-only --max-tokens 50
  python eval_fwr_generation.py --mode all --field-weight 0.0 --ret-weight 0.5
"""
import os, sys, json, argparse, torch, torch.nn.functional as F, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from transformers import AutoModelForCausalLM, AutoTokenizer
from modules.kvr_hook import KVRHook

MODEL = "D:/Software_Development/Project/models/Llama-3.2-1B"
SHORT_PROMPT = "The capital of France is Paris, a city known for its rich history and culture. The city"
LONG_PROMPT = (
    "The history of computing spans centuries. Babbage designed the Analytical Engine in 1837. "
    "Alan Turing proposed the Turing machine in 1936. The first electronic computer ENIAC was 1945. "
    "Transistors replaced vacuum tubes in the 1950s. Integrated circuits emerged in the 1960s. "
    "Intel released the 4004 in 1971. The Altair 8800 launched in 1975. Apple II in 1977. IBM PC 1981. "
    "ARPANET began in 1969. TCP/IP 1983. Tim Berners-Lee invented the Web in 1989. Mosaic 1993. "
    "Netscape went public 1995. Google founded 1998. Wikipedia 2001. Facebook 2004. YouTube 2005. "
    "The transformer architecture revolutionized deep learning. Attention is all you need. "
    "Large language models can understand and generate human-like text. "
    "Memory-efficient attention variants reduce the quadratic complexity of full attention. "
    "This allows processing much longer sequences than previously possible. "
    "The field of natural language processing continues to advance rapidly. "
)


def js_div(p, q):
    p_s = F.softmax(p, -1).clamp(1e-12)
    q_s = F.softmax(q, -1).clamp(1e-12)
    m = 0.5 * (p_s + q_s)
    return 0.5 * (F.kl_div(p_s.log().clamp(-50, 50), m, reduction='batchmean') +
                  F.kl_div(q_s.log().clamp(-50, 50), m, reduction='batchmean'))


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL)
    p.add_argument("--prompt", default=SHORT_PROMPT)
    p.add_argument("--max-tokens", type=int, default=30)
    p.add_argument("--window-size", type=int, default=2048)
    p.add_argument("--top-k", type=int, default=128)
    p.add_argument("--field-weight", type=float, default=None)
    p.add_argument("--ret-weight", type=float, default=1.0)
    p.add_argument("--mode", choices=["window-only", "window-field", "window-retrieval", "all"],
                   default="window-only")
    p.add_argument("--json", default="eval_fwr_generation.json")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval(); dev = model.device

    inp = tok(args.prompt, return_tensors="pt").to(dev)
    n_prompt = inp["input_ids"].shape[1]

    d = model.config.head_dim or (model.config.hidden_size // model.config.num_attention_heads)
    n_kv = model.config.num_key_value_heads; n_q = model.config.num_attention_heads
    n_layers = model.config.num_hidden_layers
    print(f"Model: layers={n_layers}, KV={n_kv}, Q={n_q}, d={d}, prompt={n_prompt} tok")
    print(f"Mode: {args.mode}, window={args.window_size}, top_k={args.top_k}, "
          f"ret_w={args.ret_weight}")

    # ── Native baseline ──
    print("\nRunning native baseline...")
    native_out = model.generate(**inp, max_new_tokens=args.max_tokens, do_sample=False,
        pad_token_id=tok.eos_token_id, return_dict_in_generate=True, output_logits=True)
    native_ids = native_out.sequences[0, n_prompt:]
    native_text = tok.decode(native_ids.cpu(), skip_special_tokens=True)
    print(f"Native: {native_text[:150]}")

    # ── Resolve mode params ──
    if args.mode == "window-only":
        rw = 0.0
    elif args.mode == "window-field":
        rw = 0.0
    elif args.mode == "window-retrieval":
        rw = args.ret_weight
    else:
        rw = args.ret_weight

    # ── FWR generation with per-step JS ──
    fwr = KVRHook(model, window_size=args.window_size, top_k=args.top_k,
                  ret_weight=rw, device=dev)
    fwr.prefill(inp["input_ids"])
    fwr.register()

    js_vals = []
    gen_ids = []
    try:
        for step in range(args.max_tokens):
            if step == 0:
                cur_input = inp["input_ids"]
            else:
                cur_input = torch.cat([inp["input_ids"], torch.cat(gen_ids, dim=-1)], dim=1)

            out = model(cur_input, use_cache=False)
            logits = out.logits[:, -1, :].float()

            next_id = logits.argmax(dim=-1, keepdim=True)
            gen_ids.append(next_id)
            fwr._step += 1
            fwr._context_len += 1

            if step < len(native_out.logits):
                nl = native_out.logits[step][0].float().cpu()
                js = float(js_div(logits.cpu(), nl))
                js_vals.append(js)
                if step < 5 or step % 10 == 0:
                    ntok = tok.decode([int(nl.argmax())])
                    ftok = tok.decode([int(logits.argmax())])
                    print(f"  Step {step}: JS={js:.4f}  Native={ntok}  FWR={ftok}")
    finally:
        fwr.remove()

    gen_text = tok.decode(torch.cat(gen_ids, dim=-1).cpu()[0], skip_special_tokens=True)
    print(f"\nFWR: {gen_text[:150]}")
    if js_vals:
        print(f"JS mean={np.mean(js_vals):.4f}, max={max(js_vals):.4f}, "
              f"min={min(js_vals):.4f}, n_steps={len(js_vals)}")

    report = {
        "js": js_vals,
        "js_mean": float(np.mean(js_vals)) if js_vals else 0,
        "js_max": float(max(js_vals)) if js_vals else 0,
        "js_min": float(min(js_vals)) if js_vals else 0,
        "native": native_text,
        "fwr": gen_text,
        "args": vars(args),
        "n_prompt": n_prompt,
    }
    json.dump(report, open(args.json, "w"), indent=2)
    print(f"\nSaved {args.json}")


if __name__ == "__main__":
    main()


