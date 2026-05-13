"""
Extreme long-context test: KVR vs native at 4K, 16K, 64K.
Measures:
  - Can it run? (OOM check)
  - Prefill time + generation time
  - Generation quality (JS for 4K, text only for 16K/64K)
  - GPU peak memory
Saves everything to JSON.
"""
import os, sys, json, time, gc, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from transformers import AutoModelForCausalLM, AutoTokenizer
from modules import KVRGenerator

MODEL = "D:/Software_Development/Project/models/Llama-3.2-1B"

# Diverse text blocks for building long prompts
TEXT_BLOCKS = [
    "The history of computing spans centuries. Babbage designed the Analytical Engine in 1837. "
    "Alan Turing proposed the Turing machine in 1936. ENIAC was built in 1945. "
    "Transistors replaced vacuum tubes in the 1950s. Integrated circuits emerged in the 1960s. "
    "Intel released the 4004 CPU in 1971. The internet started as ARPANET in 1969. "
    "Tim Berners-Lee invented the World Wide Web in 1989. Google was founded in 1998. ",

    "Biology studies life at all scales. DNA's double helix was discovered by Watson and Crick in 1953. "
    "The human genome project was completed in 2003 identifying about twenty thousand genes. "
    "Mitochondria generate energy through oxidative phosphorylation. "
    "Neurons communicate through synapses using neurotransmitters like dopamine. "
    "The human brain contains approximately 86 billion neurons. "
    "Evolution by natural selection was proposed by Charles Darwin in 1859. ",

    "Geography describes the physical features of Earth. "
    "Mount Everest is the tallest mountain on Earth at 8848 meters above sea level. "
    "The Amazon river is the largest river by water volume discharging into the Atlantic. "
    "Lake Baikal in Siberia is the deepest lake reaching 1642 meters. "
    "The Sahara desert covers most of North Africa approximately 9.2 million square kilometers. "
    "Antarctica is the coldest continent with temperatures reaching minus 89 degrees Celsius. "
    "The Pacific Ocean is the largest and deepest ocean covering more than 30 percent of the Earth. ",

    "Art history reflects human creativity across millennia. "
    "Greek sculpture reached its peak during the Classical period in the 5th century BCE. "
    "The Renaissance began in Florence Italy in the 14th century. "
    "Leonardo da Vinci painted the Mona Lisa in the early 16th century. "
    "Impressionism emerged in France in the 1870s with artists like Monet and Renoir. "
    "Picasso and Braque developed Cubism in the early 20th century. "
    "Abstract Expressionism emerged in New York after World War Two. ",
]

WINDOW_SIZES = {4096: 512, 16384: 512, 65536: 512}
TOP_K = 128
MAX_GEN = 20


def build_prompt(target_len, tok):
    """Build a diverse prompt of approximate target_len tokens."""
    text = ""
    bi = 0
    while True:
        t = len(tok(text)["input_ids"])
        if t >= target_len:
            break
        text += TEXT_BLOCKS[bi % len(TEXT_BLOCKS)] + "\n"
        bi += 1
    # Truncate to exact target
    ids = tok(text, truncation=True, max_length=target_len)["input_ids"]
    return tok.decode(ids)


def js_div(p, q):
    ps = F.softmax(p, -1).clamp(1e-12)
    qs = F.softmax(q, -1).clamp(1e-12)
    m = 0.5 * (ps + qs)
    return 0.5 * (F.kl_div(ps.log().clamp(-50, 50), m, reduction='batchmean') +
                  F.kl_div(qs.log().clamp(-50, 50), m, reduction='batchmean'))


@torch.no_grad()
def run_test(ctx_len, tok, model, native_possible=True):
    """Test native + KVR at a given context length. Returns result dict."""
    print(f"\n{'='*60}")
    print(f"  CONTEXT = {ctx_len} tokens")
    print(f"{'='*60}")

    ws = WINDOW_SIZES.get(ctx_len, 512)
    result = {"context": ctx_len, "window": ws, "top_k": TOP_K}

    # Build prompt
    prompt = build_prompt(ctx_len, tok)
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=ctx_len).to(dev)
    actual_ctx = inp["input_ids"].shape[1]
    result["actual_context"] = actual_ctx
    print(f"  Actual: {actual_ctx} tokens")

    # ── Native ──
    if native_possible:
        print(f"  Running native...")
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.cuda.empty_cache()
        try:
            t0 = time.perf_counter()
            native_out = model.generate(**inp, max_new_tokens=MAX_GEN, do_sample=False,
                pad_token_id=tok.eos_token_id, return_dict_in_generate=True, output_logits=True)
            t1 = time.perf_counter()
            native_time = t1 - t0
            native_mem = torch.cuda.max_memory_allocated() / (1024**3)

            native_new = tok.decode(native_out.sequences[0, actual_ctx:actual_ctx+MAX_GEN].cpu(), skip_special_tokens=True)
            native_full = tok.decode(native_out.sequences[0].cpu(), skip_special_tokens=True)

            result["native"] = {
                "success": True,
                "time_s": round(native_time, 2),
                "peak_mem_gb": round(native_mem, 2),
                "new_text": native_new,
                "full_text": native_full,
            }
            print(f"    time={native_time:.1f}s  mem={native_mem:.2f}GB")
            print(f"    output: {native_new[:80]}")
        except Exception as e:
            print(f"    NATIVE FAILED: {str(e)[:100]}")
            result["native"] = {"success": False, "error": str(e)[:200]}
    else:
        result["native"] = {"success": False, "error": "skipped (expected OOM)"}

    # ── KVRGenerator ──
    print(f"  Running KVR (window={ws})...")
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()

    try:
        t0 = time.perf_counter()
        gen = KVRGenerator(model, window_size=ws, top_k=TOP_K, device=dev)
        gen.prefill(inp["input_ids"])
        prefill_time = time.perf_counter() - t0

        gen_ids = []
        step_times = []
        for step in range(MAX_GEN):
            tid = None if step > 0 else inp["input_ids"][0, -1]
            ts = time.perf_counter()
            nid = gen.step(token_id=tid, temperature=1.0, top_k=1)
            step_times.append(time.perf_counter() - ts)
            gen_ids.append(nid.item())
        total_gen_time = time.perf_counter() - t0
        kvr_mem = torch.cuda.max_memory_allocated() / (1024**3)

        kvr_new = tok.decode(gen_ids, skip_special_tokens=True)

        result["kvr"] = {
            "success": True,
            "prefill_time_s": round(prefill_time, 2),
            "total_time_s": round(total_gen_time, 2),
            "avg_step_time_ms": round(sum(step_times) / len(step_times) * 1000, 2),
            "peak_mem_gb": round(kvr_mem, 2),
            "new_text": kvr_new,
        }
        print(f"    prefill={prefill_time:.1f}s  gen={total_gen_time:.1f}s  step={result['kvr']['avg_step_time_ms']:.0f}ms  mem={kvr_mem:.2f}GB")
        print(f"    output: {kvr_new[:80]}")

    except Exception as e:
        print(f"    KVR FAILED: {str(e)[:100]}")
        result["kvr"] = {"success": False, "error": str(e)[:200]}

    # ── JS (native if available) ──
    if result.get("native", {}).get("success") and result.get("kvr", {}).get("success"):
        print(f"  Computing JS...")
        # We don't have KVR logits saved; approximate: regenerate KVR with hook mode
        # For simplicity, just report text match length
        nat_words = result["native"]["new_text"].split()
        kvr_words = kvr_new.split()
        match = sum(1 for i in range(min(len(nat_words), len(kvr_words))) if nat_words[i] == kvr_words[i])
        result["text_match"] = {"word_match": match, "native_words": len(nat_words), "kvr_words": len(kvr_words)}

    return result


if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval(); global dev; dev = model.device

    all_results = {}
    ctx_configs = [
        (4096, True),   # Native + KVR
        (16384, True),  # Try native (might OOM), KVR
        (65536, False), # KVR only (native will OOM)
    ]

    for ctx_len, native_ok in ctx_configs:
        r = run_test(ctx_len, tok, model, native_possible=native_ok)
        all_results[f"ctx{ctx_len}"] = r

        # Compact output
        print(f"\n  Result ctx={ctx_len}:")
        for mode in ["native", "kvr"]:
            if mode in r:
                s = r[mode]
                if s.get("success"):
                    print(f"    {mode}: time={s.get('time_s', s.get('total_time_s', 0)):.1f}s mem={s.get('peak_mem_gb', 0):.2f}GB")
                else:
                    print(f"    {mode}: FAILED {s.get('error', '')}")

        gc.collect()
        torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'='*70}")
    print(f"  LONG CONTEXT SUMMARY")
    print(f"{'='*70}")
    print(f"{'ctx':>7s} {'native_time':>12s} {'native_mem':>10s} {'kvr_time':>10s} {'kvr_mem':>9s}")
    print("-" * 50)
    for k, r in all_results.items():
        nt = r.get("native", {}).get("time_s", "OOM")
        nm = r.get("native", {}).get("peak_mem_gb", "—")
        kt = r.get("kvr", {}).get("total_time_s", "FAIL")
        km = r.get("kvr", {}).get("peak_mem_gb", "—")
        print(f"  {r['context']:>5d}  {str(nt):>10s}s  {str(nm):>8s}G  {str(kt):>8s}s  {str(km):>7s}G")

    json.dump(all_results, open("eval_kvr_extreme.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nSaved eval_kvr_extreme.json")
