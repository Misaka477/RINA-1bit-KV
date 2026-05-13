"""
V quantization quality verification: int2/int3/int4/int6 V per-head vs per-dim scale.
Compares attention output cos_sim for various V quantization schemes.
"""
import os, sys, torch, torch.nn.functional as F, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

MODEL = "D:/Software_Development/Project/models/Llama-3.2-1B"
PROMPT = (
    "The history of computing spans centuries. Babbage designed the Analytical Engine in 1837. "
    "Alan Turing proposed the Turing machine in 1936. ENIAC was built in 1945. "
    "Transistors replaced vacuum tubes in the 1950s. Integrated circuits emerged in the 1960s. "
    "Intel released the 4004 CPU in 1971. The personal computer revolution began with the Altair 8800. "
) * 4


def quantize_per_dim(x, bits):
    d = x.shape[-1]
    mx = x.abs().max(dim=0).values.clamp(min=1e-8)
    n_lv = 2 ** bits
    half_n = n_lv // 2
    step = 2 * mx / n_lv
    qcodes = torch.round(x / step).clamp(-half_n, half_n - 1)
    return qcodes * step.unsqueeze(0), qcodes.to(torch.int8)


def quantize_per_head(x, bits):
    h, d = x.shape[0], x.shape[-1]
    mx = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
    n_lv = 2 ** bits
    half_n = n_lv // 2
    step = 2 * mx / n_lv
    qcodes = torch.round(x / step).clamp(-half_n, half_n - 1)
    return qcodes * step, qcodes.to(torch.int8)


@torch.no_grad()
def main():
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval(); dev = model.device

    inp = tok(PROMPT, return_tensors="pt", truncation=True, max_length=512).to(dev)
    seq = inp["input_ids"].shape[1]
    d = model.config.head_dim or (model.config.hidden_size // model.config.num_attention_heads)
    n_kv = model.config.num_key_value_heads; n_q = model.config.num_attention_heads
    g = n_q // n_kv
    print(f"Seq={seq}, d={d}, KV={n_kv}, Q={n_q}")

    with torch.no_grad(): out = model(**inp, output_hidden_states=True)
    hiddens = out.hidden_states

    bits_list = [2, 3, 4, 6]
    layers = [0, 3, 7, 11, 15]
    scales = ["per-head", "per-dim"]
    scale_label = {"per-head": "ph", "per-dim": "pd"}

    results = {}

    for li in layers:
        if li >= len(hiddens) - 1: continue
        attn = model.model.layers[li].self_attn
        layer = model.model.layers[li]
        h_states = hiddens[li][0].to(dtype=torch.float16)
        h_norm = layer.input_layernorm(h_states)

        k_pre = attn.k_proj(h_norm).float().view(seq, n_kv, d)
        v_val = attn.v_proj(h_norm).float().view(seq, n_kv, d)
        q_pre = attn.q_proj(h_norm).float().view(seq, n_q, d)

        pos_ids = torch.arange(seq, device=dev).unsqueeze(0)
        q_4d = q_pre.unsqueeze(0); k_4d = k_pre.unsqueeze(0)
        cos, sin = attn.rotary_emb(q_4d, position_ids=pos_ids)
        q_rot, k_rot = apply_rotary_pos_emb(q_4d, k_4d, cos, sin, unsqueeze_dim=2)
        q_rot = q_rot[0]; k_rot = k_rot[0]
        sc = d ** 0.5

        q_start = max(seq // 2, 0)
        q_pos = list(range(q_start, seq - 3))

        print(f"\nLayer {li} ({len(q_pos)} queries):")
        header = f"{'bits':>5s}"
        for s in scales:
            header += f"  {s:>10s}"
        print(header)
        print("-" * (5 + 11 * len(scales)))

        for b in bits_list:
            row = f"  {b:>3d}"
            for sm in scales:
                cos_list = []
                for qi in q_pos:
                    qh = q_rot[qi, 0, :]  # Q head 0 for kv head 0
                    kh = k_rot[:qi, 0, :]  # K for kv head 0
                    v_fp16 = v_val[:qi, 0, :]  # fp16 V

                    # fp16 attention (reference)
                    scores = (qh.unsqueeze(0) @ kh.T) / sc
                    w = F.softmax(scores, dim=-1)[0]
                    ref_out = w @ v_fp16

                    # Quantized V attention
                    if sm == "per-head":
                        v_q, _ = quantize_per_head(v_fp16, b)
                    else:
                        v_q, _ = quantize_per_dim(v_fp16, b)
                    q_out = w @ v_q
                    cos_list.append(F.cosine_similarity(
                        ref_out.flatten().unsqueeze(0),
                        q_out.flatten().unsqueeze(0)).item())

                mean_c = float(np.mean(cos_list))
                row += f"  {mean_c:>10.6f}"
                results[(li, b, sm)] = mean_c
            print(row)

    # Summary: best scheme
    print(f"\n{'='*60}")
    print("SUMMARY: Mean cos_sim across all layers")
    header = f"{'bits':>5s}"
    for s in scales:
        header += f"  {s:>10s}"
    print(header)
    print("-" * (5 + 11 * len(scales)))
    for b in bits_list:
        row = f"  {b:>3d}"
        for sm in scales:
            vals = [results[(li, b, sm)] for li in layers if (li, b, sm) in results]
            row += f"  {np.mean(vals):>10.6f}"
        print(row)


if __name__ == "__main__":
    main()
