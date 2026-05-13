"""
Verify: W-predicted V + int2 residual V vs int4 V quality.
Goal: int2 residual should give cos matching int4 V (~0.99).
"""
import os, sys, torch, torch.nn.functional as F, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "D:/Software_Development/Project/models/Llama-3.2-1B"
PROMPT = ("The history of computing spans centuries. Babbage designed the Analytical Engine in 1837. "
          "Alan Turing proposed the Turing machine in 1936. "
         ) * 8

@torch.no_grad()
def quantize_per_head(x, bits):
    h, d = x.shape[0], x.shape[-1]
    mx = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    n_lv = 2 ** bits; half = n_lv // 2
    step = 2 * mx / n_lv
    q = torch.round(x / step).clamp(-half, half - 1)
    return q * step, q.to(torch.int8)

@torch.no_grad()
def main():
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval(); dev = model.device

    inp = tok(PROMPT, return_tensors="pt", truncation=True, max_length=1024).to(dev)
    seq = inp["input_ids"].shape[1]
    d = model.config.head_dim or (model.config.hidden_size // model.config.num_attention_heads)
    n_kv = model.config.num_key_value_heads
    print(f"Seq={seq}, d={d}, n_kv={n_kv}")

    with torch.no_grad(): out = model(**inp, output_hidden_states=True)
    hiddens = out.hidden_states
    half = seq // 2

    for li in range(len(hiddens) - 1):
        attn = model.model.layers[li].self_attn
        layer = model.model.layers[li]
        h = hiddens[li][0].to(dtype=torch.float16)
        h_norm = layer.input_layernorm(h)

        K_pre = attn.k_proj(h_norm).float().view(seq, n_kv * d)
        V = attn.v_proj(h_norm).float().view(seq, n_kv * d)

        K_train, V_train = K_pre[:half], V[:half]
        K_test, V_test = K_pre[half:], V[half:]

        # Fit W
        K_mu = K_train.mean(0, keepdim=True)
        V_mu = V_train.mean(0, keepdim=True)
        Kc = K_train - K_mu; Vc = V_train - V_mu
        W = (Kc.T @ Vc).T @ torch.linalg.pinv(Kc.T @ Kc)

        # W prediction
        V_w = (K_test - K_mu) @ W.T + V_mu
        cos_w = F.cosine_similarity(V_w, V_test, dim=-1).mean().item()

        # Residual: int2 per-head
        V_res = V_test - V_w
        V_res_shape = V_res.reshape(-1, n_kv, d)
        V_res_q, _ = quantize_per_head(V_res_shape.reshape(-1, d), 2)
        V_res_dq = V_res_q.view(-1, n_kv * d)
        V_total = V_w + V_res_dq
        cos_wr = F.cosine_similarity(V_total, V_test, dim=-1).mean().item()

        # Baseline: int4 V directly
        V4_shape = V_test.reshape(-1, n_kv, d)
        V4_q, _ = quantize_per_head(V4_shape.reshape(-1, d), 4)
        V4 = V4_q.view(-1, n_kv * d)
        cos_v4 = F.cosine_similarity(V4, V_test, dim=-1).mean().item()

        print(f"  L{li:>2d}  W_pred={cos_w:.6f}  W+int2res={cos_wr:.6f}  int4V={cos_v4:.6f}  best={'W+res' if cos_wr > cos_v4 else 'int4V'}")

if __name__ == "__main__":
    main()
