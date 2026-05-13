"""
Test: Can V be linearly predicted from pre-RoPE K within each layer?
If yes, FWR stores only int4 K_pre (no V), predicts V on-the-fly.
"""
import os, sys, torch, torch.nn.functional as F, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

MODEL = "D:/Software_Development/Project/models/Llama-3.2-1B"
PROMPT = (
    "The history of computing spans centuries. Babbage designed the Analytical Engine in 1837. "
    "Alan Turing proposed the Turing machine in 1936. ENIAC was built in 1945. "
    "Transistors replaced vacuum tubes in the 1950s. Integrated circuits emerged in the 1960s."
) * 8


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
    kv_dim = n_kv * d
    print(f"Seq={seq}, d={d}, n_kv={n_kv}, kv_dim={kv_dim}")

    with torch.no_grad(): out = model(**inp, output_hidden_states=True)
    hiddens = out.hidden_states

    half = seq // 2

    for li in range(len(hiddens) - 1):
        attn = model.model.layers[li].self_attn
        layer = model.model.layers[li]
        h = hiddens[li][0].to(dtype=torch.float16)
        h_norm = layer.input_layernorm(h)

        K_pre = attn.k_proj(h_norm).float().view(seq, kv_dim)
        V = attn.v_proj(h_norm).float().view(seq, kv_dim)

        K_train = K_pre[:half]
        V_train = V[:half]
        K_test = K_pre[half:]
        V_test = V[half:]

        K_mean = K_train.mean(dim=0, keepdim=True)
        V_mean = V_train.mean(dim=0, keepdim=True)
        Kc = K_train - K_mean
        Vc = V_train - V_mean

        KtK_inv = torch.linalg.pinv(Kc.T @ Kc)
        W = (Kc.T @ Vc).T @ KtK_inv

        V_pred = (K_test - K_mean) @ W.T + V_mean

        cos = F.cosine_similarity(V_pred, V_test, dim=-1).mean().item()

        print(f"  L{li:>2d}  cos={cos:.6f}")


if __name__ == "__main__":
    main()
