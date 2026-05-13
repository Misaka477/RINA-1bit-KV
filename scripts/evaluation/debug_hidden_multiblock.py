"""Compare hidden states: block prefill vs native forward for 512 tokens."""
import sys, torch, torch.nn.functional as F
sys.path.insert(0, ".")
from transformers import AutoModelForCausalLM, AutoTokenizer
from modules.kvr_hook import KVRHook, _apply_rotary

MODEL = "D:/Software_Development/Project/models/Llama-3.2-1B"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
model.eval(); dev = model.device

prompt = "The history of computing spans centuries. Babbage designed the Analytical Engine. " * 50
inp = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(dev)
n = inp["input_ids"].shape[1]
print(f"Prompt: {n} tokens")

# Native: full forward with flash attention
with torch.no_grad():
    native = model(**inp, output_hidden_states=True)

# Block prefill: capture hidden states after each layer
# We need to instrument the kvr_hook to get intermediate h values
kvr = KVRHook(model, window_size=2048, top_k=128, device=dev)

# Run block prefill with hidden state capture
hidden_size = model.config.hidden_size
n_kv = model.config.num_key_value_heads; n_q = model.config.num_attention_heads
g = n_q // n_kv; d = model.config.head_dim or (hidden_size // n_q)

h = model.model.embed_tokens(inp["input_ids"])[0].to(torch.float16)
pf_k = [None] * kvr.n_layers; pf_v = [None] * kvr.n_layers

for li in range(kvr.n_layers):
    layer = model.model.layers[li]; attn = layer.self_attn
    for bi in range(0, n, 512):
        bs = bi; be = min(bs + 512, n); bsz = be - bs
        h_block = h[bs:be]

        h_norm = layer.input_layernorm(h_block)
        q = attn.q_proj(h_norm).view(bsz, n_q, d)
        k_pre = attn.k_proj(h_norm).view(bsz, n_kv, d)
        v = attn.v_proj(h_norm).view(bsz, n_kv, d)

        pos = torch.arange(bs, be, device=dev)
        c = kvr.cos_tbl[pos].to(h_norm.dtype); s = kvr.sin_tbl[pos].to(h_norm.dtype)
        cq = c.unsqueeze(1).expand(-1, n_q, -1).reshape(-1, d)
        sq = s.unsqueeze(1).expand(-1, n_q, -1).reshape(-1, d)
        ck = c.unsqueeze(1).expand(-1, n_kv, -1).reshape(-1, d)
        sk = s.unsqueeze(1).expand(-1, n_kv, -1).reshape(-1, d)
        q_rot = _apply_rotary(q.reshape(-1, d), cq, sq).view(bsz, n_q, d)
        k_rot = _apply_rotary(k_pre.reshape(-1, d), ck, sk).view(bsz, n_kv, d)

        kf = k_rot.float(); vf = v.float()
        if pf_k[li] is None:
            pf_k[li] = kf; pf_v[li] = vf
        else:
            pf_k[li] = torch.cat([pf_k[li], kf], dim=0)
            pf_v[li] = torch.cat([pf_v[li], vf], dim=0)

        all_k = pf_k[li]; all_v = pf_v[li]
        if bsz == 0: continue
        n_past = all_k.shape[0] - bsz

        qg = q_rot.float().reshape(bsz, n_kv, g, d)
        n_total = all_k.shape[0]
        gs = 512
        rmax = torch.full((bsz, n_kv, g, 1), float("-inf"), device=dev, dtype=torch.float32)
        rsum = torch.zeros(bsz, n_kv, g, 1, device=dev, dtype=torch.float32)
        aout = torch.zeros(bsz, n_kv, g, d, device=dev, dtype=torch.float32)

        for cstart in range(0, n_total, gs):
            cend = min(cstart + gs, n_total)
            kc = all_k[cstart:cend]; vc = all_v[cstart:cend]
            s = torch.einsum("bngd, cnd -> bngc", qg, kc) / (d ** 0.5)
            if cstart >= n_past:
                for i in range(bsz):
                    if i + 1 < s.shape[-1]:
                        s[i, :, :, i + 1:] = float("-inf")
            nm = torch.maximum(rmax, s.amax(dim=-1, keepdim=True))
            es = torch.exp(s - nm)
            rsum = rsum * torch.exp(rmax - nm) + es.sum(dim=-1, keepdim=True)
            aout = aout * torch.exp(rmax - nm) + torch.einsum("bngc, cnd -> bngd", es, vc)
            rmax = nm
        attn_out = (aout / rsum).reshape(bsz, -1)
        h[bs:be] = h_block + attn.o_proj(attn_out.half())
        h_norm2 = layer.post_attention_layernorm(h[bs:be])
        h[bs:be] = h[bs:be] + layer.mlp(h_norm2)

    pf_k[li] = None; pf_v[li] = None

    # Compare with native
    native_h = native.hidden_states[li + 1][0]
    cos = F.cosine_similarity(h.flatten().unsqueeze(0), native_h.flatten().unsqueeze(0)).item()
    print(f"Layer {li:>2d}: cos={cos:.8f}")
