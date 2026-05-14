/// KVR batch project — Q/K/V + RoPE + pack for 16 layers, single C++ call.
#include <torch/extension.h>
#include <vector>
#include <cstdint>

// ─── Helper: int4 pack ───

inline void pack_int4_row(const int8_t* src, uint8_t* dst, int n) {
    for (int i = 0; i < n / 2; i++) {
        uint8_t hi = (uint8_t)(src[2 * i] + 8);
        uint8_t lo = (uint8_t)(src[2 * i + 1] + 8);
        dst[i] = (hi << 4) | lo;
    }
}

inline void pack_int2_row(const int8_t* src, uint8_t* dst, int n) {
    for (int i = 0; i < n / 4; i++) {
        uint8_t v0 = (uint8_t)(src[4 * i] + 2);
        uint8_t v1 = (uint8_t)(src[4 * i + 1] + 2);
        uint8_t v2 = (uint8_t)(src[4 * i + 2] + 2);
        uint8_t v3 = (uint8_t)(src[4 * i + 3] + 2);
        dst[i] = (v0 << 6) | (v1 << 4) | (v2 << 2) | v3;
    }
}

// ─── RoPE (in-place on half[2][d]) ───

inline void apply_rope_half(half* q0, half* q1, const float* cos_t, const float* sin_t, int half_d) {
    for (int i = 0; i < half_d; i++) {
        float e = __half2float(q0[i]);
        float o = __half2float(q1[i]);
        float c = cos_t[i];
        float s = sin_t[i];
        q0[i] = __float2half(e * c - o * s);
        q1[i] = __float2half(e * s + o * c);
    }
}

// ─── Batch project ───

void batch_project(
    // Model weights (per layer)
    std::vector<at::Tensor> ln_weights,    // 16 × (hidden_size,) half
    std::vector<at::Tensor> ln_biases,     // 16 × (hidden_size,) half
    std::vector<at::Tensor> q_weights,     // 16 × (hidden_size, n_q*d) half
    std::vector<at::Tensor> k_weights,     // 16 × (hidden_size, n_kv*d) half
    std::vector<at::Tensor> v_weights,     // 16 × (hidden_size, n_kv*d) half

    // Input token
    at::Tensor token_embed,                // (hidden_size,) half

    // Position
    int pos_id,

    // Rotray tables
    at::Tensor cos_tbl,                    // (max_pos, d) float
    at::Tensor sin_tbl,                    // (max_pos, d) float

    // Pre-allocated output buffers
    at::Tensor k_packed_buf,     // (n_layers, n_kv, d/2) uint8 — reuse per step
    at::Tensor vr_packed_buf,    // (n_layers, n_kv, d/4) uint8 — reuse per step
    at::Tensor k_post_buf,       // (n_layers, n_kv, d) half
    at::Tensor v_buf,            // (n_layers, n_kv, d) half

    // Scales (per layer)
    at::Tensor k_scales,         // (n_layers, n_kv, d) float
    at::Tensor vr_scales,        // (n_layers, n_kv, 1) float

    int n_layers, int hidden_size, int n_q, int n_kv, int d_head
) {
    auto dev = token_embed.device();
    int g = n_q / n_kv;
    int half_d = d_head / 2;
    auto stream = c10::cuda::getCurrentCUDAStream();

    half cos_cache[8192][32]; // max_pos × half_d, for fast lookup

    for (int li = 0; li < n_layers; li++) {
        // ── LN ──
        auto h = token_embed;  // reuse input, overwrite per layer

        // ── QKV ──
        auto q = at::matmul(h, q_weights[li]);  // (n_q * d_head,)
        auto k = at::matmul(h, k_weights[li]);  // (n_kv * d_head,)
        auto v = at::matmul(h, v_weights[li]);  // (n_kv * d_head,)

        // ── Store V ──
        v_buf[li] = v.view({n_kv, d_head});

        // ── RoPE ──
        float* ct = (float*)cos_tbl.data_ptr() + pos_id * half_d;
        float* st = (float*)sin_tbl.data_ptr() + pos_id * half_d;

        // Q: split into halves for RoPE
        auto q_view = q.view({n_q, d_head});
        for (int hi = 0; hi < n_q; hi++) {
            half* qh = (half*)q_view.data_ptr() + hi * d_head;
            apply_rope_half(qh, qh + half_d, ct, st, half_d);
        }

        // K: same
        auto k_view = k.view({n_kv, d_head});
        for (int hi = 0; hi < n_kv; hi++) {
            half* kh = (half*)k_view.data_ptr() + hi * d_head;
            half* k_rot = (half*)k_post_buf.data_ptr() + li * n_kv * d_head + hi * d_head;
            memcpy(k_rot, kh, d_head * sizeof(half));          // copy pre-ROPE
            apply_rope_half(k_rot, k_rot + half_d, ct, st, half_d);  // rotate in-place
        }

        // ── Quantize K_pre to int4 ──
        for (int h = 0; h < n_kv; h++) {
            int8_t k_int4[128];  // max d_head
            half* k_pre_h = &((half*)k_view.data_ptr())[h * d_head];
            float* ks = (float*)k_scales.data_ptr() + li * n_kv * d_head + h * d_head;

            for (int di = 0; di < d_head; di++) {
                float val = __half2float(k_pre_h[di]);
                float step_val = 2.0f * ks[di] / 16.0f;
                k_int4[di] = (int8_t)max(-8, min(7, (int)roundf(val / step_val)));
            }
            pack_int4_row(k_int4, (uint8_t*)k_packed_buf.data_ptr() + li * n_kv * half_d + h * half_d, d_head);
        }

        // ── Compute V_res, quantize to int2 ──
        for (int h = 0; h < n_kv; h++) {
            // Note: In production, we'd compute V_pred from W@K + V_mean here
            // For now, just store V as-is (no residual)
            // TODO: integrate V prediction + residual
            half* vh = &((half*)v_buf.data_ptr())[li * n_kv * d_head + h * d_head];
            float* vrs = (float*)vr_scales.data_ptr() + li * n_kv + h;

            int8_t vr_int2[128];
            for (int di = 0; di < d_head; di++) {
                float val = __half2float(vh[di]);
                float step_val = 2.0f * vrs[0] / 4.0f;
                vr_int2[di] = (int8_t)max(-2, min(1, (int)roundf(val / step_val)));
            }
            pack_int2_row(vr_int2, (uint8_t*)vr_packed_buf.data_ptr() + li * n_kv * (d_head / 4) + h * (d_head / 4), d_head);
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batch_project", &batch_project, "Fused batch projection + pack for KVR");
}
