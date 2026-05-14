/// KVR CUDA kernel — fused score search.
/// Each thread: one (token, KV head). Unpack int4 + deq + RoPE + dot.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>

// ─── Kernel ───

__global__ void score_kernel(
    const float* __restrict__ q,         // (n_kv, d) — pre-averaged Q
    const uint8_t* __restrict__ k_pack, // (n_stored, n_kv, d/2)
    const float* __restrict__ k_scales,  // (n_kv, d)
    const float* __restrict__ cos_tbl,  // (max_pos, d)
    const float* __restrict__ sin_tbl,  // (max_pos, d)
    float* __restrict__ scores,          // (n_stored, n_kv)
    int n_kv, int d, int n_stored
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_stored) return;
    int kvh = blockIdx.y;

    int half = d / 2;
    float score = 0.0f;

    const uint8_t* p_token = k_pack + tid * n_kv * half + kvh * half;
    const float* sc = k_scales + kvh * d;
    const float* c_t = cos_tbl + tid * half;
    const float* s_t = sin_tbl + tid * half;

    for (int i = 0; i < half; i++) {
        uint8_t packed = p_token[i];
        float hi_val = (float)((packed >> 4) - 8);
        float lo_val = (float)((packed & 0x0F) - 8);

        float k_even = hi_val * sc[2 * i] / 8.0f;
        float k_odd  = lo_val * sc[2 * i + 1] / 8.0f;

        float cos_v = c_t[i];
        float sin_v = s_t[i];
        float k0r = fmaf(k_even, cos_v, -k_odd * sin_v);
        float k1r = fmaf(k_even, sin_v,  k_odd * cos_v);

        float q_even = q[kvh * d + 2 * i];
        float q_odd  = q[kvh * d + 2 * i + 1];

        score = fmaf(q_even, k0r, fmaf(q_odd, k1r, score));
    }
    scores[tid * n_kv + kvh] = score / sqrtf((float)d);
}

// ─── C++ wrapper ───

torch::Tensor compute_scores(
    torch::Tensor q,         // (n_kv, d) float, pre-averaged Q
    torch::Tensor k_pack,    // (n_stored, n_kv, d/2) uint8
    torch::Tensor k_scales,  // (n_kv, d) float
    torch::Tensor cos_tbl,   // (max_pos, d) float
    torch::Tensor sin_tbl,   // (max_pos, d) float
    int n_kv, int d, int n_stored
) {
    auto scores = torch::empty({n_stored, n_kv}, torch::TensorOptions().dtype(torch::kFloat32).device(q.device()));

    int threads = 256;
    dim3 grid((n_stored + threads - 1) / threads, n_kv);

    score_kernel<<<grid, threads>>>(
        q.data_ptr<float>(),
        k_pack.data_ptr<uint8_t>(),
        k_scales.data_ptr<float>(),
        cos_tbl.data_ptr<float>(),
        sin_tbl.data_ptr<float>(),
        scores.data_ptr<float>(),
        n_kv, d, n_stored
    );

    return scores;
}

// ─── PyTorch binding ───

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_scores", &compute_scores, "Fused KVR score search (unpack+deq+RoPE+dot)");
}
