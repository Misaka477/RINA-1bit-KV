"""KVR CUDA kernels — JIT compiled via torch.utils.cpp_extension."""
import os, torch
from torch.utils.cpp_extension import load

_cuda_module = None

def _get_cuda_module():
    global _cuda_module
    if _cuda_module is not None:
        return _cuda_module
    cu_path = os.path.join(os.path.dirname(__file__), "kvr_cuda.cu")
    _cuda_module = load(
        name="kvr_cuda_kernel",
        sources=[cu_path],
        verbose=False,
    )
    return _cuda_module


def run_score_kernel_cuda(q_avg, k_packed, k_scales, cos_tbl, sin_tbl):
    """CUDA fused score search. cos_tbl: (n_stored, d//2) — one cos per pair."""
    mod = _get_cuda_module()
    n_kv, d, n_stored = k_packed.shape[1], k_packed.shape[2] * 2, k_packed.shape[0]
    q_float = q_avg.float()
    ks_float = k_scales.float()
    return mod.compute_scores(q_float, k_packed, ks_float, cos_tbl, sin_tbl, n_kv, d, n_stored)
