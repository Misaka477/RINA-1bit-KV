"""R.I.N.A (Residual-Integrated Neural Architecture) — PyTorch Modules."""

from .residual_pursuit import (
    ResidualBinaryPursuit,
    decode_from_bases,
    differential_encode_decode,
    adaptive_encode_matrix,
    encode_matrix,
)
from .svd_noise_shaping import (
    SVDNoiseShaper,
    compute_q_covariance,
    compute_nullspace_projector,
    compute_per_head_nullspace_projectors,
    compute_shared_nullspace_projector,
)
from .differential_cancellation import (
    DifferentialCanceller,
    PerturbationStrategy,
)
from .tile_4x4 import (
    encode_tile,
    decode_tile,
    encode_4x4_matrix,
    decode_4x4_matrix,
    detect_outlier_tiles,
    dynamic_log_quantize_4bit,
    dynamic_log_dequantize_4bit,
    fixed_log_quantize_4bit,
    fixed_log_dequantize_4bit,
    linear_quantize_4bit,
    linear_dequantize_4bit,
    compute_4x4_metrics,
)

__all__ = [
    "ResidualBinaryPursuit",
    "decode_from_bases",
    "differential_encode_decode",
    "adaptive_encode_matrix",
    "encode_matrix",
    "SVDNoiseShaper",
    "compute_q_covariance",
    "compute_nullspace_projector",
    "compute_per_head_nullspace_projectors",
    "compute_shared_nullspace_projector",
    "DifferentialCanceller",
    "PerturbationStrategy",
    "encode_tile",
    "decode_tile",
    "encode_4x4_matrix",
    "decode_4x4_matrix",
    "detect_outlier_tiles",
    "dynamic_log_quantize_4bit",
    "dynamic_log_dequantize_4bit",
    "fixed_log_quantize_4bit",
    "fixed_log_dequantize_4bit",
    "linear_quantize_4bit",
    "linear_dequantize_4bit",
    "compute_4x4_metrics",
]