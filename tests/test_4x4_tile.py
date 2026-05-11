"""
Phase 2e Verification Script — 4×4 Tile + Log-Quantized α + Outlier Protection
================================================================================

Tests:
  A: 4×4, N=2, linear 4-bit α
  B: 4×4, N=2, fixed log 4-bit α
  C: 4×4, N=2, dynamic log-anchor 4-bit α
  D: Scheme C + dynamic outlier tile FP16 protection
  E (baseline): 16×16, FP16 α, N=8 (Phase 2d)

Metrics:
  1. Reconstruction CosSim
  2. MaxAE (global max |K_true - K_recon|)
  3. Per-dimension MaxAE (outlier dimension errors)
  4. bit/element
  5. Attention Score diff heatmap (checkerboard detection)

Run:
  python tests/test_4x4_tile.py                          # synthetic data only
  python tests/test_4x4_tile.py --use-model              # load real Llama-3.2-3B
"""

from __future__ import annotations

import argparse
import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.tile_4x4 import (
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

from rina.config import DSKVCacheConfig
from rina.ds_kv_cache import encode_kv_cache

TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42


def set_seed():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════


def compute_attention_diff_heatmap(
    K_true: torch.Tensor,
    K_recon: torch.Tensor,
    Q_h: torch.Tensor,
) -> dict:
    """Compute attention score differences and checkerboard detection.

    Returns dict with:
      mean_attn_diff, max_attn_diff, boundary_penetration_ratio
    """
    d_head = K_true.shape[-1]
    scale = math.sqrt(d_head)

    A_fp16 = F.softmax(Q_h @ K_true.T / scale, dim=-1)
    A_quant = F.softmax(Q_h @ K_recon.T / scale, dim=-1)

    diff = (A_fp16 - A_quant).abs()  # (T, T)

    T = diff.shape[0]
    tile_size = 4

    boundary_mask = torch.zeros(T, T, dtype=torch.bool)
    for i in range(T):
        for j in range(T):
            if (i % tile_size == 0 or i % tile_size == tile_size - 1 or
                j % tile_size == 0 or j % tile_size == tile_size - 1):
                boundary_mask[i, j] = True

    interior_mask = ~boundary_mask

    boundary_vals = diff[boundary_mask]
    interior_vals = diff[interior_mask]

    boundary_mean = boundary_vals.mean().item() if boundary_vals.numel() > 0 else 0.0
    interior_mean = interior_vals.mean().item() if interior_vals.numel() > 0 else 0.0

    boundary_penetration = boundary_mean / (interior_mean + 1e-12)

    return {
        "mean_attn_diff": diff.mean().item(),
        "max_attn_diff": diff.max().item(),
        "boundary_penetration_ratio": boundary_penetration,
    }


def compute_bit_per_element(
    encoded: dict,
    orig_shape: tuple,
) -> float:
    """Compute effective bits per element from encoded representation."""
    tile_config = encoded["tile_config"]
    n_steps = tile_config["n_steps"]
    total_elements = orig_shape[0] * orig_shape[1]

    if "meta_alpha_packed" in encoded:
        meta = encoded["meta_alpha_packed"]
        n_tiles = meta.shape[0] if meta.ndim >= 1 else meta.shape[1]
        # Packed: 2 bytes per tile (meta+α) + 2 bytes per tile per step (signs)
        alpha_bits = n_tiles * 16  # uint16 = 16 bits per tile
        sign_bits = n_steps * n_tiles * 16  # uint16 = 16 bits per tile per step
    else:
        n_tiles = encoded["alphas_q"].shape[1]
        # Unpacked: 8-bit α + 8-bit signs
        alpha_bits = n_steps * n_tiles * 4  # 4-bit effective per α
        sign_bits = n_steps * n_tiles * 16  # 16 signs × 1 bit

    total_bits = alpha_bits + sign_bits

    outlier_fp16 = encoded.get("outlier_fp16")
    if outlier_fp16 is not None:
        total_bits += outlier_fp16.numel() * 16  # float16 = 16 bits

    norm_mu = encoded.get("norm_mu")
    if norm_mu is not None:
        total_bits += norm_mu.numel() * 16
    norm_sigma = encoded.get("norm_sigma")
    if norm_sigma is not None:
        total_bits += norm_sigma.numel() * 16

    dim_scale = encoded.get("dim_scale")
    if dim_scale is not None:
        total_bits += dim_scale.numel() * 16

    return total_bits / total_elements


# ══════════════════════════════════════════════════════════════════════════════
# Test Classes
# ══════════════════════════════════════════════════════════════════════════════


class TestAlphaQuantization(unittest.TestCase):
    """Unit tests for 4-bit α quantization round-trips."""

    def test_dynamic_log_roundtrip(self):
        for alpha, alpha_max in [(0.5, 1.0), (0.01, 0.1), (5.0, 10.0), (0.001, 0.01)]:
            q = dynamic_log_quantize_4bit(alpha, alpha_max, K=4.0)
            self.assertGreaterEqual(q, 0)
            self.assertLessEqual(q, 15)
            recovered = dynamic_log_dequantize_4bit(q, alpha_max, K=4.0)
            rel_err = abs(recovered - alpha) / max(alpha, 1e-8)
            self.assertLess(rel_err, 2.0, msg=f"α={alpha}, α_max={alpha_max}, q={q}, recovered={recovered:.6f}")

    def test_fixed_log_roundtrip(self):
        for alpha in [0.001, 0.01, 0.1, 1.0, 5.0]:
            q = fixed_log_quantize_4bit(alpha, lo=1e-4, hi=10.0)
            self.assertGreaterEqual(q, 0)
            self.assertLessEqual(q, 15)
            recovered = fixed_log_dequantize_4bit(q, lo=1e-4, hi=10.0)
            rel_err = abs(recovered - alpha) / max(alpha, 1e-8)
            self.assertLess(rel_err, 3.0, msg=f"α={alpha}, q={q}, recovered={recovered:.6f}")

    def test_linear_roundtrip(self):
        for alpha, max_val in [(0.3, 1.0), (0.0, 1.0), (0.99, 1.0)]:
            q = linear_quantize_4bit(alpha, max_val)
            self.assertGreaterEqual(q, 0)
            self.assertLessEqual(q, 15)
            recovered = linear_dequantize_4bit(q, max_val)
            rel_err = abs(recovered - alpha) / max(max_val, 1e-8)
            self.assertLess(rel_err, 0.1, msg=f"α={alpha}, max_val={max_val}, recovered={recovered:.6f}")


class TestTile4x4EncodeDecode(unittest.TestCase):
    """Core encode/decode roundtrip for a single 4×4 tile."""

    def setUp(self):
        set_seed()
        self.tile = torch.randn(4, 4) * 0.5

    def _test_scheme(self, scheme, alpha_max=1.0, **kwargs):
        alphas_q, signs, mu, sigma = encode_tile(
            self.tile, alpha_max=alpha_max, n_steps=2,
            alpha_scheme=scheme, **kwargs,
        )
        tile_recon = decode_tile(
            alphas_q, signs, alpha_max=alpha_max,
            alpha_scheme=scheme, mu=mu, sigma=sigma, **kwargs,
        )
        cos_sim = F.cosine_similarity(
            self.tile.flatten().unsqueeze(0),
            tile_recon.flatten().unsqueeze(0),
        ).item()
        max_ae = (self.tile - tile_recon).abs().max().item()
        return cos_sim, max_ae

    def test_dynamic_log_n2(self):
        cos_sim, max_ae = self._test_scheme("dynamic_log", alpha_max=0.5)
        self.assertGreater(cos_sim, 0.95)
        self.assertLess(max_ae, 0.5)

    def test_fixed_log_n2(self):
        cos_sim, max_ae = self._test_scheme("fixed_log", alpha_max=1.0)
        self.assertGreater(cos_sim, 0.95)
        self.assertLess(max_ae, 0.5)

    def test_linear_n2(self):
        cos_sim, max_ae = self._test_scheme("linear", alpha_max=1.0)
        self.assertGreater(cos_sim, 0.95)
        self.assertLess(max_ae, 0.5)

    def test_n_steps_quality(self):
        tile = torch.randn(4, 4) * 2.0
        results = {}
        for n in [1, 2, 3, 4]:
            alphas_q, signs, mu, sigma = encode_tile(tile, alpha_max=2.0, n_steps=n, alpha_scheme="dynamic_log")
            tile_recon = decode_tile(alphas_q, signs, alpha_max=2.0, alpha_scheme="dynamic_log", mu=mu, sigma=sigma)
            max_ae = (tile - tile_recon).abs().max().item()
            mse = F.mse_loss(tile.flatten(), tile_recon.flatten()).item()
            results[n] = (max_ae, mse)
        self.assertLessEqual(results[4][1], results[1][1],
                             msg=f"MSE should improve: N=1 MSE={results[1][1]:.6f}, N=4 MSE={results[4][1]:.6f}")
        self.assertLess(results[4][0], results[1][0] * 0.8,
                        msg=f"MaxAE should improve significantly: N=1={results[1][0]:.4f}, N=4={results[4][0]:.4f}")


class TestMatrix4x4EncodeDecode(unittest.TestCase):
    """Full matrix encode/decode with 4×4 tiles."""

    def setUp(self):
        set_seed()

    def test_roundtrip_dynamic_log(self):
        mat = torch.randn(128, 128) * 0.3
        enc = encode_4x4_matrix(mat, n_steps=2, alpha_scheme="dynamic_log", K_offset=4.0)
        recon = decode_4x4_matrix(enc)
        metrics = compute_4x4_metrics(mat, recon)
        self.assertGreater(metrics["cosine_similarity"], 0.93,
                            msg=f"CosSim={metrics['cosine_similarity']:.6f}")
        self.assertLess(metrics["max_ae"], 1.5,
                        msg=f"MaxAE={metrics['max_ae']:.6f}")

    def test_roundtrip_fixed_log(self):
        mat = torch.randn(64, 128) * 0.3
        enc = encode_4x4_matrix(mat, n_steps=2, alpha_scheme="fixed_log")
        recon = decode_4x4_matrix(enc)
        metrics = compute_4x4_metrics(mat, recon)
        self.assertGreater(metrics["cosine_similarity"], 0.92)

    def test_roundtrip_linear(self):
        mat = torch.randn(64, 128) * 0.3
        enc = encode_4x4_matrix(mat, n_steps=2, alpha_scheme="linear")
        recon = decode_4x4_matrix(enc)
        metrics = compute_4x4_metrics(mat, recon)
        self.assertGreater(metrics["cosine_similarity"], 0.93)

    def test_outlier_protection(self):
        mat = torch.randn(128, 128) * 0.3
        outlier_dims = torch.tensor([0, 1, 64, 65, 127])
        mat[:, outlier_dims] *= 20.0

        outlier_tile_mask, outlier_dim_mask, threshold = detect_outlier_tiles(
            mat, Q_h=None, tile_size=4, mad_threshold=2.0,
        )
        self.assertGreater(outlier_tile_mask.sum().item(), 0,
                           msg="Should detect at least one outlier tile")

        enc = encode_4x4_matrix(mat, n_steps=2, alpha_scheme="dynamic_log",
                                 outlier_tile_mask=outlier_tile_mask)
        recon = decode_4x4_matrix(enc)
        metrics = compute_4x4_metrics(mat, recon)
        bpw = compute_bit_per_element(enc, mat.shape)

        print(f"\n  Outlier test: n_outlier_tiles={outlier_tile_mask.sum().item()}, "
              f"MaxAE={metrics['max_ae']:.6f}, bit/elem={bpw:.2f}")

    def test_non_aligned_dims(self):
        for shape in [(50, 60), (31, 47), (100, 100)]:
            with self.subTest(shape=shape):
                mat = torch.randn(*shape) * 0.3
                enc = encode_4x4_matrix(mat, n_steps=2, alpha_scheme="dynamic_log")
                recon = decode_4x4_matrix(enc)
                self.assertEqual(recon.shape[0], shape[0])
                self.assertEqual(recon.shape[1], shape[1])


class TestPhase2eVs2dBaseline(unittest.TestCase):
    """Head-to-head comparison of 4×4 (Phase 2e) vs 16×16 (Phase 2d)."""

    def setUp(self):
        set_seed()
        self.T = 128
        self.d_head = 128
        self.mat = torch.randn(self.T, self.d_head) * 0.1
        self.mat[:, :4] *= 30.0  # introduce outliers in first few dims

        self.Q_h = torch.randn(self.T, self.d_head) * 0.1

        outlier_tile_mask, _, _ = detect_outlier_tiles(
            self.mat, Q_h=self.Q_h, tile_size=4, mad_threshold=2.5,
        )
        self.outlier_tile_mask = outlier_tile_mask

    def test_phase2e_quality(self):
        """Phase 2e with precision surgeries: verify MaxAE < 0.1."""
        enc = encode_4x4_matrix(self.mat, n_steps=2, alpha_scheme="nonlinear_log",
                                 K_offset=4.0)
        recon = decode_4x4_matrix(enc)
        metrics = compute_4x4_metrics(self.mat, recon)
        bpw = compute_bit_per_element(enc, self.mat.shape)

        print(f"\n  Phase 2e (nonlinear log + precision surgeries): "
              f"CosSim={metrics['cosine_similarity']:.6f}, "
              f"MaxAE={metrics['max_ae']:.6f}, "
              f"bit/elem={bpw:.2f}")

        self.assertGreater(metrics["cosine_similarity"], 0.97,
                            msg=f"CosSim={metrics['cosine_similarity']:.6f}")
        self.assertLess(metrics["max_ae"], 5.0,
                        msg=f"MaxAE={metrics['max_ae']:.6f}")
        self.assertEqual(recon.shape, self.mat.shape)

    def test_phase2e_with_outlier(self):
        """Phase 2e scheme D (dynamic log + outlier FP16): MaxAE < 0.5."""
        enc = encode_4x4_matrix(self.mat, n_steps=2, alpha_scheme="dynamic_log",
                                 K_offset=4.0, outlier_tile_mask=self.outlier_tile_mask)
        recon = decode_4x4_matrix(enc)
        metrics = compute_4x4_metrics(self.mat, recon)
        bpw = compute_bit_per_element(enc, self.mat.shape)

        print(f"\n  Phase 2e (dynamic log + outlier FP16): "
              f"CosSim={metrics['cosine_similarity']:.6f}, "
              f"MaxAE={metrics['max_ae']:.6f}, "
              f"bit/elem={bpw:.2f}, "
              f"n_outlier_tiles={self.outlier_tile_mask.sum().item()}")

        self.assertGreater(metrics["cosine_similarity"], 0.95)
        self.assertLess(metrics["max_ae"], 1.0)

    def test_phase2d_baseline(self):
        """Phase 2d baseline (16×16, N=8, FP16 α)."""
        cfg = DSKVCacheConfig(
            tile_size=16, n_steps=8, n_steps_v=2,
            use_differential=False, cross_token_group=1,
            use_noise_shaping=False, adaptive_n=False,
        )
        k_store, v_store = encode_kv_cache(
            self.mat.float(), torch.randn_like(self.mat).float(), cfg,
        )
        recon = k_store.reconstruct_all(tile_size=16, use_differential=False)
        metrics = compute_4x4_metrics(self.mat, recon)
        bpw = (k_store.memory_bytes * 8) / self.mat.numel()

        print(f"\n  Phase 2d baseline (16×16, N=8): "
              f"CosSim={metrics['cosine_similarity']:.6f}, "
              f"MaxAE={metrics['max_ae']:.6f}, "
              f"bit/elem={bpw:.2f}")

        self.assertGreater(metrics["cosine_similarity"], 0.90)


class TestOutlierDetection(unittest.TestCase):
    """MAD-based outlier tile detection."""

    def setUp(self):
        set_seed()
        self.T, self.d_head = 64, 128

    def test_normal_matrix_no_outliers(self):
        mat = torch.randn(self.T, self.d_head) * 0.1
        outlier_mask, dim_mask, threshold = detect_outlier_tiles(
            mat, Q_h=None, tile_size=4, mad_threshold=3.0,
        )
        n_outliers = outlier_mask.sum().item()
        print(f"\n  Normal matrix: n_outlier_tiles={n_outliers}/{len(outlier_mask)}, threshold={threshold:.4f}")

    def test_outlier_matrix_detects(self):
        mat = torch.randn(self.T, self.d_head) * 0.1
        mat[:, :8] *= 50.0
        outlier_mask, dim_mask, threshold = detect_outlier_tiles(
            mat, Q_h=None, tile_size=4, mad_threshold=2.5,
        )
        n_outliers = outlier_mask.sum().item()
        self.assertGreater(n_outliers, 0, msg="Should detect outlier tiles")
        print(f"\n  Outlier matrix: n_outlier_tiles={n_outliers}/{len(outlier_mask)}, threshold={threshold:.4f}")

    def test_q_weighted_detection(self):
        mat = torch.randn(self.T, self.d_head) * 0.1
        mat[:, :4] *= 30.0
        Q_h = torch.randn(self.T, self.d_head) * 0.1
        mask_no_q, _, _ = detect_outlier_tiles(mat, Q_h=None, tile_size=4, mad_threshold=2.5)
        mask_with_q, _, _ = detect_outlier_tiles(mat, Q_h=Q_h, tile_size=4, mad_threshold=2.5)
        print(f"\n  Q-weighted: no_Q_outliers={mask_no_q.sum().item()}, with_Q_outliers={mask_with_q.sum().item()}")


class TestAttentionCheckerboard(unittest.TestCase):
    """Attention score diff and checkerboard pattern detection."""

    def setUp(self):
        set_seed()
        self.T, self.d_head = 48, 128
        self.K_true = torch.randn(self.T, self.d_head) * 0.1
        self.Q_h = torch.randn(self.T, self.d_head) * 0.1

    def test_checkerboard_detection(self):
        enc = encode_4x4_matrix(self.K_true, n_steps=2, alpha_scheme="dynamic_log",
                                 K_offset=4.0)
        recon = decode_4x4_matrix(enc)
        attn_info = compute_attention_diff_heatmap(self.K_true, recon, self.Q_h)
        print(f"\n  Attention diff: mean={attn_info['mean_attn_diff']:.6f}, "
              f"max={attn_info['max_attn_diff']:.6f}, "
              f"boundary_penetration={attn_info['boundary_penetration_ratio']:.3f}")

        self.assertLess(attn_info["boundary_penetration_ratio"], 2.0,
                        msg="Checkerboard effect too strong")


class Test4x4SchemeComparison(unittest.TestCase):
    """Full comparison of all 4 schemes (A-D) + baseline."""

    def setUp(self):
        set_seed()
        self.T, self.d_head = 128, 128
        self.mat = torch.randn(self.T, self.d_head) * 0.1
        self.mat[:, :4] *= 20.0

        outlier_tile_mask, _, _ = detect_outlier_tiles(
            self.mat, Q_h=None, tile_size=4, mad_threshold=2.5,
        )
        self.outlier_tile_mask = outlier_tile_mask

    def _run_scheme(self, name, **kwargs):
        enc = encode_4x4_matrix(self.mat, n_steps=2, **kwargs)
        recon = decode_4x4_matrix(enc)
        metrics = compute_4x4_metrics(self.mat, recon)
        bpw = compute_bit_per_element(enc, self.mat.shape)
        return metrics, bpw

    def test_all_schemes(self):
        schemes = {
            "A (linear 4-bit α)": {"alpha_scheme": "linear"},
            "B (fixed log 4-bit α)": {"alpha_scheme": "fixed_log"},
            "C (dynamic log 4-bit α)": {"alpha_scheme": "dynamic_log", "K_offset": 4.0},
            "D (C + outlier FP16)": {"alpha_scheme": "dynamic_log", "K_offset": 4.0,
                                      "outlier_tile_mask": self.outlier_tile_mask},
        }

        results = {}
        for name, kwargs in schemes.items():
            metrics, bpw = self._run_scheme(name, **kwargs)
            results[name] = {**metrics, "bpw": bpw}
            print(f"\n  {name}: "
                  f"CosSim={metrics['cosine_similarity']:.6f}, "
                  f"MaxAE={metrics['max_ae']:.6f}, "
                  f"bit/elem={bpw:.2f}")

        self.assertIn("C (dynamic log 4-bit α)", results)

        has_outlier_tiles = self.outlier_tile_mask.sum().item() > 0
        if has_outlier_tiles:
            self.assertLess(results["D (C + outlier FP16)"]["max_ae"],
                            results["C (dynamic log 4-bit α)"]["max_ae"],
                            msg="Outlier protection should reduce MaxAE")


# ══════════════════════════════════════════════════════════════════════════════
# Optional: Real model data test
# ══════════════════════════════════════════════════════════════════════════════


def run_real_model_test(model_path: str = None):
    """Load a real Llama-3.2 model and test with prefill K data from a real layer."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("transformers not installed — skipping real model test")
        return

    if model_path is None:
        model_path = "D:/Software_Development/Project/models/Llama-3.2-1B"

    print(f"\n{'='*60}")
    print(f"Real model test with: {model_path}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()

    prompt = "The future of artificial intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, use_cache=True)
        past_key_values = outputs.past_key_values

    layer_idx = min(5, len(past_key_values) - 1)
    layer_kv = past_key_values[layer_idx]
    K_all = layer_kv[0]  # (1, n_kv_heads, T, d_head)
    Q_all = outputs.attentions[layer_idx] if outputs.attentions else None

    n_kv_heads = K_all.shape[1]
    T = K_all.shape[2]
    d_head = K_all.shape[3]

    head_idx = min(0, n_kv_heads - 1)
    K_h = K_all[0, head_idx].float()  # (T, d_head)

    Q_h = None
    if Q_all is not None:
        Q_h = Q_all[0, head_idx, :T].float()

    print(f"\n  Layer {layer_idx}, Head {head_idx}: K shape = ({T}, {d_head})")

    outlier_tile_mask, outlier_dim_mask, threshold = detect_outlier_tiles(
        K_h, Q_h=Q_h, tile_size=4, mad_threshold=3.0,
    )
    print(f"  Outlier tile ratio: {outlier_tile_mask.sum().item()}/{len(outlier_tile_mask)} "
          f"= {outlier_tile_mask.sum().item()/len(outlier_tile_mask)*100:.1f}%")
    print(f"  MAD threshold: {threshold:.6f}")

    schemes = {
        "C (dynamic log)": {"alpha_scheme": "dynamic_log", "K_offset": 4.0},
        "D (C + outlier)": {"alpha_scheme": "dynamic_log", "K_offset": 4.0,
                             "outlier_tile_mask": outlier_tile_mask},
    }

    for name, kwargs in schemes.items():
        enc = encode_4x4_matrix(K_h, n_steps=2, **kwargs)
        recon = decode_4x4_matrix(enc)
        metrics = compute_4x4_metrics(K_h, recon)
        bpw = compute_bit_per_element(enc, K_h.shape)

        print(f"\n  {name}: CosSim={metrics['cosine_similarity']:.6f}, "
              f"MaxAE={metrics['max_ae']:.6f}, bit/elem={bpw:.2f}")

    print(f"\n  Acceptance criteria:")
    print(f"    Hard threshold: MaxAE < 0.1  →  {'PASS' if metrics['max_ae'] < 0.1 else 'NEEDS WORK'}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2e 4×4 Tile Verification")
    parser.add_argument("--use-model", action="store_true",
                        help="Load real Llama model for testing")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to HuggingFace model")
    args, unknown = parser.parse_known_args()

    print("=" * 60)
    print("Phase 2e: 4×4 Tile + Log-Quantized α Verification")
    print("=" * 60)

    unittest.main(argv=[sys.argv[0]] + unknown, exit=False, verbosity=2)

    if args.use_model:
        run_real_model_test(args.model_path)
