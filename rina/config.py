# DS-KVCache Configuration (RINA_Core/rina/config.py)
# =============================================================================
# Centralised configuration for the Σ-Δ modulated 1-bit KV cache.
#
# See Also
# --------
# RINA_Whitepaper §8 (Parameter Reference) & §10.3 (Ablation guardrails)
# RINA_Whitepaper §8.4 (Parameter Reference) & §10 (Prototype Results)

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)


@dataclass
class DSKVCacheConfig:
    """Configuration for DS-KVCache encoding and decoding.

    Parameters
    ----------
    tile_size : int
        Base tile size for Tile-Granular Σ-Δ encoding.  Must be a power of 2.
        The effective tile dimension may be larger when *cross_token_group* > 1.
        Default 16.
    n_steps : int
        Number of 1-bit bases (Σ-Δ iterations) per tile for both K and V.
        Controls compression ratio: higher → better quality, lower compression.
        Default 3.
    n_steps_k : int | None
        Separate step count for Key path.  None → uses *n_steps*.
    n_steps_v : int | None
        Separate step count for Value path.  None → uses *n_steps*.
    beta : float
        Σ-Δ integrator feedback coefficient.  Higher → faster error correction
        but may overshoot.  Recommended 0.05 – 0.15.
    proj_beta : float
        Cross-token projection feedback coefficient.  Higher → stronger
        inter-token error cancellation.  Recommended 0.3 – 0.6.
    adaptive_eta : float
        Adaptive step-size scaling for sigma-delta modulation.
        When > 0, the effective beta is scaled by local gradient energy,
        reducing step size in high-curvature regions.
        Recommended 0.0 (off) or 0.05 – 0.15 for adaptive.
    order2_gamma : float
        Second-order integrator gain.  When > 0, a second integrator is
        added to the Σ-Δ loop, forming a Type-II tracking loop that
        eliminates steady-state error for linearly-varying signals.
        Recommended 0.0 (off) or 0.05 – 0.20.
        See *RINA_Whitepaper §8.3 (Second-Order Σ-Δ Modulation)*.
    order2_c1 : float
        Second-order lead compensator numerator.
        See *RINA_Whitepaper §8.3 (Second-Order Σ-Δ Modulation)*.
    order2_c2 : float
        Second-order lead compensator denominator.
        See *RINA_Whitepaper §8.3 (Second-Order Σ-Δ Modulation)*.
    zero_mean_integrator2 : bool
        If True, initialise the second integrator state such that its mean
        is zero over each tile.  Helps when the second integrator accumulates
        a large DC offset that would saturate the 1-bit quantiser.
        See *RINA_Whitepaper §8.3 (Second-Order Σ-Δ Modulation)*.
    use_differential : bool
        Enable two-stage residual encoding for finer reconstruction.
    diff_strategy : str
        Differential cancellation strategy: 'residual' or 'cancellation'.
    diff_residual_n_steps : int
        Number of 1-bit bases for residual stage.
    diff_residual_gamma : float
        Blending coefficient for residual stage (V path).
    diff_residual_gamma_k : float
        Blending coefficient for residual stage (K path).
    adaptive_n : bool
        Assign extra 1-bit bases to high-energy tiles.
        See *RINA_Whitepaper §8.1.3 and §10.3*.
    n_upper_bound : int
        Maximum number of bases per tile under adaptive N.
    energy_threshold_factor : float
        Energy threshold ratio for adaptive N.
    cross_head_error_share : bool
        Enable inter-head Σ-Δ error propagation for GQA models.
    cross_head_error_bias : float
        Coupling strength for cross-head error sharing.
    cross_head_residual : bool
        Enable Route 2 — Cross-Head Reconstruction Residual Injection.
        When True, each KV head encodes K/V with residual bias from the
        previous head's reconstruction error (ε = X_orig - decode(encode(X))).
        This distributes quantization error across KV heads in GQA groups.
        Default False.
    cross_head_residual_gamma : float
        EMA decay factor for reconstruction residual accumulation
        across KV heads (Route 2).  Range 0.05 – 0.50.
        Default 0.25.
    cross_token_group : int
        Number of tokens to group into a single encoding tile.
        1 = no grouping (standard per-token tile).
        2+ = group *cross_token_group* consecutive tokens along the token
        dimension before encoding, effectively treating them as a wider
        tile.  This increases the effective tile dimension to
        *tile_size* × *cross_token_group*, which improves energy compaction
        at the cost of coarser temporal granularity.
        Must be a power of 2.
        NOTE: When cross_token_group > 1, V-orthogonal rotation is
        automatically disabled because the rotation matrix shape (d_head, d_head)
        does not match the cross-token expanded tile dimension.
    dynamic_tile_size : bool
        Enable adaptive tile sizing for incremental (token-by-token) encoding.
    min_tile_size : int
        Minimum tile dimension when dynamic_tile_size is enabled.
    incremental_buffer_size : int
        Number of tokens to buffer before encoding during incremental decode.
    base_dtype : str
        Base storage dtype for raw buffers and input.
    v_orthogonal_transform : bool
        Apply an orthogonal rotation to V before encoding, constructed from
        a QR decomposition of the first-iteration FWHT coefficients.
        This decorrelates the V dimension and improves Σ-Δ encoding efficiency.
        Disabled automatically when *cross_token_group* > 1.
    adaptive_masking : bool
        Enable adaptive bit-rate masking (Roadmap 1).
        Detects outlier tiles and boosts their encoding budget.
    mask_outlier_threshold : float
        Max-abs / std ratio threshold for outlier detection.
    mask_n_steps_boost : int
        Number of extra 1-bit iterations for outlier tiles.
    mask_proj_beta_boost : float
        Multiplicative boost for proj_beta on outlier tiles.
    layer_step_map : dict[int, tuple[int, int]] | None
        Per-layer step override.  Keys are layer indices (0-based),
        values are (n_steps_k, n_steps_v).  Layers not in the map use
        the global *n_steps* / *n_steps_k* / *n_steps_v* values.
    verbose : bool
        Enable diagnostic logging per encoding operation.
    """

    # ── Core parameters ─────────────────────────────────────────────────
    tile_size: int = 16
    n_steps: int = 3
    n_steps_k: Optional[int] = None
    n_steps_v: Optional[int] = None

    beta: float = 0.10
    proj_beta: float = 0.5
    adaptive_eta: float = 0.0
    order2_gamma: float = 0.20
    order2_c1: float = 0.85
    order2_c2: float = 0.15
    zero_mean_integrator2: bool = False

    # ── Differential (two-stage) ────────────────────────────────────────
    use_differential: bool = True
    diff_strategy: str = "residual"
    diff_residual_n_steps: int = 1
    diff_residual_gamma: float = 0.25
    diff_residual_gamma_k: float = 0.25

    # ── Adaptive N scheduling ──────────────────────────────────────────
    adaptive_n: bool = False
    n_upper_bound: int = 10
    energy_threshold_factor: float = 0.5

    # ── Cross-head error sharing ───────────────────────────────────────
    cross_head_error_share: bool = True
    cross_head_error_bias: float = 0.15

    # ── Route 2: Cross-head residual bias injection ────────────────────
    cross_head_residual: bool = False
    """Enable Route 2 — Cross-Head Reconstruction Residual injection (§8.2)."""
    cross_head_residual_gamma: float = 0.25
    """EMA decay factor for reconstruction residual accumulation across KV heads
    in a GQA group.  Higher = more aggressive bias injection."""

    # ── Cross-token grouping (RINA Whitepaper §8.1.4) ─────────────────
    cross_token_group: int = 1
    """Number of tokens to group into a single encoding tile.
    1 = no grouping.  Must be a power of 2.
    When > 1, v_orthogonal_transform is auto-disabled."""

    # ── Dynamic tile size ─────────────────────────────────────────────
    dynamic_tile_size: bool = True
    min_tile_size: int = 4

    # ── Incremental buffer ─────────────────────────────────────────────
    incremental_buffer_size: int = 128

    # ── IO / precision ─────────────────────────────────────────────────
    base_dtype: str = "fp16"
    verbose: bool = False

    # ── Transform roadmap (§9.2) ───────────────────────────────────────
    v_orthogonal_transform: bool = True
    """Apply QR-based orthogonal rotation to V before encoding.
    Auto-disabled when cross_token_group > 1 (shape mismatch)."""

    # ── Adaptive masking (Roadmap 1) ─────────────────────────────────────
    adaptive_masking: bool = False
    mask_outlier_threshold: float = 3.0
    mask_n_steps_boost: int = 1
    mask_proj_beta_boost: float = 0.5

    use_mask_gating: bool = True
    """If True, zero out padding regions in the Σ-Δ state at each iteration
    (single-step accumulation).  Prevents encoding bits from being wasted on
    zero-padding and improves reconstruction quality for partially-filled
    tiles (§10.3.1)."""

    # ── Noise shaping (cross-token projection) ──────────────────────────
    use_noise_shaping: bool = True
    proj_rank: int = 8
    """Rank for cross-token noise shaping projection. 0 disables."""

    # ── Weighted reconstruction (§8.1.7) ─────────────────────────────────
    use_recon_weights: bool = False
    """If True, apply per-step importance weighting during reconstruction."""
    recon_weight_temperature: float = 0.5
    """Temperature for reconstruction weight softmax."""

    # ── Periodic FP16 bypass (anchor token refresh) ────────────────────
    refresh_interval: int = 0
    """If > 0, every ``refresh_interval``-th decode token's KV is stored at
    full FP16 precision (bypasses Σ-Δ encoding).  This periodically resets
    cumulative quantization error to zero.  Default 0 = disabled.
    Recommended 8–16 for long generation."""

    # ── Per-layer step mapping ──────────────────────────────────────────
    layer_step_map: Optional[Dict[int, Tuple[int, int]]] = None

    # ── Protected layers (§8.1.8) ────────────────────────────────────────
    protected_layers: List[int] = field(default_factory=lambda: [0, -1])
    """Layer indices (0-based) that bypass 1-bit encoding.  Default: [0, -1]
    (first and last layers).  -1 is interpreted as self._num_layers-1 at runtime."""

    # ── Prefill FP16 protection (Step 0 Bypass enhancement) ───────────────
    prefill_protected: bool = False
    """If True, all prefill K/V tokens are stored as raw FP16 (bypass Σ-Δ encoding).
    This ensures the decode phase starts from zero quantization error.
    Default False."""

    # ── Adaptive bypass (Phase 1 Quality — DEPRECATED, use adaptive_residual) ──
    bypass_adaptive: bool = False
    """DEPRECATED.  If True, bypass decision is based on per-tile L∞ reconstruction error."""
    bypass_threshold: float = 0.5
    """DEPRECATED.  L∞ norm threshold for adaptive bypass."""

    # ── Key Position Protection (Attention Phase 1) ──
    decode_protect_steps: int = 3
    """Number of initial decode steps to store at FP16 precision via bypass_map_fp16.
    Protects attention initialization from 1-bit quantization error.
    Default 3.  Higher → more stable but slightly less CR."""

    decode_protect_layers: str = "last_4"
    """Which layers to protect in key-position protection.
    Options:
      \"all\"        — all layers (Stage 1 behavior)
      \"last_4\"     — last 4 layers (recommended for Stage 2)
      \"first_last\" — first and last layer only
      \"none\"       — disable per-layer protection
    """

    # ── Confidence-Masked Attention (Phase 3) ──
    confidence_mask: bool = False
    """If True, apply per-token confidence penalty to attention scores."""

    confidence_beta: float = 0.3
    """Penalty scaling factor.  0.3 = moderate, 0.7 = aggressive."""

    # ── Temporal Attention Smoothing (Phase 4) ──
    attn_smoothing_alpha: float = 1.0
    """Temporal attention smoothing factor.  1.0 = no smoothing (use current only),
    0.9 = 90% current + 10% previous step's attention distribution.
    Lower → smoother but may lag behind.  Recommended 0.85-0.95 when enabled."""

    # ── Adaptive 1-bit Residual Correction ──
    adaptive_residual: bool = False
    """If True, encode per-tile reconstruction error (delta = tile - primary_recon)
    using 1-bit Σ-Δ encoding and store as bases_residual / alphas_residual.
    This keeps the entire pipeline at 1-bit density — each correction tile adds
    only 4 packed-bits + 1 FP16 alpha ≈ 0.25 bits/element."""

    adaptive_residual_threshold: float = 0.2
    """L∞ threshold for adaptive residual.  When any token's reconstruction
    error in a tile exceeds this, the full tile's delta is 1-bit encoded.
    Lower = more residuals (better quality, higher CR cost).
    Recommended range: 0.1 – 0.5."""

    adaptive_residual_n_steps: int = 1
    """Number of Σ-Δ steps for adaptive residual encoding.
    1 step = 4 packed-bits ≈ 0.25 bits/element.  Higher = more precise but
    less memory-efficient."""

    # ── Pyramid prefill (Phase 3) ──
    prefill_system_protect_len: int = 128
    """Number of initial prefill tokens stored at full precision via bypass_map.
    Typical system prompt length. Default 128."""
    prefill_tail_protect_len: int = 32
    """Number of final prefill tokens stored at full precision via bypass_map.
    Last tokens are most critical for first-decode-step attention.
    Default 32."""

    # ── Dual store (Phase 4) ──
    prefill_n_steps: Optional[int] = None
    """If set, prefill uses this many bases per tile (e.g., 8 for high quality).
    Decode still uses the global n_steps (e.g., 3).
    None → no dual store, uses global n_steps for all tokens."""

    # ── Beta decay for decode (§8.1.11) ─────────────────────────────────
    beta_decay_start: Optional[float] = None
    """Initial beta for decode steps.  If None, uses self.beta."""
    beta_decay_end: float = 0.02
    """Final beta after decay completes."""
    beta_decay_tokens: int = 256
    """Number of decode steps to decay over."""

    def __post_init__(self):
        """Validate constraints and auto-adjust inconsistent settings.

        Enforces the parameter interaction rules documented in
        *RINA_Whitepaper §8.4.2 (Parameter Interaction Matrix)*
        and *§10.3 (Ablation guardrails)*.
        """
        # ── §10.3 guard: n_upper_bound must be ≥ n_steps ────────────
        if self.adaptive_n and self.n_upper_bound < self.n_steps:
            _logger.warning(
                f"n_upper_bound ({self.n_upper_bound}) < n_steps ({self.n_steps}); "
                f"auto-raising to n_steps + 2.  Set n_upper_bound explicitly to avoid "
                f"silent storage inflation."
            )
            self.n_upper_bound = self.n_steps + 2

        # ── §10.3 guard: n_upper_bound must cover n_steps_v too ─────
        if self.adaptive_n and self.n_steps_v is not None and self.n_upper_bound < self.n_steps_v:
            _logger.warning(
                f"n_upper_bound ({self.n_upper_bound}) < n_steps_v ({self.n_steps_v}); "
                f"auto-raising to n_steps_v + 2."
            )
            self.n_upper_bound = self.n_steps_v + 2

        # ── §8.4.2 guard: cross_token_group must be power-of-2 or 1 ──
        if self.cross_token_group > 1:
            log2 = math.log2(self.cross_token_group)
            if log2 != int(log2):
                _logger.warning(
                    f"cross_token_group ({self.cross_token_group}) is not a power of 2; "
                    f"rounding up to {2**math.ceil(log2)} for reshape alignment."
                )
                self.cross_token_group = 2 ** math.ceil(log2)

            # ── §8.4.2 guard: disable V rotation when cross-token grouping ──
            if self.v_orthogonal_transform:
                _logger.warning(
                    f"cross_token_group={self.cross_token_group} > 1 is incompatible with "
                    f"v_orthogonal_transform=True (rotation matrix shape d_head×d_head does not "
                    f"match the cross-token expanded tile dimension). "
                    f"Auto-disabling v_orthogonal_transform."
                )
                self.v_orthogonal_transform = False

        # ── §8.4.2 guard: beta reduced when order2 enabled ───────────
        if self.order2_gamma > 0 and self.beta > 0.20:
            _logger.warning(
                f"beta ({self.beta}) may cause overshoot with order2_gamma ({self.order2_gamma}). "
                f"Consider reducing beta to ≤0.15 for stability (§8.4.2 Parameter Interaction Matrix)."
            )

    # ── Convenience accessors ──────────────────────────────────────────

    def get_n_steps_k(self) -> int:
        """Effective n_steps for Key path."""
        return self.n_steps_k if self.n_steps_k is not None else self.n_steps

    def get_n_steps_v(self) -> int:
        """Effective n_steps for Value path."""
        return self.n_steps_v if self.n_steps_v is not None else self.n_steps

    def get_diff_residual_gamma_k(self) -> float:
        """Effective diff_residual_gamma for Key path."""
        return self.diff_residual_gamma_k if self.diff_residual_gamma_k is not None else self.diff_residual_gamma

    def get_layer_steps_k(self, layer_idx: int) -> int:
        """Get n_steps_k for a specific layer, falling back to global default."""
        if self.layer_step_map and layer_idx in self.layer_step_map:
            k_steps, _ = self.layer_step_map[layer_idx]
            return k_steps
        return self.get_n_steps_k()

    def get_layer_steps_v(self, layer_idx: int) -> int:
        """Get n_steps_v for a specific layer, falling back to global default."""
        if self.layer_step_map and layer_idx in self.layer_step_map:
            _, v_steps = self.layer_step_map[layer_idx]
            return v_steps
        return self.get_n_steps_v()

    def get_layer_config(self, layer_idx: int, num_layers: int) -> DSKVCacheConfig:
        """Return a per-layer configuration snapshot for the given layer.

        Returns *self* unchanged if no per-layer overrides are active;
        otherwise returns a shallow copy with the layer-specific
        ``n_steps_k`` and ``n_steps_v`` applied.

        Parameters
        ----------
        layer_idx : int
            0-based layer index.
        num_layers : int
            Total number of layers (used to resolve ``-1`` in
            *protected_layers*).
        """
        if self.layer_step_map and layer_idx in self.layer_step_map:
            k, v = self.layer_step_map[layer_idx]
            import copy
            cfg = copy.copy(self)
            cfg.n_steps_k = k
            cfg.n_steps_v = v
            return cfg
        return self

    def is_layer_protected(self, layer_idx: int, num_layers: int) -> bool:
        """Check if a layer should bypass 1-bit encoding."""
        for idx in self.protected_layers:
            if idx == -1:
                idx = num_layers - 1
            if idx == layer_idx:
                return True
        return False

    def get_decode_protect_layers(self, num_layers: int) -> set:
        """Resolve decode_protect_layers string config to a set of layer indices."""
        if self.decode_protect_layers == "all":
            return set(range(num_layers))
        elif self.decode_protect_layers == "last_4":
            return set(range(max(0, num_layers - 4), num_layers))
        elif self.decode_protect_layers == "first_last":
            return {0, num_layers - 1}
        else:
            return set()

    def get_beta_for_decode_step(self, step_idx: int) -> float:
        """Get the Σ-Δ beta for a decode step, with optional decay.

        Parameters
        ----------
        step_idx : int
            0-based decode step index.
        """
        if self.beta_decay_start is None:
            return self.beta
        if step_idx >= self.beta_decay_tokens:
            return self.beta_decay_end
        # Linear decay
        alpha = step_idx / max(self.beta_decay_tokens, 1)
        return self.beta_decay_start + alpha * (self.beta_decay_end - self.beta_decay_start)

    def to_dict(self) -> dict:
        """Serialize to a plain dict (JSON-compatible)."""
        import copy
        d = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if isinstance(value, dict):
                # Convert int keys to str for JSON
                d[field_name] = {str(k): v for k, v in value.items()}
            else:
                d[field_name] = copy.deepcopy(value)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "DSKVCacheConfig":
        """Deserialize from dict."""
        import copy
        d_copy = copy.deepcopy(d)
        # Restore int keys for layer_step_map
        if "layer_step_map" in d_copy and isinstance(d_copy["layer_step_map"], dict):
            d_copy["layer_step_map"] = {int(k): tuple(v) for k, v in d_copy["layer_step_map"].items()}
        return cls(**{k: v for k, v in d_copy.items() if k in cls.__dataclass_fields__})
