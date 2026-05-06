"""
§A Unified DS-KVCache Configuration
====================================

Central configuration for the full DS-KVCache pipeline.
Keys on §§4-8 of the whitepaper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DSKVCacheConfig:
    # ── Core encoding (Whitepaper §4) ──────────────────────────────────
    n_steps: int = 5
    """Number of 1-bit bases (oversampling ratio N).  5 is the sweet spot.
    This is the DEFAULT; it may be overridden per-path by n_steps_k/n_steps_v."""

    n_steps_k: Optional[int] = None
    """Number of 1-bit bases for Key path.  If None, falls back to n_steps."""

    n_steps_v: Optional[int] = 8
    """Number of 1-bit bases for Value path.  If None, falls back to n_steps.
    Google findings show V needs more bits → set n_steps_v > n_steps_k.
    Default 8 for V: ~1.6× more bases than K, needed because V residuals
    accumulate through the attention softmax-weighted sum (§8.1.6)."""

    tile_size: int = 16
    """Tile dimension for block-wise encoding.  Must match GPU Tensor Core
    alignment (16 recommended)."""

    beta: float = 0.15
    """First-order Σ-Δ momentum coefficient.  0 → no momentum, 0.10–0.20
    provides mild noise shaping that reduces per-step error accumulation.
    C3 optimal — 0.15 provides moderate shaping without overshoot."""

    # ── Noise shaping (Whitepaper §8.1) ────────────────────────────────
    use_noise_shaping: bool = True
    """Enable SVD nullspace noise shaping."""

    proj_rank: int = 8
    """Number of principal components retained in signal subspace.  Higher =
    tighter nullspace but more compute for SVD calibration.  4-16 is
    sufficient for d_head=64-128."""

    proj_beta: float = 0.3
    """Noise-shaping strength ∈ [0, 1].  0.3 is a conservative default;
    0.5-0.8 for aggressive shaping."""

    adaptive_eta: bool = True
    """Ramp proj_beta from 0→peak linearly across encoding steps (§8.1.1)."""

    # ── Second-order Σ-Δ (§8.1.2) ─────────────────────────────────────
    order2_gamma: float = 0.0
    """Second integrator coupling strength.  0 = first-order only.
    C3 optimal: disable second-order to avoid integrator drift."""

    order2_c1: float = 1.0
    """First integrator gain coefficient."""

    order2_c2: float = 0.5
    """Second integrator gain coefficient."""

    # ── Cross-token joint encoding (§8.1.5) ─────────────────────────────
    cross_token_group: int = 4
    """Number of consecutive tokens to group into a single matrix row before
    tile encoding.  G=4 means each 16×16 tile spans 64 tokens of elements,
    distributing quantization noise across 4× more tokens.  The noise that
    would otherwise accumulate on token N gets spread across tokens [N-4, N+4].
    1 = disabled (each tile covers consecutive 16 tokens).
    Recommended: 4 for long sequences, 2 for <512 tokens."""

    # ── V orthogonal transform (§8.1.4 — Google-style "energy dispersion") ──
    v_orthogonal_transform: bool = True
    """Apply an orthogonal rotation to V before encoding, constructed from
    Q's SVD right-singular vectors.  This disperses V's outlier-heavy
    distribution before 1-bit quantisation, analogous to Google's rotary
    transform for V-path compression."""

    # ── FWHT Walsh-Hadamard transform (§8.1.11) ────────────────────────
    use_fwht: bool = False
    """Apply Fast Walsh-Hadamard Transform to each tile before residual pursuit.
    Rotates circuit-space tile vectors into Walsh basis, spreading outlier energy
    evenly across frequencies so the Σ-Δ quantizer sees a flat spectrum.
    Zero-cost on GPU (only addition/subtraction, no multiplications).
    Phase 3 experiment shows FWHT degrades match_rate (0.1477→0.1023): Σ-Δ
    depends on structured energy distribution that FWHT destroys.
    Set False — revert to differential residual baseline which achieves 0.193+."""

    zero_mean_integrator2: bool = False
    """Remove DC component from the second-order Σ-Δ integrator after each step.
    Prevents integrator DC drift from saturating downstream stages, analogous to
    AC coupling in analog Σ-Δ modulators.  Has no effect when order2_gamma=0.
    Phase 3 experiment shows no benefit when combined with FWHT; set False
    to revert to differential residual baseline configuration."""

    # ── Per-layer adaptive step allocation (§8.1.6) ───────────────────
    layer_step_map: Optional[dict] = field(default_factory=lambda: _default_layer_step_map())
    """Per-layer override of (n_steps_k, n_steps_v).  Keys are 0-indexed layer
    indices; values are (k_steps, v_steps) tuples.  Layers not in the map
    fall back to global n_steps_k / n_steps_v.
    
    Default map (for 16-layer LLaMA-style models):
      0-4  (shallow):  k=3, v=4  — basic semantic detection, low energy
      5-9  (middle):   k=4, v=5  — current default
      10-15 (deep):    k=5, v=6  — fine-grained positional info
    """

    # ── Protected layers (§8.1.8) ──────────────────────────────────────
    protected_layers: list[int] = field(default_factory=list)
    """Layer indices whose K/V are stored at FP16 with zero encoding loss.
    First and last layers are primary candidates (critical at entry/exit of
    the transformer stack).  Empty list = no protection (all layers compressed)."""

    # ── Dynamic Beta decay (§8.1.11) ────────────────────────────────────
    beta_decay_start: float = 0.30
    """Initial β (Σ-Δ momentum) at token 1 of decode.
    Higher initial momentum pushes early quantization error harder into
    high frequencies where it is less likely to accumulate.  0.25-0.35
    recommended for 4-8 n_steps; keep ≤0.15 when order2_gamma > 0."""

    beta_decay_end: float = 0.05
    """Terminal β after decay window.  Low residual momentum prevents
    oscillation in late decode when Σ-Δ integrators are saturated."""

    beta_decay_tokens: int = 0
    """Number of decode tokens over which to linearly decay beta.
    0 = decay disabled (use fixed beta).  Typical: 10-20 for short prompts,
    20-50 for 4K+ generation."""

    # ── Weighted reconstruction (§8.1.7) ────────────────────────────────
    use_recon_weights: bool = True
    """Enable energy-based per-step reconstruction weights.
    When True, weights w_i = softmax(mean(|alpha_i|) / temperature) are
    computed from the encoded alphas, normalised so max w_i = 1.0.
    Higher-energy steps contribute more to the final reconstruction."""

    recon_weight_temperature: float = 0.5
    """Softmax temperature for reconstruction weight computation.
    0.5 = moderate sharpness, 1.0 = near-uniform, 0.1 = nearly argmax."""

    # ── Two-stage differential cancellation (§7) ────────────────────────
    use_differential: bool = True
    """Enable two-stage residual encoding for finer reconstruction."""

    diff_strategy: str = "residual"
    """Differential cancellation strategy: 'residual' or 'cancellation'."""

    diff_residual_n_steps: int = 1
    """Number of 1-bit bases for residual stage.
    C3 optimal: single step avoids over-correction."""

    diff_residual_gamma: float = 0.25
    """Blending coefficient for residual stage (V path).
    C3 optimal: 0.25 avoids over-correction that amplifies quantization noise."""

    diff_residual_gamma_k: float = 0.25
    """Blending coefficient for residual stage (K path).
    C3 optimal: matched to V for symmetric correction."""

    # ── Adaptive N scheduling (Whitepaper §8.1.3) ──────────────────────
    adaptive_n: bool = False
    """Assign extra 1-bit bases to high-energy tiles.  OFF by default
    because the extra bases must be budgeted from n_upper_bound, which
    if too large (> n_steps) silently inflates storage 2-5×.
    Enable explicitly when n_upper_bound is tuned to the workload."""

    n_upper_bound: int = 10
    """Maximum number of bases per tile under adaptive N.  Will be raised
    automatically if *n_steps* exceeds this value."""

    energy_threshold_factor: float = 0.5
    """Energy threshold ratio for adaptive N.  Tiles with energy above
    this fraction of max energy get extra bases."""

    # ── Cross-head error sharing (§8.1.9 — GQA-aware noise distribution) ──
    cross_head_error_share: bool = True
    """Enable inter-head Σ-Δ error propagation for GQA models.
    In GQA (Grouped Query Attention), N_kv KV heads serve N_q query heads
    (e.g., 8 KV → 32 Q).  Each KV head's quantisation error amplifies
    across its Q-head group.  When True, the Σ-Δ error state from head i
    is passed as initial bias to head i+1, forming a ring that distributes
    error energy evenly across all KV heads.
    This prevents any single head from accumulating disproportionate
    quantisation error that then corrupts its entire Q-head group.
    Default True for GQA models, has no effect on MHA (N_kv == N_q)."""

    cross_head_error_bias: float = 0.15
    """Coupling strength for cross-head error sharing.  Controls how much
    of the previous head's error state is injected into the current head's
    Σ-Δ initial condition.  0 = no sharing, 1 = full sharing.
    Recommended 0.10-0.25; higher values risk overshoot in small-tile regimes."""

    # ── Dynamic tile size (§8.1.10 — incremental-adaptive tiling) ──────
    dynamic_tile_size: bool = True
    """Enable adaptive tile sizing for incremental (token-by-token) encoding.
    Batch encoding always uses the full tile_size (16) for Tensor Core
    efficiency.  But incremental encoding accumulates tokens one at a time:
    the first 15 tokens sit in a raw FP16 buffer before the first tile is
    encoded.  When True, tiles are formed as soon as ≥ min_tile_size tokens
    are available, using the actual buffer depth as the tile dimension.
    This avoids long raw-buffer residency while preserving power-of-2
    alignment for Tensor Core compatibility.
    Default True; set False to force fixed tile_size in all paths."""

    min_tile_size: int = 4
    """Minimum tile dimension when dynamic_tile_size is enabled.  Must be
    a power of 2 and ≤ tile_size.  4 is the pragmatic minimum: still
    vectorisable on GPU warp (32 threads) while allowing earlier encoding.
    Lower values (2) risk excessive padding overhead per tile."""

    # ── Incremental buffer ─────────────────────────────────────────────
    incremental_buffer_size: int = 128
    """Number of tokens to buffer before encoding during incremental decode.
    128 tokens = 2 KB at d_head=128 FP16, keeping GPU L1 hit rates high."""

    # ── IO / precision ─────────────────────────────────────────────────
    base_dtype: str = "fp16"
    """Base storage dtype for raw buffers and input."""

    verbose: bool = False
    """Enable diagnostic logging per encoding operation."""

    def __post_init__(self):
        """Validate constraints and auto-adjust inconsistent settings.

        Enforces the parameter interaction rules documented in
        *RINA_Whitepaper §8.4.2 (Parameter Interaction Matrix)*
        and *§10.3 (Ablation guardrails)*.
        """
        import logging
        _logger = logging.getLogger(__name__)

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
            import math
            log2 = math.log2(self.cross_token_group)
            if log2 != int(log2):
                _logger.warning(
                    f"cross_token_group ({self.cross_token_group}) is not a power of 2; "
                    f"rounding up to {2**math.ceil(log2)} for reshape alignment."
                )
                self.cross_token_group = 2 ** math.ceil(log2)

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

    def is_protected_layer(self, layer_idx: int) -> bool:
        """Check if a layer is in the protected (FP16 passthrough) list."""
        return layer_idx in self.protected_layers

    def get_beta_for_decode_step(self, decode_step: int) -> float:
        """Compute the decayed beta value for a given decode step index.

        Uses linear decay from beta_decay_start to beta_decay_end over
        beta_decay_tokens samples.  Returns self.beta when decay is
        disabled (beta_decay_tokens == 0) or when decode_step exceeds
        the decay window.

        Parameters
        ----------
        decode_step:
            0-indexed decode step number (0 = first generated token).

        Returns
        -------
        float: effective beta for this decode step.
        """
        if self.beta_decay_tokens <= 0:
            return self.beta
        if decode_step < 0:
            return self.beta_decay_start
        if decode_step >= self.beta_decay_tokens:
            return self.beta_decay_end
        t = decode_step / max(self.beta_decay_tokens - 1, 1)
        return self.beta_decay_start + t * (self.beta_decay_end - self.beta_decay_start)

    def get_layer_config(self, layer_idx: int) -> "DSKVCacheConfig":
        """Return a DSKVCacheConfig with per-layer step overrides applied.

        Creates a shallow copy of self, replacing n_steps_k and n_steps_v
        with the layer-specific values from layer_step_map (if present).
        All other parameters (tile_size, beta, order2_gamma, noise shaping,
        differential, cross_token_group, etc.) remain unchanged.

        This ensures downstream code (encode_matrix, adaptive_encode_matrix,
        incremental_encode_step) gets the correct per-layer n_steps
        without needing explicit layer_idx plumbing.

        Parameters
        ----------
        layer_idx:
            0-indexed transformer layer index.

        Returns
        -------
        DSKVCacheConfig with per-layer n_steps_k/n_steps_v overrides.
        """
        import copy
        cfg = copy.copy(self)  # shallow copy: shares mutable defaults safely
        cfg.n_steps_k = self.get_layer_steps_k(layer_idx)
        cfg.n_steps_v = self.get_layer_steps_v(layer_idx)
        return cfg

    def to_dict(self) -> dict:
        """Serialize all config fields to a plain dict.
        
        Handles non-serializable types:
        - layer_step_map keys (int) → str keys for JSON compatibility
        - protected_layers (set/list) → sorted list
        """
        import dataclasses
        d = {}
        for field in dataclasses.fields(self):
            val = getattr(self, field.name)
            if field.name == "layer_step_map" and isinstance(val, dict):
                val = {str(k): v for k, v in val.items()}
            elif field.name == "protected_layers":
                if isinstance(val, set):
                    val = sorted(list(val))
                elif isinstance(val, list):
                    val = sorted(val)
            d[field.name] = val
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "DSKVCacheConfig":
        """Reconstruct a DSKVCacheConfig from a plain dict.
        
        Handles reverse of to_dict() conversions:
        - str keys → int keys for layer_step_map
        - list → preserves as-is (DSKVCacheConfig expects list[int])
        """
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(cls)}
        kwargs = {}
        for k, v in d.items():
            if k not in field_names:
                continue  # skip unknown keys
            if k == "layer_step_map" and isinstance(v, dict):
                v = {int(layer): tuple(steps) for layer, steps in v.items()}
            kwargs[k] = v
        return cls(**kwargs)


def _default_layer_step_map() -> dict:
    """Default per-layer step allocation for 16-layer LLaMA-style models.
    
    Shallow layers (0-4):  k=3, v=4  — basic semantics, low energy
    Middle layers (5-9):   k=4, v=5  — current default  
    Deep layers (10-15):   k=5, v=6  — fine-grained positional info
    
    Layers not in the map fall back to global n_steps_k / n_steps_v.
    """
    return {
        0:  (3, 4), 1:  (3, 4), 2:  (3, 4), 3:  (3, 4), 4:  (3, 4),
        5:  (4, 5), 6:  (4, 5), 7:  (4, 5), 8:  (4, 5), 9:  (4, 5),
        10: (5, 6), 11: (5, 6), 12: (5, 6), 13: (5, 6), 14: (5, 6), 15: (5, 6),
    }