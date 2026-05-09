"""
§2 DS-KVCache Core — Incremental Tile-Based Encoding (16×16)
=============================================================

Key design:
  • raw_buffer stores FP16 K/V rows until 16 tokens accumulate
  • On tile-full (len(buf)==16), trigger R.I.N.A 1-bit encode → append to bit-packed store
  • reconstruct_all() = decode(bit_packed_history) + raw_buffer (tail <16)
  • K: 3 steps, V: 5 steps, V orthogonal transform ON
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from rina.config import DSKVCacheConfig

from modules.residual_pursuit import (
    ResidualBases,
    ResidualAlphas,
    encode_matrix,
    decode_from_bases,
)
from rina.utils.bit_packing import pack_bases, unpack_bases

_logger = logging.getLogger(__name__)


def _quantize_int8(t: torch.Tensor) -> Tuple[torch.Tensor, float]:
    scale = t.abs().max().item() / 127.0
    if scale == 0:
        scale = 1.0
    q = (t.float() / scale).round().clamp(-128, 127).to(torch.int8)
    return q, scale


def _quantize_int8_batch(tokens: torch.Tensor) -> List[Tuple[torch.Tensor, float]]:
    """Quantize a batch of tokens (N, d_head) to INT8, returning one (q, scale)
    tuple per token.  Avoids per-token CUDA→CPU sync by computing all
    per-row absmax in a single reduction."""
    absmax = tokens.abs().amax(dim=1, keepdim=True)  # (N, 1)
    scales = absmax / 127.0
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)  # avoid div-by-zero
    q_batch = (tokens.float() / scales).round().clamp(-128, 127).to(torch.int8)
    return [(q_batch[i], float(scales[i].item())) for i in range(tokens.shape[0])]


def _dequantize_int8(q: torch.Tensor, scale: float) -> torch.Tensor:
    return q.float() * scale


@dataclass
class DSKVCacheStore:
    """On-device storage for a single head's DS-encoded K/V cache.

    Incremental mode: raw_buffer holds <16 un-encoded rows; when it hits 16,
    a tile is encoded and appended to the bit-packed store.
    """

    tile_size: int = 16

    # ── Bit-packed encoded tiles (already committed) ──
    bases: Optional[torch.Tensor] = None          # (N_steps, n_tiles_encoded, M_packed) int32
    bases_shape_M: Optional[int] = None
    alphas: Optional[ResidualAlphas] = None        # (N_steps, n_tiles_encoded)
    orig_shape: Optional[Tuple[int, int]] = None   # (n_encoded_tokens, d_head)

    # ── Two-stage residual differential ──
    bases_residual: Optional[torch.Tensor] = None
    bases_shape_M_residual: Optional[int] = None
    alphas_residual: Optional[ResidualAlphas] = None
    diff_gamma: float = 0.0

    # ── Cross-token joint encoding (§8.1.5) ──────────────────────────
    cross_token_group: int = 1
    """Number of tokens grouped per matrix row before tile encoding.
    1 = per-token (original), 4 = 4-token groups."""
    original_n_tokens: Optional[int] = None
    """Pre-reshape token count; used to un-reshape after decode."""

    # ── Orthogonal transform state (V only) ──
    v_rotation_matrix: Optional[torch.Tensor] = None

    # ── Decode cache ──
    full_k_hat: Optional[torch.Tensor] = None

    # ── Weighted reconstruction (§8.1.7) ──
    recon_weights: Optional[torch.Tensor] = None
    """Per-step reconstruction weights w_i for weighted sum:
        recon = sum(w_i * alpha_i * B_i).
    If None, uses uniform w_i=1.0 (standard sum)."""

    # ── Dynamic tile size (§8.1.10) ──
    tile_pad_counts: Optional[List[int]] = None
    """Number of zero-padded rows per encoded tile when dynamic tile size
    triggers early encoding (e.g. tile_size=16 but only 4 tokens available).
    reconstruct_all() strips these rows after decoding."""

    # ── Protected mode (§8.1.8) ────────────────────────────────────────
    protected: bool = False
    """If True, all K/V are stored at FP16 in raw_buffer without any
    1-bit encoding.  Used for critical layers (first/last) where
    quantization error propagates disproportionately."""

    # ── Orthogonal transform mode (§8.2 / Roadmap 3 — DCT/DWT/Hybrid) ──
    use_fwht: bool = False  # DEPRECATED — superseded by transform_mode
    """If True, FWHT was applied during encoding; IFWHT must be applied
    during decode.  Persisted from config so reconstruct_all can
    correctly invert the Walsh-Hadamard transform.
    DEPRECATED: use ``transform_mode`` and ``transform_decisions`` for
    the DCT/DWT/Hybrid engine (§8.2)."""

    transform_mode: str = "none"
    """Transform mode applied during encoding.  One of ``"none"``,
    ``"dct"``, ``"dwt"``, ``"hybrid"``, ``"auto"``, ``"fwht"``.
    Mirrors ``DSKVCacheConfig.transform_mode``; persisted so
    reconstruct_all can apply the correct inverse transform."""

    transform_decisions: Optional[List[str]] = None
    """Per-tile transform decisions (required when transform_mode is
    ``"auto"``, ``"hybrid"``, or ``"dwt"``).  Each element is one of
    ``"dct"``, ``"dwt"``, ``"hybrid"``, or ``"fwht"``.
    Stored alongside bases/alphas so decode can invert exactly."""

    transform_pad_rows: int = 0
    """Number of zero-pad rows added to ensure total elements are
    divisible by tile_size² for 2-D transforms (DCT/DWT/Hybrid).
    reconstruct_all strips these rows after inverse transform."""

    # ── Adaptive Bit-Rate Masking (§A / Roadmap 1) ──────────────────────
    masking_decisions: Optional[List[bool]] = None
    """Per-tile sensitivity decisions from adaptive_masking.
    True = sensitive tile (boosted proj_beta / extra steps applied).
    Persisted for diagnostics; not needed during decode (already baked
    into the stored bases/alphas)."""

    # ── Original matrix shape (before transform reshape) ──
    _original_mat_shape: Optional[Tuple[int, int]] = None
    """Pre-transform matrix shape (N_orig, d_head_orig).
    Set by encode_kv_cache / _encode_and_append_tile so reconstruct_all
    can reshape from tile-space back to the original (N, d_head)."""

    # ── Encoding segments for multi-segment reconstruct (§8.1.13) ─
    _encode_segments: List[Tuple[int, int, int, int, int, int]] = field(default_factory=list)
    """List of (start_tile, end_tile, unpadded_rows, real_tokens, cross_token_pad,
    transform_pad_rows) tuples.  Each segment can be decoded independently,
    then concatenated token-by-token to avoid tile-alignment padding contamination
    across bulk and incremental encoding passes."""

    # ── Calibration (noise shaping) ──
    svd_shaper: Optional[Dict] = None

    # ── Periodic FP16 bypass (P1 anchor token refresh) ──
    _bypass_map: Dict[int, Tuple[torch.Tensor, float]] = field(default_factory=dict)
    """Position → (int8_tensor, scale).  Each entry stores a quantized INT8
    version of the original FP16 tensor with a per-token scale factor.
    reconstruct_all dequantizes and overwrites the corresponding positions,
    giving higher-quality K/V at anchor points and resetting accumulated
    quantization error."""

    _bypass_map_fp16: Dict[int, torch.Tensor] = field(default_factory=dict)
    """Position → FP16 tensor.  Used by pyramid prefill (Phase 3) for critical
    system/tail tokens that require full precision.  reconstruct_all checks
    this map after _bypass_map, so FP16 entries take priority."""

    # ── Incremental buffer (§5) ──
    raw_buffer: Optional[torch.Tensor] = None      # (B, d_head) FP16, B < tile_size
    buffer_full: int = 0

    # ── Stats ──
    memory_bytes: int = 0
    fp16_memory_bytes: int = 0
    compression_ratio: float = 0.0

    # ── Cross-token unreshape helper (§8.1.13) ─────────────────────────
    def _unreshape_to_tokens(
        self,
        mat: torch.Tensor,
        n_real_tokens: int,
        seg_pad: int = 0,
    ) -> torch.Tensor:
        """Cross-token unreshape + padding trim for a single segment.

        Parameters
        ----------
        mat:
            Decoded matrix in grouped format ``(N_tiles*G, G*d_head)``.
        n_real_tokens:
            Number of real (non-pad) tokens in this segment.
        seg_pad:
            Cross-token pad tokens added during ``_reshape_for_cross_token``.

        Returns
        -------
        Token sequence ``(n_keep, d_head)``.
        """
        Gd = mat.shape[1]
        d_head = Gd // self.cross_token_group
        flat = mat.reshape(-1, d_head)
        if seg_pad > 0:
            flat = flat[:-seg_pad]
        n_keep = min(n_real_tokens, flat.shape[0])
        return flat[:n_keep]

    # ── Weighted reconstruction (§8.1.7) ────────────────────────────────
    def compute_recon_weights(self, temperature: float = 0.5):
        """Compute energy-based per-step reconstruction weights from alphas.

        Each step's mean |alpha| indicates its contribution to the
        reconstruction.  Steps with higher energy get larger weight:
            w_i = softmax(mean_tiles(|alpha_i|) / temperature).

        Parameters
        ----------
        temperature:
            Softmax temperature.  0.5 = moderate sharpness (default).
            1.0 = near-uniform, 0.1 = nearly argmax.

        Side effects
        ------------
        Sets ``self.recon_weights`` to a ``(N_steps,)`` tensor on the
        same device as alphas.  Call after encoding is complete.
        """
        if self.alphas is None:
            return
        alpha_med = self.alphas.float().abs().median(dim=-1).values  # (N_steps,)
        if alpha_med.numel() <= 1:
            return
        weights = torch.softmax(alpha_med / temperature, dim=0)
        # Normalise so max weight = 1.0 (avoids inflating overall scale)
        weights = weights / weights.max()
        self.recon_weights = weights.to(self.alphas.dtype)

    @property
    def n_tokens(self) -> int:
        """Total logical tokens: encoded + buffered.
        Uses original_n_tokens when cross_token_group > 1."""
        # When cross_token_group > 1, orig_shape was reshaped;
        # original_n_tokens = pre-reshape count (or total for per-tile incremental)
        if self.original_n_tokens is not None and self.cross_token_group > 1:
            return self.original_n_tokens + self.buffer_full
        encoded = self.orig_shape[0] if self.orig_shape is not None else 0
        return encoded + self.buffer_full

    @property
    def n_tiles(self) -> int:
        """Number of encoded tiles in bit-packed store."""
        if self.bases is None:
            return 0
        return self.bases.shape[1]

    # ------------------------------------------------------------------
    # Incremental append (§5 — tile trigger)
    # ------------------------------------------------------------------

    def append_incremental(
        self,
        new_vec: torch.Tensor,
        *,
        cfg: DSKVCacheConfig,
        svd_shaper: Optional[dict] = None,
        v_rotation: Optional[torch.Tensor] = None,
        initial_momentum: Optional[torch.Tensor] = None,
        initial_integrator2: Optional[torch.Tensor] = None,
        bypass: bool = False,
    ) -> tuple:
        """Add one or more FP16 K/V rows.  When >= tile_size rows accumulate,
        encode a tile and commit to the bit-packed store.

        Protected mode (§8.1.8): raw_buffer grows unbounded, NO tile encoding
        ever triggered.  reconstruct_all() returns the raw buffer as-is.

        Periodic FP16 bypass (P1): when ``bypass=True``, the token is still
        Σ-Δ encoded normally AND also stored at full FP16 precision in
        ``_bypass_map``.  reconstruct_all replaces the encoded token with
        the FP16 version, effectively resetting accumulated error at that
        position.

        Parameters
        ----------
        new_vec: (B, d_head) — 1 or more new token vectors.
        cfg: Pipeline config (heterogeneous n_steps_k / n_steps_v).
        svd_shaper: Optional per-head noise shaper.
        v_rotation: Orthogonal rotation matrix (V path only).
        initial_momentum: Cross-head Σ-Δ momentum from previous head (§8.1.9).
        initial_integrator2: Cross-head second-order integrator from previous head.
        bypass: If True, record FP16 in _bypass_map (P1 anchor refresh).

        Returns
        -------
        (momentum, integrator2) — final Σ-Δ state after encoding, or (None, None)
        if no tile was encoded in this call.  Pass to next head for cross-head
        error sharing (§8.1.9).
        """
        B, d_head = new_vec.shape
        tile_size = self.tile_size
        is_v = v_rotation is not None  # V path flag: determines n_steps later

        # ── Periodic FP16 bypass (P1 anchor refresh): record current position ──
        # Skip pre-encode bypass when adaptive bypass is active — the adaptive
        # check runs after tile encoding in _encode_and_append_tile instead.
        bypass_adaptive = getattr(cfg, 'bypass_adaptive', False)
        if bypass and not self.protected and not bypass_adaptive:
            pos = self.n_tokens  # logical position before this append
            if v_rotation is not None:
                bypass_vec = (new_vec.float() @ v_rotation.float()).squeeze(0)
            else:
                bypass_vec = new_vec.squeeze(0)
            self._bypass_map[pos] = _quantize_int8(bypass_vec)

        # ── Protected mode: just accumulate raw FP16, never encode ──
        if self.protected:
            if self.raw_buffer is None:
                self.raw_buffer = new_vec.to(torch.float16)
                self.buffer_full = B
            else:
                self.raw_buffer = torch.cat([self.raw_buffer, new_vec.to(torch.float16)], dim=0)
                self.buffer_full += B
            if self.original_n_tokens is None:
                self.original_n_tokens = 0
            self.original_n_tokens += B
            if self.orig_shape is None:
                self.orig_shape = (self.raw_buffer.shape[0], d_head)
            else:
                encoded = self.orig_shape[0]
                self.orig_shape = (encoded + B, d_head)
            return initial_momentum, initial_integrator2

        # ── V-orthogonal: rotate BEFORE storing so raw_buffer is always in rotated space ──
        # This avoids the reconstruct_all bug where the buffer tail (in original space)
        # gets incorrectly un-rotated alongside the rotated encoded tiles.
        if v_rotation is not None:
            new_vec = new_vec.to(torch.float32) @ v_rotation.to(torch.float32)

        # ── First call: initialise raw_buffer ──
        if self.raw_buffer is None:
            self.raw_buffer = new_vec.to(torch.float16)
            self.buffer_full = B
        else:
            self.raw_buffer = torch.cat([self.raw_buffer, new_vec.to(torch.float16)], dim=0)
            self.buffer_full += B

        # ── Determine cross-token group policy (must happen before any tracking) ──
        # §8.1.5: persist on store so reconstruct_all can unreshape correctly.
        # K path uses at most 2-token groups; V path uses full cfg group.
        if self.cross_token_group <= 1:
            self.cross_token_group = (
                max(1, cfg.cross_token_group) if is_v
                else min(2, max(1, cfg.cross_token_group))
            )

        cross_token_group = self.cross_token_group

        # Track original_n_tokens — per-tile mode only; cross-token mode
        # handles its own tracking inside _encode_and_append_tile
        if cross_token_group <= 1:
            if self.original_n_tokens is None:
                self.original_n_tokens = 0
            self.original_n_tokens += B

        # ── Encode tiles while we have enough rows ──
        # Buffer is already in V-rotated space → _encode_and_append_tile must NOT re-rotate.
        #
        # §8.1.5 Cross-token joint encoding:
        # When cross_token_group > 1, accumulate G * tile_size tokens before encoding.
        # Reshape (G*T, d_head) → (T, G*d_head) so each tile spans G tokens,
        # distributing quantisation noise across adjacent tokens instead of
        # accumulating independently per token.

        if cross_token_group > 1:
            # ── Cross-token mode: group G * tile_size tokens per encoding unit ──
            group_trigger = tile_size * cross_token_group
            momentum, integrator2 = initial_momentum, initial_integrator2
            token_offset = self.n_tokens - self.buffer_full
            while self.buffer_full >= group_trigger:
                group_tokens = self.raw_buffer[:group_trigger].to(torch.float32)
                # Reshape: (G*T, d_head) → (T, G*d_head)
                group_reshaped = group_tokens.reshape(tile_size, cross_token_group * d_head)
                ret_momentum, ret_integrator2 = self._encode_and_append_tile(
                    group_reshaped, cfg=cfg, svd_shaper=svd_shaper, is_v=is_v,
                    initial_momentum=momentum, initial_integrator2=integrator2,
                    n_real_tokens=group_trigger,
                    tile_token_start=token_offset,
                )
                momentum, integrator2 = ret_momentum, ret_integrator2
                token_offset += group_trigger
                self.raw_buffer = self.raw_buffer[group_trigger:]
                self.buffer_full = self.raw_buffer.shape[0] if self.raw_buffer.numel() > 0 else 0
                if self.buffer_full == 0:
                    self.raw_buffer = None
                # No padding in incremental mode — exact group_trigger tokens always taken
                self._cross_token_pad = 0
        else:
            # ── Per-tile mode: encode each tile_size block independently ──
            # §8.1.10 Dynamic tile size: when enabled, use the largest
            # power-of-2 ≤ min(buffer_full, tile_size) as the effective
            # tile dimension.  This avoids long raw-buffer residency for
            # the first 15 tokens while preserving Tensor Core alignment.
            momentum, integrator2 = initial_momentum, initial_integrator2
            token_offset = self.n_tokens - self.buffer_full

            while True:
                # ── Determine effective tile size for this iteration ──
                if cfg.dynamic_tile_size and self.buffer_full < tile_size:
                    # Find largest power-of-2 ≤ buffer_full that's ≥ min_tile_size
                    dyn_ts = tile_size
                    min_ts = getattr(cfg, 'min_tile_size', 4)
                    while dyn_ts > self.buffer_full and dyn_ts > min_ts:
                        dyn_ts //= 2
                    if dyn_ts < min_ts or self.buffer_full < dyn_ts:
                        break  # not enough tokens for even the minimum tile
                    effective_tile_size = dyn_ts
                elif self.buffer_full >= tile_size:
                    effective_tile_size = tile_size
                else:
                    break  # buffer not full enough and dynamic not applicable

                # Pad to full tile_size for encode_matrix compatibility;
                # decode_from_bases will produce tile_size rows, we strip
                # the zero-padded tail afterwards via tile_pad_counts.
                tile_raw = self.raw_buffer[:effective_tile_size].to(torch.float32)
                pad_rows = tile_size - effective_tile_size
                tile = F.pad(tile_raw, (0, 0, 0, pad_rows), mode='constant', value=0.0)

                ret_momentum, ret_integrator2 = self._encode_and_append_tile(
                    tile, cfg=cfg, svd_shaper=svd_shaper, is_v=is_v,
                    initial_momentum=momentum, initial_integrator2=integrator2,
                    n_real_tokens=effective_tile_size,
                    tile_token_start=token_offset,
                )
                momentum, integrator2 = ret_momentum, ret_integrator2
                token_offset += effective_tile_size

                # Track padding so reconstruct_all can strip it
                if self.tile_pad_counts is None:
                    self.tile_pad_counts = []
                self.tile_pad_counts.append(pad_rows)

                # Keep the remainder
                self.raw_buffer = self.raw_buffer[effective_tile_size:]
                self.buffer_full = self.raw_buffer.shape[0] if self.raw_buffer.numel() > 0 else 0
                if self.buffer_full == 0:
                    self.raw_buffer = None

        return momentum, integrator2

    def _encode_and_append_tile(
        self,
        tile: torch.Tensor,
        *,
        cfg: DSKVCacheConfig,
        svd_shaper: Optional[dict] = None,
        is_v: bool = False,
        initial_momentum: Optional[torch.Tensor] = None,
        initial_integrator2: Optional[torch.Tensor] = None,
        n_real_tokens: Optional[int] = None,
        tile_token_start: int = 0,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Encode a single (tile_size, d_head) tile and concatenate to store.

        Parameters
        ----------
        is_v: True for V-path tiles → uses n_steps_v instead of n_steps_k.
            Tile is assumed already in V-rotated space; no further rotation applied.
        initial_momentum: Cross-head Σ-Δ momentum from previous head (§8.1.9).
        initial_integrator2: Cross-head second-order integrator from previous head.
        n_real_tokens: Actual number of real tokens encoded in this tile.
            None defaults to tile_size * cross_token_group.
        tile_token_start: Global token index of the first token in this tile
            (used for adaptive bypass position tracking).

        Returns
        -------
        (momentum, integrator2) — final Σ-Δ state after encoding this tile.
        """
        tile_size, d_head = tile.shape
        n_steps = cfg.get_n_steps_v() if is_v else cfg.get_n_steps_k()

        if n_real_tokens is None:
            n_real_tokens = tile_size * self.cross_token_group

        # ── Noise-shaping projector ──
        proj_matrix = None
        if cfg.use_noise_shaping and cfg.proj_rank > 0 and cfg.proj_beta > 0:
            if svd_shaper is not None:
                proj_matrix = svd_shaper.get("projector", None)
            # else: skip for incremental (too expensive per-tile)

        # ── Cross-head error sharing: request momentum return ──
        do_cross_head = (
            cfg.cross_head_error_share
            and cfg.order2_gamma > 0
        )

        # ── Primary encode (use same path as bulk: _encode_single_path) ──
        if do_cross_head:
            if initial_momentum is None:
                initial_momentum = torch.zeros(1, tile_size * tile_size, device=tile.device, dtype=tile.dtype)
                initial_integrator2 = torch.zeros(1, tile_size * tile_size, device=tile.device, dtype=tile.dtype) if cfg.order2_gamma > 0 else None
            result = _encode_single_path(
                tile,
                n_steps=n_steps,
                cfg=cfg,
                proj_matrix=proj_matrix,
                initial_momentum=initial_momentum,
                initial_integrator2=initial_integrator2,
                return_momentum=True,
            )
            bases, alphas, shape, final_momentum, final_integrator2, tile_xform_decisions, tile_mask_decisions, _pad_rows = result
        else:
            result = _encode_single_path(
                tile,
                n_steps=n_steps,
                cfg=cfg,
                proj_matrix=proj_matrix,
            )
            bases, alphas, shape, tile_xform_decisions, tile_mask_decisions, _pad_rows = result
            final_momentum, final_integrator2 = None, None

        # ── Align n_steps with existing store (adaptive_masking fix) ──
        # encode_matrix with adaptive_masking=True trims bases to max_used
        # across all tiles in the batch.  For a single incremental tile,
        # max_used may be lower than the bulk-encoded store, causing a
        # dimension-0 mismatch on concatenation.  Pad the new tile (or
        # existing store) with neutral values to align.
        if self.bases is not None:
            store_n = self.bases.shape[0]       # packed: dim 0 = n_steps
            new_n = bases.shape[0]              # float: dim 0 = n_steps
            if new_n < store_n:
                # Pad new tile with neutral values: bases=1.0, alphas=0.0
                pad_n = store_n - new_n
                pad_bases = torch.ones(pad_n, *bases.shape[1:],
                                       device=bases.device, dtype=bases.dtype)
                pad_alphas = torch.zeros(pad_n, *alphas.shape[1:],
                                         device=alphas.device, dtype=alphas.dtype)
                bases = torch.cat([bases, pad_bases], dim=0)
                alphas = torch.cat([alphas, pad_alphas], dim=0)
            elif new_n > store_n:
                # Rare: existing store has fewer steps.  Pad the store
                # (already packed + fp16) to match.
                pad_n = new_n - store_n
                # Packed all-ones bases = -1 in int32
                all_ones_packed_val = -1
                pad_packed = torch.full(
                    (pad_n, *self.bases.shape[1:]),
                    all_ones_packed_val,
                    device=self.bases.device, dtype=self.bases.dtype,
                )
                self.bases = torch.cat([self.bases, pad_packed], dim=0)
                pad_alphas = torch.zeros(
                    pad_n, *self.alphas.shape[1:],
                    device=self.alphas.device, dtype=self.alphas.dtype,
                )
                self.alphas = torch.cat([self.alphas, pad_alphas], dim=0)
                # Also pad residual store if present
                if self.bases_residual is not None:
                    pad_res = torch.full(
                        (pad_n, *self.bases_residual.shape[1:]),
                        all_ones_packed_val,
                        device=self.bases_residual.device,
                        dtype=self.bases_residual.dtype,
                    )
                    self.bases_residual = torch.cat([self.bases_residual, pad_res], dim=0)
                    pad_alpha_res = torch.zeros(
                        pad_n, *self.alphas_residual.shape[1:],
                        device=self.alphas_residual.device,
                        dtype=self.alphas_residual.dtype,
                    )
                    self.alphas_residual = torch.cat(
                        [self.alphas_residual, pad_alpha_res], dim=0,
                    )

        bases_M = bases.shape[-1]
        packed = pack_bases(bases)

        # ── Primary reconstruction (used by differential AND adaptive bypass) ──
        bypass_adaptive = getattr(cfg, 'bypass_adaptive', False)
        need_primary = (cfg.use_differential and cfg.diff_strategy == "residual") or (bypass_adaptive and not self.protected)
        if need_primary:
            primary = decode_from_bases(bases, alphas, shape, tile_size=tile_size)

        # ── Adaptive bypass (Phase 1): record FP16 for tokens with high L∞ error ──
        if bypass_adaptive and not self.protected:
            bypass_threshold = getattr(cfg, 'bypass_threshold', 0.5)
            cg = self.cross_token_group if self.cross_token_group >= 1 else 1
            if cg > 1:
                d_head_orig = tile.shape[1] // cg
                tile_tokens = tile.reshape(-1, d_head_orig)
                primary_tokens = primary.reshape(-1, d_head_orig)
            else:
                tile_tokens = tile
                primary_tokens = primary
            n_tokens_in_tile = min(tile_tokens.shape[0], n_real_tokens if n_real_tokens else tile_tokens.shape[0])
            for i in range(n_tokens_in_tile):
                err = (tile_tokens[i] - primary_tokens[i]).abs().max().item()
                if err > bypass_threshold:
                    self._bypass_map[tile_token_start + i] = _quantize_int8(tile_tokens[i])

        # ── Two-stage residual ──
        bases_res, alphas_res = None, None
        bases_shape_M_res = None
        if cfg.use_differential and cfg.diff_strategy == "residual":
            # Compute residual in transform domain to match primary tile layout.
            transform_mode = getattr(cfg, 'transform_mode', 'none')
            if transform_mode and transform_mode not in ("none", "", None, "fwht"):
                from rina.utils.transforms import apply_transform, TransformMode
                try:
                    tf_mode = TransformMode[transform_mode.upper()]
                except KeyError:
                    tf_mode = TransformMode(transform_mode)
                tile_d = tile_size ** 2
                N_t, d_t = tile.shape
                total_elems = N_t * d_t
                if total_elems % tile_d != 0:
                    needed_elems = ((total_elems + tile_d - 1) // tile_d) * tile_d
                    pad_elems = needed_elems - total_elems
                    pad_rows = (pad_elems + d_t - 1) // d_t
                    tile_padded = F.pad(tile, (0, 0, 0, pad_rows), mode='constant', value=0.0)
                else:
                    tile_padded = tile
                tile_transformed, _ = apply_transform(
                    tile_padded, mode=tf_mode, tile_size=tile_size,
                )
                residual = tile_transformed - primary
            else:
                residual = tile - primary
            bases_res, alphas_res, _res_shape, _ = encode_matrix(
                residual,
                n_steps=cfg.diff_residual_n_steps,
                tile_size=tile_size,
                beta=cfg.beta,
                proj_matrix=None,
                proj_beta=0.0,
                adaptive_eta=False,
            )
            bases_shape_M_res = bases_res.shape[-1]
            bases_res = pack_bases(bases_res)
            alphas_res = alphas_res.to(torch.float16)

        alphas = alphas.to(torch.float16)

        # ── Align residual n_steps with existing residual store ──
        # Same adaptive_masking dimension-drift issue as primary bases.
        if bases_res is not None and self.bases_residual is not None:
            store_n = self.bases_residual.shape[0]
            new_n = bases_res.shape[0]
            if new_n < store_n:
                pad_n = store_n - new_n
                all_ones = -1  # packed all-ones value
                pad_bases_res = torch.full(
                    (pad_n, *bases_res.shape[1:]),
                    all_ones,
                    device=bases_res.device, dtype=bases_res.dtype,
                )
                bases_res = torch.cat([bases_res, pad_bases_res], dim=0)
                pad_alpha_res = torch.zeros(
                    pad_n, *alphas_res.shape[1:],
                    device=alphas_res.device, dtype=alphas_res.dtype,
                )
                alphas_res = torch.cat([alphas_res, pad_alpha_res], dim=0)
            elif new_n > store_n:
                pad_n = new_n - store_n
                all_ones = -1
                pad_store_res = torch.full(
                    (pad_n, *self.bases_residual.shape[1:]),
                    all_ones,
                    device=self.bases_residual.device,
                    dtype=self.bases_residual.dtype,
                )
                self.bases_residual = torch.cat(
                    [self.bases_residual, pad_store_res], dim=0,
                )
                pad_alpha_store = torch.zeros(
                    pad_n, *self.alphas_residual.shape[1:],
                    device=self.alphas_residual.device,
                    dtype=self.alphas_residual.dtype,
                )
                self.alphas_residual = torch.cat(
                    [self.alphas_residual, pad_alpha_store], dim=0,
                )

        # ── Set diff_gamma for incremental path (was missing → residual never applied) ──
        if cfg.use_differential and bases_res is not None:
            self.diff_gamma = cfg.get_diff_residual_gamma_k() if not is_v else cfg.diff_residual_gamma

        # ── Persist transform mode from config (first tile only) ──
        transform_mode = getattr(cfg, 'transform_mode', 'none')
        if transform_mode and transform_mode not in ("none", "", None):
            if not self.transform_mode or self.transform_mode == "none":
                self.transform_mode = transform_mode

        # ── Concat to existing store ──
        old_n_tiles = 0 if self.bases is None else self.bases.shape[1]
        if self.bases is None:
            self.bases = packed                # (N, 1, M_packed)
            self.alphas = alphas               # (N, 1)
            self.orig_shape = shape
            # Track original (pre-reshape) token count for cross-token unreshape
            if self.cross_token_group > 1:
                # (tile_size, G*d_head) encodes tile_size * G real tokens
                self.original_n_tokens = tile_size * self.cross_token_group
            if bases_res is not None:
                self.bases_residual = bases_res
                self.bases_shape_M_residual = bases_shape_M_res
                self.alphas_residual = alphas_res
            self.bases_shape_M = bases_M
            # ── Roadmap 3 & 1: store per-tile decisions ──
            if tile_xform_decisions is not None:
                self.transform_decisions = list(tile_xform_decisions)
            if tile_mask_decisions is not None:
                self.masking_decisions = list(tile_mask_decisions)
        else:
            # Concatenate bases along tile dim (dim=1)
            self.bases = torch.cat([self.bases, packed], dim=1)
            self.alphas = torch.cat([self.alphas, alphas], dim=1)
            # orig_shape tracks pre-padding matrix dimensions.
            # Increment by the padded row count to keep total tiles consistent:
            # ceil(orig_shape[0]/tile_size)*tile_size + pad_rows_for_new_tile
            encoded_tokens = self.orig_shape[0]
            padded_new = ((shape[0] + tile_size - 1) // tile_size) * tile_size
            self.orig_shape = (encoded_tokens + padded_new, shape[1])
            # Accumulate original (pre-reshape) token count
            if self.cross_token_group > 1:
                if self.original_n_tokens is None:
                    # Transition from per-tile to cross-token: convert existing count
                    self.original_n_tokens = encoded_tokens
                self.original_n_tokens += tile_size * self.cross_token_group
            if bases_res is not None:
                if self.bases_residual is not None:
                    self.bases_residual = torch.cat([self.bases_residual, bases_res], dim=1)
                    self.alphas_residual = torch.cat([self.alphas_residual, alphas_res], dim=1)
                else:
                    self.bases_residual = bases_res
                    self.bases_shape_M_residual = bases_shape_M_res
                    self.alphas_residual = alphas_res
            # ── Roadmap 3 & 1: append per-tile decisions ──
            if tile_xform_decisions is not None:
                if self.transform_decisions is None:
                    self.transform_decisions = list(tile_xform_decisions)
                else:
                    self.transform_decisions.extend(tile_xform_decisions)
            if tile_mask_decisions is not None:
                if self.masking_decisions is None:
                    self.masking_decisions = list(tile_mask_decisions)
                else:
                    self.masking_decisions.extend(tile_mask_decisions)

        # ── Append segment metadata for multi-segment reconstruct (§8.1.13) ──
        n_new_tiles = packed.shape[1]
        self._encode_segments.append((
            old_n_tiles,
            old_n_tiles + n_new_tiles,
            shape[0],
            n_real_tokens,
            0,
            _pad_rows,
        ))

        # Invalidate decode cache
        self.full_k_hat = None

        # ── Weighted reconstruction (§8.1.7) — recompute after each append ──
        if hasattr(cfg, 'use_recon_weights') and cfg.use_recon_weights:
            self.compute_recon_weights(temperature=getattr(cfg, 'recon_weight_temperature', 0.5))

        return final_momentum, final_integrator2

    # ------------------------------------------------------------------
    # Full reconstruction
    # ------------------------------------------------------------------

    def reconstruct_all(
        self,
        tile_size: int = 16,
        use_differential: bool = True,
    ) -> torch.Tensor:
        """Return (original_n_tokens, d_head) — decoded bit-packed history + raw_buffer tail.
        
        Handles cross-token unreshape when cross_token_group > 1.
        V un-rotation is applied AFTER cross-token unreshape to ensure
        the rotation operates on the correct d_head dimension.

        §A Roadmap 3: Inverse DCT/DWT/Hybrid transform applied after
        primary decode (and residual if active), BEFORE cross-token
        unreshape and V un-rotation.

        §8.1.13 Multi-segment decode: When ``_encode_segments`` has >1 entry,
        each segment is decoded independently to avoid tile-alignment padding
        contamination across bulk and incremental encoding passes.
        """
        # ── Determine transform inversion policy ────────────────────────
        transform_mode = getattr(self, 'transform_mode', 'none')
        transform_decisions = getattr(self, 'transform_decisions', None)
        do_inverse_transform = (
            transform_mode and transform_mode not in ("none", "", "fwht", None)
        )

        segments = getattr(self, '_encode_segments', None)

        # ── Multi-segment decode path (§8.1.13) ─────────────────────────
        if segments is not None and len(segments) > 1 and self.bases is not None:
            # Invalidate cache — multi-segment decode always rebuilds
            self.full_k_hat = None
            bases = unpack_bases(self.bases)
            if self.bases_shape_M is not None and bases.shape[-1] > self.bases_shape_M:
                bases = bases[..., :self.bases_shape_M]

            bases_res = None
            if use_differential and self.bases_residual is not None and self.diff_gamma > 0:
                bases_res = unpack_bases(self.bases_residual)
                if self.bases_shape_M_residual is not None and bases_res.shape[-1] > self.bases_shape_M_residual:
                    bases_res = bases_res[..., :self.bases_shape_M_residual]

            # Resolve transform mode enum once
            tf_mode = None
            if do_inverse_transform:
                from rina.utils.transforms import apply_inverse_transform, TransformMode
                if isinstance(transform_mode, str):
                    try:
                        tf_mode = TransformMode[transform_mode.upper()]
                    except KeyError:
                        tf_mode = TransformMode(transform_mode)
                else:
                    tf_mode = transform_mode

            token_parts = []
            for start, end, seg_unpadded_rows, n_real_tokens, seg_pad, seg_transform_pad in segments:
                seg_bases = bases[:, start:end]
                seg_alphas = self.alphas[:, start:end] if self.alphas is not None else None
                seg_orig_shape = (seg_unpadded_rows, self.orig_shape[1])

                # ── Primary decode ──
                mat_seg = decode_from_bases(
                    seg_bases, seg_alphas, seg_orig_shape, tile_size=tile_size,
                    recon_weights=self.recon_weights,
                    use_fwht=self.use_fwht,
                )

                # ── Differential residual ──
                if bases_res is not None:
                    seg_bases_res = bases_res[:, start:end]
                    seg_alphas_res = self.alphas_residual[:, start:end] if self.alphas_residual is not None else None
                    mat_res_seg = decode_from_bases(
                        seg_bases_res, seg_alphas_res, seg_orig_shape, tile_size=tile_size,
                        use_fwht=self.use_fwht,
                    )
                    mat_seg = mat_seg + self.diff_gamma * mat_res_seg

                # ── Inverse DCT/DWT/Hybrid transform ──
                if do_inverse_transform and tf_mode is not None:
                    if self.cross_token_group > 1:
                        spatial_rows = seg_unpadded_rows * self.cross_token_group
                        spatial_cols = self.orig_shape[1] // self.cross_token_group
                    else:
                        # orig_shape is in transform domain (e.g. (n_tiles, 256)).
                        # Use _original_mat_shape to recover pre-transform spatial dims.
                        orig_mat = getattr(self, '_original_mat_shape', None)
                        if orig_mat is not None:
                            d_head_spatial = orig_mat[1]
                            spatial_rows = n_real_tokens + seg_transform_pad
                            spatial_cols = d_head_spatial
                        else:
                            spatial_rows, spatial_cols = seg_orig_shape
                    seg_xform_decisions = transform_decisions[start:end] if transform_decisions is not None else None
                    mat_seg = apply_inverse_transform(
                        mat_seg,
                        mode=tf_mode,
                        tile_size=tile_size,
                        decisions=seg_xform_decisions,
                        original_shape=(spatial_rows, spatial_cols),
                    )
                    if seg_transform_pad > 0 and mat_seg.shape[0] >= seg_transform_pad:
                        mat_seg = mat_seg[:-seg_transform_pad]

                # ── Cross-token unreshape → token sequence ──
                if self.cross_token_group > 1:
                    token_seq = self._unreshape_to_tokens(mat_seg, n_real_tokens, seg_pad)
                else:
                    token_seq = mat_seg[:n_real_tokens]

                token_parts.append(token_seq)

            result = torch.cat(token_parts, dim=0)

            # ── Append raw buffer tail ──
            if self.raw_buffer is not None and self.buffer_full > 0:
                tail = self.raw_buffer.to(torch.float32)
                result = torch.cat([result, tail], dim=0)

            # ── V un-rotation (on final concatenated result) ──
            if self.v_rotation_matrix is not None:
                R_T = self.v_rotation_matrix.T.to(result.dtype)
                if R_T.shape[-1] == result.shape[-1]:
                    result = result @ R_T
                else:
                    _logger.debug(
                        "V rotation shape %s does not match result %s; skipping un-rotation. "
                        "This is expected when cross_token_group > 1.",
                        R_T.shape, result.shape,
                    )

        # ── Periodic FP16 bypass (P1): override bypass positions (INT8 + FP16) ──
            if self._bypass_map:
                for pos, (q, s) in self._bypass_map.items():
                    if pos < result.shape[0]:
                        result[pos] = _dequantize_int8(q, s).to(result.dtype)
            if self._bypass_map_fp16:
                for pos, fp16_tensor in self._bypass_map_fp16.items():
                    if pos < result.shape[0]:
                        result[pos] = fp16_tensor.to(result.dtype)

            return result

        # ── Single-segment / legacy path ────────────────────────────────
        decoded_parts = []

        # Decode bit-packed encoded tiles
        if self.bases is not None:
            if self.full_k_hat is not None:
                mat = self.full_k_hat
            else:
                bases = unpack_bases(self.bases)
                if self.bases_shape_M is not None and bases.shape[-1] > self.bases_shape_M:
                    bases = bases[..., :self.bases_shape_M]
                mat_primary = decode_from_bases(
                    bases, self.alphas, self.orig_shape, tile_size=tile_size,
                    recon_weights=self.recon_weights,
                    use_fwht=self.use_fwht,
                )

                if use_differential and self.bases_residual is not None and self.diff_gamma > 0:
                    bases_res = unpack_bases(self.bases_residual)
                    if self.bases_shape_M_residual is not None and bases_res.shape[-1] > self.bases_shape_M_residual:
                        bases_res = bases_res[..., :self.bases_shape_M_residual]
                    mat_residual = decode_from_bases(
                        bases_res, self.alphas_residual, self.orig_shape, tile_size=tile_size,
                        use_fwht=self.use_fwht,
                    )
                    mat = mat_primary + self.diff_gamma * mat_residual
                else:
                    mat = mat_primary

                # ── §A Roadmap 3: Inverse DCT/DWT/Hybrid transform ─────
                if do_inverse_transform:
                    from rina.utils.transforms import apply_inverse_transform, TransformMode
                    # Resolve string → TransformMode enum
                    if isinstance(transform_mode, str):
                        try:
                            tf_mode = TransformMode[transform_mode.upper()]
                        except KeyError:
                            tf_mode = TransformMode(transform_mode)
                    else:
                        tf_mode = transform_mode
                    # Compute spatial-domain shape (pre-transform) for proper
                    # inverse reshape.  orig_shape is in transform domain
                    # (e.g., (2, 256) for cross-token-grouped DCT).
                    # Convert to spatial padded shape: (N*G, d_head).
                    transform_pad_rows = getattr(self, 'transform_pad_rows', 0)
                    if self.cross_token_group > 1:
                        spatial_rows = self.orig_shape[0] * self.cross_token_group
                        spatial_cols = self.orig_shape[1] // self.cross_token_group
                    else:
                        orig_mat = getattr(self, '_original_mat_shape', None)
                        if orig_mat is not None:
                            d_head_spatial = orig_mat[1]
                            spatial_rows = orig_mat[0] + transform_pad_rows
                            spatial_cols = d_head_spatial
                        else:
                            spatial_rows, spatial_cols = self.orig_shape
                    mat = apply_inverse_transform(
                        mat,
                        mode=tf_mode,
                        tile_size=tile_size,
                        decisions=transform_decisions,
                        original_shape=(spatial_rows, spatial_cols),
                    )
                    # Strip transform padding rows (added before forward
                    # transform for tile_size² alignment)
                    if transform_pad_rows > 0 and mat.shape[0] >= transform_pad_rows:
                        mat = mat[:-transform_pad_rows]

                # ── Cross-token unreshape (§8.1.5) ─────────────────────
                if self.cross_token_group > 1 and self.original_n_tokens is not None:
                    Gd = mat.shape[1]
                    # Determine if mat is already per-token (d_head cols)
                    # or still in grouped format (G*d_head cols).
                    real_d_head = None
                    orig_mat_shape = getattr(self, '_original_mat_shape', None)
                    if orig_mat_shape is not None:
                        real_d_head = orig_mat_shape[1]
                    if real_d_head is not None and Gd == real_d_head:
                        # Already per-token — trim pad tokens only
                        mat = mat[:self.original_n_tokens]
                    else:
                        d_head = Gd // self.cross_token_group
                        pad_tokens = getattr(self, '_cross_token_pad', 0)
                        N_encoded = mat.shape[0]
                        flat = mat.reshape(N_encoded * self.cross_token_group, d_head)
                        if pad_tokens > 0:
                            flat = flat[:-pad_tokens]
                        mat = flat[:self.original_n_tokens]

                # ── Strip dynamic tile pad rows (§8.1.10) ─────────────
                if self.tile_pad_counts is not None:
                    total_pad = sum(self.tile_pad_counts)
                    if total_pad > 0 and mat.shape[0] >= len(self.tile_pad_counts) * tile_size:
                        n_full_tiles = len(self.tile_pad_counts)
                        mat_tiles = mat[:n_full_tiles * tile_size].reshape(
                            n_full_tiles, tile_size, -1,
                        )
                        keep_chunks = []
                        for i in range(n_full_tiles):
                            keep = tile_size - self.tile_pad_counts[i]
                            if keep > 0:
                                keep_chunks.append(mat_tiles[i, :keep])
                        if keep_chunks:
                            mat = torch.cat(keep_chunks, dim=0)
                        else:
                            mat = mat[:0]  # empty (all padding) → no rows

                self.full_k_hat = mat  # cache for next call
            decoded_parts.append(mat)

        # Append raw buffer tail
        # Tail is in spatial domain (never forward-transformed during
        # incremental accumulation) — skip inverse transform.
        if self.raw_buffer is not None and self.buffer_full > 0:
            tail = self.raw_buffer.to(torch.float32)
            decoded_parts.append(tail)

        if not decoded_parts:
            return torch.empty(0, 0)

        result = torch.cat(decoded_parts, dim=0)

        # ── V un-rotation (applied after cross-token unreshape) ──
        if self.v_rotation_matrix is not None:
            R_T = self.v_rotation_matrix.T.to(result.dtype)
            # Safety: only apply if rotation matrix dimension matches result
            # (may be disabled when cross_token_group > 1 changes effective tile width)
            if R_T.shape[-1] == result.shape[-1]:
                result = result @ R_T
            else:
                _logger.debug(
                    "V rotation shape %s does not match result %s; skipping un-rotation. "
                    "This is expected when cross_token_group > 1.",
                    R_T.shape, result.shape,
                )

        # ── Restore original matrix shape (undo transform reshape) ──
        # When a 2-D transform (DCT/DWT/Hybrid) is used, the encoding
        # reshapes (N_orig, d_head) → (n_tiles, tile_size²).  After
        # inverse transform the result may still be in tile-space.
        # _original_mat_shape records the pre-transform shape so we can
        # reshape back.
        #
        # CRITICAL: Only reshape when result is NOT already in token-space
        # (last dim != d_orig).  When cross-token unreshape or raw buffer
        # tail has already converted the result to (N, d_head), the reshape
        # must be a no-op to avoid silently discarding tail tokens.
        orig_shape = getattr(self, '_original_mat_shape', None)
        if orig_shape is not None:
            N_orig, d_orig = orig_shape
            if result.shape[-1] != d_orig and result.numel() >= N_orig * d_orig:
                # Result is in grouped/tile format — reshape the encoding
                # prefix only; additional elements (raw buffer tail) are
                # already in token-space and must be preserved.
                excess = result.numel() - N_orig * d_orig
                if excess > 0:
                    # Separate encoding prefix from tail, reshape prefix
                    prefix_flat = result.flatten()[:N_orig * d_orig]
                    prefix = prefix_flat.reshape(N_orig, d_orig)
                    suffix_flat = result.flatten()[N_orig * d_orig:]
                    if suffix_flat.numel() > 0:
                        result = torch.cat([
                            prefix.flatten(), suffix_flat,
                        ]).reshape(-1, d_orig)
                    else:
                        result = prefix
                else:
                    result = result.reshape(N_orig, d_orig)
            elif result.numel() == N_orig * d_orig and result.shape != (N_orig, d_orig):
                result = result.reshape(N_orig, d_orig)

        # ── Strip zero-pad rows added for 2-D transform tile alignment ──
        # Only applies when _original_mat_shape reshape did NOT already
        # crop to the exact N_orig*d_head element count (which already
        # implicitly strips the pad rows).
        if orig_shape is None:
            transform_pad_rows = getattr(self, 'transform_pad_rows', 0)
            if transform_pad_rows > 0 and result.shape[0] >= transform_pad_rows:
                result = result[:-transform_pad_rows]

        # ── Periodic FP16 bypass (P1): override bypass positions (INT8 + FP16) ──
        if self._bypass_map:
            for pos, (q, s) in self._bypass_map.items():
                if pos < result.shape[0]:
                    result[pos] = _dequantize_int8(q, s).to(result.dtype)
        if self._bypass_map_fp16:
            for pos, fp16_tensor in self._bypass_map_fp16.items():
                if pos < result.shape[0]:
                    result[pos] = fp16_tensor.to(result.dtype)

        return result

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def update_stats(self):
        """Recalculate memory footprint."""
        d_head = self.orig_shape[1] if self.orig_shape is not None else (
            self._original_mat_shape[1] if self._original_mat_shape is not None else
            self.raw_buffer.shape[1] if self.raw_buffer is not None else 0
        )
        total_tokens = self.n_tokens
        self.fp16_memory_bytes = total_tokens * d_head * 2

        total = 0
        for packed_attr in ("bases", "bases_residual"):
            tensor = getattr(self, packed_attr, None)
            if tensor is not None:
                total += (tensor.numel() * 32) // 8
        for fp16_attr in ("alphas", "alphas_residual"):
            tensor = getattr(self, fp16_attr, None)
            if tensor is not None:
                total += (tensor.numel() * 16) // 8
        if self.raw_buffer is not None and self.buffer_full > 0:
            total += self.buffer_full * d_head * 2
        if self._bypass_map:
            for q, _ in self._bypass_map.values():
                total += q.numel()  # INT8 = 1 byte per element
            total += len(self._bypass_map) * 4  # float32 scales
        if self._bypass_map_fp16:
            for t in self._bypass_map_fp16.values():
                total += t.numel() * 2  # FP16 = 2 bytes per element

        self.memory_bytes = total
        self.compression_ratio = self.fp16_memory_bytes / (total + 1e-12)


# ══════════════════════════════════════════════════════════════════════════════
# Legacy bulk encoder (for eval scripts)
# ══════════════════════════════════════════════════════════════════════════════


def _encode_single_path(
    mat: torch.Tensor,
    n_steps: int,
    cfg: DSKVCacheConfig,
    proj_matrix: Optional[torch.Tensor] = None,
    initial_momentum: Optional[torch.Tensor] = None,
    initial_integrator2: Optional[torch.Tensor] = None,
    return_momentum: bool = False,
) -> Tuple:
    """Encode a single matrix path with optional cross-head momentum.

    Returns extend to ``(bases, alphas, orig_shape, momentum, integrator2,
    transform_decisions, masking_decisions)`` when ``return_momentum=True``
    (§8.1.9 cross-head error sharing).

    Roadmaps wired here:
      §A Roadmap 1 — Adaptive Bit-Rate Masking (per-tile outlier/anchor detection)
      §A Roadmap 3 — DCT/DWT/Hybrid orthogonal transform engine
    """
    tile_size = cfg.tile_size
    tile_d = tile_size ** 2
    per_tile_proj = proj_matrix

    if proj_matrix is not None and proj_matrix.shape[-1] != tile_d:
        proj_matrix = None
        per_tile_proj = None

    # ── Pad mat so total elements are divisible by tile_size² for 2-D transforms ──
    N_orig, d_head_orig = mat.shape
    transform_pad_rows = 0
    transform_mode = getattr(cfg, 'transform_mode', 'none')
    if transform_mode and transform_mode not in ("none", "", None, "fwht"):
        total_elems = N_orig * d_head_orig
        if total_elems % tile_d != 0:
            needed_elems = ((total_elems + tile_d - 1) // tile_d) * tile_d
            pad_elems = needed_elems - total_elems
            transform_pad_rows = (pad_elems + d_head_orig - 1) // d_head_orig
            mat = F.pad(mat, (0, 0, 0, transform_pad_rows), mode='constant', value=0.0)

    # ── §A Roadmap 3: Orthogonal transform BEFORE encoding ────────────────
    transform_decisions = None
    mat_enc = mat
    if transform_mode and transform_mode not in ("none", "", None):
        from rina.utils.transforms import apply_transform, TransformMode
        # Resolve string → TransformMode enum
        if isinstance(transform_mode, str):
            try:
                tf_mode = TransformMode[transform_mode.upper()]
            except KeyError:
                tf_mode = TransformMode(transform_mode)
        else:
            tf_mode = transform_mode
        mat_enc, transform_decisions = apply_transform(
            mat_enc,
            mode=tf_mode,
            tile_size=tile_size,
            smooth_threshold=getattr(cfg, 'transform_smooth_threshold', 0.05),
            outlier_threshold=getattr(cfg, 'transform_outlier_threshold', 3.0),
        )

    # ── §A Roadmap 1: Adaptive Bit-Rate Masking ──────────────────────────
    adaptive_masking = getattr(cfg, 'adaptive_masking', False)
    mask_decisions = None
    n_steps_per_tile = n_steps  # default uniform
    if adaptive_masking and transform_decisions is None:
        # Compute sensitivity per tile from raw mat (before any transform)
        from rina.utils.transforms import compute_tile_diagnostics
        # mat may not be tile-aligned — pad to tile_size² boundary
        flat_diag = mat.reshape(-1)
        pad_diag = (tile_d - flat_diag.numel() % tile_d) % tile_d
        if pad_diag > 0:
            flat_diag = F.pad(flat_diag, (0, pad_diag))
        tiled_diag = flat_diag.reshape(-1, tile_size, tile_size)
        variances, max_abs_vals = compute_tile_diagnostics(tiled_diag)
        stds = variances.sqrt().clamp_min(1e-8)
        outlier_thr = getattr(cfg, 'mask_outlier_threshold', 3.0)
        mask_decisions = (max_abs_vals > outlier_thr * stds).tolist()
        # Per-tile extra steps for sensitive tiles
        n_boost = getattr(cfg, 'mask_n_steps_boost', 1)
        if any(mask_decisions) and n_boost > 0:
            # We handle per-tile n_steps by encoding sensitive tiles with extra steps
            # Simple approach: encode all tiles with base n_steps, then re-encode
            # sensitive tiles with extra steps
            pass  # handled in encode_matrix via per-tile adaptive logic
    if adaptive_masking and transform_mode not in ("none", "", None):
        # When transform is active, compute mask on transformed tiles
        from rina.utils.transforms import compute_tile_diagnostics
        tiled_diag = mat_enc.reshape(-1, tile_size, tile_size)
        variances, max_abs_vals = compute_tile_diagnostics(tiled_diag)
        stds = variances.sqrt().clamp_min(1e-8)
        outlier_thr = getattr(cfg, 'mask_outlier_threshold', 3.0)
        mask_decisions = (max_abs_vals > outlier_thr * stds).tolist()

    # ── Build encode kwargs ──────────────────────────────────────────────
    encode_kwargs = dict(
        tile_size=tile_size,
        beta=cfg.beta,
        proj_matrix=per_tile_proj,
        proj_beta=cfg.proj_beta if per_tile_proj is not None else 0.0,
        adaptive_eta=cfg.adaptive_eta,
        order2_gamma=cfg.order2_gamma,
        order2_c1=cfg.order2_c1,
        order2_c2=cfg.order2_c2,
        initial_momentum=initial_momentum,
        initial_integrator2=initial_integrator2,
        return_momentum=return_momentum,
        use_fwht=cfg.use_fwht if transform_mode in ("none", "", None, "fwht") else False,
        zero_mean_integrator2=cfg.zero_mean_integrator2,
        use_mask_gating=getattr(cfg, 'use_mask_gating', True),
    )

    # §A Roadmap 1: Adaptive Bit-Rate Masking (§8.2.1)
    # Forward adaptive_masking + all per-tile boost config to encode_matrix.
    # encode_matrix's adaptive_masking branch handles per-tile sensitivity
    # internally using tile diagnostics (variance/max-abs), so we don't
    # need to precompute mask_decisions here — just pass the config.
    if adaptive_masking:
        encode_kwargs['adaptive_masking'] = True
        encode_kwargs['mask_smooth_threshold'] = getattr(cfg, 'mask_smooth_threshold', 0.05)
        encode_kwargs['mask_outlier_threshold'] = getattr(cfg, 'mask_outlier_threshold', 3.0)
        encode_kwargs['mask_proj_beta_boost'] = getattr(cfg, 'mask_proj_beta_boost', 0.5)
        encode_kwargs['mask_n_steps_boost'] = getattr(cfg, 'mask_n_steps_boost', 1)

    if cfg.adaptive_n:
        from modules.residual_pursuit import adaptive_encode_matrix
        n_extra = max(cfg.n_upper_bound - n_steps, 2)
        # adaptive_encode_matrix doesn't accept momentum/tracker kwargs
        adaptive_kwargs = {
            k: v for k, v in encode_kwargs.items()
            if k not in ("initial_momentum", "initial_integrator2", "return_momentum",
                         "use_fwht", "zero_mean_integrator2", "use_mask_gating")
        }
        result = adaptive_encode_matrix(
            mat_enc,
            n_steps_base=n_steps,
            n_steps_extra=n_extra,
            energy_threshold_ratio=cfg.energy_threshold_factor,
            **adaptive_kwargs,
        )
        if return_momentum:
            bases, alphas, _, orig_shape, momentum, integrator2 = result
            return bases, alphas, orig_shape, momentum, integrator2, transform_decisions, mask_decisions, transform_pad_rows
        bases, alphas, _, orig_shape = result
        return bases, alphas, orig_shape, transform_decisions, mask_decisions, transform_pad_rows
    else:
        result = encode_matrix(mat_enc, n_steps=n_steps, **encode_kwargs)
        if return_momentum:
            # encode_matrix returns (bases, alphas, orig_shape, xform_dec, momentum, integrator2)
            bases, alphas, orig_shape, _inner_xform, momentum, integrator2 = result
            return bases, alphas, orig_shape, momentum, integrator2, transform_decisions, mask_decisions, transform_pad_rows
        # encode_matrix returns (bases, alphas, orig_shape, xform_dec)
        bases, alphas, orig_shape, _inner_xform = result
        return bases, alphas, orig_shape, transform_decisions, mask_decisions, transform_pad_rows


def _build_v_rotation(k: torch.Tensor) -> Optional[torch.Tensor]:
    """Build a square (d_head × d_head) orthogonal transform from K's SVD.

    Using ``full_matrices=True`` guarantees a rotation that preserves
    dimensionality — critical because V will later be multiplied by this
    matrix before encoding, and reconstructed V must stay shape (N, d_head).

    For K ∈ ℝ^{N×d_head} with N < d_head, ``full_matrices=False`` would
    return Vt ∈ ℝ^{N×d_head}, giving a (d_head × N) rotation that collapses
    the d_head dimension down to N — that destroys information.
    """
    _, d_head = k.shape
    if d_head < 8:
        return None
    try:
        _, _, Vt = torch.linalg.svd(k.float(), full_matrices=True)
        # Vt is (d_head, d_head) — perfect orthogonal rotation
        return Vt.T.to(k.dtype)
    except Exception:
        _logger.warning("SVD for V rotation failed — falling back to identity")
        return None


def _pad_for_tile_inversion(
    mat: torch.Tensor,
    tile_size: int,
) -> torch.Tensor:
    """Zero-pad mat so its token count is divisible by tile_size.

    Used when applying inverse DCT/DWT/Hybrid to raw buffer tail
    (which typically has < tile_size rows).  Padding guarantees
    tile-aligned reshape for per-tile inverse transform.
    """
    N, d = mat.shape
    if N % tile_size == 0:
        return mat
    pad = tile_size - (N % tile_size)
    return F.pad(mat, (0, 0, 0, pad), mode='constant', value=0.0)


def _reshape_for_cross_token(
    mat: torch.Tensor,
    group: int,
) -> Tuple[torch.Tensor, int]:
    """Reshape (N, d) → (N//G, G*d) for cross-token joint encoding.
    
    Returns (reshaped, pad_tokens).  pad_tokens=0 if N divisible by G.
    """
    if group <= 1:
        return mat, 0
    N, d = mat.shape
    pad = (group - (N % group)) % group
    if pad > 0:
        mat = F.pad(mat, (0, 0, 0, pad), mode='constant', value=0.0)
        N += pad
    return mat.reshape(N // group, group * d), pad


def encode_kv_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    cfg: DSKVCacheConfig,
    svd_shaper: Optional[dict] = None,
    protected: bool = False,
) -> Tuple[DSKVCacheStore, DSKVCacheStore]:
    """Bulk-encode K/V matrices (used by eval scripts).
    
    Parameters
    ----------
    protected:
        If True, store K/V raw at FP16 with zero encoding loss.
        Used for critical layers (first/last) where quantization error
        propagates disproportionately through the transformer stack.
    """
    assert k.ndim == 2 and v.ndim == 2
    assert k.shape == v.shape
    n_tokens_original, d_head = k.shape

    # ── Protected mode: store raw FP16, skip all encoding ───────────
    if protected:
        k_store = DSKVCacheStore(
            tile_size=cfg.tile_size,
            protected=True,
            raw_buffer=k.to(torch.float16),
            buffer_full=k.shape[0],
            orig_shape=k.shape,
            cross_token_group=1,
        )
        v_store = DSKVCacheStore(
            tile_size=cfg.tile_size,
            protected=True,
            raw_buffer=v.to(torch.float16),
            buffer_full=v.shape[0],
            orig_shape=v.shape,
            cross_token_group=1,
        )
        k_store.update_stats()
        v_store.update_stats()
        return k_store, v_store

    # ── V orthogonal transform: apply BEFORE cross-token reshape ──
    v_rotation = None
    if cfg.v_orthogonal_transform:
        v_rotation = _build_v_rotation(k)
    v_rotated = v @ v_rotation if v_rotation is not None else v

    # ── Cross-token joint encoding: K and V use different groups ──
    # §8.1.5: K has fewer steps (4) → can't afford row-resolution loss
    # from grouping, so cap at 2.  V (8 steps) can use the full group.
    group_v = max(1, cfg.cross_token_group)
    group_k = min(2, group_v)  # K gets at most 2-token grouping
    k_enc, k_pad = _reshape_for_cross_token(k, group_k)
    v_enc, v_pad = _reshape_for_cross_token(v_rotated, group_v)

    n_steps_k = cfg.get_n_steps_k()
    n_steps_v = cfg.get_n_steps_v()
    transform_mode = getattr(cfg, 'transform_mode', 'none')

    proj_matrix = None
    if cfg.use_noise_shaping and cfg.proj_rank > 0 and cfg.proj_beta > 0:
        if svd_shaper is not None:
            proj_matrix = svd_shaper.get("projector", None)
        else:
            from modules.svd_noise_shaping import compute_per_head_nullspace_projectors
            projectors = compute_per_head_nullspace_projectors(k.unsqueeze(0), energy_ratio=0.95)
            proj_matrix = projectors[0][0] if 0 in projectors else None

    k_result = _encode_single_path(k_enc, n_steps_k, cfg, proj_matrix)
    v_result = _encode_single_path(v_enc, n_steps_v, cfg, proj_matrix)
    k_bases, k_alphas, k_shape, k_xform_decisions, k_mask_decisions, k_pad_rows = k_result
    v_bases, v_alphas, v_shape, v_xform_decisions, v_mask_decisions, v_pad_rows = v_result

    # ── Two-stage residual differential ───────────────────────────────
    k_bases_res, k_alphas_res, k_shape_res = None, None, None
    v_bases_res, v_alphas_res, v_shape_res = None, None, None

    if cfg.use_differential and cfg.diff_strategy == "residual":
        # Compute residual in TRANSFORM domain to match primary tile layout.
        # The primary bases were encoded from a transform-domain matrix
        # (mat_enc after apply_transform in _encode_single_path), so the
        # residual must be computed in the same domain to produce matching
        # tile counts during reconstruction.
        k_hat_primary = decode_from_bases(k_bases, k_alphas, k_shape, tile_size=cfg.tile_size,
                                          use_fwht=cfg.use_fwht)
        if transform_mode and transform_mode not in ("none", "", None, "fwht"):
            from rina.utils.transforms import apply_transform, apply_inverse_transform, TransformMode
            try:
                tf_mode = TransformMode[transform_mode.upper()]
            except KeyError:
                tf_mode = TransformMode(transform_mode)
            tile_d = cfg.tile_size ** 2
            N_k, d_k = k_enc.shape
            total_elems = N_k * d_k
            if total_elems % tile_d != 0:
                needed_elems = ((total_elems + tile_d - 1) // tile_d) * tile_d
                pad_elems = needed_elems - total_elems
                pad_rows = (pad_elems + d_k - 1) // d_k
                k_enc_padded = F.pad(k_enc, (0, 0, 0, pad_rows), mode='constant', value=0.0)
            else:
                k_enc_padded = k_enc
            k_enc_transformed, _ = apply_transform(
                k_enc_padded, mode=tf_mode, tile_size=cfg.tile_size,
            )
            k_residual = k_enc_transformed - k_hat_primary
        else:
            k_residual = k_enc - k_hat_primary
        k_bases_res, k_alphas_res, k_shape_res, _ = encode_matrix(
            k_residual, n_steps=cfg.diff_residual_n_steps, tile_size=cfg.tile_size,
            beta=cfg.beta, proj_matrix=None, proj_beta=0.0, adaptive_eta=False,
        )

        v_hat_primary = decode_from_bases(v_bases, v_alphas, v_shape, tile_size=cfg.tile_size,
                                          use_fwht=cfg.use_fwht)
        if transform_mode and transform_mode not in ("none", "", None, "fwht"):
            from rina.utils.transforms import apply_transform, TransformMode
            try:
                tf_mode = TransformMode[transform_mode.upper()]
            except KeyError:
                tf_mode = TransformMode(transform_mode)
            tile_d = cfg.tile_size ** 2
            N_v, d_v = v_enc.shape
            total_elems = N_v * d_v
            if total_elems % tile_d != 0:
                needed_elems = ((total_elems + tile_d - 1) // tile_d) * tile_d
                pad_elems = needed_elems - total_elems
                pad_rows = (pad_elems + d_v - 1) // d_v
                v_enc_padded = F.pad(v_enc, (0, 0, 0, pad_rows), mode='constant', value=0.0)
            else:
                v_enc_padded = v_enc
            v_enc_transformed, _ = apply_transform(
                v_enc_padded, mode=tf_mode, tile_size=cfg.tile_size,
            )
            v_residual = v_enc_transformed - v_hat_primary
        else:
            v_residual = v_enc - v_hat_primary
        v_bases_res, v_alphas_res, v_shape_res, _ = encode_matrix(
            v_residual, n_steps=cfg.diff_residual_n_steps, tile_size=cfg.tile_size,
            beta=cfg.beta, proj_matrix=None, proj_beta=0.0, adaptive_eta=False,
        )

    k_bases_M = k_bases.shape[-1]
    v_bases_M = v_bases.shape[-1]

    transform_mode = getattr(cfg, 'transform_mode', 'none')
    k_store = DSKVCacheStore(
        tile_size=cfg.tile_size,
        bases=pack_bases(k_bases),
        bases_shape_M=k_bases_M,
        alphas=k_alphas.to(torch.float16),
        orig_shape=k_shape,
        svd_shaper=svd_shaper,
        bases_residual=pack_bases(k_bases_res) if k_bases_res is not None else None,
        bases_shape_M_residual=k_bases_res.shape[-1] if k_bases_res is not None else None,
        alphas_residual=k_alphas_res.to(torch.float16) if k_alphas_res is not None else None,
        diff_gamma=cfg.get_diff_residual_gamma_k() if cfg.use_differential else 0.0,
        cross_token_group=group_k,
        original_n_tokens=n_tokens_original,
        use_fwht=cfg.use_fwht if transform_mode in ("none", "", None, "fwht") else False,
        transform_mode=transform_mode if transform_mode else "none",
        transform_decisions=k_xform_decisions if k_xform_decisions is not None else None,
        masking_decisions=k_mask_decisions if k_mask_decisions is not None else None,
        transform_pad_rows=k_pad_rows,
        _encode_segments=[(
            0,
            k_bases.shape[1],
            k_shape[0],
            n_tokens_original,
            k_pad,
            k_pad_rows,
        )],
    )
    # Store pad tokens for unreshape
    k_store._cross_token_pad = k_pad  # type: ignore
    k_store._original_mat_shape = (n_tokens_original, d_head)

    v_store = DSKVCacheStore(
        tile_size=cfg.tile_size,
        bases=pack_bases(v_bases),
        bases_shape_M=v_bases_M,
        alphas=v_alphas.to(torch.float16),
        orig_shape=v_shape,
        svd_shaper=svd_shaper,
        bases_residual=pack_bases(v_bases_res) if v_bases_res is not None else None,
        bases_shape_M_residual=v_bases_res.shape[-1] if v_bases_res is not None else None,
        alphas_residual=v_alphas_res.to(torch.float16) if v_alphas_res is not None else None,
        diff_gamma=cfg.diff_residual_gamma if cfg.use_differential else 0.0,
        v_rotation_matrix=v_rotation,
        cross_token_group=group_v,
        original_n_tokens=n_tokens_original,
        use_fwht=cfg.use_fwht if transform_mode in ("none", "", None, "fwht") else False,
        transform_mode=transform_mode if transform_mode else "none",
        transform_decisions=v_xform_decisions if v_xform_decisions is not None else None,
        masking_decisions=v_mask_decisions if v_mask_decisions is not None else None,
        transform_pad_rows=v_pad_rows,
        _encode_segments=[(
            0,
            v_bases.shape[1],
            v_shape[0],
            n_tokens_original,
            v_pad,
            v_pad_rows,
        )],
    )
    v_store._cross_token_pad = v_pad  # type: ignore
    v_store._original_mat_shape = (n_tokens_original, d_head)

    # ── Weighted reconstruction (§8.1.7): compute per-step weights from alphas ──
    if cfg.use_recon_weights:
        k_store.compute_recon_weights(temperature=cfg.recon_weight_temperature)
        v_store.compute_recon_weights(temperature=cfg.recon_weight_temperature)

    k_store.update_stats()
    v_store.update_stats()

    if cfg.verbose:
        _log_diagnostics("K", k, k_store, cfg)
        _log_diagnostics("V", v, v_store, cfg)

    return k_store, v_store


# ══════════════════════════════════════════════════════════════════════════════
# Legacy decode (for eval scripts)
# ══════════════════════════════════════════════════════════════════════════════


def decode_kvcache_store(
    store: DSKVCacheStore,
    tile_size: int = 16,
    use_differential: bool = True,
) -> torch.Tensor:
    """Legacy decode path — delegates to reconstruct_all()."""
    return store.reconstruct_all(tile_size=tile_size, use_differential=use_differential)


# ══════════════════════════════════════════════════════════════════════════════
# Diagnostics
# ══════════════════════════════════════════════════════════════════════════════


def _log_diagnostics(
    tag: str,
    original: torch.Tensor,
    store: DSKVCacheStore,
    cfg: DSKVCacheConfig,
):
    approx = store.reconstruct_all(cfg.tile_size, cfg.use_differential)

    mse = F.mse_loss(approx.float(), original.float()).item()
    signal_power = (original.float() ** 2).mean().item()
    noise_power = ((original.float() - approx.float()) ** 2).mean().item()
    snr_db = 10 * math.log10(signal_power / (noise_power + 1e-12))

    cos_sim = F.cosine_similarity(
        approx.float().flatten().unsqueeze(0),
        original.float().flatten().unsqueeze(0),
    ).item()

    original_bytes = original.element_size() * original.numel()
    comp_ratio = original_bytes / (store.memory_bytes + 1e-12)

    _logger.info(
        f"[DS-KVCache {tag}] tokens={store.n_tokens}, "
        f"tiles={store.n_tiles}, "
        f"bases={store.bases.shape[0] if store.bases is not None else 0} steps, "
        f"MSE={mse:.6f}, SNR={snr_db:.2f}dB, "
        f"CosSim={cos_sim:.6f}, "
        f"CompressRatio={comp_ratio:.1f}x ({original_bytes}→{store.memory_bytes} bytes)"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Incremental store helpers (moved from incremental_decode.py)
# ══════════════════════════════════════════════════════════════════════════════


def _resolve_dtype(dtype_str: str) -> torch.dtype:
    _map = {
        "fp16": torch.float16, "float16": torch.float16,
        "fp32": torch.float32, "float32": torch.float32,
        "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
    }
    key = dtype_str.lower().strip()
    if key in _map:
        return _map[key]
    try:
        return getattr(torch, dtype_str)
    except AttributeError:
        return torch.float16


def init_incremental_store(
    d_head: int,
    cfg: DSKVCacheConfig,
) -> DSKVCacheStore:
    """Create an empty DSKVCacheStore with pre-allocated raw buffer."""
    dtype = _resolve_dtype(cfg.base_dtype)
    buffer = torch.zeros(cfg.incremental_buffer_size, d_head, dtype=dtype)
    return DSKVCacheStore(
        tile_size=cfg.tile_size,
        raw_buffer=buffer,
        buffer_full=0,
    )


def incremental_encode_step(
    new_token_vec: torch.Tensor,
    store: DSKVCacheStore,
    cfg: DSKVCacheConfig,
    is_key: bool = True,
) -> DSKVCacheStore:
    """Append one new token vector to the store, encoding batch if full."""
    if store.raw_buffer is None:
        dtype = _resolve_dtype(cfg.base_dtype)
        store.raw_buffer = torch.zeros(
            cfg.incremental_buffer_size, new_token_vec.shape[0], dtype=dtype,
        )
    idx = store.buffer_full
    store.raw_buffer[idx, :] = new_token_vec.to(store.raw_buffer.dtype)
    store.buffer_full += 1
    if store.buffer_full >= cfg.incremental_buffer_size:
        return _flush_incremental_buffer(store, cfg, is_key)
    return store


def incremental_encode_batch(
    new_token_matrix: torch.Tensor,
    store: DSKVCacheStore,
    cfg: DSKVCacheConfig,
    is_key: bool = True,
) -> DSKVCacheStore:
    """Append a batch of new token vectors, handling partial buffer fill."""
    n_new = new_token_matrix.shape[0]
    pos = 0
    while pos < n_new:
        free = store.raw_buffer.shape[0] - store.buffer_full
        chunk = min(free, n_new - pos)
        store.raw_buffer[store.buffer_full : store.buffer_full + chunk, :] = \
            new_token_matrix[pos : pos + chunk].to(store.raw_buffer.dtype)
        store.buffer_full += chunk
        pos += chunk
        if store.buffer_full >= cfg.incremental_buffer_size:
            store = _flush_incremental_buffer(store, cfg, is_key)
    return store


def finalize_store(
    store: DSKVCacheStore,
    cfg: DSKVCacheConfig,
    is_key: bool = True,
) -> DSKVCacheStore:
    """Flush any remaining tokens in the raw buffer and cache reconstruction."""
    if store.buffer_full > 0:
        store = _flush_incremental_buffer(store, cfg, is_key)
    store.full_k_hat = store.reconstruct_all(cfg.tile_size, cfg.use_differential)
    store.update_stats()
    return store


def _flush_incremental_buffer(
    store: DSKVCacheStore,
    cfg: DSKVCacheConfig,
    is_key: bool,
) -> DSKVCacheStore:
    """Encode raw buffer tokens and merge into the existing store.

    Uses the UnifiedEncoder for consistent transform-aware encoding.
    """
    if store.buffer_full == 0:
        return store

    from rina.unified_encoder import UnifiedEncoder
    from rina.encoded_data import EncodedData
    from rina.metadata import Metadata

    new_mat = store.raw_buffer[:store.buffer_full, :].float()
    N_new, d_head = new_mat.shape
    transform_mode = getattr(cfg, 'transform_mode', 'none')

    # ── Determine existing transform state ───────────────────────────────
    old_has_transform = (
        getattr(store, 'transform_mode', 'none') not in ("none", "", None)
    )
    old_xform_decisions = getattr(store, 'transform_decisions', None)

    # ── Build existing EncodedData from store fields ─────────────────────
    existing = None
    if store.bases is not None:
        existing = EncodedData(
            bases=store.bases,
            bases_shape_M=store.bases_shape_M,
            alphas=store.alphas,
            orig_shape=store.orig_shape,
            bases_residual=store.bases_residual,
            bases_shape_M_residual=store.bases_shape_M_residual,
            alphas_residual=store.alphas_residual,
        )

    encoder = UnifiedEncoder(cfg, tile_size=store.tile_size)
    n_steps = cfg.get_n_steps_k() if is_key else cfg.get_n_steps_v()

    # ── Encode via temporary buffer ──────────────────────────────────────
    from rina.encode_buffer import EncodeBuffer
    temp_buf = EncodeBuffer(data=store.raw_buffer[:store.buffer_full].clone(), buffer_full=store.buffer_full)

    merged_encoded, xform_info = encoder.encode_buffer_and_merge(
        temp_buf, existing,
        n_steps=n_steps, is_key=is_key,
        svd_shaper=store.svd_shaper,
        existing_has_transform=old_has_transform,
        existing_xform_decisions=old_xform_decisions,
    )

    # ── Write back to store ──────────────────────────────────────────────
    store.bases = merged_encoded.bases
    store.bases_shape_M = merged_encoded.bases_shape_M
    store.alphas = merged_encoded.alphas
    store.orig_shape = merged_encoded.orig_shape
    store.bases_residual = merged_encoded.bases_residual
    store.bases_shape_M_residual = merged_encoded.bases_shape_M_residual
    store.alphas_residual = merged_encoded.alphas_residual

    if xform_info.get("transform_decisions") is not None:
        store.transform_decisions = xform_info["transform_decisions"]
        store.transform_mode = transform_mode
        store.use_fwht = False
    else:
        store.transform_decisions = None
        store.transform_mode = "none"
        store.use_fwht = (transform_mode == "fwht")

    # Differential residual becomes stale after re-encode
    if cfg.use_differential:
        store.bases_residual = None
        store.alphas_residual = None

    store.raw_buffer.zero_()
    store.buffer_full = 0
    store.full_k_hat = None
    store.update_stats()
    return store