"""
§7 UnifiedEncoder — single encode/decode interface for bulk and incremental paths.

Merges encode_kv_cache from ds_kv_cache.py and _flush_buffer/_concat_stores
from incremental_decode.py into one consistent API.  Replaces the dual-path
complexity with a single encoder that handles both prefill (bulk) and
decode-loop (incremental) encoding using the same code path.

Design:
    UnifiedEncoder knows how to encode a matrix into EncodedData using the
    full RINA pipeline (transform → Σ-Δ RBP → bit-packing → residual).
    It also handles incremental accumulation via EncodeBuffer and merging
    of new and existing encoded data.
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from rina.config import DSKVCacheConfig
from rina.encoded_data import EncodedData
from rina.metadata import Metadata
from rina.encode_buffer import EncodeBuffer
from rina.utils.bit_packing import pack_bases, unpack_bases
from rina.utils.transform_pipeline import (
    TransformContext,
    TransformPipeline,
    resolve_transform_mode,
)

from modules.residual_pursuit import (
    ResidualBases,
    ResidualAlphas,
    encode_matrix,
    decode_from_bases,
)

_logger = logging.getLogger(__name__)


class UnifiedEncoder:
    """Single encoder handling both prefill (bulk) and decode-loop (incremental).

    Parameters
    ----------
    cfg: DSKVCacheConfig
        Pipeline configuration (tile_size, n_steps, transform_mode, etc.).
    tile_size: int (default from cfg)
        Override tile size (for compatibility with per-layer configs).
    """

    def __init__(self, cfg: DSKVCacheConfig, tile_size: Optional[int] = None):
        self.cfg = cfg
        self.tile_size = tile_size or cfg.tile_size
        self._pipeline = TransformPipeline(tile_size=self.tile_size)

    # ══════════════════════════════════════════════════════════════════════
    # Bulk encode
    # ══════════════════════════════════════════════════════════════════════

    def encode_matrix(
        self,
        mat: torch.Tensor,
        n_steps: int,
        *,
        proj_matrix: Optional[torch.Tensor] = None,
        initial_momentum: Optional[torch.Tensor] = None,
        initial_integrator2: Optional[torch.Tensor] = None,
        return_momentum: bool = False,
        is_v: bool = False,
    ) -> dict:
        """Encode a single matrix path returning a dict of results.

        Parameters
        ----------
        mat: ``(N, d_head)`` — matrix to encode.
        n_steps: Number of Σ-Δ steps.
        proj_matrix: Optional noise-shaping projector.
        initial_momentum/integrator2: Cross-head Σ-Δ state.
        return_momentum: If True, return final Σ-Δ state.
        is_v: True for V path (affects residual gamma).

        Returns
        -------
        dict with keys: bases, alphas, orig_shape, and optionally
        momentum, integrator2, transform_decisions, masking_decisions,
        transform_pad_rows, encoded_data.
        """
        tile_size = self.tile_size
        cfg = self.cfg

        # ── Pad for transform alignment ──────────────────────────────────
        N_orig, d_head_orig = mat.shape
        transform_pad_rows = 0
        transform_mode = getattr(cfg, 'transform_mode', 'none')
        if transform_mode and transform_mode not in ("none", "", None, "fwht"):
            tile_d = tile_size * tile_size
            total_elems = N_orig * d_head_orig
            if total_elems % tile_d != 0:
                needed = ((total_elems + tile_d - 1) // tile_d) * tile_d
                pad_elems = needed - total_elems
                transform_pad_rows = (pad_elems + d_head_orig - 1) // d_head_orig
                mat = F.pad(mat, (0, 0, 0, transform_pad_rows), mode='constant', value=0.0)

        # ── Resolve transform pipeline ───────────────────────────────────
        tf_mode = resolve_transform_mode(transform_mode)
        tf_ctx = TransformContext(mode=tf_mode, tile_size=tile_size)

        # ── Forward transform ────────────────────────────────────────────
        transform_decisions = None
        mat_enc = mat
        if tf_ctx.is_active:
            mat_enc, tf_ctx = self._pipeline.forward(
                mat_enc, tf_ctx,
                smooth_threshold=getattr(cfg, 'transform_smooth_threshold', 0.05),
                outlier_threshold=getattr(cfg, 'transform_outlier_threshold', 3.0),
            )
            transform_decisions = tf_ctx.decisions

        # ── Adaptive masking (Roadmap 1) ─────────────────────────────────
        adaptive_masking = getattr(cfg, 'adaptive_masking', False)
        mask_decisions = None
        if adaptive_masking:
            from rina.utils.transforms import compute_tile_diagnostics
            tiled_diag = mat_enc.reshape(-1, tile_size, tile_size)
            variances, max_abs_vals = compute_tile_diagnostics(tiled_diag)
            stds = variances.sqrt().clamp_min(1e-8)
            outlier_thr = getattr(cfg, 'mask_outlier_threshold', 3.0)
            mask_decisions = (max_abs_vals > outlier_thr * stds).tolist()

        # ── Build encode kwargs ──────────────────────────────────────────
        per_tile_proj = proj_matrix
        if proj_matrix is not None and proj_matrix.shape[-1] != tile_size * tile_size:
            per_tile_proj = None

        encode_kwargs = dict(
            tile_size=tile_size,
            beta=cfg.beta,
            encode_mode=cfg.encode_mode,
            proj_matrix=per_tile_proj,
            proj_beta=cfg.proj_beta if per_tile_proj is not None else 0.0,
            adaptive_eta=cfg.adaptive_eta,
            order2_gamma=cfg.order2_gamma,
            order2_c1=cfg.order2_c1,
            order2_c2=cfg.order2_c2,
            initial_momentum=initial_momentum,
            initial_integrator2=initial_integrator2,
            return_momentum=return_momentum,
            use_fwht=cfg.use_fwht if not tf_ctx.is_active else False,
            zero_mean_integrator2=cfg.zero_mean_integrator2,
        )

        if adaptive_masking:
            encode_kwargs['adaptive_masking'] = True
            encode_kwargs['mask_smooth_threshold'] = getattr(cfg, 'mask_smooth_threshold', 0.05)
            encode_kwargs['mask_outlier_threshold'] = getattr(cfg, 'mask_outlier_threshold', 3.0)
            encode_kwargs['mask_proj_beta_boost'] = getattr(cfg, 'mask_proj_beta_boost', 0.5)
            encode_kwargs['mask_n_steps_boost'] = getattr(cfg, 'mask_n_steps_boost', 1)

        # ── Encode ───────────────────────────────────────────────────────
        if cfg.adaptive_n:
            from modules.residual_pursuit import adaptive_encode_matrix
            n_extra = max(cfg.n_upper_bound - n_steps, 2)
            adaptive_kwargs = {
                k: v for k, v in encode_kwargs.items()
                if k not in ("initial_momentum", "initial_integrator2", "return_momentum",
                             "use_fwht", "zero_mean_integrator2")
            }
            result = adaptive_encode_matrix(
                mat_enc, n_steps_base=n_steps, n_steps_extra=n_extra,
                energy_threshold_ratio=cfg.energy_threshold_factor,
                **adaptive_kwargs,
            )
        else:
            result = encode_matrix(mat_enc, n_steps=n_steps, **encode_kwargs)

        # ── Unpack result ────────────────────────────────────────────────
        if return_momentum:
            bases, alphas, orig_shape, _inner_xform, momentum, integrator2 = result
        else:
            bases, alphas, orig_shape, _inner_xform = result
            momentum, integrator2 = None, None

        output = {
            "bases": bases,
            "alphas": alphas,
            "orig_shape": orig_shape,
            "transform_decisions": transform_decisions,
            "masking_decisions": mask_decisions,
            "transform_pad_rows": transform_pad_rows,
        }
        if return_momentum:
            output["momentum"] = momentum
            output["integrator2"] = integrator2

        return output

    # ══════════════════════════════════════════════════════════════════════
    # Incremental encode
    # ══════════════════════════════════════════════════════════════════════

    def encode_incremental_tile(
        self,
        tile: torch.Tensor,
        *,
        n_steps: int,
        is_v: bool = False,
        proj_matrix: Optional[torch.Tensor] = None,
        initial_momentum: Optional[torch.Tensor] = None,
        initial_integrator2: Optional[torch.Tensor] = None,
    ) -> dict:
        """Encode a single full-tile matrix and return packed result.

        Used by the incremental path (_encode_and_append_tile).
        Returns dict with keys: bases_packed, alphas, orig_shape,
        bases_shape_M, bases_residual_packed, alphas_residual,
        bases_shape_M_residual, transform_decisions, masking_decisions,
        momentum, integrator2.
        """
        do_cross_head = (
            self.cfg.cross_head_error_share
            and self.cfg.order2_gamma > 0
        )

        if do_cross_head:
            if initial_momentum is None:
                device = tile.device
                dtype = tile.dtype
                initial_momentum = torch.zeros(1, self.tile_size * self.tile_size, device=device, dtype=dtype)
                initial_integrator2 = torch.zeros(1, self.tile_size * self.tile_size, device=device, dtype=dtype) if self.cfg.order2_gamma > 0 else None

        result = self.encode_matrix(
            tile, n_steps=n_steps,
            proj_matrix=proj_matrix,
            initial_momentum=initial_momentum if do_cross_head else None,
            initial_integrator2=initial_integrator2 if do_cross_head else None,
            return_momentum=do_cross_head,
            is_v=is_v,
        )

        bases = result["bases"]
        alphas = result["alphas"]
        shape = result["orig_shape"]

        bases_M = bases.shape[-1]
        packed = pack_bases(bases)

        # ── Two-stage residual ───────────────────────────────────────────
        bases_res, alphas_res = None, None
        bases_shape_M_res = None
        if self.cfg.use_differential and self.cfg.diff_strategy == "residual":
            primary = decode_from_bases(bases, alphas, shape, tile_size=self.tile_size)
            residual = tile - primary
            bases_res, alphas_res, _res_shape, _ = encode_matrix(
                residual, n_steps=self.cfg.diff_residual_n_steps,
                tile_size=self.tile_size, beta=self.cfg.beta,
                proj_matrix=None, proj_beta=0.0, adaptive_eta=False,
            )
            bases_shape_M_res = bases_res.shape[-1]
            bases_res = pack_bases(bases_res)
            alphas_res = alphas_res.to(torch.float16)

        alphas = alphas.to(torch.float16)

        diff_gamma = self.cfg.get_diff_residual_gamma_k() if not is_v else self.cfg.diff_residual_gamma

        return {
            "bases_packed": packed,
            "alphas": alphas,
            "orig_shape": shape,
            "bases_shape_M": bases_M,
            "bases_residual_packed": bases_res,
            "alphas_residual": alphas_res,
            "bases_shape_M_residual": bases_shape_M_res,
            "diff_gamma": diff_gamma,
            "transform_decisions": result.get("transform_decisions"),
            "masking_decisions": result.get("masking_decisions"),
            "momentum": result.get("momentum"),
            "integrator2": result.get("integrator2"),
        }

    # ══════════════════════════════════════════════════════════════════════
    # Flush and merge (replaces _flush_buffer + _concat_stores)
    # ══════════════════════════════════════════════════════════════════════

    def encode_buffer_and_merge(
        self,
        buffer: EncodeBuffer,
        existing: Optional[EncodedData] = None,
        *,
        n_steps: int,
        is_key: bool = True,
        svd_shaper: Optional[dict] = None,
        existing_has_transform: bool = False,
        existing_xform_decisions: Optional[List[str]] = None,
    ) -> Tuple[EncodedData, dict]:
        """Encode buffer contents and merge with existing encoded data.

        Parameters
        ----------
        buffer: Accumulated raw tokens to encode.
        existing: Previously encoded data (None = first encode).
        n_steps: Σ-Δ step count.
        is_key: True for K path.
        svd_shaper: Per-head noise shaper for re-encoding.
        existing_has_transform: Whether existing data is in transform domain.
        existing_xform_decisions: Existing per-tile transform decisions.

        Returns
        -------
        (merged_encoded_data, extra_info_dict).
        """
        if buffer.is_empty():
            return existing, {}

        new_mat = buffer.peek_all()
        N_new, d_head = new_mat.shape
        transform_mode = getattr(self.cfg, 'transform_mode', 'none')

        # ── Forward-transform buffer if active ───────────────────────────
        new_transform_decisions = None
        mat_enc = new_mat
        if transform_mode and transform_mode not in ("none", "", None, "fwht"):
            tf_mode = resolve_transform_mode(transform_mode)
            tf_ctx = TransformContext(mode=tf_mode, tile_size=self.tile_size)
            mat_enc, tf_ctx = self._pipeline.forward(
                mat_enc, tf_ctx,
                smooth_threshold=getattr(self.cfg, 'transform_smooth_threshold', 0.05),
                outlier_threshold=getattr(self.cfg, 'transform_outlier_threshold', 3.0),
            )
            new_transform_decisions = tf_ctx.decisions

        # ── Encode buffer ────────────────────────────────────────────────
        proj_matrix = svd_shaper.get("projector") if svd_shaper else None
        result = self.encode_matrix(mat_enc, n_steps=n_steps, proj_matrix=proj_matrix)
        new_bases = result["bases"]
        new_alphas = result["alphas"]
        new_shape = result["orig_shape"]

        if existing is not None and existing.bases is not None:
            # ── Merge with existing ──────────────────────────────────────
            old_bases = unpack_bases(existing.bases)
            if existing.bases_shape_M is not None and old_bases.shape[-1] > existing.bases_shape_M:
                old_bases = old_bases[..., :existing.bases_shape_M]

            merged_bases, merged_alphas, merged_shape, merged_xform = _merge_encoded_stores(
                old_bases, existing.alphas, existing.orig_shape,
                new_bases, new_alphas, new_shape,
                self.tile_size, self.cfg, is_key,
                old_has_transform=existing_has_transform,
                old_xform_decisions=existing_xform_decisions,
                new_has_transform=(new_transform_decisions is not None),
                new_xform_decisions=new_transform_decisions,
                d_head=d_head,
            )

            merged = EncodedData(
                bases=pack_bases(merged_bases),
                bases_shape_M=merged_bases.shape[-1],
                alphas=merged_alphas,
                orig_shape=merged_shape,
            )
            merged_xform_info = {"transform_decisions": merged_xform}

            # Clear buffer after successful encode+merge
            buffer.reset()
            return merged, merged_xform_info
        else:
            # ── First encode (no existing data) ──────────────────────────
            new_bases_M = new_bases.shape[-1]
            encoded = EncodedData(
                bases=pack_bases(new_bases),
                bases_shape_M=new_bases_M,
                alphas=new_alphas,
                orig_shape=new_shape,
            )
            buffer.reset()
            return encoded, {"transform_decisions": new_transform_decisions}

    # ══════════════════════════════════════════════════════════════════════
    # Decode
    # ══════════════════════════════════════════════════════════════════════

    def decode(
        self,
        encoded: EncodedData,
        meta: Metadata,
        buffer: Optional[EncodeBuffer] = None,
    ) -> torch.Tensor:
        """Decode EncodedData back to (N_orig, d_head) matrix.

        Parameters
        ----------
        encoded: Bit-packed encoded data.
        meta: Metadata with transform state, cross-token params, etc.
        buffer: Optional raw buffer tail (unencoded tokens).

        Returns
        -------
        ``(N_total, d_head)`` reconstructed matrix.
        """
        decoded_parts = []

        if encoded.bases is not None:
            bases = unpack_bases(encoded.bases)
            if encoded.bases_shape_M is not None and bases.shape[-1] > encoded.bases_shape_M:
                bases = bases[..., :encoded.bases_shape_M]

            mat = decode_from_bases(
                bases, encoded.alphas, encoded.orig_shape, tile_size=self.tile_size,
                recon_weights=meta.recon_weights,
                use_fwht=meta.use_fwht,
            )

            # ── Differential residual ────────────────────────────────────
            if encoded.has_residual and meta.diff_gamma > 0:
                bases_res = unpack_bases(encoded.bases_residual)
                if encoded.bases_shape_M_residual is not None and bases_res.shape[-1] > encoded.bases_shape_M_residual:
                    bases_res = bases_res[..., :encoded.bases_shape_M_residual]
                mat_res = decode_from_bases(
                    bases_res, encoded.alphas_residual, encoded.orig_shape,
                    tile_size=self.tile_size, use_fwht=meta.use_fwht,
                )
                mat = mat + meta.diff_gamma * mat_res

            # ── Inverse transform ────────────────────────────────────────
            if meta.has_transform() and meta.transform_context is not None:
                mat = self._pipeline.inverse(mat, meta.transform_context)

            # ── Cross-token unreshape ────────────────────────────────────
            if meta.cross_token_group > 1 and meta.original_n_tokens > 0:
                from rina.utils.tile_ops import unreshape_cross_token
                mat = unreshape_cross_token(
                    mat, meta.cross_token_group,
                    meta.original_n_tokens, meta.cross_token_pad,
                )

            # ── Strip dynamic tile pad rows ──────────────────────────────
            if meta.tile_pad_counts is not None:
                total_pad = sum(meta.tile_pad_counts)
                if total_pad > 0 and mat.shape[0] >= len(meta.tile_pad_counts) * self.tile_size:
                    n_full_tiles = len(meta.tile_pad_counts)
                    mat_tiles = mat[:n_full_tiles * self.tile_size].reshape(
                        n_full_tiles, self.tile_size, -1,
                    )
                    keep_chunks = []
                    for i in range(n_full_tiles):
                        keep = self.tile_size - meta.tile_pad_counts[i]
                        if keep > 0:
                            keep_chunks.append(mat_tiles[i, :keep])
                    if keep_chunks:
                        mat = torch.cat(keep_chunks, dim=0)
                    else:
                        mat = mat[:0]

            # ── V un-rotation ────────────────────────────────────────────
            if meta.v_rotation_matrix is not None:
                R_T = meta.v_rotation_matrix.T.to(mat.dtype)
                if R_T.shape[-1] == mat.shape[-1]:
                    mat = mat @ R_T

            decoded_parts.append(mat)

        # ── Append raw buffer tail ───────────────────────────────────────
        if buffer is not None and not buffer.is_empty():
            tail = buffer.peek_all()
            if meta.has_transform() and meta.transform_context is not None:
                from rina.utils.tile_ops import pad_rows_to_tile_multiple
                tail_padded, _pad = pad_rows_to_tile_multiple(tail, self.tile_size)
                tail_tf_ctx = TransformContext(mode=meta.transform_context.mode, tile_size=self.tile_size)
                tail_transformed, _ = self._pipeline.forward(tail_padded, tail_tf_ctx)
                tail = self._pipeline.inverse(tail_transformed, tail_tf_ctx)
                tail = tail[:tail.shape[0]]
            decoded_parts.append(tail)

        if not decoded_parts:
            return torch.empty(0, 0)

        result = torch.cat(decoded_parts, dim=0)

        # ── Restore original shape ───────────────────────────────────────
        if meta.transform_context is not None and meta.transform_context.original_mat_shape is not None:
            N_orig, d_orig = meta.transform_context.original_mat_shape
            if result.numel() >= N_orig * d_orig and result.shape != (N_orig, d_orig):
                result = result.flatten()[:N_orig * d_orig].reshape(N_orig, d_orig)

        return result


# ══════════════════════════════════════════════════════════════════════════════
# Internal merge helper (replaces _concat_stores from incremental_decode.py)
# ══════════════════════════════════════════════════════════════════════════════


def _merge_encoded_stores(
    old_bases, old_alphas, old_shape,
    new_bases, new_alphas, new_shape,
    tile_size: int,
    cfg: DSKVCacheConfig,
    is_key: bool,
    old_has_transform: bool = False,
    old_xform_decisions=None,
    new_has_transform: bool = False,
    new_xform_decisions=None,
    d_head: int = 0,
):
    """Decode, concatenate, and re-encode two tile-sets.

    When either store is in transform domain, applies inverse transform
    before concat and forward transform after.  Merged data is returned
    in the transform domain if transform is active.
    """
    # ── Decode both to dense ─────────────────────────────────────────────
    mat_old = decode_from_bases(old_bases, old_alphas, old_shape, tile_size)
    mat_new = decode_from_bases(new_bases, new_alphas, new_shape, tile_size)

    # ── Inverse-transform if in transform domain ─────────────────────────
    if old_has_transform:
        mat_old = _inverse_transform_tiles(mat_old, cfg, tile_size, old_xform_decisions)
    if new_has_transform:
        mat_new = _inverse_transform_tiles(mat_new, cfg, tile_size, new_xform_decisions)

    # ── Determine d_head ─────────────────────────────────────────────────
    if not d_head:
        d_head_old = old_shape[1] if old_shape[1] != tile_size**2 else 0
        d_head_new = new_shape[1] if new_shape[1] != tile_size**2 else 0
        d_head = d_head_old or d_head_new or tile_size

    if d_head == tile_size**2:
        mat_total = torch.cat([mat_old, mat_new], dim=0)
    else:
        mat_old_2d = mat_old.reshape(-1, d_head).contiguous()
        mat_new_2d = mat_new.reshape(-1, d_head).contiguous()
        mat_total = torch.cat([mat_old_2d, mat_new_2d], dim=0)

    # ── Forward-transform if active ──────────────────────────────────────
    transform_decisions_merged = None
    mat_enc = mat_total
    transform_mode = getattr(cfg, 'transform_mode', 'none')
    if transform_mode and transform_mode not in ("none", "", None, "fwht"):
        from rina.utils.transforms import apply_transform, TransformMode
        if isinstance(transform_mode, str):
            try:
                tf_mode = TransformMode[transform_mode.upper()]
            except KeyError:
                tf_mode = TransformMode(transform_mode)
        else:
            tf_mode = transform_mode
        mat_enc, transform_decisions_merged, _grid_shape = apply_transform(
            mat_enc, mode=tf_mode, tile_size=tile_size,
        )

    # ── Re-encode ────────────────────────────────────────────────────────
    n_steps = cfg.get_n_steps_k() if is_key else cfg.get_n_steps_v()
    proj_matrix = None
    if cfg.use_noise_shaping:
        from modules.residual_pursuit import _build_proj_matrix
        proj_matrix = _build_proj_matrix(mat_enc, tile_size, cfg.proj_rank)

    if cfg.adaptive_n:
        from modules.residual_pursuit import adaptive_encode_matrix
        n_extra = cfg.n_upper_bound - n_steps
        if n_extra <= 0:
            n_extra = 2
        bases, alphas, _, orig_shape = adaptive_encode_matrix(
            mat_enc, n_steps_base=n_steps, n_steps_extra=n_extra,
            tile_size=tile_size, beta=cfg.beta,
            proj_matrix=proj_matrix, proj_beta=cfg.proj_beta,
            adaptive_eta=cfg.adaptive_eta,
            energy_threshold_ratio=cfg.energy_threshold_factor,
            order2_gamma=cfg.order2_gamma, order2_c1=cfg.order2_c1,
            order2_c2=cfg.order2_c2,
        )
    else:
        bases, alphas, orig_shape, _ = encode_matrix(
            mat_enc, n_steps=n_steps, tile_size=tile_size,
            beta=cfg.beta, proj_matrix=proj_matrix, proj_beta=cfg.proj_beta,
            adaptive_eta=cfg.adaptive_eta,
            order2_gamma=cfg.order2_gamma, order2_c1=cfg.order2_c1,
            order2_c2=cfg.order2_c2,
        )

    return bases, alphas, orig_shape, transform_decisions_merged


def _inverse_transform_tiles(
    tiles: torch.Tensor,
    cfg: DSKVCacheConfig,
    tile_size: int,
    decisions=None,
) -> torch.Tensor:
    """Apply inverse transform to tile-space data."""
    from rina.utils.transforms import apply_inverse_transform, TransformMode
    transform_mode = getattr(cfg, 'transform_mode', 'none')
    if isinstance(transform_mode, str):
        try:
            tf_mode = TransformMode[transform_mode.upper()]
        except KeyError:
            tf_mode = TransformMode(transform_mode)
    else:
        tf_mode = transform_mode
    return apply_inverse_transform(
        tiles, mode=tf_mode, tile_size=tile_size,
        decisions=decisions, original_shape=None,
    )


__all__ = ["UnifiedEncoder"]
