"""Stub: DCT/DWT/FWHT transforms removed (experimentally destructive, §8.1.11).
Only ``compute_tile_diagnostics`` and identity no-ops remain for adaptive_masking."""

from __future__ import annotations

from enum import Enum
from typing import Optional, Tuple

import torch


class TransformMode(Enum):
    """Transform mode — only NONE retained after cleanup; others are aliases
    preserving enum compatibility for residual_pursuit.py."""
    AUTO = "auto"
    DCT = "dct"
    DWT = "dwt"
    HYBRID = "hybrid"
    NONE = "none"
    FWHT = "fwht"


def apply_transform(tiles, mode, tile_size, **kwargs):
    """Identity passthrough (transform_mode is always 'none' in balanced)."""
    n_tiles = tiles.shape[0]
    M = tile_size * tile_size
    return tiles.reshape(n_tiles, M), ["none"] * n_tiles


def apply_inverse_transform(transformed, mode, tile_size,
                            decisions=None, original_shape=None):
    """Identity passthrough + optional reshape."""
    if original_shape is not None:
        N_orig, d_head = original_shape
        n_tiles = transformed.shape[0]
        M = tile_size * tile_size
        N_padded = n_tiles * M // d_head
        result = transformed.reshape(N_padded, d_head)
        return result[:N_orig]
    return transformed


def compute_tile_diagnostics(tiles_2d):
    """Per-tile variance / max-abs for adaptive_masking decisions."""
    flat = tiles_2d.reshape(tiles_2d.shape[0], -1) if tiles_2d.dim() == 3 else tiles_2d
    variances = flat.var(dim=-1)
    max_abs_vals = flat.abs().max(dim=-1).values
    return variances, max_abs_vals


# Stub symbols — preserve import compatibility
def dct_2d(x): return x
def idct_2d(x): return x
def dwt_haar_2d(x): return x, x, x, x
def idwt_haar_2d(ll, lh, hl, hh): return ll + lh + hl + hh
