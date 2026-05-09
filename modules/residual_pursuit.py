"""
§4 Σ-Δ Residual Binary Pursuit (Noise-Shaped variant §8.1)
===========================================================

Converts a full-precision weight matrix into N 1-bit bases with L1-optimal
scaling factors.  Each 16×16 tile is encoded independently to bound
element-wise error drift.  Momentum-augmented variant (second-order Σ-Δ)
supported via ``beta``.

Noise-Shaped RBP (§8.1):
    Introduces ``proj_matrix`` (P_signal = UU^T) and ``proj_beta`` to
    push quantization noise into the nullspace (perceptual blind spot).
    Residual update becomes:
        e = W - Ŵ_k
        r_{k+1} = e - proj_beta * (I - P_signal) · e

Adaptive η Scheduling (§8.1.1):
    Instead of applying full ``proj_beta`` from step 0, η ramps linearly
    from 0 to its peak across the first ``eta_peak_step`` iterations.
    This prevents early-step oversuppression of signal components when
    the residual is still dominated by signal energy.

Differential Noise Cancellation (§8.2):
    Encodes the same weight matrix twice with different strategies
    (momentum shift or extra step) and averages the two reconstructions.
    This is the S/W analogue of a differential circuit: two slightly
    different encoding paths whose partially independent quantisation
    errors cancel upon averaging.

Reference: R.I.N.A Whitepaper §4.1, §8.1, §8.1.1, §8.2
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pad_to_tile_multiple(
    w: torch.Tensor, tile_size: int
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Right-/bottom-pad *w* so both dims are multiples of *tile_size*."""
    rows, cols = w.shape[-2], w.shape[-1]
    pad_r = (tile_size - rows % tile_size) % tile_size
    pad_c = (tile_size - cols % tile_size) % tile_size
    if pad_r == 0 and pad_c == 0:
        return w, (0, 0)
    return F.pad(w, (0, pad_c, 0, pad_r)), (pad_r, pad_c)


def _unpad(w: torch.Tensor, orig_shape: Tuple[int, ...]) -> torch.Tensor:
    """Remove padding added by _pad_to_tile_multiple."""
    return w[..., : orig_shape[-2], : orig_shape[-1]]


def _tile_count(shape_2d: Tuple[int, ...], tile_size: int) -> Tuple[int, int]:
    """Number of tiles in (rows, cols) directions."""
    return (
        math.ceil(shape_2d[-2] / tile_size),
        math.ceil(shape_2d[-1] / tile_size),
    )


def _build_proj_matrix(
    w: torch.Tensor,
    tile_size: int,
    proj_rank: int,
) -> torch.Tensor:
    """Build signal-subspace projection matrix P_signal = V @ V^T.

    Runs PCA across all tiles of *w* to find the *proj_rank* principal
    directions in R^{tile_size²}.  Returns ``(M, M)`` where M = tile_size².
    """
    assert w.dim() == 2, "Expected 2-D weight matrix"
    M = tile_size * tile_size

    w_padded, _ = _pad_to_tile_multiple(w, tile_size)
    patches = F.unfold(
        w_padded.unsqueeze(0).unsqueeze(0),
        kernel_size=tile_size,
        stride=tile_size,
    )  # (1, M, n_tiles)
    tiles = patches.squeeze(0).t()  # (n_tiles, M)

    # Centre for PCA
    tiles_centered = tiles - tiles.mean(dim=0, keepdim=True)

    # Randomized SVD for top-k right singular vectors
    k = min(proj_rank, min(tiles_centered.shape) - 1)
    if k < 1:
        return torch.eye(M, device=w.device, dtype=torch.float32)

    _, _, V = torch.pca_lowrank(tiles_centered.float(), q=k)
    # V: (M, k) — principal directions in R^M
    P_signal = V @ V.T  # (M, M)
    return P_signal


# ---------------------------------------------------------------------------
# Core pursuit – works on a *flat* collection of tiles
# ---------------------------------------------------------------------------

ResidualBases = torch.Tensor  # shape (N, *tiles, M) where M = tile_size**2
ResidualAlphas = torch.Tensor  # shape (N, *tiles)

# ---------------------------------------------------------------------------
# Bit-packing utilities (§12 — compressed storage)
# ---------------------------------------------------------------------------
# Canonical implementations live in rina.utils.bit_packing.
# Duplicated here to avoid circular imports (modules → rina → modules).

BITS_PER_PACK = 32  # pack 32 signs per int32 element


def pack_bases(bases: ResidualBases) -> torch.Tensor:
    """Pack float (-1/+1) bases into bit-packed int32 tensor."""
    shape = bases.shape
    M = shape[-1]
    M_packed = (M + BITS_PER_PACK - 1) // BITS_PER_PACK
    pad_len = M_packed * BITS_PER_PACK - M
    if pad_len > 0:
        bases = torch.nn.functional.pad(bases, (0, pad_len), value=1.0)
    bits = (bases > 0).to(torch.uint8)
    bits = bits.reshape(*shape[:-1], M_packed, BITS_PER_PACK)
    bit_weights = 1 << torch.arange(BITS_PER_PACK, device=bases.device, dtype=torch.int32)
    packed = (bits.to(torch.int32) * bit_weights).sum(dim=-1)
    return packed.to(torch.int32)


def unpack_bases(packed: torch.Tensor) -> ResidualBases:
    """Unpack bit-packed int32 tensor back to float (-1/+1) bases."""
    device = packed.device
    shape = packed.shape
    M_packed = shape[-1]
    bit_weights = 1 << torch.arange(BITS_PER_PACK, device=device, dtype=torch.int32)
    packed_expanded = packed.unsqueeze(-1)
    bits = (packed_expanded & bit_weights) != 0
    bases = bits.to(torch.float32) * 2.0 - 1.0
    bases = bases.reshape(*shape[:-1], M_packed * BITS_PER_PACK)
    return bases


def residual_pursuit_nd(
    w_flat: torch.Tensor,
    n_steps: int,
    beta: float = 0.0,
    *,
    return_bases: bool = True,
    proj_matrix: Optional[torch.Tensor] = None,
    proj_beta: float = 0.0,
    sign_flip: float = 1.0,
    adaptive_eta: bool = False,
    eta_peak_step: Optional[int] = None,
    order2_gamma: float = 0.0,
    order2_c1: float = 1.0,
    order2_c2: float = 0.5,
    zero_mean_integrator2: bool = False,
    mask: Optional[torch.Tensor] = None,
    use_mask_gating: bool = True,
    initial_momentum: Optional[torch.Tensor] = None,
    initial_integrator2: Optional[torch.Tensor] = None,
    return_momentum: bool = False,
) -> Tuple[Optional[ResidualBases], ResidualAlphas, torch.Tensor]:
    """Core Σ-Δ Residual Binary Pursuit on ``(..., M)``-shaped tensors.

    Parameters
    ----------
    w_flat:
        Weight values for one or many tiles, shape ``(..., M)``.
        Each row/group of size ``M`` is an independent tile.
    n_steps:
        Number of 1-bit bases (oversampling ratio N).
    beta:
        Momentum coefficient for second-order Σ-Δ.
        ``beta=0`` → first-order.  ``beta∈(0,1)`` → damped momentum.
    return_bases:
        If ``True``, return the sign bases ``B_k`` (memory overhead).
        Set ``False`` for decode-only or compact storage.
    proj_matrix:
        ``(M, M)`` signal-subspace projector P_signal = UU^T.
        ``None`` disables noise shaping (§8.1).
    proj_beta:
        Noise-shaping strength ∈ [0, 1].  0 = off, 1 = nullspace fully
        suppressed each step.
    sign_flip:
        Multiplier for ``sign(target)`` in the 1-bit quantisation step.
        ``+1.0`` (default) → standard RBP.  ``-1.0`` → complementary
        encoding for differential noise cancellation (§8.2).
    adaptive_eta:
        If ``True``, *proj_beta* ramps linearly from 0 to its peak
        across the first *eta_peak_step* iterations and stays constant
        thereafter.  This prevents early-step oversuppression of signal
        components when the residual is still dominated by signal energy.
        (§8.1.1 Adaptive η Scheduling)
    eta_peak_step:
        Step at which η reaches *proj_beta* (inclusive). Defaults to
        ``max(2, n_steps // 2)`` when *adaptive_eta* is True.
    order2_gamma:
        Second-order Σ-Δ coupling strength ∈ [0, 1].
        0 → standard first-order momentum (§4).
        >0 → cascaded second integrator: the second integrator
        accumulates the output of the first, creating a (1−z⁻¹)²
        noise transfer function for stronger low-frequency
        noise suppression (§8.1.2 Second-Order Σ-Δ).
    order2_c1:
        Gain on first integrator output. Default 1.0.
    order2_c2:
        Gain on second integrator output. Default 0.5.
    zero_mean_integrator2:
        If True, subtract per-tile mean from integrator2 after update
        to prevent DC drift in the feedback loop (§8.1.12).
    mask:
        ``(..., M)`` binary validity mask.  1 = valid element (not padding),
        0 = padding.  When provided, alpha is normalised by the number of
        valid elements per tile rather than ``M``, preventing amplitude
        underestimation for partially-filled tiles (§10.3).
    use_mask_gating:
        If ``True`` and *mask* is provided, zero out padding regions in
        ``w_hat``, ``remaining``, ``momentum``, and ``integrator2`` at the
        end of each Σ-Δ iteration.  This prevents encoding bits from being
        wasted on zero-padding and improves reconstruction quality for
        partially-filled tiles (§10.3.1).
    initial_momentum:
        ``(..., M)`` initial first-order integrator state.  Used for
        cross-head error sharing (§8.1.9) where the Σ-Δ error from head i
        seeds head i+1 to distribute quantisation error evenly across
        the KV-head group in GQA models.
    initial_integrator2:
        ``(..., M)`` initial second-order integrator state.
        Companion to *initial_momentum* for second-order Σ-Δ.
    return_momentum:
        If ``True``, return ``(bases, alphas, w_hat, momentum, integrator2)``
        so callers can propagate error state to subsequent heads/tiles.

    Returns
    -------
    bases:
        ``(n_steps, ..., M)`` tensor of ``{-1, +1}`` if *return_bases*,
        else ``None``.
    alphas:
        ``(n_steps, ...)`` L1-optimal scaling per tile.
    w_hat:
        ``(..., M)`` reconstructed tensor, ``Ŵ = Σ α_k B_k``.
    momentum: (only if return_momentum=True)
        ``(..., M)`` final first-order integrator state.
    integrator2: (only if return_momentum=True)
        ``(..., M)`` final second-order integrator state.
    """
    M = w_flat.shape[-1]
    dtype = w_flat.dtype
    device = w_flat.device

    w_hat = torch.zeros_like(w_flat)
    remaining = w_flat.clone()

    # First-order momentum (integrator 1) — accept cross-head seed
    if initial_momentum is not None:
        momentum = initial_momentum.to(device=device, dtype=dtype)
    else:
        momentum = torch.zeros_like(w_flat)
    # Second-order accumulated error (integrator 2)
    if initial_integrator2 is not None:
        integrator2 = initial_integrator2.to(device=device, dtype=dtype)
    else:
        integrator2 = torch.zeros_like(w_flat)

    alphas_list: List[torch.Tensor] = []
    bases_list: List[torch.Tensor] = [] if return_bases else None

    use_noise_shape = proj_matrix is not None and proj_beta > 0.0
    use_order2 = order2_gamma > 0.0
    _proj = proj_matrix.to(device=device, dtype=dtype) if use_noise_shape else None

    # ---- Per-tile valid element count (§10.3) ----
    if mask is not None:
        valid_count = mask.sum(dim=-1)  # (...,)
        # Clip to ≥1 to avoid div-by-zero on fully-padded tiles
        valid_count = valid_count.clamp(min=1)
    else:
        valid_count = None

    # ---- Adaptive η scheduling (§8.1.1) ----
    # Ramp proj_beta linearly from 0 → peak across early steps.
    # Prevents early-step oversuppression when residual is still dominated
    # by signal energy (not yet encoded into bases).
    if adaptive_eta and use_noise_shape:
        _peak_step = eta_peak_step if eta_peak_step is not None else max(2, n_steps // 2)
        _peak_step = max(1, _peak_step)
    else:
        _peak_step = None

    for _step in range(n_steps):
        # ---- adaptive η for this step ----
        step_eta = proj_beta
        if _peak_step is not None:
            if _step <= _peak_step:
                step_eta = proj_beta * (_step / max(_peak_step, 1.0))
            else:
                step_eta = proj_beta

        # ---- second-order Σ-Δ target (§8.1.2) ----
        # Standard first-order: target = remaining + β·momentum
        # Second-order adds: target = remaining + c₁·β·integrator1 + c₂·γ·integrator2
        # integrator2 accumulates integrator1 → (1−z⁻¹)² NTF
        target = remaining + beta * momentum
        if use_order2:
            target = target + order2_gamma * order2_c2 * integrator2

        # ---- L1-optimal alpha (Whitepaper §4.1 step 1) ----
        # §10.3: when mask is provided, normalise by valid element count
        # instead of M to avoid amplitude underestimation in partial tiles.
        if valid_count is not None:
            alpha = target.abs().sum(dim=-1) / valid_count  # (...,)
        else:
            alpha = target.abs().sum(dim=-1) / M  # (...,)

        # ---- 1-bit quantization (step 2) ----
        B = sign_flip * torch.sign(target)  # {-1, +1}

        # ---- contribution (step 3) ----
        contribution = alpha.unsqueeze(-1) * B

        # ---- update reconstruction (step 3) ----
        w_hat = w_hat + contribution
        remaining = w_flat - w_hat

        # ---- store momentum for next step ----
        # First-order error (quantisation error this step)
        momentum = target - contribution

        # ---- second-order integrator update (§8.1.2) ----
        # Integrator2 accumulates the first integrator's output,
        # creating a second pole at DC for stronger low-freq suppression.
        if use_order2:
            integrator2 = order2_c1 * beta * momentum + integrator2
            # §8.1.12 DC-drift suppression: remove per-tile mean
            if zero_mean_integrator2:
                integrator2 = integrator2 - integrator2.mean(dim=-1, keepdim=True)

        # ---- noise-shaping: inject nullspace-suppressed error into momentum (§8.1) ----
        # Δ-Σ error feedback: push the nullspace component of residual into
        # the momentum so the NEXT step's quantiser focuses on signal directions.
        # We modify momentum (not remaining) to preserve the true residual for
        # convergence tracking.
        if use_noise_shape:
            # e_null = (I - P_signal)·remaining
            e_null = remaining - torch.matmul(remaining, _proj.T)
            # momentum ← momentum - step_eta * e_null
            #   = (prev_target - prev_contribution) - step_eta * e_null
            momentum = momentum - step_eta * e_null

        # ---- mask-based gating: zero out padding regions (§10.3.1) ----
        # After all updates for this step, multiply state tensors by mask
        # to prevent Σ-Δ error from accumulating in zero-padded regions.
        # The bases tensor B is deliberately NOT masked (it stays ±1 for
        # bit-packing compatibility; its contribution in padding is zeroed
        # by the reconstructed w_hat state being masked below).
        if mask is not None and use_mask_gating:
            w_hat = w_hat * mask.to(w_hat.dtype)
            remaining = remaining * mask.to(remaining.dtype)
            momentum = momentum * mask.to(momentum.dtype)
            if use_order2:
                integrator2 = integrator2 * mask.to(integrator2.dtype)

        alphas_list.append(alpha)
        if return_bases:
            bases_list.append(B)

    alphas = torch.stack(alphas_list, dim=0)  # (N, ...,)
    bases = torch.stack(bases_list, dim=0) if return_bases else None

    if return_momentum:
        return bases, alphas, w_hat, momentum, integrator2
    return bases, alphas, w_hat


# ---------------------------------------------------------------------------
# Tile-based encoder / decoder – public API
# ---------------------------------------------------------------------------


def encode_matrix(
    w: torch.Tensor,
    n_steps: int = 5,
    tile_size: int = 16,
    beta: float = 0.0,
    *,
    proj_matrix: Optional[torch.Tensor] = None,
    proj_beta: float = 0.0,
    adaptive_eta: bool = False,
    eta_peak_step: Optional[int] = None,
    order2_gamma: float = 0.0,
    order2_c1: float = 1.0,
    order2_c2: float = 0.5,
    zero_mean_integrator2: bool = False,
    use_fwht: bool = False,  # DEPRECATED: use transform_mode="fwht"
    transform_mode: str = "none",
    transform_smooth_threshold: float = 0.05,
    transform_outlier_threshold: float = 3.0,
    adaptive_masking: bool = False,
    mask_smooth_threshold: float = 0.05,
    mask_outlier_threshold: float = 3.0,
    mask_proj_beta_boost: float = 0.5,
    mask_n_steps_boost: int = 1,
    use_mask_gating: bool = True,
    initial_momentum: Optional[torch.Tensor] = None,
    initial_integrator2: Optional[torch.Tensor] = None,
    return_momentum: bool = False,
) -> Tuple:
    """1-bit tile encoder for a 2-D weight matrix.

    Splits ``w`` into ``tile_size×tile_size`` blocks, encodes each
    independently with Residual Binary Pursuit, then reassembles.

    Parameters
    ----------
    w:
        Full-precision matrix, shape ``(rows, cols)``.
    n_steps:
        Number of 1-bit bases (N).
    tile_size:
        Tile dimension (default 16 — GPU Tensor Core native).
    beta:
        Momentum coefficient (0 = first-order, >0 = second-order).
    proj_matrix:
        Optional ``(tile_size², tile_size²)`` signal-subspace projector.
    proj_beta:
        Noise-shaping strength ∈ [0, 1].
    adaptive_eta:
        If ``True``, ramps *proj_beta* from 0 → peak (§8.1.1).
    eta_peak_step:
        Step at which η reaches peak (default: n_steps//2).
    order2_gamma:
        Second-order Σ-Δ coupling strength (§8.1.2).
    order2_c1, order2_c2:
        Gain coefficients for integrator cascading (§8.1.2).
    zero_mean_integrator2:
        If True, subtract per-tile mean from integrator2 after each step
        to prevent DC drift in the feedback loop (§8.1.12).
    use_fwht:
        DEPRECATED.  If True, equivalent to ``transform_mode="fwht"``.
        Prefer ``transform_mode`` for the hybrid DCT/DWT engine (§8.2).
    transform_mode:
        Orthogonal transform applied to tiles before Σ-Δ encoding (§8.2).
        One of ``"none"``, ``"dct"``, ``"dwt"``, ``"hybrid"``, ``"auto"``,
        ``"fwht"``.  ``"auto"`` selects DCT/DWT/hybrid per tile based on
        variance and max-abs statistics.
    transform_smooth_threshold:
        Variance threshold for ``"auto"`` mode: tiles below → DCT.
    transform_outlier_threshold:
        Max-abs threshold for ``"auto"`` mode: tiles above → DWT.
    adaptive_masking:
        If True, enable per-tile bit-rate scaling (§8.2.1):
        outlier tiles get boosted ``proj_beta`` and/or extra ``n_steps``
        to allocate more bits to attention "anchor" regions.
    mask_smooth_threshold:
        Variance threshold: tiles below this are "smooth".
    mask_outlier_threshold:
        Max-abs threshold: tiles above this are "outlier/sensitive".
    mask_proj_beta_boost:
        Fractional boost applied to ``proj_beta`` for sensitive tiles.
        E.g. 0.5 means proj_beta * 1.5.
    mask_n_steps_boost:
        Extra pursuit iterations for sensitive tiles (≥0).
    use_mask_gating:
        Forwarded to :func:`residual_pursuit_nd`.  When ``True`` and a
        mask is available, zeroes out padding regions in the Σ-Δ state
        at each iteration.  See §10.3.1.
    initial_momentum:
        ``(..., M)`` initial first-order integrator.  Used for cross-head
        error sharing (§8.1.9).
    initial_integrator2:
        ``(..., M)`` initial second-order integrator.
    return_momentum:
        If True, return ``(bases, alphas, orig_shape, momentum, integrator2)``.

    Returns
    -------
    bases:
        ``(N, n_tiles, tile_size**2)`` – packed 1-bit bases.
    alphas:
        ``(N, n_tiles)`` – per-tile, per-step scaling factors.
    orig_shape:
        ``(rows, cols)`` — needed to reconstruct the original matrix
        dimensions after unpadding.
    transform_decisions:
        ``list[str]`` — per-tile transform decision (for inverse).
    momentum: (only if return_momentum=True)
        ``(n_tiles, tile_size**2)`` final first-order integrator state.
    integrator2: (only if return_momentum=True)
        ``(n_tiles, tile_size**2)`` final second-order integrator state.
    """
    assert w.dim() == 2, "encode_matrix expects a 2-D weight matrix"
    rows, cols = w.shape

    # Backward compat: use_fwht → transform_mode
    if use_fwht and transform_mode == "none":
        transform_mode = "fwht"

    # Resolve transform mode enum
    from rina.utils.transforms import TransformMode, apply_transform, compute_tile_diagnostics
    _tm = TransformMode(transform_mode)

    # Pad to tile-aligned shape
    w_padded, (pad_r, pad_c) = _pad_to_tile_multiple(w, tile_size)
    n_tr, n_tc = _tile_count(w_padded.shape, tile_size)

    # Unfold into tiles: (rows, cols) → (1, 1, ...  →  n_tr*n_tc, tile_size**2)
    # We use F.unfold which expects 4-D input (N,C,H,W).
    w_4d = w_padded.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    patches = F.unfold(
        w_4d,
        kernel_size=tile_size,
        stride=tile_size,
    )  # (1, tile_size**2, n_tiles)

    # (n_tiles, tile_size**2)
    tiles = patches.squeeze(0).transpose(0, 1).contiguous()

    # ---- Tile diagnostics (for both transform & adaptive masking) ----
    tile_vars, tile_maxabs = compute_tile_diagnostics(tiles)

    # Build validity mask: 1=real data, 0=padding (§10.3)
    # This prevents alpha underestimation for partially-filled tiles.
    validity = torch.ones_like(w_padded)  # (H_pad, W_pad)
    if pad_r > 0:
        validity[-pad_r:, :] = 0
    if pad_c > 0:
        validity[:, -pad_c:] = 0
    mask_4d = validity.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    mask_patches = F.unfold(
        mask_4d,
        kernel_size=tile_size,
        stride=tile_size,
    ).squeeze(0).transpose(0, 1).contiguous()  # (n_tiles, M)

    # Determine if mask is non-trivial (i.e. there is padding)
    has_padding = (pad_r > 0) or (pad_c > 0)

    # ---- Apply orthogonal transform (§8.2) ----
    tiles, transform_decisions = apply_transform(
        tiles,
        mode=_tm,
        tile_size=tile_size,
        smooth_threshold=transform_smooth_threshold,
        outlier_threshold=transform_outlier_threshold,
    )

    # ---- Adaptive masking: per-tile bit-rate boost (§8.2.1) ----
    if adaptive_masking:
        # Encode tiles one-by-one so we can vary n_steps/proj_beta per tile
        n_tiles = tiles.shape[0]
        M = tile_size * tile_size
        max_n_steps = n_steps + mask_n_steps_boost  # safe ceiling

        bases_all = torch.ones(max_n_steps, n_tiles, M,
                               device=tiles.device, dtype=tiles.dtype)
        alphas_all = torch.zeros(max_n_steps, n_tiles,
                                 device=tiles.device, dtype=tiles.dtype)
        n_steps_used = torch.zeros(n_tiles, dtype=torch.int32, device=tiles.device)

        if return_momentum:
            momentum_out = torch.zeros(n_tiles, M, device=tiles.device, dtype=tiles.dtype)
            integrator2_out = torch.zeros(n_tiles, M, device=tiles.device, dtype=tiles.dtype)

        # ── Expand (1, M) momentum / integrator2 to (n_tiles, M) ──
        if initial_momentum is not None and initial_momentum.shape[0] == 1 and n_tiles > 1:
            initial_momentum = initial_momentum.expand(n_tiles, -1).contiguous()
        if initial_integrator2 is not None and initial_integrator2.shape[0] == 1 and n_tiles > 1:
            initial_integrator2 = initial_integrator2.expand(n_tiles, -1).contiguous()

        for i in range(n_tiles):
            v = tile_vars[i].item()
            m = tile_maxabs[i].item()

            # Determine per-tile boost
            is_sensitive = (v >= mask_smooth_threshold or m >= mask_outlier_threshold)
            n_i = n_steps + (mask_n_steps_boost if is_sensitive else 0)
            pb_i = proj_beta * (1.0 + mask_proj_beta_boost) if is_sensitive else proj_beta

            n_steps_used[i] = n_i

            tile_i = tiles[i:i+1].unsqueeze(0)       # (1, 1, M)
            mask_i = mask_patches[i:i+1].unsqueeze(0) if has_padding else None

            result = residual_pursuit_nd(
                tile_i,
                n_steps=n_i,
                beta=beta,
                return_bases=True,
                proj_matrix=proj_matrix,
                proj_beta=pb_i,
                adaptive_eta=adaptive_eta,
                eta_peak_step=eta_peak_step,
                order2_gamma=order2_gamma,
                order2_c1=order2_c1,
                order2_c2=order2_c2,
                zero_mean_integrator2=zero_mean_integrator2,
                mask=mask_i,
                use_mask_gating=use_mask_gating,
                initial_momentum=initial_momentum[i:i+1] if initial_momentum is not None else None,
                initial_integrator2=initial_integrator2[i:i+1] if initial_integrator2 is not None else None,
                return_momentum=return_momentum,
            )

            if return_momentum:
                b_i, a_i, _, mom_i, int2_i = result
                bases_all[:n_i, i, :] = b_i.squeeze(1).squeeze(1)
                alphas_all[:n_i, i] = a_i.squeeze(1).squeeze(1)
                momentum_out[i] = mom_i.squeeze(0).squeeze(0)
                integrator2_out[i] = int2_i.squeeze(0).squeeze(0)
            else:
                b_i, a_i, _ = result
                bases_all[:n_i, i, :] = b_i.squeeze(1).squeeze(1)
                alphas_all[:n_i, i] = a_i.squeeze(1).squeeze(1)

        # Trim to actual max steps used
        max_used = int(n_steps_used.max().item())
        bases_all = bases_all[:max_used]
        alphas_all = alphas_all[:max_used]

        if return_momentum:
            return (bases_all, alphas_all, (rows, cols), transform_decisions,
                    momentum_out, integrator2_out)
        return bases_all, alphas_all, (rows, cols), transform_decisions

    # ---- Standard parallel encoding (no adaptive masking) ----
    result = residual_pursuit_nd(
        tiles,
        n_steps=n_steps,
        beta=beta,
        return_bases=True,
        proj_matrix=proj_matrix,
        proj_beta=proj_beta,
        adaptive_eta=adaptive_eta,
        eta_peak_step=eta_peak_step,
        order2_gamma=order2_gamma,
        order2_c1=order2_c1,
        order2_c2=order2_c2,
        zero_mean_integrator2=zero_mean_integrator2,
        mask=mask_patches if has_padding else None,
        use_mask_gating=use_mask_gating,
        initial_momentum=initial_momentum,
        initial_integrator2=initial_integrator2,
        return_momentum=return_momentum,
    )

    if return_momentum:
        bases, alphas, _, momentum, integrator2 = result
        return bases, alphas, (rows, cols), transform_decisions, momentum, integrator2
    else:
        bases, alphas, _ = result
        return bases, alphas, (rows, cols), transform_decisions


def encode_matrix_sequential(
    w: torch.Tensor,
    n_steps: int = 5,
    tile_size: int = 16,
    beta: float = 0.0,
    *,
    inter_tile_feedback: float = 0.0,
    proj_matrix: Optional[torch.Tensor] = None,
    proj_beta: float = 0.0,
    adaptive_eta: bool = False,
    eta_peak_step: Optional[int] = None,
    order2_gamma: float = 0.0,
    order2_c1: float = 1.0,
    order2_c2: float = 0.5,
) -> Tuple[ResidualBases, ResidualAlphas, Tuple[int, int]]:
    """§10.3 Cross-Tile Σ-Δ Noise Shaping encoder.

    Like encode_matrix, but processes tiles SEQUENTIALLY instead of in
    parallel.  After encoding tile i and reconstructing Ŵ_i, the residual
    error r = w_i − Ŵ_i is multiplied by inter_tile_feedback and injected
    into tile i+1's input BEFORE encoding.  This mimics the Δ-Σ modulator's
    noise-shaping feedback loop across the spatial (tile) dimension: the
    quantisation noise from tile i is partially encoded into the next
    tile's bases, pushing it into a region where it is less perceptually
    harmful (lower visual/semantic sensitivity).

    When inter_tile_feedback = 0, falls back to standard parallel encoding.
    Recommended: inter_tile_feedback ∈ [0.05, 0.30].
    """
    assert w.dim() == 2, "encode_matrix_sequential expects a 2-D weight matrix"
    rows, cols = w.shape

    # Pad and unfold
    w_padded, (pad_r, pad_c) = _pad_to_tile_multiple(w, tile_size)
    n_tr, n_tc = _tile_count(w_padded.shape, tile_size)

    w_4d = w_padded.unsqueeze(0).unsqueeze(0)
    patches = F.unfold(
        w_4d, kernel_size=tile_size, stride=tile_size,
    ).squeeze(0).transpose(0, 1).contiguous()  # (n_tiles, M)

    n_tiles, M = patches.shape

    # Validity mask
    validity = torch.ones_like(w_padded)
    if pad_r > 0:
        validity[-pad_r:, :] = 0
    if pad_c > 0:
        validity[:, -pad_c:] = 0
    mask_4d = validity.unsqueeze(0).unsqueeze(0)
    mask_patches = F.unfold(
        mask_4d, kernel_size=tile_size, stride=tile_size,
    ).squeeze(0).transpose(0, 1).contiguous()
    has_padding = (pad_r > 0) or (pad_c > 0)

    # Number each tile with its spatial position
    tile_rows = torch.arange(n_tr, device=w.device).repeat_interleave(n_tc)
    tile_cols = torch.arange(n_tc, device=w.device).repeat(n_tr)

    # Allocate output
    device = patches.device
    dtype = patches.dtype
    bases_all = torch.ones(n_steps, n_tiles, M, device=device, dtype=dtype)
    alphas_all = torch.zeros(n_steps, n_tiles, device=device, dtype=dtype)

    # ---- Sequential encoding with cross-tile feedback ----
    prev_residual = torch.zeros(M, device=device, dtype=dtype)
    idx_mask = mask_patches[0] if has_padding else None
    for i in range(n_tiles):
        tile_input = patches[i].clone()
        if inter_tile_feedback > 0 and i > 0:
            # Inject previous tile's residual into this tile's input
            tile_input = tile_input + inter_tile_feedback * prev_residual

        # Single-tile pursuit
        tile_data = tile_input.unsqueeze(0).unsqueeze(0)  # (1, 1, M)
        tile_mask = idx_mask.unsqueeze(0).unsqueeze(0) if has_padding else None
        b_i, a_i, w_hat_i = residual_pursuit_nd(
            tile_data,
            n_steps=n_steps,
            beta=beta,
            return_bases=True,
            proj_matrix=proj_matrix,
            proj_beta=proj_beta,
            adaptive_eta=adaptive_eta,
            eta_peak_step=eta_peak_step,
            order2_gamma=order2_gamma,
            order2_c1=order2_c1,
            order2_c2=order2_c2,
            mask=tile_mask,
        )
        bases_all[0:n_steps, i, :] = b_i.squeeze(1).squeeze(1)
        alphas_all[0:n_steps, i] = a_i.squeeze(1).squeeze(1)

        # Compute residual for next tile (vs original, not perturbed input)
        original_tile = patches[i]
        reconstructed = w_hat_i.squeeze(0).squeeze(0)
        prev_residual = original_tile - reconstructed

    return bases_all, alphas_all, (rows, cols)


def decode_from_bases(
    bases: ResidualBases,
    alphas: ResidualAlphas,
    orig_shape: Tuple[int, int],
    tile_size: int = 16,
    *,
    recon_weights: Optional[torch.Tensor] = None,
    use_fwht: bool = False,  # DEPRECATED: use transform_mode="fwht"
    transform_mode: str = "none",
    transform_decisions: Optional[List[str]] = None,
) -> torch.Tensor:
    """Reconstruct the full matrix from encoded 1-bit bases.

    Parameters
    ----------
    bases:
        ``(N, n_tiles, M)`` — packed bases.
    alphas:
        ``(N, n_tiles)`` — per-step, per-tile scaling.
    orig_shape:
        ``(rows, cols)`` — original matrix dims (without padding).
    tile_size:
        Tile size used during encoding.
    recon_weights:
        Optional ``(N,)`` tensor of per-step weights w_i for weighted
        reconstruction: Ŵ = Σ w_i · α_i · B_i instead of Σ α_i · B_i.
        If None, uniform w_i=1.0 (standard sum).
    use_fwht:
        DEPRECATED.  If True, equivalent to ``transform_mode="fwht"``.
        Prefer ``transform_mode`` for the hybrid DCT/DWT engine (§8.2).
    transform_mode:
        Orthogonal transform mode used during encoding.  Must match
        the mode passed to ``encode_matrix`` (§8.2).
        One of ``"none"``, ``"dct"``, ``"dwt"``, ``"hybrid"``, ``"auto"``,
        ``"fwht"``.
    transform_decisions:
        Per-tile transform decision list (required when transform_mode
        is ``"auto"``, ``"hybrid"``, or ``"dwt"``).  Each element is one
        of ``"dct"``, ``"dwt"``, ``"hybrid"``, or ``"fwht"``.

    Returns
    -------
    Ŵ:
        Reconstructed matrix, shape ``orig_shape``.
    """
    N, n_tiles, M = bases.shape
    assert M == tile_size**2
    assert alphas.shape == (N, n_tiles)

    # Backward compat: use_fwht → transform_mode
    if use_fwht and transform_mode == "none":
        transform_mode = "fwht"

    # Reconstruct per tile
    if recon_weights is not None:
        assert recon_weights.shape == (N,), \
            f"recon_weights shape {recon_weights.shape} != ({N},)"
        weighted_alphas = alphas.float() * recon_weights.unsqueeze(1).to(alphas.device)
        w_tiles = torch.einsum("nt,ntm->tm", weighted_alphas, bases.float())
    else:
        w_tiles = torch.einsum("nt,ntm->tm", alphas.float(), bases.float())

    # ---- Apply inverse orthogonal transform (§8.2) ----
    from rina.utils.transforms import TransformMode, apply_inverse_transform
    _tm = TransformMode(transform_mode)

    # Handle legacy FWHT mode through the new pipeline
    if _tm == TransformMode.FWHT:
        w_tiles, _ = apply_inverse_transform(
            w_tiles,
            mode=TransformMode.FWHT,
            tile_size=tile_size,
            transform_decisions=["fwht"] * n_tiles,
        )
    elif _tm != TransformMode.NONE:
        w_tiles, _ = apply_inverse_transform(
            w_tiles,
            mode=_tm,
            tile_size=tile_size,
            transform_decisions=transform_decisions,
        )

    # Invert F.unfold
    pad_r = (tile_size - orig_shape[0] % tile_size) % tile_size
    pad_c = (tile_size - orig_shape[1] % tile_size) % tile_size
    padded_h = orig_shape[0] + pad_r
    padded_w = orig_shape[1] + pad_c
    n_tr = padded_h // tile_size
    n_tc = padded_w // tile_size

    w_4d = F.fold(
        w_tiles.transpose(0, 1).unsqueeze(0),  # (1, tile_size**2, n_tiles)
        output_size=(padded_h, padded_w),
        kernel_size=tile_size,
        stride=tile_size,
    )  # (1, 1, H, W)

    w_rec = w_4d.squeeze(0).squeeze(0)
    return _unpad(w_rec, orig_shape)


# ---------------------------------------------------------------------------
# Differential encode / decode (§8.2)
# ---------------------------------------------------------------------------


def differential_encode_decode(
    w: torch.Tensor,
    n_steps: int = 5,
    tile_size: int = 16,
    beta: float = 0.0,
    *,
    proj_matrix: Optional[torch.Tensor] = None,
    proj_beta: float = 0.0,
    adaptive_eta: bool = False,
    eta_peak_step: Optional[int] = None,
    order2_gamma: float = 0.0,
    order2_c1: float = 1.0,
    order2_c2: float = 0.5,
    average_alpha: bool = True,
) -> Tuple[torch.Tensor, dict]:
    """Differential noise cancellation via diverse-basis encodings (§8.2).

    Encodes the same weight matrix twice with **different strategies**
    to produce genuinely dissimilar error patterns:

    * **Path A**: ``n_steps=N``, ``beta=β``             (standard RBP)
    * **Path B**: ``n_steps=N``, ``beta=β+0.15``        (momentum-perturbed RBP)

    The two reconstructions are averaged:

        Ŵ_diff = (Ŵ_A + Ŵ_B) / 2

    This is the S/W analogue of a differential circuit: two encoding
    paths whose partially independent quantisation errors cancel upon
    averaging.

    .. note::

        Using ``sign_flip=-1`` produces ``±Ŵ`` — catastrophic anti-correlation
        rather than complementary noise.  The momentum-shift / extra-step
        approach avoids this by keeping both paths as standard (sign_flip=+1)
        encodings.

    Parameters
    ----------
    w:
        Full-precision matrix, shape ``(rows, cols)``.
    n_steps:
        Number of 1-bit bases for Path A.
        Path B uses ``n_steps+1`` for narrow mats or ``n_steps`` for wide.
    tile_size:
        Tile dimension for block-wise encoding.
    beta:
        Momentum coefficient for Path A.  Path B adds 0.15 for wide mats.
    proj_matrix:
        Optional signal-subspace projector for noise shaping.
    proj_beta:
        Noise-shaping strength (applied independently per encoding).
    average_alpha:
        If ``True`` (default), (Ŵ_A + Ŵ_B) / 2 — equal-weight average.

    Returns
    -------
    Ŵ_diff:
        Differential reconstruction, shape ``(rows, cols)``.
    diag:
        Dictionary with diagnostic keys:

        - ``nrr``: Noise Reduction Ratio, 1 − ‖ε_diff‖ / mean(‖ε_A‖, ‖ε_B‖).
          Perfect cancellation → 1.0, no effect → 0.0, anti-effect → < 0.0.
        - ``cross_corr``:  ``⟨ε_A, ε_B⟩ / (‖ε_A‖·‖ε_B‖)``.
          Negative → complementary noise, positive → correlated noise.
        - ``mse_a`` / ``mse_b`` / ``mse_diff``:  Individual and combined MSE.
        - ``cosine_a`` / ``cosine_b`` / ``cosine_diff``:  Cosine similarities.
        - ``snr_a_db`` / ``snr_b_db`` / ``snr_diff_db``:  SNR values.
    """
    assert w.dim() == 2, "differential_encode_decode expects 2-D matrix"

    rows, cols = w.shape
    w_padded, (pad_r, pad_c) = _pad_to_tile_multiple(w, tile_size)
    w_4d = w_padded.unsqueeze(0).unsqueeze(0)
    patches = F.unfold(
        w_4d, kernel_size=tile_size, stride=tile_size,
    ).squeeze(0).transpose(0, 1).contiguous()

    # Build validity mask: 1=real data, 0=padding (§10.3)
    validity = torch.ones_like(w_padded)
    if pad_r > 0:
        validity[-pad_r:, :] = 0
    if pad_c > 0:
        validity[:, -pad_c:] = 0
    mask_4d = validity.unsqueeze(0).unsqueeze(0)
    mask_patches = F.unfold(
        mask_4d, kernel_size=tile_size, stride=tile_size,
    ).squeeze(0).transpose(0, 1).contiguous()
    has_padding = (pad_r > 0) or (pad_c > 0)

    # ---- Determine B-strategy (always momentum perturbation) ----
    # Path B differs from Path A via a small momentum shift.
    # At beta=0, we inject beta=0.15 to break the deterministic tie.
    # At beta>0, we add 0.15 to create a genuinely different trajectory.
    _beta_b = beta + 0.15

    _common_kwargs = dict(
        proj_matrix=proj_matrix, proj_beta=proj_beta,
        adaptive_eta=adaptive_eta, eta_peak_step=eta_peak_step,
        order2_gamma=order2_gamma, order2_c1=order2_c1, order2_c2=order2_c2,
    )

    # ---- Encode A: standard RBP ----
    bases_a, alphas_a, _ = residual_pursuit_nd(
        patches, n_steps=n_steps, beta=beta, return_bases=True,
        sign_flip=+1.0,
        mask=mask_patches if has_padding else None,
        **_common_kwargs,
    )
    w_tiles_a = torch.einsum("nt,ntm->tm", alphas_a.float(), bases_a.float())

    # ---- Encode B: momentum-perturbed RBP (same cardinality) ----
    bases_b, alphas_b, _ = residual_pursuit_nd(
        patches, n_steps=n_steps, beta=_beta_b, return_bases=True,
        sign_flip=+1.0,
        mask=mask_patches if has_padding else None,
        **_common_kwargs,
    )
    w_tiles_b = torch.einsum("nt,ntm->tm", alphas_b.float(), bases_b.float())

    # ---- Reconstruct per-encoding matrices ----
    padded_h = w_padded.shape[0]
    padded_w_ = w_padded.shape[1]

    def _fold(tiles: torch.Tensor) -> torch.Tensor:
        mat = F.fold(
            tiles.transpose(0, 1).unsqueeze(0),
            output_size=(padded_h, padded_w_),
            kernel_size=tile_size, stride=tile_size,
        ).squeeze(0).squeeze(0)
        return _unpad(mat, (rows, cols))

    ŵ_a = _fold(w_tiles_a)
    ŵ_b = _fold(w_tiles_b)

    # ---- Differential reconstruction ----
    ŵ_diff = (ŵ_a + ŵ_b) / 2

    # ---- Pack bases for storage (§12) ----
    # Return both path's bases+alphas so callers can store them
    # instead of a full-reconstruction cache (fixes full_k_hat leak).
    bases_a_storage = pack_bases(bases_a)
    bases_b_storage = pack_bases(bases_b)
    bases_shape_M = bases_a.shape[-1]

    # ---- Diagnostics ----
    w_flat = w.reshape(-1)
    w_hat_a = ŵ_a.reshape(-1)
    w_hat_b = ŵ_b.reshape(-1)
    w_hat_diff = ŵ_diff.reshape(-1)

    eps_a = w_flat - w_hat_a
    eps_b = w_flat - w_hat_b
    eps_diff = w_flat - w_hat_diff

    norm_a = eps_a.norm().item()
    norm_b = eps_b.norm().item()
    norm_diff = eps_diff.norm().item()

    mean_norm = (norm_a + norm_b) / 2
    nrr = 1.0 - (norm_diff / max(mean_norm, 1e-12))

    cross_corr = torch.dot(eps_a, eps_b).item() / max(norm_a * norm_b, 1e-12)

    mse_a = F.mse_loss(ŵ_a, w).item()
    mse_b = F.mse_loss(ŵ_b, w).item()
    mse_diff = F.mse_loss(ŵ_diff, w).item()

    def _cos_sim(h: torch.Tensor) -> float:
        return F.cosine_similarity(
            w.reshape(-1).unsqueeze(0), h.reshape(-1).unsqueeze(0)
        ).item()

    def _snr_db(h: torch.Tensor) -> float:
        noise = ((w - h) ** 2).mean()
        signal = (w ** 2).mean()
        return 10 * math.log10(
            max(signal.item() / max(noise.item(), 1e-12), 1e-12)
        )

    diag = {
        "nrr": nrr,
        "cross_corr": cross_corr,
        "mse_a": mse_a,
        "mse_b": mse_b,
        "mse_diff": mse_diff,
        "cosine_a": _cos_sim(ŵ_a),
        "cosine_b": _cos_sim(ŵ_b),
        "cosine_diff": _cos_sim(ŵ_diff),
        "snr_a_db": _snr_db(ŵ_a),
        "snr_b_db": _snr_db(ŵ_b),
        "snr_diff_db": _snr_db(ŵ_diff),
        # Dual-path bases for storage (fixes full_k_hat leak)
        "bases_a": bases_a_storage,
        "bases_b": bases_b_storage,
        "alphas_a": alphas_a,
        "alphas_b": alphas_b,
        "bases_shape_M": bases_shape_M,
    }
    return ŵ_diff, diag


# ---------------------------------------------------------------------------
# Adaptive N Encoder (§10.2.3 Adaptive N Scheduling)
# ---------------------------------------------------------------------------


def adaptive_encode_matrix(
    w: torch.Tensor,
    n_steps_base: int = 5,
    n_steps_extra: int = 3,
    tile_size: int = 16,
    beta: float = 0.0,
    *,
    proj_matrix: Optional[torch.Tensor] = None,
    proj_beta: float = 0.0,
    adaptive_eta: bool = False,
    eta_peak_step: Optional[int] = None,
    order2_gamma: float = 0.0,
    order2_c1: float = 1.0,
    order2_c2: float = 0.5,
    energy_threshold_ratio: float = 0.5,
) -> Tuple[ResidualBases, ResidualAlphas, torch.Tensor, Tuple[int, int]]:
    """Adaptive N 1-bit tile encoder with energy-based step allocation (§10.2.3).

    High-energy tiles (‖tile‖² > threshold) receive *n_steps_base + n_steps_extra*
    steps.  Low-energy tiles receive *n_steps_base* steps only.

    This mirrors the Δ-Σ DAC insight: higher-amplitude signals need more
    oversampling to maintain the same SNR.  By allocating extra steps only
    to the tiles that contribute most to MSE, we improve **weighted error
    symmetry** while controlling the **effective N ratio** (Eq. 10.2c).

    Parameters
    ----------
    w:
        Full-precision matrix, shape ``(rows, cols)``.
    n_steps_base:
        Base number of 1-bit bases for low-energy tiles.
    n_steps_extra:
        Extra steps allocated to high-energy tiles.
        Total for high-energy tiles = n_steps_base + n_steps_extra.
    tile_size:
        Tile dimension (default 16).
    beta:
        Momentum coefficient.
    proj_matrix:
        Optional signal-subspace projector.
    proj_beta:
        Noise-shaping strength ∈ [0, 1].
    adaptive_eta:
        If ``True``, ramps η per tile.
    eta_peak_step:
        Step for η peak.
    order2_gamma:
        Second-order Σ-Δ coupling strength.
    order2_c1, order2_c2:
        Integrator gain coefficients.
    energy_threshold_ratio:
        Fraction of mean energy above which tiles are considered
        "high-energy".  Default 0.5 (→ tiles above 0.5×mean receive
        extra steps).

    Returns
    -------
    bases:
        Packed 1-bit bases.  Shape ``(N_max, n_tiles, M)`` where
        ``N_max = n_steps_base + n_steps_extra``.  For low-energy
        tiles, the trailing *n_steps_extra* bases are filled with
        ``+1.0`` (neutral) and corresponding alphas are **zero**.
    alphas:
        Per-step, per-tile scaling.  Shape ``(N_max, n_tiles)``.
        Trailing rows for low-energy tiles are zero.
    n_steps_per_tile:
        ``(n_tiles,)`` tensor of actual step counts per tile.
    orig_shape:
        ``(rows, cols)`` for reconstruction.
    """
    assert w.dim() == 2, "adaptive_encode_matrix expects a 2-D weight matrix"
    rows, cols = w.shape

    # Pad and unfold
    w_padded, (pad_r, pad_c) = _pad_to_tile_multiple(w, tile_size)
    w_4d = w_padded.unsqueeze(0).unsqueeze(0)
    patches = F.unfold(
        w_4d,
        kernel_size=tile_size,
        stride=tile_size,
    ).squeeze(0).transpose(0, 1).contiguous()  # (n_tiles, M)

    n_tiles, M = patches.shape
    N_max = n_steps_base + n_steps_extra

    # Build validity mask: 1=real data, 0=padding (§10.3)
    validity = torch.ones_like(w_padded)
    if pad_r > 0:
        validity[-pad_r:, :] = 0
    if pad_c > 0:
        validity[:, -pad_c:] = 0
    mask_4d = validity.unsqueeze(0).unsqueeze(0)
    mask_patches = F.unfold(
        mask_4d,
        kernel_size=tile_size,
        stride=tile_size,
    ).squeeze(0).transpose(0, 1).contiguous()  # (n_tiles, M)
    has_padding = (pad_r > 0) or (pad_c > 0)

    # ---- Determine which tiles are high-energy ----
    energy_per_tile = (patches ** 2).sum(dim=-1)  # (n_tiles,)
    mean_energy = energy_per_tile.mean()
    threshold = energy_threshold_ratio * mean_energy
    high_energy_mask = energy_per_tile > threshold  # (n_tiles,) bool
    n_steps_per_tile = torch.where(
        high_energy_mask,
        torch.full_like(energy_per_tile, N_max, dtype=torch.long),
        torch.full_like(energy_per_tile, n_steps_base, dtype=torch.long),
    )

    # ---- Encode each tile group ----
    device = patches.device
    dtype = patches.dtype

    # Allocate output tensors — filled with neutral bases and zero alphas
    bases_all = torch.ones(N_max, n_tiles, M, device=device, dtype=dtype)
    alphas_all = torch.zeros(N_max, n_tiles, device=device, dtype=dtype)

    # Encode high-energy tiles with extra steps
    if high_energy_mask.any():
        hi_patches = patches[high_energy_mask]
        hi_bases, hi_alphas, _ = residual_pursuit_nd(
            hi_patches,
            n_steps=N_max,
            beta=beta,
            return_bases=True,
            proj_matrix=proj_matrix,
            proj_beta=proj_beta,
            adaptive_eta=adaptive_eta,
            eta_peak_step=eta_peak_step,
            order2_gamma=order2_gamma,
            order2_c1=order2_c1,
            order2_c2=order2_c2,
            mask=mask_patches[high_energy_mask] if has_padding else None,
        )
        hi_indices = high_energy_mask.nonzero(as_tuple=True)[0]
        bases_all[:, hi_indices, :] = hi_bases
        alphas_all[:, hi_indices] = hi_alphas

    # Encode low-energy tiles with base steps only
    if (~high_energy_mask).any():
        lo_patches = patches[~high_energy_mask]
        lo_bases, lo_alphas, _ = residual_pursuit_nd(
            lo_patches,
            n_steps=n_steps_base,
            beta=beta,
            return_bases=True,
            proj_matrix=proj_matrix,
            proj_beta=proj_beta,
            adaptive_eta=adaptive_eta,
            eta_peak_step=eta_peak_step,
            order2_gamma=order2_gamma,
            order2_c1=order2_c1,
            order2_c2=order2_c2,
            mask=mask_patches[~high_energy_mask] if has_padding else None,
        )
        lo_indices = (~high_energy_mask).nonzero(as_tuple=True)[0]
        bases_all[:n_steps_base, lo_indices, :] = lo_bases
        alphas_all[:n_steps_base, lo_indices] = lo_alphas
        # trailing rows already zero/ones from init

    return bases_all, alphas_all, n_steps_per_tile, (rows, cols)


# ---------------------------------------------------------------------------
# nn.Module wrapper
# ---------------------------------------------------------------------------


class ResidualBinaryPursuit(nn.Module):
    """Trainable-light PyTorch module wrapping Σ-Δ Residual Binary Pursuit.

    Supports Noise-Shaped RBP (§8.1) via ``proj_rank`` and ``proj_beta``,
    and Adaptive η Scheduling (§8.1.1) via ``adaptive_eta``.

    Usage::

        # Plain RBP
        rbp = ResidualBinaryPursuit(n_steps=5, tile_size=16, beta=0.0)
        w_hat = rbp(weight_matrix)

        # Noise-Shaped RBP with adaptive η
        ns_rbp = ResidualBinaryPursuit(
            n_steps=5, tile_size=16, beta=0.0,
            proj_rank=16, proj_beta=0.8, adaptive_eta=True,
        )
        w_hat = ns_rbp(weight_matrix)

    Parameters
    ----------
    n_steps:
        Number of 1-bit bases (oversampling ratio N).
        N=5 is the recommended sweet-spot.
    tile_size:
        Size of each tile.  Must be ≥ 1. 16 matches GPU Tensor Cores.
    beta:
        Momentum coefficient for second-order Σ-Δ.
    proj_rank:
        Number of principal components for noise-shaping projection.
        ``0`` disables noise shaping.  Recommended: 8–32.
    proj_beta:
        Noise-shaping strength ∈ [0, 1].
        0 = off, 1 = full nullspace suppression per step.
        Recommended: 0.5–0.8.
    adaptive_eta:
        If ``True``, ramps *proj_beta* from 0 → its peak linearly
        across the first half of the encoding steps.  Prevents
        early-step oversuppression.  (§8.1.1)
    """

    def __init__(
        self,
        n_steps: int = 5,
        tile_size: int = 16,
        beta: float = 0.0,
        proj_rank: int = 0,
        proj_beta: float = 0.0,
        adaptive_eta: bool = False,
    ) -> None:
        super().__init__()
        if n_steps < 1:
            raise ValueError(f"n_steps must be ≥ 1, got {n_steps}")
        if tile_size < 1:
            raise ValueError(f"tile_size must be ≥ 1, got {tile_size}")

        self.n_steps = n_steps
        self.tile_size = tile_size
        self.beta = beta
        self.proj_rank = proj_rank
        self.proj_beta = proj_beta
        self.adaptive_eta = adaptive_eta

    # ------------------------------------------------------------------
    # Projection matrix builder
    # ------------------------------------------------------------------

    def _ensure_proj_matrix(self, w: torch.Tensor) -> Optional[torch.Tensor]:
        """Return P_signal projector, building it on first call if needed."""
        if self.proj_rank <= 0 or self.proj_beta <= 0.0:
            return None
        return _build_proj_matrix(w, self.tile_size, self.proj_rank)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(
        self, w: torch.Tensor
    ) -> Tuple[ResidualBases, ResidualAlphas, Tuple[int, int]]:
        """Encode a 2-D weight matrix to 1-bit bases + scaling factors.

        Returns
        -------
        bases:
            ``(N, n_tiles, tile_size²)`` packed bases.
        alphas:
            ``(N, n_tiles)`` per-step, per-tile L1-optimal scales.
        orig_shape:
            ``(rows, cols)`` un-padded shape for later reconstruction.
        """
        proj = self._ensure_proj_matrix(w)
        return encode_matrix(
            w,
            n_steps=self.n_steps,
            tile_size=self.tile_size,
            beta=self.beta,
            proj_matrix=proj,
            proj_beta=self.proj_beta,
        )

    def decode(
        self,
        bases: ResidualBases,
        alphas: ResidualAlphas,
        orig_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """Reconstruct the matrix from encoded representation."""
        return decode_from_bases(
            bases,
            alphas,
            orig_shape,
            tile_size=self.tile_size,
        )

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """Encode → decode round-trip (convenience, not for inference)."""
        bases, alphas, shape, _ = self.encode(w)
        return self.decode(bases, alphas, shape)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def compute_metrics(
        self, w_orig: torch.Tensor, w_hat: torch.Tensor
    ) -> dict:
        """Compute reconstruction quality metrics.

        When ``proj_rank > 0``, also computes **effective CosSim** in the
        signal subspace (projecting both w and Ŵ through the same SVD
        basis), which is the metric that matters for downstream softmax
        sensitivity.
        """
        w = w_orig.float()
        h = w_hat.float()

        mse = F.mse_loss(h, w).item()

        signal_power = (w**2).mean()
        noise_power = ((w - h) ** 2).mean()
        snr_db = 10 * math.log10(
            max(signal_power.item() / max(noise_power.item(), 1e-12), 1e-12)
        )

        cos_sim = F.cosine_similarity(
            w.reshape(-1).unsqueeze(0), h.reshape(-1).unsqueeze(0)
        ).item()

        result = {"mse": mse, "snr_db": snr_db, "cosine_similarity": cos_sim}

        # Effective metrics in signal subspace
        proj = self._ensure_proj_matrix(w_orig)
        if proj is not None:
            result.update(
                self._compute_effective_metrics(w_orig, w_hat, proj)
            )

        return result

    def _compute_effective_metrics(
        self,
        w_orig: torch.Tensor,
        w_hat: torch.Tensor,
        proj_matrix: torch.Tensor,
    ) -> dict:
        """Compute metrics after projecting into the signal subspace."""
        w = w_orig.float()
        h = w_hat.float()

        # Pad and unfold both into tiles
        w_pad, (pad_r, pad_c) = _pad_to_tile_multiple(w, self.tile_size)
        h_pad, _ = _pad_to_tile_multiple(h, self.tile_size)

        patches_w = F.unfold(
            w_pad.unsqueeze(0).unsqueeze(0),
            kernel_size=self.tile_size,
            stride=self.tile_size,
        ).squeeze(0).t()  # (n_tiles, M)

        patches_h = F.unfold(
            h_pad.unsqueeze(0).unsqueeze(0),
            kernel_size=self.tile_size,
            stride=self.tile_size,
        ).squeeze(0).t()  # (n_tiles, M)

        # Project each tile into signal subspace
        proj = proj_matrix.to(device=patches_w.device, dtype=patches_w.dtype)
        w_sig = torch.matmul(patches_w, proj.T)  # (n_tiles, M)
        h_sig = torch.matmul(patches_h, proj.T)  # (n_tiles, M)

        eff_mse = F.mse_loss(h_sig, w_sig).item()

        sig_power = (w_sig**2).mean()
        noise_power = ((w_sig - h_sig) ** 2).mean()
        eff_snr_db = 10 * math.log10(
            max(sig_power.item() / max(noise_power.item(), 1e-12), 1e-12)
        )

        eff_cos = F.cosine_similarity(
            w_sig.reshape(-1).unsqueeze(0),
            h_sig.reshape(-1).unsqueeze(0),
        ).item()

        return {
            "effective_mse": eff_mse,
            "effective_snr_db": eff_snr_db,
            "effective_cosine_similarity": eff_cos,
        }

    @staticmethod
    def compute_metrics_static(
        w_orig: torch.Tensor, w_hat: torch.Tensor
    ) -> dict:
        """Compute basic reconstruction quality metrics (static helper)."""
        w = w_orig.float()
        h = w_hat.float()

        mse = F.mse_loss(h, w).item()

        signal_power = (w**2).mean()
        noise_power = ((w - h) ** 2).mean()
        snr_db = 10 * math.log10(
            max(signal_power.item() / max(noise_power.item(), 1e-12), 1e-12)
        )

        cos_sim = F.cosine_similarity(
            w.reshape(-1).unsqueeze(0), h.reshape(-1).unsqueeze(0)
        ).item()

        return {"mse": mse, "snr_db": snr_db, "cosine_similarity": cos_sim}

    def extra_repr(self) -> str:
        parts = [
            f"n_steps={self.n_steps}",
            f"tile_size={self.tile_size}",
            f"beta={self.beta}",
        ]
        if self.proj_rank > 0:
            parts.append(f"proj_rank={self.proj_rank}")
            parts.append(f"proj_beta={self.proj_beta}")
        if self.adaptive_eta:
            parts.append("adaptive_eta=True")
        return ", ".join(parts)