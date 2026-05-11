"""
Phase 2e: Configurable Tile Encoding (8×8 default) with Log-Quantized α and Outlier Protection
===============================================================================================

Core functions for tile-based Σ-Δ encoding with 4-bit α quantization and
MAD-based outlier tile detection.

Tile size is configurable — default 8×8 for optimal CR/quality tradeoff.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


ALPHA_SCHEMES = ("linear", "fixed_log", "dynamic_log", "nonlinear_log")


def encode_tile(
    tile: torch.Tensor,
    alpha_max: float,
    n_steps: int = 3,
    alpha_scheme: str = "nonlinear_log",
    K_offset: float = 4.0,
    log_min: float = 1e-4,
    log_max: float = 10.0,
    nonlinear_gamma: float = 0.55,
    normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    M = tile.numel()
    w_flat = tile.flatten().float()

    if normalize:
        mu = w_flat.mean().item()
        sigma = w_flat.std().clamp(min=1e-8).item()
        z = (w_flat - mu) / sigma
    else:
        mu = 0.0
        sigma = 1.0
        z = w_flat

    w_hat = torch.zeros_like(z)

    alphas_q_list = []
    signs_list = []

    for _ in range(n_steps):
        residual = z - w_hat
        alpha = residual.abs().mean().item()
        B = torch.sign(residual)
        alpha_q = _quantize_alpha(alpha, alpha_scheme, alpha_max, K_offset, log_min, log_max, nonlinear_gamma)
        alpha_fp = _dequantize_alpha(alpha_q, alpha_scheme, alpha_max, K_offset, log_min, log_max, nonlinear_gamma)
        w_hat = w_hat + alpha_fp * B
        alphas_q_list.append(alpha_q)
        signs_list.append(B.to(torch.int8))
    return (torch.tensor(alphas_q_list, dtype=torch.uint8), torch.stack(signs_list), mu, sigma)


def decode_tile(
    alphas_q: torch.Tensor, signs: torch.Tensor, alpha_max: float,
    alpha_scheme: str = "nonlinear_log", K_offset: float = 4.0,
    log_min: float = 1e-4, log_max: float = 10.0, nonlinear_gamma: float = 0.55,
    tile_size: int = 4, mu: float = 0.0, sigma: float = 1.0,
) -> torch.Tensor:
    n_steps = alphas_q.shape[0]
    M = signs.shape[-1]  # actual M from sign tensor
    w_hat = torch.zeros(M, dtype=torch.float32)
    for step in range(n_steps):
        aq = int(alphas_q[step].item())
        alpha_fp = _dequantize_alpha(aq, alpha_scheme, alpha_max, K_offset, log_min, log_max, nonlinear_gamma)
        w_hat = w_hat + alpha_fp * signs[step].float()
    z_hat = w_hat.reshape(tile_size, tile_size)
    return z_hat * sigma + mu


# ── α Quantization Schemes ──────────────────────────────────────────────────

def dynamic_log_quantize_4bit(alpha: float, alpha_max: float, K: float = 4.0) -> int:
    n = math.log2(max(alpha, 1e-12) / max(alpha_max, 1e-8))
    return min(15, max(0, int(round(n * K + 8))))

def dynamic_log_dequantize_4bit(q: int, alpha_max: float, K: float = 4.0) -> float:
    return alpha_max * (2.0 ** ((q - 8) / K))

def nonlinear_log_quantize_4bit(alpha: float, alpha_max: float, g: float = 0.55) -> int:
    n = math.log2(max(alpha, 1e-12) / max(alpha_max, 1e-8))
    x = max(n + 8.0, 0.0) / 8.0
    return min(15, max(0, int(round((x ** g) * 15))))

def nonlinear_log_dequantize_4bit(q: int, alpha_max: float, g: float = 0.55) -> float:
    x = (q / 15.0) ** (1.0 / g)
    return alpha_max * (2.0 ** (x * 8.0 - 8.0))

def fixed_log_quantize_4bit(alpha: float, lo: float = 1e-4, hi: float = 10.0) -> int:
    v = math.log2(max(alpha, 1e-12)); L = math.log2(lo); H = math.log2(hi)
    return min(15, max(0, int(round((v - L) / (H - L) * 15))))

def fixed_log_dequantize_4bit(q: int, lo: float = 1e-4, hi: float = 10.0) -> float:
    L = math.log2(lo); H = math.log2(hi)
    return 2.0 ** (L + (q / 15.0) * (H - L))

def fixed_log_quantize_5bit(alpha: float, lo: float = 1e-7, hi: float = 1.0) -> int:
    v = math.log2(max(alpha, 1e-12)); L = math.log2(lo); H = math.log2(hi)
    return min(31, max(0, int(round((v - L) / (H - L) * 31))))

def fixed_log_dequantize_5bit(q: int, lo: float = 1e-7, hi: float = 1.0) -> float:
    L = math.log2(lo); H = math.log2(hi)
    return 2.0 ** (L + (q / 31.0) * (H - L))

def linear_quantize_4bit(alpha: float, mx: float) -> int:
    if mx <= 0: return 0
    return min(15, max(0, int(round(alpha / mx * 15))))

def linear_dequantize_4bit(q: int, mx: float) -> float:
    return (q / 15.0) * mx

def _quantize_alpha(a, s, amax, K, lo, hi, g=0.55, bits=4):
    if bits == 5:
        return _quantize_alpha_5bit(a, s, amax, K, lo, hi, g)
    if s == "linear": return linear_quantize_4bit(a, amax)
    if s == "fixed_log": return fixed_log_quantize_4bit(a, lo, hi)
    if s == "dynamic_log": return dynamic_log_quantize_4bit(a, amax, K)
    return nonlinear_log_quantize_4bit(a, amax, g)

def _dequantize_alpha(q, s, amax, K, lo, hi, g=0.55, bits=4):
    if bits == 5:
        return _dequantize_alpha_5bit(q, s, amax, K, lo, hi, g)
    if s == "linear": return linear_dequantize_4bit(q, amax)
    if s == "fixed_log": return fixed_log_dequantize_4bit(q, lo, hi)
    if s == "dynamic_log": return dynamic_log_dequantize_4bit(q, amax, K)
    return nonlinear_log_dequantize_4bit(q, amax, g)

def _quantize_alpha_5bit(a, s, amax, K, lo, hi, g=0.55):
    if s == "nonlinear_log":
        n = math.log2(max(a, 1e-12) / max(amax, 1e-8))
        x = max(n + 8.0, 0.0) / 8.0
        return min(31, max(0, int(round((x ** g) * 31))))
    if s == "dynamic_log":
        n = math.log2(max(a, 1e-12) / max(amax, 1e-8))
        return min(31, max(0, int(round(n * K + 16))))
    if s == "linear":
        if amax <= 0: return 0
        return min(31, max(0, int(round(a / amax * 31))))
    if s == "fixed_log":
        v = math.log2(max(a, 1e-12)); L = math.log2(lo); H = math.log2(hi)
        return min(31, max(0, int(round((v - L) / (H - L) * 31))))
    return 0

def _dequantize_alpha_5bit(q, s, amax, K, lo, hi, g=0.55):
    if s == "nonlinear_log":
        x = (q / 31.0) ** (1.0 / g)
        return amax * (2.0 ** (x * 8.0 - 8.0))
    if s == "dynamic_log":
        return amax * (2.0 ** ((q - 16) / K))
    if s == "linear":
        return (q / 31.0) * amax
    if s == "fixed_log":
        L = math.log2(lo); H = math.log2(hi)
        return 2.0 ** (L + (q / 31.0) * (H - L))
    return 0.0


# ── Outlier Detection ──────────────────────────────────────────────────────

def detect_outlier_tiles(K_h, Q_h=None, tile_size=8, mad_threshold=3.0, outlier_ratio=0.2):
    T, d_head = K_h.shape
    K_f = K_h.float(); device = K_f.device
    score = K_f.norm(p=2, dim=0)
    if Q_h is not None:
        score = Q_h.float().mean(dim=0).abs() * score
    med = score.median()
    mad = (score - med).abs().median().clamp(min=1e-8)
    dim_mask = score > med + mad_threshold * mad
    tr = T // tile_size; nc = d_head // tile_size
    mask = torch.zeros(tr * nc, dtype=torch.bool, device=device)
    thresh = max(1, int(tile_size * outlier_ratio))
    for r in range(tr):
        for c in range(nc):
            if dim_mask[c * tile_size:(c + 1) * tile_size].sum().item() > thresh:
                mask[r * nc + c] = True
    return mask, dim_mask, med + mad_threshold * mad


# ── Full Matrix Encoder ────────────────────────────────────────────────────

TILE_SIGN_WORDS = {4: 1, 8: 4, 16: 16}  # tile_size -> number of uint16 per sign vector


def encode_4x4_matrix(
    mat: torch.Tensor,
    n_steps: int = 3,
    alpha_scheme: str = "nonlinear_log",
    K_offset: float = 4.0,
    log_min: float = 1e-4,
    log_max: float = 10.0,
    outlier_tile_mask: Optional[torch.Tensor] = None,
    tile_size: int = 4,
    packed: bool = True,
    maxae_fp16_threshold: float = 0.1,
    maxae_boost_threshold: float = 0.05,
    boost_n_steps: int = 4,
    nonlinear_gamma: float = 0.55,
    use_relative_threshold: bool = True,
    normalize: bool = True,
    position_weight_power: float = 0.0,
    plane_refine_steps: int = 1,
) -> dict:
    """Encode a matrix using configurable tile_size (default 8xd78).

    N auto-scaling: tile_rows <=1 -> N>=4, <=2 -> N>=3, else base n_steps.

    position_weight_power: >0 enables RoPE-frequency-weighted residual.
      Higher power -> more protection for fine position (high-k) dims.
      Power=2.0: last dim gets 4x weight of first dim.
      Power=0.0: uniform weighting (default, disabled).
    """
    T, d_head = mat.shape
    mat = mat.float()
    ts = tile_size
    M = ts * ts
    nt_dim = d_head // ts
    nt_tok = T // ts
    nt = nt_tok * nt_dim
    device = mat.device

    # N auto-scaling
    if nt_tok <= 1: base_min = 4
    elif nt_tok <= 2: base_min = 3
    else: base_min = n_steps
    effective_n = min(max(n_steps, base_min), 6)
    eff_boost = min(max(boost_n_steps, effective_n + 1), 6)
    max_N = max(effective_n, eff_boost)
    n_sw = TILE_SIGN_WORDS.get(ts, (ts * ts + 15) // 16)  # uint16 per sign vector

    if outlier_tile_mask is None:
        outlier_tile_mask = torch.zeros(nt, dtype=torch.bool, device=device)

    # Extract tiles
    tiles = []
    for tr in range(nt_tok):
        for tc in range(nt_dim):
            t = mat[tr * ts:(tr + 1) * ts, tc * ts:(tc + 1) * ts]
            if t.shape != (ts, ts):
                t = F.pad(t, (0, ts - t.shape[1], 0, ts - t.shape[0]))
            tiles.append(t.flatten())
    tiles = torch.stack(tiles)

    n_sb = (nt_tok // 2) * (nt_dim // 2)

    # ── Normalization: per-tile μ/σ to eliminate α dilution ──────────────
    tiles_f = tiles.float()
    use_normalize = normalize and n_sb > 0 and ts >= 8
    if use_normalize:
        mu = tiles_f.mean(dim=1)
        sigma = tiles_f.std(dim=1).clamp(min=1e-8)
        src = (tiles_f - mu.unsqueeze(1)) / sigma.unsqueeze(1)
    else:
        mu = torch.zeros(nt, dtype=torch.float32, device=device)
        sigma = torch.ones(nt, dtype=torch.float32, device=device)
        src = tiles_f

    # Tile-local α_max: group tiles by energy for per-group amax scaling
    if use_normalize and n_sb > 0:
        tile_mag = src.abs().mean(dim=1)
        n_out = ~outlier_tile_mask
        sorted_idx = tile_mag[n_out].argsort()
        n_valid = n_out.sum().item()
        if n_valid >= 3:
            n_per_group = max(1, n_valid // 3)
            g0_end = n_per_group
            g1_end = min(2 * n_per_group, n_valid)
            g0_indices = sorted_idx[:g0_end]
            g1_indices = sorted_idx[g0_end:g1_end]
            g2_indices = sorted_idx[g1_end:]
            g0_mean = tile_mag[g0_indices].mean().item()
            g1_mean = tile_mag[g1_indices].mean().item()
            g2_mean = tile_mag[g2_indices].mean().item()
            global_mag_mean = (g0_mean + g1_mean + g2_mean) / 3.0
            if global_mag_mean > 0:
                group_scales = [g0_mean / global_mag_mean, g1_mean / global_mag_mean, g2_mean / global_mag_mean]
            else:
                group_scales = [1.0, 1.0, 1.0]
            tile_group = torch.zeros(nt, dtype=torch.uint8, device=device)
            tile_group[g1_indices] = 1
            tile_group[g2_indices] = 2
        else:
            group_scales = [1.0, 1.0, 1.0]
            tile_group = torch.zeros(nt, dtype=torch.uint8, device=device)
    else:
        group_scales = [1.0, 1.0, 1.0]
        tile_group = torch.zeros(nt, dtype=torch.uint8, device=device)

# Phase 1: encode with effective_n
    recon = torch.zeros(nt, M, dtype=torch.float32, device=device)
    alphas = torch.zeros(max_N, nt, dtype=torch.uint8, device=device)
    signs = torch.zeros(max_N, nt, M, dtype=torch.int8, device=device)
    alphas_max = []
    n_planes_per_tile = ts // 2
    use_per_plane = plane_refine_steps > 0 and n_sb > 0
    plane_alphas = torch.zeros(max_N, nt, n_planes_per_tile, dtype=torch.uint8, device=device) if use_per_plane else None

    for step in range(effective_n):
        resid = src - recon
        ta = resid.abs().mean(dim=1)
        n_out = ~outlier_tile_mask
        amax = ta[n_out].max().item() if n_out.any() else 1e-4
        amax = max(amax, 1e-8); alphas_max.append(amax)
        for ti in range(nt):
            if outlier_tile_mask[ti]: continue
            B = torch.sign(resid[ti])
            gid = int(tile_group[ti].item())
            effective_amax = amax * group_scales[gid]
            if use_per_plane and step >= effective_n - plane_refine_steps:
                for pi in range(n_planes_per_tile):
                    d0, d1 = pi * 2, pi * 2 + 1
                    plane_elements = resid[ti].view(ts, ts)[:, d0:d1+1].reshape(-1)
                    pa = plane_elements.abs().mean().item()
                    paq = fixed_log_quantize_5bit(pa, lo=1e-7, hi=1.0)
                    pafp = fixed_log_dequantize_5bit(paq, lo=1e-7, hi=1.0)
                    plane_alphas[step, ti, pi] = paq
                    recon[ti].view(ts, ts)[:, d0:d1+1] += pafp * B.view(ts, ts)[:, d0:d1+1]
                alphas[step, ti] = 0
            else:
                aq = _quantize_alpha(ta[ti].item(), alpha_scheme, effective_amax, K_offset, log_min, log_max, nonlinear_gamma,
                                     bits=5 if step == 0 else 4)
                afp = _dequantize_alpha(aq, alpha_scheme, effective_amax, K_offset, log_min, log_max, nonlinear_gamma,
                                        bits=5 if step == 0 else 4)
                alphas[step, ti] = aq
                recon[ti] = recon[ti] + afp * B.float()
            signs[step, ti] = B.to(torch.int8)

    # Phase 2: relative error check
    per_tile_n = torch.full((nt,), effective_n, dtype=torch.uint8, device=device)
    fp16_reroute = torch.zeros(nt, dtype=torch.bool, device=device)
    boosted = []
    for ti in range(nt):
        if outlier_tile_mask[ti]: continue
        err = (src[ti] - recon[ti]).abs().max().item()
        if use_relative_threshold:
            mag = tiles_f[ti].abs().max().item()
            loc_fp16 = max(maxae_fp16_threshold, mag * 0.2) if mag > 1e-6 else maxae_fp16_threshold
            loc_boost = max(maxae_boost_threshold, mag * 0.1) if mag > 1e-6 else maxae_boost_threshold
        else:
            loc_fp16, loc_boost = maxae_fp16_threshold, maxae_boost_threshold
        if err > loc_fp16:
            fp16_reroute[ti] = True
        elif err > loc_boost and eff_boost > effective_n:
            boosted.append(ti)

    combined_fp16 = outlier_tile_mask | fp16_reroute

    # Phase 3: re-encode boosted
    if boosted:
        for ti in boosted:
            per_tile_n[ti] = eff_boost
            wh = recon[ti].clone()
            for step in range(effective_n, eff_boost):
                resid = src[ti] - wh
                B = torch.sign(resid)
                gid = int(tile_group[ti].item())
                all_a = []
                for t2 in range(nt):
                    if not combined_fp16[t2]:
                        gid2 = int(tile_group[t2].item())
                        all_a.append(src[t2].abs().mean().item() * group_scales[gid2])
                amax = max(all_a) if all_a else 1e-4
                amax = max(amax, 1e-8)
                alphas_max.append(amax)
                if step == effective_n:
                    while len(alphas_max) < eff_boost:
                        alphas_max.append(amax)
                effective_amax = amax * group_scales[gid]
                aq = _quantize_alpha(resid.abs().mean().item(), alpha_scheme, effective_amax, K_offset, log_min, log_max, nonlinear_gamma)
                afp = _dequantize_alpha(aq, alpha_scheme, effective_amax, K_offset, log_min, log_max, nonlinear_gamma)
                alphas[step, ti] = aq
                signs[step, ti] = B.to(torch.int8)
                wh = wh + afp * B.float()
                recon[ti] = wh

    while len(alphas_max) < max_N:
        alphas_max.append(alphas_max[-1] if alphas_max else 1.0)

    # Phase 4: Sparse residual patches — target worst-case elements for MaxAE reduction
    if use_normalize:
        recon_orig = recon * sigma.unsqueeze(1) + mu.unsqueeze(1)
    else:
        recon_orig = recon
    residual = tiles_f - recon_orig
    error_abs = residual.abs()
    threshold_val = max(error_abs.mean().item() * 3.0, 0.01)
    patch_count = torch.zeros(nt, dtype=torch.int32, device=device)
    patch_dim_weight = None
    if position_weight_power > 0 and M >= 4:
        max_deviation = min(position_weight_power * 0.25, 0.75)
        w = torch.linspace(1.0 - max_deviation, 1.0 + max_deviation, M, device=device).clamp(min=0.25)
        patch_dim_weight = w.unsqueeze(0)  # (1, M)
    patch_idx_list = []
    patch_val_list = []
    for ti in range(nt):
        if combined_fp16[ti]:
            continue
        err = error_abs[ti]
        if patch_dim_weight is not None:
            score = err * patch_dim_weight.squeeze(0)
            mask = score > threshold_val
        else:
            mask = err > threshold_val
        cnt = mask.sum().item()
        if cnt > 0 and cnt <= 15:
            patch_count[ti] = cnt
            idx = mask.nonzero(as_tuple=True)[0]
            vals = residual[ti, idx]
            for j in range(int(idx.shape[0])):
                patch_idx_list.append(int(idx[j].item()))
                patch_val_list.append(float(vals[j].item()))
    use_residual_patches = patch_count.sum().item() > 0
    if use_residual_patches:
        vals = [v for v in patch_val_list]
        bias = sum(vals) / len(vals)
        patch_val_list = [v - bias for v in patch_val_list]

    # Collect FP16 tiles (in superblock order for packed mode)
    def _collect_fp16():
        lst = []
        if packed and (nt_tok // 2) > 0 and (nt_dim // 2) > 0:
            for sr in range(nt_tok // 2):
                for sc in range(nt_dim // 2):
                    for s in range(4):
                        ti = (sr * 2) * nt_dim + (sc * 2) + (s & 1) + ((s >> 1) * nt_dim)
                        if combined_fp16[ti]:
                            lst.append(tiles[ti].to(torch.float16))
        else:
            for ti in range(nt):
                if combined_fp16[ti]:
                    lst.append(tiles[ti].to(torch.float16))
        return lst

    fp16_list = _collect_fp16()
    fp16_t = torch.stack(fp16_list) if fp16_list else None

    result = {
        "alphas_max_fp16": torch.tensor(alphas_max[:max_N], dtype=torch.float16, device=device),
        "outlier_mask": combined_fp16, "outlier_fp16": fp16_t,
        "orig_shape": (T, d_head),
        "tile_config": {"tile_size": ts, "n_steps": effective_n, "boost_n_steps": eff_boost,
                         "alpha_scheme": alpha_scheme, "K_offset": K_offset,
                         "log_min": log_min, "log_max": log_max, "nonlinear_gamma": nonlinear_gamma,
                         "position_weight_power": position_weight_power,
                         "plane_refine_n_steps": plane_refine_steps,
                         "plane_refine_lo": 1e-7, "plane_refine_hi": 1.0},
        "per_tile_n_steps": per_tile_n, "n_fp16_reroute": int(fp16_reroute.sum().item()), "n_boosted": len(boosted),
        "norm_mu": mu.to(torch.float16) if use_normalize else None,
        "norm_sigma": sigma.to(torch.float16) if use_normalize else None,
        "tile_group": tile_group,
        "group_scales": torch.tensor(group_scales, dtype=torch.float16, device=device) if use_normalize and n_sb > 0 else None,
        "residual_patch_count": patch_count if use_residual_patches else None,
        "residual_patch_idx": torch.tensor(patch_idx_list, dtype=torch.int16, device=device) if use_residual_patches else None,
        "residual_patch_val": torch.tensor(patch_val_list, dtype=torch.float16, device=device) if use_residual_patches else None,
    }

    # Pack
    if packed:
        n_sb = (nt_tok // 2) * (nt_dim // 2)
        if n_sb > 0:
            mr = 1 + max_N + 2
            meta = torch.zeros(n_sb, mr, dtype=torch.uint32, device=device)
            sflat, soff = [], [0]
            pflat, poff = [], [0] if use_per_plane else (None, None)
            n_planes_per_sb = 4 * n_planes_per_tile
            group_scales_fp16_bits = [int(torch.tensor([s], dtype=torch.float16).view(torch.uint16).item()) for s in group_scales]
            for sr in range(nt_tok // 2):
                for sc in range(nt_dim // 2):
                    sidx = sr * (nt_dim // 2) + sc
                    tiles4 = [(sr * 2) * nt_dim + (sc * 2) + s + (0 if s < 2 else nt_dim) for s in range(2)] + \
                             [(sr * 2 + 1) * nt_dim + (sc * 2) + s + (0 if s < 2 else nt_dim) for s in range(2)]
                    # Actually simpler:
                    sb_tiles = [
                        (sr * 2) * nt_dim + (sc * 2),
                        (sr * 2) * nt_dim + (sc * 2 + 1),
                        (sr * 2 + 1) * nt_dim + (sc * 2),
                        (sr * 2 + 1) * nt_dim + (sc * 2 + 1),
                    ]
                    mw = sum((1 << s) for s, ti in enumerate(sb_tiles) if combined_fp16[ti])
                    nsw = 0
                    gid_byte = 0
                    for s, ti in enumerate(sb_tiles):
                        if not combined_fp16[ti]:
                            steps_minus_2 = min(int(per_tile_n[ti].item()) - 2, 3)
                            nsw |= (steps_minus_2 & 3) << (s * 2)
                        gid_byte |= (int(tile_group[ti].item()) & 3) << (s * 2)
                    meta[sidx, 0] = mw | (nsw << 4) | (gid_byte << 12)
                    for step in range(max_N):
                        ar = 0
                        if step == 0:
                            for s, ti in enumerate(sb_tiles):
                                ar |= (int(alphas[step, ti].item()) & 0x1F) << (s * 5)
                        else:
                            for s, ti in enumerate(sb_tiles):
                                ar |= (int(alphas[step, ti].item()) & 0xF) << (s * 4)
                        meta[sidx, 1 + step] = ar
                    for s, ti in enumerate(sb_tiles):
                        if not combined_fp16[ti]:
                            an = int(per_tile_n[ti].item())
                            for step in range(an):
                                sw = sum((1 << i) for i in range(M) if signs[step, ti, i].item() > 0)
                                for chunk_i in range(0, M, 16):
                                    sflat.append((sw >> chunk_i) & 0xFFFF)
                            if use_per_plane:
                                for prstep in range(plane_refine_steps):
                                    sref = effective_n - plane_refine_steps + prstep
                                    for pi in range(n_planes_per_tile):
                                        pq = int(plane_alphas[sref, ti, pi].item()) & 0x1F
                                        pflat.append(pq)
                    meta[sidx, 1 + max_N] = group_scales_fp16_bits[0] | (group_scales_fp16_bits[1] << 16)
                    meta[sidx, 1 + max_N + 1] = group_scales_fp16_bits[2]
                    soff.append(len(sflat))
            result["meta_alpha_packed"] = meta
            result["signs_flat"] = torch.tensor(sflat, dtype=torch.uint16, device=device) if sflat else torch.zeros(0, dtype=torch.uint16, device=device)
            result["sign_offsets"] = torch.tensor(soff, dtype=torch.int32, device=device)
            if use_per_plane:
                result["plane_refine_alpha_flat"] = torch.tensor(pflat, dtype=torch.uint8, device=device) if pflat else torch.zeros(0, dtype=torch.uint8, device=device)
                result["plane_refine_alpha_offsets"] = torch.tensor(poff, dtype=torch.int32, device=device)
                result["plane_refine_n_steps"] = plane_refine_steps
        else:
            for ti in range(nt):
                if int(per_tile_n[ti].item()) >= 4 and not combined_fp16[ti]:
                    combined_fp16[ti] = True
            fp16_list2 = _collect_fp16()
            fp16_t = torch.stack(fp16_list2) if fp16_list2 else None
            result["outlier_mask"] = combined_fp16
            result["outlier_fp16"] = fp16_t
            n_sw = (M + 15) // 16
            meta = torch.zeros(nt, dtype=torch.uint16, device=device)
            signs_p = torch.zeros(max_N, nt, n_sw, dtype=torch.uint16, device=device)
            for ti in range(nt):
                word = 0
                if combined_fp16[ti]:
                    word |= 1 << 15
                else:
                    an = int(per_tile_n[ti].item())
                    word |= (an - 2) << 13
                    for step in range(min(an, 3)):
                        word |= (int(alphas[step, ti].item()) & 0xF) << (step * 4)
                    for step in range(an):
                        sw = sum((1 << i) for i in range(M) if signs[step, ti, i].item() > 0)
                        for ci in range(n_sw):
                            signs_p[step, ti, ci] = (sw >> (ci * 16)) & 0xFFFF
                meta[ti] = word
            result["meta_alpha_packed"] = meta
            result["signs_packed"] = signs_p
    else:
        result["alphas_q"] = alphas
        result["signs"] = signs

    return result


# ── Decoder ────────────────────────────────────────────────────────────────

def decode_4x4_matrix(encoded: dict) -> torch.Tensor:
    tc = encoded["tile_config"]
    ts = tc["tile_size"]
    M = ts * ts
    n_steps = tc.get("n_steps", 3)
    amax = encoded["alphas_max_fp16"]
    ofp16 = encoded.get("outlier_fp16")
    T, d_head = encoded["orig_shape"]
    device = amax.device
    nt_tok = T // ts; nt_dim = d_head // ts
    max_N = amax.shape[0]
    mg = tc.get("nonlinear_gamma", 0.55)

    def _deq(aq, s, bits=4, gid=-1):
        effective_amax = float(amax[s].item())
        if gid >= 0 and group_scales is not None and gid < len(group_scales):
            effective_amax *= group_scales[gid]
        return _dequantize_alpha(aq, tc["alpha_scheme"], effective_amax,
                                   tc.get("K_offset", 4.0), tc.get("log_min", 1e-4),
                                   tc.get("log_max", 10.0), mg, bits)

    packed = "meta_alpha_packed" in encoded
    sb_mode = packed and encoded["meta_alpha_packed"].ndim == 2
    norm_mu = encoded.get("norm_mu")
    norm_sigma = encoded.get("norm_sigma")
    group_scales = encoded.get("group_scales")
    if group_scales is not None:
        group_scales = group_scales.tolist()

    patch_count = encoded.get("residual_patch_count")
    patch_idx_flat = encoded.get("residual_patch_idx")
    patch_val_flat = encoded.get("residual_patch_val")
    patch_offset = None
    if patch_count is not None and patch_count.sum().item() > 0:
        patch_offset = torch.zeros(patch_count.shape[0] + 1, dtype=torch.int32, device=device)
        torch.cumsum(torch.cat([
            torch.zeros(1, dtype=torch.int32, device=device),
            patch_count.int()]), dim=0, out=patch_offset)

    plane_refine_n = tc.get("plane_refine_n_steps", 0)
    plane_refine_flat = encoded.get("plane_refine_alpha_flat")
    plane_refine_offsets = encoded.get("plane_refine_alpha_offsets")

    mat = torch.zeros(T, d_head, dtype=torch.float32, device=device)
    oidx = [0]

    if sb_mode:
        meta = encoded["meta_alpha_packed"]
        sflat = encoded.get("signs_flat", encoded.get("signs_packed"))
        soff = encoded.get("sign_offsets")
        n_sb_tok = nt_tok // 2; n_sb_dim = nt_dim // 2
        for sr in range(n_sb_tok):
            for sc in range(n_sb_dim):
                sidx = sr * n_sb_dim + sc
                if sidx >= meta.shape[0]: continue
                msb = meta[sidx]
                mw = int(msb[0].item())
                fm = mw & 0xF; nsw = (mw >> 4) & 0xFF; gid_byte = (mw >> 12) & 0xFF
                sb_tiles = [
                    (sr * 2) * nt_dim + (sc * 2),
                    (sr * 2) * nt_dim + (sc * 2 + 1),
                    (sr * 2 + 1) * nt_dim + (sc * 2),
                    (sr * 2 + 1) * nt_dim + (sc * 2 + 1),
                ]
                start = int(soff[sidx].item()) if soff is not None else 0
                end = int(soff[sidx + 1].item()) if soff is not None else sflat.shape[0]
                sp = 0
                pp = 0
                for s, ti in enumerate(sb_tiles):
                    if fm & (1 << s):
                        if oidx[0] < (ofp16.shape[0] if ofp16 is not None else 0):
                            tv = ofp16[oidx[0]].float().reshape(ts, ts)
                            oidx[0] += 1
                        else:
                            tv = torch.zeros(ts, ts, device=device)
                    else:
                        ns = 2 + ((nsw >> (s * 2)) & 3)
                        gid = (gid_byte >> (s * 2)) & 3
                        sign_chunks = M // 16
                        wh = torch.zeros(M, dtype=torch.float32, device=device)
                        for step in range(ns):
                            is_plane_step = (plane_refine_n > 0 and plane_refine_flat is not None and
                                step >= n_steps - plane_refine_n)
                            if is_plane_step:
                                p_step = step - (n_steps - plane_refine_n)
                                sw = 0
                                for ci in range(sign_chunks):
                                    if soff is not None:
                                        sw |= int(sflat[start + sp].item()) << (ci * 16)
                                    sp += 1
                                B = wh.new_zeros(M)
                                for i in range(M):
                                    B[i] = 1.0 if (sw >> i) & 1 else -1.0
                                n_planes_ts = ts // 2
                                for pi in range(n_planes_ts):
                                    aq = int(plane_refine_flat[pp].item()) & 0x1F; pp += 1
                                    afp = fixed_log_dequantize_5bit(aq, lo=1e-7, hi=1.0)
                                    d0, d1 = pi * 2, pi * 2 + 1
                                    wh.view(ts, ts)[:, d0] += afp * B.view(ts, ts)[:, d0]
                                    wh.view(ts, ts)[:, d1] += afp * B.view(ts, ts)[:, d1]
                            else:
                                if step == 0:
                                    aq = (int(msb[1].item()) >> (s * 5)) & 0x1F
                                    afp = _deq(aq, 0, bits=5, gid=gid)
                                else:
                                    aq = (int(msb[1 + step].item()) >> (s * 4)) & 0xF
                                    afp = _deq(aq, step, gid=gid)
                                sw = 0
                                for ci in range(sign_chunks):
                                    if soff is not None:
                                        sw |= int(sflat[start + sp].item()) << (ci * 16)
                                    else:
                                        sw |= int(sflat[step, sidx, s].item()) << (ci * 16)
                                    sp += 1
                                for i in range(M):
                                    wh[i] += afp * (1.0 if (sw >> i) & 1 else -1.0)
                        tv = wh.reshape(ts, ts)
                    if norm_mu is not None:
                        tv = tv * norm_sigma[ti].item() + norm_mu[ti].item()
                    if patch_offset is not None and ti < patch_count.shape[0]:
                        p_start = int(patch_offset[ti].item())
                        p_end = int(patch_offset[ti + 1].item())
                        for p in range(p_start, p_end):
                            elem_idx = int(patch_idx_flat[p].item())
                            pval = float(patch_val_flat[p].item())
                            pr, pc = elem_idx // ts, elem_idx % ts
                            if pr < tv.shape[0] and pc < tv.shape[1]:
                                tv[pr, pc] += pval
                    r0 = (sr * 2 + (s >> 1)) * ts; c0 = (sc * 2 + (s & 1)) * ts
                    mat[r0:min(r0 + ts, T), c0:min(c0 + ts, d_head)] = tv[:min(ts, T - r0), :min(ts, d_head - c0)]
        return mat

    # Per-tile fallback
    pt_n = encoded.get("per_tile_n_steps")
    for tr in range(nt_tok):
        for tc in range(nt_dim):
            ti = tr * nt_dim + tc
            r0 = tr * ts; c0 = tc * ts
            this_n = int(pt_n[ti].item()) if pt_n is not None else n_steps
            if packed:
                meta = encoded["meta_alpha_packed"]
                word = int(meta[ti].item())
                if word & (1 << 15):
                    if oidx[0] < (ofp16.shape[0] if ofp16 is not None else 0):
                        tv = ofp16[oidx[0]].float().reshape(ts, ts)
                        oidx[0] += 1
                    else:
                        tv = torch.zeros(ts, ts, device=device)
                else:
                    wh = torch.zeros(M, dtype=torch.float32, device=device)
                    for step in range(this_n):
                        aq = (word >> (step * 4)) & 0xF if step < 3 else 0
                        afp = _deq(aq, step)
                        sw = 0
                        n_sw_decode = (M + 15) // 16
                        sp_data = encoded.get("signs_packed", encoded.get("signs_flat"))
                        if sp_data.ndim == 3:
                            for ci in range(n_sw_decode):
                                sw |= int(sp_data[step, ti, ci].item()) << (ci * 16)
                        else:
                            sw = int(sp_data[step, ti].item()) if hasattr(sp_data, 'item') else 0
                        for i in range(M):
                            wh[i] += afp * (1.0 if (sw >> i) & 1 else -1.0)
                    tv = wh.reshape(ts, ts)
                    if norm_mu is not None:
                        tv = tv * norm_sigma[ti].item() + norm_mu[ti].item()
            else:
                aq = encoded["alphas_q"]; sg = encoded["signs"]
                om = encoded.get("outlier_mask", torch.zeros(aq.shape[1], dtype=torch.bool, device=device))
                if om[ti]:
                    if oidx[0] < (ofp16.shape[0] if ofp16 is not None else 0):
                        tv = ofp16[oidx[0]].float().reshape(ts, ts)
                        oidx[0] += 1
                    else:
                        tv = torch.zeros(ts, ts, device=device)
                else:
                    wh = torch.zeros(M, dtype=torch.float32, device=device)
                    for step in range(this_n):
                        afp = _deq(int(aq[step, ti].item()), step)
                        wh = wh + afp * sg[step, ti].float()
                    tv = wh.reshape(ts, ts)
                    if norm_mu is not None:
                        tv = tv * norm_sigma[ti].item() + norm_mu[ti].item()
            if patch_offset is not None and ti < patch_count.shape[0]:
                p_start = int(patch_offset[ti].item())
                p_end = int(patch_offset[ti + 1].item())
                for p in range(p_start, p_end):
                    elem_idx = int(patch_idx_flat[p].item())
                    pval = float(patch_val_flat[p].item())
                    pr, pc = elem_idx // ts, elem_idx % ts
                    if pr < tv.shape[0] and pc < tv.shape[1]:
                        tv[pr, pc] += pval
            mat[r0:min(r0 + ts, T), c0:min(c0 + ts, d_head)] = tv[:min(ts, T - r0), :min(ts, d_head - c0)]
    return mat


# ── Metrics ────────────────────────────────────────────────────────────────

def compute_4x4_metrics(mat_true: torch.Tensor, mat_recon: torch.Tensor) -> dict:
    tf = mat_true.float().flatten()
    rf = mat_recon.float().flatten()
    ae = (mat_true.float() - mat_recon.float()).abs()
    return {"cosine_similarity": F.cosine_similarity(tf.unsqueeze(0), rf.unsqueeze(0)).item(),
            "max_ae": ae.max().item(), "mse": F.mse_loss(rf, tf).item(),
            "per_dim_max_ae": ae.max(dim=0).values.clone()}
