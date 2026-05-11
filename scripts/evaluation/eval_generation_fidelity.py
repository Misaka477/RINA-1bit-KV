"""Generation Fidelity Evaluation — DS-KVCache vs Native FP16 text consistency.

Based on eval_padding_masking.py with:
  - JSON + CSV structured output
  - Per-prompt generated text saved to files
  - Optional KV fidelity measurement (--measure-kv)
  - Quality presets (--quality balanced|high)
  - Repetition score for n-gram loop detection
  - Logits-diff fork-point diagnostic (--logits-diff)

Run::
    python scripts/evaluation/eval_generation_fidelity.py
    python scripts/evaluation/eval_generation_fidelity.py --measure-kv --json-output results.json --csv-output results.csv --text-output-dir gen_texts
    python scripts/evaluation/eval_generation_fidelity.py --quality high --logits-diff --logits-output fork_logits.json
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from rina.config import DSKVCacheConfig
from rina.ds_kv_cache import DSKVCacheStore
from rina.model_wrapper import DSKVCacheModel, _past_get_kv

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
_logger = logging.getLogger("eval_gen_fidelity")

PROMPTS = [
    "The future of AI is",
    "Once upon a time in a galaxy far far away",
    "The capital of France is",
    "To solve the equation x^2 + 2x + 1 = 0 we",
    "Deep learning has revolutionized computer vision by",
    "The three laws of thermodynamics state that",
    "In quantum mechanics the Schr\u00f6dinger equation describes",
    "Python is a high-level programming language that",
]


@dataclass
class MaskRoute:
    name: str
    label: str
    adaptive_masking: bool = False
    use_mask_gating: bool = False
    extra_kwargs: dict = field(default_factory=dict)


def build_routes() -> List[MaskRoute]:
    return [
        MaskRoute("native",   "native",   adaptive_masking=False, use_mask_gating=False),
        MaskRoute("baseline", "baseline", adaptive_masking=False, use_mask_gating=False),
    ]


def make_config(route: MaskRoute, quality: str = "balanced", cross_token_group: int = 2,
                 n_steps_override: Optional[int] = None, n_steps_v_override: Optional[int] = None,
                 refresh_interval: int = 0, prefill_protected: bool = False,
                 bypass_adaptive: bool = False, bypass_threshold: float = 0.5,
                 prefill_system_protect_len: int = 0, prefill_tail_protect_len: int = 0,
                 prefill_n_steps: Optional[int] = None,
                 adaptive_residual: bool = False, adaptive_residual_threshold: float = 0.2,
                 adaptive_residual_n_steps: int = 1,
                 decode_protect_steps: int = 3,
                 decode_protect_layers: str = "all",
                 confidence_mask: bool = False, confidence_beta: float = 0.3,
                 attn_smoothing_alpha: float = 1.0,
                 k_outlier_dims: int = 0, k_outlier_compress_steps: int = 3,
                 k_bias_compensate: bool = False,
                 tile_size: int = 16,
                 alpha_scheme: str = "dynamic_log", alpha_K_offset: float = 4.0,
                 outlier_protect: bool = False, outlier_mad_threshold: float = 3.0) -> Optional[DSKVCacheConfig]:
    """Build DSKVCacheConfig from route + quality preset.

    quality="balanced": current production settings (n_steps=5, ctg=2)
    quality="high": higher-fidelity settings (n_steps=8, ctg=1, DCT, etc.)

    n_steps_override / n_steps_v_override: override n_steps in the config
    (useful for ablation sweeps).  n_steps_v_override defaults to n_steps_override.
    refresh_interval: periodic FP16 bypass interval (0=disabled).
    bypass_adaptive: if True, use L∞-based per-token bypass (Phase 1, deprecated).
    bypass_threshold: L∞ threshold for adaptive bypass (default 0.5).
    prefill_system_protect_len: pyramid prefill system prompt length (Phase 3).
    prefill_tail_protect_len: pyramid prefill tail length (Phase 3).
    prefill_n_steps: dual store prefill n_steps (Phase 4). None=disabled.
    adaptive_residual: if True, use 1-bit residual correction (replaces bypass).
    adaptive_residual_threshold: L∞ threshold for adaptive residual (default 0.2).
    adaptive_residual_n_steps: Σ-Δ steps for residual encoding (default 1).
    """
    if route.name == "native":
        return None

    _n = n_steps_override if n_steps_override is not None else (8 if quality == "high" else 5)
    _nv = n_steps_v_override if n_steps_v_override is not None else 2

    if quality == "high":
        return DSKVCacheConfig(
            n_steps=_n,
            n_steps_k=_n,
            n_steps_v=_nv,
            tile_size=tile_size,
            beta=0.10,
            use_noise_shaping=True,
            proj_rank=8,
            proj_beta=0.4,
            adaptive_eta=True,
            adaptive_n=False,
            use_differential=True,
            diff_strategy="residual",
            diff_residual_gamma=0.25,
            diff_residual_n_steps=4,
            v_orthogonal_transform=True,
            order2_gamma=0.15,
            cross_token_group=1,
            use_recon_weights=True,
            cross_head_error_share=True,
            adaptive_masking=True,
            mask_outlier_threshold=route.extra_kwargs.get("mask_outlier_threshold", 3.0),
            mask_n_steps_boost=route.extra_kwargs.get("mask_n_steps_boost", 0),
            mask_proj_beta_boost=route.extra_kwargs.get("mask_proj_beta_boost", 0.0),
            use_mask_gating=True,
            dynamic_tile_size=False,
            refresh_interval=refresh_interval,
            prefill_protected=prefill_protected,
            bypass_adaptive=bypass_adaptive,
            bypass_threshold=bypass_threshold,
            prefill_system_protect_len=prefill_system_protect_len,
            prefill_tail_protect_len=prefill_tail_protect_len,
            prefill_n_steps=prefill_n_steps,
            adaptive_residual=adaptive_residual,
            adaptive_residual_threshold=adaptive_residual_threshold,
            adaptive_residual_n_steps=adaptive_residual_n_steps,
            decode_protect_steps=decode_protect_steps,
            decode_protect_layers=decode_protect_layers,
            confidence_mask=confidence_mask,
            confidence_beta=confidence_beta,
            attn_smoothing_alpha=attn_smoothing_alpha,
            base_dtype="fp16",
            verbose=False,
            k_outlier_dims=k_outlier_dims,
            k_outlier_compress_steps=k_outlier_compress_steps,
            k_bias_compensate=k_bias_compensate,
            alpha_scheme=alpha_scheme,
            alpha_K_offset=alpha_K_offset,
            outlier_protect=outlier_protect,
            outlier_mad_threshold=outlier_mad_threshold,
        )
    else:
        # balanced (current defaults)
        return DSKVCacheConfig(
            n_steps=_n,
            n_steps_k=_n,
            n_steps_v=_nv,
            tile_size=tile_size,
            beta=0.12,
            use_noise_shaping=True,
            proj_rank=8,
            proj_beta=0.4,
            adaptive_eta=True,
            adaptive_n=False,
            use_differential=True,
            diff_strategy="residual",
            diff_residual_gamma=0.25,
            diff_residual_n_steps=2,
            v_orthogonal_transform=True,
            order2_gamma=0.15,
            cross_token_group=cross_token_group,
            use_recon_weights=False,
            cross_head_error_share=False,
            adaptive_masking=route.adaptive_masking,
            mask_outlier_threshold=route.extra_kwargs.get("mask_outlier_threshold", 3.0),
            mask_n_steps_boost=route.extra_kwargs.get("mask_n_steps_boost", 0),
            mask_proj_beta_boost=route.extra_kwargs.get("mask_proj_beta_boost", 0.0),
            use_mask_gating=route.use_mask_gating,
            refresh_interval=refresh_interval,
            prefill_protected=prefill_protected,
            bypass_adaptive=bypass_adaptive,
            bypass_threshold=bypass_threshold,
            prefill_system_protect_len=prefill_system_protect_len,
            prefill_tail_protect_len=prefill_tail_protect_len,
            prefill_n_steps=prefill_n_steps,
            adaptive_residual=adaptive_residual,
            adaptive_residual_threshold=adaptive_residual_threshold,
            adaptive_residual_n_steps=adaptive_residual_n_steps,
            decode_protect_steps=decode_protect_steps,
            decode_protect_layers=decode_protect_layers,
            confidence_mask=confidence_mask,
            confidence_beta=confidence_beta,
            attn_smoothing_alpha=attn_smoothing_alpha,
            base_dtype="fp16",
            verbose=False,
            k_outlier_dims=k_outlier_dims,
            k_outlier_compress_steps=k_outlier_compress_steps,
            k_bias_compensate=k_bias_compensate,
            alpha_scheme=alpha_scheme,
            alpha_K_offset=alpha_K_offset,
            outlier_protect=outlier_protect,
            outlier_mad_threshold=outlier_mad_threshold,
        )


def char_match_ratio(a: str, b: str) -> float:
    mn = min(len(a), len(b))
    mx = max(len(a), len(b))
    if mx == 0:
        return 1.0
    matches = sum(1 for i in range(mn) if a[i] == b[i])
    return matches / mx


def prefix_match_len(a: str, b: str) -> int:
    n = 0
    for ca, cb in zip(a, b):
        if ca == cb:
            n += 1
        else:
            break
    return n


def repetition_score(text: str, n: int = 4) -> float:
    """Detect repeated n-grams — high scores indicate repetition loops.

    Returns max_repeats / unique_ngrams, where max_repeats is the count
    of the most frequent n-gram.  A score of 1.0 means all unique n-grams
    are the same (extreme repetition).  A score near 0 means high diversity.
    """
    if len(text) < n:
        return 0.0
    ngrams = [text[i:i + n] for i in range(len(text) - n + 1)]
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    return max(counts.values()) / len(counts)


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    """Jensen-Shannon divergence between two probability distributions."""
    m = (p + q) / 2
    return float((F.kl_div(m.log(), p, reduction='sum') + F.kl_div(m.log(), q, reduction='sum')) / 2)


def compute_kv_fidelity(orig_k: torch.Tensor, orig_v: torch.Tensor, k_store: DSKVCacheStore, v_store: DSKVCacheStore, cfg: DSKVCacheConfig) -> dict:
    k_recon = k_store.reconstruct_all(cfg.tile_size, cfg.use_differential).float()
    v_recon = v_store.reconstruct_all(cfg.tile_size, cfg.use_differential).float()
    orig_k_f = orig_k.float()
    orig_v_f = orig_v.float()

    # K metrics
    k_mse = F.mse_loss(k_recon, orig_k_f).item()
    k_signal = (orig_k_f ** 2).mean().item()
    k_noise = ((orig_k_f - k_recon) ** 2).mean().item()
    k_snr = 10 * math.log10(k_signal / (k_noise + 1e-12))
    k_cos = F.cosine_similarity(k_recon.flatten().unsqueeze(0), orig_k_f.flatten().unsqueeze(0)).item()
    k_max_abs = (k_recon - orig_k_f).abs().max().item()

    # V metrics
    v_mse = F.mse_loss(v_recon, orig_v_f).item()
    v_signal = (orig_v_f ** 2).mean().item()
    v_noise = ((orig_v_f - v_recon) ** 2).mean().item()
    v_snr = 10 * math.log10(v_signal / (v_noise + 1e-12))
    v_cos = F.cosine_similarity(v_recon.flatten().unsqueeze(0), orig_v_f.flatten().unsqueeze(0)).item()
    v_max_abs = (v_recon - orig_v_f).abs().max().item()

    k_bytes = orig_k.element_size() * orig_k.numel()
    v_bytes = orig_v.element_size() * orig_v.numel()
    comp_ratio = (k_bytes + v_bytes) / (k_store.memory_bytes + v_store.memory_bytes + 1e-12)

    return {
        "k_cos_sim": float(k_cos),
        "k_mse": float(k_mse),
        "k_snr_db": float(k_snr),
        "k_max_abs_error": float(k_max_abs),
        "v_cos_sim": float(v_cos),
        "v_mse": float(v_mse),
        "v_snr_db": float(v_snr),
        "v_max_abs_error": float(v_max_abs),
        "compression_ratio": float(comp_ratio),
    }


def run_native_greedy_with_logits(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
) -> Tuple[str, List[int], List[torch.Tensor]]:
    """Run native FP16 greedy decode with per-step logit capture.

    Returns (text, token_ids, logits_list) where logits_list[i] is
    a float32 tensor of shape (vocab_size,) for step i.
    """
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    past = None
    native_logits: List[torch.Tensor] = []
    native_ids: List[int] = []

    with torch.no_grad():
        # Step 0: prefill
        out = model(input_ids=input_ids, use_cache=True, past_key_values=None)
        past = out.past_key_values
        step_logits = out.logits[0, -1, :].float()
        native_logits.append(step_logits)
        next_token_id = step_logits.argmax().item()
        native_ids.append(next_token_id)
        next_input_ids = torch.tensor([[next_token_id]], device=device)

        # Decode steps 1..max_new_tokens-1
        for step in range(1, max_new_tokens):
            out = model(input_ids=next_input_ids, use_cache=True, past_key_values=past)
            past = out.past_key_values
            step_logits = out.logits[0, -1, :].float()
            native_logits.append(step_logits)
            next_token_id = step_logits.argmax().item()
            native_ids.append(next_token_id)
            if next_token_id == tokenizer.eos_token_id:
                break
            next_input_ids = torch.tensor([[next_token_id]], device=device)

    generated_ids = input_ids[0].tolist() + native_ids
    gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return gen_text, native_ids, native_logits


def generate_with_kv_measurement(
    wrapper: DSKVCacheModel,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    return_logits: bool = False,
) -> tuple[str, dict]:
    device = wrapper.model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        output = wrapper.model(
            input_ids=input_ids,
            use_cache=True,
            past_key_values=None,
        )
    past_key_values = output.past_key_values

    # Save original K/V per layer per head before encoding
    orig_kv = {}
    for layer_idx in range(wrapper._num_layers):
        k_full, v_full = _past_get_kv(past_key_values, layer_idx)
        orig_kv[layer_idx] = []
        for h in range(wrapper._num_kv_heads):
            orig_kv[layer_idx].append((
                k_full[0, h].clone(),
                v_full[0, h].clone(),
            ))

    # Bulk-encode prefill into DS stores
    wrapper._bulk_encode_from_prefill(past_key_values, input_ids)

    # Compute KV fidelity per layer per head
    has_prefill = hasattr(wrapper, '_ds_prefill_layers') and wrapper._ds_prefill_layers
    kv_results = []
    for layer_idx in range(wrapper._num_layers):
        k_stores = wrapper._ds_layers[layer_idx][0]
        v_stores = wrapper._ds_layers[layer_idx][1]
        layer_cfg = wrapper.cfg.get_layer_config(layer_idx, wrapper._num_layers)

        for h in range(wrapper._num_kv_heads):
            orig_k, orig_v = orig_kv[layer_idx][h]
            if has_prefill and layer_idx < len(wrapper._ds_prefill_layers):
                pk_stores = wrapper._ds_prefill_layers[layer_idx][0]
                if h < len(pk_stores) and pk_stores[h] is not None:
                    pk = pk_stores[h]
                    pv = wrapper._ds_prefill_layers[layer_idx][1][h]
                    k_recon = pk.reconstruct_all(layer_cfg.tile_size, layer_cfg.use_differential)
                    v_recon = pv.reconstruct_all(layer_cfg.tile_size, layer_cfg.use_differential)
                    if k_stores[h].n_tokens > 0:
                        k_recon = torch.cat([k_recon, k_stores[h].reconstruct_all(layer_cfg.tile_size, layer_cfg.use_differential)], dim=0)
                        v_recon = torch.cat([v_recon, v_stores[h].reconstruct_all(layer_cfg.tile_size, layer_cfg.use_differential)], dim=0)
                    k_mse = F.mse_loss(k_recon.float(), orig_k.float()).item()
                    k_cos = F.cosine_similarity(k_recon.float().flatten().unsqueeze(0), orig_k.float().flatten().unsqueeze(0)).item()
                    v_mse = F.mse_loss(v_recon.float(), orig_v.float()).item()
                    v_cos = F.cosine_similarity(v_recon.float().flatten().unsqueeze(0), orig_v.float().flatten().unsqueeze(0)).item()
                    comp_ratio = (orig_k.element_size() * orig_k.numel() + orig_v.element_size() * orig_v.numel()) / (pk.memory_bytes + pv.memory_bytes + k_stores[h].memory_bytes + v_stores[h].memory_bytes + 1e-12)
                    fidelity = {"k_cos_sim": float(k_cos), "k_mse": float(k_mse), "k_snr_db": 0.0, "v_cos_sim": float(v_cos), "v_mse": float(v_mse), "v_snr_db": 0.0, "compression_ratio": float(comp_ratio)}
                else:
                    fidelity = compute_kv_fidelity(orig_k, orig_v, k_stores[h], v_stores[h], layer_cfg)
            else:
                fidelity = compute_kv_fidelity(orig_k, orig_v, k_stores[h], v_stores[h], layer_cfg)
            fidelity["layer"] = layer_idx
            fidelity["head"] = h
            kv_results.append(fidelity)

    # Aggregate KV fidelity
    all_k_cos = [r["k_cos_sim"] for r in kv_results]
    all_v_cos = [r["v_cos_sim"] for r in kv_results]
    all_k_snr = [r["k_snr_db"] for r in kv_results]
    all_v_snr = [r["v_snr_db"] for r in kv_results]
    all_cr = [r["compression_ratio"] for r in kv_results]

    kv_summary = {
        "avg_k_cos_sim": float(sum(all_k_cos) / len(all_k_cos)),
        "avg_v_cos_sim": float(sum(all_v_cos) / len(all_v_cos)),
        "avg_k_snr_db": float(sum(all_k_snr) / len(all_k_snr)),
        "avg_v_snr_db": float(sum(all_v_snr) / len(all_v_snr)),
        "avg_compression_ratio": float(sum(all_cr) / len(all_cr)),
        "per_layer_per_head": kv_results,
    }

    # Manual decode loop
    past = wrapper._build_past_from_ds()

    generated_ids = input_ids[0].tolist()
    first_token_id = output.logits[0, -1, :].argmax().item()
    generated_ids.append(first_token_id)

    ds_logits: List[torch.Tensor] = []
    ds_ids: List[int] = [first_token_id]
    if return_logits:
        ds_logits.append(output.logits[0, -1, :].float().clone())

    for step in range(1, max_new_tokens):
        last_token = torch.tensor([[generated_ids[-1]]], device=device)

        with torch.no_grad():
            output = wrapper.model(
                input_ids=last_token,
                use_cache=True,
                past_key_values=past,
            )

        wrapper._append_incremental(output.past_key_values, new_token_idx=-1, decode_step=step - 1)
        past = wrapper._build_past_from_ds()

        next_token_id = output.logits[0, -1, :].argmax().item()
        generated_ids.append(next_token_id)
        ds_ids.append(next_token_id)

        if return_logits:
            ds_logits.append(output.logits[0, -1, :].float().clone())

        if next_token_id == tokenizer.eos_token_id:
            break

    gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    if return_logits:
        kv_summary["ds_logits"] = ds_logits
        kv_summary["ds_token_ids"] = ds_ids

    return gen_text, kv_summary


def find_fork_point(
    native_logits: List[torch.Tensor],
    ds_logits: List[torch.Tensor],
    tokenizer,
    top_k: int = 10,
) -> dict:
    """Find first step where argmax differs, compute JS-div timeline, top-k table."""
    num_steps = min(len(native_logits), len(ds_logits))
    step_logits_data = []
    fork_step = None
    fork_native_id = None
    fork_ds_id = None

    for step in range(num_steps):
        native_l = native_logits[step]
        ds_l = ds_logits[step]

        native_prob = F.softmax(native_l, dim=-1)
        ds_prob = F.softmax(ds_l, dim=-1)
        js_div = js_divergence(native_prob, ds_prob)

        nat_id = int(native_l.argmax().item())
        ds_id = int(ds_l.argmax().item())

        entry = {
            "step": step,
            "native_token": nat_id,
            "ds_token": ds_id,
            "js_div": round(js_div, 6),
        }

        if fork_step is None and nat_id != ds_id:
            fork_step = step
            fork_native_id = nat_id
            fork_ds_id = ds_id

            # Build top-k comparison at fork point
            topk_data = []
            # Take union of top-k from both distributions
            combined_topk = set()
            combined_topk.update(native_l.topk(top_k).indices.tolist())
            combined_topk.update(ds_l.topk(top_k).indices.tolist())

            rows = []
            for tid in combined_topk:
                rows.append({
                    "token_id": tid,
                    "token": tokenizer.decode([tid]),
                    "native_logit": round(float(native_l[tid]), 4),
                    "ds_logit": round(float(ds_l[tid]), 4),
                    "diff": round(float(native_l[tid] - ds_l[tid]), 4),
                    "native_prob": round(float(native_prob[tid]), 4),
                    "ds_prob": round(float(ds_prob[tid]), 4),
                })
            rows.sort(key=lambda r: r["native_logit"], reverse=True)
            entry["topk"] = rows[:top_k]

            _logger.info("=" * 80)
            _logger.info("Fork at step %d: native=%d (%s) ds=%d (%s)",
                         fork_step, fork_native_id, tokenizer.decode([fork_native_id]),
                         fork_ds_id, tokenizer.decode([fork_ds_id]))
            _logger.info("")
            _logger.info(f"{'Token':<20s} {'Native logit':>12s} {'DS logit':>12s} {'Diff':>10s} {'Native prob':>12s} {'DS prob':>12s}")
            _logger.info("-" * 80)
            for row in rows[:top_k]:
                _logger.info(
                    f"{row['token']:<20s} {row['native_logit']:>12.4f} {row['ds_logit']:>12.4f} "
                    f"{row['diff']:>10.4f} {row['native_prob']:>12.4f} {row['ds_prob']:>12.4f}"
                )
            _logger.info("=" * 80)

        step_logits_data.append(entry)

    # Print JS divergence timeline (summary)
    js_values = [e["js_div"] for e in step_logits_data]
    _logger.info("JS divergence timeline (min/mean/max): %.6f / %.6f / %.6f",
                 min(js_values) if js_values else 0,
                 sum(js_values) / len(js_values) if js_values else 0,
                 max(js_values) if js_values else 0)
    if fork_step is not None:
        _logger.info("JS divergence at fork step %d: %.6f", fork_step, js_values[fork_step])

    return {
        "fork_step": fork_step,
        "native_fork_token": fork_native_id,
        "ds_fork_token": fork_ds_id,
        "num_compared_steps": num_steps,
        "js_div_timeline": js_values,
        "js_div_mean": round(sum(js_values) / len(js_values), 6) if js_values else 0,
        "step_logits": step_logits_data,
    }


def save_text_outputs(
    text_dir: Path,
    route_name: str,
    prompt_idx: int,
    gen_text: str,
    native_text: str,
):
    route_dir = text_dir / route_name
    route_dir.mkdir(parents=True, exist_ok=True)
    filepath = route_dir / f"prompt_{prompt_idx}.txt"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(gen_text)

    if route_name != "native" and native_text:
        native_dir = text_dir / "native"
        native_dir.mkdir(parents=True, exist_ok=True)
        native_file = native_dir / f"prompt_{prompt_idx}.txt"
        with open(native_file, "w", encoding="utf-8") as f:
            f.write(native_text)


def main():
    p = argparse.ArgumentParser(description="Generation fidelity evaluation with structured output")
    p.add_argument("--model", type=str,
                   default="D:/Software_Development/Project/models/Llama-3.2-1B")
    p.add_argument("--max-tokens", type=int, default=50)
    p.add_argument("--prompts", type=str, nargs="*", default=None,
                   help="Override prompt list")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--cross-token-group", type=int, default=2)
    p.add_argument("--quality", type=str, default="balanced", choices=["balanced", "high"],
                   help="Quality preset: balanced (current) or high (better KV fidelity)")
    p.add_argument("--n-steps", type=int, default=None,
                   help="Override n_steps (both K and V) in the config")
    p.add_argument("--n-steps-v", type=int, default=None,
                   help="Override n_steps_v separately from n_steps")
    p.add_argument("--refresh-interval", type=int, default=0,
                   help="Periodic FP16 bypass interval (0=disabled). Every N-th decode token stores FP16 in bypass map.")
    p.add_argument("--prefill-protected", action="store_true",
                   help="Store all prefill K/V as FP16 (bypass Σ-Δ encoding), giving decode zero-error start.")
    p.add_argument("--bypass-adaptive", action="store_true",
                   help="Enable adaptive bypass: L∞-based per-token bypass (Phase 1).")
    p.add_argument("--bypass-threshold", type=float, default=0.5,
                   help="L∞ threshold for adaptive bypass (default 0.5). Lower = more bypasses.")
    p.add_argument("--adaptive-residual", action="store_true",
                   help="Enable adaptive 1-bit residual correction. Encodes tile reconstruction error "
                        "at ~0.25 bits/element, only for tiles exceeding L∞ threshold.")
    p.add_argument("--adaptive-residual-threshold", type=float, default=0.2,
                   help="L∞ threshold for adaptive residual (default 0.2). Lower = more corrections.")
    p.add_argument("--adaptive-residual-n-steps", type=int, default=1,
                   help="Σ-Δ steps for adaptive residual (default 1). 1=4 packed-bits per tile.")
    p.add_argument("--prefill-system-protect", type=int, default=0,
                   help="Pyramid prefill: number of initial prompt tokens stored at full precision (default 0=disabled).")
    p.add_argument("--prefill-tail-protect", type=int, default=0,
                   help="Pyramid prefill: number of final prompt tokens stored at full precision (default 0=disabled).")
    p.add_argument("--prefill-n-steps", type=int, default=None,
                   help="Dual store: prefill n_steps (e.g., 8). Decode uses global n_steps. None=disabled.")
    p.add_argument("--decode-protect-steps", type=int, default=3,
                   help="Number of initial decode steps stored at FP16 precision (Key Position Protection). "
                        "Default 3. Higher → more stable but slightly less CR.")
    p.add_argument("--decode-protect-layers", type=str, default="last_4",
                   help="Which layers to protect: 'all', 'last_4', 'first_last', 'none'. Default: last_4.")
    p.add_argument("--decode-gap-threshold", type=float, default=0.5,
                   help="Logits Top-2 gap threshold for P1 forking protection. "
                        "When gap < threshold, triggers extra 1-bit sign residual. Default: 0.5.")
    p.add_argument("--residual-cos-threshold", type=float, default=0.9999,
                   help="Cosine similarity threshold for 1-bit sign residual skip. "
                        "When cos(tile, primary_full) > threshold, skip 1-bit sign encoding. "
                        "0.9999 = very conservative (only skips near-perfect direction). Default: 0.9999.")
    p.add_argument("--residual-n-steps", type=int, default=1,
                   help="Number of Σ-Δ steps for the 1-bit sign residual encoding. "
                        "Higher = more precise correction but higher bit cost. Default: 1.")
    p.add_argument("--encode-mode", type=str, default="sigma_delta", choices=["sigma_delta", "matching_pursuit"],
                   help="Σ-Δ encoding mode: 'sigma_delta' (default, with momentum/integrator) or "
                        "'matching_pursuit' (no momentum, no structured noise).")
    p.add_argument("--confidence-mask", action="store_true",
                   help="Enable confidence-masked attention (Stage 3). "
                        "Applies per-position penalty to attention scores based on KV encoding quality.")
    p.add_argument("--confidence-beta", type=float, default=0.3,
                   help="Confidence mask penalty scaling factor. 0.3 = moderate, 0.7 = aggressive. Default: 0.3.")
    p.add_argument("--attn-smoothing-alpha", type=float, default=1.0,
                   help="Temporal attention smoothing factor (Stage 4). 1.0 = off. "
                        "0.9 = 90%% current + 10%% previous attention distribution. Default: 1.0.")
    p.add_argument("--use-fwht", action="store_true",
                   help="Apply FWHT before Σ-Δ encoding. Distributes quantization error uniformly "
                        "across frequency components. Zero bit-cost.")
    p.add_argument("--adaptive-n", action="store_true",
                   help="Enable per-tile adaptive Σ-Δ step count. High-energy tiles get more bases.")
    p.add_argument("--beta-decay-start", type=float, default=None,
                   help="Initial beta for decode steps. If set, beta decays from this value to "
                        "--beta-decay-end over --beta-decay-tokens decode steps.")
    p.add_argument("--beta-decay-end", type=float, default=0.02,
                   help="Final beta after decay completes (default 0.02).")
    p.add_argument("--beta-decay-tokens", type=int, default=256,
                   help="Number of decode steps to decay beta over (default 256).")
    p.add_argument("--k-outlier-dims", type=int, default=0,
                   help="Phase 2d: K outlier dims protected at FP16 (0=disabled, 2=recommended for 128 dims).")
    p.add_argument("--k-outlier-compress-steps", type=int, default=3,
                   help="Phase 2d: n_steps for non-outlier K dims (default 3).")
    p.add_argument("--k-bias-compensate", action="store_true",
                    help="Phase 2d: Compute per-head K/V bias during prefill and compensate in reconstruction.")
    p.add_argument("--tile-size", type=int, default=16,
                    help="Tile size for encoding. 16 = Phase 2d. 4 = Phase 2e 4x4 tile. Default: 16.")
    p.add_argument("--alpha-scheme", type=str, default="dynamic_log",
                    choices=["dynamic_log", "nonlinear_log", "fixed_log", "linear"],
                    help="Phase 2e: α quantization scheme. Default: dynamic_log.")
    p.add_argument("--alpha-K-offset", type=float, default=4.0,
                    help="Phase 2e: dynamic log precision (3.0-5.0). Default: 4.0.")
    p.add_argument("--outlier-protect", action="store_true",
                    help="Phase 2e/2d: Enable outlier tile FP16 protection.")
    p.add_argument("--outlier-mad-threshold", type=float, default=3.0,
                    help="Phase 2e: MAD multiplier for outlier detection (2.5=tight, 3.0=default).")
    p.add_argument("--json-output", type=str, default="eval_gen_fidelity_results.json",
                   help="JSON output file")
    p.add_argument("--csv-output", type=str, default=None,
                   help="CSV output file (optional)")
    p.add_argument("--text-output-dir", type=str, default=None,
                   help="Directory for per-prompt text files (optional)")
    p.add_argument("--measure-kv", action="store_true",
                   help="Enable KV fidelity measurement (slower but richer)")
    p.add_argument("--repetition-threshold", type=int, default=3,
                   help="Minimum n-gram count to flag as repetition (default: 3)")
    p.add_argument("--logits-diff", action="store_true",
                   help="Capture per-step logits, print fork-point analysis")
    p.add_argument("--logits-output", type=str, default=None,
                   help="JSON path for per-step logits data (requires --logits-diff)")
    args = p.parse_args()

    # Auto-enable --measure-kv when --quality high
    if args.quality == "high" and not args.measure_kv:
        args.measure_kv = True
        _logger.info("--quality high → auto-enabling --measure-kv")

    # Force --measure-kv when --logits-diff (DS needs manual decode loop)
    if args.logits_diff and not args.measure_kv:
        args.measure_kv = True
        _logger.info("--logits-diff → auto-enabling --measure-kv for DS logit capture")

    prompts = args.prompts if args.prompts else PROMPTS
    routes = build_routes()
    ctg = args.cross_token_group

    _logger.info(f"Model: {args.model}")
    _logger.info(f"Max tokens: {args.max_tokens}")
    _logger.info(f"Quality: {args.quality}")
    _logger.info(f"Prompts: {len(prompts)}")
    _logger.info(f"cross_token_group: {ctg}")
    _logger.info(f"Measure KV: {args.measure_kv}")
    _logger.info(f"Logits diff: {args.logits_diff}")
    _logger.info(f"Repetition threshold: {args.repetition_threshold}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _logger.info("Loading model (shared across all routes) ...")
    shared_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    shared_model.eval()

    # Native FP16 baseline
    native_outputs: Dict[str, str] = {}
    native_logits_data: Dict[str, dict] = {}  # prompt -> {token_ids, logits}
    _logger.info("Native baseline (greedy) ...")

    if args.logits_diff:
        # Manual greedy loop with logit capture
        for prompt in prompts:
            _logger.info(f"  Native: \"{prompt[:40]}...\"")
            gen_text, nat_ids, nat_logits = run_native_greedy_with_logits(
                shared_model, tokenizer, prompt, args.max_tokens,
            )
            native_outputs[prompt] = gen_text
            native_logits_data[prompt] = {
                "token_ids": nat_ids,
                "logits": nat_logits,
            }
    else:
        for prompt in prompts:
            _logger.info(f"  Native: \"{prompt[:40]}...\"")
            inputs = tokenizer(prompt, return_tensors="pt").to(shared_model.device)
            with torch.no_grad():
                out = shared_model.generate(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            native_outputs[prompt] = text

    if args.text_output_dir:
        for i, prompt in enumerate(prompts):
            save_text_outputs(Path(args.text_output_dir), "native", i, native_outputs[prompt], native_outputs[prompt])

    _logger.info("Native baseline complete.")

    # DS routes
    results: List[dict] = []
    ds_routes = [r for r in routes if r.name != "native"]
    text_dir = Path(args.text_output_dir) if args.text_output_dir else None

    # Stores logits data for --logits-output
    all_logits_output: List[dict] = []

    for ri, route in enumerate(ds_routes):
        cfg = make_config(route, quality=args.quality, cross_token_group=ctg,
                          n_steps_override=args.n_steps, n_steps_v_override=args.n_steps_v,
                          refresh_interval=args.refresh_interval,
                          prefill_protected=args.prefill_protected,
                          bypass_adaptive=args.bypass_adaptive,
                          bypass_threshold=args.bypass_threshold,
                          prefill_system_protect_len=args.prefill_system_protect,
                          prefill_tail_protect_len=args.prefill_tail_protect,
                          prefill_n_steps=args.prefill_n_steps,
                          adaptive_residual=args.adaptive_residual,
                          adaptive_residual_threshold=args.adaptive_residual_threshold,
                          adaptive_residual_n_steps=args.adaptive_residual_n_steps,
                          decode_protect_steps=args.decode_protect_steps,
                          decode_protect_layers=args.decode_protect_layers,
                          confidence_mask=args.confidence_mask,
                          confidence_beta=args.confidence_beta,
                          attn_smoothing_alpha=args.attn_smoothing_alpha,
                          k_outlier_dims=args.k_outlier_dims,
                          k_outlier_compress_steps=args.k_outlier_compress_steps,
                          k_bias_compensate=args.k_bias_compensate,
                          tile_size=args.tile_size,
                          alpha_scheme=args.alpha_scheme,
                          alpha_K_offset=args.alpha_K_offset,
                          outlier_protect=args.outlier_protect,
                          outlier_mad_threshold=args.outlier_mad_threshold)
        label = route.label

        _logger.info(f"[{ri+1}/{len(ds_routes)}] {label}: "
                     f"adaptive_masking={route.adaptive_masking} "
                     f"use_mask_gating={route.use_mask_gating}")

        t0 = time.time()
        wrapper = None
        try:
            wrapper = DSKVCacheModel(shared_model, tokenizer, cfg=cfg)

            # Register Stage 3-4 attention hooks (confidence mask + smoothing)
            if cfg.confidence_mask or cfg.attn_smoothing_alpha < 1.0:
                wrapper._register_stage_hooks()

            for pi, prompt in enumerate(prompts):
                _logger.info(f"    Prompt: \"{prompt[:40]}...\"")

                kv_fidelity = None
                ds_logits_for_comparison = None

                if args.measure_kv:
                    gen_text, kv_fidelity = generate_with_kv_measurement(
                        wrapper, tokenizer, prompt, args.max_tokens,
                        return_logits=args.logits_diff,
                    )
                    if args.logits_diff and kv_fidelity is not None:
                        ds_logits_for_comparison = {
                            "token_ids": kv_fidelity.pop("ds_token_ids", []),
                            "logits": kv_fidelity.pop("ds_logits", []),
                        }
                else:
                    gen_text = wrapper.generate(
                        prompt,
                        max_new_tokens=args.max_tokens,
                        do_sample=False,
                    )

                native_text = native_outputs.get(prompt, "")
                prompt_only = prompt
                gen_new = gen_text[len(prompt_only):] if gen_text.startswith(prompt_only) else gen_text
                nat_new = native_text[len(prompt_only):] if native_text.startswith(prompt_only) else native_text

                rep_score = repetition_score(gen_new, n=args.repetition_threshold)

                result = {
                    "route": label,
                    "prompt": prompt_only[:40],
                    "char_match": round(char_match_ratio(gen_new, nat_new), 4) if native_text else None,
                    "prefix_match": prefix_match_len(gen_new, nat_new) if native_text else None,
                    "repetition_score": round(rep_score, 4),
                    "time_s": round(time.time() - t0, 1),
                    "generated_text": gen_text,
                    "native_text": native_text,
                }
                if kv_fidelity is not None:
                    result["kv_cos_sim_k"] = kv_fidelity["avg_k_cos_sim"]
                    result["kv_cos_sim_v"] = kv_fidelity["avg_v_cos_sim"]
                    result["kv_snr_db_k"] = kv_fidelity["avg_k_snr_db"]
                    result["kv_snr_db_v"] = kv_fidelity["avg_v_snr_db"]
                    result["kv_fidelity"] = kv_fidelity

                # Fork-point analysis
                if args.logits_diff and ds_logits_for_comparison and prompt in native_logits_data:
                    nat_data = native_logits_data[prompt]
                    fork_info = find_fork_point(
                        nat_data["logits"], ds_logits_for_comparison["logits"],
                        tokenizer, top_k=10,
                    )
                    result["fork_point"] = fork_info
                    all_logits_output.append({
                        "prompt": prompt,
                        "route": label,
                        "fork_step": fork_info["fork_step"],
                        "js_div_mean": fork_info["js_div_mean"],
                        "step_logits": fork_info["step_logits"],
                    })

                results.append(result)

                if text_dir:
                    save_text_outputs(text_dir, label, pi, gen_text, native_text)

        except Exception as e:
            _logger.error(f"Route {label} FAILED: {e}", exc_info=True)
            for pi, prompt in enumerate(prompts):
                results.append({
                    "route": label,
                    "prompt": prompt[:40],
                    "char_match": None,
                    "prefix_match": None,
                    "repetition_score": 0.0,
                    "time_s": round(time.time() - t0, 1),
                    "generated_text": f"[ROUTE ERROR: {e}]",
                    "native_text": "",
                })
        finally:
            if wrapper is not None:
                try:
                    wrapper._unregister_stage_hooks()
                except Exception:
                    pass
                del wrapper

    # Native entries
    for pi, prompt in enumerate(prompts):
        native_text = native_outputs.get(prompt, "")
        prompt_only = prompt
        nat_new = native_text[len(prompt_only):] if native_text.startswith(prompt_only) else native_text
        rep_score = repetition_score(nat_new, n=args.repetition_threshold)
        results.append({
            "route": "native",
            "prompt": prompt_only[:40],
            "char_match": 1.0,
            "prefix_match": None,
            "repetition_score": round(rep_score, 4),
            "time_s": 0.0,
            "generated_text": native_text,
            "native_text": native_text,
        })

    del shared_model
    torch.cuda.empty_cache()

    # Save logits JSON if requested
    if args.logits_diff and args.logits_output:
        logits_path = Path(args.logits_output)
        logits_path.parent.mkdir(parents=True, exist_ok=True)
        with open(logits_path, "w", encoding="utf-8") as f:
            json.dump(all_logits_output, f, indent=2, ensure_ascii=False)
        _logger.info(f"Logits data saved to {logits_path}")

    # Build route-level summaries
    route_summaries = {}
    for label in ["native", "baseline", "baseline_mask", "r1", "r1_mask"]:
        route_results = [r for r in results if r["route"] == label]
        if not route_results:
            continue

        char_vals = [r["char_match"] for r in route_results if r["char_match"] is not None]
        prefix_vals = [r["prefix_match"] for r in route_results if r["prefix_match"] is not None]
        rep_vals = [r["repetition_score"] for r in route_results if r["repetition_score"] is not None]
        time_vals = [r["time_s"] for r in route_results]

        summary = {
            "avg_char_match": round(sum(char_vals) / len(char_vals), 4) if char_vals else None,
            "avg_prefix_match": round(sum(prefix_vals) / len(prefix_vals), 1) if prefix_vals else None,
            "avg_repetition_score": round(sum(rep_vals) / len(rep_vals), 4) if rep_vals else None,
            "avg_time_s": round(sum(time_vals) / len(time_vals), 1) if time_vals else None,
        }

        if args.measure_kv and label != "native":
            kv_cos_k_vals = [r.get("kv_cos_sim_k") for r in route_results if r.get("kv_cos_sim_k") is not None]
            kv_cos_v_vals = [r.get("kv_cos_sim_v") for r in route_results if r.get("kv_cos_sim_v") is not None]
            if kv_cos_k_vals:
                summary["kv_fidelity"] = {
                    "avg_cos_sim_k": round(sum(kv_cos_k_vals) / len(kv_cos_k_vals), 6),
                    "avg_cos_sim_v": round(sum(kv_cos_v_vals) / len(kv_cos_v_vals), 6),
                }

        # Fork-point summary
        fork_vals = [r.get("fork_point", {}).get("fork_step") for r in route_results
                     if r.get("fork_point", {}).get("fork_step") is not None]
        if fork_vals:
            summary["fork_step_range"] = [min(fork_vals), max(fork_vals)]
            summary["avg_fork_step"] = round(sum(fork_vals) / len(fork_vals), 1)
            summary["fork_found_in"] = f"{len(fork_vals)}/{len(route_results)} prompts"

        route_summaries[label] = summary

    # Build full report
    config_dict = make_config(build_routes()[1], quality=args.quality, cross_token_group=ctg,
                              n_steps_override=args.n_steps, n_steps_v_override=args.n_steps_v,
                              refresh_interval=args.refresh_interval,
                              prefill_protected=args.prefill_protected,
                              bypass_adaptive=args.bypass_adaptive,
                              bypass_threshold=args.bypass_threshold,
                              prefill_system_protect_len=args.prefill_system_protect,
                              prefill_tail_protect_len=args.prefill_tail_protect,
                              prefill_n_steps=args.prefill_n_steps,
                              adaptive_residual=args.adaptive_residual,
                              adaptive_residual_threshold=args.adaptive_residual_threshold,
                              adaptive_residual_n_steps=args.adaptive_residual_n_steps,
                              decode_protect_steps=args.decode_protect_steps,
                              decode_protect_layers=args.decode_protect_layers,
                              confidence_mask=args.confidence_mask,
                              confidence_beta=args.confidence_beta,
                              attn_smoothing_alpha=args.attn_smoothing_alpha,
                              k_outlier_dims=args.k_outlier_dims,
                              k_outlier_compress_steps=args.k_outlier_compress_steps,
                              k_bias_compensate=args.k_bias_compensate,
                              tile_size=args.tile_size,
                              alpha_scheme=args.alpha_scheme,
                              alpha_K_offset=args.alpha_K_offset,
                              outlier_protect=args.outlier_protect,
                              outlier_mad_threshold=args.outlier_mad_threshold)
    report = {
        "config": {
            "model": args.model,
            "max_tokens": args.max_tokens,
            "quality": args.quality,
            "cross_token_group": ctg,
            "measure_kv": args.measure_kv,
            "logits_diff": args.logits_diff,
            "repetition_threshold": args.repetition_threshold,
            "tile_size": args.tile_size,
            "alpha_scheme": args.alpha_scheme,
            "outlier_protect": args.outlier_protect,
            "ds_config": config_dict.to_dict() if config_dict else None,
        },
        "route_results": route_summaries,
        "per_prompt": results,
    }

    # Save JSON
    json_path = Path(args.json_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    _logger.info(f"JSON saved to {json_path}")

    # Save CSV
    if args.csv_output:
        csv_path = Path(args.csv_output)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["route", "prompt", "char_match", "prefix_match", "repetition_score", "time_s"]
        if args.measure_kv:
            fieldnames.extend(["kv_cos_sim_k", "kv_cos_sim_v", "kv_snr_db_k", "kv_snr_db_v"])
        if args.logits_diff:
            fieldnames.append("fork_step")
        fieldnames.extend(["generated_text", "native_text"])

        # Flatten fork_step for CSV
        csv_results = []
        for r in results:
            row = dict(r)
            if args.logits_diff:
                fork_info = r.get("fork_point", {})
                row["fork_step"] = fork_info.get("fork_step", "")
            csv_results.append(row)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(csv_results)
        _logger.info(f"CSV saved to {csv_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print("=== Generation Fidelity Evaluation ===")
    print(f"Quality: {args.quality}")
    print(f"{'Route':<18s} {'char_match':>10s} {'prefix_match':>12s} {'rep_score':>9s} {'time':>8s}")
    if args.measure_kv:
        print(f"{'':18s} {'K CosSim':>10s} {'V CosSim':>10s}")
    print(f"{'-'*60}")

    for label in ["native", "baseline", "baseline_mask", "r1", "r1_mask"]:
        s = route_summaries.get(label)
        if not s:
            continue
        cm = s.get('avg_char_match')
        line = f"{label:<18s} {cm:>10.4f} " if cm is not None else f"{label:<18s} {'N/A':>10s} "
        if s['avg_prefix_match'] is not None:
            line += f"{s['avg_prefix_match']:>12.1f} "
        else:
            line += f"{'':>12s} "
        rs = s.get('avg_repetition_score', 0)
        line += f"{rs:>9.4f} {s['avg_time_s']:>8.1f}"
        print(line)
        if args.measure_kv and s.get("kv_fidelity"):
            kv = s["kv_fidelity"]
            print(f"{'':18s} {kv['avg_cos_sim_k']:>10.6f} {kv['avg_cos_sim_v']:>10.6f}")
        if args.logits_diff and s.get("avg_fork_step") is not None:
            print(f"{'':18s} Fork step: avg={s['avg_fork_step']} range={s['fork_step_range']} found={s['fork_found_in']}")

    # Print per-prompt details
    for r in results:
        try:
            gen_snippet = r["generated_text"][:40]
        except Exception:
            gen_snippet = "[?]"
        try:
            kv_extra = ""
            if args.measure_kv and r.get("kv_cos_sim_k") is not None:
                kv_extra = f" K_CosSim={r['kv_cos_sim_k']:.6f} V_CosSim={r['kv_cos_sim_v']:.6f}"
            fork_extra = ""
            if args.logits_diff and r.get("fork_point", {}).get("fork_step") is not None:
                fork_extra = f" fork@step={r['fork_point']['fork_step']}"
            line = f"  [{r['route']}] {r['prompt'][:30]:<30s} char={r['char_match']} " \
                   f"pref={r['prefix_match']} rep={r['repetition_score']:.4f} " \
                   f"gen=\"{gen_snippet}\"{kv_extra}{fork_extra}"
            print(line.encode('utf-8', errors='replace').decode('utf-8'))
        except Exception:
            pass

    print(f"\n[OK] eval_generation_fidelity complete.")
    if args.text_output_dir:
        print(f"Text outputs saved to: {args.text_output_dir}/")


if __name__ == "__main__":
    main()
