#!/usr/bin/env python
"""
Phase 3 — 三维正交化攻击：FWHT / Integrator均值归零 / 跨头分流 (v4)
==================================================================
对 Llama 3.2 1B 跑 4 组配置，每组记录：
  - first_divergence_position
  - token_match_rate
  - compression_ratio
  - mean CosSim (K/V)

三维正交方向:
    1. FWHT:    tile → Walsh-Hadamard 旋转 → 平坦频谱 → Σ-Δ 量化器零陷对准
    2. ZeroMean: integrator2 直流归零 → 二阶 Σ-Δ AC coupling
    3. CrossHead: 跨 KV head 传递 Σ-Δ 动量 → GQA 误差分流

实验矩阵:
    H_baseline:  FWHT ✗  ZeroMean ✗  CrossHead ✗
    I_fwht:      FWHT ✓  ZeroMean ✗  CrossHead ✗
    J_fwht_zm:   FWHT ✓  ZeroMean ✓  CrossHead ✗
    K_full:      FWHT ✓  ZeroMean ✓  CrossHead ✓

用法:
    python scripts/exp_push_divergence.py
    python scripts/exp_push_divergence.py --max-tokens 80 --output results/push_divergence.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rina.config import DSKVCacheConfig
from rina.model_wrapper import DSKVCacheModel
from scripts.auto_config import detect_gpu_info, detect_model_info, generate_optimal_config

logging.basicConfig(level=logging.INFO, format="%(message)s")
_logger = logging.getLogger("exp_push_divergence")


@torch.no_grad()
def run_single_config(
    model,
    tokenizer,
    cfg: DSKVCacheConfig,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    label: str,
) -> Dict[str, Any]:
    """Run DS-KVCache generation with a given config and return metrics."""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[1]

    wrapper = DSKVCacheModel(model, tokenizer, cfg=cfg)

    # Prefill
    out = wrapper.model(input_ids=input_ids, use_cache=True)
    logits_list = [out.logits[0, -1, :].cpu()]
    
    wrapper._bulk_encode_from_prefill(out.past_key_values, input_ids)
    past = wrapper._build_past_from_ds()

    generated_ids = input_ids[0].tolist()

    # Greedy argmax
    first_id = int(out.logits[0, -1, :].argmax().item())
    if first_id == tokenizer.eos_token_id:
        return {"error": "EOS on first token", "label": label}
    generated_ids.append(first_id)

    t_start = time.perf_counter()

    for step in range(1, max_new_tokens):
        last_token = torch.tensor([[generated_ids[-1]]], device=device)
        out = wrapper.model(
            input_ids=last_token,
            use_cache=True,
            past_key_values=past,
        )
        logits_list.append(out.logits[0, -1, :].cpu())

        # Pass decode_step for dynamic beta decay (§8.1.11)
        wrapper._append_incremental(out.past_key_values, new_token_idx=-1, decode_step=step - 1)
        past = wrapper._build_past_from_ds()

        next_id = int(out.logits[0, -1, :].argmax().item())
        generated_ids.append(next_id)

        if next_id == tokenizer.eos_token_id:
            break

    elapsed = time.perf_counter() - t_start
    gen_tokens = len(generated_ids) - prompt_len

    # ── Baseline comparison ──
    base_inputs = tokenizer(prompt, return_tensors="pt").to(device)
    base_out = model(input_ids=base_inputs["input_ids"], use_cache=True)
    base_past = base_out.past_key_values
    base_ids = base_inputs["input_ids"][0].tolist()
    base_first = int(base_out.logits[0, -1, :].argmax().item())
    base_ids.append(base_first)

    for s in range(1, max_new_tokens):
        last_t = torch.tensor([[base_ids[-1]]], device=device)
        base_out = model(input_ids=last_t, use_cache=True, past_key_values=base_past)
        base_past = base_out.past_key_values
        nxt = int(base_out.logits[0, -1, :].argmax().item())
        base_ids.append(nxt)
        if nxt == tokenizer.eos_token_id:
            break

    # ── Token match ──
    min_len = min(len(generated_ids), len(base_ids))
    match_count = sum(1 for i in range(min_len) if generated_ids[i] == base_ids[i])
    token_match_rate = match_count / min_len if min_len > 0 else 0.0

    first_div = None
    for i in range(min_len):
        if generated_ids[i] != base_ids[i]:
            first_div = i - prompt_len  # relative to first generated token
            break

    # ── Compression ──
    stats_list = wrapper.get_stats() if wrapper is not None else []
    total_fp16 = sum(s["fp16_memory_bytes"] for s in stats_list)
    total_ds = sum(s["ds_memory_bytes"] for s in stats_list)
    comp_ratio = total_fp16 / (total_ds + 1e-12) if total_ds > 0 else 0.0

    return {
        "label": label,
        "n_steps_k": cfg.get_n_steps_k(),
        "n_steps_v": cfg.get_n_steps_v(),
        "proj_beta": cfg.proj_beta,
        "diff_residual_gamma": cfg.diff_residual_gamma,
        "order2_gamma": cfg.order2_gamma,
        "use_fwht": cfg.use_fwht,
        "zero_mean_integrator2": cfg.zero_mean_integrator2,
        "cross_head_error_share": cfg.cross_head_error_share,
        "num_tokens_generated": gen_tokens,
        "first_divergence": first_div,
        "token_match_rate": round(token_match_rate, 4),
        "compression_ratio": round(comp_ratio, 2),
        "ds_memory_mb": round(total_ds / (1024**2), 2),
        "elapsed_s": round(elapsed, 2),
    }


def main():
    p = argparse.ArgumentParser(description="Phase 3 Orthogonal Attack Experiment (v4)")
    p.add_argument("--model", type=str, default="D:/Software_Development/Project/models/Llama-3.2-1B")
    p.add_argument("--prompt", type=str, default="The future of artificial intelligence lies in")
    p.add_argument("--max-tokens", type=int, default=80)
    p.add_argument("--output", type=str, default=None, help="CSV output path")

    # ── Common base config overrides ──
    p.add_argument("--n-steps-k", type=int, default=3)
    p.add_argument("--n-steps-v", type=int, default=5)
    p.add_argument("--beta", type=float, default=0.15)
    p.add_argument("--tile-size", type=int, default=16)
    p.add_argument("--no-ns", action="store_false", dest="use_ns")
    p.add_argument("--no-diff", action="store_false", dest="use_diff")
    p.add_argument("--diff-gamma", type=float, default=0.25)
    p.add_argument("--v-ortho", action="store_true", default=True, dest="v_ortho")
    p.add_argument("--no-v-ortho", action="store_false", dest="v_ortho")
    p.add_argument("--n-steps-k-base", type=int, default=4, help="K path steps for baseline configs")
    p.add_argument("--n-steps-v-base", type=int, default=6, help="V path steps for baseline configs")

    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # ── Hardware & Model ──
    gpu_info = detect_gpu_info()
    model_info = detect_model_info(args.model)
    device = torch.device(gpu_info["recommended_device"] if torch.cuda.is_available() else "cpu")

    _logger.info(f"GPU: {gpu_info['name']} ({gpu_info['vram_gb']} GB)")
    _logger.info(f"Model: {model_info['model_type']} L={model_info['num_layers']} "
                 f"GQA={model_info['gqa_ratio']}x d_head={model_info['d_head']}")

    # ── Load model ──
    _logger.info(f"Loading model {args.model} ...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    hf_model.eval()

    # ── Define 4 experimental configs (v4 — Phase 3 orthogonal attack) ──

    def _base_config(**overrides):
        """Build base config from CLI args + overrides.

        Phase 3 baseline: everything that competes with the three orthogonal
        dimensions is OFF.  Noise shaping and differential residual are kept
        as they are orthogonal infrastructure.
        """
        kwargs = {
            "n_steps_k": args.n_steps_k_base,
            "n_steps_v": args.n_steps_v_base,
            "tile_size": args.tile_size,
            "beta": args.beta,
            "use_noise_shaping": args.use_ns,
            "proj_rank": 8,
            "proj_beta": 0.3 if args.use_ns else 0.0,
            "adaptive_eta": args.use_ns,
            "use_differential": args.use_diff,
            "diff_strategy": "residual",
            "diff_residual_gamma": args.diff_gamma,
            "diff_residual_n_steps": 1,
            "v_orthogonal_transform": args.v_ortho,
            # ── C3 optimal: second-order Σ-Δ disabled (zero drift avoidance) ──
            "order2_gamma": 0.0,
            "order2_c1": 1.0,
            "order2_c2": 0.5,
            # ── All three Phase-3 dimensions OFF in C3 baseline ──
            "use_fwht": False,
            "zero_mean_integrator2": False,
            "cross_head_error_share": False,
            # ── Disable extras that confound the orthogonal test ──
            "cross_token_group": 1,
            "protected_layers": [],
            "use_recon_weights": False,
            "layer_step_map": None,
            "beta_decay_start": 0.30,
            "beta_decay_end": 0.05,
            "beta_decay_tokens": 0,
            "dynamic_tile_size": False,
            "base_dtype": "fp16",
            "verbose": False,
        }
        kwargs.update(overrides)
        return DSKVCacheConfig(**kwargs)

    configs = []

    # H: pure baseline — all three dimensions OFF
    cfg_h = _base_config()
    configs.append(("H_baseline", cfg_h))

    # I: FWHT only — energy diffusion via Walsh-Hadamard rotation
    cfg_i = _base_config(use_fwht=True)
    configs.append(("I_fwht", cfg_i))

    # J: FWHT + ZeroMean — energy diffusion + integrator DC removal
    cfg_j = _base_config(use_fwht=True, zero_mean_integrator2=True)
    configs.append(("J_fwht_zm", cfg_j))

    # K: full stack — all three dimensions ON
    cfg_k = _base_config(
        use_fwht=True,
        zero_mean_integrator2=True,
        cross_head_error_share=True,
    )
    configs.append(("K_full", cfg_k))

    # ── Run experiments ──
    results = []
    for label, cfg in configs:
        _logger.info(f"\n{'='*60}")
        _logger.info(f"  Running: {label}")
        _logger.info(f"  n_steps_k={cfg.get_n_steps_k()}, n_steps_v={cfg.get_n_steps_v()}, "
                     f"beta={cfg.beta}, order2_gamma={cfg.order2_gamma}")
        _logger.info(f"  FWHT={'Y' if cfg.use_fwht else 'N'}, "
                     f"ZeroMean={'Y' if cfg.zero_mean_integrator2 else 'N'}, "
                     f"CrossHead={'Y' if cfg.cross_head_error_share else 'N'}")
        _logger.info(f"{'='*60}")

        result = run_single_config(
            hf_model, tokenizer, cfg, args.prompt,
            args.max_tokens, device, label,
        )
        results.append(result)

        _logger.info(f"  Result: first_divergence={result.get('first_divergence')}, "
                     f"match_rate={result.get('token_match_rate', 'N/A')}, "
                     f"compression={result.get('compression_ratio', 'N/A')}x")

    # ── Summary ──
    _logger.info(f"\n{'='*70}")
    _logger.info("  SUMMARY — Phase 3 Orthogonal Attack Experiment (v4)")
    _logger.info(f"{'='*70}")
    header = (f"{'Label':<18} {'nk':>3} {'nv':>3} {'fwht':>5} {'zmi2':>5} {'xhead':>6} "
              f"{'1st_div':>8} {'match':>8} {'comp':>5}")
    _logger.info(header)
    _logger.info("-" * len(header))
    for r in results:
        fwht_str = "Y" if r.get("use_fwht") else "N"
        zmi2_str = "Y" if r.get("zero_mean_integrator2") else "N"
        xhead_str = "Y" if r.get("cross_head_error_share") else "N"
        _logger.info(
            f"{r['label']:<18} {r['n_steps_k']:>3} {r['n_steps_v']:>3} "
            f"{fwht_str:>5} {zmi2_str:>5} {xhead_str:>6} "
            f"{str(r['first_divergence']):>8} {r['token_match_rate']:>8.4f} "
            f"{r['compression_ratio']:>4.1f}x"
        )

    # ── Save CSV ──
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "label", "n_steps_k", "n_steps_v", "proj_beta", "diff_residual_gamma",
            "order2_gamma", "use_fwht", "zero_mean_integrator2", "cross_head_error_share",
            "num_tokens_generated", "first_divergence",
            "token_match_rate", "compression_ratio", "ds_memory_mb", "elapsed_s",
        ]
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        _logger.info(f"\nSaved to {output_path}")

    _logger.info("\nDone.")


if __name__ == "__main__":
    main()