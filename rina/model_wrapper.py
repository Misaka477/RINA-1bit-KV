"""§6 DS-KVCache Model Wrapper — Incremental 16×16 Tile Pipeline
==============================================================

Wraps a HuggingFace Llama model with DS-KVCache 1-bit compression.

Key flow:
  • Prefill: bulk-encode ALL prompt K/V into DSKVCacheStore via encode_kv_cache()
  • Decode loop: each step produces 1 new K/V token → append_incremental()
  • reconstruct_all() = decoded bit-packed tiles + raw_buffer tail
  • past_key_values passed to model = reconstruct_all() as (1, n_kv_heads, T, d_head)
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

from rina.config import DSKVCacheConfig
from rina.ds_kv_cache import (
    DSKVCacheStore,
    encode_kv_cache,
)

_logger = logging.getLogger("model_wrapper")


def _past_get_kv(past, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract K, V from HuggingFace past_key_values for a specific layer."""
    if isinstance(past, DynamicCache):
        k = past.key_cache[layer_idx]   # (1, n_kv_heads, T, d_head)
        v = past.value_cache[layer_idx]
    elif isinstance(past, tuple):
        layer = past[layer_idx]
        k, v = layer[0], layer[1]      # each (1, n_kv_heads, T, d_head)
    else:
        raise TypeError(f"Unsupported past_key_values type: {type(past)}")
    return k, v


class DSKVCacheModel:
    """Model wrapper that applies DS-KVCache 1-bit compression to K/V caches.

    Usage::

        model = DSKVCacheModel(model, tokenizer, cfg)
        text = model.generate("The future of AI is", max_new_tokens=50)
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        cfg: Optional[DSKVCacheConfig] = None,
        *,
        auto_detect: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer

        if cfg is None and auto_detect:
            from rina.model_adapter import ModelProfile, HardwareProfile, ModelAdapter
            hw = HardwareProfile.detect()
            profile = ModelProfile.from_hf_config(model.config)
            adapter = ModelAdapter(profile, hw)
            self.cfg = adapter.recommend_config(quality="balanced")
            _logger.info("Auto-detected config: n_steps_k=%d, n_steps_v=%d, tile_size=%d, d_head=%d",
                         self.cfg.get_n_steps_k(), self.cfg.get_n_steps_v(),
                         self.cfg.tile_size, profile.d_head)
        elif cfg is None:
            self.cfg = DSKVCacheConfig(
                n_steps_k=3,
                n_steps_v=5,
                tile_size=16,
                beta=0.15,
                use_noise_shaping=True,
                proj_rank=8,
                proj_beta=0.3,
                adaptive_eta=True,
                use_differential=True,
                diff_strategy="residual",
                diff_residual_gamma=0.25,
                diff_residual_n_steps=1,
                v_orthogonal_transform=True,
            )
        else:
            self.cfg = cfg

        # Per-layer DS stores: list of (k_store, v_store) tuples
        self._ds_layers: List[Tuple[DSKVCacheStore, DSKVCacheStore]] = []
        self._v_rotations: List[Optional[torch.Tensor]] = []

        # Number of layers in the model
        self._num_layers = model.config.num_hidden_layers
        self._num_kv_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
        self._d_head = model.config.head_dim if hasattr(model.config, "head_dim") else (
            model.config.hidden_size // model.config.num_attention_heads
        )

        _logger.info(
            "DSKVCacheModel: %d layers, %d KV heads, d_head=%d, K=%d-steps, V=%d-steps",
            self._num_layers, self._num_kv_heads, self._d_head,
            self.cfg.get_n_steps_k(), self.cfg.get_n_steps_v(),
        )

    # ------------------------------------------------------------------
    # Bulk encode (prefill)
    # ------------------------------------------------------------------

    def _bulk_encode_from_prefill(
        self,
        past_key_values,
        input_ids: torch.Tensor,
    ):
        """Encode the entire prefill K/V cache into DS stores using bulk path.

        Uses encode_kv_cache() with cross_token_group for bulk encoding.
        Short prefill sequences (< tile_size) naturally stay in raw_buffer
        with perfect fp16 fidelity via the store's incremental path — no
        padding degradation.

        Phase 4: When prefill_n_steps is set and differs from the global n_steps,
        creates a separate prefill store with higher n_steps and an empty decode
        store.  _build_past_from_ds concatenates both reconstructions.
        """
        from rina.ds_kv_cache import (DSKVCacheStore, _build_v_rotation,
                                      encode_kv_cache)

        self._ds_layers = []
        self._ds_prefill_layers = []
        self._v_rotations = []
        n_kv = self._num_kv_heads

        use_prefill_protect = getattr(self.cfg, 'prefill_protected', False)
        sys_protect = getattr(self.cfg, 'prefill_system_protect_len', 0)
        tail_protect = getattr(self.cfg, 'prefill_tail_protect_len', 0)
        use_pyramid = (not use_prefill_protect) and (sys_protect > 0 or tail_protect > 0)
        prefill_n = getattr(self.cfg, 'prefill_n_steps', None)

        for layer_idx in range(self._num_layers):
            k_full, v_full = _past_get_kv(past_key_values, layer_idx)
            # k_full: (1, n_kv_heads, T, d_head)

            # Get per-layer config (overrides n_steps_k/n_steps_v if layer_step_map)
            layer_cfg = self.cfg.get_layer_config(layer_idx, self._num_layers)

            # ── Protected layer (§8.1.8): skip 1-bit encoding ──────────
            is_protected = layer_idx in self.cfg.protected_layers
            use_dual = (prefill_n is not None
                        and prefill_n != layer_cfg.get_n_steps_k()
                        and not is_protected
                        and not use_prefill_protect)

            layer_k_stores = []
            layer_v_stores = []
            prefill_k_stores = []
            prefill_v_stores = []

            for h in range(n_kv):
                k_h = k_full[0, h].float()  # (T, d_head)
                v_h = v_full[0, h].float()  # (T, d_head)
                T = k_h.shape[0]

                if use_dual:
                    # ── Dual store: prefill with high n_steps, decode starts empty ──
                    prefill_cfg = copy.copy(layer_cfg)
                    prefill_cfg.n_steps = prefill_n
                    prefill_cfg.n_steps_k = prefill_n
                    prefill_cfg.n_steps_v = prefill_n
                    prefill_k, prefill_v = encode_kv_cache(
                        k_h, v_h, prefill_cfg, protected=is_protected,
                    )
                    # Pyramid bypass on prefill store (FP16 precision)
                    if use_pyramid:
                        sys_len = min(sys_protect, T)
                        for i in range(sys_len):
                            prefill_k._bypass_map_fp16[i] = k_h[i].half()
                            prefill_v._bypass_map_fp16[i] = v_h[i].half()
                        tail_len = min(tail_protect, T)
                        if tail_len > 0:
                            for i in range(tail_len):
                                pos = T - tail_len + i
                                prefill_k._bypass_map_fp16[pos] = k_h[pos].half()
                                prefill_v._bypass_map_fp16[pos] = v_h[pos].half()

                    # Decode store starts empty (inherits V rotation from prefill)
                    k_store = DSKVCacheStore(
                        tile_size=layer_cfg.tile_size,
                        cross_token_group=layer_cfg.cross_token_group,
                    )
                    v_store = DSKVCacheStore(
                        tile_size=layer_cfg.tile_size,
                        cross_token_group=layer_cfg.cross_token_group,
                    )
                    v_store.v_rotation_matrix = prefill_v.v_rotation_matrix
                    prefill_k_stores.append(prefill_k)
                    prefill_v_stores.append(prefill_v)
                else:
                    # ── Single store: encode + decode share same store ──
                    k_store, v_store = encode_kv_cache(
                        k_h, v_h, layer_cfg,
                        protected=is_protected or use_prefill_protect,
                    )

                    # ── Pyramid prefill (Phase 3): overlay system prompt + tail (FP16) ──
                    if use_pyramid and not is_protected:
                        sys_len = min(sys_protect, T)
                        for i in range(sys_len):
                            k_store._bypass_map_fp16[i] = k_h[i].half()
                            v_store._bypass_map_fp16[i] = v_h[i].half()
                        tail_len = min(tail_protect, T)
                        if tail_len > 0:
                            for i in range(tail_len):
                                pos = T - tail_len + i
                                k_store._bypass_map_fp16[pos] = k_h[pos].half()
                                v_store._bypass_map_fp16[pos] = v_h[pos].half()

                    prefill_k_stores.append(None)
                    prefill_v_stores.append(None)

                layer_k_stores.append(k_store)
                layer_v_stores.append(v_store)

            self._ds_layers.append((layer_k_stores, layer_v_stores))
            self._ds_prefill_layers.append((prefill_k_stores, prefill_v_stores))

    # ------------------------------------------------------------------
    # Decode loop: append_incremental per new token
    # ------------------------------------------------------------------

    def _append_incremental(
        self,
        past_key_values,
        new_token_idx: int = -1,
        decode_step: int = 0,
    ):
        """Append the LAST token's K/V from model's new past to our DS stores.

        Called after each decode-step forward pass.
        past_key_values contains FULL sequence: DS-decoded history + 1 raw new token.
        We slice out only the new token (position -1) and append_incremental it.

        Cross-head error sharing (§8.1.9): Σ-Δ momentum/integrator2 state
        from head h-1 is passed as the initial condition for head h's encoding.
        This distributes quantization error across all KV heads in GQA models,
        preventing any single head from accumulating disproportionate error
        that would corrupt its entire Q-head group.

        Dynamic beta decay (§8.1.11): beta is linearly decayed from
        beta_decay_start to beta_decay_end over beta_decay_tokens decode
        steps.  High early beta pushes early quantization error into high
        frequencies; low late beta prevents oscillation when integrators
        are saturated.
        """
        # Compute decayed beta for this decode step (once for all layers)
        decayed_beta = self.cfg.get_beta_for_decode_step(decode_step)

        for layer_idx in range(self._num_layers):
            k_full, v_full = _past_get_kv(past_key_values, layer_idx)
            # k_full: (1, n_kv_heads, T_total, d_head)
            n_kv = k_full.shape[1]

            k_stores = self._ds_layers[layer_idx][0]
            v_stores = self._ds_layers[layer_idx][1]

            # Get per-layer config (overrides n_steps_k/n_steps_v if layer_step_map)
            layer_cfg = self.cfg.get_layer_config(layer_idx, self._num_layers)

            # Apply decayed beta for this decode step
            layer_cfg = copy.copy(layer_cfg)
            layer_cfg.beta = decayed_beta

            # ── Cross-head error sharing (§8.1.9): chain Σ-Δ state across heads ──
            k_momentum, k_integrator2 = None, None
            v_momentum, v_integrator2 = None, None

            # ── Periodic FP16 bypass (P1 anchor refresh) ──
            is_bypass = (self.cfg.refresh_interval > 0 and
                         (decode_step == 0 or (decode_step + 1) % self.cfg.refresh_interval == 0))

            for h in range(n_kv):
                # Slice out the LAST token only (use new_token_idx: without stop)
                k_new = k_full[0, h, new_token_idx:]  # (1, d_head)
                v_new = v_full[0, h, new_token_idx:]  # (1, d_head)

                v_rot = v_stores[h].v_rotation_matrix
                k_momentum, k_integrator2 = k_stores[h].append_incremental(
                    k_new, cfg=layer_cfg, svd_shaper=None, v_rotation=None,
                    initial_momentum=k_momentum, initial_integrator2=k_integrator2,
                    bypass=is_bypass,
                )
                v_momentum, v_integrator2 = v_stores[h].append_incremental(
                    v_new, cfg=layer_cfg, svd_shaper=None, v_rotation=v_rot,
                    initial_momentum=v_momentum, initial_integrator2=v_integrator2,
                    bypass=is_bypass,
                )

    # ------------------------------------------------------------------
    # Build past_key_values from DS stores for next forward pass
    # ------------------------------------------------------------------

    def _build_past_from_ds(self, device: Optional[torch.device] = None) -> DynamicCache:
        """Reconstruct full past_key_values from DS stores.

        Returns DynamicCache with shape (1, n_kv_heads, total_tokens, d_head).

        Phase 4: When _ds_prefill_layers exists, concatenates prefill store
        reconstruction with decode store reconstruction per head.
        """
        if device is None:
            device = self.model.device

        new_past = DynamicCache()
        has_prefill = hasattr(self, '_ds_prefill_layers') and self._ds_prefill_layers

        for layer_idx in range(self._num_layers):
            k_stores = self._ds_layers[layer_idx][0]
            v_stores = self._ds_layers[layer_idx][1]
            n_kv = len(k_stores)

            k_list, v_list = [], []
            for h in range(n_kv):
                k_recon = k_stores[h].reconstruct_all(
                    self.cfg.tile_size, self.cfg.use_differential,
                )
                v_recon = v_stores[h].reconstruct_all(
                    self.cfg.tile_size, self.cfg.use_differential,
                )

                if has_prefill and layer_idx < len(self._ds_prefill_layers):
                    pk_stores = self._ds_prefill_layers[layer_idx][0]
                    if h < len(pk_stores) and pk_stores[h] is not None:
                        pv_stores = self._ds_prefill_layers[layer_idx][1]
                        k_prefill = pk_stores[h].reconstruct_all(
                            self.cfg.tile_size, self.cfg.use_differential,
                        )
                        v_prefill = pv_stores[h].reconstruct_all(
                            self.cfg.tile_size, self.cfg.use_differential,
                        )
                        if k_recon.numel() == 0:
                            k_recon = k_prefill
                            v_recon = v_prefill
                        else:
                            k_recon = torch.cat([k_prefill, k_recon], dim=0)
                            v_recon = torch.cat([v_prefill, v_recon], dim=0)

                k_list.append(k_recon)
                v_list.append(v_recon)

            k_recon_stack = torch.stack(k_list, dim=0).to(
                dtype=torch.float16, device=device,
            )
            v_recon_stack = torch.stack(v_list, dim=0).to(
                dtype=torch.float16, device=device,
            )

            # Add batch dim: (1, n_kv_heads, T, d_head)
            new_past.key_cache.append(k_recon_stack.unsqueeze(0))
            new_past.value_cache.append(v_recon_stack.unsqueeze(0))

        return new_past

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = False,
        baseline: bool = False,
    ) -> str:
        """Generate text, optionally using DS-KVCache compression.

        Parameters
        ----------
        baseline:
            If True, skip DS encoding entirely (pure FP16 model for comparison).
        """
        device = self.model.device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

        if baseline:
            # Vanilla generation (FP16 KV cache)
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if do_sample else 1.0,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )
            return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # ── Prefill: forward pass over prompt ──
        _logger.info("DS-KVCache generate: prompt_len=%d", input_ids.shape[1])
        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                use_cache=True,
                output_hidden_states=False,
                past_key_values=None,
            )

        # Bulk-encode prefill into DS stores
        self._bulk_encode_from_prefill(output.past_key_values, input_ids)

        # Build DS-decoded past for next step
        past = self._build_past_from_ds()

        generated_ids = input_ids[0].tolist()
        first_token = output.logits[0, -1, :].argmax().item()
        generated_ids.append(first_token)

        # ── Decode Loop ──
        for step in range(1, max_new_tokens):
            last_token = torch.tensor([[generated_ids[-1]]], device=device)

            with torch.no_grad():
                output = self.model(
                    input_ids=last_token,
                    use_cache=True,
                    output_hidden_states=False,
                    past_key_values=past,
                )

            # output.past_key_values = [DS_decoded_0..T-1, raw_KV_T]
            # Incrementally encode only the NEW token (position -1)
            # step=1 → decode_step=0 (the first auto-regressive token after prefill)
            self._append_incremental(output.past_key_values, new_token_idx=-1, decode_step=step - 1)
            past = self._build_past_from_ds()

            # ── Sample next token ──
            logits = output.logits[0, -1, :]
            if temperature > 0 and do_sample:
                logits = logits / temperature
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    remove_mask = cumulative_probs > top_p
                    remove_mask[1:] = remove_mask[:-1].clone()
                    remove_mask[0] = False
                    indices_to_remove = sorted_indices[remove_mask]
                    logits[indices_to_remove] = float('-inf')
                probs = torch.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token_id = torch.argmax(logits).item()

            generated_ids.append(next_token_id)

            if next_token_id == self.tokenizer.eos_token_id:
                break

            if step % 10 == 0:
                _logger.info("  Step %d: %d tokens total", step, len(generated_ids))

        result = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        _logger.info("Generated %d tokens", len(generated_ids))
        return result

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> List[dict]:
        """Return per-layer memory stats."""
        stats = []
        for layer_idx, (k_stores, v_stores) in enumerate(self._ds_layers):
            total_fp16 = 0
            total_ds = 0
            for h in range(len(k_stores)):
                k_stores[h].update_stats()
                v_stores[h].update_stats()
                total_fp16 += k_stores[h].fp16_memory_bytes + v_stores[h].fp16_memory_bytes
                total_ds += k_stores[h].memory_bytes + v_stores[h].memory_bytes
            stats.append({
                "layer": layer_idx,
                "fp16_memory_bytes": total_fp16,
                "ds_memory_bytes": total_ds,
                "compression_ratio": total_fp16 / (total_ds + 1e-12),
            })
        return stats

    def print_stats(self):
        """Pretty-print memory compression stats."""
        stats = self.get_stats()
        if not stats:
            print("No DS-KVCache data.")
            return

        header = f"{'Layer':>6} {'FP16 MB':>10} {'DS MB':>10} {'Ratio':>8}"
        print(header)
        print("-" * len(header))

        total_fp16 = 0
        total_ds = 0
        for s in stats:
            fp16_mb = s["fp16_memory_bytes"] / (1024**2)
            ds_mb = s["ds_memory_bytes"] / (1024**2)
            total_fp16 += fp16_mb
            total_ds += ds_mb
            print(
                f"{s['layer']:>6} {fp16_mb:>10.2f} {ds_mb:>10.2f} {s['compression_ratio']:>8.1f}x"
            )

        print("-" * len(header))
        print(
            f"{'TOTAL':>6} {total_fp16:>10.2f} {total_ds:>10.2f} {total_fp16/(total_ds+1e-12):>8.1f}x"
        )