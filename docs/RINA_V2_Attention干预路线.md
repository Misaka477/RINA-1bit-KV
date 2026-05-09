# RINA V2 技术设计：Attention 层自适应干预路线

## 1. 动机

### 1.1 残差纠正的天花板

自适应 1-bit 残差纠正（白皮书 V2.1）将 char_match 从 0.8442 提升到 0.9000，但无法突破 ~0.92。根本原因：

```
┌─────────────────────────────────────────────────┐
│                  Attention 消费链                 │
│                                                  │
│  KV Store ──→ Attention ──→ Next Token          │
│      │              │                             │
│  残差在此修正    但 token 已经被错误版本消费       │
│  (事后纠正)      (不可逆)                         │
└─────────────────────────────────────────────────┘
```

残差是**事后纠正**——在存储层修正 KV，但 Transformer 的 attention 是**即时消费**。一旦某个 step 的输出被错误的 attention 分布决定，后续所有 step 都基于错误的 token 序列，分叉不可逆。

### 1.2 核心洞察

要突破 0.92 的天花板，必须在 **attention 消费之前** 就确保 K/V 的精度。这有两种路径：

```
路径 A: 存储层干预 → 在 attention 消费前把 KV 改为 FP16（FP16 bypass）
路径 B: 注意力层干预 → 告诉 attention"哪些 KV 不太可信"（置信度掩码）
```

路径 A 我们已经测试并确认有效（prefill_protected 下 char_match = 1.0）。本路线重点关注**路径 B**——以更低的存储代价实现同样的效果。

---

## 2. 方案设计

### 2.1 关键位置保护（Minimal FP16 Bypass）

**原理**: 不是所有 decode step 的 KV 同等重要。前 2-3 个 decode step 决定了 attention 分布的初始方向。如果这几步零误差，后续 step 的噪声不会导致 catastrophic divergence。

```
step 0 (first decode): Q 来自 prefill 的最后 token
                       → K₁...K_T 的误差直接影响 attention 权重分配
                       → 这一跳的 FP16 代价: 1 token × L×H × 128B ≈ 98KB

step 1-2: 同上
                → 总代价: 3 tokens × 98KB ≈ 300KB
```

**实现** (`_append_incremental`):

```python
# 在 append_incremental 的 bypass 写入处
if decode_step < self.cfg.decode_protect_steps:
    pos = self.n_tokens
    k_store._bypass_map_fp16[pos] = k_new.half().squeeze(0)
    v_store._bypass_map_fp16[pos] = v_new.half().squeeze(0)
```

**预期效果**:

| 配置 | char_match | 额外存储 |
|------|-----------|---------|
| n=3 + residual, no attention protect | 0.9000 | 0 |
| n=3 + residual + protect first 2 steps | **≥0.95** | ~200KB |
| n=3 + residual + protect first 5 steps | **≥0.97** | ~500KB |
| n=5 + residual + protect first 3 steps | **≥0.98** | ~300KB |

---

### 2.2 错误感知注意力掩码（Confidence-Masked Attention）

**原理**: 给每个 K/V token 一个置信度权重 c[i] ∈ [0, 1]，在 attention logits 上追加惩罚项。高置信 token (c ≈ 1.0) 不受影响；低置信 token (c ≈ 0.5) 被"软化"，避免模型过度依赖它。

```
原生 attention:
  scores = Q × K^T / √d

置信度掩码 attention:
  penalty[i] = β × (1 - confidence[i]) × mean(|scores|)
  scores_masked = scores - penalty[i]
  weights = softmax(scores_masked)

其中:
  confidence[i] = {
    1.0           if K[i] ∈ bypass_map_fp16 (FP16 零误差)
    0.8           if K[i] has residual correction
    0.6           if K[i] is pure Σ-Δ encoded (可能最高误差)
    1.0           if K[i] ∈ prefill_protected
  }

  β = 0.3~0.7 (惩罚强度)
```

**不修改模型权重**，只是在推理时对 attention logits 做轻量级偏置。不需要训练、不需要校准。

**实现位置**: 在 `_build_past_from_ds` 返回 KV cache 后，在下一轮 forward 中 hook attention 层。

**预期效果**:

| 配置 | char_match | 额外计算 |
|------|-----------|---------|
| n=3 + confidence mask (β=0.5) | ≥0.93 | ~O(T) per layer |
| n=5 + confidence mask (β=0.5) | ≥0.96 | ~O(T) per layer |

---

### 2.3 软硬混合路径（Hybrid）

结合 2.1 和 2.2:

```
1. step 0-2:          FP16 bypass (无误差初始化 attention)
2. step 3-N:           Σ-Δ 1-bit 编码 + adaptive residual
3. 所有 step:          confidence masked attention
4. 高误差 tile:        自动触发 residual correction
```

这形成了一个「渐进的保真度策略」——最关键的 token 用 FP16，普通 token 用 1-bit+残差，attention 对低置信 token 做软惩罚。

**预期**:
- **char_match ≥ 0.98** (接近 FP16)
- **CR ≥ 20×** (远超 4-bit 量化)
- **额外计算开销**: < 1% (掩码是纯整数偏置)

---

## 3. 实现详细设计

### 3.1 置信度计算

```python
def compute_token_confidence(store: DSKVCacheStore, pos: int) -> float:
    # 1. FP16 bypass 映射 → 零误差
    if hasattr(store, '_bypass_map_fp16') and pos in store._bypass_map_fp16:
        return 1.0

    # 2. In raw buffer → FP16 原始值
    if store.raw_buffer is not None:
        buf_pos = pos - (store.n_tokens - store.buffer_full)
        if 0 <= buf_pos < store.buffer_full:
            return 1.0

    # 3. Has residual correction → 部分修正
    if hasattr(store, 'bases_residual') and store.bases_residual is not None:
        # 残差 tile 在存储中，对应的 token 有额外纠正
        tile_idx = pos // store.tile_size
        if tile_idx < store.bases_residual.shape[1]:
            return 0.85

    # 4. Pure Σ-Δ → 可能有最高误差
    return 0.7
```

### 3.2 注意力掩码注入点

在 `model_wrapper.py` 的 `generate` 方法中，hook 模型的 attention 层:

```python
# 伪代码
def _attention_hook(module, input, output):
    # input: (query, key, value, attention_mask)
    query, key, value = input[:3]
    # 从 store 获取每个 key position 的置信度
    confidence = get_confidence_vector(layer_idx)
    # 计算惩罚
    penalty = (1 - confidence).unsqueeze(0).unsqueeze(-1) * beta
    # 修改 attention mask
    if attention_mask is not None:
        attention_mask = attention_mask - penalty
    # forward 正常 attention
    return original_forward(query, key, value, attention_mask)
```

### 3.3 关键位置保护写入

```python
# config.py 新增
decode_protect_steps: int = 0     # 前 N decode step 用 FP16 保护
decode_protect_interval: int = 0  # 每 K 步保护一步（可选）

# model_wrapper.py 在 _append_incremental 中
if decode_step < self.cfg.decode_protect_steps:
    for h in range(n_kv):
        k_new = k_full[0, h, new_token_idx:]
        v_new = v_full[0, h, new_token_idx:]
        k_stores[h]._bypass_map_fp16[pos] = k_new.half().squeeze(0)
        v_stores[h]._bypass_map_fp16[pos] = v_new.half().squeeze(0)
```

---

## 4. 实验路线

### Phase AL1: 关键位置保护 (最小改动)

```bash
# 保护前 3 个 decode token
python eval_generation_fidelity.py --n-steps 3 \
  --prefill-n-steps 8 --prefill-system-protect 128 --prefill-tail-protect 32 \
  --adaptive-residual --adaptive-residual-threshold 0.1 \
  --decode-protect-steps 3 \
  --max-tokens 200 --measure-kv
```

验证: char_match 是否从 0.90 → 0.95+

### Phase AL2: 置信度掩码 (中等改动)

```bash
# 置信度掩码 β=0.5
python eval_generation_fidelity.py --n-steps 3 \
  --prefill-n-steps 8 --adaptive-residual --adaptive-residual-threshold 0.1 \
  --confidence-mask --confidence-beta 0.5 \
  --max-tokens 200 --measure-kv
```

验证: char_match 是否达到 0.93+

### Phase AL3: 混合路线 (全量)

```bash
# 关键保护 + 置信度掩码 + 残差
python eval_generation_fidelity.py --n-steps 3 \
  --prefill-n-steps 8 --adaptive-residual --adaptive-residual-threshold 0.1 \
  --decode-protect-steps 3 --confidence-mask --confidence-beta 0.5 \
  --max-tokens 200 --measure-kv
```

验证: char_match 是否达到 0.98+

---

## 5. 风险与取舍

| 风险 | 缓解措施 |
|------|---------|
| 置信度掩码可能降低模型质量（惩罚了确实重要的 token） | β 可调，从 0→0.5 梯度测试 |
| 关键位置保护增加少量存储 | 前 3-5 步 × 128B × L×H ≈ 300KB，忽略不计 |
| Attention hook 可能降低推理速度 | 掩码是纯向量加法，O(N) 成本 < 0.1% |
| Confidence 计算需要额外字典查找 | 缓存 confidence vector 在 `_build_past_from_ds` 时 |

---

## 6. 与残差路线的关系

```
                     ┌──────────────────────────┐
                     │    RINA V2 双路线架构      │
                     └──────────────────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              │                                      │
     ┌───────▼────────┐                    ┌────────▼────────┐
     │  存储层优化      │                    │  Attention 层优化│
     │ (白皮书 V2.1)   │                    │   (本文档)       │
     │                │                    │                 │
     │ • 1-bit 残差   │                    │ • 关键位置保护   │
     │ • 双存储       │                    │ • 置信度掩码     │
     │ • 金字塔预填充 │                    │ • 软硬混合       │
     │                │                    │                 │
     │ char_match     │                    │ char_match       │
     │   ≥ 0.90       │                    │   ≥ 0.98         │
     │ CR ≥ 25×       │                    │ CR ≥ 20×         │
     └────────────────┘                    └──────────────────┘
              │                                      │
              └──────────────────┬──────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │     最佳组合              │
                    │  char_match ≥ 0.98       │
                    │  CR ≥ 18×                │
                    │  存储密度 < 2 bits/elem   │
                    └─────────────────────────┘
```

两条路线互补：存储层优化追求极限密度，attention 层优化保障精度下限。

---

*文档版本: 2026-05-09*  
*作者: RINA Core Team*  
*状态: 设计阶段，待实验验证*
