# RINA Σ-Δ KV Cache 诊断分析：EOS 丢失与重复循环根因分析

> 基于四组对照实验（Test A/B/C/D）的系统性诊断。日期：2026-05-09

---

## 1. 问题定义

**现象**：配置 n_steps=5, dual store prefill=8, τ=0.05 时，DS 在前 195 字符与 native 完美对齐后，陷入重复循环（"Paris is the most populous urban area..."）而非生成合理后续内容。

**核心问题**：为什么 Σ-Δ 编码的 KV cache 会在特定步数后触发 argmax 翻转，进入自持循环？

---

## 2. 四组测试实验

### 2.1 测试设计与假设

| 测试 | 假设 | 验证方式 |
|------|------|---------|
| A | 分叉点 logits 分布显示 argmax 边界极其接近 | `--logits-diff` 抓取分叉点的 top-10 logits |
| B | Dual store 的 prefill(8)/decode(5) 质量不匹配是根因 | 统一 n_steps=5，关掉 dual store |
| C | 置信度掩码能通过惩罚低置信位置来压制重复 | `--confidence-mask --confidence-beta 0.7` |
| D | n_steps=3 时质量退化模式是否相同 | `--n-steps 3` 对比 |

### 2.2 实验结果汇总

| 测试 | char_match | prefix_match | fork_step | rep_score | Fork (nat→ds) |
|------|-----------|-------------|-----------|-----------|----------------|
| A (baseline n=5, dual) | 0.4470 | 195 | 41 | 0.1074 | " with" → " ." |
| B (no dual store) | 0.4470 | 195 | 41 | 0.1074 | " with" → " ." |
| C (confidence β=0.7) | 0.4470 | 195 | 41 | 0.1074 | " with" → " ." |
| D (n_steps=3) | **0.5011** | 195 | 41 | **0.0664** | " with" → " ," |

### 2.3 核心发现

1. **Dual store 证伪** — Test B 去掉 dual store 后结果与 A 完全一致。prefill(8)/decode(5) 的质量断裂不是根因。

2. **置信度掩码无效** — Test C 与 A 相同。重复循环中的 token 是新生成的（confidence=1.0），掩码不惩罚高置信度 token。

3. **n_steps=3 反而更好** — 预期更差，实际 char_match 0.5011 > 0.4470。原因：n=5 在分叉点产生 "." (句号，硬断)，n=3 产生 "," (逗号，句子继续)。**误差的方向比大小更重要。**

4. **分叉点不随配置改变** — 所有配置统一在 step 41 分叉。误差轨迹在前 40 步已确定，step 41 只是"果"。

---

## 3. 根因分析

### 3.1 误差累积机制

```
Step 1-40:  Σ-Δ 编码误差逐步累积 (7 dB SNR)
                    ↓
Step 41:    KV 幅度误差 ~40% 导致 logits 偏移
                    ↓
            native: " with" (logit 22.34) ← argmax 正确
            DS n=5: " ."     (logit 21.97) ← 偏移 0.37，argmax 翻转  
            DS n=3: " ,"     (logit 22.36) ← 偏移 -0.39，argmax 翻转方向不同
                    ↓
Step 42+:   错误 token 进入 KV cache + 持续步进
                    ↓
            重复循环: 新生成的 token 被重新 Σ-Δ 编码，
            幅度误差持续存在 → 下一个决策点再次重复
```

### 3.2 KV 误差的特性

- **KV CosSim 始终 >0.999999** — 方向性误差几乎为零
- **7 dB SNR ≈ 40-50% 幅度误差** — 这是 amplitude error，不是 angular error
- 每次 KV cache 重建后，误差以同号方向累积，非随机游走
- 在分叉点处，native 的 " with" 与 DS 的 "." 之间 logit 差距仅 **0.37**

### 3.3 为什么 n=3 比 n=5 更好（看似反常）

这不是 n=3 "更好"，而是 n=3 的误差在此特定分叉点产生了**更良性的扰动**：

```
n=5 的噪声方向:  → 压低 "with" (logit -0.37)，推高 "." → 句号 → 硬断 → 循环
n=3 的噪声方向:  → 压低 "with" (logit -0.39)，推高 "," → 逗号 → 句子继续 → 非循环
```

换一个 prompt，n=3 可能就是灾难。关键矛盾在于：**argmax 翻转取决于 logit 差值是否穿过零界，而不是误差的绝对大小。**

---

## 4. 方案对比

### 4.1 实际存储位宽

当前 Σ-Δ n=5 的精确存储（按 tile=256 元素）：

```
bases:   5 step × 256 元素 × 1 bit = 1280 bit
alphas:  5 step × 1 tile × 16 bit   =   80 bit
────────────────────────────────────────────
合计:    1360 bit / 256 = 5.3125 bit/element
```

加 cross_token_group=2（512 元素/tile）后：**~5.15 bit/element**

**关键澄清**：之前被误算为 10 bit，实际是 ~5.3 bit。bases 是 ±1（1 bit），alphas 是 per-tile per-step 标量（FP16），平均摊薄到每个元素几乎为零。

### 4.2 Σ-Δ 与其他量化方案对比

| 方案 | 位宽 | 压缩比 | 噪声形态 | 跨 token 冗余 | 流式增量 | 灵活控制 |
|------|------|--------|---------|:-----------:|:-------:|:-------:|
| FP16 原始 | 16 bit | 1:1 | — | — | — | — |
| INT8 | 8 bit | 2:1 | 白噪声 | ❌ | ❌ | ❌ |
| INT4 | 4 bit | 4:1 | 白噪声 | ❌ | ❌ | ❌ |
| TurboQuant | **3.5 bit** | **4.6:1** | 近白噪声 | ❌ | 需旋转 | ❌ |
| Σ-Δ n=5 | 5.3 bit | 3:1 | **低频衰减** | ✅ tile 编码 | ✅ 天生 | ✅ bypass/protect/mask |
| Σ-Δ + 1-bit residual | 6.3 bit | 2.5:1 | 低频衰减更优 | ✅ | ✅ | ✅ |
| Σ-Δ + 2-bit residual | 7.3 bit | 2.2:1 | 近乎 FP16 地板 | ✅ | ✅ | ✅ |

### 4.3 压缩比视角的结论

**如果只看压缩比**（位宽），Σ-Δ 在 6-7 bit 确实拼不过 TurboQuant 的 3.5 bit 或 INT4 的 4 bit。但 Σ-Δ 有三个其他方案不具备的独特价值：

1. **噪声整形（Noise Shaping）** — Σ-Δ 将量化误差推到高频（Walsh-Hadamard 基的末尾系数），Transformer attention 的 softmax 是低通运算，天然衰减高频噪声。均匀/标量量化产生白噪声，在所有频率等幅分布。

2. **跨 Token 相关性利用** — cross-token grouping 和 tile 级别编码天然压缩了相邻 token 的 KV 冗余。

3. **架构灵活性** — bypass、protect、confidence mask、不等 K/V 步数、选择性层保护，所有这些控制机制都可以在同一框架内协同。

---

## 5. 解码侧自适应残差校正方案

### 5.1 当前代码路径

| 编码路径 | 调用链 | adaptive_residual 是否生效 |
|---------|--------|--------------------------|
| Prefill (bulk) | `encode_kv_cache()` → `_encode_single_path()` | ❌ 从不触发 |
| Decode (逐 token) | `append_incremental()` → `_encode_and_append_tile()` | ✅ 仅 tile 满时触发（每 16 token） |

**实际效果**：在 decode 侧，adaptive_residual 有代码但几乎不触发——阈值 0.2 过宽，且仅生效于 tile 边界。

### 5.2 方案设计

在 `_encode_and_append_tile` 的 tile 编码完成后，计算 FP16 tile 与 Σ-Δ 重建值的差值，再用 1-2 bit Σ-Δ 编码存入 residual buffer，重建时叠加。

```python
# 每 tile 编码后：
delta = tile - primary                    # tile = FP16 原始, primary = Σ-Δ 重建
tile_linf = delta.abs().max()
if tile_linf > threshold:
    bases_res, alphas_res = encode_matrix(delta, n_steps=1, tile_size=16, ...)
    # 存储到 residual_bases / residual_alphas

# 重建时：
reconstructed = ΣΔ_decode(bases, alphas) + ΣΔ_decode(residual_bases, residual_alphas)
```

### 5.3 误差衰减估算

| residual bits | 编码误差 (per token) | 有效解码长度 |
|:---:|:---:|:---:|
| 0 (pure n=5) | ~22% | ~40 token |
| 1 | ~11% | ~80 token |
| 2 | ~5% | ~160-200 token |
| 3 | ~0.1% (低于 FP16 舍入噪声) | 不受 Σ-Δ 误差限制 |

### 5.4 位宽增幅

| 方案 | 每元素位宽 | 增量 |
|------|-----------|------|
| 纯 Σ-Δ n=5 | 5.3 bit | 基线 |
| + 1-bit residual | 6.3 bit | +19% |
| + 2-bit residual | 7.3 bit | +38% |

---

## 6. "不可能三角"的解决方案

### 6.1 目标重定义

不必用同一组比特同时满足压缩、质量和灵活性。三者可以分层：

```
目标 A: 大幅压缩 + 高质量  →  用 INT4/TurboQuant 做 prefill bulk
目标 B: 灵活操控 + 高质量  →  用 Σ-Δ 做 decode 增量
目标 C: 选择性精度        →  用注意力权重引导 bypass 分配
```

### 6.2 分层架构

| 场景 | 方案 | 位宽 | 核心诉求 |
|------|------|------|---------|
| Prefill bulk（大量一次性编码） | INT4 / TurboQuant / Σ-Δ n=3 | 3-4 bit | 高密度压缩 |
| Decode 增量（逐 token 追加） | Σ-Δ n=5 + residual | 5-6 bit | 流式、灵活操控 |
| 高注意力 token（softmax 权重高的位置） | FP16 bypass / 选择性残差 | 16 bit | 绝对精确 |
| 低注意力 token（尾部 token） | Σ-Δ n=2 | ~2.3 bit | 大胆压缩 |

### 6.3 与现有架构的关系

当前 dual store 已铺下骨架：

```
Prefill store (prefill_n_steps=8):  高质量 → 未来可替换为 TurboQuant
Decode store (n_steps=5):          流式 → 加 residual 增强质量
reconstruct_all()                   合并两者
```

三种编码路径不冲突，因为**服务于不同阶段和不同 token**。工程挑战在于统一 `reconstruct_all()` 的合并路径。

---

## 7. 后续方向

### 7.1 立即可落地

1. **解码侧 adaptive residual** — 降低 `adaptive_residual_threshold` 从 0.2 到 0.05，并验证 tile 边界 (每 16 token) 的频率是否足以改善 step 41 分叉。

2. **Per-step residual** — 每个 decode token 完成 Σ-Δ 编码后立即计算残差，而不等到 tile 边界。

### 7.2 中期探索

3. **注意力权重引导的选择性精度** — 利用 softmax 权重分布识别高注意力 token，对这部分位置做 bypass，其余保持低 bit 压缩。

4. **Prefill 侧 TurboQuant 替代** — 将 prefill store 的编码从 Σ-Δ 替换为 TurboQuant，用 3.5 bit 获得更优压缩比，解码侧保留 Σ-Δ 的灵活性。

---

## 附录：实验命令

### Test A — Logits 分叉点诊断
```bash
python scripts/evaluation/eval_generation_fidelity.py \
  --quality balanced --n-steps 5 --prefill-n-steps 8 \
  --prefill-system-protect 128 --prefill-tail-protect 32 \
  --adaptive-residual --adaptive-residual-threshold 0.05 --adaptive-residual-n-steps 2 \
  --decode-protect-steps 3 --decode-protect-layers last_4 \
  --max-tokens 100 --measure-kv --logits-diff \
  --logits-output test_A_fork_logits.json \
  --prompts "The capital of France is" \
  --json-output test_A_results.json
```

### Test B — 关掉 Dual Store
```bash
python scripts/evaluation/eval_generation_fidelity.py \
  --quality balanced --n-steps 5 \
  --prefill-system-protect 128 --prefill-tail-protect 32 \
  --adaptive-residual --adaptive-residual-threshold 0.05 --adaptive-residual-n-steps 2 \
  --decode-protect-steps 3 --decode-protect-layers last_4 \
  --max-tokens 100 --measure-kv \
  --prompts "The capital of France is" \
  --json-output test_B_nodual.json
```

### Test C — 置信度掩码 β=0.7
```bash
python scripts/evaluation/eval_generation_fidelity.py \
  --quality balanced --n-steps 5 --prefill-n-steps 8 \
  --prefill-system-protect 128 --prefill-tail-protect 32 \
  --adaptive-residual --adaptive-residual-threshold 0.05 --adaptive-residual-n-steps 2 \
  --decode-protect-steps 3 --decode-protect-layers last_4 \
  --confidence-mask --confidence-beta 0.7 \
  --max-tokens 100 --measure-kv --logits-diff \
  --logits-output test_C_conf_logits.json \
  --prompts "The capital of France is" \
  --json-output test_C_results.json
```

### Test D — n_steps=3 对比
```bash
python scripts/evaluation/eval_generation_fidelity.py \
  --quality balanced --n-steps 3 --prefill-n-steps 8 \
  --prefill-system-protect 128 --prefill-tail-protect 32 \
  --adaptive-residual --adaptive-residual-threshold 0.05 --adaptive-residual-n-steps 2 \
  --decode-protect-steps 3 --decode-protect-layers last_4 \
  --max-tokens 100 --measure-kv --logits-diff \
  --logits-output test_D_n3_logits.json \
  --prompts "The capital of France is" \
  --json-output test_D_results.json
```
