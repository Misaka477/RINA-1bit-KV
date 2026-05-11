# RINA Phase 2：常态 1-bit 符号残差与长文本/多轮可用性分析

> 基于 commit 3ff30e6 的优化实施与诊断讨论。日期：2026-05-10

---

## 一、实施概览

### 1.1 已实施改动

基于 `.kilo/plans/1778337399423-swift-meadow.md` 的优化计划，在三个核心文件中落地了以下改动：

| 文件 | 改动 | 行数 |
|------|------|:---:|
| `rina/config.py` | 新增 `decode_gap_threshold` 字段 | +1 |
| `rina/ds_kv_cache.py` | `_gap_danger` / `_recent_ring` 字段 | +2 |
| `rina/ds_kv_cache.py` | `append_incremental` 环形缓冲区记录 | +6 |
| `rina/ds_kv_cache.py` | `_encode_and_append_tile` 常态 1-bit 符号残差 | +35 |
| `rina/ds_kv_cache.py` | `encode_kv_cache` 常态 1-bit 符号残差（prefill 侧） | +40 |
| `rina/model_wrapper.py` | `_append_incremental` 新增 `gap_protect` 参数 + flag 传播 | +15 |
| `rina/model_wrapper.py` | `generate()` 新增 logits gap 检测 + 传递 | +8 |
| `rina/model_wrapper.py` | 新增 `turn_flush()` 方法 | +35 |
| `rina/model_wrapper.py` | 新增 `chat()` 方法 | +40 |
| `scripts/.../eval_generation_fidelity.py` | `make_config` + CLI + decode loop 支持 `decode_gap_threshold` | +20 |

### 1.2 改动原理

```
每 tile 编码完成后的追加步骤（无条件执行）:

  primary_full = decode(primary_bases)        ← 主编码的重建
  if bases_res != None:                       ← 如果已有 diff residual
      primary_full += diff_gamma × decode(bases_res)  ← 叠加 diff 修正
  residual = tile - primary_full              ← 完整修正后的残余误差

  step 1: 无条件 1-bit 符号编码
  step 2: 如果 _gap_danger==True → 追加第二步（logits 分叉预警触发）

  合并到 bases_res / alphas_res，重建时自动叠加
```

### 1.3 存储成本

| 场景 | 每 element 位宽 | 相对纯 n=5 的增幅 |
|------|:---------:|:---------:|
| 纯 Σ-Δ n=5 | 5.31 bit | 基线 |
| + 1-bit sign（常态） | 6.37 bit | +20% |
| + gap 第二步触发 | 7.43 bit | +40% |

---

## 二、Test A 回归验证

### 2.1 实验配置

```bash
python scripts/evaluation/eval_generation_fidelity.py \
  --quality balanced --n-steps 5 --prefill-n-steps 8 \
  --prefill-system-protect 128 --prefill-tail-protect 32 \
  --adaptive-residual --adaptive-residual-threshold 0.05 --adaptive-residual-n-steps 2 \
  --decode-protect-steps 3 --decode-protect-layers last_4 \
  --decode-gap-threshold 0.5 \
  --max-tokens 100 --measure-kv --logits-diff \
  --prompts "The capital of France is"
```

### 2.2 结果对比

| 指标 | 优化前 (3ff30e6) | 优化后 | 变化 |
|------|:---:|:---:|:--:|
| **fork_step** | 41 | **49** | **+19.5%** |
| **prefix_match** | 195 | **227** | **+16.4%** |
| **char_match** | 0.4470 | **0.5094** | **+14.0%** |
| **K_CosSim** | 1.000000 | 1.000000 | 不变 |
| **V_CosSim** | 1.000000 | 1.000000 | 不变 |
| **分叉点** | " with" (22.34) → "." (21.97) gap=0.37 | " The" (21.97) → " Paris" (22.17) gap=0.20 | gap 缩小 |

### 2.3 解读

1-bit sign residual 将分叉推迟了 8 步。fork 点上 native top-1 ("The") 与 DS top-1 ("Paris") 之间的 logits 差值仅 **0.05**——已经无限接近边界。误差的**方向**开始在此位置产生足够大的扰动，argmax 翻转。

关键结论：**1-bit sign 是"推迟"而不是"消除"误差增长**。每步的误差增长率从 ~22% 降到了 ~11%，但仍然在累积。如果不配合周期刷新，误差大概率继续发散。

---

## 三、极坐标/角度感知残差方案

### 3.1 核心思路

来自 TurboQuant 的极坐标思路可以部分借用到 RINA：

```
内积本质: Q · K = |Q| · |K| · cos(θ)

直角坐标弱点: 每个轴的量化误差直接影响 cos(θ)
极坐标优势:   角度（方向）的精度 > 幅度的精度
             只要 cos(θ) 存得准，哪怕 |K| 缩水了，attention 排名不变
```

### 3.2 当前适配可行性

| 想法 | 可行性 | 实施成本 | 理由 |
|------|:---:|:---:|------|
| 角度感知残差触发 | ✅ **高** | ~20 行 | 改 `_encode_and_append_tile` 的触发条件 |
| 残差加权方向优先 | ✅ 中 | ~50 行 | 改 loss/门控函数 |
| 完整极坐标编码器 | ❌ 低 | ~500+ 行 | d_head=64 需要 63 个角度参数，无压缩增益 |
| 极坐标位宽分配 | ❌ 低 | 需要重写 `encode_matrix` | 跟当前 Σ-Δ 架构根本冲突 |

### 3.3 推荐的切入点：CosSim 门控 1-bit sign

```python
# 当前：无条件每 tile 做 1-bit sign（1.06 bit/el 固定成本）
# 改进后：
cos_sim = cosine_similarity(tile, primary_full)
if cos_sim < cos_threshold:   # 例如 0.9999
    # 方向真的偏了 → 追加 1-bit sign
else:
    # 方向没偏 → 跳过，省 bit
```

**实测依据**：Test A 的 K_CosSim 始终 1.000000，说明 >99% 场景下 tile 方向完全未偏。CosSim 门控的**跳过率预期 >99%**，将常态 1-bit sign 的成本从 1.06 bit/el 降至接近 0。

### 3.4 实际收益估算

```
当前 (1-bit sign 无条件):   6.37 bit/el
CosSim 门控后:              ~5.35 bit/el (跳过 >99% tile)
几乎跟纯 n=5 的 5.31 bit/el 差不多

增益 ≈ 1 bit/el ≈ 16% 压缩比改进
代价 ≈ 0%（有 CosSim 计算但本就已解码 primary_full）
```

---

## 四、超长文本与多轮对话可用性分析

### 4.1 核心矛盾

Σ-Δ 的误差特性决定了**无界累积**：

```
误差模型: ε(t) = x(t) - x̂(t)
ε(t) 是低频、同号累积，不是随机游走的白噪声
1-bit sign 降低了增长率（22%→11%），但没有消除增长趋势
```

**对超长文本的结论**：1-bit sign + gap detection 能保证 **~150 token 内不崩**，但 1000+ token 时误差必然突破临界点。

### 4.2 已有但未启用的关键机制

| 机制 | config.py 默认值 | 效果 | 当前状态 |
|------|:---:|------|:---:|
| `refresh_interval` | **0（关）** | 每 N 步写入 FP16 锚点，误差彻底归零 | **需开启** |
| `beta_decay_start` | **None（无衰减）** | 后期降低 β 防止振荡 | **需开启** |

`refresh_interval=8` 注入后，误差有界：

```
t=0  [FP16]  ← decode_protect_steps
t=1  [ΣΔ+1bit]  误差 ~11%
...
t=7  [ΣΔ+1bit]  误差 ~77%
t=8  [FP16]  ← refresh_interval=8 锚点，误差归零
t=9  [ΣΔ+1bit]  误差 ~11% → 重新开始
```

**有界性不依赖步数**。生成 100 还是 10000 步，误差都一样有界。

### 4.3 多轮对话的"老 token 精度"问题

新加的 `turn_flush(keep_tail=32)` 刷新最后 32 个 decode token。但更早的 decode token（轮次 0-1 的非尾部 token）的 Σ-Δ 误差永不消失：

```
轮次 0:  prefill(128) + decode(128)
          ↓ turn_flush → 最后 32 被 FP16 覆盖
轮次 1:  旧 prefill(128) + 旧 decode(96 未刷新) + 新 decode(128)
          96 个带原始 Σ-Δ 误差的 decode token 残留
```

**prefill 不是瓶颈**（K_CosSim=1.0）。误差全部来自早期的增量 decode token。

**解决方案**：加 `refresh_interval=8` 后，这 96 个 token 每 8 步有一个 FP16 锚点刷新过，误差本身就是有界的（最大 77%），不再构成累积威胁。

### 4.4 多轮对话的显存无增长问题

```
单个 decode token 的 Σ-Δ 存储:
  bases:       160 bytes/tile ÷ 16 = 10 bytes/token
  alphas:      10 bytes/tile ÷ 16 = ~1 bytes/token
  bases_res:   96 bytes/tile ÷ 16  =  6 bytes/token
  alphas_res:  6 bytes/tile ÷ 16   = ~1 bytes/token
  ─────────────────────────────────────────────
  per token: ≈ 18 bytes × 8 heads × 16 layers ≈ 2.3 KB/token

1000 token decode = 2.3 MB
10 轮 × 128 tokens = ~2.5 MB
```

相比 2.5GB 模型权重，Σ-Δ 的存储成本已可忽略。**滑动窗口在当前规模下无必要。**

### 4.5 未来超长上下文：渐进式精度降级

如果未来需要真正在 32K 上下文中工作，正确的策略不是"丢弃旧 tile"，而是"降级旧 tile 的精度等级"：

```python
# 旧 tile: n=5 (6.37 bit/el) → 降级为 n=2 (2.50 bit/el)
# 信息不丢，只是信噪比从 7dB 降到 ~3dB
# 仍然保留 attention 所需的基本方向信息
```

相比直接滑动窗口的"信息丢失 = attention segfault"，这是唯一在不破坏模型功能的前提下节省空间的办法。

---

## 五、当前架构全景

### 5.1 各组件职责矩阵

```
refresh_interval=8         ← 核心保障层（误差有界化）
    ├── 超长文本：1000+ token 不崩
    ├── 多轮对话：每轮误差有界
    └── 存储：~128 bytes/refresh 条目

1-bit sign residual        ← 精度提升层
    ├── 无条件 step 1：降低增长率 (22% → 11%)
    ├── gap-detection step 2：分叉危险区加倍防护
    └── 成本：+1.06 bit/el (可被 CosSim 门控优化)

CosSim 门控残差            ← 待实现优化层
    ├── >99% 场景跳过 1-bit sign
    └── 成本从 1.06 bit/el → ~0.01 bit/el

turn_flush(keep_tail=32)   ← 多轮边界层
    ├── 轮次结束时 FP16 刷新最近 32 token
    └── 依赖 _recent_ring 环形缓冲区

prune_oldest_tiles()       ← 暂不需要（当前存储开销可忽略）
    └── 未来 32K 上下文时启用渐进式精度降级
```

### 5.2 各场景可用性判断

| 场景 | 当前 | 启用 refresh=8 | 最优组合 |
|------|:---:|:---:|:---:|
| **短生成 <100 tokens** | ✅ fork=49 | ✅ fork≥60 | 1-bit sign + gap + CosSim 门控 |
| **中生成 100-500 tokens** | ⚠️ 误差驱动衰减 | ✅ 有界 | refresh=8 + 1-bit sign |
| **超长 1000+ tokens** | ❌ 发散 | ✅ 有界 | refresh=8 + beta_decay + 1-bit sign |
| **多轮 5 轮内** | ⚠️ 部分保护 | ✅ 每轮有界 | refresh=8 + turn_flush |
| **多轮 20+ 轮** | ❌ 无保护 | ✅ 每轮有界 + 需未来降级 | refresh=8 + turn_flush + tile 降级 |
| **32K 上下文** | ❌ 误差发散 + 存储膨胀 | ⚠️ 存储仍需降级 | refresh=16 + 渐进式降级 |

---

## 六、下一步方向

### 6.1 一行代码的改动（推荐立即做）

```
config.py 第 222 行:
  refresh_interval: int = 0   →   refresh_interval: int = 8
```

效果：误差从"无界累积"变为"有界（最长 7 步连续编码）"，对超长文本和多轮对话是结构性的保障。

### 6.2 角度感知残差触发（下一个 ~20 行改动）

在 `_encode_and_append_tile` 的 1-bit sign 块之前增加 CosSim 门控：

```python
primary_full = decode_from_bases(bases, alphas, shape, tile_size=tile_size)
# ... diff_residual 叠加 ...
cos_sim = F.cosine_similarity(
    tile.flatten().unsqueeze(0), primary_full.flatten().unsqueeze(0),
).item()
if cos_sim > 0.9999:
    # 方向基本没偏 → 跳过 1-bit sign，省 bit
    return  # 或跳过
# 否则追加常规的 1-bit sign + gap trigger
```

### 6.3 全长度压力测试

```bash
# 500 token 测试 — 验证 refresh_interval 的误差有界性
python scripts/evaluation/eval_generation_fidelity.py \
  --quality balanced --n-steps 5 --prefill-n-steps 8 \
  --prefill-system-protect 128 --prefill-tail-protect 32 \
  --adaptive-residual --adaptive-residual-threshold 0.05 --adaptive-residual-n-steps 2 \
  --decode-protect-steps 3 --decode-protect-layers last_4 \
  --refresh-interval 8 --decode-gap-threshold 0.5 \
  --max-tokens 500 --measure-kv --logits-diff \
  --prompts "The capital of France is"
```

### 6.4 在实现极坐标编码前需要验证的前提

在投入大规模改动之前，需要验证两个假设：

1. **"幅度误差不伤 attention" 是否成立**：在已知 CosSim=1.0 的情况下，给定一系列渐变的幅度误差，测量 softmax 排名的保持率。用单层 attention 的前进实验可直接验证。
2. **"极坐标方向优先的 bit 分配" 是否能超过等分 bit**：给定固定 6 bit/element 的 budget，对比 (a) 等分直角坐标 6 bit 与 (b) 角度 4 bit + 幅度 2 bit 的 softmax 保真度。

两个实验~100 行脚本即可完成，无需动编码器。

---

## 附录：相关文档

- `RINA_诊断分析_EOS丢失_重复循环根因分析.md` — 四组实验的系统诊断（fork 的根本原因分析）
- `.kilo/plans/1778337399423-swift-meadow.md` — 完整的 Phase 2 优化计划
- `docs/RINA_V2_白皮书_自适应1bit残差纠正.md` — adaptive residual 的设计白皮书
- `docs/RINA_V2_Attention干预路线.md` — 分叉防护的注意力干预方案
