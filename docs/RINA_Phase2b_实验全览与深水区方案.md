# RINA Phase 2b：实验全览、根因分析与深水区方案

> 日期：2026-05-10
> 范围：基于 Phase 2 (commit 3ff30e6，常态 1-bit 符号残差) 的 CosSim 门控、refresh_interval、
> confidence_mask、n_steps 提升、自适应步数、beta decay、正交变换等 7 项实验，
> 以及后续对抗性噪声脱钩、低精度浮点残差、多头位宽分配等 3 项深水区方案。

---

## 一、实验总览

### 1.1 实验基准

- 模型：Llama-3.2-1B（d_head=64, 16 layers, 8 KV heads）
- Prompt：`"The capital of France is"`
- 评估：每步 logits 与 native greedy 对比，检测 argmax 首次分叉（fork）
- 配置基线：

```bash
--quality balanced --n-steps 5 --prefill-n-steps 8 \
--prefill-system-protect 128 --prefill-tail-protect 32 \
--adaptive-residual --adaptive-residual-threshold 0.05 --adaptive-residual-n-steps 2 \
--decode-protect-steps 3 --decode-protect-layers last_4 \
--decode-gap-threshold 0.5 --refresh-interval 8 \
--residual-cos-threshold 0.9999
```

### 1.2 实验树

```
3ff30e6 (Phase 2 前，无 1-bit sign residual)
  ├── 配置：n=5, p=3, ctg=2, 无 1-bit sign
  └── fork=41, char=0.4470, pref=195

Phase 2 (+1-bit 符号残差)
  ├── 配置：n=5, p=3, ctg=2, 1-bit sign residual
  └── fork=49 ★, char=0.5094, pref=227

    ├── [实验 1] +CosSim 门控 + refresh_interval=8 (Phase 2b 全量)
    │   ├── CosSim 日志：所有 tile cos_sim ∈ [0.96, 0.998]，全 < 0.9999阈值
    │   │   → 门控全触发，未跳过任何 tile，零 bit 节省
    │   └── fork=49, char=0.5285, pref=227
    │       → char_match +3.7%（可能是 refresh_interval 的 FP16 周期刷新贡献）

    ├── [实验 2] +confidence_mask + confidence_beta=0.5
    │   ├── 惩罚幅度仅 0.075–0.175（vs logit 范围 ~2.7）→ 完全无力阻止 flip
    │   └── fork=49，零效果

    ├── [实验 3] +n_steps=6 + decode_protect_steps=8
    │   ├── 预期 fork 推迟；实际 fork 提前到 41
    │   └── fork=41, char=0.4989, pref=195 ← 退步
    │       → n_steps 增加改变所有 token 的编码分布，噪声频谱变动后
    │          敏感步骤提前触发分叉

    ├── [实验 4] +adaptive_n + beta_decay (ctg=2)
    │   ├── adaptive_n：per-tile 按 L∞ 误差分配更多步数
    │   ├── beta_decay：前 64 步 β 从 0.12 衰减到 0.03
    │   └── fork=34, char=0.3835, pref=167 ← 退步
    │       → 改变了整体误差频谱，分叉更早触发

    ├── [实验 5] +use_fwht + ctg=1 (+ v_orthogonal_transform)
    │   ├── fork=6 ← 灾难性退化
    │   ├── 根因：FWHT 的 apply_transform / apply_inverse_transform 是空桩
    │   │   （仅 identity passthrough + reshape），
    │   │   但 transform_decisions/padding 账户全部激活，破坏了数据 shape
    │   └── ctg=1 同时启用了 v_orthogonal_transform (且与 ctg 交互不兼容)

    └── [实验 6] +adaptive_n + beta_decay + ctg=1 (去掉 fwht)
        └── fork=6（同样灾难 → ctg=1 本身是崩溃点）

    ├── [实验 7] 500 token 长文压力测试 (refresh_interval=8)
    │   ├── 禁 logits-diff，只跑基线
    │   ├── fork=49（同 100 token 测试），fork 之后开始严重重复
    │   └── char=0.2908, rep=0.1792
    │       → refresh_interval 无法阻止前 49 步的误差累积

    └── [实验 8] 200 token 测试 (refresh_interval=0 未跑)
        └── fork=49, char=0.2908, rep=0.1792（一致）
```

### 1.3 核心结论

| 现象 | 结论 |
|------|------|
| **唯一正收益方案** | Phase 2 的 `1-bit sign residual`（fork: 41 → 49，+19.5%） |
| **所有后续实验** | 零效或退步。加更多 bit 不解决问题 |
| **fork 位置可变** | 每次改动编码方式，fork 在不同位置重新出现（6 / 34 / 41 / 49） |
| **JS divergence 轨迹** | 前 33 步噪声 < 0.0001，step 34 开始 exponential 爆发放大 |
| **transform stub 问题** | FWHT/DCT/DWT 被明确标记"removed (experimentally destructive)" |
| **1-bit 量子的物理极限** | n=5 后 cos_sim ≈ 0.985，要到 0.998 需要 ~10 步 = 位宽翻倍 |
| **单一 prompt** | 多次实验 fork 日志数据完全一致，即已充分体现稳定性 |

---

## 二、根因分析

### 2.1 Σ-Δ 噪声是伪随机但非白噪声

Σ-Δ 编码产生的是**结构化量化噪声**——积分器把低频误差推到高频，相邻坐标的误差高度负相关。当一个 tile 的 d_head=64 的误差模式呈现 `[+1, -1, +1, -1, ...]` 的规律，而某个 query vector 恰好与这个振荡模式对齐，点积就会**放大**而非抵消误差。

这正是为什么：
- `n_steps=5` 优于 `n_steps=6`：不同步数产生不同的噪声频率分布。n=5 的频谱恰好与敏感 query 错开。
- 所有改变编码结构的操作（加步数、adaptive_n、beta decay）都会换一种噪声频谱。新频谱**可能更早**撞上敏感 query。

### 2.2 误差是非线性放大的

JS divergence 时间线显示 `step 33: ~0 → step 34: 0.0045 → step 49: 0.66`。误差不是线性增长，而是 exponential——attention 机制把量化误差逐层、逐头、逐步放大。

```
n_steps=5 路径: ERROR → step 0-33: 潜伏期 → step 34: 爆发初期 → step 49: 分叉
n_steps=6 路径: ERROR → step 0-33: 潜伏期 → step 34: 爆发初期 → step 41: 分叉
```

误差结构不同步数不同，但最终都会在某个"敏感步骤"处 exponential 爆炸。

### 2.3 1-bit Σ-Δ 在 h=64 上的物理极限

```
1 个 1-bit base 修正幅度 ≈ 1 / sqrt(64) ≈ 12.5% of signal RMS
n=5 后残余误差 ≈ 1.5% → cos_sim ≈ 0.985–0.995
n=6 后残余误差 ≈ 1.0% → cos_sim ≈ 0.990–0.997
n=∞ 理论极限处 ≈ 0.2% → cos_sim ≈ 0.998–0.999+
```

**每增加 1 步的边际收益递减**（第 n 步跟第 n+1 步的幅度差距越来越小）。n=5 之后再加步数，每步只改善零点几个点的 cos_sim，却要付出 +1 bit/element 的位宽代价。在本文的 baseline 压缩比 5.31 bit/element 中，每步 = ~19% 的开销，而精度提升微乎其微。

**根本问题不是精度不够，而是 Σ-Δ 的结构化噪声与模型 attention 的模式对齐性不可预测。**

### 2.4 变换域方案失败的原因

| 变换 | 失败原因 |
|------|---------|
| **DCT** | 低频 DC 系数幅度远大于 AC 系数 → Σ-Δ 的 ±1 量子对 AC 系数是毁灭性过量化 |
| **DWT** | LL 子带信号远大于 LH/HL/HH 子带 → 细节子带被 ±1 彻底摧毁 |
| **FWHT** | 均匀扩散误差 → Σ-Δ 无法集中火力；目前是空桩，毫无实施 |
| **v_orthogonal_transform** | 在 ctg=1 下启用时与其他组件交互不兼容 |

---

## 三、前进方向

### 3.1 Per-head 自适应位宽分配（低风险 / ~100 行代码）

**核心想法：** 不是所有 attention head 同等重要。约 20% 的 head 拿 80% 的 attention 权重。给这些"attention sink head"多分配步数（6步），给低权重 head 少分配（4步），整体位宽不变但精度针对性地提升。

**实施要点：**
- 在 prefill 后统计每个 head 的平均 attention 权重
- 根据权重排名分配 η different n_steps_per_head
- 解码时每个 head 独立配置

**风险：** 中—低。改的是位宽分配策略而非编码算法本身，与 Σ-Δ 兼容。

### 3.2 噪声脱钩器（低风险 / ~50 行代码）

**核心想法：** 编码前给 tile 加点白噪声（nondeterministic dither），破坏 Σ-Δ 噪声的结构化模式，让它变成期望为零的白噪声。白噪声在 attention 中不产生系统性偏置。

```python
# 在 _encode_and_append_tile 中，tile 进入编码之前：
dither = torch.randint(0, 2, tile.shape) * 2 - 1  # ±1 random
tile_dithered = tile + dither_amplitude * dither
# 编码 tile_dithered，解码时减去 dither
```

**风险：** 极低。改的是噪声结构而非信号信号，+0.1% 额外噪声换取 -1000% 系统性偏置。

**验证：** 用现有 Test A 框架评估 dither_amplitude = 0.1 / 0.2 / 0.5 三个等级的 fork_step 变化。

### 3.3 低精度浮点残差（中风险 / ~200 行代码）

**核心想法：** 当前的 1-bit sign residual 也是 ±1 量子，跟主编码同样是结构化的。换成 FP4 E2M1 残差——2 位精度，但误差是白噪声（舍入误差，与信号独立）。白噪声在 softmax 中期望值约等于零。

```
n_steps=3 主编码（暴力压缩）
+ FP4 E2M1 残差（4 位 = 2 per element，但白噪結構）
+ refresh_interval=4
```

**比较：**
- 当前：n=5（5.31 bit） + 1-bit sign（常触发 = 6.37 bit） ≈ 6.37 bit → fork=49
- 方案：n=3（3.19 bit） + FP4（5.19 bit）≈ 5.19 bit → fork?

**风险：** 中。FP4 需要新增量化和解量化逻辑，且需要验证"更少 bit + 更白噪声"是否优于"更多 bit + 结构化噪声"。

### 3.4 Per-head SVD 旋转 + 主成分优先编码（中风险 / ~150 行代码）

**核心想法：** 用 K 的 SVD 对 V 做旋转（当前 `v_orthogonal_transform` 的部分思路），让主方向（前几个奇异值）得到更多编码步数，小方向（尾部奇异值）可以用更少步或跳过。最终重建时反旋转回来。

**跟当前 `v_orthogonal_transform` 的区别：**
- 当前版本旋转了 V，但用相同 n_steps 编码所有方向
- 新方案旋转后用**可变步数**编码主/小方向

**风险：** 中。需要验证跟 ctg>1 的交互兼容性（当前 SVD 在 ctg>1 下被 auto-disable）。

### 3.5 大模型验证（设施需求）

**核心想法：** 1B 模型的 h=64 自由度小，同一个 h 要处理 8 个 head 的注意力，信息密度高。更大的模型（3B+）h=128 或更多，每个 head 有更多自由度来分散误差。

**验证方法：** 拿 3B 或 7B 模型跑同一套 Test A 框架，看 fork_step 是否自然推后。

---

## 四、优先级建议

| 优先级 | 方案 | 理由 |
|:---:|------|------|
| **P0** | 噪声脱钩器（dither） | 5分钟改完 + 1小时测试 = 就能锁定根因假设 |
| **P1** | Per-head 自适应 n_steps | 若 dither 验证了"结构 vs 白噪声"的假设，这里用位宽分配进一步优化 |
| **P2** | FP4 低精度残差 | 若 dither 有效，可替换 1-bit sign 为 FP4 残差 |
| **P3** | SVD + 主成分优先 | 高层方案，取决于前三个的结果 |
| **P4** | 大模型验证 | 依赖 GPU 资源 |

**建议执行顺序：**
1. 先跑 dither 验证——如果 fork_step 有明显改善（5-10 步以上），整个"白噪声 vs 结构化噪声"的假设成立
2. 按多 prompt 跑 per-head 方案
3. FP4 残差
4. 上大模型

---

## 五、附录：JS divergence 完整时间线

```
Step  fork=49 路径：
  0–33: JS < 0.00001  → 误差在潜伏期，量不到
  34:   JS = 0.0045   → 首次跳变（某层某 head 的误差被 attention 放大）
  35–48: JS 0.001–0.02 → 震荡累积，各层 cross-head 相互作用
  49:   JS = 0.66 ≈ ln(2) → 分叉，logits 完全不相关

Step  fork=34 路径 (adaptive_n + beta_decay):
  0–32: JS < 0.00001
  33:   JS = 0.002
  34:   JS = 0.005 + fork
  → 更早进入爆发阶段

Step  fork=6 路径 (ctg=1):
  0–5: JS < 0.0001
  6:   JS = 0.006 + fork
  → 完全崩溃
```
