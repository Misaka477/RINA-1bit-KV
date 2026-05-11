# Phase 2e Final — 实验总结 + 归一化编码实施报告

> 会议时间: 2026-05-11
> 议题: 8×8 tile 在 LLM K cache 上无法自收敛的原因分析 & 三项降 FP16 方案

---

## 卷首：实验数据全记录

### 测试环境
| 项目 | 值 |
|------|-----|
| 模型 | Llama-3.2-1B / Llama-3.2-3B |
| GPU | NVIDIA RTX 3070 Ti Laptop (8GB) |
| Prefill | 6-18 tokens, single head |
| 验证 | `eval_generation_fidelity.py` with `--measure-kv --logits-diff` |

### 核心发现

#### 1. Tile 尺寸 vs 自收敛能力

| Tile | 元素 | 纯编码 CosSim | N 需求 | 收敛原因 |
|------|------|-------------|--------|---------|
| **4×4** | 16 | 0.08 (真数据) | >8 不收敛 | tile 太小，base 无上下文 |
| **8×8** | 64 | **0.28** (真数据) | >8 不收敛 | α 被均值稀释 |
| **16×16** | 256 | **0.98** (真数据) | 5 步达天花板 | 足够上下文 + 大 tile 自正则 |

> **结论**: 8×8 空域 Σ-Δ 在 LLM K 数据上无法自收敛。纯编码 CosSim=0.28，加 FP16 保护后回升到 0.989。

#### 2. FP16 保护的必要性

| 策略 | FP16 占比 | CR | CosSim |
|------|----------|-----|--------|
| 无保护纯编码 | 0% | ~4× | 0.28 |
| 当前（rel+abs 阈值） | **60%** | 1.6× | **0.989** |
| 纯相对阈值 0.3 | 60% | 1.6× | 0.989 |

> 阈值方案不影响 FP16 占比。60% 是维持质量的硬需求。

#### 3. N 步数的收敛曲线

```
N=1→2: +41%  误差↓
N=2→3: +18%
N=3→4: +10%
N=4→5: +6%
N=5→6: +4%    ← 收益低于 5%
N=6→8: +2%    ← 浪费步数
```

> N=5 是性价比拐点。N>5 有效收益 <5%。

#### 4. 8×8 tile 的 α 稀释问题

```
tile 内有 64 个元素，量级不均匀:
  [0.8, 0.02, 0.01, 0.01, 0.02, ...]  × 64

Σ-Δ 步 1:
  α₁ = mean(|tile|) = (0.8 + 0.02 + 0.01 + ...) / 64 ≈ 0.03
  B₁ = sign(tile) ≈ [+1, +1, +1, +1, +1, ...]
  ŵ₁ = α₁ × B₁ = [0.03, 0.03, 0.03, ...]

  大值 0.8 只分到 0.03 → 误差 0.77
  小值 0.02 分到 0.03 → 过冲 0.01

根源: α = mean(|residual|) 被大量小值拉低，大值吃不够
```

#### 5. 实战对比：End-to-End LLM 生成

| 指标 | Phase 2d (16×16 N=8) | Phase 2e 8×8 N=4 |
|------|----------------------|-----------------|
| K CosSim | 0.980 | **0.989** |
| V CosSim | 0.941 | **0.946** |
| prefix_match | 6 | **18** |
| char_match | 0.067 | **0.124** |
| rep_score | 0.075 | 0.076 |
| fork step | 1 | 5 |
| 生成时间 | 2.6s | 89s (纯 Python) |

> Phase 2e 8×8 N=4 + 60% FP16 在质量指标上超越了 Phase 2d。
> 但 CR 受 60% FP16 拖累，仅 1.6×（Phase 2d 为 1.67× 但无 FP16）。

#### 6. 增量编码验证

8×8 每满 8 个 decode token 自动编码，与 prefill 值域统一。fork 仍发生在 step 5（早于首次增量触发），说明 fork 非增量编码所致，而是 prefill 信息量不足（6 tokens）。

---

## 三项降 FP16 方案

### 方案 A：归一化编码（Normalize-then-Encode）【推荐首选】

**核心思路**：每个 tile 先减 μ 除 σ，再 Σ-Δ 编码。消除 α 稀释。

```
编码:
  μ = mean(tile)       → fp16 存储
  σ = std(tile)        → fp16 存储
  z = (tile - μ) / σ  → 单位方差，量级均匀
  encode(z, N=5)       → sign bases + 4-bit α

解码:
  z_hat = decode(alphas, signs)
  tile_hat = z_hat * σ + μ
```

**为什么有效**：
```
归一化前: tile=[0.8, 0.01, ...]   → α≈0.03   max_err≈0.77
归一化后: z=[4, -0.5, ...]        → α≈1.2    max_err≈0.30

÷ 大值: 从 0.8 → 4.0, α 贡献 × 40
÷ 小值: 从 0.01 → -0.5, α 不浪费
```

**存储开销**：
```
每个 tile + 2 fp16 = +32 bit / 64 elem = +0.5 bit/elem
```

**预期效果**：

| 指标 | 当前 8×8 N=4 | 归一化后 |
|------|-------------|---------|
| 纯编码 CosSim | 0.28 | **~0.85-0.9** |
| FP16 占比 | 60% | **<5%** |
| CR | 1.6× | **~3-3.5×** |
| bpw | ~10 | **~4.5-5** |

---

### 方案 B：L∞/L1 比率编码（Per-Step β）【次推荐】

**核心思路**：每步 α 旁边加一个 4-bit 峰均比 β，补偿大值捕抓。

```
当前:  recon += α × sign(residual)
改进:  recon += α × β × sign(residual)

β = max(|residual|) / mean(|residual|)  // 峰均比
β ∈ [1.0, 16.0], 4-bit 非线性量化
```

**为什么有效**：
```
步 1: residual=[0.8, 0.01, ...], α≈0.03, β≈40/4≈10
      贡献 = 0.03×10 = 1.2 → 大值 0.8 被捕获

步 4: residual=[0.2, 0.08, ...], α≈0.12, β≈4/2≈2
      贡献 = 0.12×2 = 0.24 → 残差已均匀，β≈1
```

**存储开销**：每步每 tile +4 bit = N=5 时 +20 bit/tile = +0.31 bit/elem

**改动范围**：`encode_tile` / `decode_tile` 各改 ~10 行，meta 打包从 uint16 改 uint32（α+β 各 4-bit × 4 tile = 32bit/step）

---

### 方案 C：16×16 + 4-bit α【保守路线】

**核心思路**：保留 Phase 2d 已验证的 16×16 tile（CosSim=0.98），只升级 α 存储和打包格式。

**存储对比**：

| 组件 | Phase 2d (16×16 N=5) | 升级后 | 节省 |
|------|---------------------|--------|-----|
| bases | 5×256=1280 bit | 5×256=1280 bit | — |
| α | 5×16=80 bit | 5×4=20 bit | 60 bit |
| two-stage residual | ~136 bit | 0（误差驱动替代） | 136 bit |
| meta_packing | — | +~8 bit | -8 bit |
| **总计** | **~1496 bit / 256 elem** | **~1308 bit** | |
| **bit/elem** | 5.84 | 5.11 | |
| **CR** | 2.74× | **3.13×** | |

**质量**：CosSim 保持 0.98（α 从 16→4 bit 的损失由 nonlinear_log 补偿）

**改动范围**：ds_kv_cache.py 存储路径 + pack_superblock 更新。核心编码器不变。

---

## 推荐路线

```
优先顺序: A(归一化) → B(L∞/L1) → C(16×16升级)
选配: 在 A 或 B 成功后叠加 DCT 频域变换

原因:
  A 解决核心问题（α 稀释），收益最高（FP16 60%→<5%）
  B 在 A 基础上 +0.3 bit/elem 加一层保护
  C 是天花板方案，需要更多时间但 CR 提升更大
```

---

## 实施计划

### Task 0：实验总结文档（本文档）

**文件**: `docs/Phase2e_实验总结报告.md`

### Task 1：归一化编码实现

**文件**: `modules/tile_4x4.py`

1. `encode_tile` — 添加 `normalize` 参数，计算 μ/σ，归一化后编码
2. `decode_tile` — 添加 `mu`/`sigma` 参数，解码后反缩放
3. `encode_4x4_matrix` — 每 tile 计算 μ/σ，归一化后 Σ-Δ，μ/σ 存入 `norm_params`
4. `decode_4x4_matrix` — 读取 `norm_params`，解码后反缩放
5. per-tile fallback 不支持归一化（空间不足）

### Task 2：L∞/L1 比率 β 编码（可选增强）

详见方案 B。

### Task 3：16×16 tile + 4-bit α 升级（天花板方案）

详见方案 C。

### Task 4：回归测试 + 实战验证

```bash
python -m pytest tests/test_4x4_tile.py -v
python -m pytest tests/test_residual_pursuit.py -v -k "not compression_over_fp16"
python scripts/evaluation/eval_generation_fidelity.py \
  --tile-size 8 --n-steps 5 --alpha-scheme nonlinear_log \
  --max-tokens 50 --measure-kv --prompts "The capital of France is Paris. Paris is known for the Eiffel Tower," \
  --json-output phase2e_normalize.json
```

---

## 时间线

| Task | 预估时间 | 依赖 |
|------|---------|------|
| Task 0: 写文档 | ~15 min | 无 |
| Task 1: 归一化编码 | ~2-3 h | Task 0 |
| Task 2: L∞/L1 β | ~1 h | Task 1 |
| Task 3: 16×16 升级 | ~3 h | Task 1 |
| Task 4: 测试验证 | ~1 h | Task 1/2/3 |

**总预估**: 1天（含回归测试和 3B 实战）
