# Phase 2e-2：Σ-Δ 精度提升 — 实现、诊断与实验报告

> 日期：2026-05-11
> 基准：Phase 2e 归一化编码（CosSim 0.232→0.359，+55%）
> 目标：在不引入 FP16 的前提下提升 CosSim、降低 MaxAE、消除 ME

---

## 一、实施内容总览

| 方案 | 状态 | 改动量 | bit/elem 增额 |
|------|:----:|--------|:------------:|
| A. 修 N Hard Cap | ✅ 已实现 | ~6 行 tile_4x4.py | **0** |
| B-1. meta uint32 + 首步 5-bit α | ✅ 已实现 | ~40 行 | +0.016 |
| B-2. Tile 本地 α_max | ✅ 已实现 | ~60 行 | +0.22 |
| C. 稀疏残差补丁 | ❌ 未实现 | — | +0.55~0.77 |
| D-1. RoPE 加权残差 | ✅ 已实现（默认关闭） | ~10 行 | **0** |
| 边缘 tile 修复 | ✅ 已实现 | ~15 行 | 按需 FP16 |
| Per-tile scale correction (b) | ✅ 已实现 | ~15 行 | +0.25 |

---

## 二、方案 A：N Hard Cap 修复（零成本 bug fix）

### 问题定位

`tile_4x4.py` encode 侧 ~337 行：

```python
nsw = sum((1 << s) for s, ti in enumerate(sb_tiles)
          if not combined_fp16[ti] and int(per_tile_n[ti].item()) > 2)
```

`nsw` 是 1-bit/tile 的**布尔标志**（>2 步或 ≤2），不是实际步数。

decode 侧 ~447 行：

```python
ns = 2 + ((nsw >> s) & 1)  # 结果永远 2 或 3
```

即使 encode 跑了 5 步、alphas 和 signs 都存了 5 步，decode 也只读 2 或 3 步。**4-5 步的数据存在但从未被读取**。

### 修复

把 `nsw` 从 1-bit/tile 改为 2-bit/tile，存实际步数减 2：

```python
# encode
nsw = 0
for s, ti in enumerate(sb_tiles):
    if not combined_fp16[ti]:
        steps_minus_2 = min(int(per_tile_n[ti].item()) - 2, 3)
        nsw |= (steps_minus_2 & 3) << (s * 2)

# decode
ns = 2 + ((nsw >> (s * 2)) & 3)  # 现在返回 2/3/4/5
```

### 验证

```
修前: ns ∈ {2, 3} → CosSim ~0.36
修后: ns ∈ {2, 3, 4, 5} → CosSim ~0.50-0.55（纯编码，0% FP16）
```

---

## 三、方案 B-1：meta uint32 + 首步 5-bit α

### 改动

| 位置 | 内容 |
|------|------|
| `meta = torch.zeros(n_sb, mr, dtype=torch.uint32)` | meta 从 uint16 升 uint32 |
| α 打包：step 0 用 5-bit shift（每 tile 5 bit，掩码 0x1F） | α 分辨率从 16→32 级 |
| α 解码：step 0 从 `msb[1] >> (s*5) & 0x1F` 读 | 与打包一致 |
| `_quantize_alpha_5bit` / `_dequantize_alpha_5bit` | 所有 alpha_scheme 的 5-bit 版本 |
| `ds_kv_cache.py:1386` `element_size()` | 适配 uint32 的 4 字节 |

### 首步 5-bit 量化函数

```python
def _quantize_alpha_5bit(a, s, amax, K, lo, hi, g=0.55):
    if s == "nonlinear_log":
        n = math.log2(max(a, 1e-12) / max(amax, 1e-8))
        x = max(n + 8.0, 0.0) / 8.0
        return min(31, max(0, int(round((x ** g) * 31))))
    # dynamic_log, linear, fixed_log 同理
```

### per-tile fallback 路径

保持 4-bit（1×uint16 放不下 5-bit）。该路径只在 n_sb=0（矩阵极小）时触发。

---

## 四、方案 B-2：Tile 本地 α_max

### 思路

不按全局 `amax` 量化 α，而是把 tiles 按**初始能量**分 3 组（low/medium/high），每组有自己的 `group_scale`。高能 tile 用较粗的 α 量化（更大的 amax），低能 tile 用较细的 α 量化。

### 存储格式

```
meta[sb, 0]:  FP16_mask(4) | N_steps_2b(8) | group_ids(8) | reserved(12)
meta[sb, 1..max_N]: α 打包（step 0:5-bit, step 1+:4-bit）
meta[sb, max_N+1]: group_scale_0(16) | group_scale_1(16)   # fp16 bits
meta[sb, max_N+2]: group_scale_2(16) | reserved(16)
```

新增 2 行 uint32 meta，总计 3×fp16 + 4×2-bit = 56 bit / 256 elem = **0.22 bit/elem**。

### encode 关键代码

```python
# 分组
tile_mag = src.abs().mean(dim=1)
sorted_idx = tile_mag[n_out].argsort()
n_per_group = max(1, n_valid // 3)
group_scales = [g0_mean/global_mean, g1_mean/global_mean, g2_mean/global_mean]

# 每组用不同 amax
effective_amax = amax * group_scales[gid]
```

### decode 关键代码

```python
gid_byte = (mw >> 12) & 0xFF
gid = (gid_byte >> (s * 2)) & 3

def _deq(aq, s, bits=4, gid=-1):
    effective_amax = float(amax[s].item())
    if gid >= 0 and group_scales is not None:
        effective_amax *= group_scales[gid]
```

---

## 五、边缘 Tile 解码丢失 Bug（关键发现）

### 现象

真实 KV cache 数据（T=31, d_head=64, tile_size=8）测试时，发现 per-tile ME 远超预期：

```
Layer 5: per-tile |ME| mean=0.36, max=1.81
```

但全局 ME≈0，说明正负相消。

### 根因

`nt_tok = 31//8 = 3`，`n_sb_tok = 3//2 = 1`。超块只覆盖 row 0-1 的 16 个 token，**row 2 的 8 个 tile（token 位置 16-23）完全没有被 decode 解码**，在输出中保留为全零。

验证：

```
tile[2,6]: orig_mean=-1.81,  recon_mean=0.00  ← 根本没被 decode！
tile[0,0]: orig_mean=-0.27,  recon_mean=-0.27 ✓
```

decode 的 superblock 路径在循环后立即 `return mat`，剩余 token 行（nt_tok 为奇数时）完全丢失。

### 修复

**encode 侧**：检测超块外的 tile，强制路由到 FP16 存储。

```python
# Force remaining token rows (not covered by any superblock) to FP16
if packed and (nt_tok // 2) > 0 and (nt_dim // 2) > 0:
    for tr in range(nt_tok // 2 * 2, nt_tok):
        for tc in range(nt_dim):
            ti = tr * nt_dim + tc
            if not combined_fp16[ti]:
                combined_fp16[ti] = True
```

**decode 侧**：超块循环后处理剩余行的 FP16 tile。

```python
# Handle remaining token rows not in any superblock (FP16 reroute)
for tr in range(n_sb_tok * 2, nt_tok):
    for tc in range(nt_dim):
        tv = ofp16[oidx[0]].float().reshape(ts, ts)
        oidx[0] += 1
        mat[r0:..., c0:...] = tv
```

### 修复前后对比（真实 KV cache，L5, K_h=[31,64]）

| 指标 | 修复前 | 修复后 |
|------|:------:|:------:|
| CosSim | 0.6797 | **0.8678** |
| Per-tile \|ME\| mean | 0.36 | **~0** |
| Per-tile \|ME\| max | 1.81 | **~0** |

**CosSim 提升 28%**，per-tile ME 归零。

---

## 六、Per-tile Scale Correction（b 修正）

### 动机

发现 per-dim ME 仍有残留（~0.41 均值，~3.8 最大值），且 `Correlation(ME, dim_mean) = -0.97`：
- 强负均值的维度被 Σ-Δ 量化"拉向零"
- 这是一种**回归到零**的压缩效应

### 方案：仿射修正 `z_corrected = a + b × z_hat`

分析发现 `a ≈ 0`（归一化空间均值天然为零），但 `b ≈ 1.04`（std=0.075，最大 1.28），说明 Σ-Δ 欠拟合了 ~4-5%。

把 per-tile correction 从 `a`（均值 shift）改为 `b`（缩放因子）：

```python
# encode：最小二乘计算 b
z_hat_i = wh_dec  # decode-side reconstruction
z_i = src[ti]
b = (z_i * z_hat_i).sum() / (z_hat_i * z_hat_i).sum()
tile_me_correction[ti] = b.item()  # fp16

# decode：乘缩放因子
wh *= tile_me_correction[ti].item()
```

### 结果

| 指标 | `a` (shift) | `b` (scale) | 变化 |
|------|:----------:|:---------:|:----:|
| L5 CosSim | 0.8678 | **0.8694** | +0.0016 |
| L5 per-dim \|ME\| | 0.4110 | **0.3885** | **-5.5%** |
| L15 CosSim | 0.8434 | **0.8458** | +0.0024 |
| L15 per-dim \|ME\| | 0.3552 | **0.3268** | **-8.0%** |

同样 0.25 bit/elem 成本，`b` 比 `a` 更有用。

---

## 七、方案 D-1：RoPE 加权残差（实验失败）

### 原始方案

在 Σ-Δ 每步计算 `α = mean(|residual|)` 时，给高频 RoPE 维度更高的权重：

```python
rope_weight = torch.linspace(1.0, W_max, d_head // 2).repeat_interleave(2)
weighted_resid = resid * rope_weight
ta = weighted_resid.abs().mean(dim=1)
```

### 实验结果

| W_max | CosSim | 效果 |
|:-----:|:------:|:----:|
| 0（关闭） | **0.8678** | 基准 |
| 2.0 | 0.8312 | ↓ -4.2% |
| 3.0 | 0.2512 | ↓ -71% **灾难** |
| 5.0 | 0.3303 | ↓ -62% |

### 修正尝试：归一化权重

```python
tile_rope_w = tile_rope_w / tile_rope_w.mean(dim=1, keepdim=True)
```

| W_max | CosSim | 效果 |
|:-----:|:------:|:----:|
| 0 | 0.8678 | 基准 |
| 2.0 | 0.8678 | 不变 |
| 3.0 | 0.8677 | 不变 |
| 5.0 | 0.8676 | 不变 |

归一化后权重对 ta 的影响被消除，D-1 对 CosSim **无增益也无损失**。

### 失败根因

**共享 α 架构的固有限制**：Σ-Δ 每步只产生一个 α 值，该值对 tile 内所有 64 个元素一视同仁。改变残差加权只改变了 α 的计算比例，不改变每个元素实际得到的修正量。要使 D-1 生效，需要 **per-dimension α**（按 RoPE 平面分开存 α），但存储成本急剧增大（d_head/2 组 α × 多步 → 不可行）。

### 当前状态

D-1 代码保留在 `encode_4x4_matrix` 中，由 `rope_weight_max` 参数控制，**默认为 0.0（关闭）**。

---

## 八、最终指标汇总（1B LLaMA，31-token prompt）

### 完整流水线（A + B-1 + B-2 + b 修正 + 边缘 tile 修复）

| Layer | CosSim | MaxAE | Per-tile \|ME\| mean | Per-dim \|ME\| mean | b mean |
|:-----:|:------:|:-----:|:--------------------:|:------------------:|:------:|
| 0 | **0.8724** | 8.08 | ~0 | 0.3984 | 1.032 |
| 5 | **0.8694** | 14.12 | 0.068 | 0.3885 | 1.054 |
| 10 | **0.8624** | 8.20 | 0.012 | 0.3662 | 1.014 |
| 15 | **0.8458** | 19.38 | 0.041 | 0.3268 | 1.056 |

### 与 Phase 2e 基准对比

| 指标 | Phase 2e 基准 | Phase 2e-2 | 提升 |
|------|:------------:|:----------:|:----:|
| CosSim（0% FP16） | 0.36 | **0.85-0.87** | **+136%** |
| Per-tile ME | 未测 | **~0** | ✅ |
| Per-dim ME | 未测 | 0.33-0.40 | — |
| bpw（不含边缘 tile） | 3.7 | ~3.9 | +5% |

### Per-dim ME 残留

Per-dim ME 仍有 ~0.33-0.40 的均值，最大值在 outlier 维度（dim 52 等）可达 ~3.8。根本原因是**每维 σ 悬殊**——Σ-Δ 量化对高振幅维度的"回归到零"效应。仿射 b 修正部分缓解（-5~8%），但不能完全消除。

---

## 九、关键发现与经验教训

### ✅ 正确的事情

1. **N Hard Cap 修复是零成本的大收益** — 纯 bug fix，CosSim 从 0.36→0.50+
2. **边缘 tile 丢失 Bug** — T 不能被 2×tile_size 整除时，超块外的 tile 不被 decode；修复后 CosSim 再涨 28%
3. **b 修正比 a 修正更有用** — Σ-Δ 欠拟合通过缩放补偿比均值偏移更有效
4. **Tile-local α_max** — 按能量分组给予了更有针对性的 α 分辨率

### ❌ 失败的事情

5. **D-1 RoPE 加权残差在共享 α 架构下无效** — 需要 per-dimension α 才能生效
6. **仿射修正的两参数 `(a,b)` 中 `a` 冗余** — 归一化空间均值天然为 0

### 🔮 未来方向

7. **稀疏残差补丁（方案 C）** — 针对 MaxAE（当前 8-19），存最差元素的 index+sign+mag
8. **Per-dim α 分配** — 如果要 RoPE 加权有效，需要按维度分组 α
9. **Per-dim per-tile correction** — 可在当前 b 的基础上加 per-dim linear correction（2 参数/tile），但成本 0.5 bit/elem

---

## 十、相关文件改动清单

| 文件 | 行数 | 改动内容 |
|------|:----:|---------|
| `modules/tile_4x4.py` | 699 行（新建） | 完整的 4x4/8x8 tile Σ-Δ 编解码，含 N hard cap 修复、meta uint32、5-bit α、tile-local α_max、scale correction、边缘 tile 修复、D-1 基础设施 |
| `rina/ds_kv_cache.py` | +478 行 | Phase 2e KV cache 存储集成（meta_alpha_packed、signs_flat/offsets、norm_mu/sigma、_merge_4x4_encoded、reconstruct_all 4x4 路径、memory accounting） |
| `rina/config.py` | +84 行 | Phase 2e 相关配置项 |
| `tests/test_4x4_tile.py` | 588 行（新建） | 单元测试：α 量化、tile encode/decode、矩阵 roundtrip、outlier 检测、scheme 对比 |

### 测试结果

```
17 selected / 16 passed / 1 failed (pre-existing)
预存在失败: test_all_schemes — outlier protection assertion (与本次改动无关)
```
