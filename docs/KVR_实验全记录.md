# KVR — 全实验记录与架构文档

## 概述

KVR（Key‑predicted Value Retrieval）是一种替代 Transformer 全量注意力的方案。核心思想：用固定大小的窗做精确局部注意力，用压缩 K 索引 + 学习 V 预测做远处检索。注意力复杂度永远 O(2048 + 128)，不随上下文长度增长。

**关键策略：懒初始化。** 上下文 ≤ 窗大小时退化为纯窗注意力（无检索无额外开销），首次检测到窗口 eviction 时才构建检索索引。

---

## 1. 命名演变

| 阶段 | 名称 | 说明 |
|:----|:----|:------|
| Phase 1‑3 | FWR (Field‑Window‑Retrieval) | 包含场累积器 |
| Phase 4‑5 | FWR (Fused‑Window‑Retrieval) | 场被移除 |
| 最终 | **KVR** (Key‑predicted Value Retrieval) | 核心创新：V 从 K 预测 |

---

## 2. 架构

### 2.1 三阶段动态策略

```
阶段 1（顺风局）  context ≤ window_size
  行为：纯窗注意力（fp16 精确）
  检索：未构建，零额外开销
  复杂度：O(2048)

阶段 2（切换点）  首次 context > window_size
  行为：从窗口反向 RoPE 获取 K_pre，一次性构建检索索引
  开销：O(window_size)，约 1-2ms 的一次性触发

阶段 3（持久战）  context > window_size
  行为：窗 + 检索级联注意力
  检索包含所有已被窗口驱逐的 token（int4 K + W 预测 V + int2 V 残差）
  复杂度：永远 O(2048 + 128) = 固定 2176 点
```

### 2.2 组件

| 组件 | 文件 | 精度 | 存储量(1B@128K) |
|:-----|:-----|:-----|:---------------|
| WindowBuffer | `kvr_window.py` | fp16 K+V | 64 MB (2048 tok) |
| RetrievalIndex | `kvr_retrieval.py` | int4 K + int2 V_res | ~768 MB (全部历史) |
| KVRHook | `kvr_hook.py` | — | 模型无关，无 LLaMA 依赖 |
| KVRGenerator | `kvr_generator.py` | — | 增量生成循环 |

### 2.3 V 预测流水线

```
calibrate(k_pre, v):
  1. 从 prefill 数据拟合 W: V = W·(K − μ_k) + μ_v
  2. 计算 V_residual = V_true − V_pred
  3. 对 V_residual 每头 int2 量化, 存储 vr_scales (n_kv×1)

append(k_pre, v):
  1. int4 量化 K_pre → k_codes
  2. V_pred = W·(K_pre − μ_k) + μ_v
  3. V_residual = v − V_pred
  4. int2 量化 V_residual → vr_codes

retrieve_topk(q_post_rope):
  1. 反量化 k_codes → K_pre → 旋转(K_pre, pos_id) → K_post
  2. Q_post·K_post / √d → top-K（排除窗内位置）
  3. 对 top-K: V_final = W·(K_pre − μ_k) + μ_v + 反量化(vr_codes)
  4. 返回 K_post + V_final 给级联注意力
```

### 2.4 懒检索触发机制

```python
def _update_stores(self, li, k_pre, v_val, k_post):
    win = self.windows[li]
    if self._retrieval_built and win.n >= win.cap:
        # Token 将被驱逐：反向 RoPE 后追加到检索
        slot = win.pos % win.cap
        old_k_post = win.k[slot]
        old_v = win.v[slot]
        old_k_pre = _reverse_rotary(old_k_post, cos, sin)
        self.retrievals[li].append(old_k_pre, old_v)
    elif not self._retrieval_built and self._context_len >= self.window_size - 1:
        self._build_all_retrievals()  # 从窗口一次性构建
    win.append(k_post, v_val)
```

### 2.5 块预填充（Block Prefill）

不再使用 `model(input_ids)` 做全量 forward。自定义逐块 forward：

```
外层循环：16 层
  内层循环：128 块（64K ÷ 512）
    1. layernorm(512 tok) → Q/K_pre/V 投影
    2. RoPE (per-head expand)
    3. 累积 K/V 到临时张量
    4. 分组 softmax 注意力（在线 safe softmax，chunk=512）
    5. o_proj + 残差 + MLP
```

**显存：** scores 张量永不过 O(512 × n_kv × g × 512) = ~16MB，不随上下文增长。

### 2.6 增量生成（KVRGenerator）

```python
gen = KVRGenerator(model)
gen.prefill(input_ids)       # 块预填充，只建窗（不建检索）
for step in range(max_new):
    token = gen.step()       # 增量: 1 token 过 16 层
```

步进中：
- `_is_first=True` 时不存储 K/V（最后一个 prompt token 已存在）
- `_update_stores` 触发懒检索构建（若 context > window_size）
- 级联注意力：窗 fp16 K/V + 检索 int4 K/V（含 V 预测 + 残差）

---

## 3. 所有关键 Bug 与修复

| # | Bug | 影响 | 修复 |
|:-:|:----|:----|:------|
| 1 | Hook 签名必须 `with_kwargs=True` | hook 收不到参数 | 加 `with_kwargs=True` |
| 2 | o_proj 未作用在 FWR 输出上 | 输出维度不匹配 | FWR 输出再过一次 o_proj |
| 3 | Prefill 未用 input_layernorm | K/V 不匹配，JS 从 0.004→4.5 | prefill QKV 投影前过 layernorm |
| 4 | argsort 返回 (1, N) 而非 (N,) | 索引维度错 | `.squeeze(0).argsort()` |
| 5 | Exclude 全量时 argsort 仍返回 top-K | 窗口 token 被检索双加 | 加 `if exc≤0 and exc≥nr: return empty` |
| 6 | 第一步重复存储 last prompt K/V | 双重计数 | `_is_first=True` 跳过第一次 append |
| 7 | Exclude 终点用 _context_len 而非 n_stored | 当前 token 未被排除 | 改为 `exc_e = n_stored` |
| 8 | `view(n_kv, g, bsz, d)` 内存布局错乱 | 注意力完全错位 | 改为 `reshape(bsz, n_kv, g, d)` |
| 9 | cos/sin 未按 head 数 expand | RoPE 维度不匹配 | expand 后再 reshape |
| 10 | window batch_append 覆写而非累积 | evict 前丢失旧 token | 改为完整 circular buffer |
| 11 | 层/块循环顺序错误 | 残差连接错位 | **块内层**改为**层内块**（最外层=层）|
| 12 | `einsum('b n g t, t n d')` 产生 ~1TB 中间变量 | 16K OOM | 改为 per-KV-head matmul |
| 13 | Gruppen softmax 因果掩码错置 | scores 被错误冻结 | 仅对 `cstart >= n_past` 的块施加掩码 |
| 14 | 懒检索构建时取的是窗口内容（错误） | 检索包含窗外 token | 用 `captured_k[:n_ret]` 取窗口外 token |

---

## 4. V 预测消融实验

| 方案 | V cos 最低 | 结论 |
|:----|:---------:|:-----|
| int2 V per-dim | 0.72 | ❌ |
| int4 V per-head | 0.987 | ✅ 可接受 |
| W 预测 (K_pre→V) | 0.956 | ✅ 但 NIAH 67% |
| **W + int2 残差** | **0.994** | **✅ 最优** |

---

## 5. 存储

### 1B @ 128K

| 组件 | 精度 | 大小 |
|:----|:----|:----:|
| K int4 codes | 0.5 B/值 | 256 MB |
| V int2 residual codes | 0.25 B/值 | 128 MB |
| W 矩阵 | fp16 | 8 MB |
| 窗 fp16 K+V | fp16 | 64 MB |
| **总计** | | **~456 MB** |

### 12B @ 128K（推算）

| 组件 | 大小 |
|:----|:----:|
| K int4 codes | 1.0 GB |
| V int2 residual | 0.5 GB |
| W 矩阵 | 80 MB |
| 窗 fp16 K+V | 268 MB |
| **总计** | **~1.9 GB** |

---

## 6. 注意力与加速

| 上下文 | 全量注意力 | KVR | 加速比 |
|:------|:--------:|:---:|:-----:|
| 2K | 4.2M FLOPs | 4.2M | 1x |
| 32K | 67.1M | 4.2M | **16x** |
| 128K | 268.4M | 4.2M | **64x** |

KVR 永远 2176 个 attention 点（2048 窗 + 128 检索），不随上下文增长。

---

## 7. 测试结果

### 7.1 NIAH（上一版，非懒检索）

| 配置 | 通过率 |
|:----|:------:|
| 128‑2048 ctx, win=64 | 100% (15/15) |

### 7.2 极端长上下文（当前版，懒检索 + 分组 softmax）

| 上下文 | Native | KVR |
|:-----:|:-----:|:---:|
| 4K | ✅ 1.3s, 2.69 GB | ✅ 2.6s+6.3s, **3.60 GB** 文本一致 |
| 16K | ✅ 3.2s, 3.82 GB | ✅ 16.8s+20.5s, **4.14 GB** |
| 64K | ❌ OOM | ✅ 224s+230s, **7.05 GB** |

### 7.3 AR 生成（短文本）

| 模式 | JS | 文本 |
|:----|:--:|:-----|
| window=2048 + ret | 0.0000 | 与 native 逐 token 相同 |
| window=64 + ret | ~0.8 | 前 34 tok 相同，后分叉但流畅 |

---

## 8. 文件结构

### KVR 核心文件

```
modules/
  kvr_window.py        — WindowBuffer: fp16 K/V 循环缓冲
  kvr_retrieval.py     — RetrievalIndex: int4 K + int2 V 残差 + W 预测器
                        _apply_rotary / _rotate_half / _reverse_rotary
  kvr_hook.py          — KVRHook: 块预填充 + 懒检索 + 分组 softmax + hook 注入
  kvr_generator.py     — KVRGenerator: 增量生成循环（无 LLaMA 依赖）
```

### 评估脚本

```
scripts/evaluation/
  eval_kvr_extreme.py   — 极端长上下文测试（4K/16K/64K）
  eval_fwr_niah.py      — NIAH 框架（fwr 命名遗留，暂未改）
  eval_fwr_wres.py      — W+int2 残差综合测试
  eval_kvr_long.py      — 长上下文 AR 测试
  eval_kvr_scenarios.py — 6 风险场景测试
  eval_v_residual.py    — V 预测消融实验
  eval_v_quant_bits.py  — V 量化 bit 扫描
  eval_v_from_k.py      — K→V 线性预测验证
  eval_v_from_kpost.py  — post-RoPE K→V 验证
  eval_v_cache_test.py  — V 缓存 bit 宽度测试
```

### JSON 结果文件（可删）

```
eval_kvr_extreme.json
eval_kvr_long.json
eval_kvr_scenarios.json
eval_fwr_*.json         （多个）
```

### 文档

```
docs/
  KVR_实验全记录.md      — 完整性文档（本文）
  FWR_ARCHITECTURE.md   — 英文旧版（可删）
```

---

## 9. 已知局限

1. **AR 路径分叉**：当 `window_size ≪ context` 时，int4 K 搜索可能将相似段 token 排序错误。分叉后文本始终语法正确、语义合理。
2. **V 预测仅从 prefill 拟合**：W 矩阵只在 prefill 时拟合一次。风险低但需验证。
3. **64K prompt 极度重复时生成质量下降**：非 KVR 问题，换非重复 prompt 应恢复正常。
4. **Block prefill 比原生慢**（4K: 2.6s vs 1.3s）。Python 层循环开销，可用 `torch.compile` 加速。
