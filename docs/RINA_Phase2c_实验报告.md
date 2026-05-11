# RINA Phase 2c：噪声脱钩与精度路线 — 完整实验报告

> 日期：2026-05-10 ~ 2026-05-11  
> 模型：Llama-3.2-1B (d_head=64)、Llama-3.2-3B (d_head=128)  
> 状态：**多 Prompt 验证暴露了精度上限**——n=5+sign 在单 prompt ("The capital of France is") 上无 fork，但多 prompt（5 组异构）下 3/5 fork。prefill-tail-protect=32 解决了早期 fork，但 decode 后期仍有残余 fork（step 39-68）。匹配追踪（MP）实验结论：fork 与噪声结构性无关，纯属精度不足。

---

## 测试命令参数速查

### 基础参数

| 参数 | 类型 | 默认 | 含义 |
|------|:----:|------|------|
| `--model` | str | LLaMA-3.2-1B | 模型路径 |
| `--max-tokens` | int | 50 | 生成的最大 token 数 |
| `--prompts` | str* | "The capital of France is" | 输入提示词 |
| `--json-output` | str | — | 结果 JSON 输出路径 |
| `--measure-kv` | flag | — | 启用 KV 缓存保真度测量 |
| `--logits-diff` | flag | — | 每步对比 native vs DS 的 logits，检测分叉点 |

### 编码参数

| 参数 | 类型 | 默认 | 含义 |
|------|:----:|:----:|------|
| `--n-steps` | int | 5 | Σ-Δ 主编码步数（解码阶段）。每步 ≈1.06 bit/element |
| `--prefill-n-steps` | int | — | Σ-Δ 预填充编码步数。None=用 n-steps |
| `--encode-mode` | str | sigma_delta | 编码模式：`sigma_delta`（Σ-Δ，默认）或 `matching_pursuit`（无积分器匹配追踪） |
| `--cross-token-group` | int | 2 | 跨 token 分组大小。1=不分组 |

### 残差保护参数

| 参数 | 类型 | 默认 | 含义 |
|------|:----:|:----:|------|
| `--residual-n-steps` | int | 1 | **1-bit sign 残差**的 Σ-Δ 步数。（常开残差，每步 ~1.06 bit/element） |
| `--residual-cos-threshold` | float | 0.9999 | 余弦相似度阈值。当 `cos(tile, primary_recon) < threshold` 时触发 1-bit sign。默认 0.9999=几乎总触发 |
| `--adaptive-residual` | flag | — | 启用 **adaptive 1-bit 残差**（gamma=1.0）。tile 重建误差超过 threshold 时额外编码 |
| `--adaptive-residual-threshold` | float | 0.2 | adaptive 残差的 L∞ 触发阈值。越低触发越频繁 |
| `--adaptive-residual-n-steps` | int | 1 | adaptive 残差的 Σ-Δ 步数 |

### 保护 / 旁路参数

| 参数 | 类型 | 默认 | 含义 |
|------|:----:|:----:|------|
| `--decode-protect-steps` | int | 3 | 解码开头几个 token 用 FP16 存储在旁路映射中（保护注意力初始化） |
| `--decode-protect-layers` | str | last_4 | 哪些层应用 decode_protect：`all`/`last_4`/`first_last`/`none` |
| `--decode-gap-threshold` | float | 0.5 | Logits top-2 间隙阈值。低于此值的步骤触发额外 1-bit sign 保护步 |
| `--refresh-interval` | int | 8 | 每 N 步用 FP16 旁路刷新一帧 KV。0=禁用 |
| `--prefill-system-protect` | int | 0 | 预填充中前 N 个 token 用 FP16 旁路保护（零量化误差）。短 prompt 会被全覆盖 |
| `--prefill-tail-protect` | int | 0 | 预填充中后 N 个 token 用 FP16 旁路保护 |

### quality=balanced 预设内部参数

这些在执行命令时会被 `make_config()` 自动注入，无法通过 CLI 显式控制（除非添加新 flag）：

| 参数 | balanced 默认值 | 含义 |
|------|:--------------:|------|
| `use_differential` | True | 启用 diff_residual（预填充阶段给残差额外 2 步 Σ-Δ 编码） |
| `diff_residual_n_steps` | 2 | diff 残差的 Σ-Δ 步数 |
| `diff_residual_gamma` | 0.25 | diff 残差混合系数。**gamma=0.25 只有 adaptive_residual（gamma=1.0）的 1/4 强度** |

---

## 实验全景

### 1B 模型实验 (Llama-3.2-1B)

#### 基线

```bash
python scripts/evaluation/eval_generation_fidelity.py --quality balanced --n-steps 5 --prefill-n-steps 8 --prefill-system-protect 128 --prefill-tail-protect 32 --adaptive-residual --adaptive-residual-threshold 0.05 --adaptive-residual-n-steps 2 --decode-protect-steps 3 --decode-protect-layers last_4 --decode-gap-threshold 0.5 --refresh-interval 8 --residual-cos-threshold 0.9999 --max-tokens 100 --measure-kv --logits-diff --prompts "The capital of France is" --json-output test_dither_0.0.json
```

| 指标 | 值 |
|------|----|
| fork_step | **49** |
| char_match | 0.5285 |
| JS max | 0.693 |

#### 多 Prompt 诊断

```bash
# 5 个 prompt 同时测试
python scripts/evaluation/eval_generation_fidelity.py ... --prompts "The capital of France is" "The meaning of life is" "In the beginning, God created" "The best way to learn programming is" "According to the latest research," ...
```

| Prompt | fork_step | char_match | JS max |
|--------|:---------:|:----------:|:------:|
| "The capital of France is" | 49 | 0.5285 | 0.69 |
| "The meaning of life is" | 无 fork | 1.0 | 0.023 |
| "In the beginning, God created" | 无 fork | 1.0 | 0.014 |
| "The best way to learn programming is" | 无 fork | 1.0 | 0.001 |
| "According to the latest research," | 无 fork | 1.0 | 0.012 |

**结论：4/5 prompt 完美匹配 native greedy。**

#### dither 噪声脱钩实验

```bash
# dither=0.1
python ... --dither-amplitude 0.1 ... --json-output test_dither_0.1.json
# dither=0.2
python ... --dither-amplitude 0.2 ... --json-output test_dither_0.2.json
# dither=0.5
python ... --dither-amplitude 0.5 ... --json-output test_dither_0.5.json
```

| Dither | fork | char | JS@fork |
|:------:|:----:|:----:|:-------:|
| 0.0 | **49** | 0.5285 | 0.0025 |
| 0.1 | 34~41 | 0.35~0.44 | 0.04~0.12 |
| 0.2 | 34 | 0.37~0.42 | 0.07~0.15 |
| 0.5 | 33 | 0.38~0.40 | 0.61~0.66 |

**结论：假设证伪**——白噪声比结构化 Σ-Δ 噪声更有害，单调退化。

#### residual_n_steps 扫描

```bash
# residual_n_steps=2
python ... --residual-n-steps 2 ... --json-output test_residual_n2.json
```

| residual_n_steps | fork | char |
|:----------------:|:----:|:----:|
| 1（基线） | **49** | 0.5285 |
| 2 | **41** | 0.5101 |

**结论：回到固定点 41。**

#### decode-gap-threshold 调优

```bash
python ... --decode-gap-threshold 0.1 ... --json-output test_gap_0.1.json
```

| gap | fork |
|:---:|:----:|
| 0.5 | 49 |
| 0.1 | 49 |

**结论：零效。**

#### FP4 E2M1 残差替换

```bash
python ... --use-fp4-residual ... --json-output test_fp4_n5.json
```

| 配置 | fork | char |
|------|:----:|:----:|
| n=5 + 1-bit sign | **49** | 0.5285 |
| n=5 + FP4 残差 | **41** | 0.4565 |

**结论：回到固定点 41。FP4 代码已删除。**

---

### 3.2B 模型实验 (Llama-3.2-3B)

#### 实验 A：full 配置（FP16 prompt protect + adaptive_residual）

```bash
python scripts/evaluation/eval_generation_fidelity.py --model "D:/Software_Development/Project/models/Llama-3.2-3B" --quality balanced --n-steps 5 --prefill-n-steps 8 --prefill-system-protect 128 --prefill-tail-protect 32 --adaptive-residual --adaptive-residual-threshold 0.05 --adaptive-residual-n-steps 2 --decode-protect-steps 3 --decode-protect-layers last_4 --decode-gap-threshold 0.5 --refresh-interval 8 --residual-cos-threshold 0.9999 --max-tokens 100 --measure-kv --logits-diff --prompts "The capital of France is" --json-output test_3B_baseline.json
```

| 指标 | 值 |
|------|:--:|
| fork_step | **无 fork** |
| char_match | 1.0000 |
| JS max | 0.008 |
| K CosSim | 1.000 |
| V CosSim | 1.000 |
| decode 步数 | ~10（5 primary + 2 diff + 2 adaptive + 1 sign） |
| CR | ~1.5x |

#### 实验 B：n=3 bare（最小位宽）

```bash
python scripts/evaluation/eval_generation_fidelity.py --model "D:/Software_Development/Project/models/Llama-3.2-3B" --quality balanced --n-steps 3 --prefill-n-steps 3 --decode-protect-steps 3 --decode-protect-layers last_4 --decode-gap-threshold 0.5 --refresh-interval 8 --max-tokens 100 --measure-kv --logits-diff --prompts "The capital of France is" --json-output test_3B_n3.json
```

| 指标 | 值 |
|------|:--:|
| fork_step | **9** |
| char_match | 0.137 |
| K CosSim | 0.950 |
| decode 步数 | 3（无任何 residual） |
| CR | ~5.0x |

#### 实验 C：n=4 bare

```bash
python scripts/evaluation/eval_generation_fidelity.py --model "D:/Software_Development/Project/models/Llama-3.2-3B" --quality balanced --n-steps 4 --prefill-n-steps 4 --decode-protect-steps 3 --decode-protect-layers last_4 --decode-gap-threshold 0.5 --refresh-interval 8 --max-tokens 100 --measure-kv --logits-diff --prompts "The capital of France is" --json-output test_3B_n4.json
```

| 指标 | 值 |
|------|:--:|
| fork_step | **11** |
| char_match | 0.165 |
| K CosSim | 0.969 |
| decode 步数 | 4（无任何 residual） |
| CR | ~3.8x |

#### 实验 D：n=5 lean（无 FP16 protect、无 adaptive_residual）

```bash
python scripts/evaluation/eval_generation_fidelity.py --model "D:/Software_Development/Project/models/Llama-3.2-3B" --quality balanced --n-steps 5 --prefill-n-steps 5 --decode-protect-steps 3 --decode-protect-layers last_4 --decode-gap-threshold 0.5 --refresh-interval 8 --residual-cos-threshold 0.9999 --residual-n-steps 1 --max-tokens 100 --measure-kv --logits-diff --prompts "The capital of France is" --json-output test_3B_n5_lean.json
```

| 指标 | 值 |
|------|:--:|
| fork_step | **21** |
| char_match | 0.225 |
| K CosSim | 0.978 |
| decode 步数 | 8（5 primary + 2 diff(gamma=0.25) + 1 sign） |
| CR | ~1.9x |

#### 实验 E：n=5 + adaptive_residual（无 FP16 protect）

```bash
python scripts/evaluation/eval_generation_fidelity.py --model "D:/Software_Development/Project/models/Llama-3.2-3B" --quality balanced --n-steps 5 --prefill-n-steps 5 --adaptive-residual --adaptive-residual-threshold 0.05 --adaptive-residual-n-steps 2 --decode-protect-steps 3 --decode-protect-layers last_4 --decode-gap-threshold 0.5 --refresh-interval 8 --residual-cos-threshold 0.9999 --residual-n-steps 1 --max-tokens 100 --measure-kv --logits-diff --prompts "The capital of France is" --json-output test_3B_no_fp16.json
```

| 指标 | 值 |
|------|:--:|
| fork_step | **21** |
| char_match | 0.269 |
| K CosSim | 0.978 |

**失败原因：adaptive_residual_threshold=0.05 太严格，primary + sign 已经把 tile 误差压到 0.05 以下 → adaptive 从未触发。有效 decode 只有 6 步（5 + 1 sign）。**

#### 实验 F：n=6 bare

```bash
python scripts/evaluation/eval_generation_fidelity.py --model "D:/Software_Development/Project/models/Llama-3.2-3B" --quality balanced --n-steps 6 --prefill-n-steps 6 --decode-protect-steps 3 --decode-protect-layers last_4 --decode-gap-threshold 0.5 --refresh-interval 8 --residual-cos-threshold 0.9999 --residual-n-steps 1 --max-tokens 100 --measure-kv --logits-diff --prompts "The capital of France is" --json-output test_3B_n6.json
```

| 指标 | 值 |
|------|:--:|
| fork_step | **11** |
| K CosSim | 0.984 |
| decode 步数 | 7（6 primary + 1 sign） |
| CR | ~2.15x |

**结论：裸主编码有硬上限——n=4/5/6 均在步 11 附近分叉。增加主编码步数只改善 CosSim，不推迟 fork。**

#### 实验 G：prefill=8 + diff_residual(gamma=0.25) 无 FP16

```bash
python scripts/evaluation/eval_generation_fidelity.py --model "D:/Software_Development/Project/models/Llama-3.2-3B" --quality balanced --n-steps 5 --prefill-n-steps 8 --decode-protect-steps 3 --decode-protect-layers last_4 --decode-gap-threshold 0.5 --refresh-interval 8 --residual-cos-threshold 0.9999 --residual-n-steps 1 --max-tokens 100 --measure-kv --logits-diff --prompts "The capital of France is" --json-output test_3B_no_fp16_prefill8.json
```

| 指标 | 值 |
|------|:--:|
| fork_step | **44** |
| char_match | 0.470 |
| K CosSim | 0.991 |

**diff_residual 的 gamma=0.25 太弱，fork=44 说明 prefill_n_steps=8 + diff_residual 还不够。

#### 实验 H：prefill=8 + adaptive_residual（decode protect + refresh 仍保留）

```bash
python scripts/evaluation/eval_generation_fidelity.py --model "D:/Software_Development/Project/models/Llama-3.2-3B" --quality balanced --n-steps 5 --prefill-n-steps 8 --adaptive-residual --adaptive-residual-threshold 0.05 --adaptive-residual-n-steps 2 --decode-protect-steps 3 --decode-protect-layers last_4 --decode-gap-threshold 0.5 --refresh-interval 8 --residual-cos-threshold 0.9999 --residual-n-steps 1 --max-tokens 100 --measure-kv --logits-diff --prompts "The capital of France is" --json-output test_3B_adaptive_no_fp16.json
```

| 指标 | 值 |
|------|:--:|
| fork_step | **无 fork** |
| char_match | **1.0000** |
| JS max | 0.094 |
| K CosSim | 0.991 |
| decode 步数 | 8（5 primary + 2 adaptive + 1 sign） |
| CR | **~1.88x** |

**prefill 阶段零 FP16，但 decode 开头 3 个 token（last_4 层）和 refresh（每 8 步）仍用 FP16 旁路。**

#### 实验 I：全程零 FP16 ★ 最终验证

```bash
python scripts/evaluation/eval_generation_fidelity.py --model "D:/Software_Development/Project/models/Llama-3.2-3B" --quality balanced --n-steps 5 --prefill-n-steps 8 --adaptive-residual --adaptive-residual-threshold 0.05 --adaptive-residual-n-steps 2 --decode-protect-steps 0 --decode-protect-layers none --decode-gap-threshold 0.5 --refresh-interval 0 --residual-cos-threshold 0.9999 --residual-n-steps 1 --max-tokens 100 --measure-kv --logits-diff --prompts "The capital of France is" --json-output test_3B_zero_fp16.json
```

| 指标 | 值 |
|------|:----:|
| fork_step | **无 fork** |
| char_match | **1.0000** |
| JS max | 0.095 |
| K CosSim | 0.991 |
| V CosSim | 0.992 |
| decode 步数 | 8（5 primary + 2 adaptive + 1 sign） |
| CR | **~1.88x** |

**全程零 FP16 存储——prefill 全部 Σ-Δ 编码 + decode 全部 Σ-Δ 编码，无任何 FP16 旁路。** 结果与实验 H（有少量 FP16）完全一致——FP16 旁路在 100 token 短测试中对结果无影响。

#### 实验 J：prefill=8 + adaptive_residual only（去掉 decode_protect + refresh）

```bash
python scripts/evaluation/eval_generation_fidelity.py --model "D:/Software_Development/Project/models/Llama-3.2-3B" --quality balanced --n-steps 5 --prefill-n-steps 8 --adaptive-residual --adaptive-residual-threshold 0.05 --adaptive-residual-n-steps 2 --residual-cos-threshold 0.9999 --residual-n-steps 1 --decode-protect-steps 0 --decode-protect-layers none --decode-gap-threshold 0.5 --refresh-interval 0 --max-tokens 100 --measure-kv --logits-diff --prompts "The capital of France is" --json-output test_3B_adaptive_only.json
```

| 指标 | 值 |
|------|:----:|
| fork_step | **无 fork** |
| char_match | **1.0000** |
| JS max | 0.095 |
| K CosSim | 0.991 |
| V CosSim | 0.992 |

**确认：adaptive_residual 单独就够**——去掉 decode_protect 和 refresh_interval 后仍然无 fork。FP16 旁路对结果零影响。

#### 实验 K：prefill=7（去掉 adaptive_residual）

```bash
python scripts/evaluation/eval_generation_fidelity.py --model "D:/Software_Development/Project/models/Llama-3.2-3B" --quality balanced --n-steps 5 --prefill-n-steps 7 --residual-cos-threshold 0.9999 --residual-n-steps 1 --decode-protect-steps 0 --decode-protect-layers none --decode-gap-threshold 0.5 --refresh-interval 0 --max-tokens 100 --measure-kv --logits-diff --prompts "The capital of France is" --json-output test_3B_prefill7.json
```

| 指标 | 值 |
|------|:--:|
| fork_step | **11** |
| K CosSim | 0.988 |

**prefill=7 不够**——直接退回到裸 n=6 的水平（fork=11）。

#### 实验 L：prefill=6（去掉 adaptive_residual）

```bash
python scripts/evaluation/eval_generation_fidelity.py --model "D:/Software_Development/Project/models/Llama-3.2-3B" --quality balanced --n-steps 5 --prefill-n-steps 6 --adaptive-residual --adaptive-residual-threshold 0.005 --adaptive-residual-n-steps 2 --residual-cos-threshold 0.9999 --residual-n-steps 1 --decode-protect-steps 0 --decode-protect-layers none --decode-gap-threshold 0.5 --refresh-interval 0 --max-tokens 100 --measure-kv --logits-diff --prompts "The capital of France is" --json-output test_3B_prefill6.json
```

| 指标 | 值 |
|------|:--:|
| fork_step | **11** |
| K CosSim | 0.984 |

**prefill=6 也不够**——adaptive_residual threshold 降到 0.005 仍然不触发（n=5 primary + sign 精度已足够高）。

#### 实验 M：prefill=8 + 无 adaptive（去 adaptive_residual 的对照组）

```bash
python scripts/evaluation/eval_generation_fidelity.py --model "D:/Software_Development/Project/models/Llama-3.2-3B" --quality balanced --n-steps 5 --prefill-n-steps 8 --residual-cos-threshold 0.9999 --residual-n-steps 1 --decode-protect-steps 0 --decode-protect-layers none --decode-gap-threshold 0.5 --refresh-interval 0 --max-tokens 100 --measure-kv --logits-diff --prompts "The capital of France is" --json-output test_3B_final.json
```

| 指标 | 值 |
|------|:--:|
| fork_step | **44** |
| char_match | 0.470 |
| K CosSim | 0.991 |

**去 adaptive → fork=44**。和实验 G（diff_residual）结果一致——prefill=8 + sign 的裸配置只能撑到步 44。adaptive_residual 的 gamma=1.0 是推过步 44 的关键。

#### 实验 N：n=4 + adaptive + prefill=8 ★ 更高 CR 无 fork

```bash
python scripts/evaluation/eval_generation_fidelity.py --model "D:/Software_Development/Project/models/Llama-3.2-3B" --quality balanced --n-steps 4 --prefill-n-steps 8 --adaptive-residual --adaptive-residual-threshold 0.05 --adaptive-residual-n-steps 2 --residual-cos-threshold 0.9999 --residual-n-steps 1 --decode-protect-steps 0 --decode-protect-layers none --decode-gap-threshold 0.5 --refresh-interval 0 --max-tokens 100 --measure-kv --logits-diff --prompts "The capital of France is" --json-output test_3B_n4_adaptive.json
```

| 指标 | 值 |
|------|:----:|
| fork_step | **无 fork** |
| char_match | **1.0000** |
| K CosSim | 0.991 |
| decode 步数 | ~7（4 primary + 2 adaptive + 1 sign） |
| CR | **~2.7x** |

**n=4 裸跑 fork=11，加 adaptive 后无 fork。** adaptive_residual 2 步残差完美补回了 n=4 vs n=5 的 1 步主编码精度损失。

#### 实验 O：n=3 + adaptive + prefill=8（adaptive 极限测试）

```bash
python scripts/evaluation/eval_generation_fidelity.py --model "D:/Software_Development/Project/models/Llama-3.2-3B" --quality balanced --n-steps 3 --prefill-n-steps 8 --adaptive-residual --adaptive-residual-threshold 0.05 --adaptive-residual-n-steps 2 --residual-cos-threshold 0.9999 --residual-n-steps 1 --decode-protect-steps 0 --decode-protect-layers none --decode-gap-threshold 0.5 --refresh-interval 0 --max-tokens 100 --measure-kv --logits-diff --prompts "The capital of France is" --json-output test_3B_n3_adaptive.json
```

| 指标 | 值 |
|------|:--:|
| fork_step | **42** |
| char_match | 0.541 |
| K CosSim | 0.991 |

**n=3 + adaptive fork=42。** adaptive 把 fork 从 9 推到 42（救了 33 步），但 n=3 底子太弱，2 步残差不足以补 2 步主编码的损失。adaptive 的补偿上限在 n=4。

#### 实验 P：n=4 + adaptive + prefill=6

```bash
python scripts/evaluation/eval_generation_fidelity.py --model "D:/Software_Development/Project/models/Llama-3.2-3B" --quality balanced --n-steps 4 --prefill-n-steps 6 --adaptive-residual --adaptive-residual-threshold 0.05 --adaptive-residual-n-steps 2 --residual-cos-threshold 0.9999 --residual-n-steps 1 --decode-protect-steps 0 --decode-protect-layers none --decode-gap-threshold 0.5 --refresh-interval 0 --max-tokens 100 --measure-kv --logits-diff --prompts "The capital of France is" --json-output test_3B_n4_prefill6.json
```

| 指标 | 值 |
|------|:--:|
| fork_step | **11** |
| K CosSim | 0.984 |

**prefill=6 不够。** 回到裸 n=4 的水平。

#### 实验 Q：n=4 + adaptive + prefill=7

```bash
python scripts/evaluation/eval_generation_fidelity.py --model "D:/Software_Development/Project/models/Llama-3.2-3B" --quality balanced --n-steps 4 --prefill-n-steps 7 --adaptive-residual --adaptive-residual-threshold 0.05 --adaptive-residual-n-steps 2 --residual-cos-threshold 0.9999 --residual-n-steps 1 --decode-protect-steps 0 --decode-protect-layers none --decode-gap-threshold 0.5 --refresh-interval 0 --max-tokens 100 --measure-kv --logits-diff --prompts "The capital of France is" --json-output test_3B_n4_prefill7.json
```

| 指标 | 值 |
|------|:--:|
| fork_step | **11** |
| K CosSim | 0.988 |

**prefill=7 也不够。** prepill=8 是硬边界——无论 n=4 还是 n=5，prefill 低于 8 就 fork。

---

### 3.2B 模型 — Phase 2c：匹配追踪（MP）实验

#### 实验背景

Σ-Δ 编码器的 momentum 机制导致相邻步量化误差强负相关，产生结构化噪声。假设是去掉 momentum 但保留 1-bit 迭代分解结构（匹配追踪，Matching Pursuit），可以消除结构化噪声而不显著增加总误差幅度。代码中新增 `--encode-mode matching_pursuit` 开关。

#### 实验 R：Σ-Δ n=5 多 Prompt 基线 ★ 重要

```bash
python scripts/evaluation/eval_generation_fidelity.py --model "D:/Software_Development/Project/models/Llama-3.2-3B" --n-steps 5 --prefill-n-steps 8 --adaptive-residual --adaptive-residual-threshold 0.05 --adaptive-residual-n-steps 2 --residual-cos-threshold 0.9999 --residual-n-steps 1 --decode-protect-steps 0 --decode-protect-layers none --refresh-interval 0 --max-tokens 100 --measure-kv --logits-diff --prompts "The capital of France is" "The meaning of life is" "In the beginning, God created" "The best way to learn programming is" "According to the latest research," --json-output test_3B_multiprompt.json
```

| Prompt | K CosSim | 结果 |
|--------|:--------:|:----:|
| The capital of France is | 0.9913 | 无 fork ✅ |
| The meaning of life is | 0.9915 | fork@1 ❌ |
| In the beginning, God created | 0.9842 | 无 fork ✅ |
| The best way to learn programming is | 0.9920 | fork@6 ❌ |
| According to the latest research, | 0.9838 | fork@4 ❌ |
| **平均** | **0.9885** | **3/5 fork** |

**结论：** 之前单 prompt 实验（"The capital of France is"）的"无 fork"结论是误导性的——这个 prompt 太容易。多 prompt 异构下，即使 K CosSim=0.9915 的 prompt 也在 step 1 就 fork。

#### 实验 S：MP n=5

```bash
# 同实验 R，加 --encode-mode matching_pursuit
python scripts/evaluation/eval_generation_fidelity.py ... --encode-mode matching_pursuit ... --json-output test_mp_n5.json
```

| 指标 | Σ-Δ n=5 | MP n=5 |
|------|:-------:|:------:|
| K CosSim 平均 | **0.9885** | 0.9823 |
| V CosSim 平均 | **0.9899** | 0.9856 |
| Fork | 3/5 | **4/5** |

**结论：MP n=5 比 Σ-Δ n=5 差。** CosSim 下降 ~0.006，fork 多一个。

#### 实验 T：MP n=6

```bash
python scripts/evaluation/eval_generation_fidelity.py --model "D:/Software_Development/Project/models/Llama-3.2-3B" --n-steps 6 --prefill-n-steps 8 --adaptive-residual --adaptive-residual-threshold 0.05 --adaptive-residual-n-steps 2 --residual-cos-threshold 0.9999 --residual-n-steps 1 --decode-protect-steps 0 --decode-protect-layers none --refresh-interval 0 --max-tokens 100 --measure-kv --logits-diff --encode-mode matching_pursuit --prompts "The capital of France is" "The meaning of life is" "In the beginning, God created" "The best way to learn programming is" "According to the latest research," --json-output test_mp_n6.json
```

| 指标 | 值 |
|------|----|
| K CosSim 平均 | 0.988542 |
| Fork | 3/5（fork@1,4,6） |

**与 Σ-Δ n=5、Σ-Δ n=6、MP n=5 结果完全一致。** n_steps 和 encode_mode 参数在 decode 路径上均无效果——这是一个系统性 bug 或编码饱和。

#### 实验 U：Σ-Δ n=6 多 Prompt（对比）

```bash
python scripts/evaluation/eval_generation_fidelity.py --model "D:/Software_Development/Project/models/Llama-3.2-3B" --n-steps 6 --prefill-n-steps 8 --adaptive-residual --adaptive-residual-threshold 0.05 --adaptive-residual-n-steps 2 --residual-cos-threshold 0.9999 --residual-n-steps 1 --decode-protect-steps 0 --decode-protect-layers none --refresh-interval 0 --max-tokens 100 --measure-kv --logits-diff --prompts "The capital of France is" "The meaning of life is" "In the beginning, God created" "The best way to learn programming is" "According to the latest research," --json-output test_3B_n6_full.json
```

| 指标 | 值 |
|------|----|
| K CosSim 平均 | 0.988542 |
| Fork | 3/5（fork@1,4,6） |

**与 n=5 完全一致。n_steps 从 5 提高到 6 对 decode 路径零效。**

### 3.2B 模型 — Phase 2d：Prefill 尾部保护实验

#### 实验 V：prefill-tail-protect=32 + adaptive

```bash
python scripts/evaluation/eval_generation_fidelity.py --model "D:/Software_Development/Project/models/Llama-3.2-3B" --n-steps 5 --prefill-n-steps 8 --prefill-tail-protect 32 --adaptive-residual --adaptive-residual-threshold 0.05 --adaptive-residual-n-steps 2 --residual-cos-threshold 0.9999 --residual-n-steps 1 --decode-protect-steps 0 --decode-protect-layers none --refresh-interval 0 --max-tokens 100 --measure-kv --logits-diff --prompts "The capital of France is" "The meaning of life is" "In the beginning, God created" "The best way to learn programming is" "According to the latest research," --json-output test_3B_n5_tail32.json
```

| Prompt | K CosSim | 结果 |
|--------|:--------:|:----:|
| The capital of France is | 1.000 | 无 fork ✅ |
| The meaning of life is | 1.000 | 无 fork ✅ **(之前 fork@1！)** |
| In the beginning, God created | 1.000 | 无 fork ✅ |
| The best way to learn programming is | 1.000 | fork@68 |
| According to the latest research, | 1.000 | fork@39 |
| **平均** | **1.000** | **2/5 fork（均在后期）** |

**结论：prefill 尾巴 FP16 保护解决了早期 fork。** fork@1（"The meaning of life"）、fork@4（"According to"的前期）全部消除。剩余两个 fork 在 39/68 步，属 decode 后期累积误差。

#### 实验 W：prefill-tail=32 + refresh=64

```bash
# 同实验 V，加 --refresh-interval 64
```

| 指标 | 值 |
|------|----|
| Fork | 2/5（fork@39,68） |

**refresh=64 无效。** 与无 refresh 结果完全一致。

#### 实验 X：prefill-tail=32 + refresh=32

```bash
# 同实验 V，加 --refresh-interval 32
```

| 指标 | 值 |
|------|----|
| Fork | 3/5（fork@39,42,68） |

**refresh=32 反而恶化。** "The meaning of life" 从无 fork 退化为 fork@42。refresh 可能干扰注意力分布。

#### 实验 Y：decode-protect-steps=1（对比）

```bash
python scripts/evaluation/eval_generation_fidelity.py --model "D:/Software_Development/Project/models/Llama-3.2-3B" --n-steps 5 --prefill-n-steps 8 --decode-protect-steps 1 --decode-protect-layers all --adaptive-residual --adaptive-residual-threshold 0.05 --adaptive-residual-n-steps 2 --residual-cos-threshold 0.9999 --residual-n-steps 1 --refresh-interval 0 --max-tokens 100 --measure-kv --logits-diff --prompts "The capital of France is" "The meaning of life is" "In the beginning, God created" "The best way to learn programming is" "According to the latest research," --json-output test_3B_adaptive_only.json
```

| 指标 | 值 |
|------|----|
| Fork | 3/5（fork@1,4,6） |

**decode 保护无效。** fork 是 prefill 侧 KV 误差传导过来的，保护 decode 侧没用。

---

### 匹配追踪实验结论

| 假设 | 结果 |
|------|------|
| 去掉 momentum 消除结构化噪声 → 无 fork | **❌ 证伪** |
| MP 能保持同精度 | **❌** MP K CosSim=0.982 < SD K CosSim=0.989 |
| fork 原因是噪声结构性 | **❌** fork 纯属总精度不足 |
| n_steps 提升（5→6）能改善精度 | **❌** decode 路径上 n_steps 零效 |

---

## 3.2B 全部实验结果汇总

| 实验 | fork | K CosSim | 关键区别 |
|------|:----:|:--------:|------|
| R | 3/5 | 0.989 | **Σ-Δ n=5 + prefill=8 + adaptive + sign — 多 Prompt 基线** |
| S | 4/5 | 0.982 | MP n=5（差于 Σ-Δ） |
| T | 3/5 | 0.989 | MP n=6（同 Σ-Δ n=5，n_steps 零效） |
| U | 3/5 | 0.989 | Σ-Δ n=6（同 n=5，n_steps 零效） |
| V | **2/5** | **1.000** | **prefill-tail=32，早期 fork 全消！** |
| W | 2/5 | 1.000 | tail=32 + refresh=64（无效） |
| X | 3/5 | 1.000 | tail=32 + refresh=32（恶化） |
| Y | 3/5 | 0.989 | decode-protect=1（无效，fork 是 prefill 侧的） |
| J | 无★ | 0.991 | n=5 + sign + prefill=8（单 prompt "The capital of France is"） |
| N | 无★ | 0.991 | n=4 + adaptive + prefill=8（单 prompt） |
| G | 44 | 0.991 | n=5 + sign + diff(gamma=0.25)（单 prompt） |
| O | 42 | 0.991 | n=3 + adaptive + sign（单 prompt） |
| D | 21 | 0.978 | n=5 + sign + diff + prefill=5（单 prompt） |
| Q | 11 | 0.988 | n=4 + adaptive + prefill=7（单 prompt） |
| F | 11 | 0.984 | n=6 bare（单 prompt） |
| C | 11 | 0.969 | n=4 bare（单 prompt） |
| B | 9 | 0.950 | n=3 bare（单 prompt） |

> ★ 单 prompt 实验在 "The capital of France is" 上无 fork，但不代表多 prompt 可靠性。

---

## 核心结论

### 1. 单 Prompt 实验结论已被多 Prompt 验证推翻

**之前的"无 fork"结论仅对 easy prompt ("The capital of France is") 成立。** 多 prompt（5 组异构）下，Σ-Δ n=5 + sign + prefill=8 + adaptive 仍然 3/5 fork，fork 分布在 step 1~6。

即使 K CosSim 高达 0.991~0.992 的 prompt 也在早期就 fork（如 "The meaning of life is" 的 fork@1），说明平均 CosSim 无法预测 fork。问题出在特定 attention head 特定子空间上的精度不够。

### 2. 匹配追踪（MP）假设全面证伪

| 假设 | 结果 |
|------|:--:|
| 去掉 momentum → 消除结构化噪声 → 无 fork | ❌ |
| MP 保持同精度 | ❌ (0.982 < 0.989) |
| fork 原因是噪声结构性 | ❌ 纯精度问题 |

**MP 方案废弃。** 代码中的 `--encode-mode matching_pursuit` 保留作学术对照，但不应用于生产。

### 3. n_steps 在 decode 路径上零效（疑似 Bug）

Σ-Δ n=5、Σ-Δ n=6、MP n=5、MP n=6 在全部多 prompt 测试中产生**完全一致**的结果（K CosSim、fork step、logit diff 值全部相同）。这意味着 decode 路径上的 `n_steps` 参数被忽略或编码路径未实际执行。**需后续排查。**

### 4. prefill-tail-protect=32 解决了早期 fork（关键发现）

| Prompt | 无 protect | tail=32 |
|--------|:---:|:---:|
| The capital of France is | ✅ | ✅ |
| The meaning of life is | fork@1 ❌ | ✅ |
| In the beginning, God created | ✅ | ✅ |
| The best way to learn programming is | fork@6 ❌ | fork@68 |
| According to the latest research, | fork@4 ❌ | fork@39 |

早期 fork（step 1~6）全部消除。K CosSim 和 V CosSim 均达到 **1.000**（所有 prompt）。

### 5. 后期 fork 问题仍未解决

2/5 prompt 在 step 39~68 仍 fork。这些是 decode 阶段量化误差累积导致的。`refresh-interval` 无效甚至有害。

### 6. 当前最可靠配置

| 参数 | 推荐值 | 说明 |
|------|:------:|------|
| n-steps | **5** | 主编码步数（decode 侧 n_steps 实际上零效） |
| prefill-n-steps | **8** | 硬边界 |
| prefill-tail-protect | **32** | **关键项**——解决早期 fork |
| adaptive-residual | 开 | 阈值 0.05，2 步 |
| residual-cos-threshold | 0.9999 | 几乎总是触发 sign 残差 |
| residual-n-steps | 1 | 1-bit sign 残差 |
| refresh-interval | 0 | 无效，不推荐 |
| decode-protect-steps | 0 | 无效（fork 在 prefill 侧） |

**CR 估算：** ~1.9x（5 primary + 2 adaptive + 1 sign = 8 步 decode，prefill tail 32 FP16 token 对长文本影响可忽略）

### 7. CR 计算公式

```
每步 = 1 base bit + 0.0625 alpha bit = 1.0625 bit/element
CR = 16 / (步数 × 1.0625)
```

### 8. prefill_n_steps=8 是硬边界

prefill 低于 8 在任何配置下都导致 fork=11（3.2B）。与 n_steps、encode_mode 无关。

### 9. 1B 模型的瓶颈：d_head=64

1B 的 `1/√64 ≈ 12.5%` 量化误差修正幅度在 5 步内无法消除足够噪声。3.2B 的 `1/√128 ≈ 8.8%` 跨过了这个门槛。

1B 上的 fork=49 是物理硬限——`1/√64 ≈ 12.5%` 的修正幅度无法在 5 步内消除足够多的量化噪声。3.2B 的 `1/√128 ≈ 8.8%` 直接把这个门槛移除了。

### 6. 四个无效分支已清理

`baseline_mask`、`r1`、`r1_mask` 三个路由 + `dither` `FP4` 两个实验特性——全已从代码中删除。

---

## 评估脚本修改

```python
# scripts/evaluation/eval_generation_fidelity.py

# 1) 路由简化为 native + baseline only
def build_routes() -> List[MaskRoute]:
    return [
        MaskRoute("native",   "native",   adaptive_masking=False, use_mask_gating=False),
        MaskRoute("baseline", "baseline", adaptive_masking=False, use_mask_gating=False),
    ]

# 2) make_config 移除 quality 参数，固定使用原 "balanced" 参数
#    默认 n_steps=5（--n-steps 可覆盖）

# 3) 新增 CLI 参数
#    --encode-mode {sigma_delta,matching_pursuit}  编码模式开关
#    --encode_mode: str = "sigma_delta"  in DSKVCacheConfig
```

---

## 技术遗产

### 保留
- `residual_n_steps`（config + ds_kv_cache + CLI）——合法参数
- `decode_gap_threshold`（已有）
- `encode_mode`（config + ds_kv_cache + unified_encoder + CLI）——学术对照
- eval 脚本改为只跑 baseline + native
- `build_routes()` 只保留 native + baseline

### 已删除
- `--quality` CLI 参数及 `make_config` 的 quality 分支
- `dither_amplitude`（config + ds_kv_cache + model_wrapper + CLI）
- `use_fp4_residual`（config + ds_kv_cache + CLI）
- `rina/utils/fp4.py`（整个文件）
- `baseline_mask` / `r1` / `r1_mask` 路由
