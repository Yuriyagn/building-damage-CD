# Stage-2 v2 pre-change optical / SAR-enhancement exploration plan

日期：2026-06-26

目标：检查 pre-event optical RGB 是否能给 post-event SAR 提供更稳定的灾损判别上下文，并检查一个轻量 SAR texture/gradient 通道是否提升 SAR damage evidence。该组实验使用 clean human-reviewed strict manifest 与 oracle building support，便于和已经完成的 O1/O2/O3 oracle-prior 探索直接比较。

## 1. 对照关系

已完成基线：

| ID | 输入 | 目的 |
| --- | --- | --- |
| O1 | oracle prior only | prior / label bias baseline |
| O2 | SAR + oracle prior | paired SAR baseline |
| O3 | shuffled SAR + oracle prior | SAR correspondence negative control |

新增实验：

| ID | 输入 | channels | 目的 |
| --- | --- | ---: | --- |
| C1 | pre optical RGB + oracle prior, SAR zeroed | 6 | pre/context + prior baseline |
| C2 | pre optical RGB + paired SAR + oracle prior | 6 | 检查 pre optical 是否帮助模型利用 SAR |
| C3 | pre optical RGB + shuffled SAR + oracle prior | 6 | C2 的 correspondence control |
| C4 | pre optical RGB + paired SAR + SAR gradient + oracle prior | 8 | 检查轻量 SAR texture 是否增强判别 |
| C5 | pre optical RGB + shuffled SAR + SAR gradient + oracle prior | 8 | C4 的 correspondence control |

关键比较：

```text
C2 - C1: 加入 paired SAR 是否超过 pre/context+prior baseline
C2 - C3: paired SAR 是否超过 shuffled SAR control
C4 - C2: SAR gradient/texture 是否带来额外收益
C4 - C5: SAR gradient/texture 下 paired SAR 是否仍超过 shuffled control
C2/C4 vs O2: pre optical / texture 是否优于原 SAR+oracle prior
```

## 2. 输入定义

新增 dataset input modes：

```text
pre_prior:
  [pre_R, pre_G, pre_B, 0, prior, 0]

pre_sar_prior:
  [pre_R, pre_G, pre_B, SAR, prior, SAR * prior]

pre_sar_texture_prior:
  [pre_R, pre_G, pre_B, SAR, SAR_grad, prior, SAR * prior, SAR_grad * prior]
```

`SAR_grad` 是每张 crop 内由 post-SAR 灰度计算的简单梯度幅值，并按 99 分位裁剪归一化。它不是新数据源，只是轻量纹理提示。

## 3. 训练原则

- 只跑 seed 42，探索优先。
- 不用 test 调参；先看 val，再一次性评估 test。
- 仍报告 BO three-grade macro F1、damage macro、binary damage、per-event 与 event-macro。
- 仍必须看 shuffled-SAR control。若 C2/C4 只超过 O2 但不超过 C3/C5，不能声称 SAR correspondence 有效。

## 4. 预期判读

若 C2 > C1 且 C2 > C3，说明 pre optical 确实帮助模型把 paired SAR 转化为灾损证据。

若 C2 > C1 但 C2 ≈ C3，说明提升主要来自 pre optical/context 或事件 bias，不是 SAR correspondence。

若 C4 > C2 且 C4 > C5，说明简单 SAR texture 对损伤分级有帮助；可进一步考虑更系统的 SAR texture/change features。

若 C1/C2/C3 均接近，说明当前瓶颈更可能来自标签/事件漂移或单时相 SAR 与 pre optical 的跨模态对齐不足。
