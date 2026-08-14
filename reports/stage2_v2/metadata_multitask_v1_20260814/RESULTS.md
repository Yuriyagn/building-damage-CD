# Metadata-aware multi-task v1：正式验证报告

> 结论只来自 R4 event-group train/validation。未读取、评估或用于选择 test。

## Verdict

**M1 未通过预注册首轮筛选，分支在此停止，不运行 M2。**

加入 `0.1 × disaster classification loss` 后，三个种子的平均 BO damage macro F1 提高 `+0.03939`，平均 event macro BO F1 提高 `+0.02239`，且两个指标均有 2/3 种子为正。可是灾种分类没有泛化到未见事件：M1 的灾种 present-class macro F1 三种子均值只有 `0.23918`，低于 majority baseline `0.28`，平均差为 `-0.04082`，而预注册要求至少 `+0.10`（即 macro F1 至少 `0.38`）。

因此当前证据最多支持：**固定权重辅助损失可能带来一般正则化或优化差异，并在部分事件上改善损伤分割；不能声称灾害语义知识提升了跨事件泛化。**

## 1. 协议与可复现状态

- 代码 commit：`e0925941abd593784181aaf1a3a92aff176b5c5c`
- source tree SHA-256：`7a90fa46af194555569b4fff573dd498f071ca864bea40ca808fb9be8c92c638`
- tracked diff：空；SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- data audit SHA-256：`1832b1006d79dec59055335907b892f64218003db705a72e76c07bc508701b9c`
- 数据：1634 train / 290 validation；事件完全分离；audit hard error 0；未读取 test
- 输入：四通道 `[pre-RGB, post-SAR]`；prior 不进入模型，只保留次要门控评价
- 模型：ImageNet ResNet34 U-Net，共享 encoder 最深层 GAP + Dropout(0.2) + 7 类线性辅助头
- M0：同一结构，`lambda_disaster=0.0`；M1：`lambda_disaster=0.1`
- 种子：`42 / 3407 / 2026`；每种子 M0→M1 成对运行
- 训练：batch 8，AdamW，LR `1e-4`，weight decay `1e-4`，cosine，AMP，最多 100 epoch，最少 40，patience 30
- checkpoint：只按 validation BO 三等级 macro F1 的 raw 值保存；灾种分数不参与选择
- 初始化配对：三个种子的 M0/M1 初始模型 SHA-256 均逐字节一致
- bootstrap：先重采样种子、再在种子内成对重采样 4 个事件；10,000 次，seed `20260814`

正式运行均成功完成：M0 epochs 分别为 `56 / 82 / 73`，M1 分别为 `100 / 100 / 100`；checkpoint epochs 分别为 M0 `39 / 14 / 42`、M1 `93 / 49 / 76`。

## 2. 三种子主结果

均值后括号为样本标准差。

| condition | BO 3-grade macro F1 | BO damage macro F1 | Damaged F1 | Destroyed F1 | Binary damage F1 | Event macro BO F1 | Event macro damage F1 | Worst-event BO F1 | Disaster macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | 0.41676 (0.01187) | 0.15870 (0.02307) | 0.04276 (0.03052) | 0.27464 (0.05406) | 0.31758 (0.01115) | 0.33783 (0.01445) | 0.11367 (0.00798) | 0.24155 (0.07061) | 不解释（辅助头未训练） |
| M1 | 0.44661 (0.01607) | 0.19809 (0.02423) | 0.04892 (0.02907) | 0.34726 (0.03172) | 0.32610 (0.04320) | 0.36023 (0.03239) | 0.12866 (0.02183) | 0.26418 (0.09103) | 0.23918 (0.04311) |

主任务改善主要来自 Destroyed F1（均值约 `+0.07263`），Damaged F1 只增加约 `+0.00616`，绝对值仍极低。Binary damage F1 的标准差从 `0.01115` 增至 `0.04320`，说明收益伴随更强的种子敏感性。

## 3. 成对差值与门槛

| seed | Δ BO damage macro | Δ event macro BO | Δ event macro damage | Δ worst-event BO | M1 disaster macro | 相对 majority 0.28 |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | +0.05877 | +0.04527 | +0.01179 | +0.14283 | 0.28843 | +0.00843 |
| 3407 | +0.06091 | +0.03319 | +0.03421 | -0.00397 | 0.20829 | -0.07171 |
| 2026 | -0.00150 | -0.01128 | -0.00101 | -0.07094 | 0.22081 | -0.05919 |
| mean | **+0.03939** | **+0.02239** | +0.01500 | **+0.02264** | **0.23918** | **-0.04082** |

预注册检查：

| 检查 | 结果 |
|---|---|
| 三种子 mean Δ BO damage macro ≥ +0.01 | PASS |
| BO damage macro 至少 2/3 种子为正 | PASS |
| 三种子 mean Δ event macro BO ≥ +0.01 | PASS |
| Event macro BO 至少 2/3 种子为正 | PASS |
| Mean Δ worst-event BO ≥ -0.01 | PASS |
| M1 disaster macro 比 majority 至少高 +0.10 | **FAIL** |

层级 bootstrap 的描述性 95% 区间为：

- Δ BO damage macro F1：`[-0.00305, +0.09414]`
- Δ event macro BO F1：`[-0.01350, +0.06140]`

两条区间均跨过 0。validation 只有 4 个事件，这些区间只描述不确定性，不构成显著性证明。

## 4. 逐事件结果揭示的收益来源

下表是三种子 M1−M0 的平均差值。

| validation event | disaster | Δ event BO F1 | Δ event damage macro F1 | 解释 |
|---|---|---:|---:|---|
| bata_explosion | explosion | +0.03716 | -0.00540 | BO 提升来自 intact；损伤两类平均反而略降，且 seed2026 worst-event 大跌 |
| haiti_earthquake | earthquake | +0.00186 | -0.00281 | 基本无改善；damage F1 仍接近 0 |
| marshall_wildfire | fire | +0.04301 | +0.06189 | 最主要的真实 damage 收益来源，但 seed2026 为负 |
| morocco_earthquake | earthquake | +0.00755 | +0.00630 | 小幅改善，但 damage F1 仍接近 0.01 |

最严重的失败不是平均分，而是跨事件结构：

- M1 在 Haiti 的三种子 event damage F1 约为 `0.01535 / 0.00978 / 0.00892`。
- M1 在 Morocco 的三种子 event damage F1 约为 `0.00813 / 0.01091 / 0.01199`。
- seed2026 的 Bata BO F1 从 `0.23009` 降到 `0.15915`，使 worst-event 下降 `-0.07094`。
- 因此模型并没有解决未见地震事件的受损建筑识别，平均收益高度依赖 Marshall wildfire 和 Destroyed 类。

## 5. 灾种分类为何不能支持“知识提升泛化”

在三个 M1 最佳 damage checkpoint 上：

| seed | checkpoint epoch | train disaster macro F1 | validation disaster macro F1 |
|---:|---:|---:|---:|
| 42 | 93 | 0.99973 | 0.28843 |
| 3407 | 49 | 0.99861 | 0.20829 |
| 2026 | 76 | 0.99973 | 0.22081 |

训练分类近乎完美、未见事件验证却低于 majority，符合当前数据设计中的已知混杂：训练集每个灾种只对应一个训练事件。辅助头很容易学到事件/场景身份、成像风格或损伤分布捷径，而不是可迁移的灾害机制。完整 validation 图像评价没有消除这一差距，说明问题并非仅由 damage-aware crop 的局部视野造成。

M0 的灾种输出来自未训练随机辅助头，只用于检查结构一致性，不能与 M1 作语义比较。

## 6. 梯度诊断

M1 在共享 encoder 最后一层的任务梯度余弦：

| seed | mean cosine | negative epoch fraction | min | max |
|---:|---:|---:|---:|---:|
| 42 | -0.00499 | 0.53 | -0.34552 | +0.33772 |
| 3407 | +0.01685 | 0.36 | -0.21208 | +0.44083 |
| 2026 | +0.01449 | 0.41 | -0.19656 | +0.26596 |

平均余弦接近 0，且 36%–53% epoch 为负；辅助任务既没有稳定对齐主任务，也存在阶段性强冲突。首轮按协议没有使用 GradNorm/PCGrad，因此不能把当前结果外推为“所有多任务权重策略都无效”。

## 7. Checkpoint/early-stopping 实现细节

训练代码用 3-epoch 平滑 BO 值更新 patience，但用 raw BO 值保存 checkpoint。两者可能在不同 epoch 改善。例如 M0/seed3407 在 epoch 82 才因 patience 停止，但正式 checkpoint 是 epoch 14；M1/seed2026 多次因平滑值刷新而跑满 100 epoch，最终 raw checkpoint 是 epoch 76。

这不违反本次冻结协议，且 M0/M1 完全一致，但会增加 checkpoint 路径依赖和运行时，并可能放大 raw validation 峰值选择。下一轮实验应在启动前统一为一种明确策略（例如平滑值只早停、raw 值保存但报告峰值偏差，或两者都用同一平滑定义），不能在已完成结果上追溯修改。

## 8. 决策与下一步

本分支的正式动作是 `stop_and_report_M1_failure`：

1. **不运行 M2**。M2 只在 M1 通过全部首轮门槛时启动；当前语义门槛失败。
2. **不宣称灾害知识提升泛化**。现有 damage 收益只能标注为辅助正则化/优化相关的筛选级信号。
3. 下一步若继续 metadata 方向，先修复可识别性问题：每个灾种至少需要多个训练事件，并设计灾种内 leave-one-event-out；否则 disaster 与 event identity 无法解耦。
4. 在数据条件改善前，更值得优先验证视觉任务辅助监督（building existence / changed-vs-unchanged），因为它们可在样本级直接核验，不依赖当前混杂的事件级灾种标签。
5. 保留 M1 的正向信号作为后续对照，但必须与固定乱序标签或等参数一般辅助任务比较后，才能区分语义与正则化。

机器可读结果见 `summary.json`、`runs.csv` 与 `m0_vs_m1_paired_deltas.csv`。
