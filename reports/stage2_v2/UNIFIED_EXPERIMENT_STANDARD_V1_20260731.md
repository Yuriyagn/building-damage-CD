# Stage-2 Unified Experiment Standard v1

冻结日期：2026-07-31  
协议 ID：`stage2_unified_v1`

## 1. 目的

本协议用于比较本项目模型和外部作者方法。比较时只允许替换模型方法本身，
数据切分、输入信息、训练目标、优化预算、模型选择指标和负对照保持一致。

作者原始 UABCD 复现已证明代码可以完整运行，但它采用四类无权重 CE 和四类
mIoU 选权重。最佳 epoch 28 的 validation mIoU 为 `0.3505`，各类 IoU 为
`0.9170 / 0.4828 / 0.0015 / 0.0009`（background / intact / damaged /
destroyed）。因此该结果主要由背景和 intact 支撑，不能直接回答本项目关心的
损伤分级问题，也不能与本项目 BO F1 横向比较。

Unified v1 将 UABCD 骨架迁入本项目的训练和评估管线，消除上述协议差异。

## 2. 冻结数据与开发边界

- 唯一数据协议：
  `manifests/stage2_v2_clean_human_reviewed_20260624/predicted_prior/`
- 样本数：train `1207`，val `357`，test `388`
- 三个 split 的 SHA-256 固化在
  `configs/stage2_unified_v1/protocol.yaml`
- split 之间必须 canonical-event-disjoint，完整审计必须为 0 hard error
- 配方开发和模型选择只用 train/val
- test 在本轮迁移中封存，不运行指标、不调阈值

历史 strict test 已经在更早阶段暴露过，因此它不能重新充当全新的无偏最终
测试。本轮结论只写作 audited validation result；如需新的正式泛化结论，应
另建未触碰事件 holdout 或预注册 cross-event CV。

## 3. 统一输入与迁移方式

每个样本使用相同六通道：

```text
[pre_R, pre_G, pre_B, SAR, predicted_building_prior, SAR * prior]
```

- E2/UNet 直接接收六通道。
- UABCD 保留作者的双流结构：
  - stream A：`[pre_R, pre_G, pre_B]`
  - stream B：`[SAR, predicted_prior, SAR * prior]`
- UABCD 保留作者输入归一化 `[0,1] -> [-1,1]` 和 PVTv2-B2 ImageNet 权重。
- 外部源码不复制进主仓库；适配器只动态加载隔离目录中的固定 commit。
- 外部仓库快照无明确顶层 LICENSE，因此当前使用范围限于本地研究复现。

## 4. 统一训练标准

| 项目 | 冻结值 |
| --- | --- |
| crop | 512 |
| seeds | 42, 3407, 2026 |
| optimizer | AdamW |
| LR / weight decay | 1e-4 / 1e-4 |
| scheduler | cosine |
| effective batch | 8 |
| max epochs | 40 |
| early stop | min 12, patience 10, window 3 |
| loss | building-only weighted CE + binary auxiliary 0.3 |
| augmentation | HFlip + Rotate90；damage-aware crop 0.4/0.4/0.2 |
| checkpoint | 仅保留 BO grade macro F1 最佳权重 |

UABCD 以 micro-batch 4、梯度累积 2 达到 effective batch 8。paired 和
shuffled 配置除实验名及 SAR 配对方式外必须完全一致。

## 5. 统一指标

唯一主指标：

```text
validation building_only_macro_f1_3class
```

必须同时报告：

- BO intact / damaged / destroyed F1
- BO damage macro F1
- BO binary damage F1
- predicted-gate four-class macro F1
- predicted-gate damage macro F1
- per-disaster 与 per-event 指标

`cc_surrogate` 仅为历史诊断，禁止用于模型选择。不得以 train loss、作者原始
四类 mIoU 或 test 指标替代主指标。

## 6. 必须的负对照

- `S2U1_UABCD_paired`：真实配对 SAR
- `S2U1_UABCD_shuffled`：固定 seed `20260731` 的 within-event derangement

shuffled 置换必须无 self-pair、可复现并随 run 保存。未显著优于 shuffled 时，
不得声称模型利用了 SAR 的样本级对应关系。

## 7. 执行门槛

严格顺序：

1. 协议校验与完整数据审计
2. 单 batch smoke
3. 固定两 batch overfit，loss 相对下降至少 30%
4. paired / shuffled seed 42 validation screening
5. 仅当以下两项同时通过，才扩展到三 seeds：
   - paired BO macro F1 `>= 0.3586137203`（E2 argmax reference）
   - paired - shuffled BO macro F1 `>= 0.01`
6. gate 不通过则记录结果并停止，不打开额外调参分支

三 seed 完成后报告均值、标准差、逐 seed paired-shuffled 差值；仍不运行 test。

## 8. 可执行入口

```bash
python scripts/validate_stage2_unified_protocol.py
bash scripts/launch_stage2_unified_v1.sh paired 42 0
bash scripts/launch_stage2_unified_v1.sh shuffled 42 1
```

训练完成后分别在 val 上运行 `src/stage2/test_stage2_v2.py`，输出目录统一命名
为 `val_best_grade/`，再执行：

```bash
python scripts/evaluate_stage2_unified_v1_gate.py \
  --paired-run <paired_run_dir> \
  --shuffled-run <shuffled_run_dir>
```

若 gate 结果为 `pass`，launcher 才允许 seeds `3407` 与 `2026`。

## 9. 产物保留

每个完成 run 只保留：

- `best_bo_grade_macro_f1.pth`
- resolved config、run info、审计、置换记录
- scalar history、validation metrics、per-event/per-disaster CSV
- 唯一日志

不长期保留 `last.pth`、其他指标权重、逐 epoch 权重、predictions 或 previews。
