# UABCD 统一标准迁移结果 — 2026-07-31

协议：`stage2_unified_v1`
结论：**seed 42 硬门槛失败，UABCD 迁移分支停止，不扩展 seeds 3407/2026。**

本结论只针对 UABCD 在本项目数据、输入和训练目标下的这次迁移，不代表对作者
原方法或论文结论的否定。本轮仅使用 train/val，**未运行 test**。

## 1. 先完成工作区精简

- 删除 `99.023 GiB`、`10,337` 个文件、`424` 个明确目标。
- `outputs/` 约从 `100 GiB` 降至 `4.7 GiB`。
- 隔离复现目录约从 `4.6 GiB` 降至 `650 MiB`。
- 数据集保持只读且未改动。
- 删除非主检查点、逐 epoch 权重、OGSR 派生特征、预测图、预览图和缓存。
- 保留 Stage-1 O1–O4、strict-v1 A1–A4、E2、作者 UABCD best，以及本次
  unified-v1 两个正式 run 的唯一主检查点。
- 主仓库已整理并本地提交：
  `04953c8 Consolidate audited Stage-2 pipeline and unified UABCD protocol`。

完整清理证据见：

- `reports/workspace_hygiene/WORKSPACE_CLEANUP_20260731.md`
- `reports/workspace_hygiene/workspace_cleanup_20260731_applied.json`

## 2. 作者代码直接复现结果

隔离目录中的作者 UABCD 代码完整跑完 40 epochs。最佳为 epoch 28：

| 指标 | 值 |
| --- | ---: |
| validation 四类 mIoU | 0.350528 |
| background IoU | 0.9170 |
| intact IoU | 0.4828 |
| damaged IoU | 0.0015 |
| destroyed IoU | 0.0009 |

这证明作者工程可以在本地数据视图上完整训练，但 mIoU 几乎完全由 background
和 intact 支撑，damaged/destroyed 分级接近失效。上游日志在保存最佳权重时
使用了 `#TEST#` 字样，但实际传入的是 validation loader；这里不把它解释为
本项目 test 结果。

直接证据：

- `../cross_modal_bdm_reproduction_20260702/runs/change_detection/UABCD_20260708_114131/log.log`
- `../cross_modal_bdm_reproduction_20260702/runs/change_detection/UABCD_20260708_114131/Seg_epoch_best.pth`

## 3. 统一实验标准

冻结标准见 `reports/stage2_v2/UNIFIED_EXPERIMENT_STANDARD_V1_20260731.md`。
核心约束如下：

- 正式 manifest：
  `manifests/stage2_v2_clean_human_reviewed_20260624/predicted_prior/`
- train/val/test：`1207 / 357 / 388`
- split canonical-event-disjoint；完整内容 SHA-256 无重复
- 相同六通道输入：
  `[pre RGB, SAR, predicted prior, SAR * prior]`
- 相同 loss：building-only weighted CE + binary auxiliary `0.3`
- AdamW，LR/WD `1e-4 / 1e-4`，effective batch `8`，最多 `40` epochs
- early stop：min `12`、patience `10`、window `3`
- 唯一主指标：val `building_only_macro_f1_3class`
- 每个 run 只保留 `best_bo_grade_macro_f1.pth`
- 必须比较真实 paired SAR 和 seed `20260731` 的事件内固定打乱 SAR
- 只有 seed 42 同时超过 E2 且 paired-shuffled `>= 0.01` 才允许扩展三种子
- test 在本轮封存，不参与调参或结论

UABCD 保留作者双流结构和 PVTv2-B2 权重，但通过适配器接入统一六通道：

- stream A：`pre RGB`
- stream B：`SAR, predicted prior, SAR * prior`
- UABCD 参数量：`25,954,115`
- E2 参数量：`24,446,067`

外部代码固定在 commit
`fe8a8580949dc4e1e38a5a1e3901cee7e20f806b`，UABCD 源码 SHA-256 为
`2600e2a539aca5ea591ab5bc5c0aeca6eafcd87bec25226a149cd189bd732c3b`。
外部快照未发现明确顶层 LICENSE，因此只做本地动态加载，没有把作者源码复制进
主仓库。

## 4. 技术门槛

| 门槛 | 结果 |
| --- | --- |
| 协议校验 | pass，0 hard error，0 warning |
| 数据完整审计 | pass，0 hard error，3 个 `qc_label=unknown` 元数据 warning |
| exact-content duplicates | 0 |
| paired smoke | pass |
| shuffled smoke | pass |
| 固定两 batch overfit | pass |
| overfit 最佳 loss 相对下降 | 64.87%，要求至少 30% |
| shuffled self-pair | 0 |
| paired 正式训练 | 16 epochs，early stop |
| shuffled 正式训练 | 13 epochs，early stop |

正式训练使用的 source-tree SHA-256：
`12cf8ac046ed122f4d30b36590385b2311dda7ac59080a2abb0ad6254cfdc887`，
对应整理后的本地提交 `04953c8`。

启动时 `run_info.json` 仍读取了旧的通用审计路径；这不改变训练输入，但会误导
溯源。本次两个 run 已分别保存 `preflight_provenance.json`，指向真正执行且
通过的严格审计和协议校验文件。启动器也已修正，今后会把唯一审计路径直接传给
trainer；没有反向改写本次原始 `run_info.json`。

## 5. seed 42 独立全量 val 结果

以下均来自各自 `best_bo_grade_macro_f1.pth` 在全部 357 个 val 样本上的独立
复核；复核值与训练保存时的最佳值完全一致。

| 方法 | BO 3-grade macro F1 | BO damage macro F1 | BO binary damage F1 | intact F1 | damaged F1 | destroyed F1 | predicted-gate 4-class macro F1 | predicted-gate damage macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E2 argmax reference | **0.358614** | **0.112705** | **0.199500** | **0.850430** | 0.126116 | **0.099295** | **0.469296** | **0.092199** |
| UABCD paired | 0.311873 | 0.096698 | 0.158676 | 0.742221 | **0.169006** | 0.024391 | 0.444105 | 0.079539 |
| UABCD shuffled | 0.321145 | 0.072047 | 0.107860 | 0.819342 | 0.078956 | 0.065137 | 0.444773 | 0.038517 |

解释：

- paired 比 E2 主指标低 `0.046741`。
- paired 比 shuffled 主指标低 `0.009273`，没有证明样本级 SAR 配对带来增益。
- paired 的 damaged F1 高于 E2，但 destroyed F1 仅 `0.024391`，整体三等级
  分级明显更差。
- shuffled 的三等级宏 F1 更高，主要由 intact F1 `0.819342` 支撑；其
  damaged/destroyed 宏 F1 只有 `0.072047`。
- paired 的 damage secondary metrics 比 shuffled 好，说明真实 SAR 可能包含
  局部损伤信号；但该信号不稳定，无法达到预注册的总体分级与负对照门槛。

## 6. 事件级检查

| val event | paired BO macro F1 | shuffled BO macro F1 | paired-shuffled |
| --- | ---: | ---: | ---: |
| bata_explosion | 0.327247 | 0.283641 | +0.043606 |
| haiti_earthquake | 0.257755 | 0.309462 | -0.051707 |
| marshall_wildfire | 0.304117 | 0.307125 | -0.003009 |
| morocco_earthquake | 0.249162 | 0.297775 | -0.048613 |
| turkey_earthquake4 | 0.326710 | 0.301784 | +0.024926 |

paired 只在 2/5 个事件上优于 shuffled，在 Haiti 与 Morocco 两个地震事件上
分别落后约 `0.052` 和 `0.049`。因此不能把少数事件上的提升外推为稳定的
跨事件 SAR 增益。

## 7. 预注册门槛裁决

| 检查 | 观测值 | 门槛 | 结果 |
| --- | ---: | ---: | --- |
| paired >= E2 | 0.311873 | 0.358614 | fail |
| paired - shuffled | -0.009273 | +0.010000 | fail |

两项要求必须同时通过，但本次两项均失败。自动门槛输出：
`outputs/stage2/unified_v1/gates/seed42_gate_20260731.json`。

最终决定：

```text
stop_after_seed42_and_record_as_rejected_transfer
```

不运行 seeds 3407/2026，不进行 test，不基于该结果继续调 UABCD 分支。后续如
需继续改进，应回到数据质量、标签可分性和跨事件稳健性，而不是围绕本次失败
配方做无界调参。

## 8. 正式产物

Paired：

- run：
  `outputs/stage2/unified_v1/S2U1_UABCD_paired/seed_42/run_20260731_011153`
- log：`logs/stage2_unified_v1_paired_seed42_20260731_011153.log`
- val metrics：上述 run 下 `val_best_grade/metrics.json`
- per-event：上述 run 下 `val_best_grade/per_event_metrics.csv`

Shuffled：

- run：
  `outputs/stage2/unified_v1/S2U1_UABCD_shuffled/seed_42/run_20260731_011210`
- log：`logs/stage2_unified_v1_shuffled_seed42_20260731_011210.log`
- val metrics：上述 run 下 `val_best_grade/metrics.json`
- per-event：上述 run 下 `val_best_grade/per_event_metrics.csv`

共同审计与协议证据：

- `outputs/stage2/unified_v1_preflight/`
- `configs/stage2_unified_v1/protocol.yaml`
- `scripts/evaluate_stage2_unified_v1_gate.py`
