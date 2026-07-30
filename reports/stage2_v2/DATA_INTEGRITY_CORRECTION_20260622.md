# Stage-2 v2 数据完整性更正与 strict-v1 重分割

日期：2026-06-22  
状态：旧 test 结论已撤回；strict-v1 manifests 已构建并通过完整 SHA-256 与 canonical-event-disjoint 审计；尚未基于新 split 重新训练。

## 1. 更正结论

2026-06-21 的 Stage-2 v2 正式 test 主表和所有由它导出的 test 图、per-event 排名、v1-v2 test 对比及部署结论均被撤回，不能继续引用。

原因不是训练过程或 checkpoint 选择使用了 test，而是旧数据审计只检查相同 ID/路径，并只对同 ID 碰撞做内容哈希。数据包把相同内容改成不同 ID 和路径后，旧审计未能发现它们。

仍可保留为历史证据的部分：

- train 与 legacy val 没有完整内容重复，也没有 event ID overlap；
- Phase-1 三 seed validation 和 Phase-2 validation 的配方选择过程没有使用 test；
- 因此它们仍是 legacy event-held-out validation 结果，但不能替代新的严格 test。

## 2. 全内容哈希发现

修复后的审计对每一行的以下三个完整文件字节计算 SHA-256：

```text
pre_image + post_sar + mask_multiclass
```

只有三者都一致才定义为同一个完整样本，同时也分别统计每种文件的内容重复。

旧 manifest 并集结果：

| 项目 | 数量 |
| --- | ---: |
| legacy train/val/test 总行数 | 2613 |
| 完整内容重复组 | 661 |
| 应删除的重复行 | 661 |
| train-test 跨 split 重复组 | 105 |
| 单 split 内重复组 | 556 |
| 去重后唯一完整样本 | 1952 |

105 个 train-test 重复占旧 test 的 `18.62%`。这些组的 pre optical、post SAR 和四分类 mask 均完全相同，只是 `la_palma_volcano` 与 `spain_volcano` 的名称和路径被对调。例如：

```text
train: spain_volcano_00000620
test:  la_palma_volcano_00000620
```

三种文件的 SHA-256 都一致。内容证据同时证明 `la_palma_volcano` 和 `spain_volcano` 在该数据包中不是可独立分割的两个事件，strict-v1 将它们合并为 canonical event：

```text
spain_la_palma_volcano
```

旧 split 的修复审计结果为 `fail`：

- 13 个 train-test event ID overlap；
- 4 个 val-test event ID overlap；
- 661 个完整内容重复组，其中 105 个跨 split。

完整机器记录：`outputs/stage2/v2_data_integrity_20260622/legacy_split_full_hash_audit.json`。

## 3. 对旧 test 数值的影响

为量化泄漏影响，使用已保存的三 seed 逐样本混淆矩阵，排除 105 个与 train 完全相同的旧 test 样本。该结果只是一项 post-hoc 诊断，不是新的正式 test：

| 实验 | 样本数 | BO grade macro F1 | Intact | Damaged | Destroyed | Damage macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A1 prior-only | 459 | 0.2420 ± 0.0190 | 0.3307 | 0.3675 | 0.0279 | 0.1977 |
| A2 SAR-only | 459 | 0.2426 ± 0.0101 | 0.4065 | 0.2846 | 0.0366 | 0.1606 |
| A3 paired SAR + prior | 459 | 0.2509 ± 0.0088 | 0.3624 | 0.3746 | 0.0157 | 0.1952 |
| A4 shuffled SAR + prior | 458 | 0.2067 ± 0.0455 | 0.2753 | 0.2611 | 0.0837 | 0.1724 |

成对 bootstrap：

- A3−A1 grade macro：`+0.0089 [-0.0086, 0.0265]`，不再显著；
- A3−A2 grade macro：`+0.0083 [-0.0057, 0.0227]`，不再显著；
- A3−A4 grade macro：`+0.0442 [0.0281, 0.0601]`；
- A3−A4 damage macro：`+0.0228 [0.0046, 0.0405]`。

尤其是 A3 destroyed F1 从旧总表的 `0.4655` 降至 `0.0157`。因此旧 test 的主要性能和 destroyed 结论受重复样本严重放大。即使 A3−A4 在排除精确重复后仍为正，也不能把这项 post-hoc 结果升级为正式结论。

机器记录：`outputs/stage2/v2_data_integrity_20260622/legacy_test_exact_duplicate_filtered.json`。

## 4. strict-v1 构建规则

strict-v1 不修改或复制数据文件，只在 Git 仓库内生成新的 manifest 引用。构建规则在任何模型训练前固定：

1. 合并 legacy train/val/test 的 2613 行；
2. 按完整 `pre_image + post_sar + mask_multiclass` SHA-256 全局去重；
3. 合并由重复内容证明的 Spain/La Palma event alias；
4. 每个 canonical event 只能属于 train、val、test 中一个 split；
5. 为每个唯一内容保留一个确定性的代表路径，并保存完整 source-to-strict 映射；
6. 同步生成 master、predicted-prior、oracle-prior 和 no-prior manifests。

冻结的 split 方案在 `configs/stage2_v2/strict_split_v1.json`。

## 5. 新 split

| Split | 样本 | Events | Intact | Damaged | Destroyed | 灾种 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| train | 1207 | 7 | 71.58% | 14.50% | 13.92% | conflict, earthquake, explosion, fire, flood, hurricane, volcano |
| val | 357 | 5 | 82.13% | 10.78% | 7.09% | earthquake, explosion, fire |
| test | 388 | 5 | 60.87% | 31.54% | 7.59% | earthquake, hurricane, volcano |

Canonical event 分配：

```text
train:
  spain_la_palma_volcano, ukraine_conflict, myanmar_hurricane,
  libya_flood, beriut_explosion, hawaii_wildfire, turkey_earthquake5

val:
  bata_explosion, haiti_earthquake, marshall_wildfire,
  morocco_earthquake, turkey_earthquake4

test:
  mexico_hurricane, noto_earthquake, rwanda_volcano,
  turkey_earthquake1, turkey_earthquake3
```

数据事件数量有限，无法让每种 disaster type 同时出现在 val 和 test。strict-v1 优先满足：train 覆盖所有现有灾种、val/test canonical event 完全隔离、三个 split 都覆盖三种损伤等级，以及 test 保留足够 damaged 像素。该限制必须在后续结果中报告。

## 6. strict-v1 审计结果

使用 `--require-event-disjoint` 对新 master manifests 做完整审计：

| 检查 | 结果 |
| --- | ---: |
| hard errors | 0 |
| train/val/test event ID overlap | 0 |
| sample ID overlap | 0 |
| path overlap | 0 |
| pre-image 内容重复组 | 0 |
| post-SAR 内容重复组 | 0 |
| mask 内容重复组 | 0 |
| 完整样本内容重复组 | 0 |

仅剩 3 个 warning：三个 split 的 `qc_label` 都没有在 minimal package 中传播。这不造成 split 泄漏，但仍限制 per-QC 分析。

严格审计：`outputs/stage2/v2_data_integrity_20260622/strict_v1_full_audit.json`。

## 7. 使用边界

- 旧 checkpoints 是在旧 train 上训练的，不能直接在 strict-v1 test 上形成正式结果，因为 strict-v1 把原始行重新分配到了不同 split。
- 必须用 strict-v1 train/val 从头重新训练 A1–A4，再在 validation 冻结配方后一次性运行 strict-v1 test。
- 当前没有启动任何新正式训练。
- exact SHA-256 能排除完全相同文件，但不能证明不存在相邻 tile、空间重叠或经过重采样的近重复。数据缺少统一 geospatial ID 时，这一限制应继续保留。
- 后续 `relaxed-overlap-v1` 图像审计和用户复核已完成：legacy 诊断版按复核删除107个 train-side 样本，但仍非正式协议；strict train-test medium/low 均记录为 `not_overlap`，双方保留。正式训练 manifest 为 `stage2_v2_clean_human_reviewed_20260624`，已通过 0-hard-error 完整内容/event-disjoint 复审。详见 `HUMAN_REVIEWED_CLEAN_DATASET_20260624.md` 与 `STAGE2_OVERLAP_AUDIT_REPORT.md`。
- 项目主目标是像素级 `background/intact/damaged/destroyed` 语义输出。CC-surrogate 不再列为主实验或后续验收项；旧 CC 数值仅作为已撤回 test 的历史附属产物。

## 8. 产物与复现

主要产物：

- `manifests/stage2_v2_strict_v1/`
- `manifests/stage2_v2_strict_v1/build_summary.json`
- `manifests/stage2_v2_strict_v1/deduplication_groups.json`
- `manifests/stage2_v2_strict_v1/source_to_strict.csv`
- `manifests/stage2_v2_strict_v1/content_hashes.jsonl`
- `configs/stage2_v2_strict_v1/`

重新构建：

```bash
/home/yr/miniconda3/envs/sam3/bin/python \
  scripts/build_stage2_v2_strict_splits.py \
  --data-root '../datasets/DisasterM3_optical_sar_damage_minimal_v0.2' \
  --source-manifest-root '../datasets/DisasterM3_optical_sar_damage_minimal_v0.2/manifests' \
  --split-plan configs/stage2_v2/strict_split_v1.json \
  --output-root manifests/stage2_v2_strict_v1
```

构建脚本拒绝覆盖已有输出。若要复现，应使用新的空输出目录，并比较 `build_summary.json` 和 `content_hashes.jsonl`。

严格审计：

```bash
/home/yr/miniconda3/envs/sam3/bin/python \
  scripts/audit_stage2_v2_data.py \
  --data-root '../datasets/DisasterM3_optical_sar_damage_minimal_v0.2' \
  --manifest-root manifests/stage2_v2_strict_v1 \
  --out outputs/stage2/v2_data_integrity_20260622/strict_v1_full_audit.json \
  --require-event-disjoint
```

对应的 strict-v1 A1–A4 配置已放在 `configs/stage2_v2_strict_v1/`。正式训练仍需按三 seed、固定 negative control、validation-only selection 和 tmux 规则执行。

严格版预检与训练入口：

```bash
cd '/home/yr/code/Building damage change detection/stage1_optical_building'
bash instruction.sh stage2_v2_strict_check_runtime

# 示例；只在准备开始正式训练时执行
bash scripts/launch_stage2_v2_strict_run.sh a1 42 0
```

严格 launcher 会再次执行 full-content/event-disjoint 预检、检查 GPU 与重复 tmux 任务，并写入唯一的 audit、log 和 output 路径。旧 `launch_stage2_v2_run.sh` 已默认拒绝启动，避免误用被撤回的 legacy split。本文档修订过程未启动训练。
