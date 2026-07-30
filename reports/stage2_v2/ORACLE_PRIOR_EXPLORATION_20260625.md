# Stage-2 v2 oracle-prior exploration

日期：2026-06-25

## 目的

本实验使用真实标签派生的 oracle building mask 作为 Stage-2 输入 prior，诊断在“建筑区域完美已知”的条件下，Stage-2 是否能仅根据 post-event SAR 对建筑内像素分成：

```text
intact / damaged / destroyed
```

这不是端到端部署结果；它用于隔离 Stage-1 建筑先验误差和 Stage-2 损伤分类误差。

## 实验矩阵

| ID | Config | 输入 | 目的 |
|---|---|---|---|
| O1 | `configs/stage2_v2_oracle_explore/v2_oracle_o1_prior_only.yaml` | oracle building mask only | 测试真实建筑支撑和数据偏置能做到多少 |
| O2 | `configs/stage2_v2_oracle_explore/v2_oracle_o2_sar_oracle_prior.yaml` | SAR + oracle building mask | 主诊断：完美建筑支撑下 SAR 是否能分级 |
| O3 | `configs/stage2_v2_oracle_explore/v2_oracle_o3_shuffled_sar_oracle_prior.yaml` | shuffled SAR + oracle building mask | 负控：确认 paired SAR 是否提供有效信息 |

三组均使用 `stage2_v2_clean_human_reviewed_20260624` 的 `oracle_prior` manifests，单 seed `42`，探索优先，训练上限 `40` epochs，早停 `min_epochs=12 / patience=10`。

## 启动命令

```bash
cd '/home/yr/code/Building damage change detection/stage1_optical_building'

bash scripts/launch_stage2_v2_oracle_explore.sh o1 0
bash scripts/launch_stage2_v2_oracle_explore.sh o2 1

# O1/O2 完成后再跑
bash scripts/launch_stage2_v2_oracle_explore.sh o3 0
```

## 后评估

每个 run 完成后，用主 checkpoint 评估：

```bash
bash scripts/evaluate_stage2_v2_run.sh <RUN_DIR> <GPU_INDEX> test grade
bash scripts/evaluate_stage2_v2_run.sh <RUN_DIR> <GPU_INDEX> val grade
```

## 解读

- `O2 >> A3`：predicted prior 是主要瓶颈。
- `O2 ≈ A3`：Stage-1 prior 不是主要瓶颈，Stage-2 损伤分类能力不足。
- `O2 > O1` 且 `O2 > O3`：paired SAR 对分级有正贡献。
- `O2 ≈ O1`：模型主要依赖建筑支撑或事件/类别偏置。
- `O3 >= O2`：当前 SAR 配对信息没有形成可靠正贡献。

报告时必须同时给出 global pixel、event-macro、per-event 和 Mexico-excluded sensitivity。尤其要单独检查 `mexico_hurricane` 的 Damaged F1。
