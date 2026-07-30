# Stage-2 v2 oracle-prior exploration results

日期：2026-06-25

本报告汇总单 seed 探索实验：用真实标签建筑 mask 作为 Stage-2 prior/support，检验建筑内三分类能力。该实验用于诊断，不是正式可部署结果；正式系统仍应看 predicted-prior pipeline。

## 1. 完成状态

| ID | 实验 | 状态 | epochs | early stop | best epoch | best val BO macro | run dir |
| --- | --- | --- | --- | --- | --- | --- | --- |
| O1 | O1 oracle-prior only | completed | 40 | True | 12 | 0.2984 | outputs/stage2/v2_oracle_explore/V2_ORACLE_O1_prior_only/seed_42/run_20260625_170524 |
| O2 | O2 paired SAR + oracle prior | completed | 24 | True | 12 | 0.3591 | outputs/stage2/v2_oracle_explore/V2_ORACLE_O2_sar_oracle_prior/seed_42/run_20260625_125918 |
| O3 | O3 shuffled SAR + oracle prior | completed | 40 | False | 19 | 0.3067 | outputs/stage2/v2_oracle_explore/V2_ORACLE_O3_shuffled_sar_oracle_prior/seed_42/run_20260625_140145 |

所有 O1/O2/O3 训练均已完成，val/test 的 `best_bo_grade_macro_f1.pth` 评估产物齐全。当前没有运行中的 tmux 训练会话。

## 2. 全局指标

| split | ID | BO macro | damage macro | binary damage | intact | damaged | destroyed | oracle 4c macro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val | O1 | 0.2984 | 0.1625 | 0.3149 | 0.5700 | 0.1837 | 0.1414 | 0.4738 |
| val | O2 | 0.3591 | 0.1514 | 0.2986 | 0.7745 | 0.2624 | 0.0404 | 0.5193 |
| val | O3 | 0.3067 | 0.1496 | 0.2907 | 0.6210 | 0.2605 | 0.0387 | 0.4801 |
| test | O1 | 0.3374 | 0.2538 | 0.5368 | 0.5047 | 0.3900 | 0.1175 | 0.5031 |
| test | O2 | 0.2645 | 0.1421 | 0.2087 | 0.5092 | 0.1091 | 0.1752 | 0.4484 |
| test | O3 | 0.2597 | 0.1304 | 0.5202 | 0.5185 | 0.1546 | 0.1062 | 0.4448 |

图：

- `assets/oracle_explore_20260625/oracle_val_test_metrics.png`
- `assets/oracle_explore_20260625/oracle_event_macro.png`
- `assets/oracle_explore_20260625/oracle_test_per_event_grade_macro.png`

## 3. 关键差值

| split | comparison | Δ BO macro | Δ damage macro | Δ binary damage | Δ damaged | Δ destroyed |
| --- | --- | --- | --- | --- | --- | --- |
| val | paired SAR gain over oracle-prior only | 0.0608 | -0.0111 | -0.0163 | 0.0788 | -0.1010 |
| val | paired SAR gain over shuffled SAR control | 0.0524 | 0.0018 | 0.0079 | 0.0019 | 0.0017 |
| test | paired SAR gain over oracle-prior only | -0.0729 | -0.1116 | -0.3280 | -0.2809 | 0.0577 |
| test | paired SAR gain over shuffled SAR control | 0.0047 | 0.0117 | -0.3115 | -0.0455 | 0.0690 |

解释：

- Val 上，O2 明显高于 O1/O3：paired SAR 在 oracle mask 条件下对开发集的 BO macro 有增益。
- Test global 上，O2 低于 O1（`-0.0729`），只比 O3 高 `+0.0047`；同时 binary damage F1 明显低于 O1/O3。这不支持“真实 SAR 对当前 strict test 有稳定可泛化增益”的结论。
- O1 test 的 damaged F1 很高但 destroyed F1 很低，说明 prior-only/bias 可以命中 `Damaged` 主导事件，却不能可靠分开损伤等级。
- O2 test 的 intact 较高，但 damaged/destroyed 都低，说明模型偏向完整建筑，损伤召回/分级没有稳定起来。

和正式 clean A3（predicted prior，3 seeds test mean）相比，O2 使用真实建筑 mask 后 BO macro 为 `0.2645`，strict A3 mean 为 `0.3152`；damage macro 为 `0.1421` vs `0.1311`。这说明本轮 oracle mask 不是简单把测试性能抬高，主要瓶颈仍在建筑内损伤分级与跨事件泛化，而不只是 Stage-1 建筑支持域。

## 4. 事件宏平均

| split | ID | events | event BO macro | event damage macro | event binary damage | event damaged | event destroyed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| val | O1 | 5 | 0.2901 | 0.1361 | 0.2375 | 0.1488 | 0.1234 |
| val | O2 | 5 | 0.3453 | 0.1241 | 0.1857 | 0.1438 | 0.1043 |
| val | O3 | 5 | 0.2818 | 0.1169 | 0.2193 | 0.1579 | 0.0759 |
| test | O1 | 5 | 0.2536 | 0.1316 | 0.3160 | 0.1676 | 0.0955 |
| test | O2 | 5 | 0.2797 | 0.1353 | 0.2479 | 0.0845 | 0.1860 |
| test | O3 | 5 | 0.2253 | 0.0863 | 0.3217 | 0.1096 | 0.0631 |

事件宏平均避免 `mexico_hurricane` 的像素量过度支配。Test event-macro 下 O2 的 BO macro 和 damage macro 最高，但 binary damage 仍低于 O1/O3；结合 global 指标，结论只能定为“有局部迹象，但不稳健”。这进一步说明 oracle support 后，主要问题仍是建筑内损伤分级与跨事件泛化。

## 5. 结论

本轮探索回答了一个关键问题：即使把 Stage-1 prior 替换为真实建筑 mask，Stage-2 在 strict clean test 上也没有自然变成可靠的损伤分级器。当前证据更符合：

```text
建筑支持域误差不是唯一瓶颈；建筑内三分类本身仍不稳定。
单时相 post-SAR 在当前数据/模型/损失下，对跨事件 damage grading 的可泛化增益不足。
```

下一步不建议继续只堆 Stage-2 网络结构。更有价值的方向是：

1. 增加 pre/post temporal cue 或 SAR change feature；
2. 做 per-event 失败样本可视化，确认 O2 在哪些事件把 damage 预测成 intact；
3. 若继续 Stage-2，先从 loss/sampler 调 damaged/destroyed recall，但仍必须用 shuffled-SAR control 约束解释。

## 6. 产物

- 汇总表：`outputs/stage2/v2_oracle_explore_analysis_20260625/oracle_runs.csv`
- 差值表：`outputs/stage2/v2_oracle_explore_analysis_20260625/oracle_deltas.csv`
- 事件表：`outputs/stage2/v2_oracle_explore_analysis_20260625/oracle_per_event.csv`
- 事件宏平均：`outputs/stage2/v2_oracle_explore_analysis_20260625/oracle_event_macro.csv`
- 图像目录：`reports/stage2_v2/assets/oracle_explore_20260625/`
