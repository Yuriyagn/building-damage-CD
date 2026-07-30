# Stage-2 v2 qualitative SAR diagnosis

日期：2026-06-26

本次只做小规模定性诊断，没有启动新的正式训练，也没有用 test 结果调参。图像使用 clean human-reviewed strict manifest 和已训练完成的 oracle-prior O1/O2/O3。样本面板中的预测是在 512×512 crop 上重新前向得到；表格中的 BO 指标仍引用完整 test image 的已保存评估结果。O3 是 shuffled-SAR control，其预测使用评估协议中的 shuffled SAR 输入，面板中展示的 SAR 图仍是 paired SAR 作为视觉参考。

## 1. 生成的图

- 样本面板：`assets/qualitative_sar_diagnosis_20260626/qualitative_cases_grid.png`
- 单样本图：`assets/qualitative_sar_diagnosis_20260626/cases/`
- 样本级 O2 vs O1/O3 散点：`assets/qualitative_sar_diagnosis_20260626/sample_delta_scatter.png`
- 事件级 GT/预测类别占比：`assets/qualitative_sar_diagnosis_20260626/event_grade_share_gt_vs_predictions.png`
- SAR 灰度按事件/类别分布：`assets/qualitative_sar_diagnosis_20260626/sar_intensity_hist_by_event.png`

## 2. 选取样本

| id | event | note | O1 BO | O2 BO | O3 BO | O2 damaged | O2 destroyed |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| strictv1__test__noto_earthquake_124 | noto_earthquake | O2 positive: destroyed improves | 0.211 | 0.461 | 0.197 | 0.000 | 0.521 |
| strictv1__val__rwanda_volcano_00000093 | rwanda_volcano | O2 positive: strong destroyed cue | 0.139 | 0.517 | 0.122 | 0.062 | 0.826 |
| strictv1__test__turkey_earthquake3_272 | turkey_earthquake3 | O2 mild positive | 0.071 | 0.214 | 0.136 | 0.166 | 0.000 |
| strictv1__test__mexico_hurricane_00000042 | mexico_hurricane | O2 failure on damaged-heavy Mexico | 0.152 | 0.008 | 0.107 | 0.001 | 0.000 |
| strictv1__test__mexico_hurricane_00000060 | mexico_hurricane | O2 collapse; O1 bias works | 0.587 | 0.000 | 0.273 | 0.000 | 0.000 |
| strictv1__test__rwanda_volcano_00000028 | rwanda_volcano | O2 empty/low-confidence failure | 0.066 | 0.000 | 0.285 | 0.000 | 0.000 |

## 3. 定性判断

当前结果不能解释为“模型完全无法利用 SAR”。Noto earthquake、Rwanda volcano 和部分 Turkey earthquake3 样本中，O2 相对 O1/O3 有局部正向表现，说明模型确实可以在某些事件/纹理模式下利用 SAR。

但更关键的是：这种利用不稳定，不能跨事件可靠泛化。Mexico hurricane 是 test Damaged 像素的主要来源，O2 在大量 Mexico 样本中把 damaged/destroyed 压成 intact 或近似空损伤；prior-only O1 反而因为事件/标签偏置能拿到更高 damaged F1。O3 shuffled control 在一些样本上接近甚至超过 O2，也说明 paired SAR 对最终预测的支配力不足。

## 4. 问题位置

1. 单时相 post-SAR 的建筑损伤可分性弱。`sar_intensity_hist_by_event.png` 显示 intact/damaged/destroyed 的 SAR 灰度分布大量重叠，单靠后时相 SAR 强度/纹理很难稳定区分损伤等级。
2. 事件分布和类别分布漂移严重。模型在 Val/部分事件学到的 destroyed cue，到 strict test 的 Mexico damaged 主导场景中不成立。
3. 当前 loss/checkpoint 容易在 intact、damaged、destroyed 之间做错误折中。O2 Val BO macro 高，但 Test binary damage 低，说明模型更像在学习事件相关纹理与类别先验，而不是稳定 damage evidence。
4. Stage-1 建筑 mask 不是主瓶颈。本轮使用 oracle building mask 后，O2 仍没有在 strict test global 上超过 O1；问题主要在建筑内损伤分级。

## 5. 下一步建议

不要把结论写成“SAR 没用”。更准确的表述是：

```text
当前单时相 post-SAR + U-Net/CE 方案能在部分事件中提取有用线索，
但这些线索无法在 clean event-disjoint test 上稳定泛化；
瓶颈主要是 SAR 损伤信号弱、事件/标签分布漂移和类别决策不稳定。
```

下一步优先级：

1. 引入 pre/post change cue，而不是只用 post-SAR；
2. 做 event-aware 或 domain-robust 的损伤分类实验；
3. 若继续单时相 SAR，先把 binary damage recall 稳住，再谈 damaged vs destroyed 细分；
4. 保留 shuffled-SAR control，避免把事件先验误判为 SAR 贡献。
