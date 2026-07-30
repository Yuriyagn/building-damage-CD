# Stage-2 strict-v1 事件与损伤类别分布

日期：2026-06-22  
统计口径：四分类 mask 中 building pixels（标签 1/2/3）内部的 Intact/Damaged/Destroyed 像素占比。样本数不是实例数，connected components 不参与本统计。

## Split 总览

| Split | Events | Images | Intact | Damaged | Destroyed |
|---|---:|---:|---:|---:|---:|
| Train | 7 | 1207 | 71.58% | 14.50% | 13.92% |
| Val | 5 | 357 | 82.13% | 10.78% | 7.09% |
| Test | 5 | 388 | 60.87% | 31.54% | 7.59% |

## Train（7 events）

| Event | Images | Intact | Damaged | Destroyed |
|---|---:|---:|---:|---:|
| beriut_explosion | 6 | 23.44% | 49.49% | 27.07% |
| hawaii_wildfire | 3 | 14.34% | 5.61% | 80.05% |
| libya_flood | 8 | 51.87% | 43.48% | 4.65% |
| myanmar_hurricane | 78 | 90.52% | 8.85% | 0.63% |
| spain_la_palma_volcano | 667 | 73.33% | 0.32% | 26.35% |
| turkey_earthquake5 | 19 | 95.28% | 1.79% | 2.92% |
| ukraine_conflict | 426 | 67.19% | 31.74% | 1.07% |

## Val（5 events）

| Event | Images | Intact | Damaged | Destroyed |
|---|---:|---:|---:|---:|
| bata_explosion | 13 | 60.03% | 22.74% | 17.24% |
| haiti_earthquake | 67 | 98.38% | 1.48% | 0.14% |
| marshall_wildfire | 67 | 93.09% | 0.69% | 6.22% |
| morocco_earthquake | 143 | 99.09% | 0.83% | 0.08% |
| turkey_earthquake4 | 67 | 69.65% | 18.79% | 11.56% |

## Test（5 events）

| Event | Images | Intact | Damaged | Destroyed |
|---|---:|---:|---:|---:|
| mexico_hurricane | 172 | 14.23% | 78.71% | 7.06% |
| noto_earthquake | 26 | 83.96% | 4.74% | 11.29% |
| rwanda_volcano | 92 | 88.75% | 0.65% | 10.61% |
| turkey_earthquake1 | 72 | 87.15% | 6.20% | 6.65% |
| turkey_earthquake3 | 26 | 78.67% | 13.93% | 7.40% |

## 集中度结论

Test 的 Damaged 极度集中：`mexico_hurricane` 贡献测试集全部 Damaged 像素的 **85.09%**，HHI 为 `0.7337`，有效事件数仅 `1.36/5`。相比之下，Test Destroyed 的 top-1 事件贡献为 31.75%，有效事件数 4.16；Intact 的 top-1 为 47.64%，有效事件数 3.24。

因此需要同时报告：全局像素 F1、per-event F1、五事件等权 event-macro，以及 `mexico_hurricane` 排除后的敏感性结果。Test 仍可用于 event-held-out 比较，但不能把全局 Damaged F1 解读为对五个未知事件均匀泛化。

![各事件建筑像素损伤类别占比](../../outputs/stage2/strict_v1_distribution_20260622/event_class_distribution.png)

![Test 各类别由哪些事件贡献](../../outputs/stage2/strict_v1_distribution_20260622/test_class_concentration.png)

机器可读结果：`outputs/stage2/strict_v1_distribution_20260622/event_class_distribution.csv` 与 `distribution_summary.json`。
