# DisasterM3 第一阶段 Baseline 报告

生成日期：2026-06-16

## 结论

第一阶段 baseline 任务已经完成。四个正式实验 O1-O4 均已在服务器完成 100 epoch 训练和 test_all 测试，O1 额外完成 test_freq、test_rare、test_ok 切分测试；所有实验均保存并使用 `best_iou.pth` 进行测试。按 test_all 的建筑 IoU 排序，最佳 baseline 是 O1 U-Net ResNet34 frequent-disaster training，IoU = 0.6871，F1 = 0.8145。

主要结论：

- O1 是当前最稳的第一阶段 baseline，整体 IoU 最高。
- O4 SegFormer-B0 接近 O1，在 frequent 灾种上几乎持平，但 boundary_f1 较低。
- O2 全灾种训练牺牲了一部分 frequent 灾种性能，但对 rare 灾种更友好，尤其明显提升 fire。
- O3 DeepLabV3+ ResNet50 在本设置下不如 U-Net/SegFormer，主要表现为 precision 偏低、FP 较多。
- 各模型 boundary_f1 都偏低，说明建筑边界仍然是后续优化重点。

## 已完成工作

- 准备 Stage-1 训练代码与服务器运行脚本。
- 使用 `segmentation_models_pytorch` 建立四个 baseline。
- 完成 smoke test 和 small-batch overfit test。
- 完成 O1-O4 正式训练和测试。
- 修复 SegFormer-B0 预训练权重下载问题，将 `mit_b0.pth` 缓存到服务器。
- 汇总 test_all、frequent/rare、per-disaster 指标。
- 使用 O1 最佳 checkpoint 在测试集按灾种导出 21 个可视化样例。
- 将服务器指标和可视化结果同步回本地。

## 数据与任务

任务定义：

```text
pre-event optical RGB -> binary building mask
```

标签定义：

```text
building = mask_multiclass > 0
0 = background
1 = building
```

数据切分：

| split | sample count |
|---|---:|
| train all | 1543 |
| train frequent | 1524 |
| train rare | 19 |
| val | 506 |
| test all | 564 |
| test frequent | 409 |
| test rare | 155 |

frequent 灾种定义为 earthquake、hurricane、volcano。rare 组为 conflict、explosion、fire、flood。

## 实验设置

公共设置：

| item | value |
|---|---|
| input | pre-event optical RGB |
| native image size | 1024x1024 |
| train crop | random 512x512 crop |
| eval/test | full 1024x1024 image |
| normalization | ImageNet mean/std |
| loss | BCE + Dice |
| pos_weight | 5.0 |
| optimizer | AdamW |
| learning rate | 1e-4 |
| weight decay | 1e-4 |
| batch size | 8 |
| epochs | 100 |
| threshold | 0.5 |
| checkpoint | best validation IoU |

实验列表：

| ID | model | encoder | train split |
|---|---|---|---|
| O1 | U-Net | ResNet34 ImageNet | frequent |
| O2 | U-Net | ResNet34 ImageNet | all |
| O3 | DeepLabV3+ | ResNet50 ImageNet | frequent |
| O4 | SegFormer | MiT-B0 ImageNet | frequent |

## 训练收敛情况

| model | train count | best val epoch | best val IoU | last val IoU | note |
|---|---:|---:|---:|---:|---|
| O1 | 1524 | 91 | 0.5980 | 0.5609 | 后期有回落 |
| O2 | 1543 | 97 | 0.5968 | 0.5382 | 后期回落明显 |
| O3 | 1524 | 80 | 0.5982 | 0.5236 | 80 epoch 后收益很小 |
| O4 | 1524 | 79 | 0.5834 | 0.5501 | 79 epoch 后收益很小 |

这说明 100 epoch 对固定 baseline 是可以接受的，但后续探索阶段继续使用完整 100 epoch 效率不高。当前代码已加入可选 early stopping，探索阶段建议使用：

```bash
CUDA_VISIBLE_DEVICES=0 EARLY_STOPPING_PATIENCE=20 EARLY_STOPPING_MIN_EPOCHS=30 bash train.sh o4
```

## Test-All 总体结果

| model | IoU | F1 | precision | recall | OA | boundary_f1 |
|---|---:|---:|---:|---:|---:|---:|
| O1 U-Net ResNet34 freq | 0.6871 | 0.8145 | 0.7579 | 0.8804 | 0.9416 | 0.1643 |
| O2 U-Net ResNet34 all | 0.6765 | 0.8070 | 0.7319 | 0.8993 | 0.9374 | 0.1759 |
| O3 DeepLabV3+ ResNet50 freq | 0.6432 | 0.7829 | 0.6989 | 0.8897 | 0.9282 | 0.1512 |
| O4 SegFormer-B0 freq | 0.6793 | 0.8091 | 0.7397 | 0.8927 | 0.9387 | 0.1363 |

O1 在 test_all 上排名第一。O4 与 O1 接近，只低 0.0078 IoU。O2 的 recall 最高，但 precision 低于 O1，说明全灾种训练倾向于预测更多建筑区域。O3 recall 不低，但 precision 最低，因此整体 IoU 落后。

## Frequent vs Rare 结果

| model | frequent IoU | rare IoU | frequent F1 | rare F1 |
|---|---:|---:|---:|---:|
| O1 | 0.7309 | 0.5480 | 0.8446 | 0.7080 |
| O2 | 0.7087 | 0.5721 | 0.8295 | 0.7278 |
| O3 | 0.6852 | 0.5076 | 0.8132 | 0.6734 |
| O4 | 0.7295 | 0.5294 | 0.8436 | 0.6923 |

O1 和 O4 在 frequent 灾种上最好，说明只用 frequent 灾种训练可以强化主分布性能。O2 在 rare 灾种上最好，说明少量 rare 样本虽然数量有限，但加入训练仍能改善 rare 泛化。由于 rare train 只有 19 张，O2 对 rare 的提升有限，但方向是合理的。

## 按灾种结果

| disaster | n | O1 IoU | O2 IoU | O3 IoU | O4 IoU | best |
|---|---:|---:|---:|---:|---:|---|
| conflict | 86 | 0.5649 | 0.5541 | 0.5448 | 0.5558 | O1 |
| earthquake | 171 | 0.7039 | 0.6700 | 0.6371 | 0.7116 | O4 |
| explosion | 8 | 0.4911 | 0.4801 | 0.4390 | 0.4986 | O4 |
| fire | 55 | 0.4668 | 0.6356 | 0.4229 | 0.4190 | O2 |
| flood | 6 | 0.6849 | 0.6553 | 0.5362 | 0.6717 | O1 |
| hurricane | 105 | 0.6956 | 0.6846 | 0.6777 | 0.6859 | O1 |
| volcano | 133 | 0.8468 | 0.8536 | 0.8506 | 0.8257 | O2 |

分析：

- volcano 最容易，四个模型都达到 0.82+ IoU，O2/O3/O1 非常接近。
- earthquake 和 hurricane 是 frequent 主体，O1/O4 表现最好，符合 frequent training 的预期。
- fire 对训练分布敏感，O2 从 0.4668 提升到 0.6356，是全灾种训练最明显的收益。
- conflict、explosion 整体偏难，可能来自影像纹理复杂、灾后场景扰动或标注形态差异。
- flood 和 explosion 样本数很少，分别只有 6 和 8 张，单灾种 IoU 不宜过度解读。

## 可视化输出

使用 O1 最佳模型导出测试集可视化，因为 O1 是当前 test_all 最佳 baseline。每个灾种从 O1 的 `sample_metrics.csv` 中选择 worst、median、best 各 1 张，共 21 张。

输出目录：

```text
reports/stage1_baseline/visualizations/O1_unet_resnet34_freq_test_all/
```

主要文件：

| file/dir | content |
|---|---|
| `contact_sheet.jpg` | 21 个样例的总览图 |
| `comparisons/` | 每个样例的四联图 |
| `pred_masks/` | 模型输出二值预测掩码 |
| `gt_masks/` | GT 二值掩码 |
| `visualization_samples.csv` | 21 个样例的 ID、灾种、IoU/F1 和文件路径 |

四联图配色：

| panel | meaning |
|---|---|
| Input RGB | 原始 pre-event optical RGB |
| Label mask | GT building 区域用绿色覆盖 |
| Prediction | prediction building 区域用橙色覆盖 |
| Error overlay | TP 绿色、FP 红色、FN 蓝色 |

可以先查看：

```text
reports/stage1_baseline/visualizations/O1_unet_resnet34_freq_test_all/contact_sheet.jpg
```

## 输出文件

报告相关本地文件：

```text
reports/stage1_baseline/stage1_baseline_report.md
reports/stage1_baseline/overall_metrics_summary.csv
reports/stage1_baseline/training_summary.csv
reports/stage1_baseline/frequent_rare_metrics_summary.csv
reports/stage1_baseline/per_disaster_metrics_summary.csv
```

服务器原始指标已同步到本地：

```text
reports/stage1_baseline/raw_outputs/
```

该目录不提交 Git，只作为本地追溯材料保留。

## 是否满足第一阶段 Baseline 要求

已满足。理由：

- 数据检查、smoke test、overfit test、四个正式 baseline 训练与测试均完成。
- O1-O4 覆盖了 U-Net ResNet34、DeepLabV3+ ResNet50、SegFormer-B0，以及 frequent/all 训练数据对比。
- 测试集包含 all、frequent、rare、per-disaster 分析。
- 最佳模型 O1 已导出部分测试样例预测掩码，并与标签按明确配色进行比对。
- 结果、可视化和报告均已同步/生成在本地。

下一阶段建议以 O1 作为稳健 baseline，以 O4 作为轻量 Transformer 对照；探索阶段启用 early stopping，并重点尝试 rare-aware sampling、边界损失或后处理、阈值调优和更高分辨率 crop。
