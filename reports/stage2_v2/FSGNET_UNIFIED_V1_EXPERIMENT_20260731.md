# FSG-Net 统一迁移实验报告

- 日期：2026-07-31
- 协议：`stage2_fsgnet_unified_v1`
- 状态：迁移与实验成功完成，seed-42 晋级门槛未全部通过
- 决定：`stop_after_seed42_and_record_as_rejected_transfer`

## 1. 结论

FSG-Net 已成功迁移到本项目的 Stage-2 严格数据、三等级损伤目标和统一训练/
评估框架中。这不是代码迁移失败：协议校验、数据审计、smoke、固定两 batch
overfit、paired/shuffled 正式训练和独立全量 val 复评均已完成。

seed 42 的主结果为：

| 变体 | BO 3-grade macro F1 | 最佳 epoch | 训练结束 |
| --- | ---: | ---: | --- |
| FSG-Net paired | 0.334827 | 1 | epoch 12 early stop |
| FSG-Net shuffled | 0.323156 | 4 | epoch 15 early stop |

paired-shuffled 为 `+0.011671`，通过了预注册的 `+0.01` SAR 负对照门槛；但
paired 比 E2 reference `0.358614` 低 `0.023787`，未通过性能门槛。两项要求
必须同时满足，因此不扩展 seeds 3407/2026，不运行 test，也不追加调参分支。

在当前三种外部架构迁移中，FSG-Net paired 的主指标高于 SSFCNet paired
`0.329494` 和 UABCD paired `0.311873`，而且是唯一通过 paired-shuffled
差值门槛的方法；但它仍不能替代 E2。

## 2. 论文、源码与边界

- 论文：
  [FSG-Net: Frequency-Spatial Synergistic Gated Network for High-Resolution Remote Sensing Change Detection](https://arxiv.org/abs/2509.06482)
- 作者代码：[zxXie-Air/FSG-Net](https://github.com/zxXie-Air/FSG-Net)
- 隔离 checkout：
  `../fsgnet_reproduction_20260731/original_code`
- 冻结 commit：
  `535d7eee2f1706b1da01c076a05d35dde1cfd977`
- 作者仓库未提供模型 checkpoint；本次只保留作者架构和固定的 ResNet18
  初始化权重。
- checkout 中没有正式 LICENSE 文件。作者源码未复制进主仓库，也没有修改
  upstream 文件；主项目通过动态适配器调用隔离 checkout，并在每次建模时校验
  所有关键源文件 SHA-256。

作者方法是面向双时相 RGB 图像的 Siamese 二值变化检测网络，核心包括：

1. Discrepancy-Aware Wavelet Interaction Module（DAWIM）
2. Synergistic Temporal-Spatial Attention Module（STSAM）
3. Lightweight Gated Fusion Unit（LGFU）

作者论文在 CDD、GZ-CD 和 LEVIR-CD 上报告的二值变化 F1 与本项目的异质输入、
灾损三等级目标、严格事件不相交 split 不同，因此不得与本报告分数直接横比。

## 3. 本项目中的迁移定义

统一 Stage-2 输入为：

```text
[pre_R, pre_G, pre_B, SAR, predicted_prior, SAR*predicted_prior]
```

FSG-Net 两个三通道分支的映射为：

```text
stream A = [pre_R, pre_G, pre_B]
stream B = [SAR, predicted_prior, SAR*predicted_prior]
```

迁移保留：

- shared Siamese ResNet18
- DAWIM、STSAM、LGFU
- 作者 FPN/ASPP/fusion/DropBlock neck
- FCN head 的结构形式
- 作者数据范围对应的 `[0, 1]` identity normalization
- 固定 ResNet18 预训练初始化

为统一任务而替换：

- 作者二值 head 改为本项目 `intact/damaged/destroyed` 三等级 head
- 作者二值 loss 改为统一
  `building_only_ce_binary_aux`，binary auxiliary weight `0.3`
- 输入、采样、增强、优化器、早停、checkpoint 和评估口径均服从统一协议

因此本实验回答的是“FSG-Net 架构迁移到本项目后，在同一目标和同一训练标准下
是否优于基线并真实利用 SAR”，不是复刻作者原始数据上的论文表格。

三等级适配后的参数量为 `13,761,744`，全部可训练，与论文给出的约
`13.76M` 量级一致。

## 4. 运行时兼容处理

正式环境：

```text
Python 3.12.13
torch 2.7.0+cu126
timm 1.0.26
PyWavelets 1.8.0
dropblock 0.3.0
GPU NVIDIA GeForce RTX 4090 D
```

作者 requirements 固定 Python 3.8、timm 0.6.13、PyWavelets 1.4.1。
`PyWavelets==1.4.1` 无法在当前 Python 3.12 环境构建，因此使用兼容版本
`1.8.0`；Haar DWT/IDWT 数学定义不变。

当前 timm 的在线权重标识与作者年代不同。为避免不受控的 Hugging Face 下载，
本次固定使用本机已有的 legacy torchvision ResNet18 权重：

```text
../fsgnet_reproduction_20260731/artifacts/resnet18-5c106cde.pth
SHA-256 5c106cde386e87d4033832f2996f5493238eda96ccf559d1d62760c4de0613f8
```

加载审计只允许 classifier 的 `fc.weight/fc.bias` 为 unexpected keys；正式
训练设置 `HF_HUB_OFFLINE=1`。

还处理了两个实际兼容问题：

1. 作者自定义 wavelet autograd 在 AMP 下保存 FP32 Haar filters，而反向梯度
   为 FP16，原实现会 dtype error。适配层只把固定 Haar filters 转成当前
   autocast compute dtype，不改变 DWT/IDWT 方程；包含 DropBlock 的三 epoch
   AMP backward 回归已通过。
2. STSAM 的空间 attention 对 token 数是平方复杂度。val 原图为
   `1024x1024`，直接推理会在单个 attention 张量上触发 OOM。正式协议冻结为
   deterministic non-overlap `512x512` tiled inference，并重建原始
   `1024x1024` logits；不缩放标签、不改变训练 crop。

上述改动全部位于主项目 adapter，upstream checkout 保持 clean。

## 5. 统一协议

数据：

- manifest：
  `manifests/stage2_v2_clean_human_reviewed_20260624/predicted_prior/`
- train/val/test：`1207 / 357 / 388`
- selection split：仅 val
- test：开发期 embargoed，本实验未使用
- exact-content/full-sample duplicates：0

训练：

| 项目 | 固定值 |
| --- | --- |
| crop | 512 |
| micro batch | 4 |
| gradient accumulation | 2 |
| effective batch | 8 |
| optimizer | AdamW |
| learning rate | 1e-4 |
| weight decay | 1e-4 |
| scheduler | cosine |
| maximum epochs | 40 |
| early stop | window 3, min epoch 12, patience 10 |
| checkpoint | 仅 `best_bo_grade_macro_f1.pth` |

shuffled-SAR 使用固定 base seed `20260731` 的 within-event derangement。train
有 1207 条、val 有 357 条置换记录，均为 0 self-pair、0 singleton；val 的
确定性 split offset 后 seed 为 `20270731`。

seed-42 晋级门槛要求同时满足：

```text
paired BO 3-grade macro F1 >= 0.35861372026578153
paired - shuffled BO 3-grade macro F1 >= 0.01
```

## 6. 技术门槛与可复现性

| 检查 | 结果 |
| --- | --- |
| 协议校验 | pass，0 error，0 warning |
| 数据审计 | 0 hard error |
| 内容重复 | 各字段及 full sample 均为 0 |
| 元数据 warning | 3 类，train/val/test 的 `qc_label=unknown`；不影响输入或标签 |
| paired smoke | pass |
| shuffled smoke | pass |
| 固定两 batch overfit | pass |
| overfit 首个/最佳/最后 loss | 1.166096 / 0.603282 / 0.698837 |
| overfit 最佳相对下降 | 48.26%，要求至少 30% |
| shuffled self-pair | train 0，val 0 |
| 正式 checkpoint | 每个 run 恰好 1 个 |

正式 paired/shuffled 都从 clean 主仓库提交启动：

```text
git commit       32d3d510e9cf07abc28d96a9ba8594ff0c7db935
source-tree SHA  824088e5e8f0d552749362310ce3065e1678d32f35b10c07d11a81fee1bece2f
tracked diff     0 bytes
```

## 7. seed 42 独立全量 val 结果

以下结果均来自各 run 的唯一
`best_bo_grade_macro_f1.pth`，在全部 357 个 val 样本上重新加载并独立复评；
复评值与训练时记录的最佳值完全一致。

Building-only 指标：

| 方法 | BO macro | damage macro | binary damage | intact F1 | damaged F1 | destroyed F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E2 argmax reference | **0.358614** | **0.112705** | **0.199500** | 0.850430 | 0.126116 | **0.099295** |
| UABCD paired | 0.311873 | 0.096698 | 0.158676 | 0.742221 | **0.169006** | 0.024391 |
| UABCD shuffled | 0.321145 | 0.072047 | 0.107860 | 0.819342 | 0.078956 | 0.065137 |
| SSFCNet paired | 0.329494 | 0.056961 | 0.125865 | **0.874558** | 0.020642 | 0.093281 |
| SSFCNet shuffled | 0.340489 | 0.081564 | 0.150487 | 0.858340 | 0.056028 | 0.107099 |
| FSG-Net paired | 0.334827 | 0.072302 | 0.144819 | 0.859877 | 0.098075 | 0.046530 |
| FSG-Net shuffled | 0.323156 | 0.064185 | 0.114694 | 0.841098 | 0.111248 | 0.017122 |

统一 predicted-building gate 下的端到端指标：

| 方法 | 4-class macro F1 | damage macro F1 | binary damage F1 |
| --- | ---: | ---: | ---: |
| E2 argmax reference | **0.469296** | **0.092199** | **0.161159** |
| UABCD paired | 0.444105 | 0.079539 | 0.136952 |
| UABCD shuffled | 0.444773 | 0.038517 | 0.055446 |
| SSFCNet paired | 0.451849 | 0.050837 | 0.107752 |
| SSFCNet shuffled | 0.459267 | 0.072440 | 0.126847 |
| FSG-Net paired | 0.453751 | 0.054764 | 0.105565 |
| FSG-Net shuffled | 0.446840 | 0.049104 | 0.081556 |

关键比较：

- FSG-Net paired 比 E2 主指标低 `0.023787`。
- FSG-Net paired 比 SSFCNet paired 高 `0.005333`，比 UABCD paired 高
  `0.022954`；它是三种迁移架构中最高的 paired 主指标。
- FSG-Net paired 的 damage macro `0.072302` 仍低于 E2 `0.112705` 和
  UABCD paired `0.096698`，说明损伤等级分离仍弱。
- FSG-Net paired-shuffled 主指标为 `+0.011671`。相比 UABCD 的
  `-0.009273` 和 SSFCNet 的 `-0.010996`，FSG-Net 是唯一在本统一筛选中
  通过 SAR 负对照差值门槛的迁移方法。
- paired 相对 shuffled 的 destroyed F1 提高约 `0.029408`，binary damage
  F1 提高约 `0.030124`；但 damaged F1 反而低约 `0.013173`，仍存在两种损伤
  等级间的取舍。

## 8. 事件级检查

| val event | paired BO macro | shuffled BO macro | paired-shuffled | E2 BO macro | paired-E2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| bata_explosion | 0.341285 | 0.331400 | +0.009885 | 0.361908 | -0.020623 |
| haiti_earthquake | 0.338486 | 0.316789 | +0.021697 | 0.308206 | +0.030280 |
| marshall_wildfire | 0.300991 | 0.279972 | +0.021019 | 0.339084 | -0.038093 |
| morocco_earthquake | 0.315447 | 0.313073 | +0.002374 | 0.329740 | -0.014293 |
| turkey_earthquake4 | 0.316358 | 0.311315 | +0.005043 | 0.349356 | -0.032998 |

paired 在 5/5 个 val 事件上都高于 shuffled，说明本次正差值并非由单一事件
反转造成；但它只在 Haiti 上高于 E2，在其余 4 个事件上仍落后。可以认为
FSG-Net 捕获到一定的样本级 SAR 对应信号，但该信号尚不足以转化为总体最优的
三等级损伤识别。

## 9. 预注册门槛裁决

| 检查 | 观测值 | 门槛 | 结果 |
| --- | ---: | ---: | --- |
| paired >= E2 | 0.334827 | 0.358614 | fail |
| paired - shuffled | +0.011671 | +0.010000 | pass |

自动门槛文件：

```text
outputs/stage2/fsgnet_unified_v1/gates/seed42_gate.json
status: fail
decision: stop_after_seed42_and_record_as_rejected_transfer
test_used: false
```

因此将本分支记录为“架构迁移成功、统一性能晋级失败”。本结论不否定论文在
原始二值变化数据和作者完整训练条件下的结果。

## 10. 正式产物

Paired：

```text
outputs/stage2/fsgnet_unified_v1/S2FG1_FSGNet_paired/
  seed_42/run_20260731_161055/
```

Shuffled：

```text
outputs/stage2/fsgnet_unified_v1/S2FG1_FSGNet_shuffled/
  seed_42/run_20260731_161059/
```

每个 run 保留：

- 唯一正式 `checkpoints/best_bo_grade_macro_f1.pth`
- `completed.json`、`run_info.json`、resolved config
- train/val 固定置换记录与 class weights
- 完整训练标量历史
- `val_best_grade/metrics.json`
- per-event、per-disaster、per-region、per-sample 标量 CSV
- 唯一训练与独立复评日志

共同协议、审计、门槛证据：

```text
configs/stage2_fsgnet_unified_v1/
outputs/stage2/fsgnet_unified_v1_preflight/
outputs/stage2/fsgnet_unified_v1/gates/
```

## 11. 工作区清理

结果冻结后先执行 dry-run 并逐项检查，再应用保守清理：

- 删除两个失败/技术 overfit 的非正式 checkpoint
- 删除独立复评生成的 64 张 preview
- 删除主仓库及隔离 FSG-Net/SSFCNet checkout 的可再生 Python cache
- 初次清理复核时发现 SSFCNet 两个隔离 checkout 把 47 个历史 bytecode
  纳入了 Git。它们已全部恢复，两个 checkout 重新为 clean；清理器随后改为
  保护所有外部 Git checkout 的 tracked files，并新增对应回归测试
- 扣除上述已恢复文件并计入回归测试后重新生成的 cache，最终净删除
  `173` 个文件、`304,528,680` bytes（约 `0.284 GiB`）
- 两个 FSG-Net 正式最佳 checkpoint 与所有 E2/UABCD/SSFCNet 正式核心
  checkpoint 均通过白名单复核并保留
- 配置、源码、manifest、报告、标量、日志、审计和数据均未删除
- 最终 postcheck 为 0 个待清理条目

清理证据：

```text
reports/workspace_hygiene/
  workspace_cleanup_fsgnet_20260731_dry_run.json
  workspace_cleanup_fsgnet_20260731_applied.json
  workspace_cleanup_fsgnet_20260731_final_dry_run.json
  workspace_cleanup_fsgnet_20260731_final_applied.json
  workspace_cleanup_fsgnet_20260731_reconciled_dry_run.json
  workspace_cleanup_fsgnet_20260731_reconciled_applied.json
  workspace_cleanup_fsgnet_20260731_postcheck.json
```
