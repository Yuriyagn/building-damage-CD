# 建筑损伤变化检测项目：全景整理、实验审计与下一步决策材料

> 证据快照：2026-08-14（Asia/Shanghai）<br>
> 目的：把本文件整体交给另一个大模型，使其能够在不访问本机的情况下，理解项目背景、数据谱系、实验设计、实际进展、结论边界和当前漏洞，并据此提出下一步工作。<br>
> 重要约定：本文把“已由磁盘证据验证的事实”“历史结果”“诊断性结果”“推测/待验证假设”分开书写。不同数据划分上的分数不能横向直接比较。

---

## 0. 一页结论

### 0.1 当前项目真正回答到了什么

项目要做的是像素级四分类建筑损伤制图：`background / intact / damaged / destroyed`。现有主线将任务拆成两级：

1. Stage-1：灾前光学 RGB → 二值建筑先验；
2. Stage-2：灾后 SAR、建筑先验，部分实验再加入灾前 RGB → 建筑内部三等级 `intact / damaged / destroyed`；最后通过建筑先验门控回到四分类图。

现有实验已经相当充分地证明：**当前模型在同一数据体系内能学到某些与事件、建筑支撑和 Destroyed 类相关的信号，但尚不能证明它学到了跨事件稳定、可归因于灾后 SAR 的损伤等级判别能力。** 主要症状是：

- Damaged 与 Destroyed 在不同事件和不同随机种子间大幅互换；
- 严格划分上，`SAR + predicted prior` 在测试集反而显著差于 `prior-only`；
- paired SAR 虽有时优于 within-event shuffled SAR，但优势并不稳定地落在 `Damaged` 和 damage macro F1 上；
- 添加 BRIGHT 同事件训练样本的 R4，在新划分验证集上均值变好，但收益主要来自 Destroyed，且与更多优化步数、数据来源/标注域变化混杂；
- 当前所有已经查看过的“test”都不能再承担一个全新的、一次性最终确认角色。

### 0.2 最高优先级的实验漏洞

按会不会改变科学结论排序：

1. **旧 Stage-1 与 Stage-2 v1 基线存在精确 train-test 内容泄漏。** 旧 1543/506/564 划分中有 105 个精确 train-test 重复组。Stage-1 O1 的 564 张 test 中有 105 张（18.62%）在其训练集中出现；其全 test IoU/F1 为 0.6871/0.8145，去掉这些重复样本后仅为 0.6603/0.7954，而泄漏子集为 0.8654/0.9279。Stage-2 v1 也使用该旧划分，因此其分数只能作为受污染的历史记录，不能作为正式基线。
2. **Stage-1 → Stage-2 的 predicted prior 不是 out-of-fold。** 旧 strict-clean Stage-2 validation 中 154/357 张灾前图像曾被 Stage-1 O1 用于训练；最新 event-group validation 中仍有 119/290（41.03%）重合。Stage-2 的模型选择及输入因此享受了“见过该验证图像”的上游先验。最新 event-group test 的该项精确重合为 0/290，这是好消息，但不能修复验证选择偏差。
3. **测试集被反复暴露。** 输出树中共有 44 个 `test*/metrics.json`：5 个 Stage-2 v1、12 个已撤回 legacy-v2、12 个 strict A1–A4、3 个 oracle 探索、5 个 pre-change 探索、6 个 R0/R4 诊断测试和 1 个 legacy A0。仅 clean strict test 就被用于至少 20 个模型结果；最新 event-group test 也已被 R0/R4 三种子查看。它们可以保留作“历史诊断集”，但不能再被称为未触碰最终测试集。
4. **事件×类别支持严重绑定。** 最新 event-group test 中，Mexico Hurricane 提供约 98.80% 的 Damaged 像素；validation 的建筑像素则有 95.61% 为 Intact。模型的“类别泛化”与“事件识别/域识别”没有被充分解耦。
5. **R0/R4 的因果归因不干净。** R4 不仅多了 262 张 BRIGHT 训练图像，而且每 epoch 有更多 batch、实际早停 epoch 也更晚。按 batch size 8 粗算，三个种子的总优化步数约为 R0 的 1.34×、1.46×、2.04×；还同时引入来源域和标注流程变化。当前结果不能单独归因于“更好的数据覆盖”。

### 0.3 当前最合理的总判断

- **不能声称**：当前系统已经解决跨灾害事件的三等级损伤泛化；SAR 已被证明稳定提供了损伤等级信息；R4 已经正式优于 R0；旧 Stage-1/Stage-2 分数是可信最终基线。
- **可以声称**：精确内容去重和 canonical event-disjoint 划分已建立；paired/shuffled 负对照已制度化；外部模型统一转移实验遵守了 validation-only 硬门；事件捷径和事件×类别绑定已被实验证据发现；R4 在历史 validation 上出现了值得继续验证的数据收益信号。
- **下一步不应先继续堆架构。** 应先重建无交叉污染的 Stage-1 prior、封存/更换最终外部事件评估、做训练预算与样本来源匹配的 R0/R4 因果对照，并优先补充新的独立事件中的 Damaged/Destroyed 支撑。

---

## 1. 项目背景、目标与科学问题

### 1.1 应用目标

输入多灾害事件的灾前光学影像、灾后 SAR 影像及建筑信息，输出每个像素的四等级标签：

| 标签 | 含义 |
|---|---|
| 0 | background |
| 1 | intact building |
| 2 | damaged building |
| 3 | destroyed building |

这是跨模态、跨事件、类别极不平衡的语义分割任务。难点并不只是“有没有建筑”，而是：

- 光学与 SAR 成像机制、分辨率、配准质量不同；
- 不同灾害事件的传感器、地貌、建筑形态、强度分布和标签先验不同；
- `Damaged` 往往边界模糊、像素少、事件集中；
- 模型可能通过背景纹理、事件身份、建筑轮廓或类别先验取得表面高分，而没有使用真正的灾后变化信号。

### 1.2 两阶段建模逻辑

当前主线的因果结构可以写成：

```text
pre-event RGB ──> Stage-1 building model ──> predicted building prior
       │                                             │
       └──────────────────────┐                      │
post-event SAR ───────────────┼──> Stage-2 grade model ──> intact/damaged/destroyed
                              │                      │
                              └──────────────────────┴──> predicted-prior gate
                                                       └──> four-class map
```

其中需要分别回答三个问题：

1. Stage-1 是否在从未见过的事件/图像上产生可靠建筑先验？
2. 固定建筑支撑后，真实配对 SAR 是否比打乱 SAR、prior-only 或 pre-only 更能预测损伤等级？
3. 这种增益是否在多个独立事件、多个种子和 Damaged/Destroyed 两类上都成立？

### 1.3 正式指标层级

当前项目规定的主要指标是建筑真值区域（building oracle，BO）内三等级 macro F1：

```text
BO grade macro F1 = mean(F1_intact, F1_damaged, F1_destroyed)
damage macro F1   = mean(F1_damaged, F1_destroyed)
```

还应同时报告：

- `F1_damaged`、`F1_destroyed`；
- binary damage F1（intact vs damaged-or-destroyed）；
- event macro / per-event × per-class 指标；
- predicted-prior gate 下的端到端四分类指标。

背景占绝大多数，因此全图 accuracy 或普通 mIoU 不能取代 BO 三等级指标。Oracle support 只用于诊断，不能作为部署结果。连通域指标若出现，只能标为 `cc_surrogate`，不用于模型选择。

一个尚未完全统一的细节是：部分分析使用“先汇总像素混淆矩阵再算 F1”，部分 bootstrap 使用“逐样本算分再平均”。这两个 estimand 不同，已出现全局 P2−P1 为正、逐样本 bootstrap 均值却为负的现象。今后必须同时明确 `pooled-pixel`、`sample-macro`、`event-macro`，不能只写同一个“macro F1”。

---

## 2. 工作区、代码、环境和当前运行状态

### 2.1 目录边界

- 工作区根目录：`/home/yr/code/Building damage change detection`
- 真正的 Git 仓库：`stage1_optical_building/`
- 工作区根本身不是有效 Git 仓库。
- Stage-2 当前主数据根：`datasets/DisasterM3_optical_sar_damage_minimal_v0.2`，约 10 GB。
- 原始/备份数据约 39 GB，practice/QC 源约 4.4 GB；数据应视为只读。
- 项目 `outputs/` 约 19 GB，报告约 45 MB。

外部作者代码位于独立目录，避免与主仓混写：

- `cross_modal_bdm_reproduction_20260702/`：UABCD 作者代码在 strict-v1 symlink data view 上完整训练 40 epochs；随后由主仓动态加载做统一迁移。
- `ssfcnet_reproduction_20260731/`：作者当前 master 已删除模型和训练入口，只能固定历史 commit `284d0bf...` 的架构，因此准确名称是“作者架构迁移”，不是论文训练配方复现。
- `fsgnet_reproduction_20260731/`：固定作者 commit `535d7eee...`，把原双时相 RGB 二值变化架构迁移为异质六通道、三等级目标。

三个外部快照均未发现可依赖的正式顶层 LICENSE，作者源码没有复制进主 Git 仓；主项目通过动态适配器和源文件 SHA-256 固定边界。其论文原始任务、数据和指标也不能与本项目分数直接横比。

### 2.2 环境

规定环境：

```bash
source /home/yr/miniconda3/etc/profile.d/conda.sh
conda activate sam3
```

本次在该环境执行单元测试：

```text
Ran 47 tests in 1.533s
OK
```

仅出现依赖弃用 warning，未发现测试失败。

### 2.3 Git 与可复现性状态

当前 `main` 分支 HEAD：`3adb969 Record FSG-Net unified transfer`。工作树并不干净：9 个 tracked 文件有修改，约 372 insertions / 26 deletions，另有 event-group、event-shortcut 等未跟踪配置、脚本、manifest、报告和测试。

主要修改涉及：

- `instruction.sh`；
- Stage-2 dataset、metric、train/test 代码；
- strict distribution 汇总脚本；
- `tests/test_stage2_v2_core.py`；
- 新增 event-group / event-shortcut 工作流。

当前工作树计算出的代码状态哈希为：

```text
6d14cc347f97c1ce954baafe7b36ec2cc161cb28c29ad9302d146cc80a6ca29c
```

R0/R4 的 `run_info.json` 保存的是较早快照哈希 `245787...`；tracked diff 哈希一致，但当前多了一个未跟踪汇总脚本。这意味着运行本身留有代码快照信息，但目前尚无一个冻结、可 checkout 的完整提交代表最新实验状态。

### 2.4 2026-08-14 10:01 的运行快照

- 机器有两张 RTX 4090 D；GPU 0 被一个与本项目无关的 `train_stfdiff.py` 占用；GPU 1 空闲。
- 没有发现本项目 Stage-2 训练进程。
- 没有 tmux session。
- 本次审计没有启动训练或全量推理。

这是时间点快照，不代表之后仍然如此。

---

## 3. 数据谱系与划分演化

### 3.1 旧版 legacy 划分：已污染

旧 Stage-1 和 Stage-2 v1 使用约 1543/506/564 的 train/val/test 划分。2026-06-22 的全量 SHA-256 审计发现：

- 2613 个样本中有 661 个 full-sample 重复组；
- 105 个组跨 train-test 精确重复；
- 556 个重复组位于同一 split 内。

因此 legacy Stage-2 v2 test 被正式撤回。进一步核对表明，Stage-1 O1 和 Stage-2 v1 同样使用该旧拆分，所以它们的测试数字也不应再承担正式比较功能。

### 3.2 strict-v1 / human-reviewed clean：精确内容与事件身份层面合格

全局去重后得到 1952 个 unique 样本，正式 clean manifest 为：

```text
manifests/stage2_v2_clean_human_reviewed_20260624/
```

| split | 样本数 | canonical event 数 | 精确内容跨 split | canonical event 跨 split |
|---|---:|---:|---:|---:|
| train | 1207 | 7 | 0 | 0 |
| val | 357 | 5 | 0 | 0 |
| test | 388 | 5 | 0 | 0 |

自动审计为 0 hard errors；relaxed-overlap 候选也已经人工复核。需要保留两点边界：

1. manifest 中 `qc_label` 缺失仍产生 warning；
2. 图像内容哈希与人工近重复审查并不能证明正式的地理 footprint 独立性。

### 3.3 strict clean 的事件×类别分布

建筑区域内像素占比：

| split | Intact | Damaged | Destroyed |
|---|---:|---:|---:|
| train | 71.58% | 14.50% | 13.92% |
| val | 82.13% | 10.78% | 7.09% |
| test | 60.87% | 31.54% | 7.59% |

strict test 的 Damaged 有 85.09% 来自单一事件 `mexico_hurricane`，Damaged 的有效事件数仅约 1.36。即使没有精确数据泄漏，类别和事件仍高度绑定。

### 3.4 最新 event-group R0/R4 划分

2026-08-03 重新整理 canonical event identity，将同一灾害的别名/子事件合并：

- Turkey 子事件合并；
- Rwanda / Congo 归并为 Nyiragongo；
- Spain / La Palma 归并；
- R0：1372/290/290，分别有 7/4/3 个 event groups；
- R4：只在训练集增加 262 个 BRIGHT 样本，总训练数 1634；val/test 与 R0 完全相同。

R4 新增样本来自已有训练事件：Beirut 116、Hawaii 56、Libya 90。它增加了同事件内部支撑，**没有增加新的独立训练事件组**。

审计结果：

- exact duplicate：0；
- canonical event overlap：0；
- 3 个 near-overlap 候选人工判定为不同场景；
- 但 val/test 只有 350/580（60.34%）记录具有可用 georeference，另 230 条无法做正式 footprint 独立证明。

最新分布更极端：

| split | Intact | Damaged | Destroyed |
|---|---:|---:|---:|
| val | 95.612% | 2.138% | 2.250% |
| test | 37.506% | 54.224% | 8.270% |

- test 中 Mexico Hurricane 提供 `20,714,379 / 20,965,461 = 98.80%` 的 Damaged 像素；
- val 中 Marshall 提供 58.64% 的 Destroyed，Bata 提供 37.72%；
- test 只有 3 个事件，event-level 不确定性很难由常规样本 bootstrap 正确表达。

这使 validation 和 test 对 Damaged 的分布差异比 strict clean 旧划分还大。

### 3.5 Stage-1 prior 的跨阶段重合审计

把 Stage-2 manifest 的 `source_id/pre_image` 与 Stage-1 O1 的训练 manifest 做精确连接：

| Stage-2 划分 | 与 Stage-1 O1 train 的精确灾前图重合 |
|---|---:|
| strict-clean train | 738 / 1207 |
| strict-clean val | 154 / 357 = 43.14% |
| strict-clean test | 79 / 388 = 20.36% |
| latest event-group train | 852 / 1372 |
| latest event-group val | 119 / 290 = 41.03% |
| latest event-group test | 0 / 290 |

latest validation 的重合主要是 Morocco 112 张和 Haiti 7 张。这个问题不是 Stage-2 自身 manifest 的 train/val 重复，而是**上游 prior 生成器使用了 Stage-2 validation 的输入图像训练**。因此：

- predicted-prior gate 的绝对表现被高估的风险最大；
- prior 也是 Stage-2 输入，所以 BO grade 指标也可能间接受影响；
- val 上做架构、损失、阈值、早停选择并非完全 out-of-sample；
- latest test 精确重合为 0 是积极结果，但选模过程仍被污染。

正确修复是按 Stage-2 event split 重新训练 Stage-1，或为每个 Stage-2 fold 生成 cross-fitted / out-of-fold prior，而不是只重新导出同一 O1 的预测。

该审计的源和连接键为：Stage-1 O1 的实际训练清单 `datasets/DisasterM3_aria2_building_damage_practice/splits/stage1_optical_building/building_extract_train_sar_qc_freq.jsonl` 中的 `id`，对 Stage-2 各 `stage2_master_{split}.jsonl` 中的 `source_id`。这是精确样本身份连接，不是文件名模糊匹配。

### 3.6 测试暴露台账

按 `outputs/stage2/` 下路径名以 `test` 开头且文件名为 `metrics.json` 的产物盘点：

| 数据/阶段 | test metrics 数 | 状态 |
|---|---:|---|
| Stage-2 v1 | 5 | legacy split，受污染历史结果 |
| legacy Stage-2 v2 | 12 | 已因重复样本正式撤回 |
| strict A1–A4，3 seeds | 12 | 首次冻结结果有诊断意义，之后 test 已暴露 |
| oracle exploration | 3 | 探索性查看 clean strict test |
| pre-change C1–C5 | 5 | 探索性查看 clean strict test |
| latest event-group R0/R4，3 seeds | 6 | 明确标记 diagnostic test |
| legacy A0 | 1 | 历史规则/参考 |
| **合计** | **44** | 不再存在未触碰的现有 test |

这是一份“评估产物下界台账”，不是操作系统层面的人类文件访问日志；它能证明这些评估曾经被执行并保留，但不能还原每个结果对后续决策造成了多大影响。clean strict test 至少已经生成 12+3+5=20 个学习模型结果；latest event-group test 已生成 6 个结果。因此两者今后都应标记 `historical_diagnostic_only`。

---

## 4. 实验设计与实际进展时间线

先给出状态总表，避免把“有配置”“已经训练”“已经通过门槛”混为一谈：

| 实验族 | 实际状态 | 数据/选择范围 | 当前证据地位 |
|---|---|---|---|
| Stage-1 O1–O4 | 已完成 | legacy train/val/test | 测试受污染，需按新协议重训 |
| Stage-2 v1 S0–S3 | 已完成 | legacy split | 测试受污染，历史记录 |
| legacy Stage-2 v2 A1–A4/B2/B3 | 已完成 | legacy split | test 全部撤回；无泄漏 val 过程仅作历史开发证据 |
| strict-clean A1–A4，3 seeds | 已完成 | clean 1207/357/388 | 首次 test 为诊断；prior 非 OOF，test 后续暴露 |
| Oracle O1–O3 | 已完成 | strict clean，seed 42 | 机制诊断，查看过 test |
| Pre-change C1–C5 | 已完成 | strict clean，seed 42 | 机制诊断，查看过 test |
| Predicted-prior P1–P5 | 已完成 | strict val，seed 42 | validation-only 探索 |
| OGSR D4–D7 | 已完成 | strict val，seed 42 | 未超过 P2，分支停止 |
| Objective E1–E3 / binary-aux F1–F3 | 已完成 | strict val，seed 42 | E2 为单种子开发 reference |
| Phase-5 threshold/adaptive | 已完成 | 同一 strict val | post-hoc 多重搜索，不是确认结果 |
| UABCD/SSFCNet/FSGNet unified | 已完成 | strict train/val，seed 42 | 均未通过完整硬门，test 未用，分支停止 |
| Shortcut D1–D4 / balanced E1 | 已完成 | strict development events | 诊断捷径；E1 已失败但旧结论文档滞后 |
| event-group R0/R4，3 seeds | 已完成 | 1372/290/290；R4 train+262 | validation 有信号；test 已作历史诊断；存在预算混杂 |

### 4.1 Stage-1 光学建筑提取

Stage-1 共保留四个历史模型，均为 seed 42，并在同一旧 test-all 上评价：

| 模型 | 训练范围/架构 | test-all IoU | test-all F1 | Boundary F1 |
|---|---|---:|---:|---:|
| O1 | U-Net ResNet34，frequent disasters | 0.6871 | 0.8145 | 0.1643 |
| O2 | U-Net ResNet34，all disasters | 0.6765 | 0.8070 | 0.1759 |
| O3 | DeepLabV3+ ResNet50，frequent | 0.6432 | 0.7829 | 0.1512 |
| O4 | SegFormer-B0，frequent | 0.6793 | 0.8091 | 0.1363 |

O1 因 IoU 最高成为后续 predicted-prior 生成器。但这四组都属于 legacy protocol；至少 O1 的训练/test 泄漏已被精确量化，其他模型也不能因为尚未逐一给出过滤数字就恢复正式地位。O1 的泄漏拆分如下：

| 指标 | 全部 564 test | 去掉 105 个泄漏样本后的 459 | 105 个泄漏子集 |
|---|---:|---:|---:|
| IoU | 0.6871 | 0.6603 | 0.8654 |
| F1 | 0.8145 | 0.7954 | 0.9279 |

这里先由 `deduplication_groups.json` 找出同时含原始 train/test member 的 105 个组，再用 test member 的 `source_id` 连接 O1 的 `test_all/sample_metrics.csv`；表中数字由各子集 TP/FP/FN 汇总后重新计算，属于 pooled-pixel IoU/F1。“去掉泄漏样本”的数字只是事后诊断，不是重新设计后的正式 Stage-1 测试。Stage-1 需要按新的 event-group 协议重新训练、验证和生成 prior。

### 4.2 Stage-2 v1：历史基线，但不能再用于正式结论

旧 B1 系列的完整历史表如下：

| 模型 | BO macro | damage macro | Intact | Damaged | Destroyed |
|---|---:|---:|---:|---:|---:|
| S0 no prior | 0.3314 | 0.3233 | 0.3476 | 0.2114 | 0.4352 |
| S1 oracle prior | 0.3927 | 0.3028 | 0.5726 | 0.2152 | 0.3903 |
| S2 predicted prior | **0.4120** | **0.3376** | 0.5609 | 0.2127 | 0.4626 |
| S3 oracle prior-only | 0.3592 | 0.2684 | 0.5408 | 0.3311 | 0.2056 |
| S3 predicted prior-only | 0.3176 | 0.2800 | 0.3928 | 0.4142 | 0.1458 |

由于其使用含 105 个 train-test 精确重复组的旧拆分，上述数字只能放在“历史记录”栏，不能与 clean strict 或 event-group 新实验直接比较。

### 4.3 strict A1–A4：核心模态与负对照

设计：

| 编号 | 输入/作用 |
|---|---|
| A1 | predicted prior only |
| A2 | paired SAR only |
| A3 | paired SAR + predicted prior |
| A4 | within-event shuffled SAR + predicted prior |

三种子 validation 均值：

| 模型 | BO macro F1 | damage macro F1 |
|---|---:|---:|
| A1 | 0.33258 | 0.12904 |
| A2 | 0.35336 | 0.12916 |
| A3 | 0.35398 | 0.13049 |
| A4 | 0.30278 | 0.11122 |

A3 的总 BO 看似非常稳定，但类别语义并不稳定：

| A3 seed | val BO macro | Damaged F1 | Destroyed F1 |
|---:|---:|---:|---:|
| 42 | 0.35397 | 0.26477 | 0.02171 |
| 3407 | 0.35188 | 0.03393 | 0.21702 |
| 2026 | 0.35609 | 0.22098 | 0.02453 |

这是一项关键细节：若只看 BO macro，会误以为三种子高度复现；实际模型在“把损伤叫 Damaged 还是 Destroyed”上发生近乎相反的解。

关键差值：

- A3−A2：BO `+0.00062`，damage `+0.00133`，prior 的等级增益几乎为零；
- A3−A1：BO `+0.02140`，damage `+0.00144`，总体提升主要不是 Damaged/Destroyed 稳定提升；
- A3−A4：BO `+0.05120`，damage `+0.01927`，说明真实配对 SAR 包含某种有效信息，但还不能证明是可泛化损伤等级信息。

A3 各种子间 `Damaged ↔ Destroyed` 取舍明显，提示类别语义不稳定。

第一次冻结后 strict test 均值：

| 模型 | BO macro | damage macro | Damaged F1 | Destroyed F1 |
|---|---:|---:|---:|---:|
| A1 | 0.37728 | 0.25362 | 0.40077 | 0.10648 |
| A2 | 0.33521 | 0.16220 | — | — |
| A3 | 0.31518 | 0.13107 | 0.05265 | 0.20948 |
| A4 | 0.28852 | 0.14537 | — | — |

paired bootstrap：

- A3−A1：BO `−0.0621`，95% CI `[−0.0875, −0.0396]`；damage `−0.1226`；
- A3−A2：BO `−0.0200`；
- A3−A4：BO `+0.0267`，95% CI `[+0.0042, +0.0458]`，但 damage `−0.0143`，95% CI `[−0.0441, +0.0133]`。

因此严格解释是：A3 在总体 BO 上优于 shuffled control，但这个优势没有转化为稳定的 damage macro；在该测试分布上 prior-only 明显更好。由于此后同一 test 被继续查看，该次结果应保留为“首次冻结后的历史诊断”，不能再重复用于新模型选择。

### 4.4 Oracle prior 探索：建筑支撑并不是唯一瓶颈

seed 42：

| 模型 | val BO | test BO |
|---|---:|---:|
| O1 oracle prior only | 0.2984 | 0.3374 |
| O2 paired SAR + oracle prior | 0.3591 | 0.2645 |
| O3 shuffled SAR + oracle prior | 0.3067 | 0.2597 |

O2 在 test 仅比 O3 高约 0.0047，却比 O1 低约 0.0729。即使给定真值建筑支撑，跨事件损伤分级仍不稳定，说明瓶颈不只是 Stage-1 建筑掩膜。

### 4.5 灾前 RGB、SAR texture 与目标函数探索

#### Oracle pre-change C1–C5

`pre RGB + paired SAR + oracle prior` 的单种子 test BO 曾达到约 0.4646，shuffled 对照约 0.3840；但样本 bootstrap 的差值区间包含 0，而且这个阶段直接查看了 strict test。只能视作“灾前信息可能有帮助”的线索。

#### Predicted-prior P1–P5（validation-only，seed 42）

| 模型 | val BO |
|---|---:|
| P1 pre + predicted prior | 0.3361 |
| P2 pre + paired SAR + predicted prior | 0.3457 |
| P3 pre + shuffled SAR + predicted prior | 0.2842 |
| P4 pre + SAR texture + predicted prior | 0.3332 |
| P5 pre + shuffled texture + predicted prior | 0.2923 |

P2 的全局 pooled 指标比 P1 高约 0.0097，但逐样本 bootstrap 汇总给出相反方向，暴露了指标聚合定义未统一的问题。该阶段只有一个种子。

#### OGSR D4–D7

最佳候选 D6 val BO 约 0.3375，未超过 P2 约 0.3457，因此没有替换主线。

#### Objective E1–E3 与 binary-aux F1–F3

seed 42 validation：

| 候选 | val BO | 说明 |
|---|---:|---|
| E1 focal | 0.3302 | 未超过 P2 |
| E2 binary auxiliary | 0.358614 | 本轮最佳 |
| E3 disaster-balanced sampler | 0.2602 | 明显下降 |

E2−P2 约 `+0.0129`，但样本 bootstrap 95% CI 约 `[−0.0002, +0.0271]`，下界仍接近/穿过 0，且 event damage 指标没有同步改善。F1–F3 的 binary auxiliary 权重扫描也未超过 E2。E2 至今仍是旧 strict validation 上的 reference，但它是**多轮探索后的单种子最优项**，不是三种子正式确认结果。

### 4.6 Phase-5 后处理阈值扫描：明显的 validation 多重比较风险

基于同一个 E2 checkpoint 和同一 357 张 validation：

| 后处理 | BO macro | damage macro | predicted-gate 4-class |
|---|---:|---:|---:|
| argmax | 0.3586 | 0.1127 | 0.4693 |
| global offset −0.25 | 0.3613 | 0.1358 | 0.4733 |
| adaptive：evidence≥1 时 −0.625 | 0.3794 | 0.1596 | 0.4847 |

adaptive 规则只在 86/357 个样本上激活。它来自较大的阈值/证据扫描，并仍在同一单一 validation 上选择，没有 nested validation、预注册或多种子复验。因此 0.3794 是乐观的 post-hoc validation 结果，不应作为独立泛化证据。

### 4.7 外部模型统一转移：流程规范，但结论只限于当前 screening recipe

2026-07-31 建立统一协议：

- 同一 strict 1207/357/388；
- 统一六通道 `[pre RGB, SAR, predicted prior, SAR×prior]`；
- 相同损失、训练预算和 seed 42 screening；
- paired 与固定 within-event shuffled 成对；
- 硬门：paired BO ≥ E2 `0.358614`，且 paired−shuffled ≥ 0.01；
- 未过门不跑多种子、不看 test。

结果：

| 架构 | paired val BO | shuffled val BO | 差值 | 结论 |
|---|---:|---:|---:|---|
| UABCD unified | 0.311873 | 0.321145 | −0.009272 | 两门均失败 |
| SSFCNet unified | 0.329494 | 0.340489 | −0.010995 | 两门均失败 |
| FSGNet unified | 0.334827 | 0.323156 | +0.011671 | 负对照门通过，绝对门失败 |

作者代码直接复现的 UABCD validation mIoU 约 0.3505，但 Damaged IoU 0.0015、Destroyed IoU 0.0009，背景主导明显。

可复现性名称必须准确：UABCD 是完成作者工程直接训练后再做统一迁移；SSFCNet 因上游当前版本缺模型、训练入口、checkpoint 和完整依赖，只是固定历史源码的架构迁移；FSG-Net 原本是双时相 RGB 二值变化检测，当前实验替换为本项目三等级 head、loss、输入和评价协议，也属于受控迁移而非论文表格复现。

这组实验最值得保留的是纪律：test embargo 和硬门都执行了。它不能证明这些论文架构本身无效，因为共享超参数是受控筛选，不是每个架构的最优调参；单种子和固定 `+0.01` 门限也没有置信区间。

### 4.8 事件捷径诊断 D1–D4 与事件×类别采样 E1

#### D1–D3 shortcut controls

| 控制 | val BO | event macro | 解释边界 |
|---|---:|---:|---|
| D1 pre-only | 0.31413 | 0.29015 | 灾前外观/事件域本身有较强预测力 |
| D2 background-only | 0.32178 | 0.31884 | 高分，但不是纯背景控制 |
| D3 metadata/event prior | ≈0.3006 | ≈0.3006 | 大多退化为全 Intact，damage 为 0 |

D2 在真值建筑区域内做遮挡，建筑“洞”的形状会泄漏建筑支撑；训练仍使用 damage-aware crop，裁剪位置也受目标标签影响。因此它只能说明背景/事件上下文强，不能声称“完全不看建筑和标签也能达到 0.3218”。应增加不使用 GT mask、且裁剪策略与标签无关的背景控制。

#### D4 leave-one-event-out（strict train 内 7 folds，seed 42）

| held-out event | BO macro F1 |
|---|---:|
| Beirut explosion | 0.3757 |
| Hawaii wildfire | 0.3524 |
| Libya flood | 0.3430 |
| Myanmar hurricane | 0.3835 |
| Spain/La Palma volcano | 0.3002 |
| Turkey earthquake5 | 0.3016 |
| Ukraine conflict | 0.3104 |

event macro 约 0.3381，bootstrap CI `[0.3147, 0.3617]`。但分类退化很明显：Spain Damaged 约 0.0034，Libya Damaged 约 0.0115，Turkey Destroyed 为 0，Ukraine Destroyed 约 0.027。

LOEO fold 规模从 3 到 667 张不等，并在 held-out event 本身上选 checkpoint；这不是嵌套、无偏的外部事件估计，只能作为诊断。仅 7 个事件时 event bootstrap 的精度也很有限。

#### E1 event×class-balanced sampler（已实际完成，但文档状态滞后）

| 方案 | BO macro | event macro | worst event | Damaged | Destroyed |
|---|---:|---:|---:|---:|---:|
| E0 reference | 0.358614 | 0.337659 | — | — | — |
| E1 balanced | 0.327085 | 0.289261 | 0.2294 | 0.0704 | 0.1884 |

E1 下降，主要表现为用 Intact/Damaged 换 Destroyed。采样器均匀抽取 21 个 event×class cells；Hawaii 只有 3 张图，却在一个 epoch 中通过三个 cells 被重复抽取约 171 次，构成严重过采样和记忆风险。

当前 `D4_LOEO_CONCLUSION.md` 仍写“下一步继续 E1”，但 E1 已完成且失败；相关 summary 生成时间也早于 E1。这是状态文档未同步，而不是实验尚未运行。

### 4.9 R0 vs R4 BRIGHT 数据补充实验

设计优点：同一 val/test、三种子 42/3407/2026、相同配置家族、R4 仅向 train 添加样本；validation 冻结后才做诊断 test。

R0/R4 使用的是基础 CE 配方，目的是隔离数据变化，并没有采用旧 strict split 上筛出的 E2 binary-aux recipe。这一选择本身合理，但意味着 R0/R4 与 E2 位于不同 split、不同 recipe，二者分数绝不能直接排成“排行榜”；R4 也还没有在无污染 prior 和新验证协议下与最佳已验证方法组合。

#### validation

| 方案 | BO macro mean±sd | damage macro 变化 | 主要类别变化 |
|---|---:|---:|---|
| R0 | 0.3098 ± 0.0035 | — | — |
| R4 | 0.3494 ± 0.0238 | 均值 +0.02285 | Damaged −0.01487，Destroyed +0.06058 |

逐种子 BO 差：`+0.06474, +0.04354, +0.01055`，三次同号；event macro 差约 `+0.0417, +0.0215, −0.0009`；damage macro 差约 `+0.0670, +0.00769, −0.00613`，后者并非三种子一致。

只有 3 对种子时，paired-t 95% CI 约为 `[−0.0282, +0.1074]`；3/3 同号的简单单侧 sign-test `p=0.125`。所以它是积极信号，不是统计确认。

#### 已暴露的 diagnostic test

| 方案 | BO macro mean±sd |
|---|---:|
| R0 | 0.2719 ± 0.0196 |
| R4 | 0.3217 ± 0.0845 |

逐种子差：`−0.0185, +0.0194, +0.1485`；均值提升由 seed 2026 大幅主导。paired-t 95% CI 约 `[−0.1677, +0.2673]`。因此不能写成“R4 已在测试集稳定提升”。

#### 关键混杂

按 batch size 8 和记录的完成 epoch 粗算：

| seed | R0 epochs / 约 steps | R4 epochs / 约 steps | R4/R0 steps |
|---|---:|---:|---:|
| 42 | 40 / 6880 | 45 / 9225 | 1.34× |
| 3407 | 44 / 7568 | 54 / 11070 | 1.46× |
| 2026 | 42 / 7224 | 72 / 14760 | 2.04× |

R4 同时改变了：样本数量、每 epoch 更新数、早停时长、来源数据域、标注转换流程和部分事件×类别频率。缺少以下对照：

- R0 按 optimizer steps 与 R4 严格匹配；
- R0 重采样到与 R4 相同的训练样本访问次数；
- R4 只控制“样本数”、只控制“类别支撑”、只控制“来源域”的消融；
- 新增独立事件，而不只是同事件更多图块。

另外，R4 manifest 使用了绝对路径，如 `/home/yr/code/cvprw26/data/BRIGHT`，没有完全封装进 minimal dataset，跨机器复现仍依赖外部目录。

---

## 5. 当前代码与训练配方中的方法学细节

### 5.1 当前增强

代码中主要是：

- horizontal flip；
- 90° rotation；
- damage-aware crop，概率约 0.4 / 0.4 / 0.2。

未见系统性的 SAR 强度/对数变换、speckle、radiometric normalization、跨事件强度校准，也未见 vertical flip、连续角度仿射、elastic、MixUp/CutMix 等。这里不能直接断言“必须加更多增强”；更重要的漏洞是**尚未形成明确的 SAR 物理/统计归一化审计**。几何增强也不能创造缺失的事件×Damaged/Destroyed 支撑。

### 5.2 SAR 输入

当前数据读取大体将灰度 SAR 读成 uint8 并除以 255。若源 minimal package 已经完成统一预处理，这可能足够；但现有实验报告没有充分记录各事件的原始动态范围、裁剪/归一化方式、传感器差异和直方图漂移。由于模型可能借 SAR 统计识别事件，这一审计是必要的。

### 5.3 负对照

within-event shuffled SAR 是项目目前最有价值的设计之一：它保留事件域和大体传感器分布，破坏逐样本配对。不过还需确认并长期固化：

- permutation 完全可复现并保存；
- 尽可能 deranged，不允许样本映射到自身；
- train/val 各自独立生成；
- 对极小事件明确报告无法完全 derange 的比例；
- 不只比较 BO 总分，还比较 Damaged/Destroyed 和 event-macro paired difference。

### 5.4 event×class 指标的缺失类处理

当前部分 helper 对某事件中不存在的类别跳过计算。这样不同实验/划分的 event-class macro 可能具有不同分母。必须在报告中固定：

- absent-in-GT 类是排除、记 0，还是单独报告 NA；
- present-in-GT 但模型无预测必须记 0；
- 同时给出每个 cell 的 support，避免宏平均掩盖三张图等极小单元。

---

## 6. 漏洞清单：证据、影响与关闭条件

| 优先级 | 漏洞 | 直接证据 | 会造成什么误判 | 如何关闭 |
|---|---|---|---|---|
| P0 | legacy train-test 精确重复 | 105 跨 train-test duplicate groups；Stage-1 test 18.62% 泄漏 | 高估 Stage-1 和 Stage-2 v1 | 永久标记历史受污染；新 event-group 重训全部基线 |
| P0 | Stage-1 prior 非 OOF | latest val 119/290 灾前图被 O1 训练见过 | validation、早停、门控和输入偏乐观 | event-aware Stage-1 重训或 cross-fit prior |
| P0 | test 反复暴露 | 44 个 test metrics；clean strict 至少 20 个模型 | 后续“test 提升”已包含调参反馈 | 现有 test 降级为 historical diagnostic；获取新封存事件或盲测服务 |
| P0 | 事件×类别绑定 | latest test Damaged 98.80% 来自 Mexico | 把事件识别误认为类别识别 | 新增多个独立事件的 Damaged/Destroyed；按 event×class 报告 |
| P1 | R4 与训练预算混杂 | 总 steps 约 1.34×–2.04× | 把更多更新误称数据质量收益 | step-matched、visit-matched 和 source/count 消融 |
| P1 | validation 分布不代表 test | latest val 95.61% Intact，test 54.22% Damaged | 早停和阈值偏向 Intact | 多 event folds；预注册聚合指标；更合理的验证事件覆盖 |
| P1 | R4 没有新增事件组 | 262 张仅来自 Beirut/Hawaii/Libya | 同事件拟合改善被当跨事件泛化 | 优先收集全新 canonical events |
| P1 | 多轮单 val 搜索 | 架构、objective、threshold 在同一 val 反复选择 | winner's curse / 多重比较 | nested group CV 或独立 development folds；预注册最终规则 |
| P1 | D2 控制不纯 | GT building holes + damage-aware crop | 高估纯背景 shortcut | 不使用 GT mask且 target-independent crop 的新控制 |
| P1 | LOEO checkpoint 选择不嵌套 | held-out event 同时用于选 epoch | LOEO 估计偏乐观 | 内层按训练事件选 epoch，外层事件只评一次 |
| P1 | R4 test 高方差 | 三种子差一负两正，seed2026 主导 | 均值掩盖不稳定性 | 更多预注册种子；事件层/样本层层级 bootstrap |
| P2 | geospatial independence 未完全证明 | 230/580 val/test 无 georef | 可能仍有近邻地块/同场景相关性 | 补坐标元数据；按 footprint/scene 切分与审计 |
| P2 | SAR 归一化审计不足 | 主要为 uint8/255，缺跨事件统计报告 | 利用传感器/事件强度捷径 | 每事件分布审计；预处理消融与固定规范 |
| P2 | 指标 estimand 不统一 | pooled P2−P1 与 sample bootstrap 方向不一致 | 同名分数表达不同问题 | 固定 pooled/sample/event 三套名称和公式 |
| P2 | 小 cell 极端过采样 | Hawaii 3 张约被抽 171 次/epoch | 记忆、方差增大 | capped weights、effective-number、batch-level event coverage |
| P2 | 状态文档滞后 | 多份文档仍写“未训练/下一步 E1/R4 blocked” | 后续模型基于错误项目状态决策 | 单一实验 registry，旧报告标 superseded |
| P2 | R4 路径不可移植 | manifest 指向 BRIGHT 绝对路径 | 别处无法复现 | 将许可允许的派生最小包、哈希和相对路径封装 |

---

## 7. 文档与结果状态冲突

以下内容在历史上是正确的，但截至本快照已过时：

- `PROJECT_CONTEXT.md`、`stage2-v2.md`、`DATA_INTEGRITY_CORRECTION_20260622.md` 仍有“strict fresh A1–A4 尚未开始”的表述；实际上 A1–A4 已完成。
- `event_group_dataset_v1_20260803/DATASET_BUILD_REPORT.md` 和部分 `READINESS.json` 仍写 R4 缺 prior、`training_eligible=false` 或尚未训练；之后 prior 已生成，R0/R4 六个正式运行已完成。
- `event_shortcut_v1_20260803/D4_LOEO_CONCLUSION.md` 仍把 E1 写成下一步；E1 实际已运行并失败。
- 部分 summary 文件生成于 E1 之前，不能代表最终 shortcut 实验状态。
- 根目录 `实验改进A.md` 是 OGSR/P1–P5 等阶段的设计提案，其中多项后来已经执行，不能当作当前待办状态。
- 根目录 `数据集更新.md` 是 1207/357/388 clean strict 数据快照；内容仍有审计价值，但最新 R0/R4 已改用 1372/290/290 event-group 划分。
- 根目录 `重复审计.md` 主要是 relaxed-overlap 工作流设计；最终人工复核结果应以 `HUMAN_REVIEWED_CLEAN_DATASET_20260624.md` 及对应 manifest 为准。

建议建立一个唯一的 `EXPERIMENT_REGISTRY`，每个实验至少记录：

```text
experiment_id
question / hypothesis
data_manifest + sha256
upstream_prior_model + train-split provenance
config + resolved config
code commit / dirty patch hash
seed
selection split and checkpoint rule
test exposure status
primary/secondary metrics
gate result
status: planned/running/completed/failed/superseded/retracted
supersedes / superseded_by
```

不要删除旧报告；应在旧文件顶部加 `SUPERSEDED/RETRACTED` 及指向当前 registry 的链接。

---

## 8. 哪些结论现在仍然可辩护

### 8.1 有充分证据支持

1. legacy 拆分存在严重精确内容泄漏，旧测试结果必须撤回正式地位。
2. clean strict 和最新 event-group manifest 在已审计的内容哈希与 canonical event identity 层面是 split-disjoint 的。
3. 事件/背景上下文本身具有较强预测力，模型存在学习 shortcut 的现实风险。
4. Damaged/Destroyed 的跨事件支撑不足且高度不均衡，是当前最主要的数据问题之一。
5. 当前统一配方下 UABCD、SSFCNet、FSGNet 没有通过预设完整硬门，因此按纪律停止是正确的。
6. R4 在历史 validation 的三个种子上 BO 总分均高于 R0，值得做更严格的确认实验。

### 8.2 仅有提示性证据

1. paired SAR 相对 within-event shuffled SAR 可能提供逐样本信息；但它是否稳定对应 Damaged/Destroyed 仍未证明。
2. pre-event RGB 可能帮助变化判别；现有强结果受单种子与 test 暴露限制。
3. binary auxiliary objective 可能优于原 P2；现有证据只有一个被多轮筛选后的 validation seed。
4. BRIGHT 同事件补充可能改善 Destroyed；Damaged 并未同步改善。

### 8.3 当前不应使用的表述

- “Stage-1 IoU 已可靠达到 0.6871”；
- “Stage-2 v1 BO macro 0.4120 是当前正式基线”；
- “A3 证明 SAR 能跨事件识别损伤等级”；
- “adaptive threshold 0.3794 已泛化”；
- “R4 正式 test 提升 0.0498”；
- “event-balanced sampling 无效”（当前实现过采样过强，只能说这个实现失败）；
- “外部论文模型不适合本任务”（只能说统一 screening 配方失败）。

---

## 9. 建议的下一步工作：按依赖顺序执行

### 9.1 P0：先修评估链，不训练新架构

1. 冻结一个完整可 checkout 的代码提交，保存当前 dirty patch 和所有 manifest/report 哈希。
2. 建立 experiment registry 和 test-exposure ledger；将 legacy、strict test、event-group test 明确标记为 `historical_diagnostic_only`。
3. 为最终结论准备新的、尚未用于任何选择的独立事件盲测集。最理想的是由独立脚本/人员保管标签，只返回一次预注册指标；若短期无法获得，只能用 nested event CV 做开发，不能制造一个“新的 test”名义来掩盖旧数据已被观察的事实。
4. 固定指标定义：pooled-pixel、sample-macro、event-macro、event×class support 和 hierarchical CI。

### 9.2 P0：重建 Stage-1 → Stage-2 无污染 prior

建议两个可选协议：

#### 协议 A：event-group frozen Stage-1

- Stage-1 只使用 Stage-2 train event groups 的光学建筑标签训练；
- 在 Stage-2 val/test event groups 上直接推理；
- Stage-2 全部实验复用同一冻结 prior；
- 不允许通过 Stage-2 val 调 Stage-1。

#### 协议 B：cross-fitted prior

- 在 development events 上做 group K-fold；
- 每张 Stage-2 train/val 图的 prior 都由未见过该事件/图像的 Stage-1 fold 产生；
- 最终盲测 prior 由只在全部 development events 上训练的 Stage-1 产生。

必须同时报告 prior 在各 event 的 building IoU/F1，而不是只报全局均值。

### 9.3 P1：用最小矩阵拆解 R4 收益

保持同一模型、同一 OOF prior、同一预注册 event-CV，只跑最能区分解释的四组：

| 组 | 数据 | 总 optimizer steps | 每样本期望访问数 | 回答的问题 |
|---|---|---:|---:|---|
| C0 | R0 | 与 R4 匹配 | 自然/记录 | 单纯多训练是否足够 |
| C1 | R0 重采样 | 与 R4 匹配 | 与 R4 近似 | 样本访问次数效应 |
| C2 | R4 | 与 R0 匹配 | 受控 | 新数据在同预算下是否有效 |
| C3 | R4 | 原 recipe | 原 recipe | 复现已有信号 |

若资源允许，再拆 BRIGHT 的 Beirut/Hawaii/Libya 三个来源做 leave-one-source-out，而不是立刻加新网络。主要判据必须要求：

- BO grade macro 改善；
- damage macro 不下降；
- Damaged 和 Destroyed 至少不能靠一类大幅牺牲另一类；
- event-macro 和 worst-event 同向；
- paired vs shuffled 的差值在 event 层置信区间上有支持。

### 9.4 P1：数据采集优先级

不要按“图片总数最多”采数据，应按缺失的 `event × class` cell 采集：

1. 新的 canonical events，优先于已有事件更多 tile；
2. 每个新事件同时有 Intact、Damaged、Destroyed，尤其是当前最弱的 Damaged；
3. 限制单一事件对某类像素的最大占比；
4. 保存 scene/footprint/sensor/acquisition 元数据，使地理独立和传感器域可审计；
5. 在纳入训练前做 exact hash、near-duplicate、footprint overlap 和标注协议一致性检查。

一个可操作的入库门是：任何正式 validation/test 类别的最大单事件像素占比不超过预注册阈值（例如 50%），且每个 damage 类至少由若干独立事件提供有效支持。具体阈值应在查看新盲测结果前确定。

### 9.5 P2：之后才考虑方法改进

只有在评估链和数据对照修复后，才值得重新确认：

- E2 binary auxiliary 是否在多个 event folds 上稳定；
- pre RGB + paired SAR 是否在 OOF prior 下稳定优于 pre-only / shuffled SAR；
- SAR 事件级 normalization 或域泛化是否减少 shortcut；
- capped event×class sampler、effective-number weighting 是否优于当前极端均匀 cell sampler；
- 阈值/校准应在内层 folds 学习，再固定到外层事件。

---

## 10. 推荐的正式验证协议草案

```text
Development unit: canonical event / geospatial scene, never random tile

Outer loop:
  leave one or more unseen events out for evaluation

Inner loop:
  select epoch, objective and any threshold only on remaining event groups

Stage-1:
  train without outer event
  produce true OOF/frozen priors

Stage-2 controls:
  prior-only
  pre-only
  paired SAR
  within-event deranged SAR
  metadata/event-prior rule
  target-independent background control

Reporting:
  pooled-pixel BO macro
  sample-macro
  event-macro and worst event
  every event × class cell with support
  hierarchical bootstrap over events then samples
  paired seed differences

Final:
  lock model/config/threshold/seed ensemble
  evaluate once on a newly sealed external-event set
```

注意：若 outer event 数量仍只有 3–7 个，任何置信区间都会很宽。这不是统计方法的问题，而是独立实验单位不足；增加同事件 tile 不能替代增加事件。

---

## 11. 给后续大模型的分析任务

可以把本文件全文作为上下文，并附上下面的要求：

> 你是一名负责遥感跨模态语义分割、因果实验设计和统计验证的研究负责人。请不要只提出“换模型、加增强、调学习率”之类泛化建议。<br>
> 1. 先判断本文每个结论的证据等级，指出是否有任何过度推断；<br>
> 2. 画出 Stage-1 prior、事件身份、pre RGB、SAR、标签和 selection/test exposure 的因果图，找出泄漏与 shortcut 路径；<br>
> 3. 评估本文提出的 OOF prior、nested event CV、R0/R4 matched-budget 对照是否足以区分因果解释；<br>
> 4. 在最多 4 个训练条件、3 个种子的预算下，给出信息增益最大的实验矩阵和明确停止条件；<br>
> 5. 给出 event×class 数据采样目标，说明怎样避免 Mexico/Damaged 或 Marshall/Destroyed 式单事件垄断；<br>
> 6. 指定 pooled/sample/event 三层指标和层级 bootstrap 的实现方案；<br>
> 7. 明确哪些现有 checkpoint/结果仍可复用做诊断，哪些必须重训，哪些数字必须永久撤回；<br>
> 8. 最后输出一份按 P0/P1/P2 排序、包含依赖关系和验收条件的下一阶段计划。

建议进一步追问大模型：

1. 在新事件数量有限的情况下，GroupKFold、LOEO、leave-p-events-out 哪种估计偏差/方差最合适？
2. 如何设计 paired/shuffled SAR，使其只破坏逐样本变化关系而尽可能保留事件、传感器和边缘统计？
3. 如何区分 R4 的收益来自更多样本、更多 Destroyed、更多更新步数，还是 BRIGHT 标注域？
4. OOF Stage-1 prior 应使用硬 mask、概率图还是带校准不确定性的多通道输入？
5. 当某 event×class cell 没有 GT support 时，宏平均分母应如何定义并公开？

---

## 12. 关键证据索引

项目总览与执行约束：

- `PROJECT_CONTEXT.md`
- `stage2-v2.md`
- `SERVER_RECOVERY.md`
- `stage1_optical_building/PROJECT_EXPERIENCE.md`

数据完整性与严格数据集：

- `stage1_optical_building/reports/stage2_v2/DATA_INTEGRITY_CORRECTION_20260622.md`
- `stage1_optical_building/reports/stage2_v2/HUMAN_REVIEWED_CLEAN_DATASET_20260624.md`
- `stage1_optical_building/manifests/stage2_v2_strict_v1/deduplication_groups.json`
- `stage1_optical_building/manifests/stage2_v2_clean_human_reviewed_20260624/`
- `datasets/DisasterM3_aria2_building_damage_practice/splits/stage1_optical_building/building_extract_train_sar_qc_freq.jsonl`
- `stage1_optical_building/outputs/O1_unet_resnet34_freq/test_all/sample_metrics.csv`

strict 结果与后续探索：

- `stage1_optical_building/reports/stage2_v2/stage2_v2_experiment_report.md`
- `stage1_optical_building/reports/stage2_v2/ORACLE_PRIOR_EXPLORATION_RESULTS_20260625.md`
- `stage1_optical_building/reports/stage2_v2/PRECHANGE_EXPERIMENT_RESULTS_20260626.md`
- `stage1_optical_building/reports/stage2_v2/PHASE5_THRESHOLD_ADAPTIVE_REPORT_20260630.md`

统一外部模型协议：

- `stage1_optical_building/reports/stage2_v2/UNIFIED_EXPERIMENT_STANDARD_V1_20260731.md`
- `stage1_optical_building/reports/stage2_v2/UABCD_UNIFIED_V1_RESULTS_20260731.md`
- `stage1_optical_building/reports/stage2_v2/SSFCNET_UNIFIED_V1_EXPERIMENT_20260731.md`
- `stage1_optical_building/reports/stage2_v2/FSGNET_UNIFIED_V1_EXPERIMENT_20260731.md`

事件捷径与最新数据：

- `stage1_optical_building/reports/stage2_v2/event_class_distribution_20260803/`
- `stage1_optical_building/reports/stage2_v2/event_shortcut_v1_20260803/`
- `stage1_optical_building/reports/stage2_v2/event_group_dataset_v1_20260803/DATASET_BUILD_REPORT.md`
- `stage1_optical_building/reports/stage2_v2/event_group_dataset_v1_20260803/R0_R4_PREDICTED_PRIOR_COMPARISON_20260804.md`
- `stage1_optical_building/reports/stage2_v2/event_group_dataset_v1_20260803/footprint_audit.json`

主要配置：

- `stage1_optical_building/configs/stage2_v2_strict_v1/`
- `stage1_optical_building/configs/stage2_v2_pred_prechange_phase1/`
- `stage1_optical_building/configs/stage2_v2_phase3_objective/`
- `stage1_optical_building/configs/stage2_unified_v1/`
- `stage1_optical_building/configs/stage2_event_shortcut_v1/`
- `stage1_optical_building/configs/stage2_event_group_r0_r4_v1/`

---

## 13. 最终决策摘要

当前项目不是“模型还不够复杂”这么简单。最核心的限制是独立实验单位不足、事件与损伤类别绑定、上游 prior 非 OOF、测试集已经反复暴露，以及数据补充实验缺少预算/来源匹配对照。

因此下一阶段的成功标准不应是再得到一个更高的单次 validation BO，而应是：

1. Stage-1 prior 对每个 development/holdout event 真正 out-of-fold；
2. paired SAR 相对 prior-only 和 within-event shuffled SAR 的优势同时出现在 damage macro、Damaged、Destroyed 和 event-macro，而不是只在 pooled Intact 主导的 BO 上；
3. 数据增益在 step-matched、visit-matched 对照下仍成立；
4. 多个独立事件上的方向一致，置信区间按事件层计算；
5. 所有选择冻结后，只在新的封存外部事件上评估一次。

做到这五点之后，项目才具备对“跨事件建筑损伤分级能力”作正式论证的基础。
