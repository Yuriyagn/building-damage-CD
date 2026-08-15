# Building Damage Change Detection

跨事件光学–SAR 建筑损伤变化检测研究代码。目标是输出像素级四分类语义图：

```text
background / intact / damaged / destroyed
```

项目不是只追求一个更高的 pooled 分数，而是判断模型是否真的利用灾前/灾后变化信息，在未见事件上稳定区分 `damaged` 与 `destroyed`，而不是依赖建筑轮廓、事件身份、灾种先验或类别比例捷径。

> 状态快照：2026-08-15。旧 test 已降级为历史诊断；当前最新正式开发结果是
> [Metadata-aware multi-task v1](reports/stage2_v2/metadata_multitask_v1_20260814/RESULTS.md)。
> 更完整的数据谱系和历史审计见
> [项目全景审计](docs/PROJECT_AUDIT_20260814.md)。

## 1. 任务与模型输入

历史主线采用两阶段设计：

```text
Stage-1: pre-event optical RGB -> binary building prior
Stage-2: visual inputs + building support -> intact / damaged / destroyed
Final: background / intact / damaged / destroyed
```

Stage-2 并不是固定一种输入，仓库中已有几类受控实验：

| 研究方面 | 输入/监督 | 回答的问题 |
|---|---|---|
| 建筑支撑 | SAR、oracle/predicted prior | 建筑定位先验能解释多少分数 |
| SAR 可识别性 | paired SAR、within-event shuffled SAR、prior-only | 模型是否使用样本对应的 SAR，而不是事件纹理 |
| 灾前变化信息 | pre-RGB、post-SAR、prior 的组合 | 灾前信息是否帮助损伤分级 |
| 数据补充 | R0 与 R4（R4 增加 262 张 BRIGHT 训练图） | 更多同事件数据是否改善验证表现 |
| 架构迁移 | E2/U-Net、UABCD、SSFCNet、FSG-Net | 更复杂变化检测架构是否优于本项目基线 |
| 事件捷径 | leave-one-event-out、event×class sampler | 总分是否掩盖单事件/单类别崩溃 |
| Metadata 多任务 | RGB+SAR damage-only 与 `+ disaster loss` | 灾害类型辅助监督是否提升跨事件泛化 |

最新 Metadata 实验有意使用严格四通道 `[pre-RGB, post-SAR]`，**不把 prior 或灾种标签送入 forward 输入**。灾种只作为辅助监督；prior 只用于次要端到端门控评价。因此它与历史 `SAR + predicted prior` 实验不是只差一个分类 head，不能直接排成同一排行榜。

## 2. 指标与比较规则

正式模型选择以 GT building pixels 内的三等级宏平均为主：

```text
BO grade macro F1 = mean(F1_intact, F1_damaged, F1_destroyed)
BO damage macro F1 = mean(F1_damaged, F1_destroyed)
```

同时报告：

- `Damaged F1`、`Destroyed F1`、binary damage F1；
- pooled-pixel、event-macro、worst-event 和 event×class；
- predicted-prior gate 下的端到端四分类指标；
- paired−shuffled、paired seed 差值及层级 bootstrap。

不同 split、输入、训练预算、seed 数和 checkpoint 规则的实验只用于机制对照，**不得按分数高低直接排序**。背景主导的 accuracy/mIoU 不能替代 BO 指标；oracle support 只是建筑内分级诊断，不是部署性能。

## 3. 实验总览：用了什么，差异在哪里

| 实验家族 | 数据与 seeds | 模型/输入 | 唯一主要变化 | 当前结论 |
|---|---|---|---|---|
| Stage-1 O1–O4 | legacy split，历史 test | RGB building segmentation | U-Net/DeepLab/SegFormer、采样范围 | 历史 O1 最好，但 test 含精确重复，只保留历史记录 |
| Stage-2 v1 S0–S3 | legacy split，历史 test | SAR、oracle/predicted prior | 输入模态消融 | prior 很强；旧 test 受污染，不能作正式基线 |
| strict A1–A4 | clean 1207/357/388，3 seeds | U-Net ResNet34 | prior-only / SAR-only / SAR+prior / shuffled-SAR+prior | paired SAR 有信号，但 damage 等级与 test 泛化不稳定 |
| P1–P5 / E1–E3 | strict validation，seed 42 | pre-RGB、SAR、predicted prior | 输入组合、focal/binary auxiliary objective | E2 是单 val 多轮筛选后的 reference，不是三种子确认结果 |
| UABCD/SSFCNet/FSG-Net | strict train/val，seed 42 | 外部变化检测架构迁移 | 架构；paired 与固定 shuffled 对照 | 三者均未通过完整晋级门槛；test 未使用 |
| D4 / balanced E1 | strict development events，seed 42 | E0 recipe | LOEO 或 event×class 均匀采样 | 揭示类别崩溃；极端均匀采样反而降分 |
| R0 vs R4 | event-group 1372/290 与 1634/290，3 seeds | SAR + frozen O1 predicted prior | R4 只增加 262 张同事件 BRIGHT train 样本 | validation 有正信号，主要来自 Destroyed；预算/来源域混杂未解除 |
| Metadata M0 vs M1 | R4 1634/290，3 seeds | 四通道 pre-RGB + post-SAR | M1 增加 `0.1 × disaster CE`；结构与初始化相同 | damage 均值提高，但灾种分类低于 majority；分支停止，不运行 M2 |

## 4. 同协议内的实验结果

### 4.0 Legacy 历史结果：只用于追溯

Stage-1 四个历史模型均为 seed 42，并在旧 test-all 上评价：

| 模型 | 架构/训练范围 | Building IoU | Building F1 |
|---|---|---:|---:|
| O1 | U-Net ResNet34，frequent disasters | **0.6871** | **0.8145** |
| O2 | U-Net ResNet34，all disasters | 0.6765 | 0.8070 |
| O3 | DeepLabV3+ ResNet50，frequent | 0.6432 | 0.7829 |
| O4 | SegFormer-B0，frequent | 0.6793 | 0.8091 |

O1 因 IoU 最高成为 predicted-prior 生成器，但其 564 张 test 中有 105 张与 train 精确重复。去掉这些样本后，事后诊断 IoU/F1 从 `0.6871/0.8145` 降到 `0.6603/0.7954`；这仍不是重新设计后的正式测试。

Stage-2 v1 历史输入消融：

| 模型 | 输入 | BO macro | Damage macro | Damaged | Destroyed |
|---|---|---:|---:|---:|---:|
| S0 | SAR，无 prior | 0.3314 | 0.3233 | 0.2114 | 0.4352 |
| S1 | SAR + oracle prior | 0.3927 | 0.3028 | 0.2152 | 0.3903 |
| S2 | SAR + predicted prior | **0.4120** | **0.3376** | 0.2127 | **0.4626** |
| S3-o | oracle prior only | 0.3592 | 0.2684 | 0.3311 | 0.2056 |
| S3-p | predicted prior only | 0.3176 | 0.2800 | **0.4142** | 0.1458 |

S2 的历史总分最高，但 S3 prior-only 已能取得较高 Damaged，说明输入支撑和数据先验可以制造非零损伤分数。整个 v1 使用含 105 个 train-test 精确重复组的旧拆分，上表不能作为正式基线，也不能和下面 clean/event-group validation 分数直接比较。

### 4.1 strict A1–A4：模态与负对照

使用 clean strict split、U-Net ResNet34、building-only CE 和三种子；表中是 validation 均值。

| ID | 输入 | BO grade macro F1 | BO damage macro F1 | 与 A3 的关键差异 |
|---|---|---:|---:|---|
| A1 | predicted prior only | 0.33258 | 0.12904 | 不使用 SAR |
| A2 | paired SAR only | 0.35336 | 0.12916 | 不使用 prior |
| A3 | paired SAR + predicted prior | **0.35398** | **0.13049** | 完整历史主线输入 |
| A4 | within-event shuffled SAR + predicted prior | 0.30278 | 0.11122 | 只破坏样本级 SAR 对应关系 |

差值解释：

- A3−A4：BO `+0.05120`、damage `+0.01927`，说明真实配对 SAR 包含可用信息。
- A3−A2：BO `+0.00062`、damage `+0.00133`，predicted prior 几乎没有带来建筑内等级增益。
- A3−A1：BO `+0.02140`，但 damage 只 `+0.00144`；总分改善主要不是两种损伤等级共同提升。
- A3 在第一次冻结 strict test 上反而低于 A1；该 test 后续又被多轮探索暴露，因此只能作为历史诊断，不能继续用于新模型确认。

### 4.2 外部架构迁移：只比较 seed-42 strict validation

所有方法使用同一 strict train/val、相同优化框架和固定 within-event shuffled SAR。E2 是本项目 reference；外部架构只在通过 `paired >= E2` 且 `paired−shuffled >= 0.01` 后才允许扩展种子。

| 方法 | BO grade macro | Damage macro | Damaged | Destroyed | Paired−shuffled BO | 判定 |
|---|---:|---:|---:|---:|---:|---|
| E2 reference | **0.358614** | **0.112705** | 0.126116 | **0.099295** | — | reference |
| UABCD paired | 0.311873 | 0.096698 | **0.169006** | 0.024391 | -0.009273 | 两项门槛均失败 |
| SSFCNet paired | 0.329494 | 0.056961 | 0.020642 | 0.093281 | -0.010996 | 两项门槛均失败 |
| FSG-Net paired | 0.334827 | 0.072302 | 0.098075 | 0.046530 | **+0.011671** | SAR 对照通过，性能门槛失败 |

涨跌原因：

- UABCD 提高了 Damaged，但 Destroyed 降到 `0.02439`，相邻损伤等级发生取舍。
- SSFCNet 的 intact F1 很高，但 Damaged 几乎失效；总分主要由 intact 支撑。
- FSG-Net 是唯一稳定通过 SAR 负对照的迁移架构，5/5 validation events 上 paired 都高于 shuffled，但 damage macro 和 E2 仍有明显差距。
- 结论只针对本项目迁移 recipe，不否定原论文在原数据和完整训练设置下的结果。

### 4.3 事件捷径：D4 与 event×class-balanced E1

D4 在 strict train 的 7 个事件上做 leave-one-event-out：event macro BO 为 `0.3381`，与 E0 的 `0.3377` 几乎相同，但多个事件出现单类别近乎完全失效，例如 Spain/La Palma 的 Damaged F1 `0.0034`、Turkey 的 Destroyed F1 `0.0000`。这说明总体 event macro 会被 intact 或另一个损伤等级补偿。

| 方法 | BO macro | Event macro | Worst event | Damaged | Destroyed |
|---|---:|---:|---:|---:|---:|
| E0 reference | 0.358614 | 0.337659 | — | — | — |
| E1 event×class balanced | 0.327085 | 0.289261 | 0.2294 | 0.0704 | 0.1884 |

E1 丢分的直接原因是对极小 event×class cell 的极端重复采样：Hawaii 只有 3 张图，却约被访问 171 次/epoch。均匀 cell sampling 放大了记忆和方差，并没有得到事件不变表示。

### 4.4 R0 vs R4：增加 BRIGHT 训练样本

R0/R4 共用相同 validation，均使用 frozen O1 predicted prior、基础 CE、相同增强与三个种子。R4 只在 train 增加 262 张 BRIGHT 图像，但没有新增 canonical event。

| Validation metric | R0 | R4 | R4−R0 |
|---|---:|---:|---:|
| BO grade macro F1 | 0.3098 ± 0.0035 | **0.3494 ± 0.0238** | +0.0396 |
| BO damage macro F1 | 0.0573 ± 0.0142 | **0.0801 ± 0.0247** | +0.0229 |
| Damaged F1 | **0.0726 ± 0.0268** | 0.0578 ± 0.0376 | **-0.0149** |
| Destroyed F1 | 0.0419 ± 0.0212 | **0.1025 ± 0.0647** | +0.0606 |
| Event-macro BO F1 | 0.3183 | **0.3390** | +0.0208 |
| Worst-event BO F1 | 0.2418 | **0.3006** | +0.0588 |

R4 在三个种子的 pooled 主指标上均为正，但 event-macro 只有 2/3 种子为正。涨分主要来自 Destroyed，Damaged 反而下降。R4 还同时增加了每 epoch batch 数和实际总 optimizer steps，三个种子的总更新量约为 R0 的 `1.34× / 1.46× / 2.04×`，并引入 BRIGHT 来源域与标注流程变化。因此结果只能说明“R4 数据包存在值得确认的正信号”，不能把全部增益归因于新增样本质量。

### 4.5 Metadata-aware multi-task：M0 vs M1

这是当前最新、validation-only 的三种子实验。M0/M1 使用完全相同的 ResNet34 U-Net、多任务 head、初始化、batch 顺序和优化器；输入均为 `[pre-RGB, post-SAR]`，不把 prior、disaster type、地点或传感器 metadata 作为模型输入。

| Condition | Damage supervision | Disaster supervision | BO grade macro | BO damage macro | Event macro BO | Worst-event BO | Disaster macro |
|---|---|---|---:|---:|---:|---:|---:|
| M0 | building-only CE | `lambda=0`，head 不训练 | 0.41676 ± 0.01187 | 0.15870 ± 0.02307 | 0.33783 ± 0.01445 | 0.24155 ± 0.07061 | 不解释 |
| M1 | 同 M0 | `0.1 ×` weighted disaster CE | **0.44661 ± 0.01607** | **0.19809 ± 0.02423** | **0.36023 ± 0.03239** | **0.26418 ± 0.09103** | 0.23918 ± 0.04311 |

成对主差值：

| seed | Δ BO damage macro | Δ event macro BO | Δ worst-event BO | M1 disaster macro |
|---:|---:|---:|---:|---:|
| 42 | +0.05877 | +0.04527 | +0.14283 | 0.28843 |
| 3407 | +0.06091 | +0.03319 | -0.00397 | 0.20829 |
| 2026 | -0.00150 | -0.01128 | -0.07094 | 0.22081 |
| mean | **+0.03939** | **+0.02239** | **+0.02264** | **0.23918** |

为什么有涨分：

- Destroyed F1 平均约提高 `+0.07263`，Marshall wildfire 的 event damage F1 平均提高 `+0.06189`。
- 两个辅助目标可能改变共享特征和优化轨迹，形成一般正则化效应。

为什么仍判定失败：

- Damaged F1 只提高约 `+0.00616`，M1 绝对值仍只有 `0.04892`。
- Haiti/Morocco earthquake 的 event damage F1 仍约为 `0.01`。
- seed2026 的 event macro 和 worst-event 明显下降。
- 三个 M1 damage checkpoint 上 train disaster macro 约 `0.999`，validation 只有 `0.208–0.288`；平均 `0.23918` 低于 majority baseline `0.28`，远低于预注册要求 `0.38`。
- 训练集中每个灾种只对应一个事件，分类 head 可以学习事件/场景身份，而不是可迁移灾害机制。
- 层级 bootstrap 的 damage 与 event 差值 95% 区间都跨 0。

因此 M1 没有通过“灾害语义泛化”门槛，M2 固定乱序标签对照按协议不启动。当前只能把增益归为辅助正则化/优化信号，不能表述为“加入灾害知识提升泛化”。

## 5. 为什么会涨分或丢分：跨实验归纳

| 现象 | 证据 | 更可能的原因 |
|---|---|---|
| 总 BO 提升但 Damaged 不升 | A3、R4、M1 都主要提高 Destroyed 或 intact | 类别/事件支持不均，总指标被较易类别补偿 |
| paired 有时优于 shuffled | strict A3−A4、FSG-Net 5/5 events | SAR 中存在样本级对应信号，但不等于稳定损伤等级语义 |
| 更复杂架构未超过 E2 | UABCD/SSFCNet/FSG-Net 均失败 | 数据可识别性和事件偏差是瓶颈，不只是容量不足 |
| 均匀 event×class sampling 丢分 | E1 BO 0.3271 < E0 0.3586 | 极小 cells 被过度重复，记忆和梯度方差上升 |
| R4 涨分且方差变大 | Destroyed +0.0606，Damaged -0.0149 | 新样本覆盖、更多更新步数和来源域同时变化 |
| Disaster auxiliary 提升 damage 但分类失败 | train disaster≈0.999，val≈0.239 | 灾种与训练事件一一绑定，辅助头学习事件捷径 |
| 不同种子损伤等级互换 | A3、R4、M1 的 Damaged/Destroyed 差异 | 少事件、模糊等级边界和 checkpoint 峰值选择共同作用 |

## 6. 当前项目问题审查

### P0：会直接改变科学结论

1. **旧数据存在精确泄漏。** legacy 2613 行有 661 个 full-sample 重复组，其中 105 个跨 train-test；旧 Stage-1/Stage-2 test 只能保留为历史记录。
2. **没有新的盲测集。** clean strict test 和 event-group test 都已被多轮评估暴露，后续只能标为 `historical_diagnostic_only`。
3. **Predicted prior 不是 out-of-fold。** 最新 event-group validation 有 119/290 张 pre-image 曾参与 Stage-1 O1 训练，导致上游先验参与下游模型选择时存在验证污染。
4. **事件与类别/灾种绑定。** event-group validation 95.61% building pixels 为 Intact；训练集中每个灾种只对应一个事件；test 的 Damaged 又几乎由单一 Mexico Hurricane 提供。

### P1：会让涨分归因不成立

1. **R0/R4 训练预算不匹配。** 样本数、总 steps、来源域、标注流程和类别支撑同时改变。
2. **validation 事件太少。** Metadata validation 只有 4 个事件，bootstrap 区间宽；均值容易由 Marshall 或单个种子主导。
3. **同一 validation 被多轮搜索。** objective、架构、阈值和 sampler 都曾在同一开发集上筛选，存在 winner's curse。
4. **LOEO checkpoint 不完全嵌套。** held-out event 同时参与选 epoch，LOEO 结果可能偏乐观。
5. **Metadata 语义不可识别。** 灾种与事件身份没有解耦，当前实验无法区分灾种知识、事件识别和一般 auxiliary regularization。

### P2：工程与报告一致性问题

1. **Checkpoint 与 early stopping 双轨。** 当前代码用 3-epoch 平滑 BO 更新 patience，却按 raw BO 保存 checkpoint；例如 M0/seed3407 在 epoch 82 停止但正式 checkpoint 是 epoch 14。协议虽一致，但增加峰值选择和运行时敏感性。
2. **指标 estimand 尚未完全统一。** pooled-pixel、sample-macro 和 event-macro 曾出现方向不一致，报告必须显式命名。
3. **地理独立性未完全证明。** 部分样本缺少 georeference；内容哈希和图像近重复审查不能证明 footprint 独立。
4. **SAR 预处理审计不足。** 主要按 uint8/255 使用，缺少跨事件、传感器和强度分布的系统报告。
5. **R4 路径不可完全移植。** 部分 manifest 仍依赖外部 BRIGHT 绝对路径。
6. **历史状态文档会过时。** 部分旧文档仍写“strict A1–A4 未开始”或“E1 是下一步”；README 和冻结结果报告应作为当前入口，旧文档只保留时间点语义。

## 7. 下一步优先级

1. 重建 Stage-1 out-of-fold prior，保证 Stage-2 train/val 的 prior 都来自未见该图像/事件的 Stage-1 fold。
2. 建立新的封存外部事件或盲测服务；现有 test 不再用于新的最终确认。
3. 做 R0/R4 step-matched、visit-matched、source/count 分离实验，拆开“更多训练”和“更好数据”。
4. 在 metadata 方向，先为每个灾种补充多个训练事件，再做灾种内 leave-one-event-out；否则不继续调 classification loss。
5. 优先验证样本级可核验的视觉辅助任务：building existence、changed/unchanged；并与等参数乱序/随机辅助任务对照。
6. 统一 checkpoint/early-stopping 定义和 pooled/sample/event 指标命名，再启动下一轮正式训练。

## 8. 仓库结构与复现

```text
configs/      冻结实验配置和 event-group 映射
docs/         GitHub 入口文档、完整项目审计和文档索引
manifests/    已审计的小型/历史 manifest；大型生成 manifest 默认忽略
reports/      数据完整性、实验结论和统一外部迁移报告
scripts/      数据构建、审计、训练启动、评估和汇总脚本
src/          Stage-1 / Stage-2 数据、模型、损失和指标实现
tests/        单元与协议回归测试
tools/        报告导出和辅助工具
instruction.sh 统一命令入口
```

数据集、权重、训练输出、日志、缓存、生成预测和隔离第三方源码不进入 Git。

### 环境与数据

```bash
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate sam3

export STAGE2_DATA_ROOT=/path/to/DisasterM3_optical_sar_damage_minimal_v0.2
export PYTHON_BIN="$(command -v python)"
```

Stage-2 minimal package 应包含 pre optical、post SAR、四分类 mask、oracle building mask 和 Stage-1 prior。数据不随仓库发布。

### 快速检查

```bash
python -m unittest discover -s tests -q
bash instruction.sh stage2_v2_strict_check_runtime
```

正式 GPU 训练前必须检查 manifest、数据审计、`nvidia-smi`、tmux、输出目录和随机种子。长任务在 tmux 中运行并写入唯一日志。

## 9. 关键文档

- [完整项目审计与下一步建议](docs/PROJECT_AUDIT_20260814.md)
- [Metadata-aware multi-task 正式结果](reports/stage2_v2/metadata_multitask_v1_20260814/RESULTS.md)
- [R0/R4 BRIGHT 数据补充比较](reports/stage2_v2/event_group_dataset_v1_20260803/R0_R4_PREDICTED_PRIOR_COMPARISON_20260804.md)
- [D4 LOEO 事件捷径诊断](reports/stage2_v2/event_shortcut_v1_20260803/D4_LOEO_CONCLUSION.md)
- [UABCD 统一迁移](reports/stage2_v2/UABCD_UNIFIED_V1_RESULTS_20260731.md)
- [SSFCNet 统一迁移](reports/stage2_v2/SSFCNET_UNIFIED_V1_EXPERIMENT_20260731.md)
- [FSG-Net 统一迁移](reports/stage2_v2/FSGNET_UNIFIED_V1_EXPERIMENT_20260731.md)
- [数据完整性更正](reports/stage2_v2/DATA_INTEGRITY_CORRECTION_20260622.md)
- [人工复核 clean 数据集](reports/stage2_v2/HUMAN_REVIEWED_CLEAN_DATASET_20260624.md)
- [文档索引](docs/README.md)

## 10. 外部源码边界

UABCD、SSFCNet 和 FSG-Net 作者代码位于独立工作区，通过 commit/SHA-256 固定并动态加载，没有复制进本仓库。上游快照缺少可依赖的正式顶层 LICENSE，且原任务、数据和指标与本项目不同；统一迁移结果不应描述为论文表格的精确复现。
