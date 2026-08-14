# Building Damage Change Detection

跨事件光学–SAR 建筑损伤变化检测研究代码。项目输出像素级四分类语义图：

```text
background / intact / damaged / destroyed
```

主线采用两阶段设计：

1. **Stage-1**：灾前光学 RGB → 二值建筑先验；
2. **Stage-2**：灾前 RGB、灾后 SAR 与建筑先验 → 建筑内部
   `intact / damaged / destroyed`，再门控为完整四分类结果。

## 当前研究状态

> 状态快照：2026-08-14。请先阅读
> [完整项目审计](docs/PROJECT_AUDIT_20260814.md)。

- 旧 1543/506/564 划分发现 105 个精确 train-test 重复组，旧 Stage-1 和
  Stage-2 v1 测试数字只保留为历史记录。
- 正式 clean strict manifest 为 1207/357/388，内容哈希与 canonical event
  层面 split-disjoint；但现有 strict test 已被多轮探索暴露，不能再充当新的
  盲测集。
- 最新 event-group R0/R4 开发划分为 1372/290/290；R4 只向训练集增加
  262 个 BRIGHT 同事件样本。R4 在三个 validation seeds 上均高于 R0，但收益
  主要来自 Destroyed，且与训练步数和来源域变化混杂。
- 当前证据尚不能证明模型已获得稳定的跨事件损伤等级泛化。下一步重点是
  out-of-fold Stage-1 prior、新的封存外部事件、event × class 数据覆盖以及
  matched-budget 数据对照，而不是继续无约束堆叠模型。

## 主要指标

正式模型选择以 GT building pixels 内的三等级宏平均为主：

```text
BO grade macro F1 = mean(F1_intact, F1_damaged, F1_destroyed)
damage macro F1   = mean(F1_damaged, F1_destroyed)
```

同时报告 `Damaged`、`Destroyed`、binary damage、event-macro、worst-event、
event × class support，以及 predicted-prior gate 下的端到端四分类指标。背景主导的
accuracy/mIoU 不能替代 BO 指标。

## 仓库结构

```text
configs/      冻结实验配置和 event-group 映射
docs/         GitHub 入口文档、完整项目审计和文档索引
manifests/    已审计的小型/历史 manifest；新生成的大型 manifests 默认忽略
reports/      数据完整性、实验结论和统一外部迁移报告
scripts/      数据构建、审计、训练启动、评估和汇总脚本
src/          Stage-1 / Stage-2 数据、模型、损失和指标实现
tests/        单元与协议回归测试
tools/        报告导出和辅助工具
instruction.sh 统一命令入口
```

数据集、模型权重、训练输出、日志、缓存、生成预测和隔离的第三方源码不进入 Git。

## 环境与数据

历史实验使用 Miniconda 环境 `sam3`。建议显式传入环境和数据路径：

```bash
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate sam3

export STAGE2_DATA_ROOT=/path/to/DisasterM3_optical_sar_damage_minimal_v0.2
export PYTHON_BIN="$(command -v python)"
```

Stage-2 minimal package 应包含灾前光学、灾后 SAR、四分类 mask、oracle building
mask 和 Stage-1 prior。数据不随本仓库发布；构建边界见
[MINIMAL_PACKAGE_REPORT.md](MINIMAL_PACKAGE_REPORT.md)。该文件记录的是历史包构建，
其 legacy split 数字已经被后续完整性更正取代。

## 快速检查

```bash
python -m unittest discover -s tests -q
bash instruction.sh stage2_v2_strict_check_runtime
```

正式 GPU 训练前还必须检查目标 manifest、数据审计、`nvidia-smi`、tmux session、
输出目录和随机种子。长任务应在 tmux 中运行并写入唯一日志，不应在前台启动。

Stage-1 历史入口仍可使用：

```bash
bash instruction.sh check_env
bash instruction.sh check_data
bash instruction.sh smoke_unet_freq
```

## 数据与实验纪律

- 只在 train/validation 上开发和选择配方；已暴露测试集只作
  `historical_diagnostic_only`。
- 固定并保存 within-event deranged SAR permutation；paired/shuffled 必须成对。
- 正式比较使用 seeds `42 / 3407 / 2026`，并保存 resolved config、代码状态、
  数据审计与 checkpoint selection rule。
- 结论必须同时检查 pooled、sample、event 和 event × class 层级。
- 不提交数据、权重、输出、日志、缓存或第三方作者源码。

## 关键文档

- [文档索引](docs/README.md)
- [完整项目审计与下一步建议](docs/PROJECT_AUDIT_20260814.md)
- [数据完整性更正](reports/stage2_v2/DATA_INTEGRITY_CORRECTION_20260622.md)
- [人工复核 clean 数据集](reports/stage2_v2/HUMAN_REVIEWED_CLEAN_DATASET_20260624.md)
- [统一外部模型实验标准](reports/stage2_v2/UNIFIED_EXPERIMENT_STANDARD_V1_20260731.md)
- [R0/R4 BRIGHT 数据补充比较](reports/stage2_v2/event_group_dataset_v1_20260803/R0_R4_PREDICTED_PRIOR_COMPARISON_20260804.md)

## 外部源码边界

UABCD、SSFCNet 和 FSG-Net 的作者代码在独立工作区动态加载并用 SHA-256/commit
固定，未复制进本仓库。相关上游快照缺少可依赖的正式顶层 LICENSE，且其原任务、
数据和指标与本项目不同；仓库中保留的是适配代码、协议与实验报告，不应把统一迁移
结果描述为论文表格的精确复现。
