# Documentation index

本目录是 GitHub 读者的入口。历史报告保留用于追溯，但应按下面的证据层级阅读。

## 先读

1. [项目全景、实验审计与下一步建议](PROJECT_AUDIT_20260814.md)
2. [主 README](../README.md)
3. [Stage-2 数据完整性更正](../reports/stage2_v2/DATA_INTEGRITY_CORRECTION_20260622.md)
4. [人工复核 clean 数据集](../reports/stage2_v2/HUMAN_REVIEWED_CLEAN_DATASET_20260624.md)

## 实验协议与主要结果

- [Stage-2 baseline implementation](../STAGE2_BASELINE_IMPLEMENTATION.md)
- [Stage-1 project experience](../PROJECT_EXPERIENCE.md)
- [Stage-2 v2 实验总报告](../reports/stage2_v2/stage2_v2_experiment_report.md)
- [Oracle prior exploration](../reports/stage2_v2/ORACLE_PRIOR_EXPLORATION_RESULTS_20260625.md)
- [Pre-change exploration](../reports/stage2_v2/PRECHANGE_EXPERIMENT_RESULTS_20260626.md)
- [Phase-5 threshold/adaptive analysis](../reports/stage2_v2/PHASE5_THRESHOLD_ADAPTIVE_REPORT_20260630.md)

## 外部架构统一迁移

- [统一实验标准](../reports/stage2_v2/UNIFIED_EXPERIMENT_STANDARD_V1_20260731.md)
- [UABCD](../reports/stage2_v2/UABCD_UNIFIED_V1_RESULTS_20260731.md)
- [SSFCNet](../reports/stage2_v2/SSFCNET_UNIFIED_V1_EXPERIMENT_20260731.md)
- [FSG-Net](../reports/stage2_v2/FSGNET_UNIFIED_V1_EXPERIMENT_20260731.md)

## 事件捷径与数据补充

- [event × class distribution](../reports/stage2_v2/event_class_distribution_20260803/distribution_summary.md)
- [LOEO shortcut conclusion](../reports/stage2_v2/event_shortcut_v1_20260803/D4_LOEO_CONCLUSION.md)
- [event-group dataset build report](../reports/stage2_v2/event_group_dataset_v1_20260803/DATASET_BUILD_REPORT.md)
- [R0/R4 comparison](../reports/stage2_v2/event_group_dataset_v1_20260803/R0_R4_PREDICTED_PRIOR_COMPARISON_20260804.md)

## 状态说明

- legacy 1543/506/564 test 已因精确内容泄漏撤回。
- clean strict 1207/357/388 test 已被多轮探索暴露，只能作历史诊断。
- event-group 1372/290/290 test 已用于 R0/R4 诊断，也不再是新的盲测集。
- `MINIMAL_PACKAGE_REPORT.md`、早期 runbook 和部分 readiness 文件记录的是当时状态；
  当它们与 2026-08-14 审计冲突时，以项目审计和磁盘冻结产物为准。
