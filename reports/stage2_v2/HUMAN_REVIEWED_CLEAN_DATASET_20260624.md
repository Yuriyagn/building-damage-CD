# Stage-2 v2 human-reviewed clean dataset

日期：2026-06-24

## 结论

正式后续训练入口已整理为：

```text
manifests/stage2_v2_clean_human_reviewed_20260624/
```

该目录基于 strict-v1 的全局去重与 canonical-event-disjoint split，并合并人工复核记录。它保留 `1207/357/388` 个 train/val/test 样本，完整内容重复为 0，event overlap 为 0，复审 hard errors 为 0。唯一 warning 仍是 minimal package 没有传播 `qc_label`。

## 人工复核记录

### legacy 审计候选

| 类别 | 数量 |
|---|---:|
| 总候选 | 115 |
| 已复核 | 115 |
| High confirmed overlap, 删除 train 侧 | 106 |
| Medium likely overlap, 删除 train 侧 | 1 |
| Medium/Low not overlap, 保留双方 | 8 |

LOW 候选没有在默认网页筛选中显示，是因为前端默认筛选原先是 `high+medium`。该唯一 LOW 候选已按用户说明记录为 `not_overlap + keep_both_record_only`。

### strict-v1 审计候选

strict-v1 中真正影响 train-test 独立性的 2 个候选均已记录为 `not_overlap + keep_both_record_only`。其余未复核候选为同 split 内或 train-val 内部风险记录，不影响正式 train/test 独立性；后续若要做更保守的同 split 去冗余，可以单独开启新协议，不应混入本次正式协议。

## 导出的两个 manifest

| Manifest root | 用途 | Counts | 删除样本 | 审计结论 |
|---|---|---:|---:|---|
| `manifests/stage2_v2_clean_human_reviewed_20260624/` | 正式后续训练 | 1207/357/388 | 0 | 0 hard errors；0 full duplicates；event-disjoint |
| `manifests/stage2_v2_legacy_overlap_reviewed_20260624/` | legacy 诊断记录 | 1436/506/564 | 107 train-side | cross-split full duplicates 已清理；仍有 556 个 split 内完整重复组，不作为正式协议 |

legacy 诊断版只证明人工标注的 train-test high/medium 风险可以被重现地移除；它不是新的正式 test 协议，也不应用于最终跨事件泛化结论。

## 验收记录

正式 clean set：

- full audit: `outputs/stage2/clean_human_reviewed_20260624/full_audit.json`
- dataloader smoke: train/val/test 均能读取 `predicted_prior` manifest，batch shape 为 `(2, 3, 1024, 1024)` 与 `(2, 1024, 1024)`
- distribution summary: `outputs/stage2/clean_human_reviewed_20260624/distribution/`
- human review summary: `outputs/stage2/clean_human_reviewed_20260624/human_review_summary.json`

legacy 诊断版：

- audit: `outputs/stage2/legacy_overlap_reviewed_20260624/audit_skip_mask.json`
- dry-run after LOW decision: `outputs/stage2/legacy_overlap_reviewed_20260624/apply_dry_run_after_low_review.json`

## 后续训练入口

`configs/stage2_v2_strict_v1/*.yaml` 和 `scripts/launch_stage2_v2_strict_run.sh` 已指向 `stage2_v2_clean_human_reviewed_20260624`。

正式训练仍按原四个实验启动：

```bash
bash scripts/launch_stage2_v2_strict_run.sh a1 42 0
bash scripts/launch_stage2_v2_strict_run.sh a2 42 1
# 完成后
bash scripts/launch_stage2_v2_strict_run.sh a3 42 0
bash scripts/launch_stage2_v2_strict_run.sh a4 42 1
```

三 seed 正式比较仍为 `42, 3407, 2026`。不要使用 legacy launcher 作为新正式实验入口。
