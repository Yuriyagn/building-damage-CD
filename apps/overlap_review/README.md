# OverlapReview-QC

用于 `relaxed-overlap-v1` 的人工候选复核。自动分数只生成候选，不能替代人工空间重叠判断。

## 启动

```bash
cd '/home/yr/code/Building damage change detection/stage1_optical_building'
source /home/yr/miniconda3/etc/profile.d/conda.sh
conda activate sam3

export OVERLAP_AUDIT_BASE="$PWD/outputs/stage2/overlap_audit_20260622"
uvicorn apps.overlap_review.main:app --host 127.0.0.1 --port 8787
```

远程访问使用 SSH tunnel：

```bash
ssh -L 8787:127.0.0.1:8787 yr@115.156.98.196
```

浏览器打开 `http://127.0.0.1:8787`。

## 审核范围

- `train-test`：全部 high + medium 必须审核；confirmed/likely 默认删除 train 侧。
- `train-val`、`val-test`：high 必须记录；默认 `keep_both_record_only`。
- `uncertain` 使用 `needs_second_review`，不能算完成审核。

界面可在 `legacy` 与 `strict-v1` 之间切换，左栏按风险和复核状态统计；中央连续显示候选，train-test 对中 Train 在上、Test 在下。按 `1`–`6` 判定，`J/K` 移动。决定逐条原子写入各协议的 `review_decisions.csv`。

`Not Overlapped` 自动使用 `keep_both_record_only`，表示双方都保留。完成 strict-v1 必审项后，页面右上角可显式生成一个新的 `stage2_v2_strict_v1_overlap_reviewed`；原 strict-v1 不会被覆盖。等价 CLI：

```bash
python scripts/apply_overlap_decisions.py \
  --manifest-root manifests/stage2_v2_strict_v1 \
  --candidates outputs/stage2/overlap_audit_20260622/strict_v1/candidates_all.csv \
  --decisions outputs/stage2/overlap_audit_20260622/strict_v1/review_decisions.csv \
  --policy relaxed_train_test_only \
  --out-root manifests/stage2_v2_strict_v1_overlap_reviewed
```

脚本默认拒绝在必审候选未完成时生成最终 manifest，也拒绝覆盖已有输出。
