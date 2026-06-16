#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

from stage1_closeout_utils import read_json, sha256_file, write_json


def copy_required(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_optional(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", required=True, help="Stage-1 O1 output dir, e.g. outputs/O1_unet_resnet34_freq")
    parser.add_argument("--practice-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default="configs/stage1_optical_building_unet_resnet34_freq.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp_dir = Path(args.exp_dir)
    practice_root = Path(args.practice_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = exp_dir / "checkpoints" / "best_iou.pth"
    official_checkpoint = out_dir / "best_iou.pth"
    copy_required(checkpoint, official_checkpoint)

    config_path = Path(args.config)
    if config_path.exists():
        copy_required(config_path, out_dir / "config.yaml")
    copy_optional(exp_dir / "config_resolved.json", out_dir / "config_resolved.json")

    copied = {}
    metric_files = {
        "test_all_metrics.json": exp_dir / "test_all" / "metrics.json",
        "test_freq_metrics.json": exp_dir / "test_freq" / "metrics.json",
        "test_rare_metrics.json": exp_dir / "test_rare" / "metrics.json",
        "test_ok_metrics.json": exp_dir / "test_ok" / "metrics.json",
        "per_disaster_metrics.csv": exp_dir / "test_all" / "per_disaster_metrics.csv",
        "per_region_metrics.csv": exp_dir / "test_all" / "per_region_metrics.csv",
        "per_event_metrics.csv": exp_dir / "test_all" / "per_event_metrics.csv",
        "sample_metrics.csv": exp_dir / "test_all" / "sample_metrics.csv",
    }
    for dst_name, src in metric_files.items():
        copied[dst_name] = copy_optional(src, out_dir / dst_name)

    digest = sha256_file(official_checkpoint)
    (out_dir / "checkpoint_sha256.txt").write_text(f"{digest}  best_iou.pth\n", encoding="utf-8")

    test_all = read_json(out_dir / "test_all_metrics.json")
    threshold_path = out_dir / "official_threshold.json"
    threshold = read_json(threshold_path)["threshold"] if threshold_path.exists() else None
    md = f"""# Stage-1 O1 Official Baseline

Frozen at: {datetime.now().astimezone().isoformat(timespec="seconds")}

Model: U-Net ResNet34
Training split: frequent disasters
Input: pre-event optical RGB
Output: binary building mask
Label: building = mask_multiclass > 0
Checkpoint: best_iou.pth
Checkpoint sha256: `{digest}`
Official threshold: {threshold if threshold is not None else "pending threshold tuning"}

Test-all metrics:

```text
IoU_building: {test_all["iou_building"]:.10f}
F1_building: {test_all["f1_building"]:.10f}
Precision:    {test_all["precision_building"]:.10f}
Recall:       {test_all["recall_building"]:.10f}
OA:           {test_all["overall_accuracy"]:.10f}
Boundary F1:  {test_all.get("boundary_f1", 0.0):.10f}
Samples:      {test_all["sample_count"]}
```

Why O1 is official:

1. Highest Stage-1 test_all IoU among O1-O4.
2. Balanced precision and recall.
3. Simple and stable U-Net baseline.
4. Suitable source of building priors for Stage-2 SAR damage classification.

Source:

```text
exp_dir: {exp_dir}
practice_root: {practice_root}
```
"""
    (out_dir / "STAGE1_O1_OFFICIAL.md").write_text(md, encoding="utf-8")
    write_json(
        out_dir / "freeze_summary.json",
        {
            "exp_dir": str(exp_dir),
            "practice_root": str(practice_root),
            "out_dir": str(out_dir),
            "checkpoint_sha256": digest,
            "copied": copied,
            "test_all_iou": test_all["iou_building"],
            "test_all_f1": test_all["f1_building"],
        },
    )
    print(f"Frozen O1 official baseline to {out_dir}")


if __name__ == "__main__":
    main()
