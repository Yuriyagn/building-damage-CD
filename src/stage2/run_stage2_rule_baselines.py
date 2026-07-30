#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stage2.common import read_jsonl, resolve_data_path, resolve_manifest, write_csv, write_json  # noqa: E402
from stage2.datasets import Stage2DamageDataset  # noqa: E402
from stage2.metrics import GroupedStage2Meters, Stage2DamageMeter  # noqa: E402
from stage2.test_stage2 import save_preview  # noqa: E402


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def load_target(data_root: Path, row: dict[str, Any]) -> np.ndarray:
    path = resolve_data_path(data_root, str(row["mask_multiclass"]))
    if path is None:
        raise FileNotFoundError("mask_multiclass is null")
    return load_mask(path)


def compute_majority_maps(data_root: Path, train_manifest: Path) -> tuple[int, dict[str, int]]:
    rows = read_jsonl(train_manifest)
    global_counts = np.zeros(4, dtype=np.int64)
    by_disaster: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(4, dtype=np.int64))
    for row in tqdm(rows, desc="learn_majority", leave=False):
        target = load_target(data_root, row)
        counts = np.bincount(target[target > 0].reshape(-1), minlength=4)
        global_counts += counts
        by_disaster[str(row.get("disaster_type", "") or "unknown")] += counts
    global_majority = int(np.argmax(global_counts[1:4]) + 1) if global_counts[1:4].sum() else 1
    disaster_majority = {}
    for key, counts in by_disaster.items():
        disaster_majority[key] = int(np.argmax(counts[1:4]) + 1) if counts[1:4].sum() else global_majority
    return global_majority, disaster_majority


def evaluate_rule(
    name: str,
    data_root: Path,
    rows: list[dict[str, Any]],
    prior_type: str,
    output_dir: Path,
    rule: str,
    global_majority: int = 1,
    disaster_majority: dict[str, int] | None = None,
    max_previews: int = 32,
) -> dict[str, Any]:
    ds = Stage2DamageDataset(data_root=data_root, manifest=output_dir / "_tmp_empty.jsonl", train=False, prior_type=prior_type)
    # The temporary dataset object is only used for its prior-loading helpers.
    ds.rows = rows
    meter = Stage2DamageMeter()
    grouped = GroupedStage2Meters(["disaster_type", "country_or_region", "event_id"])
    sample_rows: list[dict[str, Any]] = []
    preview_count = 0
    pred_dir = output_dir / name / "predictions"
    preview_dir = output_dir / name / "previews"
    pred_dir.mkdir(parents=True, exist_ok=True)

    for row in tqdm(rows, desc=name):
        target = ds.load_target(row)
        if rule == "all_background":
            prior = np.zeros_like(target, dtype=np.uint8)
            pred = np.zeros_like(target, dtype=np.uint8)
        else:
            prior = ds.load_prior_binary(row, prior_type=prior_type)
            pred_class = 1
            if rule == "global_majority_inside_prior":
                pred_class = global_majority
            elif rule == "disaster_majority_inside_prior":
                pred_class = (disaster_majority or {}).get(str(row.get("disaster_type", "") or "unknown"), global_majority)
            pred = np.zeros_like(target, dtype=np.uint8)
            pred[prior > 0] = np.uint8(pred_class)

        meter.update_prediction(pred, target)
        grouped.update(pred, target, row)
        sample_meter = Stage2DamageMeter()
        sample_meter.update_prediction(pred, target)
        sample_rows.append(
            {
                "id": str(row.get("id", "")),
                "event_id": str(row.get("event_id", "")),
                "disaster_type": str(row.get("disaster_type", "")),
                "country_or_region": str(row.get("country_or_region", "")),
                "rule": name,
                **sample_meter.compute(),
            }
        )
        Image.fromarray(pred).save(pred_dir / f"{row.get('id', '')}.png")
        if preview_count < max_previews:
            sar = ds.load_sar(row).astype(np.float32) / 255.0
            prior_f = prior.astype(np.float32)
            save_preview(preview_dir / f"{row.get('id', '')}.png", sar, prior_f, pred, target)
            preview_count += 1

    metrics = meter.compute()
    metrics.update({"rule": name, "prior_type": prior_type, "sample_count": len(rows)})
    rule_dir = output_dir / name
    write_json(rule_dir / "metrics.json", metrics)
    write_csv(rule_dir / "sample_metrics.csv", sample_rows)
    write_csv(rule_dir / "per_disaster_metrics.csv", grouped.rows("disaster_type"))
    write_csv(rule_dir / "per_region_metrics.csv", grouped.rows("country_or_region"))
    write_csv(rule_dir / "per_event_metrics.csv", grouped.rows("event_id"))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest-root", default="manifests")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-previews", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    manifest_root = resolve_manifest(data_root, args.manifest_root) if Path(args.manifest_root).is_file() else data_root / args.manifest_root
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp = output_dir / "_tmp_empty.jsonl"
    tmp.write_text("", encoding="utf-8")

    no_prior_rows = read_jsonl(manifest_root / "no_prior" / f"{args.split}.jsonl")
    oracle_rows = read_jsonl(manifest_root / "oracle_prior" / f"{args.split}.jsonl")
    predicted_rows = read_jsonl(manifest_root / "predicted_prior" / f"{args.split}.jsonl")
    if args.limit is not None:
        no_prior_rows = no_prior_rows[: args.limit]
        oracle_rows = oracle_rows[: args.limit]
        predicted_rows = predicted_rows[: args.limit]

    train_manifest = manifest_root / "oracle_prior" / "train.jsonl"
    global_majority, disaster_majority = compute_majority_maps(data_root, train_manifest)
    write_json(
        output_dir / "majority_class_info.json",
        {
            "global_majority_inside_building": global_majority,
            "disaster_majority_inside_building": disaster_majority,
        },
    )

    summary = []
    summary.append(
        evaluate_rule(
            "R0_all_background",
            data_root,
            no_prior_rows,
            "none",
            output_dir,
            "all_background",
            max_previews=args.max_previews,
        )
    )
    for prior_type, rows, prefix in [
        ("oracle", oracle_rows, "R1_oracle"),
        ("predicted", predicted_rows, "R2_predicted"),
    ]:
        summary.append(
            evaluate_rule(
                f"{prefix}_all_intact",
                data_root,
                rows,
                prior_type,
                output_dir,
                "all_intact",
                max_previews=args.max_previews,
            )
        )
        summary.append(
            evaluate_rule(
                f"{prefix}_global_majority_inside_prior",
                data_root,
                rows,
                prior_type,
                output_dir,
                "global_majority_inside_prior",
                global_majority=global_majority,
                disaster_majority=disaster_majority,
                max_previews=args.max_previews,
            )
        )
        summary.append(
            evaluate_rule(
                f"{prefix}_disaster_majority_inside_prior",
                data_root,
                rows,
                prior_type,
                output_dir,
                "disaster_majority_inside_prior",
                global_majority=global_majority,
                disaster_majority=disaster_majority,
                max_previews=args.max_previews,
            )
        )

    write_csv(output_dir / "rule_baseline_summary.csv", summary)
    print(json.dumps({"rules": [row["rule"] for row in summary], "output_dir": str(output_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
