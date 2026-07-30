#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import load_config  # noqa: E402
from stage2.common import get_metadata, write_csv, write_json  # noqa: E402
from stage2.metrics_v2 import GroupedV2Meters, Stage2V2MeterBundle  # noqa: E402
from stage2.test_stage2_v2 import event_sets, familiarity, make_dataset, sample_row  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-2 v2 all-intact rule baseline")
    parser.add_argument("--config", default="configs/stage2_v2/v2_a1_prior_only.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = make_dataset(cfg, data_root, args.split, seed=42, limit=None)
    train_events, val_events = event_sets(cfg, data_root)
    meter = Stage2V2MeterBundle()
    grouped = GroupedV2Meters(["disaster_type", "country_or_region", "event_id", "event_familiarity"])
    rows: list[dict[str, Any]] = []
    for index, manifest_row in enumerate(tqdm(dataset.rows, desc=f"all_intact_{args.split}")):
        target = dataset.load_target(manifest_row)
        support = dataset.load_prior_binary(manifest_row).astype(bool)
        grade = np.ones_like(target, dtype=np.uint8)
        pred = np.where(support, grade, 0).astype(np.uint8)
        base = get_metadata(manifest_row)
        event_id = base["event_id"]
        metadata: dict[str, Any] = {
            **base,
            "event_familiarity": familiarity(event_id, train_events, val_events),
            "sar_source_id": dataset._sar_source_ids[index],
            "sar_is_paired": bool(dataset.sar_permutation[index] == index),
        }
        meter.update(grade, target, support)
        grouped.update(grade, target, support, metadata)
        rows.append(sample_row(metadata, grade, pred, target, support))
    metrics: dict[str, Any] = meter.compute()
    metrics.update({"rule": "all_intact", "sample_count": len(dataset), "split": args.split})
    write_json(output_dir / "metrics.json", metrics)
    write_csv(output_dir / "sample_metrics.csv", rows)
    write_csv(output_dir / "per_disaster_metrics.csv", grouped.rows("disaster_type"))
    write_csv(output_dir / "per_region_metrics.csv", grouped.rows("country_or_region"))
    write_csv(output_dir / "per_event_metrics.csv", grouped.rows("event_id"))
    write_csv(output_dir / "per_event_familiarity_metrics.csv", grouped.rows("event_familiarity"))
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
