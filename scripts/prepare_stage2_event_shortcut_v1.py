#!/usr/bin/env python3
"""Build train-only leave-one-event-out folds for shortcut diagnostics."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("manifests/stage2_v2_clean_human_reviewed_20260624/predicted_prior/train.jsonl"),
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/stage2_v2_phase3_objective/v2_obj_e2_p2_binary_aux.yaml"),
    )
    parser.add_argument(
        "--manifest-root", type=Path, default=Path("manifests/stage2_event_shortcut_v1/d4_loeo")
    )
    parser.add_argument(
        "--config-root", type=Path, default=Path("configs/stage2_event_shortcut_v1/d4_loeo")
    )
    parser.add_argument("--data-root", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def load_class_histograms(rows: list[dict[str, Any]], data_root: Path) -> dict[str, np.ndarray]:
    result = {}
    for row in rows:
        mask = np.asarray(Image.open(data_root / str(row["mask_multiclass"])).convert("L"), dtype=np.uint8)
        result[str(row["id"])] = np.bincount(mask.reshape(-1), minlength=4)[1:4].astype(np.int64)
    return result


def class_pixels(rows: list[dict[str, Any]], histograms: dict[str, np.ndarray]) -> list[int]:
    counts = sum((histograms[str(row["id"])] for row in rows), np.zeros(3, dtype=np.int64))
    return counts.astype(int).tolist()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.source_manifest)
    if not rows:
        raise ValueError("source manifest is empty")
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("source manifest contains duplicate ids")
    events = sorted({str(row["event_id"]) for row in rows})
    histograms = load_class_histograms(rows, args.data_root)
    base = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    source_sha256 = hashlib.sha256(args.source_manifest.read_bytes()).hexdigest()
    summary: dict[str, Any] = {
        "protocol_id": "stage2_event_shortcut_v1",
        "diagnostic": "D4_leave_one_event_out",
        "source_manifest": str(args.source_manifest.resolve()),
        "source_sha256": source_sha256,
        "source_rows": len(rows),
        "source_event_counts": dict(sorted(Counter(str(row["event_id"]) for row in rows).items())),
        "folds": {},
    }

    for holdout in events:
        train_rows = []
        val_rows = []
        for source in rows:
            row = copy.deepcopy(source)
            row["original_strict_split"] = str(source.get("split", "train"))
            row["loeo_holdout_event"] = holdout
            if str(source["event_id"]) == holdout:
                row["split"] = "val"
                val_rows.append(row)
            else:
                row["split"] = "train"
                train_rows.append(row)
        if not train_rows or not val_rows:
            raise ValueError(f"invalid fold {holdout}: train={len(train_rows)} val={len(val_rows)}")
        train_events = {str(row["event_id"]) for row in train_rows}
        val_events = {str(row["event_id"]) for row in val_rows}
        if train_events & val_events:
            raise AssertionError(f"event overlap in fold {holdout}")
        if {str(row["id"]) for row in train_rows} & {str(row["id"]) for row in val_rows}:
            raise AssertionError(f"id overlap in fold {holdout}")

        fold_root = args.manifest_root / holdout
        write_jsonl(fold_root / "train.jsonl", train_rows)
        write_jsonl(fold_root / "val.jsonl", val_rows)

        config = copy.deepcopy(base)
        config["protocol_id"] = "stage2_event_shortcut_v1"
        config["experiment_name"] = f"S2ES_D4_LOEO_{holdout}"
        config["dataset"]["train_manifest"] = str(fold_root / "train.jsonl")
        config["dataset"]["val_manifest"] = str(fold_root / "val.jsonl")
        config["dataset"]["test_manifest"] = str(fold_root / "val.jsonl")
        config["train"]["checkpoint_policy"] = "primary_only"
        config["train"]["save_last_checkpoint"] = False
        config["diagnostic"] = {
            "id": "D4",
            "holdout_event": holdout,
            "source_split": "strict_clean_train_only",
            "formal_test_embargoed": True,
        }
        config_path = args.config_root / f"{holdout}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        train_counts = class_pixels(train_rows, histograms)
        val_counts = class_pixels(val_rows, histograms)
        if any(value <= 0 for value in train_counts):
            raise ValueError(f"fold {holdout} training partition loses a grade: {train_counts}")
        fold_audit = {
            "status": "pass",
            "holdout_event": holdout,
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "train_events": sorted(train_events),
            "val_events": sorted(val_events),
            "event_overlap": [],
            "id_overlap": [],
            "train_class_pixels_intact_damaged_destroyed": train_counts,
            "val_class_pixels_intact_damaged_destroyed": val_counts,
            "source_sha256": source_sha256,
        }
        (fold_root / "audit.json").write_text(
            json.dumps(fold_audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        summary["folds"][holdout] = fold_audit

    if sum(int(item["val_rows"]) for item in summary["folds"].values()) != len(rows):
        raise AssertionError("LOEO holdout partitions do not cover the source manifest exactly once")
    summary_path = args.manifest_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
