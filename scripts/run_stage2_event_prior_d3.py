#!/usr/bin/env python3
"""D3 metadata-only priors for quantifying event/class predictability."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stage2.common import read_jsonl, write_csv, write_json  # noqa: E402
from stage2.metrics_v2 import summarize_event_generalization  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def load_mask(data_root: Path, row: dict[str, Any]) -> np.ndarray:
    return np.asarray(Image.open(data_root / str(row["mask_multiclass"])).convert("L"), dtype=np.uint8)


def grade_hist(mask: np.ndarray) -> np.ndarray:
    return np.bincount(mask.reshape(-1), minlength=4)[1:4].astype(np.int64)


def aggregate(rows: list[dict[str, Any]], histograms: dict[str, np.ndarray], key: str) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(3, dtype=np.int64))
    for row in rows:
        result[str(row.get(key, "") or "unknown")] += histograms[str(row["id"])]
    return dict(result)


def metrics_from_confusion(confusion: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    f1s = []
    for index, name in enumerate(("intact", "damaged", "destroyed")):
        tp = float(confusion[index, index])
        fp = float(confusion[:, index].sum() - confusion[index, index])
        fn = float(confusion[index, :].sum() - confusion[index, index])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[f"building_only_f1_{name}"] = f1
        out[f"building_only_precision_{name}"] = precision
        out[f"building_only_recall_{name}"] = recall
        out[f"building_only_support_{name}"] = float(confusion[index, :].sum())
        f1s.append(f1)
    out["building_only_macro_f1_3class"] = float(np.mean(f1s))
    out["building_only_damage_macro_f1"] = float(np.mean(f1s[1:3]))
    true_damage = float(confusion[1:3, :].sum())
    pred_damage = float(confusion[:, 1:3].sum())
    tp_damage = float(confusion[1:3, 1:3].sum())
    precision = tp_damage / pred_damage if pred_damage else 0.0
    recall = tp_damage / true_damage if true_damage else 0.0
    out["building_only_damage_binary_f1"] = (
        2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    )
    out["pixels_total"] = float(confusion.sum())
    return out


def evaluate(
    name: str,
    rows: list[dict[str, Any]],
    histograms: dict[str, np.ndarray],
    predictor: Callable[[dict[str, Any], np.ndarray], int],
    out_dir: Path,
) -> dict[str, Any]:
    global_confusion = np.zeros((3, 3), dtype=np.int64)
    event_confusions: dict[str, np.ndarray] = defaultdict(lambda: np.zeros((3, 3), dtype=np.int64))
    for row in rows:
        hist = histograms[str(row["id"])]
        grade_index = int(predictor(row, hist)) - 1
        event = str(row.get("event_id", "") or "unknown")
        global_confusion[:, grade_index] += hist
        event_confusions[event][:, grade_index] += hist
    strategy_dir = out_dir / name
    strategy_dir.mkdir(parents=True, exist_ok=True)
    event_rows = [
        {"event_id": event, **metrics_from_confusion(confusion)}
        for event, confusion in sorted(event_confusions.items())
    ]
    metrics = metrics_from_confusion(global_confusion)
    event_summary = summarize_event_generalization(event_rows)
    write_json(strategy_dir / "metrics.json", metrics)
    write_json(strategy_dir / "event_generalization.json", event_summary)
    write_csv(strategy_dir / "per_event_metrics.csv", event_rows)
    return {"strategy": name, **metrics, **event_summary}


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_jsonl(args.train_manifest)
    val_rows = read_jsonl(args.val_manifest)
    histograms = {
        str(row["id"]): grade_hist(load_mask(args.data_root, row)) for row in train_rows + val_rows
    }
    global_hist = sum((histograms[str(row["id"])] for row in train_rows), np.zeros(3, dtype=np.int64))
    global_grade = int(np.argmax(global_hist)) + 1
    disaster_hists = aggregate(train_rows, histograms, "disaster_type")
    val_event_hists = aggregate(val_rows, histograms, "event_id")
    val_sample_hists = {str(row["id"]): histograms[str(row["id"])] for row in val_rows}

    summaries = []
    summaries.append(
        evaluate("global_train_prior_on_val", val_rows, histograms, lambda row, target: global_grade, args.out_dir)
    )
    summaries.append(
        evaluate(
            "train_disaster_prior_on_val",
            val_rows,
            histograms,
            lambda row, target: int(
                np.argmax(disaster_hists.get(str(row.get("disaster_type", "") or "unknown"), global_hist))
            )
            + 1,
            args.out_dir,
        )
    )
    summaries.append(
        evaluate(
            "oracle_val_event_prior_upper_bound",
            val_rows,
            histograms,
            lambda row, target: int(np.argmax(val_event_hists[str(row["event_id"])])) + 1,
            args.out_dir,
        )
    )

    def loio_event_predictor(row: dict[str, Any], target: np.ndarray) -> int:
        remaining = val_event_hists[str(row["event_id"])] - val_sample_hists[str(row["id"])]
        source = remaining if int(remaining.sum()) > 0 else global_hist
        return int(np.argmax(source)) + 1

    summaries.append(
        evaluate(
            "leave_one_image_out_val_event_prior",
            val_rows,
            histograms,
            loio_event_predictor,
            args.out_dir,
        )
    )
    write_csv(args.out_dir / "summary.csv", summaries)
    write_json(
        args.out_dir / "run_info.json",
        {
            "diagnostic": "D3_event_prior",
            "train_manifest": str(args.train_manifest.resolve()),
            "val_manifest": str(args.val_manifest.resolve()),
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "global_train_grade": global_grade,
            "global_train_grade_hist_intact_damaged_destroyed": global_hist.astype(int).tolist(),
            "strategies": [row["strategy"] for row in summaries],
            "test_used": False,
        },
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
