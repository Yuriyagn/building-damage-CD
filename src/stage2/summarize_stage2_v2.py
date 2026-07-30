#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stage2.common import write_csv, write_json  # noqa: E402


SUMMARY_METRICS = [
    "building_only_macro_f1_3class",
    "building_only_damage_macro_f1",
    "building_only_damage_binary_f1",
    "building_only_f1_intact",
    "building_only_f1_damaged",
    "building_only_f1_destroyed",
    "predicted_gate_miou_4class",
    "predicted_gate_macro_f1_4class",
    "cc_surrogate_all_macro_f1",
    "cc_surrogate_all_f1_intact",
    "cc_surrogate_all_f1_damaged",
    "cc_surrogate_all_f1_destroyed",
    "cc_surrogate_total_components",
    "cc_surrogate_ignored_components",
    "cc_surrogate_low_purity_components",
]


def infer_run(path: Path) -> tuple[str, int]:
    seed = -1
    experiment = "unknown"
    for part in path.parts:
        match = re.fullmatch(r"seed[_-]?(\d+)", part, flags=re.IGNORECASE)
        if match:
            seed = int(match.group(1))
        if part.upper().startswith("V2_"):
            experiment = part
    return experiment, seed


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_runs(root: Path, evaluation_name: str) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    pattern = f"**/{evaluation_name}/metrics.json"
    for metrics_path in sorted(root.glob(pattern)):
        experiment, seed = infer_run(metrics_path)
        sample_path = metrics_path.parent / "sample_metrics.csv"
        if experiment == "unknown" or not sample_path.exists():
            continue
        runs.append(
            {
                "experiment": experiment,
                "seed": seed,
                "metrics_path": metrics_path,
                "evaluation_dir": metrics_path.parent,
                "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
                "samples": read_csv(sample_path),
            }
        )
    return runs


def summarize_runs(runs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        row: dict[str, Any] = {
            "experiment": run["experiment"],
            "seed": run["seed"],
            "metrics_path": str(run["metrics_path"]),
            "sample_count": run["metrics"].get("sample_count"),
        }
        for metric in SUMMARY_METRICS:
            row[metric] = run["metrics"].get(metric)
        raw_rows.append(row)
        grouped[run["experiment"]].append(row)

    summary_rows: list[dict[str, Any]] = []
    for experiment, rows in sorted(grouped.items()):
        summary: dict[str, Any] = {"experiment": experiment, "seed_count": len(rows)}
        for metric in SUMMARY_METRICS:
            values = np.asarray([float(row[metric]) for row in rows if row.get(metric) is not None])
            if len(values):
                summary[f"{metric}_mean"] = float(values.mean())
                summary[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(summary)
    return raw_rows, summary_rows


def summarize_group_file(
    runs: list[dict[str, Any]], filename: str, group_column: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_rows: list[dict[str, Any]] = []
    for run in runs:
        path = run["evaluation_dir"] / filename
        if not path.exists():
            continue
        sample_counts = defaultdict(int)
        for sample in run["samples"]:
            sample_counts[str(sample.get(group_column, ""))] += 1
        for source in read_csv(path):
            group = str(source[group_column])
            row: dict[str, Any] = {
                "experiment": run["experiment"],
                "seed": run["seed"],
                group_column: group,
                "sample_count": sample_counts[group],
            }
            for metric in SUMMARY_METRICS:
                if source.get(metric) not in (None, ""):
                    row[metric] = float(source[metric])
            raw_rows.append(row)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        grouped[(str(row["experiment"]), str(row[group_column]))].append(row)
    summary_rows: list[dict[str, Any]] = []
    for (experiment, group), rows in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "experiment": experiment,
            group_column: group,
            "seed_count": len(rows),
            "sample_count_min": min(int(row["sample_count"]) for row in rows),
            "sample_count_max": max(int(row["sample_count"]) for row in rows),
        }
        for metric in SUMMARY_METRICS:
            values = np.asarray([float(row[metric]) for row in rows if metric in row])
            if len(values):
                summary[f"{metric}_mean"] = float(values.mean())
                summary[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(summary)
    return raw_rows, summary_rows


def macro_f1(confusion: np.ndarray, classes: tuple[int, ...]) -> float:
    values = []
    for index in classes:
        tp = float(confusion[index, index])
        fp = float(confusion[:, index].sum() - confusion[index, index])
        fn = float(confusion[index, :].sum() - confusion[index, index])
        values.append(2.0 * tp / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) else 0.0)
    return float(np.mean(values))


def sample_map(run: dict[str, Any]) -> dict[tuple[str, str], np.ndarray]:
    out = {}
    for row in run["samples"]:
        key = (str(row.get("split", "")), str(row.get("id", "")))
        out[key] = np.asarray(json.loads(row["bo_confusion_3x3"]), dtype=np.int64).reshape(3, 3)
    return out


def paired_bootstrap(
    left_runs: list[dict[str, Any]],
    right_runs: list[dict[str, Any]],
    iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    left_by_seed = {int(run["seed"]): run for run in left_runs}
    right_by_seed = {int(run["seed"]): run for run in right_runs}
    common_seeds = sorted(set(left_by_seed) & set(right_by_seed))
    pairs = []
    for run_seed in common_seeds:
        left = sample_map(left_by_seed[run_seed])
        right = sample_map(right_by_seed[run_seed])
        keys = sorted(set(left) & set(right))
        if keys:
            pairs.append((run_seed, np.stack([left[key] for key in keys]), np.stack([right[key] for key in keys]), len(keys)))
    if not pairs:
        return []

    rng = np.random.default_rng(seed)
    outputs = []
    for metric_name, classes in (
        ("bo_grade_macro_f1", (0, 1, 2)),
        ("bo_damage_macro_f1", (1, 2)),
        ("bo_intact_f1", (0,)),
        ("bo_damaged_f1", (1,)),
        ("bo_destroyed_f1", (2,)),
    ):
        point_differences = []
        for _, left, right, _ in pairs:
            point_differences.append(macro_f1(left.sum(axis=0), classes) - macro_f1(right.sum(axis=0), classes))
        bootstrap_differences = np.zeros(iterations, dtype=np.float64)
        for iteration in range(iterations):
            seed_differences = []
            for _, left, right, count in pairs:
                indices = rng.integers(0, count, size=count)
                seed_differences.append(
                    macro_f1(left[indices].sum(axis=0), classes) - macro_f1(right[indices].sum(axis=0), classes)
                )
            bootstrap_differences[iteration] = float(np.mean(seed_differences))
        outputs.append(
            {
                "metric": metric_name,
                "seed_count": len(pairs),
                "common_sample_count_min": min(pair[3] for pair in pairs),
                "point_difference": float(np.mean(point_differences)),
                "ci95_low": float(np.quantile(bootstrap_differences, 0.025)),
                "ci95_high": float(np.quantile(bootstrap_differences, 0.975)),
                "probability_difference_gt_zero": float(np.mean(bootstrap_differences > 0)),
            }
        )
    return outputs


def find_experiment(runs: list[dict[str, Any]], token: str) -> list[dict[str, Any]]:
    token = token.upper()
    return [run for run in runs if token in str(run["experiment"]).upper()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Stage-2 v2 multi-seed evaluations")
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--evaluation-name", default="test_best_grade")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260620)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(Path(args.runs_root), args.evaluation_name)
    if not runs:
        raise FileNotFoundError(f"no {args.evaluation_name}/metrics.json evaluations found under {args.runs_root}")
    raw, summary = summarize_runs(runs)
    write_csv(output_dir / "runs.csv", raw)
    write_csv(output_dir / "mean_std.csv", summary)
    for filename, group_column, prefix in (
        ("per_disaster_metrics.csv", "disaster_type", "per_disaster"),
        ("per_event_metrics.csv", "event_id", "per_event"),
        ("per_event_familiarity_metrics.csv", "event_familiarity", "per_event_familiarity"),
    ):
        group_raw, group_summary = summarize_group_file(runs, filename, group_column)
        write_csv(output_dir / f"{prefix}_runs.csv", group_raw)
        write_csv(output_dir / f"{prefix}_mean_std.csv", group_summary)

    a3 = find_experiment(runs, "V2_A3")
    comparisons = []
    for control_name in ("V2_A1", "V2_A2", "V2_A4"):
        rows = paired_bootstrap(
            a3,
            find_experiment(runs, control_name),
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed,
        )
        comparisons.extend({"comparison": f"V2_A3-minus-{control_name}", **row} for row in rows)
    for candidate_name in ("V2_B2", "V2_B3"):
        rows = paired_bootstrap(
            find_experiment(runs, candidate_name),
            a3,
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed,
        )
        comparisons.extend({"comparison": f"{candidate_name}-minus-V2_A3", **row} for row in rows)
    write_csv(output_dir / "paired_bootstrap.csv", comparisons)
    payload = {
        "evaluation_name": args.evaluation_name,
        "run_count": len(runs),
        "experiments": sorted({str(run["experiment"]) for run in runs}),
        "bootstrap_iterations": args.bootstrap_iterations,
        "comparisons": comparisons,
    }
    write_json(output_dir / "summary.json", payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
