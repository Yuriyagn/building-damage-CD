#!/usr/bin/env python3
"""Quantify the withdrawn legacy test after excluding exact train-test duplicates."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stage2.common import write_json  # noqa: E402
from stage2.summarize_stage2_v2 import load_runs, macro_f1, paired_bootstrap  # noqa: E402


def leaked_test_ids(deduplication_groups: Path) -> list[str]:
    groups = json.loads(deduplication_groups.read_text(encoding="utf-8"))
    leaked: set[str] = set()
    for group in groups:
        members = list(group["members"])
        splits = {str(member["source_split"]) for member in members}
        if "train" not in splits or "test" not in splits:
            continue
        leaked.update(
            str(member["source_id"]) for member in members if str(member["source_split"]) == "test"
        )
    return sorted(leaked)


def run_metrics(run: dict[str, Any]) -> dict[str, Any]:
    confusion = np.zeros((3, 3), dtype=np.int64)
    for row in run["samples"]:
        confusion += np.asarray(json.loads(row["bo_confusion_3x3"]), dtype=np.int64).reshape(3, 3)
    return {
        "experiment": str(run["experiment"]),
        "seed": int(run["seed"]),
        "sample_count": len(run["samples"]),
        "building_only_macro_f1_3class": macro_f1(confusion, (0, 1, 2)),
        "building_only_f1_intact": macro_f1(confusion, (0,)),
        "building_only_f1_damaged": macro_f1(confusion, (1,)),
        "building_only_f1_destroyed": macro_f1(confusion, (2,)),
        "building_only_damage_macro_f1": macro_f1(confusion, (1, 2)),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["experiment"])].append(row)
    metrics = (
        "building_only_macro_f1_3class",
        "building_only_f1_intact",
        "building_only_f1_damaged",
        "building_only_f1_destroyed",
        "building_only_damage_macro_f1",
    )
    output = []
    for experiment, values in sorted(grouped.items()):
        result: dict[str, Any] = {
            "experiment": experiment,
            "seed_count": len(values),
            "sample_count_min": min(int(value["sample_count"]) for value in values),
            "sample_count_max": max(int(value["sample_count"]) for value in values),
        }
        for metric in metrics:
            array = np.asarray([float(value[metric]) for value in values])
            result[f"{metric}_mean"] = float(array.mean())
            result[f"{metric}_std"] = float(array.std(ddof=1)) if len(array) > 1 else 0.0
        output.append(result)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deduplication-groups", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    args = parser.parse_args()

    excluded = set(leaked_test_ids(Path(args.deduplication_groups)))
    runs = load_runs(Path(args.runs_root), "test_best_grade")
    filtered = []
    for run in runs:
        copy = deepcopy(run)
        copy["samples"] = [row for row in copy["samples"] if str(row["id"]) not in excluded]
        filtered.append(copy)
    run_rows = [run_metrics(run) for run in filtered]

    a3 = [run for run in filtered if "A3" in str(run["experiment"]).upper()]
    comparisons = {}
    for control in ("A1", "A2", "A4"):
        control_runs = [run for run in filtered if control in str(run["experiment"]).upper()]
        comparisons[f"A3-{control}"] = paired_bootstrap(
            a3,
            control_runs,
            iterations=max(1, int(args.bootstrap_iterations)),
            seed=20260622,
        )

    write_json(
        args.output,
        {
            "status": "post_hoc_exact-duplicate-filtered_diagnostic_not_formal_test",
            "reason": "The legacy test was contaminated by exact train-test duplicates and is withdrawn.",
            "excluded_test_sample_count": len(excluded),
            "excluded_test_ids": sorted(excluded),
            "runs": run_rows,
            "mean_std": summarize(run_rows),
            "paired_bootstrap": comparisons,
        },
    )
    print(json.dumps({"excluded_test_sample_count": len(excluded), "output": args.output}, sort_keys=True))


if __name__ == "__main__":
    main()
