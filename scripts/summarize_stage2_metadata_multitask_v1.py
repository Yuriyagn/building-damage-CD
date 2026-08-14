#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np


SEEDS = (42, 3407, 2026)
EXPERIMENTS = {
    "M0": "M0_rgb_sar_damage_only",
    "M1": "M1_rgb_sar_disaster_aux",
    "M2": "M2_rgb_sar_shuffled_disaster_aux",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def latest_run(root: Path, condition: str, seed: int) -> Path | None:
    candidates = []
    seed_root = root / EXPERIMENTS[condition] / f"seed_{seed}"
    for run in seed_root.glob("run_*"):
        if (
            (run / "completed.json").is_file()
            and (run / "val_best_grade" / "metrics.json").is_file()
            and (run / "val_best_grade" / "event_generalization.json").is_file()
        ):
            candidates.append(run)
    return max(candidates, default=None)


def confusion_metrics(confusion: np.ndarray) -> dict[str, float]:
    f1s = []
    for index in range(3):
        tp = float(confusion[index, index])
        fp = float(confusion[:, index].sum() - tp)
        fn = float(confusion[index, :].sum() - tp)
        f1s.append(2.0 * tp / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) else 0.0)
    return {
        "grade_macro": float(np.mean(f1s)),
        "damage_macro": float(np.mean(f1s[1:3])),
    }


def load_run(run: Path, condition: str, seed: int) -> dict[str, Any]:
    val_dir = run / "val_best_grade"
    metrics = read_json(val_dir / "metrics.json")
    event = read_json(val_dir / "event_generalization.json")
    run_info = read_json(run / "run_info.json")
    completed = read_json(run / "completed.json")
    per_event = {row["event_id"]: row for row in read_csv(val_dir / "per_event_metrics.csv")}
    confusions: dict[str, np.ndarray] = defaultdict(lambda: np.zeros((3, 3), dtype=np.int64))
    for row in read_csv(val_dir / "sample_metrics.csv"):
        confusion = np.asarray(json.loads(row["bo_confusion_3x3"]), dtype=np.int64).reshape(3, 3)
        confusions[row["event_id"]] += confusion
    best = completed["best_metrics"]["best_bo_grade_macro_f1.pth"]
    return {
        "condition": condition,
        "seed": seed,
        "run_dir": str(run),
        "initial_model_state_sha256": run_info["initial_model_state_sha256"],
        "checkpoint_epoch": int(best["epoch"]),
        "bo_grade_macro_f1": float(metrics["building_only_macro_f1_3class"]),
        "bo_damage_macro_f1": float(metrics["building_only_damage_macro_f1"]),
        "bo_damage_binary_f1": float(metrics["building_only_damage_binary_f1"]),
        "bo_damaged_f1": float(metrics["building_only_f1_damaged"]),
        "bo_destroyed_f1": float(metrics["building_only_f1_destroyed"]),
        "event_macro_bo_f1": float(event["event_macro_bo_f1"]),
        "event_macro_bo_damage_f1": float(event["event_macro_bo_damage_f1"]),
        "worst_event_bo_f1": float(event["worst_event_bo_f1"]),
        "worst_event_id": str(event["worst_event_id"]),
        "disaster_present_macro_f1": float(
            metrics.get("disaster_classification_present_class_macro_f1", 0.0)
        ),
        "disaster_majority_macro_f1": float(
            metrics.get("disaster_classification_majority_present_class_macro_f1", 0.0)
        ),
        "per_event": per_event,
        "event_confusions": dict(confusions),
    }


def public_row(run: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in run.items() if key not in {"per_event", "event_confusions"}}


def paired_rows(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[dict[str, Any]]:
    right_by_seed = {int(row["seed"]): row for row in right}
    rows = []
    for source in left:
        target = right_by_seed[int(source["seed"])]
        rows.append(
            {
                "left": source["condition"],
                "right": target["condition"],
                "seed": source["seed"],
                "initialization_match": (
                    source["initial_model_state_sha256"] == target["initial_model_state_sha256"]
                ),
                "delta_bo_damage_macro_f1": (
                    target["bo_damage_macro_f1"] - source["bo_damage_macro_f1"]
                ),
                "delta_event_macro_bo_f1": (
                    target["event_macro_bo_f1"] - source["event_macro_bo_f1"]
                ),
                "delta_event_macro_bo_damage_f1": (
                    target["event_macro_bo_damage_f1"] - source["event_macro_bo_damage_f1"]
                ),
                "delta_worst_event_bo_f1": (
                    target["worst_event_bo_f1"] - source["worst_event_bo_f1"]
                ),
                "disaster_macro_over_majority": (
                    target["disaster_present_macro_f1"] - target["disaster_majority_macro_f1"]
                ),
            }
        )
    return rows


def hierarchical_bootstrap(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    left_by_seed = {int(row["seed"]): row for row in left}
    right_by_seed = {int(row["seed"]): row for row in right}
    rng = np.random.default_rng(seed)
    damage_deltas = np.empty(iterations, dtype=np.float64)
    event_deltas = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        selected_seeds = rng.choice(SEEDS, size=len(SEEDS), replace=True)
        selected_damage = []
        selected_event = []
        for selected_seed in selected_seeds:
            source = left_by_seed[int(selected_seed)]
            target = right_by_seed[int(selected_seed)]
            events = sorted(set(source["per_event"]) & set(target["per_event"]))
            chosen_events = rng.choice(events, size=len(events), replace=True)
            source_confusion = np.zeros((3, 3), dtype=np.int64)
            target_confusion = np.zeros((3, 3), dtype=np.int64)
            source_event_values = []
            target_event_values = []
            for event_id in chosen_events:
                event_name = str(event_id)
                source_confusion += source["event_confusions"][event_name]
                target_confusion += target["event_confusions"][event_name]
                source_event_values.append(
                    float(source["per_event"][event_name]["building_only_macro_f1_3class"])
                )
                target_event_values.append(
                    float(target["per_event"][event_name]["building_only_macro_f1_3class"])
                )
            selected_damage.append(
                confusion_metrics(target_confusion)["damage_macro"]
                - confusion_metrics(source_confusion)["damage_macro"]
            )
            selected_event.append(float(np.mean(target_event_values) - np.mean(source_event_values)))
        damage_deltas[iteration] = float(np.mean(selected_damage))
        event_deltas[iteration] = float(np.mean(selected_event))
    return {
        "iterations": iterations,
        "seed": seed,
        "delta_bo_damage_macro_f1_ci95": [
            float(value) for value in np.quantile(damage_deltas, [0.025, 0.975])
        ],
        "delta_event_macro_bo_f1_ci95": [
            float(value) for value in np.quantile(event_deltas, [0.025, 0.975])
        ],
    }


def aggregate(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
    }


def screen_m1(rows: list[dict[str, Any]]) -> dict[str, Any]:
    damage = [float(row["delta_bo_damage_macro_f1"]) for row in rows]
    event = [float(row["delta_event_macro_bo_f1"]) for row in rows]
    worst = [float(row["delta_worst_event_bo_f1"]) for row in rows]
    classification = [float(row["disaster_macro_over_majority"]) for row in rows]
    checks = {
        "initialization_matches_all_seeds": all(bool(row["initialization_match"]) for row in rows),
        "damage_mean_at_least_0p01": mean(damage) >= 0.01,
        "damage_positive_at_least_2_seeds": sum(value > 0 for value in damage) >= 2,
        "event_mean_at_least_0p01": mean(event) >= 0.01,
        "event_positive_at_least_2_seeds": sum(value > 0 for value in event) >= 2,
        "worst_event_mean_delta_at_least_minus_0p01": mean(worst) >= -0.01,
        "disaster_macro_over_majority_at_least_0p10": mean(classification) >= 0.10,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "mean_delta_bo_damage_macro_f1": mean(damage),
        "mean_delta_event_macro_bo_f1": mean(event),
        "mean_delta_worst_event_bo_f1": mean(worst),
        "mean_disaster_macro_over_majority": mean(classification),
        "next_action": "run_M2" if all(checks.values()) else "stop_and_report_M1_failure",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Stage-2 metadata multitask validation runs")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260814)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir)
    runs: dict[str, list[dict[str, Any]]] = {}
    for condition in ("M0", "M1", "M2"):
        condition_runs = []
        for seed in SEEDS:
            path = latest_run(run_root, condition, seed)
            if path is not None:
                condition_runs.append(load_run(path, condition, seed))
        runs[condition] = condition_runs
    if len(runs["M0"]) != 3 or len(runs["M1"]) != 3:
        raise SystemExit("M0 and M1 must each have complete validation runs for all three seeds")

    m0_m1 = paired_rows(runs["M0"], runs["M1"])
    screening = screen_m1(m0_m1)
    summary: dict[str, Any] = {
        "protocol": "stage2_metadata_multitask_v1",
        "test_used": False,
        "conditions": {
            condition: {
                metric: aggregate([float(row[metric]) for row in values])
                for metric in (
                    "bo_damage_macro_f1",
                    "event_macro_bo_f1",
                    "event_macro_bo_damage_f1",
                    "worst_event_bo_f1",
                    "disaster_present_macro_f1",
                )
            }
            for condition, values in runs.items()
            if values
        },
        "M0_vs_M1": {
            "paired_rows": m0_m1,
            "screening": screening,
            "hierarchical_bootstrap": hierarchical_bootstrap(
                runs["M0"],
                runs["M1"],
                iterations=args.bootstrap_iterations,
                seed=args.bootstrap_seed,
            ),
        },
    }
    if len(runs["M2"]) == 3:
        m2_rows = paired_rows(runs["M2"], runs["M1"])
        semantic_damage = [float(row["delta_bo_damage_macro_f1"]) for row in m2_rows]
        semantic_event = [float(row["delta_event_macro_bo_f1"]) for row in m2_rows]
        semantic_checks = {
            "damage_mean_at_least_0p01": mean(semantic_damage) >= 0.01,
            "damage_positive_at_least_2_seeds": sum(value > 0 for value in semantic_damage) >= 2,
            "event_mean_at_least_0p01": mean(semantic_event) >= 0.01,
            "event_positive_at_least_2_seeds": sum(value > 0 for value in semantic_event) >= 2,
        }
        summary["M2_vs_M1"] = {
            "paired_rows": m2_rows,
            "semantic_attribution_passed": all(semantic_checks.values()),
            "checks": semantic_checks,
            "hierarchical_bootstrap": hierarchical_bootstrap(
                runs["M2"],
                runs["M1"],
                iterations=args.bootstrap_iterations,
                seed=args.bootstrap_seed,
            ),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "runs.csv", [public_row(row) for values in runs.values() for row in values])
    write_csv(output_dir / "m0_vs_m1_paired_deltas.csv", m0_m1)
    write_json(output_dir / "summary.json", summary)

    lines = [
        "# Metadata-aware multi-task v1 validation result",
        "",
        "> Test was not read or evaluated. All selection and conclusions are validation-only.",
        "",
        "## Three-seed means",
        "",
        "| condition | BO damage macro F1 | event macro BO F1 | event macro damage F1 | worst-event BO F1 | disaster macro F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in ("M0", "M1", "M2"):
        if condition not in summary["conditions"]:
            continue
        values = summary["conditions"][condition]
        lines.append(
            f"| {condition} | {values['bo_damage_macro_f1']['mean']:.6f} | "
            f"{values['event_macro_bo_f1']['mean']:.6f} | "
            f"{values['event_macro_bo_damage_f1']['mean']:.6f} | "
            f"{values['worst_event_bo_f1']['mean']:.6f} | "
            f"{values['disaster_present_macro_f1']['mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## M0 versus M1 screening",
            "",
            f"- passed: `{str(screening['passed']).lower()}`",
            f"- mean damage-macro delta: `{screening['mean_delta_bo_damage_macro_f1']:+.6f}`",
            f"- mean event-macro delta: `{screening['mean_delta_event_macro_bo_f1']:+.6f}`",
            f"- mean worst-event delta: `{screening['mean_delta_worst_event_bo_f1']:+.6f}`",
            f"- mean disaster macro-F1 over majority: `{screening['mean_disaster_macro_over_majority']:+.6f}`",
            f"- next action: `{screening['next_action']}`",
            "",
            "The four validation events are insufficient for a formal generalization claim; bootstrap intervals are descriptive.",
        ]
    )
    (output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "screening": screening}, sort_keys=True))


if __name__ == "__main__":
    main()
