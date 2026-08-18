#!/usr/bin/env python3
"""Summarize RQ1 nested-CV paired effects and apply frozen development gates."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SEEDS = (42, 3407, 2026)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def f1s(confusion: np.ndarray) -> np.ndarray:
    result = np.zeros(3, dtype=np.float64)
    for index in range(3):
        tp = confusion[index, index]
        fp = confusion[:, index].sum() - tp
        fn = confusion[index, :].sum() - tp
        result[index] = 2 * tp / max(2 * tp + fp + fn, 1)
    return result


def damage_macro(confusion: np.ndarray) -> float:
    values = f1s(confusion)
    present = confusion.sum(axis=1) > 0
    damage_present = present[1:3]
    return float(values[1:3][damage_present].mean()) if damage_present.any() else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    args = parser.parse_args()
    stage2 = args.stage2_root.resolve()
    output = args.output_root.resolve()
    event_rows: dict[tuple[str, int, str], dict[str, str]] = {}
    sample_confusions: dict[tuple[str, int, str], dict[str, np.ndarray]] = {}
    event_to_outer: dict[str, int] = {}
    for outer in range(7):
        for condition in ("C1", "C2", "C3"):
            for seed in SEEDS:
                eval_dir = stage2 / "evaluation" / f"outer_{outer}" / condition / f"seed_{seed}"
                for row in read_csv(eval_dir / "per_event_metrics.csv"):
                    event = row["event_id"]
                    event_rows[(condition, seed, event)] = row
                    event_to_outer[event] = outer
                grouped: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
                for row in read_csv(eval_dir / "sample_metrics.csv"):
                    grouped[row["event_id"]][row["id"]] = np.asarray(json.loads(row["bo_confusion_3x3"]), dtype=np.int64).reshape(3, 3)
                for event, by_id in grouped.items():
                    sample_confusions[(condition, seed, event)] = by_id
    events = sorted(event_to_outer)
    if len(events) != 14:
        raise ValueError(f"expected 14 evaluated events, found {len(events)}")

    paired = []
    for event in events:
        for seed in SEEDS:
            values: dict[str, dict[str, float]] = {}
            for condition in ("C1", "C2", "C3"):
                row = event_rows[(condition, seed, event)]
                present = [name for name in ("damaged", "destroyed") if float(row[f"building_only_support_{name}"]) > 0]
                values[condition] = {
                    "damage": float(np.mean([float(row[f"building_only_f1_{name}"]) for name in present])) if present else 0.0,
                    "grade": float(row["building_only_macro_f1_3class"]),
                    "damaged": float(row["building_only_f1_damaged"]),
                    "destroyed": float(row["building_only_f1_destroyed"]),
                    "damaged_present": float(row["building_only_support_damaged"]) > 0,
                    "destroyed_present": float(row["building_only_support_destroyed"]) > 0,
                }
            paired.append({
                "event_id": event, "outer_fold": event_to_outer[event], "seed": seed,
                "C2_minus_C3_damage": values["C2"]["damage"] - values["C3"]["damage"],
                "C2_minus_C1_damage": values["C2"]["damage"] - values["C1"]["damage"],
                "C2_minus_C3_grade": values["C2"]["grade"] - values["C3"]["grade"],
                "C2_minus_C3_damaged": values["C2"]["damaged"] - values["C3"]["damaged"],
                "C2_minus_C3_destroyed": values["C2"]["destroyed"] - values["C3"]["destroyed"],
                "damaged_present": values["C2"]["damaged_present"],
                "destroyed_present": values["C2"]["destroyed_present"],
            })

    primary = float(np.mean([row["C2_minus_C3_damage"] for row in paired]))
    secondary = float(np.mean([row["C2_minus_C1_damage"] for row in paired]))
    seed_means = {seed: float(np.mean([row["C2_minus_C3_damage"] for row in paired if row["seed"] == seed])) for seed in SEEDS}
    event_means = {event: float(np.mean([row["C2_minus_C3_damage"] for row in paired if row["event_id"] == event])) for event in events}
    damaged_delta = float(np.mean([row["C2_minus_C3_damaged"] for row in paired if row["damaged_present"]]))
    destroyed_delta = float(np.mean([row["C2_minus_C3_destroyed"] for row in paired if row["destroyed_present"]]))
    worst_event_grade_delta = min(float(np.mean([row["C2_minus_C3_grade"] for row in paired if row["event_id"] == event])) for event in events)

    rng = np.random.default_rng(20260815)
    boot = np.empty(args.bootstrap_iterations, dtype=np.float64)
    for iteration in range(args.bootstrap_iterations):
        chosen_events = rng.choice(events, size=len(events), replace=True)
        deltas = []
        for event in chosen_events:
            for seed in SEEDS:
                c2 = sample_confusions[("C2", seed, str(event))]
                c3 = sample_confusions[("C3", seed, str(event))]
                ids = sorted(set(c2) & set(c3))
                if set(c2) != set(c3):
                    raise ValueError(f"unpaired sample IDs for event={event} seed={seed}")
                chosen = rng.integers(0, len(ids), size=len(ids))
                conf2 = sum((c2[ids[int(index)]] for index in chosen), start=np.zeros((3, 3), dtype=np.int64))
                conf3 = sum((c3[ids[int(index)]] for index in chosen), start=np.zeros((3, 3), dtype=np.int64))
                deltas.append(damage_macro(conf2) - damage_macro(conf3))
        boot[iteration] = float(np.mean(deltas))

    gates = {
        "mean_C2_minus_C3_gte_0p02": primary >= 0.02,
        "positive_at_least_2_of_3_seeds": sum(value > 0 for value in seed_means.values()) >= 2,
        "positive_at_least_9_of_14_events": sum(value > 0 for value in event_means.values()) >= 9,
        "mean_C2_minus_C1_gte_0p01": secondary >= 0.01,
        "damaged_delta_gte_minus_0p01": damaged_delta >= -0.01,
        "destroyed_delta_gte_minus_0p01": destroyed_delta >= -0.01,
        "worst_event_grade_delta_gte_minus_0p01": worst_event_grade_delta >= -0.01,
    }
    passed = all(gates.values())
    summary = {
        "status": "development_supported" if passed else "failed",
        "evidence_scope": "nested_cv_development_only_no_blind_confirmation",
        "primary_mean_C2_minus_C3_event_damage_f1": primary,
        "primary_hierarchical_paired_bootstrap_ci95": [float(value) for value in np.quantile(boot, (0.025, 0.975))],
        "secondary_mean_C2_minus_C1_event_damage_f1": secondary,
        "damaged_delta": damaged_delta, "destroyed_delta": destroyed_delta,
        "worst_event_grade_delta": worst_event_grade_delta,
        "seed_means": seed_means, "event_means": event_means,
        "gates": gates, "passed": passed,
        "bootstrap": {"iterations": args.bootstrap_iterations, "units": "events then paired samples; seeds retained within events", "descriptive_only": True},
        "paired_rows": paired,
    }
    write_json(output / "summary.json", summary)
    lines = [
        "# RQ1 Nested-CV Development Result", "",
        f"Status: **{summary['status']}**", "",
        "This is development-only nested-CV evidence; no independent blind confirmation exists.", "",
        f"- Mean C2-C3 event damage F1: `{primary:+.5f}`",
        f"- 95% hierarchical paired bootstrap CI: `{summary['primary_hierarchical_paired_bootstrap_ci95']}`",
        f"- Mean C2-C1 event damage F1: `{secondary:+.5f}`", "",
        "## Frozen gates", "",
    ] + [f"- {'PASS' if value else 'FAIL'}: {name}" for name, value in gates.items()]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "passed": passed}, sort_keys=True))


if __name__ == "__main__":
    main()
