#!/usr/bin/env python3
"""Summarize the frozen R0/R4 event-group comparison from completed runs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


SEEDS = (42, 3407, 2026)
VARIANTS = {
    "r0": "R0_event_group_predicted_prior",
    "r4": "R4_bright_event_group_predicted_prior",
}
METRICS = (
    "val_building_only_macro_f1_3class",
    "val_building_only_damage_macro_f1",
    "val_building_only_f1_damaged",
    "val_building_only_f1_destroyed",
    "val_building_only_damage_binary_f1",
    "val_predicted_gate_macro_f1_4class",
)


def latest_completed(seed_dir: Path) -> Path | None:
    candidates = sorted(seed_dir.glob("run_*/completed.json"))
    return candidates[-1] if candidates else None


def selected_row(run_dir: Path, epoch: int) -> dict[str, str]:
    with (run_dir / "metrics_history.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if int(row["epoch"]) == epoch:
            return row
    raise RuntimeError(f"selected epoch {epoch} absent from {run_dir}")


def mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    records: list[dict[str, object]] = []
    missing: list[str] = []
    for variant, dirname in VARIANTS.items():
        for seed in SEEDS:
            completed_path = latest_completed(args.root / dirname / f"seed_{seed}")
            if completed_path is None:
                missing.append(f"{variant}:seed_{seed}")
                continue
            completed = json.loads(completed_path.read_text(encoding="utf-8"))
            if completed.get("status") != "completed":
                missing.append(f"{variant}:seed_{seed}:status={completed.get('status')}")
                continue
            best = completed["best_metrics"]["best_bo_grade_macro_f1.pth"]
            epoch = int(best["epoch"])
            row = selected_row(completed_path.parent, epoch)
            record: dict[str, object] = {
                "variant": variant,
                "seed": seed,
                "run_dir": str(completed_path.parent.resolve()),
                "selected_checkpoint": "best_bo_grade_macro_f1.pth",
                "selected_epoch": epoch,
                "epochs_completed": int(completed["epochs_completed"]),
                "stopped_early": bool(completed["stopped_early"]),
            }
            record.update({metric: float(row[metric]) for metric in METRICS})
            records.append(record)

    if missing and not args.allow_partial:
        raise SystemExit("incomplete comparison: " + ", ".join(missing))

    aggregates: dict[str, object] = {}
    for variant in VARIANTS:
        subset = [r for r in records if r["variant"] == variant]
        aggregates[variant] = {
            "n": len(subset),
            "metrics": {
                metric: mean_std([float(r[metric]) for r in subset]) if subset else None
                for metric in METRICS
            },
        }

    paired: list[dict[str, object]] = []
    for seed in SEEDS:
        by_variant = {
            str(r["variant"]): r for r in records if int(r["seed"]) == seed
        }
        if set(by_variant) != set(VARIANTS):
            continue
        paired.append({
            "seed": seed,
            **{
                f"delta_r4_minus_r0__{metric}": (
                    float(by_variant["r4"][metric]) - float(by_variant["r0"][metric])
                )
                for metric in METRICS
            },
        })

    payload = {
        "protocol": "stage2_v2_event_group_r0_r4_v1",
        "selection_rule": "best validation BO three-grade macro F1 checkpoint",
        "test_used_for_selection": False,
        "complete": not missing,
        "missing": missing,
        "records": records,
        "aggregates": aggregates,
        "paired_deltas": paired,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"complete": not missing, "runs": len(records), "out": str(args.out)}))


if __name__ == "__main__":
    main()
