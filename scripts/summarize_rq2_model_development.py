#!/usr/bin/env python3
"""Apply the frozen RQ2 pilot or full-development gates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


CANDIDATES = ("R2_A", "R2_B", "R2_C")
COST_RANK = {"R2_C": 0, "R2_B": 1, "R2_A": 2}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def row_metrics(row: dict[str, str]) -> dict[str, float]:
    present = [
        name
        for name in ("damaged", "destroyed")
        if float(row[f"building_only_support_{name}"]) > 0
    ]
    return {
        "damage": (
            float(np.mean([float(row[f"building_only_f1_{name}"]) for name in present]))
            if present
            else 0.0
        ),
        "grade": float(row["building_only_macro_f1_3class"]),
        "damaged": float(row["building_only_f1_damaged"]),
        "destroyed": float(row["building_only_f1_destroyed"]),
        "damaged_present": float(row["building_only_support_damaged"]) > 0,
        "destroyed_present": float(row["building_only_support_destroyed"]) > 0,
    }


def load_event_rows(
    root: Path,
    *,
    candidate: str | None,
    condition: str,
    seeds: tuple[int, ...],
) -> dict[tuple[int, str], dict[str, float]]:
    values = {}
    for outer in range(7):
        for seed in seeds:
            if candidate is None:
                path = root / "evaluation" / f"outer_{outer}" / condition / f"seed_{seed}"
            else:
                path = (
                    root
                    / "candidates"
                    / candidate
                    / "evaluation"
                    / f"outer_{outer}"
                    / condition
                    / f"seed_{seed}"
                )
            for row in read_csv(path / "per_event_metrics.csv"):
                key = (seed, row["event_id"])
                if key in values:
                    raise ValueError(f"duplicate event row for {candidate=} {condition=} {key=}")
                values[key] = row_metrics(row)
    return values


def summarize_candidate(
    rq2_root: Path,
    baseline_root: Path,
    candidate: str,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    baseline_c2 = load_event_rows(
        baseline_root, candidate=None, condition="C2", seeds=seeds
    )
    new_c2 = load_event_rows(rq2_root, candidate=candidate, condition="C2", seeds=seeds)
    new_c3 = load_event_rows(rq2_root, candidate=candidate, condition="C3", seeds=seeds)
    if set(baseline_c2) != set(new_c2) or set(new_c2) != set(new_c3):
        raise ValueError(f"unpaired RQ2 rows for {candidate}")

    paired = []
    for seed, event_id in sorted(new_c2):
        baseline = baseline_c2[(seed, event_id)]
        paired_value = new_c2[(seed, event_id)]
        deranged = new_c3[(seed, event_id)]
        paired.append(
            {
                "seed": seed,
                "event_id": event_id,
                "candidate_minus_f0_damage": paired_value["damage"] - baseline["damage"],
                "candidate_minus_f0_grade": paired_value["grade"] - baseline["grade"],
                "candidate_minus_f0_damaged": paired_value["damaged"] - baseline["damaged"],
                "candidate_minus_f0_destroyed": paired_value["destroyed"] - baseline["destroyed"],
                "candidate_c2_minus_c3_damage": paired_value["damage"] - deranged["damage"],
                "candidate_c2_minus_c3_grade": paired_value["grade"] - deranged["grade"],
                "damaged_present": paired_value["damaged_present"],
                "destroyed_present": paired_value["destroyed_present"],
            }
        )
    events = sorted({row["event_id"] for row in paired})
    if len(events) != 14:
        raise ValueError(f"expected 14 development events, found {len(events)}")
    seed_baseline_means = {
        seed: float(
            np.mean(
                [
                    row["candidate_minus_f0_damage"]
                    for row in paired
                    if row["seed"] == seed
                ]
            )
        )
        for seed in seeds
    }
    seed_correspondence_means = {
        seed: float(
            np.mean(
                [
                    row["candidate_c2_minus_c3_damage"]
                    for row in paired
                    if row["seed"] == seed
                ]
            )
        )
        for seed in seeds
    }
    event_baseline_means = {
        event: float(
            np.mean(
                [
                    row["candidate_minus_f0_damage"]
                    for row in paired
                    if row["event_id"] == event
                ]
            )
        )
        for event in events
    }
    summary = {
        "candidate": candidate,
        "seeds": list(seeds),
        "candidate_minus_f0_event_macro_damage": float(
            np.mean([row["candidate_minus_f0_damage"] for row in paired])
        ),
        "candidate_minus_f0_positive_event_count": sum(
            value > 0 for value in event_baseline_means.values()
        ),
        "candidate_minus_f0_positive_seed_count": sum(
            value > 0 for value in seed_baseline_means.values()
        ),
        "candidate_minus_f0_damaged": float(
            np.mean(
                [
                    row["candidate_minus_f0_damaged"]
                    for row in paired
                    if row["damaged_present"]
                ]
            )
        ),
        "candidate_minus_f0_destroyed": float(
            np.mean(
                [
                    row["candidate_minus_f0_destroyed"]
                    for row in paired
                    if row["destroyed_present"]
                ]
            )
        ),
        "candidate_c2_minus_c3_event_macro_damage": float(
            np.mean([row["candidate_c2_minus_c3_damage"] for row in paired])
        ),
        "candidate_c2_minus_c3_positive_seed_count": sum(
            value > 0 for value in seed_correspondence_means.values()
        ),
        "worst_event_candidate_minus_f0_grade": min(
            float(
                np.mean(
                    [
                        row["candidate_minus_f0_grade"]
                        for row in paired
                        if row["event_id"] == event
                    ]
                )
            )
            for event in events
        ),
        "worst_event_candidate_c2_minus_c3_grade": min(
            float(
                np.mean(
                    [
                        row["candidate_c2_minus_c3_grade"]
                        for row in paired
                        if row["event_id"] == event
                    ]
                )
            )
            for event in events
        ),
        "seed_candidate_minus_f0_damage": seed_baseline_means,
        "seed_candidate_c2_minus_c3_damage": seed_correspondence_means,
        "event_candidate_minus_f0_damage": event_baseline_means,
        "paired_rows": paired,
    }
    return summary


def pilot_gates(summary: dict[str, Any]) -> dict[str, bool]:
    return {
        "candidate_minus_f0_event_macro_damage_gte_0p01": summary[
            "candidate_minus_f0_event_macro_damage"
        ]
        >= 0.01,
        "candidate_minus_f0_positive_events_gte_8": summary[
            "candidate_minus_f0_positive_event_count"
        ]
        >= 8,
        "candidate_minus_f0_damaged_gte_minus_0p005": summary[
            "candidate_minus_f0_damaged"
        ]
        >= -0.005,
        "candidate_minus_f0_destroyed_gte_minus_0p01": summary[
            "candidate_minus_f0_destroyed"
        ]
        >= -0.01,
        "candidate_c2_minus_c3_event_macro_damage_gte_0p02": summary[
            "candidate_c2_minus_c3_event_macro_damage"
        ]
        >= 0.02,
        "worst_candidate_minus_f0_grade_gte_minus_0p03": summary[
            "worst_event_candidate_minus_f0_grade"
        ]
        >= -0.03,
        "worst_candidate_c2_minus_c3_grade_gte_minus_0p03": summary[
            "worst_event_candidate_c2_minus_c3_grade"
        ]
        >= -0.03,
    }


def full_gates(summary: dict[str, Any]) -> dict[str, bool]:
    return {
        "candidate_minus_f0_event_macro_damage_gte_0p02": summary[
            "candidate_minus_f0_event_macro_damage"
        ]
        >= 0.02,
        "candidate_minus_f0_positive_seeds_gte_2": summary[
            "candidate_minus_f0_positive_seed_count"
        ]
        >= 2,
        "candidate_minus_f0_positive_events_gte_10": summary[
            "candidate_minus_f0_positive_event_count"
        ]
        >= 10,
        "worst_candidate_minus_f0_grade_gte_minus_0p01": summary[
            "worst_event_candidate_minus_f0_grade"
        ]
        >= -0.01,
        "worst_candidate_c2_minus_c3_grade_gte_minus_0p01": summary[
            "worst_event_candidate_c2_minus_c3_grade"
        ]
        >= -0.01,
        "candidate_minus_f0_damaged_gte_0": summary["candidate_minus_f0_damaged"] >= 0,
        "candidate_minus_f0_destroyed_gte_0": summary["candidate_minus_f0_destroyed"] >= 0,
        "candidate_c2_minus_c3_event_macro_damage_gte_0p02": summary[
            "candidate_c2_minus_c3_event_macro_damage"
        ]
        >= 0.02,
        "candidate_c2_minus_c3_positive_seeds_gte_2": summary[
            "candidate_c2_minus_c3_positive_seed_count"
        ]
        >= 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "full"), required=True)
    parser.add_argument("--rq2-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--candidate", choices=CANDIDATES)
    args = parser.parse_args()
    rq2_root = args.rq2_root.resolve()
    baseline_root = args.baseline_root.resolve()
    output_root = args.output_root.resolve()

    if args.phase == "pilot":
        if args.candidate is not None:
            raise ValueError("pilot summary always evaluates all candidates")
        seeds = (42,)
        summaries = []
        for candidate in CANDIDATES:
            summary = summarize_candidate(rq2_root, baseline_root, candidate, seeds)
            summary["gates"] = pilot_gates(summary)
            summary["eligible"] = all(summary["gates"].values())
            summaries.append(summary)
        eligible = [summary for summary in summaries if summary["eligible"]]
        eligible.sort(
            key=lambda summary: (
                summary["candidate_minus_f0_event_macro_damage"],
                min(
                    summary["worst_event_candidate_minus_f0_grade"],
                    summary["worst_event_candidate_c2_minus_c3_grade"],
                ),
                summary["candidate_minus_f0_damaged"],
                -COST_RANK[summary["candidate"]],
            ),
            reverse=True,
        )
        selected = eligible[0]["candidate"] if eligible else None
        payload = {
            "protocol_id": "rq2_model_development_v1.0",
            "phase": "pilot",
            "evidence_scope": "14_exposed_development_events",
            "status": "candidate_selected" if selected else "stopped_no_eligible_candidate",
            "selected_candidate": selected,
            "backfill_if_full_fails": False,
            "candidates": summaries,
        }
    else:
        if args.candidate is None:
            raise ValueError("full summary requires --candidate")
        summary = summarize_candidate(
            rq2_root, baseline_root, args.candidate, (42, 3407, 2026)
        )
        summary["gates"] = full_gates(summary)
        summary["passed"] = all(summary["gates"].values())
        payload = {
            "protocol_id": "rq2_model_development_v1.0",
            "phase": "full",
            "evidence_scope": "14_exposed_development_events_no_blind_confirmation",
            "status": (
                "development_candidate_frozen"
                if summary["passed"]
                else "stopped_full_gate_failed"
            ),
            "selected_candidate": args.candidate,
            "passed": summary["passed"],
            "backfill_allowed": False,
            "candidate": summary,
        }
    write_json(output_root / "summary.json", payload)
    print(json.dumps({key: payload[key] for key in ("phase", "status")}, sort_keys=True))


if __name__ == "__main__":
    main()
