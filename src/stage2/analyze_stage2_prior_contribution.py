#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_EXPERIMENTS = {
    "S0": "B1_S0_no_prior_unet_resnet34/test",
    "S1": "B1_S1_oracle_prior_unet_resnet34/test",
    "S2": "B1_S2_predicted_prior_unet_resnet34/test",
    "S3_oracle": "B1_S3_oracle_prior_only_unet_resnet34/test",
    "S3_predicted": "B1_S3_predicted_prior_only_unet_resnet34/test",
}

METRICS = [
    "miou_4class",
    "iou_intact",
    "iou_damaged",
    "iou_destroyed",
    "f1_intact",
    "f1_damaged",
    "f1_destroyed",
    "damage_binary_f1",
    "damage_macro_f1",
    "building_only_miou_3class",
    "building_only_f1_intact",
    "building_only_f1_damaged",
    "building_only_f1_destroyed",
    "building_only_damage_binary_f1",
    "building_only_damage_macro_f1",
]

PAIRS = [
    ("S1_minus_S0", "S1", "S0", "oracle prior gain over no prior"),
    ("S2_minus_S0", "S2", "S0", "predicted prior gain over no prior"),
    ("S1_minus_S2", "S1", "S2", "Stage-1 prior error gap"),
    ("S1_minus_S3_oracle", "S1", "S3_oracle", "SAR contribution beyond oracle prior-only"),
    ("S2_minus_S3_predicted", "S2", "S3_predicted", "SAR contribution beyond predicted prior-only"),
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def locate_metrics(results_root: Path, rel: str) -> Path:
    candidates = [
        results_root / rel / "metrics.json",
        results_root / rel,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"metrics not found for {rel}; tried {candidates}")


def load_experiment_metrics(results_root: Path, mapping: dict[str, str]) -> dict[str, dict[str, Any]]:
    out = {}
    for key, rel in mapping.items():
        path = locate_metrics(results_root, rel)
        out[key] = read_json(path)
        out[key]["_metrics_path"] = str(path)
    return out


def compute_metric_deltas(metrics_by_exp: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for pair_name, left, right, meaning in PAIRS:
        if left not in metrics_by_exp or right not in metrics_by_exp:
            continue
        for metric in METRICS:
            if metric not in metrics_by_exp[left] or metric not in metrics_by_exp[right]:
                continue
            left_value = float(metrics_by_exp[left][metric])
            right_value = float(metrics_by_exp[right][metric])
            rows.append(
                {
                    "comparison": pair_name,
                    "meaning": meaning,
                    "metric": metric,
                    "left_experiment": left,
                    "right_experiment": right,
                    "left_value": left_value,
                    "right_value": right_value,
                    "delta": left_value - right_value,
                }
            )
    return rows


def group_delta_rows(results_root: Path, mapping: dict[str, str], group_file: str, group_key: str) -> list[dict[str, Any]]:
    tables: dict[str, dict[str, dict[str, str]]] = {}
    for exp, rel in mapping.items():
        base = locate_metrics(results_root, rel).parent
        rows = read_csv(base / group_file)
        tables[exp] = {str(row.get(group_key, "")): row for row in rows}

    out = []
    for pair_name, left, right, meaning in PAIRS:
        if left not in tables or right not in tables:
            continue
        groups = sorted(set(tables[left]) & set(tables[right]))
        for group in groups:
            for metric in METRICS:
                if metric not in tables[left][group] or metric not in tables[right][group]:
                    continue
                try:
                    left_value = float(tables[left][group][metric])
                    right_value = float(tables[right][group][metric])
                except ValueError:
                    continue
                out.append(
                    {
                        group_key: group,
                        "comparison": pair_name,
                        "meaning": meaning,
                        "metric": metric,
                        "left_experiment": left,
                        "right_experiment": right,
                        "left_value": left_value,
                        "right_value": right_value,
                        "delta": left_value - right_value,
                    }
                )
    return out


def write_markdown(path: Path, deltas: list[dict[str, Any]]) -> None:
    priority = {
        "S1_minus_S0": "building prior theoretical gain",
        "S2_minus_S0": "building prior practical gain",
        "S1_minus_S2": "Stage-1 error gap",
        "S1_minus_S3_oracle": "SAR gain over oracle prior-only",
        "S2_minus_S3_predicted": "SAR gain over predicted prior-only",
    }
    rows = [
        row for row in deltas
        if row["metric"] in {"building_only_damage_macro_f1", "building_only_damage_binary_f1", "damage_macro_f1", "miou_4class"}
    ]
    lines = ["# Stage-2 Prior Contribution Analysis", ""]
    for comparison, title in priority.items():
        lines.append(f"## {comparison}: {title}")
        lines.append("")
        lines.append("| metric | left | right | delta |")
        lines.append("| --- | ---: | ---: | ---: |")
        for row in rows:
            if row["comparison"] != comparison:
                continue
            lines.append(
                f"| {row['metric']} | {row['left_value']:.6f} | {row['right_value']:.6f} | {row['delta']:.6f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation Rules",
            "",
            "- S1 > S0: building prior helps.",
            "- S2 close to S1: Stage-1 prior is good enough for the two-stage pipeline.",
            "- S2 far below S1: Stage-1 prior error is a major bottleneck.",
            "- S1 close to S3_oracle: SAR adds little beyond the prior, so prior or dataset bias explains much of the result.",
            "- High mIoU with low damaged/destroyed F1 means background/intact dominate the score.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_mapping(values: list[str] | None) -> dict[str, str]:
    if not values:
        return dict(DEFAULT_EXPERIMENTS)
    mapping = {}
    for value in values:
        key, rel = value.split("=", 1)
        mapping[key] = rel
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--experiment", action="append", help="Override experiment mapping as KEY=relative/path/to/test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    mapping = parse_mapping(args.experiment)
    metrics_by_exp = load_experiment_metrics(results_root, mapping)
    metric_deltas = compute_metric_deltas(metrics_by_exp)
    write_csv(out_dir / "prior_contribution_metric_deltas.csv", metric_deltas)
    write_csv(
        out_dir / "prior_contribution_per_disaster_deltas.csv",
        group_delta_rows(results_root, mapping, "per_disaster_metrics.csv", "disaster_type"),
    )
    write_csv(
        out_dir / "prior_contribution_per_region_deltas.csv",
        group_delta_rows(results_root, mapping, "per_region_metrics.csv", "country_or_region"),
    )
    write_csv(
        out_dir / "prior_contribution_per_event_deltas.csv",
        group_delta_rows(results_root, mapping, "per_event_metrics.csv", "event_id"),
    )
    write_markdown(out_dir / "prior_contribution_report.md", metric_deltas)
    print(json.dumps({"out_dir": str(out_dir), "experiments": mapping}, sort_keys=True))


if __name__ == "__main__":
    main()

