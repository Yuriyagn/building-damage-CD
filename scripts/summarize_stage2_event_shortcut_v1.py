#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.stage2.metrics_v2 import summarize_event_generalization  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Stage-2 event-shortcut diagnostics")
    parser.add_argument(
        "--root",
        default="outputs/stage2/event_shortcut_v1",
        help="experiment output root relative to the repository",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_evaluation(experiment_dir: Path) -> Path:
    candidates = sorted(experiment_dir.glob("seed_42/run_*/val_best_grade"))
    complete = [path for path in candidates if (path / "event_generalization.json").is_file()]
    if not complete:
        raise FileNotFoundError(f"no complete validation evaluation under {experiment_dir}")
    return complete[-1]


def compact_row(label: str, evaluation: Path) -> dict[str, Any]:
    metrics = load_json(evaluation / "metrics.json")
    event = load_json(evaluation / "event_generalization.json")
    return {
        "experiment": label,
        "evaluation": str(evaluation.resolve()),
        "building_only_macro_f1_3class": metrics["building_only_macro_f1_3class"],
        "building_only_f1_intact": metrics["building_only_f1_intact"],
        "building_only_f1_damaged": metrics["building_only_f1_damaged"],
        "building_only_f1_destroyed": metrics["building_only_f1_destroyed"],
        "building_only_damage_macro_f1": metrics["building_only_damage_macro_f1"],
        "building_only_damage_binary_f1": metrics["building_only_damage_binary_f1"],
        **event,
    }


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        compact_row("D1_pre_only", latest_evaluation(root / "D1_pre_only")),
        compact_row("D2_background_only", latest_evaluation(root / "D2_background_only")),
    ]

    d3_dir = root / "D3_event_prior" / "run_20260803_final"
    for source in read_csv(d3_dir / "summary.csv"):
        rows.append({"experiment": f"D3_{source['strategy']}", "evaluation": str(d3_dir.resolve()), **source})

    d4_rows: list[dict[str, Any]] = []
    for experiment_dir in sorted(root.glob("D4_LOEO_*")):
        evaluation = latest_evaluation(experiment_dir)
        row = compact_row(experiment_dir.name, evaluation)
        rows.append(row)
        d4_rows.extend(read_csv(evaluation / "per_event_metrics.csv"))

    d4_summary = summarize_event_generalization(d4_rows)
    rows.append(
        {
            "experiment": "D4_LOEO_event_macro",
            "evaluation": "combined held-out events; no pixel-weighted global F1",
            **d4_summary,
        }
    )

    all_keys = sorted({key for row in rows for key in row})
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps({"test_used": False, "rows": rows}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir.resolve()), "row_count": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
