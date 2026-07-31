#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the Stage-2 fixed-batch overfit technical gate"
    )
    parser.add_argument("--run", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--out")
    return parser.parse_args()


def summarize_losses(losses: Iterable[float], threshold: float) -> dict[str, Any]:
    values = [float(value) for value in losses]
    if not values:
        raise ValueError("overfit history has no train_loss values")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("overfit history contains a non-finite train_loss")
    first = values[0]
    if first <= 0.0:
        raise ValueError(f"first train_loss must be positive, got {first}")
    best = min(values)
    last = values[-1]
    best_reduction = (first - best) / first
    last_reduction = (first - last) / first
    passed = best_reduction >= float(threshold)
    return {
        "status": "pass" if passed else "fail",
        "epochs_observed": len(values),
        "first_train_loss": first,
        "best_train_loss": best,
        "last_train_loss": last,
        "best_relative_loss_reduction": best_reduction,
        "last_relative_loss_reduction": last_reduction,
        "threshold": float(threshold),
        "check": "best_relative_loss_reduction_gte_threshold",
    }


def read_train_losses(path: Path) -> list[float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    try:
        return [float(row["train_loss"]) for row in rows]
    except KeyError as exc:
        raise ValueError(f"missing train_loss column in {path}") from exc


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    protocol_path = Path(args.protocol)
    if not protocol_path.is_absolute():
        protocol_path = repo_root / protocol_path
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    technical = dict(dict(protocol).get("advancement_gate", {})).get("technical", {})
    threshold = float(technical["overfit_min_relative_loss_reduction"])

    run_dir = Path(args.run).resolve()
    history_path = run_dir / "metrics_history.csv"
    if not history_path.is_file():
        raise FileNotFoundError(f"missing overfit history: {history_path}")
    result = {
        "protocol_id": protocol.get("protocol_id"),
        "run": str(run_dir),
        "history": str(history_path),
        "selection_split": "technical_overfit_only",
        "test_used": False,
        **summarize_losses(read_train_losses(history_path), threshold),
    }

    output_path = Path(args.out).resolve() if args.out else run_dir / "overfit_gate.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
