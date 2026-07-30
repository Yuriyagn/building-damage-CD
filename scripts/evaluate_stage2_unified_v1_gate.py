#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


PRIMARY_METRIC = "building_only_macro_f1_3class"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the unified-v1 seed-42 advancement gate")
    parser.add_argument("--paired-run", required=True)
    parser.add_argument("--shuffled-run", required=True)
    parser.add_argument(
        "--protocol",
        default="configs/stage2_unified_v1/protocol.yaml",
    )
    parser.add_argument(
        "--out",
        default="outputs/stage2/unified_v1/gates/seed42_gate.json",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def resolve_metrics(run_dir: Path) -> Path:
    path = run_dir / "val_best_grade" / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing validation metrics: {path}; evaluate best_bo_grade_macro_f1.pth on val first"
        )
    return path


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    protocol_path = Path(args.protocol)
    if not protocol_path.is_absolute():
        protocol_path = repo_root / protocol_path
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    gate_cfg = dict(dict(protocol).get("advancement_gate", {})).get("seed42", {})

    paired_run = Path(args.paired_run).resolve()
    shuffled_run = Path(args.shuffled_run).resolve()
    paired_metrics_path = resolve_metrics(paired_run)
    shuffled_metrics_path = resolve_metrics(shuffled_run)
    paired_metrics = load_json(paired_metrics_path)
    shuffled_metrics = load_json(shuffled_metrics_path)

    paired_value = float(paired_metrics[PRIMARY_METRIC])
    shuffled_value = float(shuffled_metrics[PRIMARY_METRIC])
    reference = float(gate_cfg["reference_e2_argmax_bo_macro_f1"])
    required_margin = float(gate_cfg["minimum_paired_minus_shuffled_bo_macro_f1"])
    margin = paired_value - shuffled_value
    checks = {
        "paired_meets_or_beats_e2_argmax": {
            "passed": paired_value >= reference,
            "value": paired_value,
            "threshold": reference,
        },
        "paired_beats_shuffled": {
            "passed": margin >= required_margin,
            "value": margin,
            "threshold": required_margin,
        },
    }
    require_both = bool(gate_cfg.get("require_both", True))
    passed_values = [bool(item["passed"]) for item in checks.values()]
    passed = all(passed_values) if require_both else any(passed_values)
    result = {
        "protocol_id": protocol.get("protocol_id"),
        "status": "pass" if passed else "fail",
        "selection_split": "val",
        "test_used": False,
        "seed": 42,
        "primary_metric": PRIMARY_METRIC,
        "paired_run": str(paired_run),
        "shuffled_run": str(shuffled_run),
        "paired_metrics": str(paired_metrics_path),
        "shuffled_metrics": str(shuffled_metrics_path),
        "paired_value": paired_value,
        "shuffled_value": shuffled_value,
        "paired_minus_shuffled": margin,
        "require_both": require_both,
        "checks": checks,
        "decision": (
            "advance_to_three_seeds"
            if passed
            else "stop_after_seed42_and_record_as_rejected_transfer"
        ),
    }
    output_path = Path(args.out)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
