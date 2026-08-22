#!/usr/bin/env python3
"""Apply preregistered RQ3 pilot/full hard gates to an aggregated metric record."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


PILOT_GATES = {
    "anchor_relative_event_macro_damage": (">=", 0.01),
    "damaged_delta": (">=", 0.005),
    "destroyed_delta": (">=", -0.01),
    "improved_event_count": (">=", 8),
    "c2_minus_c3_damage": (">=", 0.02),
    "worst_event_grade_vs_anchor": (">=", -0.03),
    "worst_event_grade_vs_c3": (">=", -0.03),
    "overall_damage_vs_original_f0": (">=", 0.0),
}
FULL_GATES = {
    "anchor_relative_damage": (">=", 0.02),
    "damaged_delta": (">=", 0.01),
    "destroyed_delta": (">=", 0.0),
    "positive_seed_count": (">=", 2),
    "positive_event_count": (">=", 10),
    "c2_minus_c3_damage": (">=", 0.02),
    "positive_c2_minus_c3_seed_count": (">=", 2),
    "c2_minus_c1_damage": (">=", 0.01),
    "worst_event_grade_vs_anchor": (">=", -0.01),
    "worst_event_grade_vs_c3": (">=", -0.01),
}


def evaluate(payload: dict[str, Any]) -> dict[str, Any]:
    phase = str(payload.get("phase"))
    thresholds = PILOT_GATES if phase == "pilot" else FULL_GATES if phase == "full" else None
    if thresholds is None:
        raise ValueError("phase must be pilot or full")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("metrics must be an object")
    gates = {}
    for name, (_, threshold) in thresholds.items():
        value = metrics.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"missing numeric metric: {name}")
        gates[name] = {"value": value, "threshold": threshold, "passed": bool(value >= threshold)}
    passed = all(record["passed"] for record in gates.values())
    return {
        "protocol_id": "rq3_damage_evidence_decomposition_v1.0",
        "phase": phase,
        "experiment": payload.get("experiment"),
        "selected_factors": payload.get("selected_factors", []),
        "passed": passed,
        "gates": gates,
        "decision": "promote_as_latest_anchor" if passed else "freeze_factor_return_to_latest_passed_anchor",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(json.loads(args.input.read_text(encoding="utf-8")))
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite verdict: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "decision": result["decision"]}, sort_keys=True))
    return 0 if result["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
