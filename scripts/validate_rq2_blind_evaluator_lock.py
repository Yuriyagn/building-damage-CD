#!/usr/bin/env python3
"""Fail closed unless the RQ2 third-party blind evaluator contract is frozen."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ZERO_SHA256 = "0" * 64


def validate_lock(payload: object) -> list[str]:
    if not isinstance(payload, dict):
        return ["lock: expected a JSON object"]
    required = {
        "protocol_id",
        "status",
        "evaluator",
        "accepted_at",
        "written_confirmation_sha256",
        "event_count",
        "event_ids_sha256",
        "input_manifest_sha256",
        "hidden_labels_sha256",
        "metric_script_sha256",
        "labels_withheld",
        "pixel_semantic_evaluation",
        "damaged_event_count",
        "destroyed_event_count",
        "geospatial_overlap_audit",
        "excluded_event_overlap",
        "format_dry_run_limit",
        "scored_submission_limit",
        "resubmission_policy",
    }
    errors = []
    missing = sorted(required - payload.keys())
    if missing:
        errors.append(f"lock: missing required keys {missing}")
    expected: dict[str, Any] = {
        "protocol_id": "rq2_model_development_v1.0",
        "status": "accepted",
        "labels_withheld": True,
        "pixel_semantic_evaluation": True,
        "excluded_event_overlap": False,
        "format_dry_run_limit": 1,
        "scored_submission_limit": 1,
        "resubmission_policy": "only_if_no_score_was_computed_or_disclosed",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            errors.append(f"lock.{key}: expected {value!r}, got {payload.get(key)!r}")
    if payload.get("geospatial_overlap_audit") not in {
        "passed",
        "custodian_attested_passed",
    }:
        errors.append("lock.geospatial_overlap_audit: no accepted passing verdict")
    for key, minimum in {
        "event_count": 3,
        "damaged_event_count": 2,
        "destroyed_event_count": 2,
    }.items():
        value = payload.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            errors.append(f"lock.{key}: expected integer >= {minimum}")
    evaluator = payload.get("evaluator")
    if not isinstance(evaluator, str) or not evaluator.strip() or evaluator.startswith("REPLACE_"):
        errors.append("lock.evaluator: real custodian name is required")
    accepted_at = payload.get("accepted_at")
    if not isinstance(accepted_at, str) or "T" not in accepted_at:
        errors.append("lock.accepted_at: ISO-8601 timestamp is required")
    for key in (
        "written_confirmation_sha256",
        "event_ids_sha256",
        "input_manifest_sha256",
        "hidden_labels_sha256",
        "metric_script_sha256",
    ):
        value = payload.get(key)
        if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
            errors.append(f"lock.{key}: lowercase SHA-256 is required")
        elif value == ZERO_SHA256:
            errors.append(f"lock.{key}: template zero digest is forbidden")
    return errors


def load_and_validate(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"blind evaluator lock does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"blind evaluator lock is invalid JSON: {path}: {exc}") from exc
    errors = validate_lock(payload)
    if errors:
        raise RuntimeError("blind evaluator lock rejected:\n- " + "\n- ".join(errors))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lock", type=Path)
    args = parser.parse_args()
    try:
        payload = load_and_validate(args.lock.resolve())
    except RuntimeError as exc:
        print(f"BLOCKED: {exc}")
        return 2
    print(
        json.dumps(
            {
                "status": "accepted",
                "evaluator": payload["evaluator"],
                "event_count": payload["event_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
