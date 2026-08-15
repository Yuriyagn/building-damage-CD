#!/usr/bin/env python3
"""Validate the project experiment registry without third-party packages."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
VALID_STATUSES = {
    "planned",
    "blocked",
    "running",
    "completed",
    "failed",
    "superseded",
    "retracted",
}
VALID_CLAIM_VERDICTS = {
    "ALIGNED",
    "OVERSTATED",
    "NOT_SUPPORTED_BY_PROVENANCE",
    "PROVENANCE_INSUFFICIENT",
}


def _require(mapping: dict, keys: set[str], context: str, errors: list[str]) -> None:
    missing = sorted(keys - mapping.keys())
    if missing:
        errors.append(f"{context}: missing required keys {missing}")


def _check_path_hash(item: object, context: str, root: Path, errors: list[str]) -> None:
    if not isinstance(item, dict):
        errors.append(f"{context}: expected object")
        return
    _require(item, {"path", "sha256"}, context, errors)
    path = item.get("path")
    digest = item.get("sha256")
    if not isinstance(path, str) or not path:
        errors.append(f"{context}.path: expected non-empty string")
    elif Path(path).is_absolute():
        errors.append(f"{context}.path: must be repository-relative")
    elif not (root / path).exists():
        errors.append(f"{context}.path: tracked artifact does not exist: {path}")
    if digest is not None and (not isinstance(digest, str) or not SHA256_RE.fullmatch(digest)):
        errors.append(f"{context}.sha256: expected lowercase 64-character digest or null")


def validate_registry(payload: object, root: Path) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["registry: expected top-level object"]
    _require(payload, {"schema_version", "updated_at", "entries"}, "registry", errors)
    if payload.get("schema_version") != "1.0":
        errors.append("registry.schema_version: expected '1.0'")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        errors.append("registry.entries: expected array")
        return errors

    required = {
        "experiment_id",
        "title",
        "research_question",
        "hypothesis",
        "status",
        "provenance_status",
        "datasets",
        "manifests_and_audits",
        "split",
        "code",
        "conditions",
        "seeds",
        "selection",
        "metrics",
        "test_exposure",
        "runs",
        "artifacts",
        "gate",
        "claims",
    }
    seen_ids: set[str] = set()
    for index, entry in enumerate(entries):
        context = f"entries[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{context}: expected object")
            continue
        _require(entry, required, context, errors)
        experiment_id = entry.get("experiment_id")
        if not isinstance(experiment_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]+", experiment_id):
            errors.append(f"{context}.experiment_id: invalid identifier")
        elif experiment_id in seen_ids:
            errors.append(f"{context}.experiment_id: duplicate {experiment_id}")
        else:
            seen_ids.add(experiment_id)

        status = entry.get("status")
        if status not in VALID_STATUSES:
            errors.append(f"{context}.status: invalid status {status!r}")

        code = entry.get("code")
        if isinstance(code, dict):
            _require(code, {"commit", "dirty_patch_sha256", "configs"}, f"{context}.code", errors)
            commit = code.get("commit")
            patch_hash = code.get("dirty_patch_sha256")
            if commit is not None and (not isinstance(commit, str) or not COMMIT_RE.fullmatch(commit)):
                errors.append(f"{context}.code.commit: expected 40-character commit or null")
            if patch_hash is not None and (
                not isinstance(patch_hash, str) or not SHA256_RE.fullmatch(patch_hash)
            ):
                errors.append(f"{context}.code.dirty_patch_sha256: invalid digest")
            for config_index, config in enumerate(code.get("configs", [])):
                _check_path_hash(config, f"{context}.code.configs[{config_index}]", root, errors)
        else:
            errors.append(f"{context}.code: expected object")

        for key in ("manifests_and_audits", "artifacts"):
            items = entry.get(key)
            if not isinstance(items, list):
                errors.append(f"{context}.{key}: expected array")
                continue
            for item_index, item in enumerate(items):
                _check_path_hash(item, f"{context}.{key}[{item_index}]", root, errors)

        conditions = entry.get("conditions")
        if not isinstance(conditions, list) or not conditions:
            errors.append(f"{context}.conditions: expected non-empty array")
            condition_ids: set[str] = set()
        else:
            condition_ids = {item.get("id") for item in conditions if isinstance(item, dict)}

        runs = entry.get("runs")
        if not isinstance(runs, list):
            errors.append(f"{context}.runs: expected array")
            runs = []
        run_pairs: set[tuple[object, object]] = set()
        for run_index, run in enumerate(runs):
            run_context = f"{context}.runs[{run_index}]"
            if not isinstance(run, dict):
                errors.append(f"{run_context}: expected object")
                continue
            _require(run, {"condition", "seed", "status", "artifact"}, run_context, errors)
            if run.get("condition") not in condition_ids:
                errors.append(f"{run_context}.condition: unknown condition {run.get('condition')!r}")
            pair = (run.get("condition"), run.get("seed"))
            if pair in run_pairs:
                errors.append(f"{run_context}: duplicate condition/seed pair {pair}")
            run_pairs.add(pair)

        claims = entry.get("claims")
        if not isinstance(claims, list) or not claims:
            errors.append(f"{context}.claims: expected non-empty array")
        else:
            for claim_index, claim in enumerate(claims):
                claim_context = f"{context}.claims[{claim_index}]"
                if not isinstance(claim, dict):
                    errors.append(f"{claim_context}: expected object")
                    continue
                _require(claim, {"claim", "verdict", "basis"}, claim_context, errors)
                if claim.get("verdict") not in VALID_CLAIM_VERDICTS:
                    errors.append(f"{claim_context}.verdict: invalid verdict")

        exposure = entry.get("test_exposure")
        if isinstance(exposure, dict):
            if exposure.get("used") is False and exposure.get("status") in {"exposed", "historical_diagnostic_only"}:
                errors.append(f"{context}.test_exposure: used=false conflicts with status")
            if exposure.get("used") is True and exposure.get("status") in {"not_used", "sealed"}:
                errors.append(f"{context}.test_exposure: used=true conflicts with status")
        else:
            errors.append(f"{context}.test_exposure: expected object")

        gate = entry.get("gate")
        if not isinstance(gate, dict):
            errors.append(f"{context}.gate: expected object")
        elif status in {"completed", "failed"} and gate.get("passed") is None:
            errors.append(f"{context}.gate.passed: completed/failed experiment needs a decision")

        if status in {"completed", "failed"} and not runs:
            errors.append(f"{context}.runs: completed/failed experiment needs run provenance")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry", type=Path)
    args = parser.parse_args()
    registry_path = args.registry.resolve()
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    root = registry_path.parent.parent
    errors = validate_registry(payload, root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"OK: {len(payload['entries'])} experiment entries validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
