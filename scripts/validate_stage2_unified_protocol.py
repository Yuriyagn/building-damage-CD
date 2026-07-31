#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the frozen Stage-2 unified-v1 protocol")
    parser.add_argument(
        "--protocol",
        default="configs/stage2_unified_v1/protocol.yaml",
        help="Protocol YAML relative to the repository root.",
    )
    parser.add_argument("--out", help="Optional JSON result path relative to the repository root.")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected mapping in {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(payload)
    return rows


def git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def canonical_candidate(cfg: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(cfg)
    payload.pop("experiment_name", None)
    dataset = dict(payload.get("dataset", {}))
    dataset.pop("sar_shuffle_mode", None)
    payload["dataset"] = dataset
    return payload


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    protocol_path = Path(args.protocol)
    if not protocol_path.is_absolute():
        protocol_path = repo_root / protocol_path
    protocol = load_yaml(protocol_path)
    errors: list[str] = []
    warnings: list[str] = []

    protocol_id = str(protocol.get("protocol_id", ""))
    supported_protocols = {
        "stage2_unified_v1",
        "stage2_ssfcnet_unified_v1",
        "stage2_fsgnet_unified_v1",
    }
    if protocol_id not in supported_protocols:
        errors.append(f"unsupported unified protocol_id: {protocol_id}")

    data_cfg = dict(protocol.get("data", {}))
    manifest_root = repo_root / str(data_cfg.get("manifest_root", ""))
    expected_counts = dict(data_cfg.get("expected_counts", {}))
    expected_hashes = dict(data_cfg.get("expected_sha256", {}))
    manifest_results: dict[str, Any] = {}
    event_sets: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        path = manifest_root / f"{split}.jsonl"
        if not path.is_file():
            errors.append(f"missing manifest: {path}")
            continue
        rows = read_jsonl(path)
        ids = [str(row.get("id", "")) for row in rows]
        events = {str(row.get("event_id", "") or "unknown") for row in rows}
        event_sets[split] = events
        actual_hash = sha256(path)
        manifest_results[split] = {
            "path": str(path),
            "count": len(rows),
            "sha256": actual_hash,
            "unique_ids": len(set(ids)),
            "event_count": len(events),
        }
        if len(rows) != int(expected_counts.get(split, -1)):
            errors.append(f"{split} count mismatch: {len(rows)}")
        if actual_hash != str(expected_hashes.get(split, "")):
            errors.append(f"{split} SHA-256 mismatch")
        if len(set(ids)) != len(ids):
            errors.append(f"{split} contains duplicate sample ids")

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sorted(event_sets.get(left, set()) & event_sets.get(right, set()))
        if overlap:
            errors.append(f"{left}/{right} event overlap: {overlap[:10]}")

    candidate_paths = dict(protocol.get("candidate_configs", {}))
    candidates: dict[str, dict[str, Any]] = {}
    for label in ("paired", "shuffled"):
        path = repo_root / str(candidate_paths.get(label, ""))
        if not path.is_file():
            errors.append(f"missing {label} candidate config: {path}")
            continue
        cfg = load_yaml(path)
        candidates[label] = cfg
        if cfg.get("protocol_id") != protocol.get("protocol_id"):
            errors.append(f"{label} config protocol_id mismatch")
        dataset = dict(cfg.get("dataset", {}))
        for split in ("train", "val", "test"):
            expected = f"{data_cfg.get('manifest_root')}/{split}.jsonl"
            if dataset.get(f"{split}_manifest") != expected:
                errors.append(f"{label} {split}_manifest is not frozen protocol data")
        train = dict(cfg.get("train", {}))
        effective_batch = int(train.get("batch_size", 0)) * int(
            train.get("gradient_accumulation_steps", 1)
        )
        if effective_batch != int(dict(protocol.get("training", {})).get("effective_batch_size", -1)):
            errors.append(f"{label} effective batch size mismatch")
        if str(train.get("checkpoint_policy", "")).lower() != "primary_only":
            errors.append(f"{label} must use primary_only checkpoint policy")
        if bool(train.get("save_last_checkpoint", True)):
            errors.append(f"{label} must not retain last.pth")
        model = dict(cfg.get("model", {}))
        expected_model_name = str(
            dict(protocol.get("external_method", {})).get(
                "model_name",
                "external_uabcd",
            )
        ).lower()
        if str(model.get("name", "")).lower() != expected_model_name:
            errors.append(f"{label} does not select {expected_model_name}")

    if candidates:
        paired_mode = str(dict(candidates.get("paired", {}).get("dataset", {})).get("sar_shuffle_mode"))
        shuffled_mode = str(
            dict(candidates.get("shuffled", {}).get("dataset", {})).get("sar_shuffle_mode")
        )
        if paired_mode != "paired":
            errors.append("paired candidate must use paired SAR")
        if shuffled_mode != "within_event":
            errors.append("shuffled candidate must use within_event SAR")
        if {"paired", "shuffled"}.issubset(candidates) and canonical_candidate(
            candidates["paired"]
        ) != canonical_candidate(candidates["shuffled"]):
            errors.append("candidate configs differ outside experiment_name/sar_shuffle_mode")

    external_cfg = dict(protocol.get("external_method", {}))
    external_root = (repo_root / str(external_cfg.get("checkout_root", ""))).resolve()
    external_result = {
        "root": str(external_root),
        "git_commit": git_head(external_root),
        "name": external_cfg.get("name"),
    }
    if external_root == repo_root or repo_root in external_root.parents:
        errors.append("external author code must remain outside the main Git repository")
    if external_result["git_commit"] != str(external_cfg.get("git_commit", "")):
        errors.append(f"external {external_cfg.get('name', 'method')} Git commit mismatch")

    artifact_specs = external_cfg.get("artifacts")
    if isinstance(artifact_specs, dict) and artifact_specs:
        artifact_results: dict[str, Any] = {}
        for label, raw_spec in artifact_specs.items():
            spec = dict(raw_spec)
            path = external_root / str(spec.get("path", ""))
            expected = str(spec.get("sha256", ""))
            artifact_result = {"path": str(path)}
            if not path.is_file():
                errors.append(f"missing external {label}: {path}")
            else:
                actual = sha256(path)
                artifact_result["sha256"] = actual
                if actual != expected:
                    errors.append(f"external {label} SHA-256 mismatch")
            artifact_results[str(label)] = artifact_result
        external_result["artifacts"] = artifact_results
    else:
        source_path = external_root / str(external_cfg.get("source_path", ""))
        backbone_path = external_root / str(external_cfg.get("backbone_path", ""))
        external_result["source_path"] = str(source_path)
        external_result["backbone_path"] = str(backbone_path)
        for label, path, expected in (
            ("source", source_path, external_cfg.get("source_sha256")),
            ("backbone", backbone_path, external_cfg.get("backbone_sha256")),
        ):
            if not path.is_file():
                errors.append(f"missing external {label}: {path}")
                continue
            actual = sha256(path)
            external_result[f"{label}_sha256"] = actual
            if actual != str(expected):
                errors.append(f"external {label} SHA-256 mismatch")

    if data_cfg.get("selection_split") != "val":
        errors.append("selection_split must remain val")
    if data_cfg.get("test_policy") != "embargoed_during_development":
        errors.append("test must remain embargoed during development")

    result = {
        "protocol_id": protocol.get("protocol_id"),
        "protocol_path": str(protocol_path),
        "status": "ok" if not errors else "error",
        "hard_error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "manifests": manifest_results,
        "external_method": external_result,
        "candidate_configs": {key: str(repo_root / value) for key, value in candidate_paths.items()},
    }

    if args.out:
        output_path = Path(args.out)
        if not output_path.is_absolute():
            output_path = repo_root / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
