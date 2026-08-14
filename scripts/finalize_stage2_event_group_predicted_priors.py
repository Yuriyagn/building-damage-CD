#!/usr/bin/env python3
"""Finalize an immutable R4 manifest bundle after frozen O1 prior export.

The source R0, BRIGHT supplement, and no/oracle R4 manifests are read-only.  A
new manifest root is created only after every supplement ID has exactly one
finite probability map, uint8 preview, and thresholded binary mask.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


SPLITS = ("train", "val", "test")
MODEL_NAME = "O1_unet_resnet34_freq"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_data_path(data_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else data_root / path


def sort_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (str(row["event_group_id"]), str(row["id"])))


def index_by_id(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    counts = Counter(str(row.get("id", "")) for row in rows)
    missing = counts.get("", 0)
    duplicates = sorted(sample_id for sample_id, count in counts.items() if sample_id and count > 1)
    if missing or duplicates:
        raise ValueError(f"{label} invalid IDs: missing={missing}, duplicates={duplicates[:10]}")
    return {str(row["id"]): row for row in rows}


def require_exact_id_coverage(expected: set[str], actual: set[str], label: str) -> None:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(f"{label} ID mismatch: missing={missing[:10]}, extra={extra[:10]}")


def validate_event_groups(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    groups = {
        split: {str(row["event_group_id"]) for row in rows_by_split[split]}
        for split in SPLITS
    }
    overlaps: dict[str, list[str]] = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlaps[f"{left}_vs_{right}"] = sorted(groups[left] & groups[right])
    if any(overlaps.values()):
        raise ValueError(f"event_group_id crosses splits: {overlaps}")
    return {
        "groups_by_split": {split: sorted(groups[split]) for split in SPLITS},
        "overlaps": overlaps,
    }


def predicted_row(row: dict[str, Any], prior: dict[str, Any], threshold: float) -> dict[str, Any]:
    output = dict(row)
    output.update(
        {
            "building_prior": prior["building_prob"],
            "building_prior_binary": prior["building_binary"],
            "building_prior_uint8": prior["building_prob_uint8"],
            "pred_building_prob": prior["building_prob"],
            "pred_building_binary": prior["building_binary"],
            "pred_building_prob_uint8": prior["building_prob_uint8"],
            "prior_type": "predicted",
            "stage1_model": MODEL_NAME,
            "stage1_threshold": threshold,
        }
    )
    return output


def audit_one_prior(data_root: Path, row: dict[str, Any], threshold: float) -> dict[str, Any]:
    prob_path = resolve_data_path(data_root, str(row["building_prob"]))
    prob_uint8_path = resolve_data_path(data_root, str(row["building_prob_uint8"]))
    binary_path = resolve_data_path(data_root, str(row["building_binary"]))
    for path in (prob_path, prob_uint8_path, binary_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    with np.load(prob_path) as payload:
        if "prob" not in payload:
            raise KeyError(f"{prob_path} does not contain prob")
        prob = np.asarray(payload["prob"], dtype=np.float32)
    prob_uint8 = np.asarray(Image.open(prob_uint8_path).convert("L"), dtype=np.uint8)
    binary = np.asarray(Image.open(binary_path).convert("L"), dtype=np.uint8)
    if prob.shape != (1024, 1024) or prob_uint8.shape != prob.shape or binary.shape != prob.shape:
        raise ValueError(
            f"invalid prior shape for id={row['id']}: prob={prob.shape}, "
            f"uint8={prob_uint8.shape}, binary={binary.shape}"
        )
    if not np.isfinite(prob).all() or float(prob.min()) < 0.0 or float(prob.max()) > 1.0:
        raise ValueError(f"invalid probability values for id={row['id']}")
    binary_values = sorted(int(value) for value in np.unique(binary))
    if not set(binary_values).issubset({0, 255}):
        raise ValueError(f"invalid binary values for id={row['id']}: {binary_values}")

    quantized = np.rint(prob * 255.0).astype(np.int16)
    max_quantization_error = int(np.max(np.abs(quantized - prob_uint8.astype(np.int16))))
    if max_quantization_error > 1:
        raise ValueError(
            f"probability PNG disagrees with float16 probability for id={row['id']}: "
            f"max error={max_quantization_error}"
        )
    thresholded = (prob >= threshold).astype(np.uint8) * 255
    binary_mismatch_pixels = int(np.count_nonzero(thresholded != binary))
    return {
        "prob_min": float(prob.min()),
        "prob_max": float(prob.max()),
        "binary_positive_pixels": int(np.count_nonzero(binary)),
        "binary_mismatch_pixels_due_to_float16_rounding": binary_mismatch_pixels,
        "max_uint8_quantization_error": max_quantization_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--r0-manifest-root", type=Path, required=True)
    parser.add_argument("--supplement-manifest-root", type=Path, required=True)
    parser.add_argument("--base-r4-manifest-root", type=Path, required=True)
    parser.add_argument("--prior-export-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--threshold-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-supplement-count", type=int, default=262)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite manifest root: {output_root}")

    data_root = args.data_root.resolve()
    r0_root = args.r0_manifest_root.resolve()
    supplement_root = args.supplement_manifest_root.resolve()
    base_r4_root = args.base_r4_manifest_root.resolve()
    prior_root = args.prior_export_root.resolve()
    checkpoint = args.checkpoint.resolve()
    threshold_json = args.threshold_json.resolve()

    threshold_info = read_json(threshold_json)
    threshold = float(threshold_info["threshold"])
    export_summary = read_json(prior_root / "export_summary.json")
    if export_summary.get("model") != MODEL_NAME:
        raise ValueError(f"unexpected prior model: {export_summary.get('model')}")
    if float(export_summary.get("threshold", -1.0)) != threshold:
        raise ValueError("prior export threshold does not match official threshold")

    supplement_rows = read_jsonl(supplement_root / "stage2_master_train.jsonl")
    prior_rows = read_jsonl(prior_root / "prior_manifest_index.jsonl")
    if len(supplement_rows) != args.expected_supplement_count:
        raise ValueError(
            f"expected {args.expected_supplement_count} supplement rows, found {len(supplement_rows)}"
        )
    supplement_by_id = index_by_id(supplement_rows, "supplement")
    prior_by_id = index_by_id(prior_rows, "prior export")
    require_exact_id_coverage(set(supplement_by_id), set(prior_by_id), "supplement prior export")

    prior_stats = []
    predicted_supplement = []
    for sample_id in sorted(supplement_by_id):
        stats = audit_one_prior(data_root, prior_by_id[sample_id], threshold)
        prior_stats.append({"id": sample_id, **stats})
        predicted_supplement.append(
            predicted_row(supplement_by_id[sample_id], prior_by_id[sample_id], threshold)
        )

    master_by_split = {
        split: read_jsonl(base_r4_root / f"stage2_master_{split}.jsonl") for split in SPLITS
    }
    event_audit = validate_event_groups(master_by_split)
    r0_predicted = {
        split: read_jsonl(r0_root / "predicted_prior" / f"{split}.jsonl") for split in SPLITS
    }
    predicted_by_split = {
        "train": sort_rows([*r0_predicted["train"], *predicted_supplement]),
        "val": sort_rows(r0_predicted["val"]),
        "test": sort_rows(r0_predicted["test"]),
    }
    for split in SPLITS:
        master_ids = set(index_by_id(master_by_split[split], f"master {split}"))
        predicted_ids = set(index_by_id(predicted_by_split[split], f"predicted {split}"))
        require_exact_id_coverage(master_ids, predicted_ids, f"combined predicted {split}")
        for row in predicted_by_split[split]:
            for field in ("building_prior", "building_prior_binary", "building_prior_uint8"):
                path = resolve_data_path(data_root, str(row[field]))
                if not path.is_file():
                    raise FileNotFoundError(path)

    output_root.mkdir(parents=True)
    for split in SPLITS:
        write_jsonl(output_root / f"stage2_master_{split}.jsonl", sort_rows(master_by_split[split]))
        for variant in ("no_prior", "oracle_prior"):
            rows = read_jsonl(base_r4_root / variant / f"{split}.jsonl")
            write_jsonl(output_root / variant / f"{split}.jsonl", sort_rows(rows))
        write_jsonl(output_root / "predicted_prior" / f"{split}.jsonl", predicted_by_split[split])

    mismatch_total = sum(int(row["binary_mismatch_pixels_due_to_float16_rounding"]) for row in prior_stats)
    summary = {
        "protocol": "stage2_v2_event_group_bright_r4_predicted_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "counts": {split: len(master_by_split[split]) for split in SPLITS},
        "supplement_prior_count": len(predicted_supplement),
        "model": MODEL_NAME,
        "threshold": threshold,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "threshold_json": str(threshold_json),
        "threshold_json_sha256": sha256_file(threshold_json),
        "export_summary_sha256": sha256_file(prior_root / "export_summary.json"),
        "prior_manifest_index_sha256": sha256_file(prior_root / "prior_manifest_index.jsonl"),
        "prob_min": min(float(row["prob_min"]) for row in prior_stats),
        "prob_max": max(float(row["prob_max"]) for row in prior_stats),
        "float16_threshold_mismatch_pixels": mismatch_total,
        "event_group_audit": event_audit,
    }
    readiness = {
        "protocol": summary["protocol"],
        "r4_predicted_prior_ready": True,
        "comparison_training_eligible": True,
        "formal_unseen_event_claim_eligible": False,
        "formal_geospatial_independence_claim_eligible": False,
        "test_execution_status": "embargoed_until_all_R0_R4_validation_runs_complete",
        "limitations": [
            "The event-group test was historically inspected and is diagnostic, not pristine unseen-event evidence.",
            "Formal geospatial independence is not proven for 230/580 current val/test PNG rows without source georeferencing.",
        ],
    }
    write_json(output_root / "event_group_audit.json", event_audit)
    write_json(output_root / "prior_audit.json", {"summary": summary, "samples": prior_stats})
    write_json(output_root / "build_summary.json", summary)
    write_json(output_root / "READINESS.json", readiness)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
