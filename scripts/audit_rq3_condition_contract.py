#!/usr/bin/env python3
"""Fail-closed C2/C3 SAR tensor and synchronized-metadata contract audit."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config  # noqa: E402
from stage2.common import read_jsonl, resolve_manifest  # noqa: E402
from stage2.numerical_integrity import strict_write_json  # noqa: E402
from stage2.train_stage2_v2 import make_dataset  # noqa: E402


METADATA_KEYS = (
    "sar_provenance",
    "sar_provider",
    "sar_platform",
    "sar_mode",
    "sar_polarization",
    "sar_product_type",
    "sar_acquisition_time",
    "sar_metadata_confidence",
    "metadata_confidence",
    "sar_incidence_angle",
    "sar_incidence_angle_deg",
    "sar_gsd",
    "sar_gsd_m",
)


def _json_digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii") + b"\0")
    digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_data_path(data_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else data_root / path


def audit_split(
    c2_cfg: dict[str, Any], c3_cfg: dict[str, Any], data_root: Path, split: str
) -> dict[str, Any]:
    seed = int(dict(c2_cfg.get("train", {})).get("seed", 42))
    # This contract audits condition construction, not ROI extraction.  Disable
    # RQ3 components to avoid recomputing connected components while preserving
    # the exact SAR load, normalization, source mapping and metadata path.
    c2_data_cfg = deepcopy(c2_cfg)
    c3_data_cfg = deepcopy(c3_cfg)
    c2_data_cfg.setdefault("model", {})["factors"] = []
    c3_data_cfg.setdefault("model", {})["factors"] = []
    c2 = make_dataset(c2_data_cfg, data_root, split, False, seed)
    c3 = make_dataset(c3_data_cfg, data_root, split, False, seed)
    manifest_key = f"{split}_manifest"
    manifest = resolve_manifest(data_root, str(dict(c2_cfg["dataset"])[manifest_key]))
    original_rows = read_jsonl(manifest)
    original_by_id = {str(row["id"]): row for row in original_rows}
    if len(c2) != len(c3) or len(c2) != len(original_rows):
        raise RuntimeError(f"{split}: C2/C3/manifest length mismatch")

    records: list[dict[str, Any]] = []
    metadata_sync_mismatches: list[str] = []
    identical_tensor_pairs = 0
    for index in range(len(c2)):
        c2_item = c2[index]
        c3_item = c3[index]
        target_id = str(c2_item["id"])
        if str(c3_item["id"]) != target_id:
            raise RuntimeError(f"{split}: target order mismatch at index {index}")
        c2_source = str(c2_item["sar_source_id"])
        c3_source = str(c3_item["sar_source_id"])
        c2_tensor = _tensor_digest(c2_item["sar"])
        c3_tensor = _tensor_digest(c3_item["sar"])
        identical_tensor_pairs += int(c2_tensor == c3_tensor)
        source_row = original_by_id[c3_source]
        remapped_row = c3.rows[index]
        expected_metadata = {key: source_row.get(key) for key in METADATA_KEYS}
        observed_metadata = {key: remapped_row.get(key) for key in METADATA_KEYS}
        if expected_metadata != observed_metadata:
            metadata_sync_mismatches.append(target_id)
        raw_path = _resolve_data_path(data_root, str(source_row["post_sar"]))
        records.append(
            {
                "target_id": target_id,
                "c2_source_id": c2_source,
                "c3_source_id": c3_source,
                "c3_source_deranged": c3_source != target_id,
                "c3_raw_sar_sha256": _file_digest(raw_path),
                "c2_tensor_sha256": c2_tensor,
                "c3_tensor_sha256": c3_tensor,
                "c3_metadata_sha256": _json_digest(observed_metadata),
                "metadata_synchronized": expected_metadata == observed_metadata,
            }
        )

    c2_summary = _json_digest([(row["target_id"], row["c2_tensor_sha256"]) for row in records])
    c3_summary = _json_digest([(row["target_id"], row["c3_tensor_sha256"]) for row in records])
    passed = (
        all(row["c3_source_deranged"] for row in records)
        and not metadata_sync_mismatches
        and c2_summary != c3_summary
    )
    return {
        "split": split,
        "sample_count": len(records),
        "c3_deranged_count": sum(row["c3_source_deranged"] for row in records),
        "c3_deranged_fraction": (
            sum(row["c3_source_deranged"] for row in records) / len(records) if records else 0.0
        ),
        "identical_c2_c3_tensor_pair_count": identical_tensor_pairs,
        "metadata_sync_mismatch_count": len(metadata_sync_mismatches),
        "metadata_sync_mismatch_ids": metadata_sync_mismatches[:20],
        "c2_tensor_aggregate_sha256": c2_summary,
        "c3_tensor_aggregate_sha256": c3_summary,
        "passed": passed,
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c2-config", type=Path, required=True)
    parser.add_argument("--c3-config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "val"), default=("train", "val"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite condition contract: {args.output}")
    c2_cfg = load_config(args.c2_config)
    c3_cfg = load_config(args.c3_config)
    results = [audit_split(c2_cfg, c3_cfg, args.data_root.resolve(), split) for split in args.splits]
    payload = {
        "schema": "rq3_c2_c3_condition_contract_v1",
        "status": "passed" if all(result["passed"] for result in results) else "failed",
        "c2_config": str(args.c2_config.resolve()),
        "c3_config": str(args.c3_config.resolve()),
        "splits": results,
    }
    strict_write_json(args.output, payload)
    print(json.dumps({"status": payload["status"], "output": str(args.output)}, sort_keys=True))
    return 0 if payload["status"] == "passed" else 4


if __name__ == "__main__":
    raise SystemExit(main())
