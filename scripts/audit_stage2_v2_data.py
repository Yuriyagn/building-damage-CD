#!/usr/bin/env python3
"""Audit Stage-2 v2 split integrity and label distribution before training."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm


SPLITS = ("train", "val", "test")
CLASS_NAMES = ("background", "intact", "damaged", "destroyed")
PATH_FIELDS = ("pre_image", "post_sar", "mask_multiclass")
METADATA_FIELDS = ("disaster_type", "event_id", "country_or_region", "qc_label")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def resolve_data_path(data_root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else data_root / path


def shares(counts: np.ndarray, denominator: float | None = None) -> dict[str, float]:
    total = float(counts.sum()) if denominator is None else float(denominator)
    return {
        name: (float(counts[index]) / total if total else 0.0)
        for index, name in enumerate(CLASS_NAMES)
    }


def duplicate_values(values: list[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if value and count > 1)


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_hash_audit(
    rows_by_split: dict[str, list[dict[str, Any]]],
    data_root: Path,
    max_group_details: int = 200,
) -> dict[str, Any]:
    """Detect identical content even when IDs and file paths have been renamed."""
    hash_cache: dict[Path, tuple[int, str] | None] = {}
    row_hashes: dict[tuple[str, int], dict[str, tuple[int, str] | None]] = {}
    field_records: dict[str, list[dict[str, Any]]] = {field: [] for field in PATH_FIELDS}

    for split in SPLITS:
        for row_index, row in enumerate(rows_by_split[split]):
            hashes: dict[str, tuple[int, str] | None] = {}
            for field in PATH_FIELDS:
                value = row.get(field)
                if not value:
                    hashes[field] = None
                    continue
                path = resolve_data_path(data_root, value).resolve()
                if path not in hash_cache:
                    if path.is_file():
                        digest = file_sha256(path)
                        hash_cache[path] = (int(path.stat().st_size), str(digest)) if digest else None
                    else:
                        hash_cache[path] = None
                fingerprint = hash_cache[path]
                hashes[field] = fingerprint
                if fingerprint is not None:
                    size, digest = fingerprint
                    field_records[field].append(
                        {
                            "split": split,
                            "row_index": row_index,
                            "id": str(row.get("id", "")),
                            "event_id": str(row.get("event_id", "")),
                            "path": str(path),
                            "size": size,
                            "sha256": digest,
                        }
                    )
            row_hashes[(split, row_index)] = hashes

    def summarize_groups(records: list[dict[str, Any]], key_fields: tuple[str, ...]) -> dict[str, Any]:
        grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[tuple(record[field] for field in key_fields)].append(record)
        duplicates = [group for group in grouped.values() if len(group) > 1]
        duplicates.sort(
            key=lambda group: [
                (str(record["split"]), str(record["id"]), str(record.get("path", "")))
                for record in group
            ]
        )
        cross_split = [group for group in duplicates if len({str(record["split"]) for record in group}) > 1]
        within_split = [group for group in duplicates if len({str(record["split"]) for record in group}) == 1]
        return {
            "duplicate_group_count": len(duplicates),
            "duplicate_row_excess": int(sum(len(group) - 1 for group in duplicates)),
            "cross_split_group_count": len(cross_split),
            "within_split_group_count": len(within_split),
            "group_details_truncated": len(duplicates) > max_group_details,
            "groups": duplicates[:max_group_details],
        }

    fields: dict[str, Any] = {}
    for field, records in field_records.items():
        fields[field] = {
            "hashed_file_count": len(records),
            **summarize_groups(records, ("size", "sha256")),
        }

    full_records: list[dict[str, Any]] = []
    for split in SPLITS:
        for row_index, row in enumerate(rows_by_split[split]):
            hashes = row_hashes[(split, row_index)]
            if any(hashes[field] is None for field in PATH_FIELDS):
                continue
            signature = tuple(hashes[field] for field in PATH_FIELDS)
            full_records.append(
                {
                    "split": split,
                    "row_index": row_index,
                    "id": str(row.get("id", "")),
                    "event_id": str(row.get("event_id", "")),
                    "content_signature": [
                        {"field": field, "size": signature[index][0], "sha256": signature[index][1]}
                        for index, field in enumerate(PATH_FIELDS)
                    ],
                    "signature_key": json.dumps(signature, separators=(",", ":")),
                }
            )
    full_summary = summarize_groups(full_records, ("signature_key",))
    for group in full_summary["groups"]:
        for record in group:
            record.pop("signature_key", None)

    return {
        "algorithm": "SHA-256 over complete file bytes; full-sample identity requires pre_image, post_sar, and mask_multiclass to all match",
        "unique_physical_files_hashed": sum(value is not None for value in hash_cache.values()),
        "fields": fields,
        "full_sample": full_summary,
    }


def audit_split(
    split: str,
    rows: list[dict[str, Any]],
    data_root: Path,
    skip_mask_stats: bool,
    hard_errors: list[str],
    max_errors: int,
) -> dict[str, Any]:
    ids = [str(row.get("id", "")) for row in rows]
    missing_ids = sum(not value for value in ids)
    duplicate_ids = duplicate_values(ids)
    if missing_ids:
        hard_errors.append(f"{split}: {missing_ids} rows have no id")
    if duplicate_ids:
        hard_errors.append(f"{split}: duplicate ids: {duplicate_ids[:20]}")

    metadata_counts = {
        field: dict(sorted(Counter(str(row.get(field, "") or "unknown") for row in rows).items()))
        for field in METADATA_FIELDS
    }
    paths_by_field: dict[str, list[str]] = {}
    for field in PATH_FIELDS:
        values = [str(row.get(field, "") or "") for row in rows]
        paths_by_field[field] = values
        missing = sum(not value for value in values)
        duplicates = duplicate_values(values)
        if missing and len(hard_errors) < max_errors:
            hard_errors.append(f"{split}: {missing} rows have no {field}")
        if duplicates and len(hard_errors) < max_errors:
            hard_errors.append(f"{split}: duplicate {field} paths: {duplicates[:20]}")

    class_counts = np.zeros(4, dtype=np.int64)
    images_with_class = np.zeros(4, dtype=np.int64)
    event_class_counts: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(4, dtype=np.int64))
    event_image_counts: Counter[str] = Counter()
    invalid_masks: list[dict[str, Any]] = []

    if not skip_mask_stats:
        for row in tqdm(rows, desc=f"audit_{split}_masks", leave=False):
            sample_id = str(row.get("id", ""))
            event_id = str(row.get("event_id", "") or "unknown")
            value = row.get("mask_multiclass")
            if not value:
                continue
            path = resolve_data_path(data_root, value)
            if not path.is_file():
                if len(hard_errors) < max_errors:
                    hard_errors.append(f"{split}/{sample_id}: missing mask {path}")
                continue
            try:
                mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
            except Exception as exc:  # noqa: BLE001
                if len(hard_errors) < max_errors:
                    hard_errors.append(f"{split}/{sample_id}: cannot read mask {path}: {exc}")
                continue
            values = np.unique(mask)
            invalid = values[(values < 0) | (values > 3)]
            if invalid.size:
                invalid_masks.append({"id": sample_id, "path": str(path), "values": invalid.astype(int).tolist()})
                if len(hard_errors) < max_errors:
                    hard_errors.append(f"{split}/{sample_id}: invalid mask values {invalid.astype(int).tolist()}")
                continue
            counts = np.bincount(mask.reshape(-1), minlength=4)[:4].astype(np.int64)
            class_counts += counts
            images_with_class += counts > 0
            event_class_counts[event_id] += counts
            event_image_counts[event_id] += 1

    building_total = int(class_counts[1:].sum())
    building_shares = {
        CLASS_NAMES[index]: (float(class_counts[index]) / building_total if building_total else 0.0)
        for index in range(1, 4)
    }
    event_stats = {}
    for event_id in sorted(event_class_counts):
        counts = event_class_counts[event_id]
        event_building_total = int(counts[1:].sum())
        event_stats[event_id] = {
            "sample_count": int(event_image_counts[event_id]),
            "class_pixels": {name: int(counts[index]) for index, name in enumerate(CLASS_NAMES)},
            "building_class_share": {
                CLASS_NAMES[index]: (float(counts[index]) / event_building_total if event_building_total else 0.0)
                for index in range(1, 4)
            },
        }

    return {
        "sample_count": len(rows),
        "unique_id_count": len(set(ids)) - (1 if "" in ids else 0),
        "missing_id_count": missing_ids,
        "duplicate_ids": duplicate_ids,
        "metadata_counts": metadata_counts,
        "paths_by_field": paths_by_field,
        "class_pixels": {name: int(class_counts[index]) for index, name in enumerate(CLASS_NAMES)},
        "full_image_class_share": shares(class_counts),
        "building_class_share": building_shares,
        "images_with_class": {name: int(images_with_class[index]) for index, name in enumerate(CLASS_NAMES)},
        "invalid_masks": invalid_masks,
        "events": event_stats,
    }


def pairwise_overlaps(split_reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for left, right in combinations(SPLITS, 2):
        pair = f"{left}_vs_{right}"
        left_report = split_reports[left]
        right_report = split_reports[right]
        fields: dict[str, list[str]] = {}
        left_ids = set(left_report["_ids"])
        right_ids = set(right_report["_ids"])
        fields["id"] = sorted((left_ids & right_ids) - {""})
        for field in PATH_FIELDS:
            left_paths = set(left_report["paths_by_field"][field]) - {""}
            right_paths = set(right_report["paths_by_field"][field]) - {""}
            fields[field] = sorted(left_paths & right_paths)
        metadata: dict[str, list[str]] = {}
        for field in ("event_id", "country_or_region", "disaster_type"):
            left_values = set(left_report["metadata_counts"][field]) - {"unknown", ""}
            right_values = set(right_report["metadata_counts"][field]) - {"unknown", ""}
            metadata[field] = sorted(left_values & right_values)
        result[pair] = {"sample_or_path": fields, "metadata": metadata}
    return result


def analyze_id_collisions(
    overlaps: dict[str, Any],
    rows_by_split: dict[str, list[dict[str, Any]]],
    data_root: Path,
) -> None:
    for pair, pair_report in overlaps.items():
        left, right = pair.split("_vs_", maxsplit=1)
        left_rows = {str(row.get("id", "")): row for row in rows_by_split[left]}
        right_rows = {str(row.get("id", "")): row for row in rows_by_split[right]}
        collision_rows: list[dict[str, Any]] = []
        for sample_id in pair_report["sample_or_path"]["id"]:
            field_matches: dict[str, bool | None] = {}
            for field in PATH_FIELDS:
                left_value = left_rows[sample_id].get(field)
                right_value = right_rows[sample_id].get(field)
                if not left_value or not right_value:
                    field_matches[field] = None
                    continue
                left_digest = file_sha256(resolve_data_path(data_root, left_value))
                right_digest = file_sha256(resolve_data_path(data_root, right_value))
                field_matches[field] = (
                    left_digest == right_digest if left_digest is not None and right_digest is not None else None
                )
            duplicate_sample = all(field_matches.get(field) is True for field in PATH_FIELDS)
            collision_rows.append(
                {
                    "id": sample_id,
                    "field_content_matches": field_matches,
                    "duplicate_sample": duplicate_sample,
                }
            )
        pair_report["id_collision_content"] = collision_rows


def distribution_warnings(split_reports: dict[str, dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    for class_name in ("intact", "damaged", "destroyed"):
        values = {split: float(split_reports[split]["building_class_share"][class_name]) for split in SPLITS}
        positive = [value for value in values.values() if value > 0]
        if len(positive) != len(SPLITS):
            warnings.append(f"class {class_name} is absent from at least one split: {values}")
        elif max(positive) / min(positive) >= 3.0:
            warnings.append(f"building-only {class_name} share differs by >=3x across splits: {values}")
    return warnings


def build_report(
    data_root: Path,
    manifest_root: Path,
    skip_mask_stats: bool = False,
    max_errors: int = 200,
    require_event_disjoint: bool = False,
) -> dict[str, Any]:
    hard_errors: list[str] = []
    warnings: list[str] = []
    split_reports: dict[str, dict[str, Any]] = {}
    rows_by_split: dict[str, list[dict[str, Any]]] = {}

    for split in SPLITS:
        manifest = manifest_root / f"stage2_master_{split}.jsonl"
        if not manifest.is_file():
            hard_errors.append(f"missing manifest: {manifest}")
            rows: list[dict[str, Any]] = []
        else:
            rows = read_jsonl(manifest)
        rows_by_split[split] = rows
        report = audit_split(split, rows, data_root, skip_mask_stats, hard_errors, max_errors)
        report["manifest"] = str(manifest)
        report["_ids"] = [str(row.get("id", "")) for row in rows]
        split_reports[split] = report

    overlaps = pairwise_overlaps(split_reports)
    analyze_id_collisions(overlaps, rows_by_split, data_root)
    hash_audit = content_hash_audit(rows_by_split, data_root, max_group_details=max_errors)
    for report in split_reports.values():
        report.pop("_ids", None)
    for pair, pair_report in overlaps.items():
        for field, values in pair_report["sample_or_path"].items():
            if values and field != "id":
                hard_errors.append(f"{pair}: {field} overlap ({len(values)}): {values[:20]}")
        for collision in pair_report["id_collision_content"]:
            if collision["duplicate_sample"]:
                hard_errors.append(f"{pair}: duplicate sample content for id={collision['id']}")
            else:
                warnings.append(
                    f"{pair}: id={collision['id']} is reused but core file contents differ; treat (split, id) as the key"
                )
        event_overlap = pair_report["metadata"]["event_id"]
        if event_overlap:
            message = f"{pair}: {len(event_overlap)} event_id values overlap; cross-event generalization is not measured"
            if require_event_disjoint:
                hard_errors.append(message)
            else:
                warnings.append(message)

    full_duplicates = hash_audit["full_sample"]
    if int(full_duplicates["duplicate_group_count"]):
        hard_errors.append(
            "global full-sample content duplicates: "
            f"{full_duplicates['duplicate_group_count']} groups / "
            f"{full_duplicates['duplicate_row_excess']} excess rows "
            f"({full_duplicates['cross_split_group_count']} cross-split, "
            f"{full_duplicates['within_split_group_count']} within-split); "
            "see content_hash_audit.full_sample"
        )

    if not skip_mask_stats:
        warnings.extend(distribution_warnings(split_reports))

    for split, report in split_reports.items():
        for field in METADATA_FIELDS:
            unknown_count = int(report["metadata_counts"][field].get("unknown", 0))
            if unknown_count:
                warnings.append(f"{split}: {unknown_count} rows have unknown {field}")
        report.pop("paths_by_field", None)

    hard_errors = hard_errors[:max_errors]
    return {
        "status": "fail" if hard_errors else ("warning" if warnings else "pass"),
        "data_root": str(data_root),
        "manifest_root": str(manifest_root),
        "skip_mask_stats": skip_mask_stats,
        "require_event_disjoint": require_event_disjoint,
        "splits": split_reports,
        "overlaps": overlaps,
        "content_hash_audit": hash_audit,
        "hard_error_count": len(hard_errors),
        "hard_errors": hard_errors,
        "warning_count": len(warnings),
        "warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--skip-mask-stats", action="store_true")
    parser.add_argument("--require-event-disjoint", action="store_true")
    parser.add_argument("--max-errors", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(
        data_root=Path(args.data_root),
        manifest_root=Path(args.manifest_root),
        skip_mask_stats=bool(args.skip_mask_stats),
        max_errors=max(1, int(args.max_errors)),
        require_event_disjoint=bool(args.require_event_disjoint),
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "hard_error_count": report["hard_error_count"],
                "warning_count": report["warning_count"],
                "out": str(output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    if report["hard_errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
