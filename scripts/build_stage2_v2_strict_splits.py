#!/usr/bin/env python3
"""Build globally deduplicated, canonical-event-disjoint Stage-2 v2 manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


SPLITS = ("train", "val", "test")
VARIANTS = ("predicted_prior", "oracle_prior", "no_prior")
CORE_FIELDS = ("pre_image", "post_sar", "mask_multiclass")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def resolve_data_path(data_root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else data_root / path


def sha256(path: Path, cache: dict[Path, tuple[int, str]]) -> tuple[int, str]:
    path = path.resolve()
    if path not in cache:
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        cache[path] = (int(path.stat().st_size), digest.hexdigest())
    return cache[path]


def signature_id(signature: tuple[tuple[int, str], ...]) -> str:
    payload = json.dumps(signature, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_map(plan: dict[str, Any], source_events: set[str]) -> dict[str, str]:
    result = {event: event for event in source_events}
    for canonical, aliases in dict(plan.get("canonical_event_aliases", {})).items():
        for alias in aliases:
            alias = str(alias)
            if alias not in source_events:
                raise KeyError(f"configured event alias is absent from source manifests: {alias}")
            if result[alias] != alias:
                raise ValueError(f"event alias assigned twice: {alias}")
            result[alias] = str(canonical)
    return result


def validate_split_plan(plan: dict[str, Any], canonical_events: set[str]) -> dict[str, str]:
    assignments: dict[str, str] = {}
    split_events = dict(plan.get("split_events", {}))
    if set(split_events) != set(SPLITS):
        raise ValueError(f"split_events must contain exactly {SPLITS}")
    for split in SPLITS:
        for event in split_events[split]:
            event = str(event)
            if event in assignments:
                raise ValueError(f"canonical event assigned to multiple splits: {event}")
            assignments[event] = split
    missing = sorted(canonical_events - set(assignments))
    extra = sorted(set(assignments) - canonical_events)
    if missing or extra:
        raise ValueError(f"split plan mismatch; missing={missing}, extra={extra}")
    return assignments


def class_stats(rows: list[dict[str, Any]], data_root: Path) -> dict[str, Any]:
    counts = np.zeros(4, dtype=np.int64)
    for row in rows:
        path = resolve_data_path(data_root, row["mask_multiclass"])
        mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
        counts += np.bincount(mask.reshape(-1), minlength=4)[:4]
    building = counts[1:]
    total = int(building.sum())
    names = ("background", "intact", "damaged", "destroyed")
    return {
        "class_pixels": {name: int(counts[index]) for index, name in enumerate(names)},
        "building_class_share": {
            names[index]: (float(counts[index]) / total if total else 0.0) for index in range(1, 4)
        },
    }


def build(
    data_root: Path,
    source_manifest_root: Path,
    plan_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output root: {output_root}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if tuple(plan.get("identity_fields", [])) != CORE_FIELDS:
        raise ValueError(f"identity_fields must be {CORE_FIELDS}")
    priority = [str(value) for value in plan.get("representative_split_priority", SPLITS)]
    if set(priority) != set(SPLITS):
        raise ValueError(f"representative_split_priority must contain {SPLITS}")
    priority_index = {split: index for index, split in enumerate(priority)}

    master_rows: list[dict[str, Any]] = []
    variant_maps: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for split in SPLITS:
        rows = read_jsonl(source_manifest_root / f"stage2_master_{split}.jsonl")
        for row_index, row in enumerate(rows):
            source = dict(row)
            source["_source_split"] = split
            source["_source_row_index"] = row_index
            master_rows.append(source)
    for variant in VARIANTS:
        mapping: dict[tuple[str, str], dict[str, Any]] = {}
        for split in SPLITS:
            for row in read_jsonl(source_manifest_root / variant / f"{split}.jsonl"):
                key = (split, str(row["id"]))
                if key in mapping:
                    raise ValueError(f"duplicate source variant key: {variant}/{key}")
                mapping[key] = row
        variant_maps[variant] = mapping

    source_events = {str(row["event_id"]) for row in master_rows}
    event_map = canonical_map(plan, source_events)
    canonical_events = {event_map[event] for event in source_events}
    assignments = validate_split_plan(plan, canonical_events)

    hash_cache: dict[Path, tuple[int, str]] = {}
    grouped: dict[tuple[tuple[int, str], ...], list[dict[str, Any]]] = defaultdict(list)
    for row in master_rows:
        signature = tuple(
            sha256(resolve_data_path(data_root, row[field]), hash_cache) for field in CORE_FIELDS
        )
        source = dict(row)
        source["_content_signature"] = signature
        source["_signature_id"] = signature_id(signature)
        source["_canonical_event_id"] = event_map[str(row["event_id"])]
        grouped[signature].append(source)

    selected: list[dict[str, Any]] = []
    dedup_groups: list[dict[str, Any]] = []
    source_to_strict: list[dict[str, Any]] = []
    for signature, group in grouped.items():
        canonical_group_events = {str(row["_canonical_event_id"]) for row in group}
        if len(canonical_group_events) != 1:
            raw_events = sorted({str(row["event_id"]) for row in group})
            raise ValueError(
                "identical content crosses canonical events; update canonical_event_aliases: "
                f"raw={raw_events}, canonical={sorted(canonical_group_events)}"
            )
        representative = min(
            group,
            key=lambda row: (
                priority_index[str(row["_source_split"])],
                str(row["id"]),
                int(row["_source_row_index"]),
            ),
        )
        canonical_event = next(iter(canonical_group_events))
        strict_split = assignments[canonical_event]
        strict_id = f"strictv1__{representative['_source_split']}__{representative['id']}"
        selected_row = dict(representative)
        selected_row["_strict_split"] = strict_split
        selected_row["_strict_id"] = strict_id
        selected.append(selected_row)
        members = []
        for member in sorted(group, key=lambda row: (str(row["_source_split"]), str(row["id"]))):
            record = {
                "source_split": str(member["_source_split"]),
                "source_id": str(member["id"]),
                "source_event_id": str(member["event_id"]),
                "strict_id": strict_id,
                "strict_split": strict_split,
                "canonical_event_id": canonical_event,
                "is_representative": member is representative,
                "core_content_signature": str(member["_signature_id"]),
            }
            source_to_strict.append(record)
            members.append(record)
        if len(group) > 1:
            dedup_groups.append(
                {
                    "core_content_signature": str(representative["_signature_id"]),
                    "canonical_event_id": canonical_event,
                    "strict_split": strict_split,
                    "members": members,
                }
            )

    output_root.mkdir(parents=True, exist_ok=False)
    outputs: dict[str, dict[str, list[dict[str, Any]]]] = {
        "master": {split: [] for split in SPLITS},
        **{variant: {split: [] for split in SPLITS} for variant in VARIANTS},
    }
    hash_rows: list[dict[str, Any]] = []
    for source in selected:
        source_key = (str(source["_source_split"]), str(source["id"]))
        strict_split = str(source["_strict_split"])
        canonical_event = str(source["_canonical_event_id"])
        strict_id = str(source["_strict_id"])
        signature = source["_content_signature"]

        def transform(row: dict[str, Any]) -> dict[str, Any]:
            out = dict(row)
            out.update(
                {
                    "id": strict_id,
                    "split": strict_split,
                    "event_id": canonical_event,
                    "source_split": source_key[0],
                    "source_id": source_key[1],
                    "source_event_id": str(source["event_id"]),
                    "strict_split_protocol": str(plan["protocol"]),
                    "core_content_signature": str(source["_signature_id"]),
                }
            )
            return out

        master = {key: value for key, value in source.items() if not key.startswith("_")}
        outputs["master"][strict_split].append(transform(master))
        for variant in VARIANTS:
            if source_key not in variant_maps[variant]:
                raise KeyError(f"missing {variant} row for source {source_key}")
            outputs[variant][strict_split].append(transform(variant_maps[variant][source_key]))
        hash_rows.append(
            {
                "id": strict_id,
                "split": strict_split,
                "event_id": canonical_event,
                "source_split": source_key[0],
                "source_id": source_key[1],
                "core_content_signature": str(source["_signature_id"]),
                **{
                    f"{field}_size": int(signature[index][0])
                    for index, field in enumerate(CORE_FIELDS)
                },
                **{
                    f"{field}_sha256": str(signature[index][1])
                    for index, field in enumerate(CORE_FIELDS)
                },
            }
        )

    for kind, split_rows in outputs.items():
        for split, rows in split_rows.items():
            rows.sort(key=lambda row: (str(row["event_id"]), str(row["id"])))
            path = (
                output_root / f"stage2_master_{split}.jsonl"
                if kind == "master"
                else output_root / kind / f"{split}.jsonl"
            )
            write_jsonl(path, rows)

    source_to_strict.sort(key=lambda row: (row["source_split"], row["source_id"]))
    with (output_root / "source_to_strict.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_to_strict[0]))
        writer.writeheader()
        writer.writerows(source_to_strict)
    write_jsonl(output_root / "content_hashes.jsonl", sorted(hash_rows, key=lambda row: row["id"]))
    write_json(output_root / "deduplication_groups.json", dedup_groups)
    write_json(output_root / "split_plan_resolved.json", plan)

    summary: dict[str, Any] = {
        "protocol": plan["protocol"],
        "data_root": str(data_root.resolve()),
        "source_manifest_root": str(source_manifest_root.resolve()),
        "source_row_count": len(master_rows),
        "unique_content_sample_count": len(selected),
        "duplicate_row_count_removed": len(master_rows) - len(selected),
        "duplicate_group_count": len(dedup_groups),
        "cross_split_duplicate_group_count": sum(
            len({member["source_split"] for member in group["members"]}) > 1
            for group in dedup_groups
        ),
        "canonical_event_count": len(canonical_events),
        "split_events": plan["split_events"],
        "splits": {},
    }
    for split in SPLITS:
        rows = outputs["master"][split]
        summary["splits"][split] = {
            "sample_count": len(rows),
            "event_count": len({str(row["event_id"]) for row in rows}),
            "disaster_types": sorted({str(row.get("disaster_type", "")) for row in rows}),
            **class_stats(rows, data_root),
        }
    write_json(output_root / "build_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--source-manifest-root", required=True)
    parser.add_argument("--split-plan", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build(
        data_root=Path(args.data_root),
        source_manifest_root=Path(args.source_manifest_root),
        plan_path=Path(args.split_plan),
        output_root=Path(args.output_root),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
