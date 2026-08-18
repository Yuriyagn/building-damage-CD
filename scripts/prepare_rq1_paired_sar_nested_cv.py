#!/usr/bin/env python3
"""Freeze the seven-fold RQ1 nested event-CV manifests and Stage-1 lineage.

This preparation step treats the existing 14-event R4 union as development
data.  It never reads historical model outputs or test metrics.  Generated
manifests are immutable: an existing output directory is rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


FOLD_PAIRS: tuple[tuple[str, str], ...] = (
    ("la_palma_volcano", "bata_explosion"),
    ("ukraine_conflict", "noto_earthquake"),
    ("turkey_eq_2023", "myanmar_hurricane"),
    ("mexico_hurricane", "haiti_earthquake"),
    ("morocco_earthquake", "hawaii_wildfire"),
    ("beirut_explosion_2020", "marshall_wildfire"),
    ("libya_flood", "nyiragongo_2021"),
)
EXPECTED_COUNTS = (680, 452, 262, 239, 202, 189, 190)
SPLITS = ("train", "val", "test")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


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


def model_id(excluded: frozenset[str]) -> str:
    label = "__".join(sorted(excluded))
    return "s1_excl_" + hashlib.sha256(label.encode()).hexdigest()[:12]


def stage1_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "image": row["pre_image"],
        "mask_binary": row["oracle_building_mask"],
        "event_id": row["event_group_id"],
        "event_group_id": row["event_group_id"],
        "disaster_type": row.get("disaster_type", ""),
        "country_or_region": row.get("country_or_region", ""),
        "source_dataset": row.get("source_dataset", "unknown"),
        "source_id": row.get("source_id", row["id"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite protocol root: {output_root}")

    rows: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    for split in SPLITS:
        path = source_root / f"stage2_master_{split}.jsonl"
        source_hashes[str(path)] = sha256_file(path)
        rows.extend(read_jsonl(path))
    rows.sort(key=lambda row: (str(row["event_group_id"]), str(row["id"])))

    ids = [str(row["id"]) for row in rows]
    if len(rows) != 2214 or len(set(ids)) != len(ids):
        raise ValueError(f"expected 2214 unique rows, found rows={len(rows)} unique_ids={len(set(ids))}")
    expected_events = {event for pair in FOLD_PAIRS for event in pair}
    actual_events = {str(row["event_group_id"]) for row in rows}
    if actual_events != expected_events:
        raise ValueError(f"event set mismatch: missing={sorted(expected_events-actual_events)}, extra={sorted(actual_events-expected_events)}")
    counts = Counter(str(row["event_group_id"]) for row in rows)
    actual_fold_counts = tuple(sum(counts[event] for event in pair) for pair in FOLD_PAIRS)
    if actual_fold_counts != EXPECTED_COUNTS:
        raise ValueError(f"fold count mismatch: {actual_fold_counts} != {EXPECTED_COUNTS}")
    signatures = [str(row["core_content_signature"]) for row in rows if row.get("core_content_signature")]
    if len(signatures) != len(set(signatures)):
        raise ValueError("duplicate canonical core_content_signature in source union")

    output_root.mkdir(parents=True)
    lineage_sets: set[frozenset[str]] = set()
    assignments: list[dict[str, Any]] = []
    all_events = frozenset(actual_events)
    for outer_index, outer_pair_tuple in enumerate(FOLD_PAIRS):
        inner_index = (outer_index + 1) % len(FOLD_PAIRS)
        inner_pair_tuple = FOLD_PAIRS[inner_index]
        outer_pair = frozenset(outer_pair_tuple)
        inner_pair = frozenset(inner_pair_tuple)
        train_events = all_events - outer_pair - inner_pair
        fold_root = output_root / "folds" / f"outer_{outer_index}"
        roles = {
            "train": train_events,
            "inner_val": inner_pair,
            "outer_eval": outer_pair,
        }
        for role, events in roles.items():
            role_rows = [row for row in rows if str(row["event_group_id"]) in events]
            write_jsonl(fold_root / "stage2_base" / f"{role}.jsonl", role_rows)

        outer_model = frozenset(outer_pair)
        lineage_sets.add(outer_model)
        assignments.append({
            "outer_fold": outer_index,
            "role": "outer_eval",
            "events": sorted(outer_pair),
            "prior_model_id": model_id(outer_model),
            "prior_excluded_events": sorted(outer_model),
        })
        for event_pair in FOLD_PAIRS:
            pair = frozenset(event_pair)
            if pair == outer_pair:
                continue
            final_excluded = frozenset(outer_pair | pair)
            lineage_sets.add(final_excluded)
            assignments.append({
                "outer_fold": outer_index,
                "role": "final_train",
                "events": sorted(pair),
                "prior_model_id": model_id(final_excluded),
                "prior_excluded_events": sorted(final_excluded),
            })
            if pair != inner_pair:
                inner_excluded = frozenset(outer_pair | inner_pair | pair)
                lineage_sets.add(inner_excluded)
                assignments.append({
                    "outer_fold": outer_index,
                    "role": "inner_train",
                    "events": sorted(pair),
                    "prior_model_id": model_id(inner_excluded),
                    "prior_excluded_events": sorted(inner_excluded),
                })

    lineage_rows: list[dict[str, Any]] = []
    for excluded in sorted(lineage_sets, key=lambda value: (len(value), sorted(value))):
        identifier = model_id(excluded)
        model_root = output_root / "stage1" / identifier
        train_rows = [stage1_row(row) for row in rows if str(row["event_group_id"]) not in excluded]
        export_rows = [stage1_row(row) for row in rows if str(row["event_group_id"]) in excluded]
        write_jsonl(model_root / "train.jsonl", train_rows)
        write_jsonl(model_root / "export.jsonl", export_rows)
        lineage_rows.append({
            "prior_model_id": identifier,
            "excluded_events": sorted(excluded),
            "train_count": len(train_rows),
            "export_count": len(export_rows),
            "train_manifest": str((model_root / "train.jsonl").resolve()),
            "export_manifest": str((model_root / "export.jsonl").resolve()),
            "train_manifest_sha256": sha256_file(model_root / "train.jsonl"),
            "export_manifest_sha256": sha256_file(model_root / "export.jsonl"),
        })

    write_json(output_root / "prior_lineage.json", {"models": lineage_rows, "assignments": assignments})
    protocol = {
        "protocol": "rq1_paired_sar_nested_cv_v1.2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence_scope": "nested_cv_development_only_no_blind_confirmation",
        "source_root": str(source_root),
        "source_manifest_sha256": source_hashes,
        "sample_count": len(rows),
        "event_count": len(actual_events),
        "folds": [
            {"outer_fold": index, "outer_events": list(pair), "inner_fold": (index + 1) % 7,
             "inner_events": list(FOLD_PAIRS[(index + 1) % 7]), "outer_count": EXPECTED_COUNTS[index]}
            for index, pair in enumerate(FOLD_PAIRS)
        ],
        "stage1_model_count": len(lineage_rows),
        "stage1_policy": "ResNet34 U-Net seed42 BCE+Dice fixed 20000 optimizer steps; no held-out checkpoint selection",
        "stage2_conditions": {
            "C0": "[0,0,0,OOF_prior,0]",
            "C1": "[pre_R,pre_G,pre_B,OOF_prior,0]",
            "C2": "[pre_R,pre_G,pre_B,OOF_prior,paired_SAR]",
            "C3": "[pre_R,pre_G,pre_B,OOF_prior,fixed_within_event_deranged_SAR]",
        },
        "historical_test_metrics_allowed": False,
        "codabench_allowed": False,
        "geospatial_independence_claim_eligible": False,
    }
    write_json(output_root / "protocol.json", protocol)
    write_json(output_root / "PREPARED.json", {
        "status": "prepared",
        "hard_error_count": 0,
        "sample_count": len(rows),
        "event_count": len(actual_events),
        "stage1_model_count": len(lineage_rows),
        "protocol_sha256": sha256_file(output_root / "protocol.json"),
        "prior_lineage_sha256": sha256_file(output_root / "prior_lineage.json"),
    })
    print(json.dumps(protocol, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
