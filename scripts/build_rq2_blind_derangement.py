#!/usr/bin/env python3
"""Build the label-free RQ2 C3 blind mapping within event/shape strata."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


FORBIDDEN_FIELD_TOKENS = ("mask", "target", "label", "annotation", "ground_truth")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_blind_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number}: expected object")
            forbidden = sorted(
                key
                for key in row
                if any(token in key.lower() for token in FORBIDDEN_FIELD_TOKENS)
            )
            if forbidden:
                raise ValueError(f"line {line_number}: blind manifest exposes label fields {forbidden}")
            missing = sorted(
                {"id", "event_id", "pre_image", "post_sar", "width", "height"} - row.keys()
            )
            if missing:
                raise ValueError(f"line {line_number}: missing fields {missing}")
            rows.append(row)
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("blind manifest sample IDs are not unique")
    if len({str(row["event_id"]) for row in rows}) < 3:
        raise ValueError("blind manifest must contain at least three canonical events")
    return rows


def build_mapping(rows: list[dict[str, Any]], seed: int = 3407) -> tuple[dict[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        key = (
            str(row["event_id"]),
            int(row["width"]),
            int(row["height"]),
        )
        groups[key].append(index)
    singleton_groups = [key for key, indices in groups.items() if len(indices) < 2]
    if singleton_groups:
        raise ValueError(
            "cannot fully derange blind inputs; singleton event/shape strata: "
            f"{singleton_groups[:20]}"
        )
    rng = random.Random(int(seed))
    mapping: dict[str, str] = {}
    records = []
    for key in sorted(groups):
        indices = list(groups[key])
        order = list(indices)
        rng.shuffle(order)
        shift = rng.randint(1, len(order) - 1)
        sources = order[shift:] + order[:shift]
        for target, source in zip(order, sources, strict=True):
            target_id = str(rows[target]["id"])
            source_id = str(rows[source]["id"])
            if target_id == source_id:
                raise AssertionError("blind derangement generated a self-pair")
            mapping[target_id] = source_id
            records.append(
                {
                    "target_id": target_id,
                    "source_id": source_id,
                    "event_id": key[0],
                    "width": key[1],
                    "height": key[2],
                }
            )
    if set(mapping.values()) != set(mapping):
        raise AssertionError("blind derangement is not bijective")
    return mapping, sorted(records, key=lambda row: row["target_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    rows = read_blind_manifest(manifest)
    mapping, records = build_mapping(rows, args.seed)
    payload = {
        "protocol_id": "rq2_model_development_v1.0",
        "mode": "within_event_shape_derangement",
        "seed": args.seed,
        "manifest_sha256": sha256_file(manifest),
        "sample_count": len(rows),
        "event_count": len({str(row["event_id"]) for row in rows}),
        "mapping": mapping,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"sample_count": len(rows), "event_count": payload["event_count"]}))


if __name__ == "__main__":
    main()
