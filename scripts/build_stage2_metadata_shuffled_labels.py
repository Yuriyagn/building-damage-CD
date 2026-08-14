#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def histogram_preserving_derangement(rows: list[dict[str, Any]]) -> dict[str, str]:
    ordered = sorted(
        ((str(row["disaster_type"]), str(row["id"])) for row in rows),
        key=lambda item: (item[0], item[1]),
    )
    labels = [label for label, _ in ordered]
    counts = Counter(labels)
    if not labels or max(counts.values()) * 2 > len(labels):
        raise ValueError("a full histogram-preserving disaster-label derangement is impossible")
    shift = max(counts.values())
    rotated = labels[shift:] + labels[:shift]
    mapping = {sample_id: rotated[index] for index, (_, sample_id) in enumerate(ordered)}
    if any(mapping[sample_id] == label for label, sample_id in ordered):
        raise AssertionError("constructed disaster-label mapping is not deranged")
    if Counter(mapping.values()) != counts:
        raise AssertionError("constructed disaster-label mapping changed the class histogram")
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the fixed M2 shuffled disaster-label control")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260814)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = Path(args.manifest)
    output = Path(args.output)
    rows = read_jsonl(manifest)
    mapping = histogram_preserving_derangement(rows)
    counts = Counter(str(row["disaster_type"]) for row in rows)
    payload = {
        "protocol": "stage2_metadata_multitask_v1_m2",
        "method": "sorted_multiset_cyclic_derangement",
        "seed": int(args.seed),
        "source_manifest": str(manifest),
        "sample_count": len(rows),
        "rotation": max(counts.values()),
        "class_counts": dict(sorted(counts.items())),
        "labels": dict(sorted(mapping.items())),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "sample_count": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
