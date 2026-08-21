#!/usr/bin/env python3
"""Create a new manifest sidecar with verified SAR provenance; never edit frozen manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


HIGH_CONFIDENCE = {"high", "verified", "stac_exact"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    if args.output_manifest.exists() or args.audit_output.exists():
        raise FileExistsError("refusing to overwrite provenance artifacts")
    rows = read_jsonl(args.input_manifest)
    mapping_rows = read_jsonl(args.mapping)
    mapping = {str(row["sample_id"]): row for row in mapping_rows}
    if len(mapping) != len(mapping_rows):
        raise ValueError("duplicate sample_id in provenance mapping")
    required = {"sample_id", "stac_id", "source_url", "provider", "mode", "polarization", "mapping_confidence", "source_sha256"}
    for row in mapping_rows:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"provenance mapping missing {sorted(missing)} for {row.get('sample_id')}")
        digest = str(row["source_sha256"])
        if len(digest) != 64 or digest == "0" * 64:
            raise ValueError(f"invalid source_sha256 for {row['sample_id']}")
    per_event = defaultdict(lambda: [0, 0])
    enriched: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        source = mapping.get(str(row["id"]))
        high = bool(source and source.get("mapping_confidence") in HIGH_CONFIDENCE)
        event = str(row.get("event_id") or "unknown")
        per_event[event][1] += 1
        if source:
            provenance = {key: value for key, value in source.items() if key != "sample_id"}
            item["sar_provenance"] = provenance
            item["metadata_confidence"] = provenance["mapping_confidence"]
        if high:
            per_event[event][0] += 1
        enriched.append(item)
    high_count = sum(values[0] for values in per_event.values())
    coverage = high_count / max(len(rows), 1)
    per_event_coverage = {event: high / total for event, (high, total) in sorted(per_event.items())}
    admissible = coverage >= 0.9 and all(value >= 0.8 for value in per_event_coverage.values())
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8") as handle:
        for row in enriched:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    audit = {
        "schema": "rq3_sar_provenance_attachment_v1",
        "input_manifest": str(args.input_manifest.resolve()),
        "input_manifest_sha256": sha256_file(args.input_manifest),
        "mapping": str(args.mapping.resolve()),
        "mapping_sha256": sha256_file(args.mapping),
        "output_manifest_sha256": sha256_file(args.output_manifest),
        "sample_count": len(rows),
        "mapped_sample_count": len(mapping),
        "high_confidence_coverage": coverage,
        "per_event_high_confidence_coverage": per_event_coverage,
        "sensor_branch_admissible": admissible,
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
