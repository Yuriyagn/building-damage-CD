#!/usr/bin/env python3
"""Recover BRIGHT GeoTIFF identity and georeferencing for an RQ3 sidecar.

This is deliberately narrower than SAR provenance recovery.  An exact content
match can recover the official BRIGHT tile and its geographic footprint, but it
does not identify the original Capella/Umbra collect or establish radiometric
calibration.  Ambiguous hashes are retained as ambiguous instead of selecting a
candidate by filename or event name.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import rasterio
from PIL import Image


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_grayscale_pixels(path: Path) -> str:
    """Hash decoded uint8 grayscale pixels with their dimensions.

    BRIGHT derivatives can preserve every SAR pixel while changing TIFF tags or
    re-encoding the container.  This canonical pixel digest intentionally
    ignores such container differences and is kept separate from the file SHA.
    """
    with Image.open(path) as source:
        image = source.convert("L")
        digest = hashlib.sha256()
        digest.update(f"L:{image.width}x{image.height}:".encode("ascii"))
        digest.update(image.tobytes())
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _hash_official_path(args: tuple[Path, Path]) -> dict[str, Any]:
    root, path = args
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "file_sha256": sha256_file(path),
        "pixel_sha256": sha256_grayscale_pixels(path),
        "size_bytes": path.stat().st_size,
    }


def index_official_tiffs(root: Path, workers: int = 8) -> list[dict[str, Any]]:
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".tif", ".tiff"})
    if not paths:
        raise ValueError(f"no GeoTIFFs found under official root: {root}")
    worker_count = max(1, workers)
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        rows = list(pool.map(_hash_official_path, ((root, path) for path in paths)))
    return sorted(rows, key=lambda row: str(row["relative_path"]))


def load_official_index(path: Path, official_root: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    required = {"relative_path", "file_sha256", "pixel_sha256", "size_bytes"}
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"official index row missing {sorted(missing)}")
        source = official_root / str(row["relative_path"])
        if not source.is_file():
            raise FileNotFoundError(f"indexed official GeoTIFF is absent: {source}")
        if int(row["size_bytes"]) != source.stat().st_size:
            raise ValueError(f"indexed official GeoTIFF size changed: {source}")
    return sorted(rows, key=lambda row: str(row["relative_path"]))


def geotiff_metadata(path: Path) -> dict[str, Any]:
    with rasterio.open(path) as dataset:
        crs = dataset.crs
        transform = dataset.transform
        bounds = dataset.bounds
        return {
            "width": dataset.width,
            "height": dataset.height,
            "count": dataset.count,
            "dtypes": list(dataset.dtypes),
            "crs": crs.to_string() if crs is not None else None,
            "epsg": crs.to_epsg() if crs is not None else None,
            "coordinate_unit": getattr(crs, "linear_units", None) if crs is not None else None,
            "transform": [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f],
            "bounds": {
                "left": bounds.left,
                "bottom": bounds.bottom,
                "right": bounds.right,
                "top": bounds.top,
            },
            "has_crs": crs is not None,
        }


def sidecar_post_sar_sha256(row: dict[str, Any]) -> str:
    hashes = row.get("source_sha256")
    if isinstance(hashes, dict) and hashes.get("post_sar"):
        return str(hashes["post_sar"])
    source = Path(str(row["post_sar"]))
    if not source.is_file():
        raise FileNotFoundError(f"sidecar post_sar is absent and has no frozen hash: {source}")
    return sha256_file(source)


def recover_rows(
    sidecar_rows: list[dict[str, Any]],
    official_index: list[dict[str, Any]],
    official_root: Path,
) -> list[dict[str, Any]]:
    by_pixel_digest: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in official_index:
        by_pixel_digest[str(row["pixel_sha256"])].append(row)

    metadata_cache: dict[str, dict[str, Any]] = {}
    recovered: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in sidecar_rows:
        sample_id = str(row["id"])
        if sample_id in seen_ids:
            raise ValueError(f"duplicate sidecar sample id: {sample_id}")
        seen_ids.add(sample_id)
        file_digest = sidecar_post_sar_sha256(row)
        pixel_digest = sha256_grayscale_pixels(Path(str(row["post_sar"])))
        matches = sorted(by_pixel_digest.get(pixel_digest, []), key=lambda item: str(item["relative_path"]))
        if not matches:
            match_status = "unmatched"
        elif len(matches) == 1:
            match_status = "unique"
        else:
            match_status = "ambiguous"

        item: dict[str, Any] = {
            "sample_id": sample_id,
            "event_id": str(row["event_id"]),
            "sidecar_post_sar": str(row["post_sar"]),
            "post_sar_file_sha256": file_digest,
            "post_sar_pixel_sha256": pixel_digest,
            "content_match_status": match_status,
            "official_candidate_count": len(matches),
            "official_candidate_relative_paths": [str(match["relative_path"]) for match in matches],
            "official_binary_exact_candidate_count": sum(
                str(match["file_sha256"]) == file_digest for match in matches
            ),
            "georeference_status": "not_recovered",
        }
        if len(matches) == 1:
            relative_path = str(matches[0]["relative_path"])
            if relative_path not in metadata_cache:
                metadata_cache[relative_path] = geotiff_metadata(official_root / relative_path)
            metadata = metadata_cache[relative_path]
            item["official_relative_path"] = relative_path
            item["official_geotiff"] = metadata
            item["georeference_status"] = "recovered" if metadata["has_crs"] else "matched_without_crs"
        recovered.append(item)
    return sorted(recovered, key=lambda row: str(row["sample_id"]))


def build_audit(
    sidecar_path: Path,
    official_root: Path,
    official_index_path: Path,
    mapping_path: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    per_event_counts: dict[str, dict[str, int]] = defaultdict(lambda: {
        "sample_count": 0,
        "unique_match_count": 0,
        "ambiguous_match_count": 0,
        "unmatched_count": 0,
        "georeference_recovered_count": 0,
    })
    for row in rows:
        counts = per_event_counts[str(row["event_id"])]
        counts["sample_count"] += 1
        counts[f"{row['content_match_status']}_match_count" if row["content_match_status"] != "unmatched" else "unmatched_count"] += 1
        if row["georeference_status"] == "recovered":
            counts["georeference_recovered_count"] += 1

    sample_count = len(rows)
    unique_count = sum(row["content_match_status"] == "unique" for row in rows)
    ambiguous_count = sum(row["content_match_status"] == "ambiguous" for row in rows)
    recovered_count = sum(row["georeference_status"] == "recovered" for row in rows)
    per_event: dict[str, dict[str, Any]] = {}
    for event, counts in sorted(per_event_counts.items()):
        total = counts["sample_count"]
        per_event[event] = {
            **counts,
            "any_content_match_coverage": (counts["unique_match_count"] + counts["ambiguous_match_count"]) / total,
            "unique_georeference_coverage": counts["georeference_recovered_count"] / total,
        }
    return {
        "schema": "rq3_bright_georeference_recovery_v1",
        "input_sidecar": str(sidecar_path.resolve()),
        "input_sidecar_sha256": sha256_file(sidecar_path),
        "official_post_root": str(official_root.resolve()),
        "official_index_sha256": sha256_file(official_index_path),
        "mapping_sha256": sha256_file(mapping_path),
        "sample_count": sample_count,
        "event_count": len(per_event),
        "unique_content_match_count": unique_count,
        "ambiguous_content_match_count": ambiguous_count,
        "unmatched_content_count": sample_count - unique_count - ambiguous_count,
        "any_content_match_coverage": (unique_count + ambiguous_count) / max(sample_count, 1),
        "unique_georeference_recovered_count": recovered_count,
        "unique_georeference_coverage": recovered_count / max(sample_count, 1),
        "per_event": per_event,
        "confidence_policy": (
            "exact SHA-256 of canonical decoded uint8 grayscale dimensions and pixels; "
            "file SHA-256 is retained separately and ambiguous pixel hashes are never auto-selected"
        ),
        "claim_boundary": (
            "This recovers BRIGHT tile identity and geographic footprint only. It does not identify the original "
            "Capella/Umbra collect, prove radiometric calibration, or satisfy the RQ3 E3 high-confidence sensor-metadata gate."
        ),
        "sensor_branch_admissible_from_this_artifact": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-sidecar", type=Path, required=True)
    parser.add_argument("--official-post-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reuse-official-index", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    sidecar_path = args.input_sidecar.resolve()
    official_root = args.official_post_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite georeference recovery: {output_dir}")
    if not sidecar_path.is_file():
        raise FileNotFoundError(sidecar_path)
    if not official_root.is_dir():
        raise NotADirectoryError(official_root)
    output_dir.mkdir(parents=True)

    if args.reuse_official_index is not None:
        official_index = load_official_index(args.reuse_official_index.resolve(), official_root)
    else:
        official_index = index_official_tiffs(official_root, workers=args.workers)
    official_index_path = output_dir / "official_geotiff_index.jsonl"
    write_jsonl(official_index_path, official_index)

    sidecar_rows = read_jsonl(sidecar_path)
    recovered = recover_rows(sidecar_rows, official_index, official_root)
    mapping_path = output_dir / "sample_georeference_mapping.jsonl"
    write_jsonl(mapping_path, recovered)
    audit = build_audit(sidecar_path, official_root, official_index_path, mapping_path, recovered)
    audit_path = output_dir / "AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
