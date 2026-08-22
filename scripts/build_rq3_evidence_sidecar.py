#!/usr/bin/env python3
"""Build the descriptive 14-event RQ3 SAR evidence sidecar without mutating manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str, data_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else data_root / path


def haar_energies(image: np.ndarray) -> dict[str, float]:
    x = image.astype(np.float32) / 255.0
    result: dict[str, float] = {}
    for level in (1, 2):
        h, w = x.shape
        x = x[: h - h % 2, : w - w % 2]
        a, b, c, d = x[0::2, 0::2], x[0::2, 1::2], x[1::2, 0::2], x[1::2, 1::2]
        bands = {
            "ll": (a + b + c + d) * 0.5,
            "lh": (a - b + c - d) * 0.5,
            "hl": (a + b - c - d) * 0.5,
            "hh": (a - b - c + d) * 0.5,
        }
        for name, band in bands.items():
            result[f"haar_l{level}_{name}_energy"] = float(np.mean(np.square(band)))
        x = bands["ll"]
    return result


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--event-effects", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-file-hashes", action="store_true")
    parser.add_argument("--reuse-hashed-samples", type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    data_root = args.data_root.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing sidecar: {out}")
    out.mkdir(parents=True)

    unique: dict[str, dict[str, Any]] = {}
    for outer in range(7):
        fold = args.manifest_root.resolve() / f"outer_{outer}"
        for name in ("final_train.jsonl", "outer_eval.jsonl"):
            for row in read_jsonl(fold / name):
                unique.setdefault(str(row["id"]), row)
    reused: dict[str, dict[str, Any]] = {}
    if args.reuse_hashed_samples is not None:
        reused = {str(row["id"]): row for row in read_jsonl(args.reuse_hashed_samples)}
        if set(reused) != set(unique):
            raise ValueError("reused hashed sample IDs do not match frozen manifests")
        if any("source_sha256" not in row for row in reused.values()):
            raise ValueError("reused samples are not fully hashed")

    effects: dict[str, dict[str, str]] = {}
    with args.event_effects.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            effects[str(row["event_id"])] = row

    event_samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sidecar_path = out / "samples.jsonl"
    with sidecar_path.open("w", encoding="utf-8") as handle:
        for sample_id in sorted(unique):
            row = unique[sample_id]
            footprint_status = str(row.get("geographic_footprint_status") or "")
            footprint = dict(row.get("geographic_footprint") or {})
            official_polygon_traceable = footprint_status in {
                "available_from_bright_target_json",
                "inherited_from_decoded_identical_bright_pair",
            }
            instance_source = row.get("instance_annotation_source")
            if not instance_source and official_polygon_traceable:
                instance_source = footprint.get("source")
            if sample_id in reused:
                item = dict(reused[sample_id])
                item.update({
                    "instances": instance_source,
                    "official_polygon_traceable": official_polygon_traceable,
                    "official_instance_metric_direct": bool(row.get("instance_annotation_source")),
                })
                handle.write(json.dumps(item, sort_keys=True) + "\n")
                event_samples[str(row["event_id"])].append(item)
                continue
            sar_path = resolve_path(str(row["post_sar"]), data_root)
            mask_path = resolve_path(str(row["mask_multiclass"]), data_root)
            prior_path = resolve_path(str(row["building_prior"]), data_root)
            sar = np.asarray(Image.open(sar_path).convert("L"), dtype=np.uint8)
            mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
            building = mask > 0
            background = ~building
            provenance = dict(row.get("sar_provenance") or {})
            provider = provenance.get("provider") or row.get("sar_provider")
            metadata_confidence = row.get("metadata_confidence") or provenance.get("confidence") or "unresolved"
            item = {
                "id": sample_id,
                "event_id": row["event_id"],
                "post_sar": str(sar_path),
                "building_prior": str(prior_path),
                "mask_multiclass": str(mask_path),
                "instances": instance_source,
                "official_polygon_traceable": official_polygon_traceable,
                "official_instance_metric_direct": bool(row.get("instance_annotation_source")),
                "instance_annotation_count": int(row.get("instance_annotation_count") or 0),
                "sar_provenance": provenance,
                "metadata_confidence": metadata_confidence,
                "provider": provider,
                "mode": provenance.get("mode") or row.get("sar_mode"),
                "polarization": provenance.get("polarization") or row.get("sar_polarization"),
                "acquisition_time": provenance.get("acquisition_time") or row.get("sar_acquisition_time"),
                "incidence_angle_deg": provenance.get("incidence_angle_deg") or row.get("sar_incidence_angle_deg"),
                "gsd_m": provenance.get("gsd_m") or row.get("sar_gsd_m"),
                "sar_mean": float(sar.mean() / 255.0),
                "sar_std": float(sar.std() / 255.0),
                "sar_building_mean": float(sar[building].mean() / 255.0) if building.any() else None,
                "sar_background_mean": float(sar[background].mean() / 255.0) if background.any() else None,
                "building_pixel_count": int(building.sum()),
                "class_pixel_counts": {str(k): int((mask == k).sum()) for k in range(4)},
                **haar_energies(sar),
            }
            if not args.skip_file_hashes:
                item["source_sha256"] = {
                    "post_sar": sha256_file(sar_path),
                    "building_prior": sha256_file(prior_path),
                    "mask_multiclass": sha256_file(mask_path),
                }
            handle.write(json.dumps(item, sort_keys=True) + "\n")
            event_samples[str(row["event_id"])].append(item)

    event_rows: list[dict[str, Any]] = []
    high_conf = {"high", "verified", "stac_exact"}
    for event_id in sorted(event_samples):
        rows = event_samples[event_id]
        event_row: dict[str, Any] = {
            "event_id": event_id,
            "sample_count": len(rows),
            "official_polygon_traceable_sample_count": sum(bool(r["official_polygon_traceable"]) for r in rows),
            "official_instance_metric_direct_sample_count": sum(bool(r["official_instance_metric_direct"]) for r in rows),
            "high_confidence_metadata_count": sum(r["metadata_confidence"] in high_conf for r in rows),
            "providers": sorted({str(r["provider"]) for r in rows if r["provider"]}),
            "sar_mean": mean_or_none([r["sar_mean"] for r in rows]),
            "sar_std": mean_or_none([r["sar_std"] for r in rows]),
            "sar_building_minus_background": mean_or_none([
                r["sar_building_mean"] - r["sar_background_mean"]
                for r in rows if r["sar_building_mean"] is not None and r["sar_background_mean"] is not None
            ]),
            "mean_building_pixels": mean_or_none([float(r["building_pixel_count"]) for r in rows]),
        }
        for key in ("haar_l1_ll_energy", "haar_l1_lh_energy", "haar_l1_hl_energy", "haar_l1_hh_energy", "haar_l2_ll_energy", "haar_l2_lh_energy", "haar_l2_hl_energy", "haar_l2_hh_energy"):
            event_row[key] = mean_or_none([float(r[key]) for r in rows])
        event_row.update(effects.get(event_id, {}))
        event_rows.append(event_row)

    (out / "event_evidence.json").write_text(json.dumps(event_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_keys = sorted({key for row in event_rows for key in row})
    with (out / "event_evidence.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_keys)
        writer.writeheader()
        writer.writerows(event_rows)
    confidence_counts = Counter(r["metadata_confidence"] for rows in event_samples.values() for r in rows)
    sample_count = len(unique)
    high_count = sum(confidence_counts[key] for key in high_conf)
    per_event_coverage = {
        event_id: sum(r["metadata_confidence"] in high_conf for r in rows) / len(rows)
        for event_id, rows in event_samples.items()
    }
    audit = {
        "schema": "rq3_sar_evidence_audit_v1",
        "sample_count": sample_count,
        "event_count": len(event_samples),
        "official_polygon_traceable_sample_count": sum(
            str(r.get("geographic_footprint_status") or "") in {
                "available_from_bright_target_json",
                "inherited_from_decoded_identical_bright_pair",
            }
            for r in unique.values()
        ),
        "official_instance_metric_direct_sample_count": sum(bool(r.get("instance_annotation_source")) for r in unique.values()),
        "metadata_confidence_counts": dict(confidence_counts),
        "high_confidence_metadata_coverage": high_count / sample_count,
        "per_event_high_confidence_coverage": per_event_coverage,
        "sensor_branch_admissible": high_count / sample_count >= 0.9 and all(v >= 0.8 for v in per_event_coverage.values()),
        "correlation_use": "descriptive_only_not_for_causal_claims_or_hyperparameter_selection",
        "file_hashes_included": bool(reused) or not args.skip_file_hashes,
        "samples_jsonl_sha256": sha256_file(sidecar_path),
    }
    (out / "AUDIT.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
