#!/usr/bin/env python3
"""Fit train-fold-only SAR median/MAD statistics with group/provider/global fallback."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def robust(values: list[np.ndarray]) -> dict[str, float]:
    merged = np.concatenate(values).astype(np.float32) if values else np.asarray([0.0], dtype=np.float32)
    median = float(np.median(merged))
    mad = float(np.median(np.abs(merged - median)))
    return {"median": median, "mad": max(mad, 1.0 / 255.0), "pixel_count": int(merged.size)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--max-background-pixels-per-sample", type=int, default=8192)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite calibration: {args.output}")
    groups: dict[str, list[np.ndarray]] = defaultdict(list)
    providers: dict[str, list[np.ndarray]] = defaultdict(list)
    group_events: dict[str, set[str]] = defaultdict(set)
    provider_events: dict[str, set[str]] = defaultdict(set)
    all_values: list[np.ndarray] = []
    sample_background_medians: list[float] = []
    incidence_angles: list[float] = []
    gsd_values: list[float] = []
    rng = np.random.default_rng(20260820)
    sample_count = 0
    with args.train_manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row: dict[str, Any] = json.loads(line)
            sar_path = Path(row["post_sar"])
            mask_path = Path(row["mask_multiclass"])
            if not sar_path.is_absolute():
                sar_path = args.data_root / sar_path
            if not mask_path.is_absolute():
                mask_path = args.data_root / mask_path
            sar = np.asarray(Image.open(sar_path).convert("L"), dtype=np.float32) / 255.0
            mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
            values = sar[mask == 0]
            if values.size > args.max_background_pixels_per_sample:
                values = rng.choice(values, args.max_background_pixels_per_sample, replace=False)
            provenance = dict(row.get("sar_provenance") or {})
            provider = str(provenance.get("provider") or row.get("sar_provider") or "unknown")
            mode = str(provenance.get("mode") or row.get("sar_mode") or "unknown")
            pol = str(provenance.get("polarization") or row.get("sar_polarization") or "unknown")
            group = "|".join((provider, mode, pol))
            groups[group].append(values)
            providers[provider].append(values)
            event_id = str(row.get("event_id") or "unknown")
            group_events[group].add(event_id)
            provider_events[provider].add(event_id)
            all_values.append(values)
            sample_background_medians.append(float(np.median(values)))
            def numeric(name: str, alias: str) -> float:
                value = provenance.get(name, provenance.get(alias, row.get(f"sar_{name}", row.get(f"sar_{alias}"))))
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return float("nan")
            incidence_angles.append(numeric("incidence_angle", "incidence_angle_deg"))
            gsd_values.append(numeric("gsd", "gsd_m"))
            sample_count += 1
    covariates = np.asarray([incidence_angles, gsd_values], dtype=np.float64).T
    response = np.asarray(sample_background_medians, dtype=np.float64)
    references = np.nanmedian(covariates, axis=0)
    valid_columns = np.isfinite(references)
    valid_rows = np.isfinite(response) & np.all(np.isfinite(covariates[:, valid_columns]), axis=1)
    coefficients = np.zeros(2, dtype=np.float64)
    if valid_columns.any() and int(valid_rows.sum()) >= 10:
        design = covariates[valid_rows][:, valid_columns] - references[valid_columns]
        coefficients[valid_columns] = np.linalg.lstsq(design, response[valid_rows] - np.median(response[valid_rows]), rcond=None)[0]
    payload = {
        "schema_version": "rq3_sar_calibration_v1",
        "fit_manifest": str(args.train_manifest.resolve()),
        "fit_split": "train_only_background_pixels",
        "sample_count": sample_count,
        "groups": {
            key: {**robust(values), "event_count": len(group_events[key])}
            for key, values in sorted(groups.items())
            if len(group_events[key]) >= 2 and "unknown" not in key.split("|")
        },
        "providers": {
            key: {**robust(values), "event_count": len(provider_events[key])}
            for key, values in sorted(providers.items())
            if len(provider_events[key]) >= 2 and key != "unknown"
        },
        "excluded_groups": sorted(
            key for key in groups
            if len(group_events[key]) < 2 or "unknown" in key.split("|")
        ),
        "excluded_providers": sorted(
            key for key in providers
            if len(provider_events[key]) < 2 or key == "unknown"
        ),
        "global": robust(all_values),
        "fallback_order": ["provider_mode_polarization", "provider", "global"],
        "continuous_modulation": {
            "fit_on": "train_sample_background_medians_only",
            "incidence_angle_reference": float(references[0]) if np.isfinite(references[0]) else None,
            "incidence_angle_coefficient": float(coefficients[0]),
            "gsd_reference": float(references[1]) if np.isfinite(references[1]) else None,
            "gsd_coefficient": float(coefficients[1]),
        },
        "physical_calibration_claim": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "sample_count": sample_count,
        "admissible_group_count": sum(
            len(events) >= 2 and "unknown" not in key.split("|")
            for key, events in group_events.items()
        ),
        "admissible_provider_count": sum(
            len(events) >= 2 and key != "unknown"
            for key, events in provider_events.items()
        ),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
