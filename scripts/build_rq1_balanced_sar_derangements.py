#!/usr/bin/env python3
"""Build fixed audited C3 SAR derangements for finalized RQ1 fold roles."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment


ROLES = ("inner_train", "inner_val", "final_train", "outer_eval")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve(data_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else data_root / path


def load_features(row: dict[str, Any], data_root: Path) -> tuple[tuple[int, int], float, np.ndarray]:
    sar = np.asarray(Image.open(resolve(data_root, str(row["post_sar"]))).convert("L"), dtype=np.float32) / 255.0
    with np.load(resolve(data_root, str(row["building_prior"]))) as payload:
        prior = np.asarray(payload["prob"], dtype=np.float32)
    if prior.shape != sar.shape:
        raise ValueError(f"SAR/prior shape mismatch for {row['id']}: {sar.shape} vs {prior.shape}")
    stats = np.asarray([sar.mean(), sar.std(), *np.percentile(sar, (10, 50, 90))], dtype=np.float64)
    return sar.shape, float(prior.mean()), stats


def deciles(values: list[float]) -> np.ndarray:
    order = np.argsort(np.asarray(values), kind="stable")
    result = np.empty(len(values), dtype=np.int64)
    for rank, index in enumerate(order):
        result[index] = min(9, (10 * rank) // max(len(values), 1))
    return result


def build(rows: list[dict[str, Any]], data_root: Path) -> dict[str, Any]:
    shapes: list[tuple[int, int]] = []
    areas: list[float] = []
    stats: list[np.ndarray] = []
    for row in rows:
        shape, area, stat = load_features(row, data_root)
        shapes.append(shape)
        areas.append(area)
        stats.append(stat)
    area_deciles = deciles(areas)
    groups: dict[tuple[str, str, tuple[int, int]], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        key = (str(row["event_group_id"]), str(row.get("source_dataset", "unknown")), shapes[index])
        groups[key].append(index)
    mapping: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    for key in sorted(groups, key=str):
        indices = groups[key]
        if len(indices) < 2:
            raise ValueError(f"singleton event/source/shape stratum cannot be deranged: {key}")
        features = np.stack([stats[index] for index in indices])
        scale = np.std(features, axis=0)
        scale[scale < 1e-6] = 1.0
        normalized = (features - np.mean(features, axis=0)) / scale
        cost = np.abs(normalized[:, None, :] - normalized[None, :, :]).mean(axis=2)
        group_deciles = area_deciles[indices]
        decile_distance = np.abs(group_deciles[:, None] - group_deciles[None, :])
        cost += 5.0 * np.maximum(decile_distance - 1, 0)
        np.fill_diagonal(cost, 1e9)
        targets, sources = linear_sum_assignment(cost)
        if len(targets) != len(indices) or any(int(target) == int(source) for target, source in zip(targets, sources)):
            raise ValueError(f"failed full derangement in stratum {key}")
        for target_local, source_local in zip(targets, sources):
            target = indices[int(target_local)]
            source = indices[int(source_local)]
            target_id = str(rows[target]["id"])
            source_id = str(rows[source]["id"])
            mapping[target_id] = source_id
            records.append({
                "target_id": target_id,
                "source_id": source_id,
                "event_group_id": key[0],
                "source_dataset": key[1],
                "image_shape": list(key[2]),
                "target_prior_area_decile": int(area_deciles[target]),
                "source_prior_area_decile": int(area_deciles[source]),
                "prior_area_decile_distance": int(abs(area_deciles[target] - area_deciles[source])),
                "nuisance_feature_l1": float(np.abs(normalized[int(target_local)] - normalized[int(source_local)]).mean()),
            })
    if set(mapping) != {str(row["id"]) for row in rows} or len(set(mapping.values())) != len(rows):
        raise ValueError("derangement is not a full bijection")
    close_fraction = float(np.mean([record["prior_area_decile_distance"] <= 1 for record in records]))
    if close_fraction < 0.90:
        raise ValueError(f"balanced derangement gate failed: decile-distance<=1 fraction={close_fraction:.4f}")
    return {
        "mapping": mapping,
        "records": records,
        "audit": {
            "count": len(rows),
            "stratum_count": len(groups),
            "bijection": True,
            "self_pair_count": 0,
            "cross_event_count": 0,
            "cross_source_dataset_count": 0,
            "cross_shape_count": 0,
            "prior_area_decile_distance_le_1_fraction": close_fraction,
            "sensor_acquisition_balance": "unknown_not_available_in_manifests",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest_root = args.manifest_root.resolve()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite derangements: {output_root}")
    summaries = []
    for outer in range(7):
        for role in ROLES:
            rows = read_jsonl(manifest_root / f"outer_{outer}" / f"{role}.jsonl")
            payload = build(rows, data_root)
            path = output_root / f"outer_{outer}" / f"{role}.json"
            write_json(path, payload)
            summaries.append({"outer_fold": outer, "role": role, **payload["audit"]})
    write_json(output_root / "AUDIT.json", {"hard_error_count": 0, "derangements": summaries})
    print(json.dumps({"status": "completed", "derangement_count": len(summaries)}, sort_keys=True))


if __name__ == "__main__":
    main()
