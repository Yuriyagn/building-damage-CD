#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from stage1_closeout_utils import read_jsonl, resolve_data_path, write_json


SPLITS = ["train", "val", "test"]
PRIOR_TYPES = ["no_prior", "oracle_prior", "predicted_prior"]
np: Any = None
Image: Any = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - keeps --help/lightweight runs dependency-free
    def tqdm(iterable, **_: Any):  # type: ignore[no-redef]
        return iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit-per-split", type=int, default=8)
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def check_image(name: str, arr: np.ndarray, errors: list[str]) -> None:
    if arr.shape[:2] != (1024, 1024):
        errors.append(f"{name} shape is {arr.shape}, expected 1024x1024")


def smoke_row(data_root: Path, row: dict[str, Any], prior_type: str, errors: list[str]) -> dict[str, Any]:
    sample_id = str(row.get("id", ""))
    pre = load_rgb(resolve_data_path(data_root, row["pre_image"]))
    post_sar = load_rgb(resolve_data_path(data_root, row["post_sar"]))
    mask = load_mask(resolve_data_path(data_root, row["mask_multiclass"]))
    check_image(f"pre_image id={sample_id}", pre, errors)
    check_image(f"post_sar id={sample_id}", post_sar, errors)
    check_image(f"mask_multiclass id={sample_id}", mask, errors)
    mask_values = set(int(x) for x in np.unique(mask))
    if not mask_values.issubset({0, 1, 2, 3}):
        errors.append(f"mask_multiclass id={sample_id} invalid values {sorted(mask_values)}")

    record: dict[str, Any] = {
        "id": sample_id,
        "pre_shape": list(pre.shape),
        "post_sar_shape": list(post_sar.shape),
        "mask_shape": list(mask.shape),
        "mask_values": sorted(mask_values),
    }

    if prior_type == "no_prior":
        if row.get("building_prior") is not None:
            errors.append(f"no_prior id={sample_id} has non-null building_prior")
        return record

    prior_binary = load_mask(resolve_data_path(data_root, row["building_prior_binary"]))
    check_image(f"building_prior_binary id={sample_id}", prior_binary, errors)
    prior_values = set(int(x) for x in np.unique(prior_binary))
    if not prior_values.issubset({0, 1, 255}):
        errors.append(f"building_prior_binary id={sample_id} invalid values {sorted(prior_values)}")
    record["building_prior_binary_shape"] = list(prior_binary.shape)
    record["building_prior_binary_values"] = sorted(prior_values)

    if prior_type == "predicted_prior":
        prob_path = resolve_data_path(data_root, row["building_prior"])
        with np.load(prob_path) as data:
            prob = data["prob"]
        if prob.shape != (1024, 1024):
            errors.append(f"predicted prob id={sample_id} shape is {prob.shape}")
        if float(np.nanmin(prob)) < 0.0 or float(np.nanmax(prob)) > 1.0:
            errors.append(f"predicted prob id={sample_id} outside [0, 1]")
        record["predicted_prob_shape"] = list(prob.shape)
        record["predicted_prob_min"] = float(np.nanmin(prob))
        record["predicted_prob_max"] = float(np.nanmax(prob))
    return record


def main() -> None:
    args = parse_args()
    global np, Image
    import numpy as np_module
    from PIL import Image as image_module

    np = np_module
    Image = image_module
    data_root = Path(args.data_root)
    manifest_root = Path(args.manifest_root)
    errors: list[str] = []
    examples: dict[str, Any] = {}
    counts: dict[str, int] = {}

    for prior_type in PRIOR_TYPES:
        for split_name in SPLITS:
            manifest = manifest_root / prior_type / f"{split_name}.jsonl"
            rows = read_jsonl(manifest)
            counts[f"{prior_type}_{split_name}"] = len(rows)
            sample_rows = rows[: args.limit_per_split]
            key = f"{prior_type}_{split_name}"
            examples[key] = []
            for row in tqdm(sample_rows, desc=f"smoke_{key}"):
                try:
                    examples[key].append(smoke_row(data_root, row, prior_type, errors))
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{key} id={row.get('id', '')}: {exc}")

    result = {
        "status": "pass" if not errors else "fail",
        "data_root": str(data_root),
        "manifest_root": str(manifest_root),
        "limit_per_split": args.limit_per_split,
        "counts": counts,
        "error_count": len(errors),
        "errors": errors,
        "examples": examples,
    }
    write_json(args.out, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
