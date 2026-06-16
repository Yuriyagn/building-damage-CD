#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from stage1_closeout_utils import read_jsonl, resolve_data_path, write_json


EXPECTED_COUNTS = {"train": 1543, "val": 506, "test": 564}
PRIOR_TYPES = ["no_prior", "oracle_prior", "predicted_prior"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-errors", type=int, default=200)
    return parser.parse_args()


def add_error(errors: list[str], message: str, max_errors: int) -> None:
    if len(errors) < max_errors:
        errors.append(message)


def load_image_array(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path))


def check_image_exists(path: Path, errors: list[str], max_errors: int, label: str) -> np.ndarray | None:
    if not path.exists():
        add_error(errors, f"missing {label}: {path}", max_errors)
        return None
    try:
        return load_image_array(path)
    except Exception as exc:  # noqa: BLE001
        add_error(errors, f"cannot read {label}: {path}: {exc}", max_errors)
        return None


def check_values(name: str, arr: np.ndarray, allowed: set[int], errors: list[str], max_errors: int) -> None:
    values = set(int(x) for x in np.unique(arr))
    if not values.issubset(allowed):
        add_error(errors, f"{name} has invalid values: {sorted(values)} allowed={sorted(allowed)}", max_errors)


def main() -> None:
    args = parse_args()
    manifest_root = Path(args.manifest_root)
    data_root = Path(args.data_root) if args.data_root else manifest_root.resolve().parents[2]
    errors: list[str] = []
    counts: dict[str, dict[str, int]] = {}
    ids_by_group: dict[tuple[str, str], list[str]] = {}

    for split_name, expected in EXPECTED_COUNTS.items():
        master_path = manifest_root / f"stage2_master_{split_name}.jsonl"
        master_rows = read_jsonl(master_path) if master_path.exists() else []
        counts[f"master_{split_name}"] = {"count": len(master_rows), "expected": expected}
        if len(master_rows) != expected:
            add_error(errors, f"master {split_name} count {len(master_rows)} != {expected}", args.max_errors)

        for prior_type in PRIOR_TYPES:
            path = manifest_root / prior_type / f"{split_name}.jsonl"
            rows = read_jsonl(path) if path.exists() else []
            counts[f"{prior_type}_{split_name}"] = {"count": len(rows), "expected": expected}
            ids_by_group[(prior_type, split_name)] = [str(row.get("id", "")) for row in rows]
            if len(rows) != expected:
                add_error(errors, f"{prior_type} {split_name} count {len(rows)} != {expected}", args.max_errors)

            for row in tqdm(rows, desc=f"check_{prior_type}_{split_name}"):
                sample_id = row.get("id", "")
                for field in ["pre_image", "post_sar", "mask_multiclass"]:
                    arr = check_image_exists(resolve_data_path(data_root, row[field]), errors, args.max_errors, f"{field} id={sample_id}")
                    if field == "mask_multiclass" and arr is not None:
                        check_values(f"mask_multiclass id={sample_id}", arr, {0, 1, 2, 3}, errors, args.max_errors)

                if prior_type == "no_prior":
                    if row.get("building_prior") is not None:
                        add_error(errors, f"no_prior building_prior not null id={sample_id}", args.max_errors)
                    continue

                if prior_type == "oracle_prior":
                    arr = check_image_exists(resolve_data_path(data_root, row["building_prior"]), errors, args.max_errors, f"oracle prior id={sample_id}")
                    if arr is not None:
                        check_values(f"oracle prior id={sample_id}", arr, {0, 1, 255}, errors, args.max_errors)
                    continue

                prob_path = resolve_data_path(data_root, row["building_prior"])
                binary_path = resolve_data_path(data_root, row["building_prior_binary"])
                uint8_path = resolve_data_path(data_root, row["building_prior_uint8"])
                if not prob_path.exists():
                    add_error(errors, f"missing predicted prob npz id={sample_id}: {prob_path}", args.max_errors)
                else:
                    try:
                        with np.load(prob_path) as data:
                            prob = data["prob"]
                        if prob.shape != (1024, 1024):
                            add_error(errors, f"predicted prob shape id={sample_id}: {prob.shape}", args.max_errors)
                        if float(np.nanmin(prob)) < 0.0 or float(np.nanmax(prob)) > 1.0:
                            add_error(errors, f"predicted prob outside [0,1] id={sample_id}", args.max_errors)
                    except Exception as exc:  # noqa: BLE001
                        add_error(errors, f"cannot read predicted prob id={sample_id}: {prob_path}: {exc}", args.max_errors)
                for label, path_value in [("predicted binary", binary_path), ("predicted uint8", uint8_path)]:
                    arr = check_image_exists(path_value, errors, args.max_errors, f"{label} id={sample_id}")
                    if arr is not None and arr.shape[:2] != (1024, 1024):
                        add_error(errors, f"{label} shape id={sample_id}: {arr.shape}", args.max_errors)
                arr = check_image_exists(binary_path, errors, args.max_errors, f"predicted binary id={sample_id}")
                if arr is not None:
                    check_values(f"predicted binary id={sample_id}", arr, {0, 255}, errors, args.max_errors)

    for split_name in EXPECTED_COUNTS:
        ref = ids_by_group.get(("no_prior", split_name), [])
        for prior_type in ["oracle_prior", "predicted_prior"]:
            other = ids_by_group.get((prior_type, split_name), [])
            if ref != other:
                add_error(errors, f"id order mismatch no_prior vs {prior_type} split={split_name}", args.max_errors)

    result = {
        "status": "pass" if not errors else "fail",
        "manifest_root": str(manifest_root),
        "data_root": str(data_root),
        "counts": counts,
        "error_count": len(errors),
        "errors": errors,
    }
    write_json(args.out, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
