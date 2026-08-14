#!/usr/bin/env python3
"""Audit only the train/validation evidence allowed by metadata experiment v1."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm


REQUIRED_FIELDS = (
    "id",
    "event_id",
    "disaster_type",
    "pre_image",
    "post_sar",
    "mask_multiclass",
)
PATH_FIELDS = ("pre_image", "post_sar", "mask_multiclass")
DISASTER_CLASSES = {
    "conflict",
    "earthquake",
    "explosion",
    "fire",
    "flood",
    "hurricane",
    "volcano",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_split(
    split: str,
    rows: list[dict[str, Any]],
    data_root: Path,
) -> tuple[dict[str, Any], list[str], dict[str, tuple[str, str, str]]]:
    errors: list[str] = []
    ids = [str(row.get("id", "")) for row in rows]
    duplicate_ids = sorted(value for value, count in Counter(ids).items() if value and count > 1)
    if duplicate_ids:
        errors.append(f"{split}: duplicate ids: {duplicate_ids[:10]}")
    paths: dict[Path, str] = {}
    resolved_rows: list[tuple[str, dict[str, Path]]] = []
    disaster_counts: Counter[str] = Counter()
    event_counts: Counter[str] = Counter()
    for index, row in enumerate(rows):
        missing = [field for field in REQUIRED_FIELDS if not str(row.get(field, "") or "")]
        if missing:
            errors.append(f"{split}[{index}]: missing fields {missing}")
            continue
        disaster = str(row["disaster_type"])
        if disaster not in DISASTER_CLASSES:
            errors.append(f"{split}[{index}]: unknown disaster_type {disaster!r}")
        disaster_counts[disaster] += 1
        event_counts[str(row["event_id"])] += 1
        sample_paths = {field: resolve(data_root, row[field]).resolve() for field in PATH_FIELDS}
        for path in sample_paths.values():
            if not path.is_file():
                errors.append(f"{split}[{index}]: missing file {path}")
            else:
                paths[path] = ""
        resolved_rows.append((str(row["id"]), sample_paths))

    existing_paths = sorted(paths)
    with ThreadPoolExecutor(max_workers=8) as executor:
        hashes = dict(
            zip(
                existing_paths,
                tqdm(
                    executor.map(sha256, existing_paths),
                    total=len(existing_paths),
                    desc=f"hash_{split}",
                    leave=False,
                ),
            )
        )
    full_hashes: dict[str, tuple[str, str, str]] = {}
    mask_pixels = Counter()
    for sample_id, sample_paths in tqdm(resolved_rows, desc=f"validate_{split}", leave=False):
        if not all(path in hashes for path in sample_paths.values()):
            continue
        try:
            with Image.open(sample_paths["pre_image"]) as pre_image:
                pre_size = pre_image.size
            with Image.open(sample_paths["post_sar"]) as sar_image:
                sar_size = sar_image.size
            with Image.open(sample_paths["mask_multiclass"]) as mask_image:
                mask = np.asarray(mask_image.convert("L"), dtype=np.uint8)
                mask_size = mask_image.size
            if not (pre_size == sar_size == mask_size):
                errors.append(
                    f"{split}:{sample_id}: shape mismatch pre={pre_size} sar={sar_size} mask={mask_size}"
                )
            values, counts = np.unique(mask, return_counts=True)
            invalid = sorted(int(value) for value in values if int(value) not in {0, 1, 2, 3})
            if invalid:
                errors.append(f"{split}:{sample_id}: invalid mask labels {invalid}")
            mask_pixels.update({int(value): int(count) for value, count in zip(values, counts)})
        except Exception as exc:
            errors.append(f"{split}:{sample_id}: image validation failed: {exc}")
        full_hashes[sample_id] = tuple(hashes[sample_paths[field]] for field in PATH_FIELDS)
    return (
        {
            "sample_count": len(rows),
            "unique_id_count": len(set(ids)),
            "event_counts": dict(sorted(event_counts.items())),
            "disaster_counts": dict(sorted(disaster_counts.items())),
            "mask_pixel_counts": {str(key): int(value) for key, value in sorted(mask_pixels.items())},
            "hashed_file_count": len(existing_paths),
        },
        errors,
        full_hashes,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    manifest_paths = {"train": Path(args.train_manifest), "val": Path(args.val_manifest)}
    rows_by_split = {split: read_jsonl(path) for split, path in manifest_paths.items()}
    summaries: dict[str, Any] = {}
    hashes_by_split: dict[str, dict[str, tuple[str, str, str]]] = {}
    errors: list[str] = []
    for split in ("train", "val"):
        summary, split_errors, full_hashes = audit_split(split, rows_by_split[split], data_root)
        summaries[split] = summary
        errors.extend(split_errors)
        hashes_by_split[split] = full_hashes
    train_events = {str(row["event_id"]) for row in rows_by_split["train"]}
    val_events = {str(row["event_id"]) for row in rows_by_split["val"]}
    event_overlap = sorted(train_events & val_events)
    if event_overlap:
        errors.append(f"train/val event overlap: {event_overlap}")
    reverse: dict[tuple[str, str, str], list[tuple[str, str]]] = defaultdict(list)
    for split, mapping in hashes_by_split.items():
        for sample_id, fingerprint in mapping.items():
            reverse[fingerprint].append((split, sample_id))
    cross_duplicates = [members for members in reverse.values() if len({split for split, _ in members}) > 1]
    if cross_duplicates:
        errors.append(f"train/val exact full-sample duplicate groups: {len(cross_duplicates)}")
    payload = {
        "protocol": "stage2_metadata_multitask_v1_train_val_only",
        "splits_read": ["train", "val"],
        "test_read": False,
        "data_root": str(data_root),
        "manifests": {key: str(value.resolve()) for key, value in manifest_paths.items()},
        "split_summaries": summaries,
        "event_overlap": event_overlap,
        "cross_split_full_sample_duplicate_group_count": len(cross_duplicates),
        "cross_split_full_sample_duplicate_examples": cross_duplicates[:20],
        "hard_error_count": len(errors),
        "hard_errors": errors[:200],
        "status": "ok" if not errors else "error",
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(output), "status": payload["status"], "hard_error_count": len(errors)}))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
