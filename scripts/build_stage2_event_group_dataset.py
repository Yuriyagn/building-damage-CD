#!/usr/bin/env python3
"""Build a real-disaster event-group split and a deduplicated BRIGHT supplement.

The source strict-v1 manifests and both source datasets are treated as read-only.
All converted BRIGHT masks are written to a separate derived-data directory, and
all manifests are emitted under new versioned roots.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw


SPLITS = ("train", "val", "test")
VARIANTS = ("predicted_prior", "oracle_prior", "no_prior")
CLASS_NAMES = ("background", "intact", "damaged", "destroyed")
CURRENT_DATASET = "disasterm3_minimal_v0.2"
BRIGHT_DATASET = "bright_cvprw26"
SAMPLE_ID_PATTERN = re.compile(r"^(?P<event>.+)_\d{8}$")


@dataclass(frozen=True)
class BrightSample:
    source_split: str
    image_id: int
    sample_id: str
    source_event_name: str
    pre_image: Path
    post_sar: Path
    target_json: Path
    width: int
    height: int
    annotations: tuple[dict[str, Any], ...]
    geographic_footprint: dict[str, Any]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise ValueError("fieldnames are required for an empty CSV")
        fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resolve_path(data_root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else data_root / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decoded_hash_task(task: tuple[str, str, str]) -> tuple[str, str]:
    key, raw_path, mode = task
    path = Path(raw_path)
    with Image.open(path) as image:
        array = np.ascontiguousarray(np.asarray(image.convert(mode), dtype=np.uint8))
    digest = hashlib.sha256()
    digest.update(f"{mode}|{array.shape}|{array.dtype}".encode("utf-8"))
    digest.update(array.tobytes())
    return key, digest.hexdigest()


def decoded_hashes(
    tasks: list[tuple[str, Path, str]], workers: int, label: str
) -> dict[str, str]:
    print(f"[INFO] decoded hashing {label}: {len(tasks)} images with {workers} workers", flush=True)
    payload = [(key, str(path), mode) for key, path, mode in tasks]
    if workers <= 1:
        results = map(_decoded_hash_task, payload)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        try:
            results = executor.map(_decoded_hash_task, payload, chunksize=8)
            output = dict(results)
        finally:
            executor.shutdown(wait=True)
        return output
    return dict(results)


def load_mapping(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    mapping: dict[tuple[str, str], dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (str(row["source_dataset"]), str(row["source_event_name"]))
            if key in mapping:
                raise ValueError(f"duplicate event mapping key: {key}")
            mapping[key] = {str(k): str(v) for k, v in row.items()}
    if not mapping:
        raise ValueError(f"empty event mapping: {path}")
    return mapping


def load_split_assignments(plan: dict[str, Any]) -> dict[str, str]:
    split_groups = dict(plan.get("split_event_groups", {}))
    if set(split_groups) != set(SPLITS):
        raise ValueError(f"split_event_groups must contain exactly {SPLITS}")
    assignments: dict[str, str] = {}
    for split in SPLITS:
        for event_group in split_groups[split]:
            event_group = str(event_group)
            if event_group in assignments:
                raise ValueError(f"event group assigned twice: {event_group}")
            assignments[event_group] = split
    return assignments


def footprint_from_target(path: Path, width: int, height: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    images = payload.get("images", [])
    if len(images) != 1 or not isinstance(images[0].get("geo_reference"), dict):
        raise ValueError(f"missing single-image geo_reference: {path}")
    geo = dict(images[0]["geo_reference"])
    origin_x = float(geo["origin_lon"])
    origin_y = float(geo["origin_lat"])
    pixel_x = float(geo["pixel_size_lon"])
    pixel_y = float(geo["pixel_size_lat"])
    tie_x = float(geo.get("tie_point_pixel_x", 0))
    tie_y = float(geo.get("tie_point_pixel_y", 0))
    xs = [origin_x + (pixel - tie_x) * pixel_x for pixel in (0, width)]
    ys = [origin_y + (pixel - tie_y) * pixel_y for pixel in (0, height)]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return {
        "epsg": int(geo["epsg"]),
        "bbox_projected": [min_x, min_y, max_x, max_y],
        "polygon_projected": [
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y],
            [min_x, min_y],
        ],
        "source": str(path),
    }


def load_bright_samples(bright_root: Path, coco_root: Path) -> list[BrightSample]:
    samples: list[BrightSample] = []
    seen_ids: set[str] = set()
    for source_split in ("train", "val"):
        payload = json.loads((coco_root / f"{source_split}.json").read_text(encoding="utf-8"))
        categories = {int(item["id"]): str(item["name"]) for item in payload.get("categories", [])}
        if categories != {1: "intact", 2: "damaged", 3: "destroyed"}:
            raise ValueError(f"unexpected BRIGHT categories in {source_split}: {categories}")
        annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in payload.get("annotations", []):
            annotations[int(annotation["image_id"])].append(annotation)
        for image in payload.get("images", []):
            post_name = str(image["file_name"])
            if not post_name.endswith("_post_disaster.tif"):
                raise ValueError(f"unexpected BRIGHT image name: {post_name}")
            sample_id = post_name.removesuffix("_post_disaster.tif")
            match = SAMPLE_ID_PATTERN.match(sample_id)
            if match is None:
                raise ValueError(f"cannot parse BRIGHT source event from {sample_id}")
            if sample_id in seen_ids:
                raise ValueError(f"duplicate BRIGHT sample id: {sample_id}")
            seen_ids.add(sample_id)
            pre = bright_root / "pre-event" / f"{sample_id}_pre_disaster.tif"
            post = bright_root / "post-event" / post_name
            target = bright_root / "target_instance_level" / f"{sample_id}_instance_damage.json"
            for required in (pre, post, target):
                if not required.is_file():
                    raise FileNotFoundError(required)
            width = int(image["width"])
            height = int(image["height"])
            samples.append(
                BrightSample(
                    source_split=source_split,
                    image_id=int(image["id"]),
                    sample_id=sample_id,
                    source_event_name=match.group("event"),
                    pre_image=pre,
                    post_sar=post,
                    target_json=target,
                    width=width,
                    height=height,
                    annotations=tuple(annotations[int(image["id"])]),
                    geographic_footprint=footprint_from_target(target, width, height),
                )
            )
    samples.sort(key=lambda sample: sample.sample_id)
    return samples


def rasterize_annotations(sample: BrightSample) -> np.ndarray:
    image = Image.new("L", (sample.width, sample.height), 0)
    draw = ImageDraw.Draw(image)
    for annotation in sample.annotations:
        category = int(annotation["category_id"])
        if category not in {1, 2, 3}:
            raise ValueError(f"invalid BRIGHT category {category} in {sample.sample_id}")
        segmentation = annotation.get("segmentation")
        if not isinstance(segmentation, list):
            raise ValueError(f"non-polygon segmentation in {sample.sample_id}")
        for polygon in segmentation:
            if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
                raise ValueError(f"invalid polygon in {sample.sample_id}")
            points = [(float(polygon[i]), float(polygon[i + 1])) for i in range(0, len(polygon), 2)]
            draw.polygon(points, fill=category)
    return np.asarray(image, dtype=np.uint8)


def load_current_bundle(
    source_manifest_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[tuple[str, str], dict[str, Any]]]]:
    masters: list[dict[str, Any]] = []
    variants: dict[str, dict[tuple[str, str], dict[str, Any]]] = {
        variant: {} for variant in VARIANTS
    }
    for directory_split in SPLITS:
        for row in read_jsonl(source_manifest_root / f"stage2_master_{directory_split}.jsonl"):
            item = dict(row)
            item["_directory_split_v1"] = directory_split
            masters.append(item)
        for variant in VARIANTS:
            for row in read_jsonl(source_manifest_root / variant / f"{directory_split}.jsonl"):
                key = (directory_split, str(row["id"]))
                if key in variants[variant]:
                    raise ValueError(f"duplicate current variant row: {variant}/{key}")
                variants[variant][key] = row
    if len({str(row["id"]) for row in masters}) != len(masters):
        raise ValueError("current strict source IDs are not globally unique")
    return masters, variants


def mapping_for_current(
    row: dict[str, Any], mapping: dict[tuple[str, str], dict[str, str]]
) -> dict[str, str]:
    source_event = str(row.get("source_event_id") or row["event_id"])
    key = (CURRENT_DATASET, source_event)
    if key not in mapping:
        raise KeyError(f"missing current event mapping: {key}")
    return mapping[key]


def transform_current_row(
    row: dict[str, Any],
    event_mapping: dict[str, str],
    new_split: str,
    directory_split: str,
    geographic_footprint: dict[str, Any] | None,
    plan: dict[str, Any],
) -> dict[str, Any]:
    output = {key: value for key, value in row.items() if not key.startswith("_")}
    output.update(
        {
            "source_dataset": CURRENT_DATASET,
            "source_event_name": str(output.get("source_event_id") or event_mapping["source_event_name"]),
            "canonical_event_name": event_mapping["canonical_event_name"],
            "event_group_id": event_mapping["event_group_id"],
            "subevent_id": event_mapping["subevent_id"],
            "tile_id": str(output.get("source_id") or output["id"]),
            "event_id": event_mapping["event_group_id"],
            "directory_split_v1": directory_split,
            "split": new_split,
            "event_group_split_protocol": str(plan["protocol"]),
            "geographic_footprint": geographic_footprint,
            "geographic_footprint_status": (
                "inherited_from_decoded_identical_bright_pair"
                if geographic_footprint is not None
                else "unavailable_in_png_minimal_package"
            ),
        }
    )
    return output


def no_prior_row(row: dict[str, Any]) -> dict[str, Any]:
    output = dict(row)
    output.update(
        {
            "building_prior": None,
            "building_prior_binary": None,
            "building_prior_uint8": None,
            "prior_type": "none",
        }
    )
    return output


def oracle_prior_row(row: dict[str, Any]) -> dict[str, Any]:
    output = dict(row)
    value = output["oracle_building_mask"]
    output.update(
        {
            "building_prior": value,
            "building_prior_binary": value,
            "building_prior_uint8": value,
            "prior_type": "oracle",
        }
    )
    return output


def sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (str(row["event_group_id"]), str(row["id"])))


def write_bundle(
    root: Path,
    master_by_split: dict[str, list[dict[str, Any]]],
    variants: dict[str, dict[str, list[dict[str, Any]]]],
) -> None:
    if root.exists():
        raise FileExistsError(f"refusing to overwrite manifest root: {root}")
    root.mkdir(parents=True)
    for split in SPLITS:
        write_jsonl(root / f"stage2_master_{split}.jsonl", sort_rows(master_by_split[split]))
    for variant, rows_by_split in variants.items():
        for split in SPLITS:
            write_jsonl(root / variant / f"{split}.jsonl", sort_rows(rows_by_split[split]))


def validate_event_groups(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    groups = {
        split: {str(row["event_group_id"]) for row in rows_by_split[split]} for split in SPLITS
    }
    overlaps: dict[str, list[str]] = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlaps[f"{left}_vs_{right}"] = sorted(groups[left] & groups[right])
    if any(overlaps.values()):
        raise ValueError(f"event_group_id crosses splits: {overlaps}")
    return {"groups_by_split": {key: sorted(value) for key, value in groups.items()}, "overlaps": overlaps}


def class_distribution(rows_by_split: dict[str, list[dict[str, Any]]], data_root: Path) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for split in SPLITS:
        event_hist: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(4, dtype=np.int64))
        event_images: Counter[str] = Counter()
        event_presence: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(4, dtype=np.int64))
        for row in rows_by_split[split]:
            path = resolve_path(data_root, row["mask_multiclass"])
            mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
            values = set(int(value) for value in np.unique(mask))
            if not values.issubset({0, 1, 2, 3}):
                raise ValueError(f"invalid mask values at {path}: {sorted(values)}")
            hist = np.bincount(mask.reshape(-1), minlength=4)[:4].astype(np.int64)
            event = str(row["event_group_id"])
            event_hist[event] += hist
            event_images[event] += 1
            event_presence[event] += hist > 0
        total = sum(event_hist.values(), np.zeros(4, dtype=np.int64))
        concentration: dict[str, Any] = {}
        nonempty_cells = 0
        positive_support: list[int] = []
        for class_id in (1, 2, 3):
            values = {event: int(hist[class_id]) for event, hist in event_hist.items()}
            class_total = sum(values.values())
            shares = {event: value / class_total if class_total else 0.0 for event, value in values.items()}
            top_event = max(shares, key=shares.get) if shares else ""
            hhi = sum(value * value for value in shares.values())
            concentration[CLASS_NAMES[class_id]] = {
                "top_event": top_event,
                "top1_share": shares.get(top_event, 0.0),
                "hhi": hhi,
                "effective_event_count": 1.0 / hhi if hhi else 0.0,
            }
            for value in values.values():
                if value > 0:
                    nonempty_cells += 1
                    positive_support.append(value)
        events: dict[str, Any] = {}
        for event, hist in sorted(event_hist.items()):
            building = int(hist[1:].sum())
            events[event] = {
                "images": int(event_images[event]),
                "class_pixels": {CLASS_NAMES[index]: int(hist[index]) for index in range(4)},
                "images_with_class": {
                    CLASS_NAMES[index]: int(event_presence[event][index]) for index in range(4)
                },
                "building_class_share": {
                    CLASS_NAMES[index]: float(hist[index] / building) if building else 0.0
                    for index in (1, 2, 3)
                },
            }
        building_total = int(total[1:].sum())
        output[split] = {
            "images": len(rows_by_split[split]),
            "event_groups": len(event_hist),
            "class_pixels": {CLASS_NAMES[index]: int(total[index]) for index in range(4)},
            "building_class_share": {
                CLASS_NAMES[index]: float(total[index] / building_total) if building_total else 0.0
                for index in (1, 2, 3)
            },
            "nonempty_event_class_cells": nonempty_cells,
            "total_event_class_cells": len(event_hist) * 3,
            "minimum_positive_event_class_pixels": min(positive_support) if positive_support else 0,
            "concentration": concentration,
            "events": events,
        }
    return output


def rectangles_overlap(left: dict[str, Any], right: dict[str, Any]) -> tuple[bool, float]:
    if int(left["epsg"]) != int(right["epsg"]):
        return False, 0.0
    lx0, ly0, lx1, ly1 = (float(value) for value in left["bbox_projected"])
    rx0, ry0, rx1, ry1 = (float(value) for value in right["bbox_projected"])
    width = max(0.0, min(lx1, rx1) - max(lx0, rx0))
    height = max(0.0, min(ly1, ry1) - max(ly0, ry0))
    area = width * height
    return area > 0.0, area


def footprint_audit(
    candidates: list[dict[str, Any]],
    current_by_split: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    comparison = current_by_split["val"] + current_by_split["test"]
    known = [row for row in comparison if isinstance(row.get("geographic_footprint"), dict)]
    overlaps: list[dict[str, Any]] = []
    for candidate in candidates:
        left = candidate["geographic_footprint"]
        for target in known:
            right = target["geographic_footprint"]
            intersects, area = rectangles_overlap(left, right)
            if intersects:
                overlaps.append(
                    {
                        "candidate_id": candidate["id"],
                        "candidate_event_group_id": candidate["event_group_id"],
                        "target_id": target["id"],
                        "target_split": target["split"],
                        "target_event_group_id": target["event_group_id"],
                        "epsg": int(left["epsg"]),
                        "intersection_area_projected": area,
                    }
                )
    footprint_keys: dict[str, list[str]] = defaultdict(list)
    for row in candidates:
        value = row["geographic_footprint"]
        key = json.dumps([value["epsg"], value["bbox_projected"]], separators=(",", ":"))
        footprint_keys[key].append(str(row["id"]))
    internal_duplicates = [ids for ids in footprint_keys.values() if len(ids) > 1]
    return {
        "candidate_count": len(candidates),
        "candidate_footprint_coverage": 1.0 if candidates else 0.0,
        "new_val_test_sample_count": len(comparison),
        "new_val_test_known_footprint_count": len(known),
        "new_val_test_footprint_coverage": len(known) / len(comparison) if comparison else 0.0,
        "projected_bbox_overlap_count": len(overlaps),
        "projected_bbox_overlaps": overlaps,
        "internal_exact_footprint_duplicate_groups": internal_duplicates,
        "status": (
            "fail"
            if overlaps or internal_duplicates
            else "partial_pass_due_to_missing_current_png_georeferencing"
        ),
        "limitation": (
            "The minimal PNG package strips georeferencing. Footprints are inherited only for current "
            "samples with a decoded-identical labeled BRIGHT pair; image-based near-overlap audit remains required."
        ),
    }


def label_compatibility(
    exact_matches: list[tuple[BrightSample, dict[str, Any]]], data_root: Path
) -> dict[str, Any]:
    confusion = np.zeros((4, 4), dtype=np.int64)
    for bright, current in exact_matches:
        current_mask = np.asarray(
            Image.open(resolve_path(data_root, current["mask_multiclass"])).convert("L"), dtype=np.uint8
        )
        bright_mask = rasterize_annotations(bright)
        if current_mask.shape != bright_mask.shape:
            raise ValueError(f"mask shape mismatch for decoded-identical pair: {bright.sample_id}")
        confusion += np.bincount(
            current_mask.reshape(-1).astype(np.int64) * 4 + bright_mask.reshape(-1).astype(np.int64),
            minlength=16,
        ).reshape(4, 4)
    intersection_building = int(confusion[1:, 1:].sum())
    union_building = int(confusion[1:, :].sum() + confusion[:, 1:].sum() - intersection_building)
    shared_class_conflict = int(intersection_building - np.trace(confusion[1:, 1:]))
    class_iou: dict[str, float] = {}
    for class_id in (1, 2, 3):
        intersection = int(confusion[class_id, class_id])
        union = int(confusion[class_id, :].sum() + confusion[:, class_id].sum() - intersection)
        class_iou[CLASS_NAMES[class_id]] = intersection / union if union else 0.0
    return {
        "pair_count": len(exact_matches),
        "confusion_rows_current_cols_bright": confusion.tolist(),
        "building_iou": intersection_building / union_building if union_building else 0.0,
        "class_iou": class_iou,
        "shared_foreground_class_conflict_pixels": shared_class_conflict,
        "shared_foreground_class_conflict_rate": (
            shared_class_conflict / intersection_building if intersection_building else 0.0
        ),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    data_root = args.data_root.resolve()
    source_manifest_root = args.source_manifest_root.resolve()
    bright_root = args.bright_root.resolve()
    coco_root = args.bright_coco_root.resolve()
    event_manifest_root = args.event_manifest_root.resolve()
    supplement_manifest_root = args.supplement_manifest_root.resolve()
    combined_manifest_root = args.combined_manifest_root.resolve()
    derived_root = args.derived_root.resolve()
    report_root = args.report_root.resolve()
    for output in (
        event_manifest_root,
        supplement_manifest_root,
        combined_manifest_root,
        derived_root,
        report_root,
    ):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {output}")

    mapping = load_mapping(args.event_mapping)
    plan = json.loads(args.split_plan.read_text(encoding="utf-8"))
    assignments = load_split_assignments(plan)
    current_raw, current_variants_raw = load_current_bundle(source_manifest_root)
    bright_samples = load_bright_samples(bright_root, coco_root)
    bright_by_id = {sample.sample_id: sample for sample in bright_samples}
    print(
        f"[INFO] loaded current={len(current_raw)} BRIGHT_labeled={len(bright_samples)}",
        flush=True,
    )

    mapped_groups = {
        mapping_for_current(row, mapping)["event_group_id"] for row in current_raw
    }
    if set(assignments) != mapped_groups:
        raise ValueError(
            f"split plan/mapping mismatch: missing={sorted(mapped_groups - set(assignments))} "
            f"extra={sorted(set(assignments) - mapped_groups)}"
        )
    for sample in bright_samples:
        key = (BRIGHT_DATASET, sample.source_event_name)
        if key not in mapping:
            raise KeyError(f"missing BRIGHT event mapping: {key}")

    current_key_to_row: dict[str, dict[str, Any]] = {}
    current_key_to_path: dict[str, Path] = {}
    current_pre_tasks: list[tuple[str, Path, str]] = []
    for row in current_raw:
        key = str(row["id"])
        current_key_to_row[key] = row
        path = resolve_path(data_root, row["pre_image"])
        current_key_to_path[key] = path
        current_pre_tasks.append((key, path, "RGB"))
    bright_pre_tasks = [(sample.sample_id, sample.pre_image, "RGB") for sample in bright_samples]
    current_pre_hash = decoded_hashes(current_pre_tasks, args.workers, "current pre-event RGB")
    bright_pre_hash = decoded_hashes(bright_pre_tasks, args.workers, "BRIGHT pre-event RGB")
    current_by_pre_hash: dict[str, list[str]] = defaultdict(list)
    for key, digest in current_pre_hash.items():
        current_by_pre_hash[digest].append(key)

    pre_matches: list[tuple[BrightSample, str]] = []
    for sample in bright_samples:
        for current_key in current_by_pre_hash.get(bright_pre_hash[sample.sample_id], []):
            pre_matches.append((sample, current_key))
    post_tasks: list[tuple[str, Path, str]] = []
    needed_current_post: set[str] = set()
    needed_bright_post: set[str] = set()
    for sample, current_key in pre_matches:
        needed_current_post.add(current_key)
        needed_bright_post.add(sample.sample_id)
    allowed_bright_events = set(plan["bright_supplement"]["allowed_source_events"])
    for sample in bright_samples:
        if sample.source_event_name in allowed_bright_events:
            needed_bright_post.add(sample.sample_id)
    for current_key in sorted(needed_current_post):
        row = current_key_to_row[current_key]
        post_tasks.append((f"current::{current_key}", resolve_path(data_root, row["post_sar"]), "L"))
    for sample_id in sorted(needed_bright_post):
        post_tasks.append((f"bright::{sample_id}", bright_by_id[sample_id].post_sar, "L"))
    post_hash = decoded_hashes(post_tasks, args.workers, "matched/candidate post-event SAR")

    exact_matches: list[tuple[BrightSample, dict[str, Any]]] = []
    pre_only_mismatches: list[dict[str, Any]] = []
    current_footprints: dict[str, dict[str, Any]] = {}
    exact_match_records: list[dict[str, Any]] = []
    matched_bright_ids: set[str] = set()
    for sample, current_key in pre_matches:
        current = current_key_to_row[current_key]
        post_equal = post_hash[f"bright::{sample.sample_id}"] == post_hash[f"current::{current_key}"]
        record = {
            "bright_id": sample.sample_id,
            "bright_source_split": sample.source_split,
            "bright_source_event_name": sample.source_event_name,
            "current_id": current_key,
            "current_directory_split_v1": str(current["_directory_split_v1"]),
            "current_source_event_name": str(current.get("source_event_id") or current["event_id"]),
            "decoded_pre_equal": True,
            "decoded_post_equal": post_equal,
        }
        if post_equal:
            exact_matches.append((sample, current))
            exact_match_records.append(record)
            matched_bright_ids.add(sample.sample_id)
            if current_key in current_footprints and current_footprints[current_key] != sample.geographic_footprint:
                raise ValueError(f"conflicting inherited footprints for {current_key}")
            current_footprints[current_key] = sample.geographic_footprint
        else:
            pre_only_mismatches.append(record)
            matched_bright_ids.add(sample.sample_id)

    group_metadata: dict[str, dict[str, str]] = {}
    for row in current_raw:
        item = mapping_for_current(row, mapping)
        event_group = item["event_group_id"]
        metadata = {
            "canonical_event_name": item["canonical_event_name"],
            "country_or_region": str(row.get("country_or_region", "")),
            "continent": str(row.get("continent", "")),
            "disaster_type": str(row.get("disaster_type", "")),
        }
        group_metadata.setdefault(event_group, metadata)

    current_master_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    current_variant_by_split: dict[str, dict[str, list[dict[str, Any]]]] = {
        variant: {split: [] for split in SPLITS} for variant in VARIANTS
    }
    transformed_info: dict[tuple[str, str], tuple[dict[str, str], str, dict[str, Any] | None]] = {}
    for raw in current_raw:
        directory_split = str(raw["_directory_split_v1"])
        event_mapping = mapping_for_current(raw, mapping)
        new_split = assignments[event_mapping["event_group_id"]]
        footprint = current_footprints.get(str(raw["id"]))
        transformed = transform_current_row(raw, event_mapping, new_split, directory_split, footprint, plan)
        current_master_by_split[new_split].append(transformed)
        transformed_info[(directory_split, str(raw["id"]))] = (event_mapping, new_split, footprint)
    for variant in VARIANTS:
        for key, raw in current_variants_raw[variant].items():
            event_mapping, new_split, footprint = transformed_info[key]
            transformed = transform_current_row(raw, event_mapping, new_split, key[0], footprint, plan)
            current_variant_by_split[variant][new_split].append(transformed)
    event_group_audit = validate_event_groups(current_master_by_split)

    derived_masks = derived_root / "masks_4class"
    derived_oracle = derived_root / "oracle_building_masks"
    derived_masks.mkdir(parents=True)
    derived_oracle.mkdir(parents=True)
    candidate_rows: list[dict[str, Any]] = []
    candidate_no_prior: list[dict[str, Any]] = []
    candidate_oracle: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    conversion_rows: list[dict[str, Any]] = []
    candidate_content_rows: list[dict[str, Any]] = []
    for sample in bright_samples:
        if sample.source_event_name not in allowed_bright_events:
            continue
        if sample.sample_id in matched_bright_ids:
            excluded_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "source_split": sample.source_split,
                    "source_event_name": sample.source_event_name,
                    "reason": "decoded_pre_match_with_current_strict_union",
                }
            )
            continue
        mask = rasterize_annotations(sample)
        hist = np.bincount(mask.reshape(-1), minlength=4)[:4].astype(np.int64)
        if bool(plan["bright_supplement"]["require_nonempty_building_mask"]) and int(hist[1:].sum()) == 0:
            excluded_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "source_split": sample.source_split,
                    "source_event_name": sample.source_event_name,
                    "reason": "empty_building_annotation",
                }
            )
            continue
        mask_path = derived_masks / f"{sample.sample_id}.png"
        oracle_path = derived_oracle / f"{sample.sample_id}.png"
        Image.fromarray(mask, mode="L").save(mask_path)
        Image.fromarray(((mask > 0) * 255).astype(np.uint8), mode="L").save(oracle_path)
        event_mapping = mapping[(BRIGHT_DATASET, sample.source_event_name)]
        event_group = event_mapping["event_group_id"]
        target_split = assignments[event_group]
        if target_split != str(plan["bright_supplement"]["target_split"]):
            raise ValueError(f"supplement event is not assigned to train: {sample.source_event_name}")
        metadata = group_metadata[event_group]
        row = {
            "id": f"brightv1__{sample.sample_id}",
            "split": target_split,
            "source_dataset": BRIGHT_DATASET,
            "source_split": sample.source_split,
            "source_id": sample.sample_id,
            "source_event_name": sample.source_event_name,
            "source_event_id": sample.source_event_name,
            "canonical_event_name": event_mapping["canonical_event_name"],
            "event_group_id": event_group,
            "event_id": event_group,
            "subevent_id": event_mapping["subevent_id"],
            "tile_id": sample.sample_id,
            "directory_split_v1": None,
            "event_group_split_protocol": str(plan["protocol"]),
            "pre_image": str(sample.pre_image),
            "post_sar": str(sample.post_sar),
            "post_modality": "SAR",
            "mask_multiclass": str(mask_path),
            "oracle_building_mask": str(oracle_path),
            "pred_building_binary": None,
            "pred_building_prob": None,
            "pred_building_prob_uint8": None,
            "stage1_model": None,
            "stage1_threshold": None,
            "target_num_classes": 4,
            "target_values": {"0": "background", "1": "intact", "2": "damaged", "3": "destroyed"},
            "instance_annotation_source": str(sample.target_json),
            "instance_annotation_count": len(sample.annotations),
            "annotation_conversion": "COCO polygon instances rasterized in source annotation order",
            "geographic_footprint": sample.geographic_footprint,
            "geographic_footprint_status": "available_from_bright_target_json",
            "country_or_region": metadata["country_or_region"],
            "continent": metadata["continent"],
            "disaster_type": metadata["disaster_type"],
            "qc_label": "bright_polygon_rasterized_v1",
        }
        candidate_rows.append(row)
        candidate_no_prior.append(no_prior_row(row))
        candidate_oracle.append(oracle_prior_row(row))
        conversion_rows.append(
            {
                "id": row["id"],
                "source_event_name": sample.source_event_name,
                "source_split": sample.source_split,
                "annotation_count": len(sample.annotations),
                **{f"{CLASS_NAMES[index]}_pixels": int(hist[index]) for index in range(4)},
                **{
                    f"{CLASS_NAMES[index]}_instances": sum(
                        int(annotation["category_id"]) == index for annotation in sample.annotations
                    )
                    for index in (1, 2, 3)
                },
            }
        )
        mask_digest = hashlib.sha256(mask.tobytes()).hexdigest()
        candidate_content_rows.append(
            {
                "id": row["id"],
                "decoded_pre_sha256": bright_pre_hash[sample.sample_id],
                "decoded_post_sha256": post_hash[f"bright::{sample.sample_id}"],
                "decoded_mask_sha256": mask_digest,
            }
        )
    candidate_rows = sort_rows(candidate_rows)
    candidate_no_prior = sort_rows(candidate_no_prior)
    candidate_oracle = sort_rows(candidate_oracle)

    pair_signatures = Counter(
        (row["decoded_pre_sha256"], row["decoded_post_sha256"]) for row in candidate_content_rows
    )
    duplicate_candidate_pairs = [signature for signature, count in pair_signatures.items() if count > 1]
    if duplicate_candidate_pairs:
        raise ValueError(f"decoded-identical pairs remain within BRIGHT supplement: {len(duplicate_candidate_pairs)}")

    supplement_master = {"train": candidate_rows, "val": [], "test": []}
    supplement_variants = {
        "no_prior": {"train": candidate_no_prior, "val": [], "test": []},
        "oracle_prior": {"train": candidate_oracle, "val": [], "test": []},
    }
    write_bundle(event_manifest_root, current_master_by_split, current_variant_by_split)
    write_bundle(supplement_manifest_root, supplement_master, supplement_variants)

    combined_master = {
        split: list(current_master_by_split[split]) + (candidate_rows if split == "train" else [])
        for split in SPLITS
    }
    combined_variants = {
        "no_prior": {
            split: list(current_variant_by_split["no_prior"][split])
            + (candidate_no_prior if split == "train" else [])
            for split in SPLITS
        },
        "oracle_prior": {
            split: list(current_variant_by_split["oracle_prior"][split])
            + (candidate_oracle if split == "train" else [])
            for split in SPLITS
        },
    }
    combined_event_audit = validate_event_groups(combined_master)
    write_bundle(combined_manifest_root, combined_master, combined_variants)

    near_view_root = supplement_manifest_root / "near_overlap_audit_view"
    near_view_root.mkdir(parents=True)
    write_jsonl(near_view_root / "stage2_master_train.jsonl", candidate_rows)
    write_jsonl(near_view_root / "stage2_master_val.jsonl", sort_rows(current_master_by_split["val"]))
    write_jsonl(near_view_root / "stage2_master_test.jsonl", sort_rows(current_master_by_split["test"]))

    cross_counts = Counter(
        (sample.source_split, str(current["_directory_split_v1"])) for sample, current in exact_matches
    )
    event_alias_counts = Counter(
        (
            sample.source_event_name,
            str(current.get("source_event_id") or current["event_id"]),
        )
        for sample, current in exact_matches
    )
    compatibility = label_compatibility(exact_matches, data_root)
    exact_overlap_audit = {
        "algorithm": (
            "SHA-256 over decoded uint8 RGB/L arrays; a duplicate pair requires decoded pre-event RGB "
            "and decoded post-event SAR equality."
        ),
        "bright_labeled_samples": len(bright_samples),
        "current_strict_samples": len(current_raw),
        "decoded_pre_match_pairs": len(pre_matches),
        "decoded_pre_post_match_pairs": len(exact_matches),
        "decoded_pre_only_post_mismatch_pairs": len(pre_only_mismatches),
        "bright_exact_pair_overlap_share": len(exact_matches) / len(bright_samples),
        "current_exact_pair_overlap_share": len(exact_matches) / len(current_raw),
        "cross_split_counts": {
            f"bright_{left}__current_{right}": count
            for (left, right), count in sorted(cross_counts.items())
        },
        "event_alias_counts": {
            f"{left}__{right}": count for (left, right), count in sorted(event_alias_counts.items())
        },
        "pre_only_post_mismatches": pre_only_mismatches,
        "label_compatibility": compatibility,
        "matches": exact_match_records,
    }
    footprint_report = footprint_audit(candidate_rows, current_master_by_split)
    r0_distribution = class_distribution(current_master_by_split, data_root)
    r4_distribution = class_distribution(combined_master, data_root)

    report_root.mkdir(parents=True)
    write_json(report_root / "exact_overlap_audit.json", exact_overlap_audit)
    write_json(report_root / "footprint_audit.json", footprint_report)
    write_json(report_root / "r0_event_group_distribution.json", r0_distribution)
    write_json(report_root / "r4_bright_event_group_distribution.json", r4_distribution)
    write_jsonl(report_root / "decoded_candidate_hashes.jsonl", sorted(candidate_content_rows, key=lambda row: row["id"]))
    write_csv(report_root / "bright_conversion_stats.csv", conversion_rows)
    write_csv(
        supplement_manifest_root / "excluded_samples.csv",
        excluded_rows,
        ["sample_id", "source_split", "source_event_name", "reason"],
    )
    write_json(event_manifest_root / "event_group_split_resolved.json", plan)
    write_json(event_manifest_root / "event_group_audit.json", event_group_audit)
    write_json(combined_manifest_root / "event_group_audit.json", combined_event_audit)

    supplement_event_counts = Counter(str(row["source_event_name"]) for row in candidate_rows)
    exclusion_counts = Counter(str(row["reason"]) for row in excluded_rows)
    readiness = {
        "protocol": plan["protocol"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training_eligible": False,
        "r0_current_only_predicted_prior_ready": True,
        "r4_predicted_prior_ready": False,
        "near_duplicate_audit": "pending",
        "exact_decoded_pair_overlap": "pass",
        "event_group_disjoint": "pass",
        "footprint_audit": footprint_report["status"],
        "test_status": plan["evaluation_status"]["event_group_v1_test"],
        "blockers": [
            "Frozen O1 predicted building priors have not been generated for BRIGHT supplement samples.",
            "The focused BRIGHT-vs-new-val/test near-duplicate audit has not yet been run.",
            "The rebuilt test contains historically inspected events and is not a pristine final test.",
        ],
    }
    summary = {
        "protocol": plan["protocol"],
        "source_manifest_root": str(source_manifest_root),
        "event_manifest_root": str(event_manifest_root),
        "supplement_manifest_root": str(supplement_manifest_root),
        "combined_manifest_root": str(combined_manifest_root),
        "derived_root": str(derived_root),
        "mapping_sha256": sha256_file(args.event_mapping),
        "split_plan_sha256": sha256_file(args.split_plan),
        "current_split_counts": {split: len(current_master_by_split[split]) for split in SPLITS},
        "combined_split_counts": {split: len(combined_master[split]) for split in SPLITS},
        "bright_labeled_count": len(bright_samples),
        "bright_exact_pair_overlap_count": len(exact_matches),
        "bright_supplement_count": len(candidate_rows),
        "bright_supplement_event_counts": dict(sorted(supplement_event_counts.items())),
        "bright_exclusion_counts": dict(sorted(exclusion_counts.items())),
        "footprint_audit_status": footprint_report["status"],
        "training_eligible": False,
    }
    write_json(event_manifest_root / "build_summary.json", summary)
    write_json(supplement_manifest_root / "build_summary.json", summary)
    write_json(combined_manifest_root / "build_summary.json", summary)
    write_json(combined_manifest_root / "READINESS.json", readiness)
    write_json(report_root / "build_summary.json", summary)
    write_json(report_root / "READINESS.json", readiness)

    current_damaged = r0_distribution["train"]["concentration"]["damaged"]
    current_destroyed = r0_distribution["train"]["concentration"]["destroyed"]
    combined_damaged = r4_distribution["train"]["concentration"]["damaged"]
    combined_destroyed = r4_distribution["train"]["concentration"]["destroyed"]
    lines = [
        "# Stage-2 event-group v1 dataset build",
        "",
        "Status: data build complete; training remains blocked and was not started.",
        "",
        "## Frozen inputs",
        "",
        f"- Historical directory split: `{source_manifest_root}`",
        f"- Event mapping: `{args.event_mapping.resolve()}`",
        f"- Event-group split plan: `{args.split_plan.resolve()}`",
        "- The historical strict-v1 manifests and source datasets were not modified.",
        "",
        "## New current-only event-group split (R0 data)",
        "",
        f"- train/val/test: `{summary['current_split_counts']}`",
        "- All Turkey 1/3/4/5 rows are assigned to the single `turkey_eq_2023` train group.",
        "- Current `rwanda_volcano` is normalized to `nyiragongo_2021`.",
        "- The test is historically observed and must not be described as a pristine unseen-disaster test.",
        "",
        "## BRIGHT supplement",
        "",
        f"- Final nonempty, decoded-deduplicated samples: **{len(candidate_rows)}**",
        f"- Per source event: `{dict(sorted(supplement_event_counts.items()))}`",
        f"- Decoded-identical BRIGHT/current pairs excluded: **{len(exact_matches)}**",
        f"- Footprint audit: `{footprint_report['status']}`; known val/test coverage "
        f"`{footprint_report['new_val_test_footprint_coverage']:.2%}`.",
        "",
        "## Event-class concentration (train)",
        "",
        "| class | R0 top-1 | R0 effective events | R4-data top-1 | R4-data effective events |",
        "|---|---:|---:|---:|---:|",
        f"| damaged | {current_damaged['top1_share']:.2%} | {current_damaged['effective_event_count']:.2f} | "
        f"{combined_damaged['top1_share']:.2%} | {combined_damaged['effective_event_count']:.2f} |",
        f"| destroyed | {current_destroyed['top1_share']:.2%} | {current_destroyed['effective_event_count']:.2f} | "
        f"{combined_destroyed['top1_share']:.2%} | {combined_destroyed['effective_event_count']:.2f} |",
        "",
        "## Blocking conditions before any R4 training",
        "",
        "1. Run and resolve the focused BRIGHT-vs-new-val/test near-duplicate audit.",
        "2. Generate frozen O1 predicted priors for the final BRIGHT supplement without using val/test for selection.",
        "3. Keep the rebuilt test labelled as historically observed; genuinely pristine testing requires new events.",
        "",
    ]
    (report_root / "DATASET_BUILD_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-manifest-root", type=Path, required=True)
    parser.add_argument("--bright-root", type=Path, required=True)
    parser.add_argument("--bright-coco-root", type=Path, required=True)
    parser.add_argument("--event-mapping", type=Path, required=True)
    parser.add_argument("--split-plan", type=Path, required=True)
    parser.add_argument("--event-manifest-root", type=Path, required=True)
    parser.add_argument("--supplement-manifest-root", type=Path, required=True)
    parser.add_argument("--combined-manifest-root", type=Path, required=True)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    summary = build(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
