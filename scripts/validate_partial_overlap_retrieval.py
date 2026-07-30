#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path

import cv2
import numpy as np

from overlap_audit_core import (
    IMAGE_FIELDS,
    LocalFeatureCache,
    Sample,
    feature_from_array,
    json_dump,
    load_samples,
    read_gray,
    score_candidate,
    write_csv,
)


POSITIVE_TRANSFORMS = ("center_crop", "shift128", "shift256", "scale08", "brightness", "jpeg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic validation for partial-overlap retrieval.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def transform(image: np.ndarray, name: str, is_mask: bool) -> np.ndarray:
    height, width = image.shape[:2]
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    border = cv2.BORDER_CONSTANT if is_mask else cv2.BORDER_REFLECT_101
    if name == "center_crop":
        y0, y1 = height // 8, height - height // 8
        x0, x1 = width // 8, width - width // 8
        return cv2.resize(image[y0:y1, x0:x1], (width, height), interpolation=interpolation)
    if name in {"shift128", "shift256"}:
        amount = int(name.removeprefix("shift"))
        matrix = np.float32([[1, 0, amount], [0, 1, amount // 2]])
        return cv2.warpAffine(image, matrix, (width, height), flags=interpolation, borderMode=border)
    if name == "scale08":
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 0.0, 0.8)
        return cv2.warpAffine(image, matrix, (width, height), flags=interpolation, borderMode=border)
    if name == "brightness":
        if is_mask:
            return image.copy()
        return np.clip(image.astype(np.float32) * 0.82 + 18.0, 0, 255).astype(np.uint8)
    if name == "jpeg":
        if is_mask:
            return image.copy()
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 62])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    raise ValueError(name)


def write_triplet(root: Path, key: str, images: dict[str, np.ndarray]) -> Sample:
    paths: dict[str, Path] = {}
    for field, image in images.items():
        path = root / f"{key}_{field}.png"
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"failed to write {path}")
        paths[field] = path
    return Sample(
        split="synthetic",
        sample_id=key,
        event_id="synthetic",
        disaster="synthetic",
        region="synthetic",
        pre_image=paths["pre_image"],
        post_sar=paths["post_sar"],
        mask=paths["mask_multiclass"],
    )


def main() -> None:
    args = parse_args()
    result_path = args.out_root / "synthetic_validation.json"
    if result_path.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {result_path}; pass --force")
    all_samples = sum(load_samples(args.data_root.resolve(), args.manifest_root.resolve()).values(), [])
    rng = random.Random(args.seed)
    selected = rng.sample(all_samples, min(args.num_samples, len(all_samples)))
    pair_rows: list[dict[str, object]] = []
    query_embeddings: list[np.ndarray] = []
    gallery_embeddings: list[np.ndarray] = []
    gallery_owner: list[int] = []
    with tempfile.TemporaryDirectory(prefix="stage2_overlap_synth_") as temp:
        temp_root = Path(temp)
        local_cache = LocalFeatureCache()
        for owner, sample in enumerate(selected):
            originals = {
                field: read_gray(sample.path_for(field))
                for field in IMAGE_FIELDS
            }
            original_sample = write_triplet(temp_root, f"base_{owner}", originals)
            original_features = {field: feature_from_array(image) for field, image in originals.items()}
            query_embeddings.append(original_features["pre_image"].embedding)
            for name in POSITIVE_TRANSFORMS:
                transformed = {
                    field: transform(image, name, field == "mask_multiclass")
                    for field, image in originals.items()
                }
                transformed_sample = write_triplet(temp_root, f"base_{owner}_{name}", transformed)
                transformed_features = {
                    field: feature_from_array(image) for field, image in transformed.items()
                }
                row = score_candidate(
                    original_sample,
                    transformed_sample,
                    original_features,
                    transformed_features,
                    local_cache,
                )
                row["transform"] = name
                row["expected_overlap"] = True
                pair_rows.append(row)
                gallery_embeddings.append(transformed_features["pre_image"].embedding)
                gallery_owner.append(owner)

            # Two disjoint halves from the same source are a hard negative for the local matcher.
            width = originals["pre_image"].shape[1]
            left_images = {
                field: cv2.resize(image[:, : width // 2], (1024, 1024), interpolation=cv2.INTER_NEAREST if field == "mask_multiclass" else cv2.INTER_LINEAR)
                for field, image in originals.items()
            }
            right_images = {
                field: cv2.resize(image[:, width // 2 :], (1024, 1024), interpolation=cv2.INTER_NEAREST if field == "mask_multiclass" else cv2.INTER_LINEAR)
                for field, image in originals.items()
            }
            left_sample = write_triplet(temp_root, f"base_{owner}_left", left_images)
            right_sample = write_triplet(temp_root, f"base_{owner}_right", right_images)
            row = score_candidate(
                left_sample,
                right_sample,
                {field: feature_from_array(image) for field, image in left_images.items()},
                {field: feature_from_array(image) for field, image in right_images.items()},
                local_cache,
            )
            row["transform"] = "adjacent_nonoverlap_halves"
            row["expected_overlap"] = False
            pair_rows.append(row)

    queries = np.stack(query_embeddings)
    gallery = np.stack(gallery_embeddings)
    similarities = queries @ gallery.T
    top20_hits = 0
    top20_total = 0
    positive_ranks: list[int] = []
    for query_index, scores in enumerate(similarities):
        order = np.argsort(-scores)
        for gallery_index, owner in enumerate(gallery_owner):
            if owner == query_index:
                positive_ranks.append(int(np.flatnonzero(order == gallery_index)[0]) + 1)
        top = order[:20]
        top20_hits += sum(gallery_owner[index] == query_index for index in top)
        top20_total += len(top)

    positives = [row for row in pair_rows if row["expected_overlap"]]
    negatives = [row for row in pair_rows if not row["expected_overlap"]]
    summary = {
        "seed": args.seed,
        "sample_count": len(selected),
        "positive_transform_count": len(positives),
        "negative_pair_count": len(negatives),
        "recall_at_20": sum(rank <= 20 for rank in positive_ranks) / max(1, len(positive_ranks)),
        "recall_at_8": sum(rank <= 8 for rank in positive_ranks) / max(1, len(positive_ranks)),
        "precision_at_20": top20_hits / max(1, top20_total),
        "median_positive_rank": float(np.median(positive_ranks)),
        "high_risk_recall": sum(float(row["overlap_score"]) >= 0.85 for row in positives) / max(1, len(positives)),
        "medium_or_high_recall": sum(float(row["overlap_score"]) >= 0.70 for row in positives) / max(1, len(positives)),
        "adjacent_negative_medium_or_high_rate": sum(float(row["overlap_score"]) >= 0.70 for row in negatives) / max(1, len(negatives)),
        "per_transform": {
            name: {
                "count": sum(row["transform"] == name for row in positives),
                "mean_score": round(float(np.mean([float(row["overlap_score"]) for row in positives if row["transform"] == name])), 6),
                "medium_or_high_recall": sum(float(row["overlap_score"]) >= 0.70 for row in positives if row["transform"] == name) / max(1, sum(row["transform"] == name for row in positives)),
            }
            for name in POSITIVE_TRANSFORMS
        },
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_root / "synthetic_pairs.csv", pair_rows)
    json_dump(result_path, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
