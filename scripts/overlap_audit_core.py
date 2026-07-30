from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


FEATURE_VERSION = "relaxed-overlap-v1-dct-patch4-v1"
SPLITS = ("train", "val", "test")
PAIR_ORDER = (("train", "test"), ("train", "val"), ("val", "test"))
IMAGE_FIELDS = ("pre_image", "post_sar", "mask_multiclass")


@dataclass(frozen=True)
class Sample:
    split: str
    sample_id: str
    event_id: str
    disaster: str
    region: str
    pre_image: Path
    post_sar: Path
    mask: Path

    def path_for(self, field: str) -> Path:
        return {
            "pre_image": self.pre_image,
            "post_sar": self.post_sar,
            "mask_multiclass": self.mask,
        }[field]


@dataclass
class ImageFeature:
    phash: int
    dhash: int
    embedding: np.ndarray
    patches: np.ndarray


def resolve_path(data_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = data_root / path
    return path.resolve()


def load_samples(data_root: Path, manifest_root: Path) -> dict[str, list[Sample]]:
    result: dict[str, list[Sample]] = {}
    for split in SPLITS:
        manifest = manifest_root / f"stage2_master_{split}.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(f"missing manifest: {manifest}")
        samples: list[Sample] = []
        seen: set[str] = set()
        for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row["id"])
            if sample_id in seen:
                raise ValueError(f"duplicate id in {manifest}:{line_number}: {sample_id}")
            seen.add(sample_id)
            samples.append(
                Sample(
                    split=split,
                    sample_id=sample_id,
                    event_id=str(row.get("event_id", "unknown")),
                    disaster=str(row.get("disaster_type", "unknown")),
                    region=str(row.get("country_or_region", "unknown")),
                    pre_image=resolve_path(data_root, row["pre_image"]),
                    post_sar=resolve_path(data_root, row["post_sar"]),
                    mask=resolve_path(data_root, row["mask_multiclass"]),
                )
            )
        result[split] = samples
    return result


def _gray_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype == np.uint8:
        return image
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(image.shape[:2], dtype=np.uint8)
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if hi <= lo:
        return np.zeros(image.shape[:2], dtype=np.uint8)
    scaled = np.clip((image.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    return np.round(scaled * 255.0).astype(np.uint8)


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return _gray_uint8(image)


def _bits_to_int(bits: np.ndarray) -> int:
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bool(bit))
    return value


def phash(gray: np.ndarray) -> int:
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:8, :8]
    median = float(np.median(low.reshape(-1)[1:]))
    return _bits_to_int(low > median)


def dhash(gray: np.ndarray) -> int:
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    return _bits_to_int(small[:, 1:] > small[:, :-1])


def hamming(left: int, right: int) -> int:
    return int((int(left) ^ int(right)).bit_count())


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = vector.astype(np.float32, copy=False).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return np.zeros_like(vector)
    return vector / norm


def dct_embedding(gray: np.ndarray, size: int = 16) -> np.ndarray:
    resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    resized -= float(resized.mean())
    coeff = cv2.dct(resized)[:size, :size]
    return _unit(coeff)


def patch_descriptors(gray: np.ndarray, grid: int = 4, dct_size: int = 8) -> np.ndarray:
    height, width = gray.shape[:2]
    descriptors: list[np.ndarray] = []
    for row in range(grid):
        y0, y1 = height * row // grid, height * (row + 1) // grid
        for col in range(grid):
            x0, x1 = width * col // grid, width * (col + 1) // grid
            patch = gray[y0:y1, x0:x1]
            patch = cv2.resize(patch, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            patch -= float(patch.mean())
            descriptors.append(_unit(cv2.dct(patch)[:dct_size, :dct_size]))
    return np.stack(descriptors).astype(np.float32)


def feature_from_array(image: np.ndarray) -> ImageFeature:
    gray = _gray_uint8(image)
    return ImageFeature(
        phash=phash(gray),
        dhash=dhash(gray),
        embedding=dct_embedding(gray),
        patches=patch_descriptors(gray),
    )


class FeatureCache:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _key(path: Path) -> str:
        stat = path.stat()
        material = f"{FEATURE_VERSION}\0{path}\0{stat.st_size}\0{stat.st_mtime_ns}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def get(self, path: Path) -> ImageFeature:
        key = self._key(path)
        target = self.root / key[:2] / f"{key}.npz"
        if target.is_file():
            with np.load(target, allow_pickle=False) as data:
                return ImageFeature(
                    phash=int(data["phash"]),
                    dhash=int(data["dhash"]),
                    embedding=data["embedding"].astype(np.float32),
                    patches=data["patches"].astype(np.float32),
                )
        feature = feature_from_array(read_gray(path))
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(f".{target.name}.{os.getpid()}.tmp.npz")
        np.savez_compressed(
            temp,
            phash=np.array(feature.phash, dtype=np.uint64),
            dhash=np.array(feature.dhash, dtype=np.uint64),
            embedding=feature.embedding,
            patches=feature.patches,
        )
        os.replace(temp, target)
        return feature


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.clip(np.dot(left.reshape(-1), right.reshape(-1)), -1.0, 1.0))


def patch_overlap(left: np.ndarray, right: np.ndarray) -> tuple[float, int, float]:
    similarity = np.clip(left @ right.T, -1.0, 1.0)
    left_best = similarity.max(axis=1)
    right_best = similarity.max(axis=0)
    best = float(similarity.max())
    high_count = int(min(np.count_nonzero(left_best >= 0.90), np.count_nonzero(right_best >= 0.90)))
    coverage = min(1.0, high_count / 4.0)
    strong_best = float(np.clip((best - 0.70) / 0.30, 0.0, 1.0))
    score = 0.65 * coverage + 0.35 * strong_best
    return float(score), high_count, best


def _topk_embedding_pairs(
    left: Sequence[ImageFeature], right: Sequence[ImageFeature], top_k: int
) -> set[tuple[int, int]]:
    if not left or not right:
        return set()
    a = np.stack([item.embedding for item in left])
    b = np.stack([item.embedding for item in right])
    pairs: set[tuple[int, int]] = set()
    block = 256
    k = min(top_k, len(right))
    for start in range(0, len(left), block):
        scores = a[start : start + block] @ b.T
        indices = np.argpartition(scores, -k, axis=1)[:, -k:]
        for local_row, columns in enumerate(indices):
            row = start + local_row
            pairs.update((row, int(column)) for column in columns)
    return pairs


def _patch_candidate_pairs(
    left: Sequence[ImageFeature], right: Sequence[ImageFeature], neighbors: int = 2
) -> set[tuple[int, int]]:
    if not left or not right:
        return set()
    train = np.concatenate([item.patches for item in right], axis=0).astype(np.float32)
    query = np.concatenate([item.patches for item in left], axis=0).astype(np.float32)
    index_params = {"algorithm": 1, "trees": 6}
    search_params = {"checks": 64}
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    matches = matcher.knnMatch(query, train, k=min(neighbors, len(train)))
    counts: dict[tuple[int, int], list[float]] = {}
    patches_per_sample = left[0].patches.shape[0]
    right_patches_per_sample = right[0].patches.shape[0]
    for query_index, row_matches in enumerate(matches):
        left_index = query_index // patches_per_sample
        for match in row_matches:
            similarity = 1.0 - float(match.distance * match.distance) / 2.0
            if similarity < 0.84:
                continue
            right_index = int(match.trainIdx) // right_patches_per_sample
            counts.setdefault((left_index, right_index), []).append(similarity)
    return {
        pair
        for pair, values in counts.items()
        if len(values) >= 2 or (values and max(values) >= 0.975)
    }


def generate_candidate_indices(
    left_by_field: dict[str, Sequence[ImageFeature]],
    right_by_field: dict[str, Sequence[ImageFeature]],
    global_top_k: int = 8,
) -> set[tuple[int, int]]:
    pairs = _topk_embedding_pairs(
        left_by_field["pre_image"], right_by_field["pre_image"], global_top_k
    )
    pairs |= _topk_embedding_pairs(
        left_by_field["post_sar"], right_by_field["post_sar"], max(2, global_top_k // 3)
    )
    pairs |= _topk_embedding_pairs(
        left_by_field["mask_multiclass"],
        right_by_field["mask_multiclass"],
        max(2, global_top_k // 4),
    )
    pairs |= _patch_candidate_pairs(left_by_field["pre_image"], right_by_field["pre_image"])
    return pairs


class LocalFeatureCache:
    def __init__(self, max_side: int = 512, nfeatures: int = 600):
        self.max_side = max_side
        self.detector = cv2.ORB_create(
            nfeatures=nfeatures,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=21,
            fastThreshold=10,
        )
        self.cache: dict[Path, tuple[list[cv2.KeyPoint], np.ndarray | None]] = {}

    def get(self, path: Path) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        if path in self.cache:
            return self.cache[path]
        gray = read_gray(path)
        scale = min(1.0, self.max_side / max(gray.shape[:2]))
        if scale < 1.0:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        self.cache[path] = (keypoints, descriptors)
        return keypoints, descriptors


def local_descriptor_candidate_pairs(
    left: Sequence[Sample],
    right: Sequence[Sample],
    cache: LocalFeatureCache,
    min_descriptor_matches: int = 6,
    max_pairs_per_left: int = 20,
    max_descriptors_per_sample: int = 160,
) -> set[tuple[int, int]]:
    """Retrieve spatial-overlap candidates from approximate ORB descriptor matches."""
    right_descriptors: list[np.ndarray] = []
    right_owners: list[np.ndarray] = []
    for index, sample in enumerate(right):
        _, descriptors = cache.get(sample.pre_image)
        if descriptors is None or len(descriptors) == 0:
            continue
        descriptors = descriptors[:max_descriptors_per_sample]
        right_descriptors.append(descriptors)
        right_owners.append(np.full(len(descriptors), index, dtype=np.int32))
    if not right_descriptors:
        return set()
    train = np.concatenate(right_descriptors, axis=0).astype(np.uint8, copy=False)
    owners = np.concatenate(right_owners)
    matcher = cv2.FlannBasedMatcher(
        {
            "algorithm": 6,  # FLANN_INDEX_LSH
            "table_number": 12,
            "key_size": 20,
            "multi_probe_level": 2,
        },
        {"checks": 64},
    )
    matcher.add([train])
    matcher.train()
    pairs: set[tuple[int, int]] = set()
    same_collection = left is right
    for left_index, sample in enumerate(left):
        _, descriptors = cache.get(sample.pre_image)
        if descriptors is None or len(descriptors) == 0:
            continue
        descriptors = descriptors[:max_descriptors_per_sample]
        match_rows = matcher.knnMatch(descriptors.astype(np.uint8, copy=False), k=4 if same_collection else 2)
        counts: dict[int, int] = {}
        for row_matches in match_rows:
            for match in row_matches:
                if match.distance > 52:
                    continue
                owner = int(owners[int(match.trainIdx)])
                if same_collection and owner == left_index:
                    continue
                counts[owner] = counts.get(owner, 0) + 1
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        for right_index, count in ranked[:max_pairs_per_left]:
            if count >= min_descriptor_matches:
                pairs.add((left_index, right_index))
    return pairs


def local_geometry(
    left_path: Path, right_path: Path, cache: LocalFeatureCache
) -> tuple[int, int, float, str, float]:
    left_kp, left_desc = cache.get(left_path)
    right_kp, right_desc = cache.get(right_path)
    if left_desc is None or right_desc is None or len(left_desc) < 4 or len(right_desc) < 4:
        return 0, 0, 0.0, "none", 0.0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = matcher.knnMatch(left_desc, right_desc, k=2)
    good = [first for first, second in raw if first.distance < 0.78 * second.distance]
    if len(good) < 4:
        return len(good), 0, 0.0, "none", 0.0
    source = np.float32([left_kp[item.queryIdx].pt for item in good])
    target = np.float32([right_kp[item.trainIdx].pt for item in good])
    matrix, mask = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=4.0, maxIters=3000
    )
    inliers = int(mask.sum()) if mask is not None else 0
    ratio = inliers / max(1, len(good))
    if matrix is not None:
        singular = np.linalg.svd(matrix[:, :2], compute_uv=False)
        valid_transform = (
            np.all(np.isfinite(matrix))
            and float(singular.min()) >= 0.25
            and float(singular.max()) <= 4.0
            and float(singular.max() / max(singular.min(), 1e-8)) <= 2.0
        )
        if not valid_transform:
            matrix = None
            inliers = 0
            ratio = 0.0
    local_score = min(1.0, inliers / 40.0) * min(1.0, ratio / 0.55)
    transform = "none_or_degenerate" if matrix is None else json.dumps(np.round(matrix, 5).tolist(), separators=(",", ":"))
    return len(good), inliers, float(ratio), transform, float(local_score)


def score_candidate(
    sample_a: Sample,
    sample_b: Sample,
    features_a: dict[str, ImageFeature],
    features_b: dict[str, ImageFeature],
    local_cache: LocalFeatureCache,
) -> dict[str, object]:
    pre_a, pre_b = features_a["pre_image"], features_b["pre_image"]
    sar_a, sar_b = features_a["post_sar"], features_b["post_sar"]
    mask_a, mask_b = features_a["mask_multiclass"], features_b["mask_multiclass"]
    pre_similarity = max(0.0, cosine(pre_a.embedding, pre_b.embedding))
    sar_similarity = max(0.0, cosine(sar_a.embedding, sar_b.embedding))
    mask_similarity = max(0.0, cosine(mask_a.embedding, mask_b.embedding))
    patch_score, patch_count, patch_best = patch_overlap(pre_a.patches, pre_b.patches)
    pre_phash = hamming(pre_a.phash, pre_b.phash)
    # Candidate generation already limits the pair count.  Run geometric verification on
    # every retained candidate because crop/translation can destroy global similarity.
    local_matches, local_inliers, local_ratio, transform, local_score = local_geometry(
        sample_a.pre_image, sample_b.pre_image, local_cache
    )
    weighted_score = (
        0.35 * pre_similarity
        + 0.25 * patch_score
        + 0.20 * local_score
        + 0.10 * mask_similarity
        + 0.10 * sar_similarity
    )
    # A geometrically consistent local match is stronger evidence of partial spatial
    # overlap than low global similarity is evidence against it.  The override only
    # activates after RANSAC and still sends the pair to human review.
    geometry_score = 0.0
    if local_inliers >= 12:
        geometry_score = 0.55 + 0.40 * local_score
    patch_override = 0.0
    if patch_count >= 3:
        patch_override = 0.62 + 0.30 * min(1.0, patch_count / 8.0)
    overlap_score = max(weighted_score, geometry_score, patch_override)
    if overlap_score >= 0.85:
        risk = "high"
    elif overlap_score >= 0.70:
        risk = "medium"
    elif overlap_score >= 0.60:
        risk = "low"
    else:
        risk = "ignore"
    pair_material = f"{sample_a.split}\0{sample_a.sample_id}\0{sample_b.split}\0{sample_b.sample_id}"
    pair_id = hashlib.sha1(pair_material.encode("utf-8")).hexdigest()[:20]
    return {
        "pair_id": pair_id,
        "split_pair": f"{sample_a.split}-{sample_b.split}",
        "split_a": sample_a.split,
        "id_a": sample_a.sample_id,
        "event_a": sample_a.event_id,
        "disaster_a": sample_a.disaster,
        "region_a": sample_a.region,
        "pre_a": str(sample_a.pre_image),
        "sar_a": str(sample_a.post_sar),
        "mask_a": str(sample_a.mask),
        "split_b": sample_b.split,
        "id_b": sample_b.sample_id,
        "event_b": sample_b.event_id,
        "disaster_b": sample_b.disaster,
        "region_b": sample_b.region,
        "pre_b": str(sample_b.pre_image),
        "sar_b": str(sample_b.post_sar),
        "mask_b": str(sample_b.mask),
        "pre_hash_distance": pre_phash,
        "pre_dhash_distance": hamming(pre_a.dhash, pre_b.dhash),
        "sar_hash_distance": hamming(sar_a.phash, sar_b.phash),
        "mask_hash_distance": hamming(mask_a.phash, mask_b.phash),
        "pre_embedding_similarity": round(pre_similarity, 6),
        "sar_embedding_similarity": round(sar_similarity, 6),
        "patch_overlap_score": round(patch_score, 6),
        "patch_high_match_count": patch_count,
        "patch_best_similarity": round(patch_best, 6),
        "local_matches": local_matches,
        "local_inliers": local_inliers,
        "local_inlier_ratio": round(local_ratio, 6),
        "estimated_transform": transform,
        "mask_shape_similarity": round(mask_similarity, 6),
        "overlap_score": round(float(overlap_score), 6),
        "risk_level": risk,
        "auto_suggestion": "manual_review" if risk in {"high", "medium"} else "sample_or_record",
        "review_label": "",
        "review_action": "",
        "review_comment": "",
    }


def write_csv(path: Path, rows: Sequence[dict[str, object]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise ValueError("fieldnames are required when writing an empty CSV")
        fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def risk_counts(rows: Iterable[dict[str, object]]) -> dict[str, int]:
    counts = {"high": 0, "medium": 0, "low": 0, "ignore": 0}
    for row in rows:
        counts[str(row["risk_level"])] += 1
    return counts
