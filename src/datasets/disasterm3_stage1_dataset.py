from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resolve_manifest(data_root: str | Path, manifest: str | Path) -> Path:
    manifest_path = Path(manifest)
    if manifest_path.is_absolute() and manifest_path.exists():
        return manifest_path

    root = Path(data_root)
    candidates = [
        root / manifest_path,
        root / "manifests" / manifest_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"manifest not found: {manifest}; tried {candidates}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_data_path(data_root: str | Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(data_root) / path


class DisasterM3Stage1Dataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        manifest: str | Path,
        train: bool,
        crop_size: int | None = None,
        positive_crop_ratio: float = 0.5,
        hflip: bool = False,
        vflip: bool = False,
        rotate90: bool = False,
        color_jitter: bool = False,
        normalize: str = "imagenet",
        limit: int | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.manifest_path = resolve_manifest(data_root, manifest)
        self.rows = read_jsonl(self.manifest_path)
        if limit is not None:
            self.rows = self.rows[:limit]
        self.train = train
        self.crop_size = crop_size
        self.positive_crop_ratio = positive_crop_ratio
        self.hflip = hflip
        self.vflip = vflip
        self.rotate90 = rotate90
        self.color_jitter = color_jitter
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_path = resolve_data_path(self.data_root, str(row["image"]))
        mask_path = resolve_data_path(self.data_root, str(row["mask_binary"]))

        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
        mask = (mask > 0).astype(np.uint8)

        if self.train and self.crop_size:
            image, mask = self._random_crop(image, mask, self.crop_size)
        if self.train:
            image, mask = self._augment(image, mask)

        image_f = image.astype(np.float32) / 255.0
        if self.normalize == "imagenet":
            image_f = (image_f - IMAGENET_MEAN) / IMAGENET_STD
        image_t = torch.from_numpy(np.ascontiguousarray(image_f.transpose(2, 0, 1))).float()
        mask_t = torch.from_numpy(np.ascontiguousarray(mask[None, :, :])).float()

        return {
            "image": image_t,
            "mask": mask_t,
            "id": str(row.get("id", "")),
            "event_id": str(row.get("event_id", "")),
            "disaster_type": str(row.get("disaster_type", "")),
            "country_or_region": str(row.get("country_or_region", "")),
            "qc_label": str(row.get("qc_label", "")),
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }

    def _random_crop(self, image: np.ndarray, mask: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
        h, w = mask.shape
        if h <= size or w <= size:
            return image, mask

        want_positive = random.random() < self.positive_crop_ratio and mask.any()
        y = random.randint(0, h - size)
        x = random.randint(0, w - size)
        if want_positive:
            for _ in range(20):
                yy = random.randint(0, h - size)
                xx = random.randint(0, w - size)
                if mask[yy:yy + size, xx:xx + size].any():
                    y, x = yy, xx
                    break

        return image[y:y + size, x:x + size], mask[y:y + size, x:x + size]

    def _augment(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.hflip and random.random() < 0.5:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=1)
        if self.vflip and random.random() < 0.5:
            image = np.flip(image, axis=0)
            mask = np.flip(mask, axis=0)
        if self.rotate90:
            k = random.randint(0, 3)
            if k:
                image = np.rot90(image, k)
                mask = np.rot90(mask, k)
        if self.color_jitter:
            image = self._color_jitter(image)
        return image, mask

    @staticmethod
    def _color_jitter(image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32)
        brightness = random.uniform(0.9, 1.1)
        contrast = random.uniform(0.9, 1.1)
        img = img * brightness
        mean = img.mean(axis=(0, 1), keepdims=True)
        img = (img - mean) * contrast + mean
        return np.clip(img, 0, 255).astype(np.uint8)
