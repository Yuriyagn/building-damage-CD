from __future__ import annotations

import random
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .common import get_metadata, read_jsonl, resolve_data_path, resolve_manifest


DEFAULT_OGSR_TEXTURE_FEATURE_KEYS = (
    "expected_sar",
    "expected_sar_grad",
    "residual_sar",
    "abs_residual_sar",
    "residual_grad",
    "abs_residual_grad",
)


def _load_gray_uint8(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def _load_rgb_uint8(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _load_prior_npz(path: Path) -> np.ndarray:
    with np.load(path) as data:
        if "prob" not in data:
            raise KeyError(f"{path} does not contain a 'prob' array")
        prob = data["prob"].astype(np.float32)
    return np.clip(prob, 0.0, 1.0)


class Stage2DamageDataset(Dataset):
    """Dataset for post-event SAR + building prior -> 4-class damage mask.

    The default returned image has three channels:
    [SAR, prior, SAR * prior]. For prior-only baselines, SAR and SAR*prior
    are zeroed while the prior channel is kept.

    Stage-2 v2 pre-change experiments can additionally use the pre-event
    optical RGB image:

    - pre_prior: [pre_R, pre_G, pre_B, 0, prior, 0]
    - pre_only: [pre_R, pre_G, pre_B, 0, 0, 0]
    - pre_sar: [pre_R, pre_G, pre_B, SAR]
    - pre_sar_background_only:
      [pre_RGB outside GT buildings, SAR outside GT buildings, 0, 0]
    - pre_sar_prior: [pre_R, pre_G, pre_B, SAR, prior, SAR * prior]
    - prior_only5: [0, 0, 0, prior, 0]
    - pre_prior5: [pre_R, pre_G, pre_B, prior, 0]
    - pre_prior_sar5: [pre_R, pre_G, pre_B, prior, SAR]
    - pre_sar_texture_prior:
      [pre_R, pre_G, pre_B, SAR, SAR_grad, prior, SAR * prior, SAR_grad * prior]
    - pre_sar_ogsr_texture_prior:
      [pre_R, pre_G, pre_B, SAR, SAR_grad, prior, expected_sar,
       expected_sar_grad, residual_sar, abs_residual_sar, residual_grad,
       abs_residual_grad]
    - pre_sar_texture_ogsr_add_prior:
      [pre_R, pre_G, pre_B, SAR, SAR_grad, prior, SAR * prior,
       SAR_grad * prior, expected_sar, expected_sar_grad, residual_sar,
       abs_residual_sar, residual_grad, abs_residual_grad]
    """

    def __init__(
        self,
        data_root: str | Path,
        manifest: str | Path,
        train: bool,
        prior_type: str = "predicted",
        input_mode: str = "sar_prior",
        crop_size: int | None = None,
        crop_probabilities: dict[str, float] | None = None,
        hflip: bool = False,
        vflip: bool = False,
        rotate90: bool = False,
        ogsr_feature_root: str | Path | None = None,
        ogsr_feature_keys: list[str] | tuple[str, ...] | None = None,
        limit: int | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.manifest_path = resolve_manifest(data_root, manifest)
        self.rows = read_jsonl(self.manifest_path)
        if limit is not None:
            self.rows = self.rows[:limit]
        self.train = train
        self.prior_type = prior_type.lower()
        self.input_mode = input_mode.lower()
        self.crop_size = crop_size
        self.crop_probabilities = crop_probabilities or {
            "damaged": 0.4,
            "destroyed": 0.4,
            "building": 0.2,
            "random": 0.0,
        }
        self.hflip = hflip
        self.vflip = vflip
        self.rotate90 = rotate90
        self.ogsr_feature_root = Path(ogsr_feature_root) if ogsr_feature_root else None
        self.ogsr_feature_keys = tuple(ogsr_feature_keys or DEFAULT_OGSR_TEXTURE_FEATURE_KEYS)
        self._ogsr_feature_index = self._load_ogsr_feature_index()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int | tuple[int, int]) -> dict[str, Any]:
        forced_crop_class: int | None = None
        if isinstance(index, tuple):
            index, forced_crop_class = int(index[0]), int(index[1])
        row = self.rows[index]
        sar = self.load_sar(row)
        target = self.load_target(row)
        prior = self.load_prior(row)
        pre = self.load_pre_image(row) if self._uses_pre_image() else None
        ogsr_features = self.load_ogsr_features(row) if self._uses_ogsr_features() else None

        if self.train and self.crop_size:
            sar, prior, target, pre, ogsr_features = self._crop(
                sar,
                prior,
                target,
                self.crop_size,
                pre,
                ogsr_features,
                forced_target_class=forced_crop_class,
            )
        if self.train:
            sar, prior, target, pre, ogsr_features = self._augment(sar, prior, target, pre, ogsr_features)

        sar_f = sar.astype(np.float32) / 255.0
        prior_f = prior.astype(np.float32)
        sar_grad = self._sar_gradient(sar_f)
        normalized_mode = self.input_mode.replace("-", "_")
        if normalized_mode in {"prior_only"}:
            image = np.stack([np.zeros_like(sar_f), prior_f, np.zeros_like(sar_f)], axis=0)
        elif normalized_mode in {"sar_only", "no_prior"}:
            image = np.stack([sar_f, np.zeros_like(sar_f), np.zeros_like(sar_f)], axis=0)
        elif normalized_mode in {"pre_only", "pre_optical_only"}:
            pre_f = self._pre_float_chw(pre)
            zeros = np.zeros_like(sar_f)[None, :, :]
            image = np.concatenate([pre_f, zeros, zeros, zeros], axis=0)
        elif normalized_mode in {"pre_sar", "rgb_sar", "pre_optical_sar"}:
            pre_f = self._pre_float_chw(pre)
            image = np.concatenate([pre_f, sar_f[None, :, :]], axis=0)
        elif normalized_mode in {"pre_sar_background_only", "background_only"}:
            pre_f = self._pre_float_chw(pre)
            background = (target == 0).astype(np.float32)
            zeros = np.zeros_like(sar_f)[None, :, :]
            image = np.concatenate(
                [
                    pre_f * background[None, :, :],
                    (sar_f * background)[None, :, :],
                    zeros,
                    zeros,
                ],
                axis=0,
            )
        elif normalized_mode in {"pre_prior", "pre_prior_only", "pre_zero_sar_prior"}:
            pre_f = self._pre_float_chw(pre)
            image = np.concatenate(
                [
                    pre_f,
                    np.zeros_like(sar_f)[None, :, :],
                    prior_f[None, :, :],
                    np.zeros_like(sar_f)[None, :, :],
                ],
                axis=0,
            )
        elif normalized_mode in {"pre_sar_prior", "pre_optical_sar_prior"}:
            pre_f = self._pre_float_chw(pre)
            image = np.concatenate(
                [
                    pre_f,
                    sar_f[None, :, :],
                    prior_f[None, :, :],
                    (sar_f * prior_f)[None, :, :],
                ],
                axis=0,
            )
        elif normalized_mode in {"prior_only5", "zero_pre_prior_zero_sar5"}:
            zeros = np.zeros_like(sar_f)[None, :, :]
            image = np.concatenate([zeros, zeros, zeros, prior_f[None, :, :], zeros], axis=0)
        elif normalized_mode in {"pre_prior5", "pre_prior_zero_sar5"}:
            pre_f = self._pre_float_chw(pre)
            image = np.concatenate(
                [pre_f, prior_f[None, :, :], np.zeros_like(sar_f)[None, :, :]], axis=0
            )
        elif normalized_mode in {"pre_prior_sar5", "pre_optical_prior_sar5"}:
            pre_f = self._pre_float_chw(pre)
            image = np.concatenate([pre_f, prior_f[None, :, :], sar_f[None, :, :]], axis=0)
        elif normalized_mode in {"pre_sar_texture_prior", "pre_sar_grad_prior", "pre_sar_gradient_prior"}:
            pre_f = self._pre_float_chw(pre)
            image = np.concatenate(
                [
                    pre_f,
                    sar_f[None, :, :],
                    sar_grad[None, :, :],
                    prior_f[None, :, :],
                    (sar_f * prior_f)[None, :, :],
                    (sar_grad * prior_f)[None, :, :],
                ],
                axis=0,
            )
        elif normalized_mode in {"pre_sar_ogsr_lite_prior", "pre_sar_ogsr_prior"}:
            pre_f = self._pre_float_chw(pre)
            feature_f = self._ogsr_select_chw(ogsr_features, ("expected_sar", "residual_sar", "abs_residual_sar"))
            image = np.concatenate(
                [
                    pre_f,
                    sar_f[None, :, :],
                    prior_f[None, :, :],
                    feature_f,
                ],
                axis=0,
            )
        elif normalized_mode in {
            "pre_sar_ogsr_texture_prior",
            "pre_sar_ogsr_texture_residual_prior",
            "pre_sar_texture_ogsr_prior",
        }:
            pre_f = self._pre_float_chw(pre)
            feature_f = self._ogsr_select_chw(ogsr_features, self.ogsr_feature_keys)
            image = np.concatenate(
                [
                    pre_f,
                    sar_f[None, :, :],
                    sar_grad[None, :, :],
                    prior_f[None, :, :],
                    feature_f,
                ],
                axis=0,
            )
        elif normalized_mode in {
            "pre_sar_texture_ogsr_add_prior",
            "pre_sar_texture_ogsr_additive_prior",
            "pre_sar_texture_ogsr_full_prior",
        }:
            pre_f = self._pre_float_chw(pre)
            feature_f = self._ogsr_select_chw(ogsr_features, self.ogsr_feature_keys)
            image = np.concatenate(
                [
                    pre_f,
                    sar_f[None, :, :],
                    sar_grad[None, :, :],
                    prior_f[None, :, :],
                    (sar_f * prior_f)[None, :, :],
                    (sar_grad * prior_f)[None, :, :],
                    feature_f,
                ],
                axis=0,
            )
        else:
            image = np.stack([sar_f, prior_f, sar_f * prior_f], axis=0)

        item: dict[str, Any] = {
            "image": torch.from_numpy(np.ascontiguousarray(image)).float(),
            "mask": torch.from_numpy(np.ascontiguousarray(target)).long(),
            "sar": torch.from_numpy(np.ascontiguousarray(sar_f[None, :, :])).float(),
            "sar_grad": torch.from_numpy(np.ascontiguousarray(sar_grad[None, :, :])).float(),
            "prior": torch.from_numpy(np.ascontiguousarray(prior_f[None, :, :])).float(),
            "manifest_path": str(self.manifest_path),
        }
        if pre is not None:
            item["pre_image"] = torch.from_numpy(np.ascontiguousarray(self._pre_float_chw(pre))).float()
        if ogsr_features is not None:
            item["ogsr_features"] = torch.from_numpy(
                np.ascontiguousarray(np.transpose(ogsr_features.astype(np.float32), (2, 0, 1)))
            ).float()
        item.update(get_metadata(row))
        return item

    def _uses_pre_image(self) -> bool:
        return "pre" in self.input_mode.replace("-", "_").split("_")

    def _uses_ogsr_features(self) -> bool:
        return "ogsr" in self.input_mode.replace("-", "_").split("_")

    @staticmethod
    def _pre_float_chw(pre: np.ndarray | None) -> np.ndarray:
        if pre is None:
            raise ValueError("input_mode requires pre_image but sample has not loaded it")
        return np.transpose(pre.astype(np.float32) / 255.0, (2, 0, 1))

    @staticmethod
    def _sar_gradient(sar_f: np.ndarray) -> np.ndarray:
        dy, dx = np.gradient(sar_f.astype(np.float32))
        grad = np.sqrt(dx * dx + dy * dy)
        hi = float(np.percentile(grad, 99.0))
        if hi <= 1e-6:
            return np.zeros_like(sar_f, dtype=np.float32)
        return np.clip(grad / hi, 0.0, 1.0).astype(np.float32)

    def _ogsr_select_chw(self, features: np.ndarray | None, keys: tuple[str, ...]) -> np.ndarray:
        if features is None:
            raise ValueError("input_mode requires OGSR features but ogsr_feature_root is not configured")
        try:
            indices = [self.ogsr_feature_keys.index(key) for key in keys]
        except ValueError as exc:
            raise ValueError(f"requested OGSR feature keys are not available: {keys}") from exc
        selected = features[:, :, indices]
        return np.transpose(selected.astype(np.float32), (2, 0, 1))

    def _load_ogsr_feature_index(self) -> dict[str, str]:
        if self.ogsr_feature_root is None:
            return {}
        splits = sorted({str(row.get("split", "") or "").strip() for row in self.rows})
        splits = [split for split in splits if split]
        if not splits:
            splits = ["train", "val", "test"]
        index: dict[str, str] = {}
        for split in splits:
            path = self.ogsr_feature_root / f"index_{split}.json"
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            features = payload.get("features", payload) if isinstance(payload, dict) else payload
            if not isinstance(features, dict):
                raise ValueError(f"invalid OGSR feature index format: {path}")
            index.update({str(key): str(value) for key, value in features.items()})
        missing = [str(row.get("id", "")) for row in self.rows if str(row.get("id", "")) not in index]
        if missing:
            examples = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"missing OGSR features for {len(missing)} samples under {self.ogsr_feature_root}; "
                f"examples: {examples}"
            )
        return index

    def load_ogsr_features(self, row: dict[str, Any]) -> np.ndarray:
        if self.ogsr_feature_root is None:
            raise ValueError("ogsr_feature_root is required for OGSR input modes")
        sample_id = str(row.get("id", ""))
        value = self._ogsr_feature_index.get(sample_id)
        if value is None:
            raise FileNotFoundError(f"missing OGSR feature entry for id={sample_id}")
        path = Path(value)
        if not path.is_absolute():
            path = self.ogsr_feature_root / path
        arrays: list[np.ndarray] = []
        with np.load(path) as data:
            for key in self.ogsr_feature_keys:
                if key not in data:
                    raise KeyError(f"{path} does not contain OGSR feature '{key}'")
                array = np.asarray(data[key], dtype=np.float32)
                if array.ndim != 2:
                    raise ValueError(f"OGSR feature '{key}' in {path} must be 2D, got shape {array.shape}")
                arrays.append(array)
        stack = np.stack(arrays, axis=-1)
        return stack.astype(np.float32)

    def load_sar(self, row: dict[str, Any]) -> np.ndarray:
        path = resolve_data_path(self.data_root, str(row["post_sar"]))
        if path is None:
            raise FileNotFoundError("post_sar is null")
        return _load_gray_uint8(path)

    def load_pre_image(self, row: dict[str, Any]) -> np.ndarray:
        path = resolve_data_path(self.data_root, str(row["pre_image"]))
        if path is None:
            raise FileNotFoundError("pre_image is null")
        return _load_rgb_uint8(path)

    def load_target(self, row: dict[str, Any]) -> np.ndarray:
        path = resolve_data_path(self.data_root, str(row["mask_multiclass"]))
        if path is None:
            raise FileNotFoundError("mask_multiclass is null")
        mask = _load_gray_uint8(path)
        values = set(int(v) for v in np.unique(mask))
        if not values.issubset({0, 1, 2, 3}):
            raise ValueError(f"invalid target values for id={row.get('id')}: {sorted(values)}")
        return mask.astype(np.uint8)

    def load_prior(self, row: dict[str, Any]) -> np.ndarray:
        prior_type = self.prior_type
        if prior_type in {"none", "no_prior", "no-prior"}:
            target = self.load_target(row)
            return np.zeros_like(target, dtype=np.float32)
        if prior_type == "oracle":
            value = row.get("building_prior") or row.get("oracle_building_mask")
            path = resolve_data_path(self.data_root, str(value))
            if path is None:
                raise FileNotFoundError("oracle prior is null")
            return (_load_gray_uint8(path) > 0).astype(np.float32)
        if prior_type == "predicted":
            value = row.get("building_prior") or row.get("pred_building_prob")
            path = resolve_data_path(self.data_root, str(value))
            if path is None:
                raise FileNotFoundError("predicted prior is null")
            if path.suffix.lower() == ".npz":
                return _load_prior_npz(path)
            return _load_gray_uint8(path).astype(np.float32) / 255.0
        raise ValueError(f"unknown prior_type: {self.prior_type}")

    def load_prior_binary(self, row: dict[str, Any], prior_type: str | None = None) -> np.ndarray:
        chosen = (prior_type or self.prior_type).lower()
        if chosen in {"none", "no_prior", "no-prior"}:
            return np.zeros_like(self.load_target(row), dtype=np.uint8)
        if chosen == "oracle":
            value = row.get("building_prior_binary") or row.get("building_prior") or row.get("oracle_building_mask")
        elif chosen == "predicted":
            value = row.get("building_prior_binary") or row.get("pred_building_binary")
        else:
            raise ValueError(f"unknown prior_type: {chosen}")
        path = resolve_data_path(self.data_root, str(value))
        if path is None:
            raise FileNotFoundError(f"{chosen} binary prior is null")
        return (_load_gray_uint8(path) > 0).astype(np.uint8)

    def _crop(
        self,
        sar: np.ndarray,
        prior: np.ndarray,
        target: np.ndarray,
        size: int,
        pre: np.ndarray | None = None,
        ogsr_features: np.ndarray | None = None,
        forced_target_class: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        h, w = target.shape
        if h <= size or w <= size:
            return sar, prior, target, pre, ogsr_features
        if forced_target_class is None:
            y, x = self._choose_crop_origin(target, size)
        else:
            origin = self._origin_for_target(target, size, forced_target_class)
            if origin is None:
                raise RuntimeError(
                    f"forced crop class {forced_target_class} is absent from the selected sample"
                )
            y, x = origin
        return (
            sar[y : y + size, x : x + size],
            prior[y : y + size, x : x + size],
            target[y : y + size, x : x + size],
            None if pre is None else pre[y : y + size, x : x + size, :],
            None if ogsr_features is None else ogsr_features[y : y + size, x : x + size, :],
        )

    def _choose_crop_origin(self, target: np.ndarray, size: int) -> tuple[int, int]:
        r = random.random()
        cumulative = 0.0
        selected = "random"
        for name, prob in self.crop_probabilities.items():
            cumulative += max(0.0, float(prob))
            if r <= cumulative:
                selected = name
                break

        fallbacks = {
            "damaged": [2, 3, "building", "random"],
            "destroyed": [3, 2, "building", "random"],
            "building": ["building", 2, 3, "random"],
            "random": ["random"],
        }.get(selected, ["random"])

        for target_class in fallbacks:
            origin = self._origin_for_target(target, size, target_class)
            if origin is not None:
                return origin
        h, w = target.shape
        return random.randint(0, h - size), random.randint(0, w - size)

    @staticmethod
    def _origin_for_target(target: np.ndarray, size: int, target_class: int | str) -> tuple[int, int] | None:
        h, w = target.shape
        if target_class == "random":
            return random.randint(0, h - size), random.randint(0, w - size)
        if target_class == "building":
            ys, xs = np.where(target > 0)
        else:
            ys, xs = np.where(target == int(target_class))
        if len(ys) == 0:
            return None
        idx = random.randrange(len(ys))
        cy = int(ys[idx])
        cx = int(xs[idx])
        y = min(max(cy - random.randint(0, size - 1), 0), h - size)
        x = min(max(cx - random.randint(0, size - 1), 0), w - size)
        return y, x

    def _augment(
        self,
        sar: np.ndarray,
        prior: np.ndarray,
        target: np.ndarray,
        pre: np.ndarray | None = None,
        ogsr_features: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        if self.hflip and random.random() < 0.5:
            sar = np.flip(sar, axis=1)
            prior = np.flip(prior, axis=1)
            target = np.flip(target, axis=1)
            if pre is not None:
                pre = np.flip(pre, axis=1)
            if ogsr_features is not None:
                ogsr_features = np.flip(ogsr_features, axis=1)
        if self.vflip and random.random() < 0.5:
            sar = np.flip(sar, axis=0)
            prior = np.flip(prior, axis=0)
            target = np.flip(target, axis=0)
            if pre is not None:
                pre = np.flip(pre, axis=0)
            if ogsr_features is not None:
                ogsr_features = np.flip(ogsr_features, axis=0)
        if self.rotate90:
            k = random.randint(0, 3)
            if k:
                sar = np.rot90(sar, k)
                prior = np.rot90(prior, k)
                target = np.rot90(target, k)
                if pre is not None:
                    pre = np.rot90(pre, k, axes=(0, 1))
                if ogsr_features is not None:
                    ogsr_features = np.rot90(ogsr_features, k, axes=(0, 1))
        return sar, prior, target, pre, ogsr_features
