from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
from skimage.measure import label

from .metrics import Stage2DamageMeter


def _prefix_metrics(prefix: str, metrics: dict[str, float], keys: set[str] | None = None) -> dict[str, float]:
    return {
        f"{prefix}_{key}": float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and (keys is None or key in keys)
    }


class Stage2V2MeterBundle:
    """Track grade diagnosis and oracle/predicted support outputs in one pass."""

    def __init__(self) -> None:
        self.grade = Stage2DamageMeter()
        self.oracle_gate = Stage2DamageMeter()
        self.predicted_gate = Stage2DamageMeter()

    def update(
        self,
        grade_pred: np.ndarray,
        target: np.ndarray,
        predicted_support: np.ndarray,
    ) -> None:
        grade = np.asarray(grade_pred, dtype=np.uint8)
        true = np.asarray(target, dtype=np.uint8)
        oracle = np.where(true > 0, grade, 0).astype(np.uint8)
        predicted = np.where(np.asarray(predicted_support, dtype=bool), grade, 0).astype(np.uint8)
        self.grade.update_prediction(grade, true)
        self.oracle_gate.update_prediction(oracle, true)
        self.predicted_gate.update_prediction(predicted, true)

    def compute(self) -> dict[str, float]:
        grade = self.grade.compute()
        oracle = self.oracle_gate.compute()
        predicted = self.predicted_gate.compute()
        out = {
            key: float(value)
            for key, value in grade.items()
            if key.startswith("building_only_") or key == "pixels_total"
        }
        full_keys = {
            "miou_4class",
            "macro_f1_4class",
            "overall_accuracy",
            "damage_macro_f1",
            "damage_binary_f1",
            "f1_background",
            "f1_intact",
            "f1_damaged",
            "f1_destroyed",
        }
        out.update(_prefix_metrics("oracle_gate", oracle, full_keys))
        out.update(_prefix_metrics("predicted_gate", predicted, full_keys))
        return out


class GroupedV2Meters:
    def __init__(self, keys: list[str]) -> None:
        self.keys = keys
        self.groups: dict[str, dict[str, Stage2V2MeterBundle]] = {
            key: defaultdict(Stage2V2MeterBundle) for key in keys
        }

    def update(
        self,
        grade_pred: np.ndarray,
        target: np.ndarray,
        predicted_support: np.ndarray,
        metadata: dict[str, Any],
    ) -> None:
        for key in self.keys:
            value = str(metadata.get(key, "") or "unknown")
            self.groups[key][value].update(grade_pred, target, predicted_support)

    def rows(self, key: str) -> list[dict[str, Any]]:
        return [{key: value, **meter.compute()} for value, meter in sorted(self.groups[key].items())]


class CCSurrogateMeter:
    """Connected-component grade metric; explicitly not a true footprint metric."""

    def __init__(self, min_area: int = 4, small_max: int = 63, medium_max: int = 255) -> None:
        self.min_area = int(min_area)
        self.small_max = int(small_max)
        self.medium_max = int(medium_max)
        self.confusions = {name: np.zeros((3, 3), dtype=np.int64) for name in ("all", "small", "medium", "large")}
        self.total_components = 0
        self.ignored_components = 0
        self.low_purity_components = 0

    def _update_components(self, target: np.ndarray, predict_region: Any) -> None:
        true = np.asarray(target, dtype=np.uint8)
        components = label(true > 0, connectivity=1)
        for component_id in range(1, int(components.max()) + 1):
            region = components == component_id
            area = int(region.sum())
            self.total_components += 1
            if area < self.min_area:
                self.ignored_components += 1
                continue
            counts = np.bincount(true[region], minlength=4)[1:4]
            gt = int(np.argmax(counts))
            purity = float(counts[gt] / max(counts.sum(), 1))
            if purity < 0.8:
                self.low_purity_components += 1
            pred = int(predict_region(region))
            size = "small" if area <= self.small_max else ("medium" if area <= self.medium_max else "large")
            self.confusions["all"][gt, pred] += 1
            self.confusions[size][gt, pred] += 1

    def update(self, logits: torch.Tensor, target: np.ndarray) -> None:
        probabilities = torch.softmax(logits.detach().float(), dim=0).cpu().numpy()
        self._update_components(target, lambda region: np.argmax(probabilities[:, region].mean(axis=1)))

    def update_prediction(self, grade_pred: np.ndarray, target: np.ndarray) -> None:
        grade = np.asarray(grade_pred, dtype=np.uint8)
        true = np.asarray(target, dtype=np.uint8)
        if grade.shape != true.shape:
            raise ValueError(f"grade/target shape mismatch: {grade.shape} vs {true.shape}")
        grade_index = np.clip(grade, 1, 3).astype(np.int64) - 1
        self._update_components(target, lambda region: np.argmax(np.bincount(grade_index[region], minlength=3)))

    @staticmethod
    def _metrics(confusion: np.ndarray) -> dict[str, float]:
        f1s = []
        out: dict[str, float] = {"count": float(confusion.sum())}
        names = ("intact", "damaged", "destroyed")
        for index, name in enumerate(names):
            tp = float(confusion[index, index])
            fp = float(confusion[:, index].sum() - confusion[index, index])
            fn = float(confusion[index, :].sum() - confusion[index, index])
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
            out[f"f1_{name}"] = f1
            f1s.append(f1)
        out["macro_f1"] = float(np.mean(f1s))
        return out

    def compute(self) -> dict[str, float]:
        out: dict[str, float] = {
            "cc_surrogate_total_components": float(self.total_components),
            "cc_surrogate_ignored_components": float(self.ignored_components),
            "cc_surrogate_low_purity_components": float(self.low_purity_components),
        }
        for size, confusion in self.confusions.items():
            for key, value in self._metrics(confusion).items():
                out[f"cc_surrogate_{size}_{key}"] = value
        return out
