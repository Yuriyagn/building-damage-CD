from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
from skimage.measure import label

from .metrics import Stage2DamageMeter


GRADE_NAMES = ("intact", "damaged", "destroyed")


def summarize_event_generalization(
    rows: list[dict[str, Any]],
    *,
    bootstrap_iterations: int = 10_000,
    bootstrap_seed: int = 20260803,
) -> dict[str, Any]:
    """Aggregate event-level metrics without treating pixels as independent samples."""
    if not rows:
        raise ValueError("event rows must not be empty")

    def damage_macro(row: dict[str, Any]) -> float:
        if "building_only_damage_macro_f1" in row:
            return float(row["building_only_damage_macro_f1"])
        return 0.5 * (
            float(row["building_only_f1_damaged"])
            + float(row["building_only_f1_destroyed"])
        )

    event_values = np.asarray(
        [float(row["building_only_macro_f1_3class"]) for row in rows], dtype=np.float64
    )
    event_damage_values = np.asarray([damage_macro(row) for row in rows], dtype=np.float64)

    def event_class_values(selected: list[dict[str, Any]]) -> np.ndarray:
        values = []
        for row in selected:
            for name in GRADE_NAMES:
                if float(row.get(f"building_only_support_{name}", 0.0)) > 0:
                    values.append(float(row[f"building_only_f1_{name}"]))
        return np.asarray(values, dtype=np.float64)

    cells = event_class_values(rows)
    rng = np.random.default_rng(bootstrap_seed)
    boot_event = np.empty(bootstrap_iterations, dtype=np.float64)
    boot_event_damage = np.empty(bootstrap_iterations, dtype=np.float64)
    boot_event_class = np.empty(bootstrap_iterations, dtype=np.float64)
    for index in range(bootstrap_iterations):
        chosen = rng.integers(0, len(rows), size=len(rows))
        sampled = [rows[int(item)] for item in chosen]
        boot_event[index] = float(np.mean([float(row["building_only_macro_f1_3class"]) for row in sampled]))
        boot_event_damage[index] = float(np.mean([damage_macro(row) for row in sampled]))
        sampled_cells = event_class_values(sampled)
        boot_event_class[index] = float(np.mean(sampled_cells)) if sampled_cells.size else 0.0

    out: dict[str, Any] = {
        "event_count": len(rows),
        "event_class_cell_count": int(cells.size),
        "event_macro_bo_f1": float(event_values.mean()),
        "event_macro_bo_f1_ci95": [float(x) for x in np.quantile(boot_event, [0.025, 0.975])],
        "event_macro_bo_damage_f1": float(event_damage_values.mean()),
        "event_macro_bo_damage_f1_ci95": [
            float(x) for x in np.quantile(boot_event_damage, [0.025, 0.975])
        ],
        "event_class_macro_f1": float(cells.mean()) if cells.size else 0.0,
        "event_class_macro_f1_ci95": [
            float(x) for x in np.quantile(boot_event_class, [0.025, 0.975])
        ],
        "worst_event_bo_f1": float(event_values.min()),
        "worst_event_id": str(rows[int(event_values.argmin())].get("event_id", "unknown")),
        "bootstrap_unit": "event",
        "bootstrap_iterations": int(bootstrap_iterations),
        "bootstrap_seed": int(bootstrap_seed),
    }
    for name in ("damaged", "destroyed"):
        values = np.asarray(
            [
                float(row[f"building_only_f1_{name}"])
                for row in rows
                if float(row.get(f"building_only_support_{name}", 0.0)) > 0
            ],
            dtype=np.float64,
        )
        out[f"{name}_event_count"] = int(values.size)
        out[f"{name}_event_mean_f1"] = float(values.mean()) if values.size else 0.0
        out[f"{name}_event_std_f1"] = float(values.std(ddof=0)) if values.size else 0.0
    return out


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
