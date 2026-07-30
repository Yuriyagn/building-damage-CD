from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch

from .common import CLASS_NAMES


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


class Stage2DamageMeter:
    def __init__(self, num_classes: int = 4) -> None:
        self.num_classes = num_classes
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.damage_binary = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        self.building_only_class = {
            c: {"tp": 0, "fp": 0, "fn": 0}
            for c in range(1, num_classes)
        }
        self.building_only_damage = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    def update_logits(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        pred = torch.argmax(logits.detach(), dim=1).cpu().numpy()
        true = target.detach().cpu().numpy()
        self.update_prediction(pred, true)

    def update_prediction(self, pred: np.ndarray, target: np.ndarray) -> None:
        pred_arr = np.asarray(pred)
        true_arr = np.asarray(target)
        if pred_arr.ndim == 3:
            pred_flat = pred_arr.reshape(-1)
            true_flat = true_arr.reshape(-1)
        else:
            pred_flat = pred_arr.reshape(-1)
            true_flat = true_arr.reshape(-1)
        valid = (true_flat >= 0) & (true_flat < self.num_classes) & (pred_flat >= 0) & (pred_flat < self.num_classes)
        encoded = self.num_classes * true_flat[valid].astype(np.int64) + pred_flat[valid].astype(np.int64)
        counts = np.bincount(encoded, minlength=self.num_classes * self.num_classes)
        self.confusion += counts.reshape(self.num_classes, self.num_classes)

        true_damage = np.isin(true_flat, [2, 3])
        pred_damage = np.isin(pred_flat, [2, 3])
        self._add_binary(self.damage_binary, pred_damage, true_damage)

        gt_building = true_flat > 0
        if gt_building.any():
            p = pred_flat[gt_building]
            t = true_flat[gt_building]
            for cls in range(1, self.num_classes):
                pred_cls = p == cls
                true_cls = t == cls
                self.building_only_class[cls]["tp"] += int(np.logical_and(pred_cls, true_cls).sum())
                self.building_only_class[cls]["fp"] += int(np.logical_and(pred_cls, ~true_cls).sum())
                self.building_only_class[cls]["fn"] += int(np.logical_and(~pred_cls, true_cls).sum())
            self._add_binary(self.building_only_damage, np.isin(p, [2, 3]), np.isin(t, [2, 3]))

    @staticmethod
    def _add_binary(bucket: dict[str, int], pred: np.ndarray, true: np.ndarray) -> None:
        bucket["tp"] += int(np.logical_and(pred, true).sum())
        bucket["fp"] += int(np.logical_and(pred, ~true).sum())
        bucket["fn"] += int(np.logical_and(~pred, true).sum())
        bucket["tn"] += int(np.logical_and(~pred, ~true).sum())

    def compute(self) -> dict[str, float]:
        out: dict[str, float] = {}
        total = int(self.confusion.sum())
        correct = int(np.trace(self.confusion))
        out["overall_accuracy"] = _safe_div(correct, total)

        ious = []
        f1s = []
        for cls in range(self.num_classes):
            name = CLASS_NAMES.get(cls, str(cls))
            tp = float(self.confusion[cls, cls])
            fp = float(self.confusion[:, cls].sum() - self.confusion[cls, cls])
            fn = float(self.confusion[cls, :].sum() - self.confusion[cls, cls])
            precision = _safe_div(tp, tp + fp)
            recall = _safe_div(tp, tp + fn)
            iou = _safe_div(tp, tp + fp + fn)
            f1 = _safe_div(2.0 * precision * recall, precision + recall)
            ious.append(iou)
            f1s.append(f1)
            out[f"iou_{name}"] = iou
            out[f"f1_{name}"] = f1
            out[f"precision_{name}"] = precision
            out[f"recall_{name}"] = recall
            out[f"support_{name}"] = float(self.confusion[cls, :].sum())
            out[f"predicted_{name}"] = float(self.confusion[:, cls].sum())

        out["miou_4class"] = float(np.mean(ious)) if ious else 0.0
        out["macro_f1_4class"] = float(np.mean(f1s)) if f1s else 0.0
        out.update(self._binary_metrics("damage_binary", self.damage_binary))
        out["damage_macro_f1"] = float(np.mean([out["f1_damaged"], out["f1_destroyed"]]))

        building_ious = []
        building_f1s = []
        for cls in range(1, self.num_classes):
            name = CLASS_NAMES.get(cls, str(cls))
            counts = self.building_only_class[cls]
            tp = float(counts["tp"])
            fp = float(counts["fp"])
            fn = float(counts["fn"])
            precision = _safe_div(tp, tp + fp)
            recall = _safe_div(tp, tp + fn)
            iou = _safe_div(tp, tp + fp + fn)
            f1 = _safe_div(2.0 * precision * recall, precision + recall)
            building_ious.append(iou)
            building_f1s.append(f1)
            out[f"building_only_iou_{name}"] = iou
            out[f"building_only_f1_{name}"] = f1
            out[f"building_only_precision_{name}"] = precision
            out[f"building_only_recall_{name}"] = recall
        out["building_only_miou_3class"] = float(np.mean(building_ious)) if building_ious else 0.0
        out["building_only_macro_f1_3class"] = float(np.mean(building_f1s)) if building_f1s else 0.0
        out.update(self._binary_metrics("building_only_damage_binary", self.building_only_damage))
        out["building_only_damage_macro_f1"] = float(
            np.mean([out["building_only_f1_damaged"], out["building_only_f1_destroyed"]])
        )
        out["pixels_total"] = float(total)
        return out

    @staticmethod
    def _binary_metrics(prefix: str, counts: dict[str, int]) -> dict[str, float]:
        tp = float(counts["tp"])
        fp = float(counts["fp"])
        fn = float(counts["fn"])
        tn = float(counts["tn"])
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2.0 * precision * recall, precision + recall)
        return {
            f"{prefix}_iou": _safe_div(tp, tp + fp + fn),
            f"{prefix}_f1": f1,
            f"{prefix}_precision": precision,
            f"{prefix}_recall": recall,
            f"{prefix}_tp": tp,
            f"{prefix}_fp": fp,
            f"{prefix}_fn": fn,
            f"{prefix}_tn": tn,
        }


class GroupedStage2Meters:
    def __init__(self, keys: list[str]) -> None:
        self.keys = keys
        self.groups: dict[str, dict[str, Stage2DamageMeter]] = {
            key: defaultdict(Stage2DamageMeter)
            for key in keys
        }

    def update(self, pred: np.ndarray, target: np.ndarray, metadata: dict[str, Any]) -> None:
        for key in self.keys:
            value = str(metadata.get(key, "") or "unknown")
            self.groups[key][value].update_prediction(pred, target)

    def rows(self, key: str) -> list[dict[str, Any]]:
        out = []
        for value, meter in sorted(self.groups[key].items()):
            out.append({key: value, **meter.compute()})
        return out

