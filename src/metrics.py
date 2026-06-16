from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch


class BinarySegmentationMeter:
    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        pred = torch.sigmoid(logits).detach() >= self.threshold
        true = target.detach() >= 0.5
        self.tp += int((pred & true).sum().item())
        self.fp += int((pred & ~true).sum().item())
        self.fn += int((~pred & true).sum().item())
        self.tn += int((~pred & ~true).sum().item())

    def update_binary(self, pred: np.ndarray, target: np.ndarray) -> None:
        p = pred.astype(bool)
        t = target.astype(bool)
        self.tp += int(np.logical_and(p, t).sum())
        self.fp += int(np.logical_and(p, ~t).sum())
        self.fn += int(np.logical_and(~p, t).sum())
        self.tn += int(np.logical_and(~p, ~t).sum())

    def compute(self) -> dict[str, float]:
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        oa = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else 0.0
        return {
            "iou_building": iou,
            "f1_building": f1,
            "precision_building": precision,
            "recall_building": recall,
            "overall_accuracy": oa,
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
            "tn": float(tn),
        }


def binary_boundary(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    if m.ndim == 3:
        m = m.squeeze()
    padded = np.pad(m, 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    eroded = (
        center
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return center ^ eroded


def boundary_f1(pred: np.ndarray, target: np.ndarray) -> float:
    pb = binary_boundary(pred)
    tb = binary_boundary(target)
    tp = np.logical_and(pb, tb).sum()
    fp = np.logical_and(pb, ~tb).sum()
    fn = np.logical_and(~pb, tb).sum()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def group_metrics(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, BinarySegmentationMeter] = defaultdict(BinarySegmentationMeter)
    for row in rows:
        group = str(row.get(key, "") or "unknown")
        grouped[group].tp += int(row["tp"])
        grouped[group].fp += int(row["fp"])
        grouped[group].fn += int(row["fn"])
        grouped[group].tn += int(row["tn"])

    out = []
    for group, meter in sorted(grouped.items()):
        item = {key: group, **meter.compute()}
        out.append(item)
    return out
