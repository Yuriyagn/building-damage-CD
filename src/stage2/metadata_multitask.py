from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


DISASTER_CLASSES = (
    "conflict",
    "earthquake",
    "explosion",
    "fire",
    "flood",
    "hurricane",
    "volcano",
)
DISASTER_CLASS_TO_INDEX = {name: index for index, name in enumerate(DISASTER_CLASSES)}


def validate_disaster_classes(values: Sequence[str] | None) -> tuple[str, ...]:
    classes = tuple(str(value) for value in (values or DISASTER_CLASSES))
    if classes != DISASTER_CLASSES:
        raise ValueError(
            f"disaster_classes must equal the frozen protocol order {DISASTER_CLASSES}, got {classes}"
        )
    return classes


def encode_disaster_types(
    values: Sequence[str],
    *,
    sample_ids: Sequence[str] | None = None,
    label_map: dict[str, str] | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    if label_map is not None and sample_ids is None:
        raise ValueError("sample_ids are required when a shuffled disaster label map is configured")
    labels: list[int] = []
    for index, value in enumerate(values):
        label = str(value)
        if label_map is not None:
            sample_id = str(sample_ids[index])
            if sample_id not in label_map:
                raise KeyError(f"missing shuffled disaster label for sample {sample_id!r}")
            label = str(label_map[sample_id])
        if label not in DISASTER_CLASS_TO_INDEX:
            raise ValueError(f"unknown or missing disaster_type {label!r}")
        labels.append(DISASTER_CLASS_TO_INDEX[label])
    return torch.tensor(labels, dtype=torch.long, device=device)


def load_label_map(path: str | Path | None) -> tuple[dict[str, str] | None, dict[str, Any]]:
    if not path:
        return None, {"mode": "true", "path": None, "sha256": None, "count": 0}
    label_path = Path(path)
    payload_bytes = label_path.read_bytes()
    payload = json.loads(payload_bytes)
    mapping = payload.get("labels", payload)
    if not isinstance(mapping, dict):
        raise ValueError(f"invalid disaster label map: {label_path}")
    normalized = {str(key): str(value) for key, value in mapping.items()}
    invalid = sorted(set(normalized.values()) - set(DISASTER_CLASSES))
    if invalid:
        raise ValueError(f"invalid labels in {label_path}: {invalid}")
    return normalized, {
        "mode": "shuffled",
        "path": str(label_path.resolve()),
        "sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "count": len(normalized),
    }


def disaster_class_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("disaster_type", "") or "") for row in rows)
    unknown = sorted(set(counts) - set(DISASTER_CLASSES))
    if unknown:
        raise ValueError(f"unknown or missing disaster_type values: {unknown}")
    missing = [name for name in DISASTER_CLASSES if counts[name] <= 0]
    if missing:
        raise ValueError(f"training manifest is missing frozen disaster classes: {missing}")
    return {name: int(counts[name]) for name in DISASTER_CLASSES}


def inverse_sqrt_class_weights(counts: dict[str, int]) -> list[float]:
    raw = np.asarray([1.0 / np.sqrt(float(counts[name])) for name in DISASTER_CLASSES])
    normalized = raw / float(raw.mean())
    return [float(value) for value in normalized]


@dataclass
class DisasterClassificationMeter:
    num_classes: int = len(DISASTER_CLASSES)

    def __post_init__(self) -> None:
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        predicted = torch.argmax(logits.detach(), dim=1).cpu().numpy().astype(np.int64)
        true = target.detach().cpu().numpy().astype(np.int64)
        encoded = self.num_classes * true + predicted
        self.confusion += np.bincount(
            encoded, minlength=self.num_classes * self.num_classes
        ).reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, float]:
        support = self.confusion.sum(axis=1).astype(np.float64)
        total = float(support.sum())
        f1s: list[float] = []
        recalls: list[float] = []
        for index in range(self.num_classes):
            tp = float(self.confusion[index, index])
            fp = float(self.confusion[:, index].sum() - tp)
            fn = float(self.confusion[index, :].sum() - tp)
            if support[index] > 0:
                f1s.append(2.0 * tp / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) else 0.0)
                recalls.append(tp / support[index])
        majority = int(np.argmax(support)) if total else 0
        majority_confusion = np.zeros_like(self.confusion)
        majority_confusion[:, majority] = support.astype(np.int64)
        majority_f1s = []
        for index in np.flatnonzero(support > 0):
            tp = float(majority_confusion[index, index])
            fp = float(majority_confusion[:, index].sum() - tp)
            fn = float(majority_confusion[index, :].sum() - tp)
            majority_f1s.append(
                2.0 * tp / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) else 0.0
            )
        return {
            "accuracy": float(np.trace(self.confusion) / total) if total else 0.0,
            "present_class_macro_f1": float(np.mean(f1s)) if f1s else 0.0,
            "present_class_balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
            "present_class_count": float(np.sum(support > 0)),
            "majority_present_class_macro_f1": (
                float(np.mean(majority_f1s)) if majority_f1s else 0.0
            ),
        }
