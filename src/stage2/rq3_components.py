from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.measure import label


@dataclass(frozen=True)
class ComponentTargets:
    component_map: np.ndarray
    valid_loss_mask: np.ndarray
    component_count: int
    valid_component_count: int
    ambiguous_component_count: int


def predicted_prior_components(
    prior: np.ndarray,
    target: np.ndarray,
    *,
    threshold: float,
    min_area: int = 4,
    label_purity: float = 0.8,
) -> ComponentTargets:
    """Build deployable ROIs from a predicted prior and label them for training.

    Geometry depends only on ``prior``.  The target is consulted solely to mark
    which predicted components have an unambiguous training label.  Validation
    and inference therefore never replace predicted geometry with GT geometry.
    """

    if prior.ndim != 2 or target.ndim != 2 or prior.shape != target.shape:
        raise ValueError("prior and target must be same-shaped 2D arrays")
    if min_area < 1:
        raise ValueError("min_area must be positive")
    if not 0.0 <= label_purity <= 1.0:
        raise ValueError("label_purity must be in [0, 1]")

    raw = label(prior >= float(threshold), connectivity=1)
    component_map = np.zeros_like(raw, dtype=np.int32)
    valid_loss_mask = np.zeros_like(raw, dtype=bool)
    next_id = 1
    valid_count = 0
    ambiguous_count = 0
    for raw_id in range(1, int(raw.max()) + 1):
        region = raw == raw_id
        if int(region.sum()) < int(min_area):
            continue
        component_map[region] = next_id
        values = target[region]
        if values.size:
            counts = np.bincount(values.astype(np.int64), minlength=4)[:4]
            majority = int(np.argmax(counts))
            purity = float(counts[majority] / int(values.size))
            if majority in {1, 2, 3} and purity >= float(label_purity):
                valid_loss_mask[region] = True
                valid_count += 1
            else:
                ambiguous_count += 1
        else:
            ambiguous_count += 1
        next_id += 1

    return ComponentTargets(
        component_map=component_map,
        valid_loss_mask=valid_loss_mask,
        component_count=next_id - 1,
        valid_component_count=valid_count,
        ambiguous_component_count=ambiguous_count,
    )
