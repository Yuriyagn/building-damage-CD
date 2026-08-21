from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F


def forward_stage2_model(
    model: torch.nn.Module,
    image: torch.Tensor,
    batch: dict[str, Any],
    device: torch.device,
) -> Any:
    if not bool(getattr(model, "requires_rq3_batch", False)):
        return model(image)
    component_map = batch.get("rq3_component_map")
    valid_mask = batch.get("rq3_instance_valid_mask")
    if not isinstance(component_map, torch.Tensor) or not isinstance(valid_mask, torch.Tensor):
        raise KeyError("RQ3 model requires component map and instance-valid mask")
    return model(
        image,
        component_map=component_map.to(device, non_blocking=True),
        instance_valid_mask=valid_mask.to(device, non_blocking=True),
    )


def target_for_stage2_loss(output: Any, target: torch.Tensor) -> torch.Tensor:
    if not isinstance(output, dict):
        return target
    valid_mask = output.get("rq3_loss_valid_mask")
    if valid_mask is None:
        return target
    if not isinstance(valid_mask, torch.Tensor) or valid_mask.shape != target.shape:
        raise ValueError("rq3_loss_valid_mask must match target shape")
    masked = target.clone()
    masked[~valid_mask.bool()] = 0
    return masked


def stage2_damage_loss(
    criterion: torch.nn.Module,
    output: Any,
    target: torch.Tensor,
) -> torch.Tensor:
    """Use one CE contribution per valid predicted component for E1."""

    if not isinstance(output, dict) or "instance_logits" not in output:
        logits = output["damage_logits"] if isinstance(output, dict) else output
        return criterion(logits, target_for_stage2_loss(output, target))
    logits = output["instance_logits"]
    pixel_index = output.get("instance_pixel_index")
    valid_mask = output.get("rq3_loss_valid_mask")
    if not isinstance(logits, torch.Tensor) or not isinstance(pixel_index, torch.Tensor):
        raise TypeError("instance supervision requires tensor logits and pixel index")
    if not isinstance(valid_mask, torch.Tensor) or valid_mask.shape != target.shape:
        raise ValueError("instance supervision requires a target-shaped valid mask")
    eligible_pixels = valid_mask.bool() & (pixel_index > 0) & (target >= 1) & (target <= 3)
    ids = pixel_index[eligible_pixels].long()
    values = target[eligible_pixels].long() - 1
    counts = logits.new_zeros((logits.shape[0] + 1, 3))
    if ids.numel():
        counts.index_add_(0, ids, torch.nn.functional.one_hot(values, num_classes=3).to(logits.dtype))
    eligible_instances = counts[1:].sum(dim=1) > 0
    if not torch.any(eligible_instances):
        return logits.sum() * 0.0
    labels = counts[1:].argmax(dim=1)[eligible_instances] + 1
    instance_logits = logits[eligible_instances].transpose(0, 1)[None, :, :, None]
    instance_target = labels[None, :, None]
    return criterion(instance_logits, instance_target)


def reliability_auxiliary_loss(
    output: Any,
    batch: dict[str, Any],
    device: torch.device,
) -> torch.Tensor | None:
    if not isinstance(output, dict):
        return None
    logits = output.get("reliability_logits")
    instance_batch_index = output.get("instance_batch_index")
    if logits is None:
        return None
    if not isinstance(logits, torch.Tensor) or not isinstance(instance_batch_index, torch.Tensor):
        raise TypeError("reliability outputs must be tensors")
    paired = batch.get("sar_is_paired")
    if not isinstance(paired, torch.Tensor):
        raise KeyError("reliability supervision requires sar_is_paired")
    if logits.numel() == 0:
        return logits.sum() * 0.0
    targets = paired.to(device=device, dtype=logits.dtype)[instance_batch_index.long()]
    return F.binary_cross_entropy_with_logits(logits.reshape(-1), targets.reshape(-1))
