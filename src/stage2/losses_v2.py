from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BuildingOnlyGradeLoss(nn.Module):
    """Three-grade CE/focal loss evaluated only where the GT is a building."""

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        focal_gamma: float = 0.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.focal_gamma = float(focal_gamma)
        self.label_smoothing = float(label_smoothing)
        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            if class_weights.numel() != 3:
                raise ValueError("Stage-2 v2 class_weights must contain intact/damaged/destroyed")
            self.register_buffer("class_weights", class_weights.float())

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4 or logits.shape[1] != 3:
            raise ValueError(f"expected [B,3,H,W] logits, got {tuple(logits.shape)}")
        if target.ndim == 4:
            target = target.squeeze(1)
        valid = target > 0
        if not torch.any(valid):
            return logits.sum() * 0.0
        flat_logits = logits.permute(0, 2, 3, 1)[valid]
        flat_target = target[valid].long() - 1
        weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
        ce = F.cross_entropy(
            flat_logits,
            flat_target,
            weight=weights,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        if self.focal_gamma > 0:
            pt = torch.softmax(flat_logits, dim=1).gather(1, flat_target[:, None]).squeeze(1)
            ce = ((1.0 - pt).clamp_min(0.0) ** self.focal_gamma) * ce
        return ce.mean()


class BuildingOnlyGradeBinaryAuxLoss(nn.Module):
    """Three-grade CE/focal plus an intact-vs-damage auxiliary term."""

    def __init__(
        self,
        grade_loss: BuildingOnlyGradeLoss,
        binary_aux_weight: float = 0.3,
        binary_pos_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.grade_loss = grade_loss
        self.binary_aux_weight = float(binary_aux_weight)
        self.register_buffer("binary_pos_weight", torch.tensor(float(binary_pos_weight), dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        grade_loss = self.grade_loss(logits, target)
        if self.binary_aux_weight <= 0:
            return grade_loss
        if target.ndim == 4:
            target = target.squeeze(1)
        valid = target > 0
        if not torch.any(valid):
            return grade_loss
        flat_logits = logits.permute(0, 2, 3, 1)[valid]
        binary_logit = torch.logsumexp(flat_logits[:, 1:3], dim=1) - flat_logits[:, 0]
        binary_target = (target[valid] > 1).float()
        binary_loss = F.binary_cross_entropy_with_logits(
            binary_logit,
            binary_target,
            pos_weight=self.binary_pos_weight.to(logits.device),
        )
        return grade_loss + self.binary_aux_weight * binary_loss


def build_stage2_v2_loss(cfg: dict, class_weights: torch.Tensor | None = None) -> nn.Module:
    loss_cfg = dict(cfg.get("loss", {}))
    name = str(loss_cfg.get("name", "building_only_ce")).lower()
    gamma = float(loss_cfg.get("focal_gamma", 0.0))
    if name in {"building_only_focal", "focal"} and gamma <= 0:
        gamma = 2.0
    if name not in {
        "building_only_ce",
        "masked_ce",
        "building_only_focal",
        "focal",
        "building_only_ce_binary_aux",
        "building_only_focal_binary_aux",
        "binary_aux",
    }:
        raise ValueError(f"unsupported Stage-2 v2 loss: {name}")
    grade_loss = BuildingOnlyGradeLoss(
        class_weights=class_weights,
        focal_gamma=gamma,
        label_smoothing=float(loss_cfg.get("label_smoothing", 0.0)),
    )
    if name in {"building_only_ce_binary_aux", "building_only_focal_binary_aux", "binary_aux"}:
        return BuildingOnlyGradeBinaryAuxLoss(
            grade_loss=grade_loss,
            binary_aux_weight=float(loss_cfg.get("binary_aux_weight", 0.3)),
            binary_pos_weight=float(loss_cfg.get("binary_pos_weight", 1.0)),
        )
    return grade_loss
