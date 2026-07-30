from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class WeightedCEDiceLoss(nn.Module):
    def __init__(
        self,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        class_weights: torch.Tensor | None = None,
        include_background: bool = True,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.include_background = include_background
        self.eps = eps
        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            self.register_buffer("class_weights", class_weights.float())

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == 4:
            target = target.squeeze(1)
        weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
        ce = F.cross_entropy(logits, target.long(), weight=weights)
        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(target.long(), num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
        if not self.include_background and logits.shape[1] > 1:
            probs = probs[:, 1:]
            one_hot = one_hot[:, 1:]
        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dim=dims)
        denominator = probs.sum(dim=dims) + one_hot.sum(dim=dims)
        dice_per_class = (2.0 * intersection + self.eps) / (denominator + self.eps)
        dice = 1.0 - dice_per_class.mean()
        return self.ce_weight * ce + self.dice_weight * dice


def build_stage2_loss(cfg: dict, class_weights: torch.Tensor | None = None) -> nn.Module:
    loss_cfg = dict(cfg.get("loss", {}))
    return WeightedCEDiceLoss(
        ce_weight=float(loss_cfg.get("ce_weight", 0.5)),
        dice_weight=float(loss_cfg.get("dice_weight", 0.5)),
        class_weights=class_weights,
        include_background=bool(loss_cfg.get("dice_include_background", True)),
    )

