from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class MetadataAwareUNet(nn.Module):
    """Attach an image-level disaster head to an SMP U-Net encoder.

    Metadata labels supervise the shared visual encoder but are never accepted as
    model inputs.  Keeping the auxiliary head present for both M0 and M1 makes
    their initialization and parameterization identical.
    """

    def __init__(
        self,
        base_model: nn.Module,
        *,
        num_disaster_classes: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if not all(hasattr(base_model, name) for name in ("encoder", "decoder", "segmentation_head")):
            raise TypeError("MetadataAwareUNet requires an SMP-style encoder/decoder model")
        encoder_channels = list(getattr(base_model.encoder, "out_channels", []))
        if not encoder_channels:
            raise ValueError("metadata-aware encoder does not expose out_channels")
        self.base_model = base_model
        self.disaster_head = nn.Sequential(
            nn.Dropout(float(dropout)),
            nn.Linear(int(encoder_channels[-1]), int(num_disaster_classes)),
        )

    @property
    def encoder(self) -> nn.Module:
        return self.base_model.encoder

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.base_model.encoder(image)
        decoder_output = self.base_model.decoder(features)
        damage_logits = self.base_model.segmentation_head(decoder_output)
        pooled = F.adaptive_avg_pool2d(features[-1], output_size=1).flatten(1)
        disaster_logits = self.disaster_head(pooled)
        return {
            "damage_logits": damage_logits,
            "disaster_logits": disaster_logits,
        }


def unpack_model_output(output: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Normalize legacy segmentation tensors and metadata-aware dictionaries."""

    if isinstance(output, torch.Tensor):
        return output, None
    if isinstance(output, dict):
        damage_logits = output.get("damage_logits")
        disaster_logits = output.get("disaster_logits")
        if not isinstance(damage_logits, torch.Tensor):
            raise TypeError("model output dictionary must contain tensor damage_logits")
        if disaster_logits is not None and not isinstance(disaster_logits, torch.Tensor):
            raise TypeError("disaster_logits must be a tensor or None")
        return damage_logits, disaster_logits
    raise TypeError(f"unsupported model output type: {type(output).__name__}")


def diagnostic_shared_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Return the last shared encoder block used for task-gradient diagnostics."""

    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return []
    block = getattr(encoder, "layer4", encoder)
    return [parameter for parameter in block.parameters() if parameter.requires_grad]


def gradient_interaction(
    damage_loss: torch.Tensor,
    disaster_loss: torch.Tensor,
    parameters: Iterable[nn.Parameter],
    *,
    disaster_weight: float,
) -> dict[str, float]:
    """Measure task-gradient norms/cosine without modifying optimizer gradients."""

    params = list(parameters)
    if not params:
        return {
            "damage_grad_norm": 0.0,
            "disaster_grad_norm": 0.0,
            "weighted_disaster_grad_norm": 0.0,
            "task_grad_cosine": 0.0,
        }
    damage_grads = torch.autograd.grad(
        damage_loss, params, retain_graph=True, allow_unused=True
    )
    disaster_grads = torch.autograd.grad(
        disaster_loss, params, retain_graph=True, allow_unused=True
    )
    damage_parts: list[torch.Tensor] = []
    disaster_parts: list[torch.Tensor] = []
    for parameter, damage_grad, disaster_grad in zip(params, damage_grads, disaster_grads):
        damage_parts.append(
            torch.zeros_like(parameter, dtype=torch.float32).reshape(-1)
            if damage_grad is None
            else damage_grad.detach().float().reshape(-1)
        )
        disaster_parts.append(
            torch.zeros_like(parameter, dtype=torch.float32).reshape(-1)
            if disaster_grad is None
            else disaster_grad.detach().float().reshape(-1)
        )
    damage_flat = torch.cat(damage_parts)
    disaster_flat = torch.cat(disaster_parts)
    damage_norm = torch.linalg.vector_norm(damage_flat)
    disaster_norm = torch.linalg.vector_norm(disaster_flat)
    denominator = damage_norm * disaster_norm
    cosine = (
        torch.dot(damage_flat, disaster_flat) / denominator
        if float(denominator.item()) > 0.0
        else damage_norm.new_tensor(0.0)
    )
    return {
        "damage_grad_norm": float(damage_norm.item()),
        "disaster_grad_norm": float(disaster_norm.item()),
        "weighted_disaster_grad_norm": float(disaster_norm.item()) * float(disaster_weight),
        "task_grad_cosine": float(cosine.item()),
    }
