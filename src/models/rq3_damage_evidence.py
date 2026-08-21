from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def haar_dwt2(image: torch.Tensor) -> torch.Tensor:
    """Deterministic one-level orthonormal Haar decomposition: LL/LH/HL/HH."""

    if image.ndim != 4 or image.shape[1] != 1:
        raise ValueError(f"Haar input must be [B,1,H,W], got {tuple(image.shape)}")
    if image.shape[-2] % 2 or image.shape[-1] % 2:
        image = F.pad(image, (0, image.shape[-1] % 2, 0, image.shape[-2] % 2), mode="replicate")
    a = image[:, :, 0::2, 0::2]
    b = image[:, :, 0::2, 1::2]
    c = image[:, :, 1::2, 0::2]
    d = image[:, :, 1::2, 1::2]
    return torch.cat(
        [
            (a + b + c + d) * 0.5,
            (a - b + c - d) * 0.5,
            (a + b - c - d) * 0.5,
            (a - b - c + d) * 0.5,
        ],
        dim=1,
    )


class HaarFeatureResidual(nn.Module):
    def __init__(self, encoder_channels: tuple[int, ...]) -> None:
        super().__init__()
        if len(encoder_channels) < 3:
            raise ValueError("wavelet residual requires at least two encoder scales")
        self.projections = nn.ModuleList(
            [
                nn.Conv2d(4, int(encoder_channels[1]), kernel_size=1, bias=False),
                nn.Conv2d(4, int(encoder_channels[2]), kernel_size=1, bias=False),
            ]
        )
        for projection in self.projections:
            nn.init.zeros_(projection.weight)

    def forward(self, features: list[torch.Tensor], sar: torch.Tensor) -> list[torch.Tensor]:
        level1 = haar_dwt2(sar)
        level2 = haar_dwt2(level1[:, 0:1])
        updated = list(features)
        for feature_index, (bands, projection) in enumerate(
            zip((level1, level2), self.projections, strict=True), start=1
        ):
            residual = projection(bands)
            if residual.shape[-2:] != updated[feature_index].shape[-2:]:
                residual = F.interpolate(
                    residual,
                    size=updated[feature_index].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            updated[feature_index] = updated[feature_index] + residual
        return updated


@dataclass
class PooledInstances:
    pooled: torch.Tensor
    pixel_index: torch.Tensor
    batch_index: torch.Tensor


def pool_component_features(features: torch.Tensor, component_map: torch.Tensor) -> PooledInstances:
    """Vectorized masked mean+max pooling for per-image contiguous component IDs."""

    if features.ndim != 4 or component_map.ndim != 3:
        raise ValueError("features must be [B,C,H,W] and component_map [B,H,W]")
    if features.shape[0] != component_map.shape[0]:
        raise ValueError("component batch size mismatch")
    if features.shape[-2:] != component_map.shape[-2:]:
        component_map = F.interpolate(
            component_map[:, None].float(), size=features.shape[-2:], mode="nearest"
        )[:, 0].long()
    else:
        component_map = component_map.long()

    global_map = torch.zeros_like(component_map)
    batch_indices: list[torch.Tensor] = []
    offset = 0
    for batch_index in range(component_map.shape[0]):
        local = component_map[batch_index]
        local_ids = torch.unique(local[local > 0], sorted=True)
        count = int(local_ids.numel())
        if count:
            for compact_id, local_id in enumerate(local_ids, start=1):
                global_map[batch_index][local == local_id] = offset + compact_id
            batch_indices.append(
                torch.full((count,), batch_index, device=features.device, dtype=torch.long)
            )
            offset += count
    if offset == 0:
        empty = features.new_zeros((0, features.shape[1] * 2))
        return PooledInstances(
            pooled=empty,
            pixel_index=global_map,
            batch_index=torch.empty(0, device=features.device, dtype=torch.long),
        )

    labels = global_map.reshape(-1)
    values = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
    keep = labels > 0
    labels_kept = labels[keep]
    values_kept = values[keep]
    sums = features.new_zeros((offset + 1, features.shape[1]))
    sums.index_add_(0, labels_kept, values_kept)
    counts = features.new_zeros((offset + 1, 1))
    counts.index_add_(0, labels_kept, features.new_ones((labels_kept.numel(), 1)))
    maxima = features.new_zeros((offset + 1, features.shape[1]))
    maxima.scatter_reduce_(
        0,
        labels_kept[:, None].expand(-1, features.shape[1]),
        values_kept,
        reduce="amax",
        include_self=False,
    )
    pooled = torch.cat([sums[1:] / counts[1:].clamp_min(1.0), maxima[1:]], dim=1)
    return PooledInstances(
        pooled=pooled,
        pixel_index=global_map,
        batch_index=torch.cat(batch_indices),
    )


def rasterize_instance_logits(
    instance_logits: torch.Tensor,
    pixel_index: torch.Tensor,
) -> torch.Tensor:
    if instance_logits.ndim != 2 or instance_logits.shape[1] != 3:
        raise ValueError("instance logits must be [N,3]")
    table = torch.cat([instance_logits.new_zeros((1, 3)), instance_logits], dim=0)
    dense = table[pixel_index.long()]
    return dense.permute(0, 3, 1, 2).contiguous()


def fuse_control_sar(
    control_logits: torch.Tensor,
    sar_logits: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Residual fusion whose gate-zero endpoint is exactly the control path."""

    return control_logits + gate * (sar_logits - control_logits)


class RQ3DamageEvidenceModel(nn.Module):
    """F0-compatible U-Net with independently switchable RQ3 evidence factors."""

    def __init__(
        self,
        *,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        in_channels: int = 5,
        instance_enabled: bool = False,
        wavelet_enabled: bool = False,
        reliability_enabled: bool = False,
    ) -> None:
        super().__init__()
        if in_channels != 5:
            raise ValueError("RQ3 freezes the input layout to five channels")
        import segmentation_models_pytorch as smp

        self.base_model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=3,
        )
        self.instance_enabled = bool(instance_enabled)
        self.wavelet_enabled = bool(wavelet_enabled)
        self.reliability_enabled = bool(reliability_enabled)
        self.requires_rq3_batch = self.instance_enabled or self.reliability_enabled
        encoder_channels = tuple(int(value) for value in self.base_model.encoder.out_channels)
        self.wavelet_residual = HaarFeatureResidual(encoder_channels) if wavelet_enabled else None
        decoder_channels = int(self.base_model.segmentation_head[0].in_channels)
        if self.instance_enabled:
            self.instance_head = nn.Linear(2 * decoder_channels, 3)
        else:
            self.instance_head = None
        if self.reliability_enabled:
            self.reliability_head = nn.Linear(2 * decoder_channels, 1)
        else:
            self.reliability_head = None

    @property
    def encoder(self) -> nn.Module:
        return self.base_model.encoder

    def _decode(self, image: torch.Tensor) -> torch.Tensor:
        features = list(self.base_model.encoder(image))
        if self.wavelet_residual is not None:
            features = self.wavelet_residual(features, image[:, 4:5])
        return self.base_model.decoder(features)

    def _dense_logits(self, decoder: torch.Tensor) -> torch.Tensor:
        return self.base_model.segmentation_head(decoder)

    def forward(
        self,
        image: torch.Tensor,
        *,
        component_map: torch.Tensor | None = None,
        instance_valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        decoder = self._decode(image)
        if not self.requires_rq3_batch:
            return {"damage_logits": self._dense_logits(decoder)}
        if component_map is None or instance_valid_mask is None:
            raise ValueError("instance/reliability factors require component maps")
        pooled = pool_component_features(decoder, component_map)
        dense_spatial = self._dense_logits(decoder)
        instance_logits: torch.Tensor | None = None
        if self.instance_head is not None:
            instance_logits = self.instance_head(pooled.pooled)
            dense_spatial = rasterize_instance_logits(instance_logits, pooled.pixel_index)

        output: dict[str, torch.Tensor] = {
            "damage_logits": dense_spatial,
            "instance_batch_index": pooled.batch_index,
            "instance_pixel_index": pooled.pixel_index,
            "rq3_loss_valid_mask": (
                instance_valid_mask.bool()
                if self.instance_enabled
                else torch.ones_like(instance_valid_mask, dtype=torch.bool)
            ),
        }
        if instance_logits is not None:
            output["instance_logits"] = instance_logits

        if self.reliability_head is None:
            return output

        control_image = image.clone()
        control_image[:, 4:5] = 0.0
        control_decoder = self._decode(control_image)
        delta_pooled = pool_component_features(torch.abs(decoder - control_decoder), component_map)
        reliability_logits = self.reliability_head(delta_pooled.pooled).reshape(-1)
        reliability = torch.sigmoid(reliability_logits)
        gate_table = torch.cat([reliability.new_zeros(1), reliability], dim=0)
        gate_map = gate_table[delta_pooled.pixel_index.long()][:, None]

        if self.instance_head is not None:
            control_instances = self.instance_head(pool_component_features(control_decoder, component_map).pooled)
            spatial_instances = instance_logits
            assert spatial_instances is not None
            fused_instances = fuse_control_sar(
                control_instances, spatial_instances, reliability[:, None]
            )
            output["damage_logits"] = rasterize_instance_logits(
                fused_instances, delta_pooled.pixel_index
            )
            output["instance_logits"] = fused_instances
        else:
            control_logits = self._dense_logits(control_decoder)
            output["damage_logits"] = fuse_control_sar(control_logits, dense_spatial, gate_map)
        output["reliability_logits"] = reliability_logits
        output["reliability"] = reliability
        output["instance_batch_index"] = delta_pooled.batch_index
        output["instance_pixel_index"] = delta_pooled.pixel_index
        return output


def build_rq3_damage_evidence(model_cfg: dict, *, no_pretrained: bool) -> RQ3DamageEvidenceModel:
    weights = None if no_pretrained else model_cfg.get("encoder_weights", "imagenet")
    if weights in {"none", "null", "None", ""}:
        weights = None
    factors = {str(value).lower() for value in model_cfg.get("factors", [])}
    unknown = factors - {"instance", "wavelet", "sensor", "reliability"}
    if unknown:
        raise ValueError(f"unknown RQ3 factors: {sorted(unknown)}")
    return RQ3DamageEvidenceModel(
        encoder_name=str(model_cfg.get("encoder", "resnet34")),
        encoder_weights=weights,
        in_channels=int(model_cfg.get("in_channels", 5)),
        instance_enabled="instance" in factors,
        wavelet_enabled="wavelet" in factors,
        reliability_enabled="reliability" in factors,
    )
