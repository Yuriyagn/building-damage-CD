from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def hierarchical_grade_log_probs(
    affected_logit: torch.Tensor,
    destroyed_given_affected_logit: torch.Tensor,
) -> torch.Tensor:
    """Return normalized intact/damaged/destroyed log-probabilities."""

    if affected_logit.shape != destroyed_given_affected_logit.shape:
        raise ValueError("hierarchical logits must have identical shapes")
    if affected_logit.ndim != 4 or affected_logit.shape[1] != 1:
        raise ValueError(
            "hierarchical logits must have shape [B,1,H,W], got "
            f"{tuple(affected_logit.shape)}"
        )
    log_affected = F.logsigmoid(affected_logit)
    log_intact = F.logsigmoid(-affected_logit)
    log_destroyed_given_affected = F.logsigmoid(destroyed_given_affected_logit)
    log_damaged_given_affected = F.logsigmoid(-destroyed_given_affected_logit)
    return torch.cat(
        [
            log_intact,
            log_affected + log_damaged_given_affected,
            log_affected + log_destroyed_given_affected,
        ],
        dim=1,
    )


class HierarchicalGradeUNet(nn.Module):
    """SMP U-Net with an affected -> damaged/destroyed factorized head."""

    def __init__(
        self,
        *,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        in_channels: int = 5,
    ) -> None:
        super().__init__()
        import segmentation_models_pytorch as smp

        self.base_model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=2,
        )

    @property
    def encoder(self) -> nn.Module:
        return self.base_model.encoder

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.base_model(image)
        affected_logit = raw[:, 0:1]
        destroyed_given_affected_logit = raw[:, 1:2]
        return {
            "damage_logits": hierarchical_grade_log_probs(
                affected_logit, destroyed_given_affected_logit
            ),
            "affected_logit": affected_logit,
            "destroyed_given_affected_logit": destroyed_given_affected_logit,
        }


class FusionProjection(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(3 * channels + 1, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        optical: torch.Tensor,
        sar: torch.Tensor,
        prior: torch.Tensor,
    ) -> torch.Tensor:
        if optical.shape != sar.shape:
            raise ValueError(
                "dual-stream feature shapes differ: "
                f"optical={tuple(optical.shape)}, sar={tuple(sar.shape)}"
            )
        resized_prior = F.interpolate(
            prior,
            size=optical.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.projection(
            torch.cat([optical, sar, torch.abs(optical - sar), resized_prior], dim=1)
        )


class DualStreamUNet(nn.Module):
    """Parameter-matched RGB/SAR encoders with fixed multi-scale fusion.

    The five input channels must be [pre_R, pre_G, pre_B, prior, post_SAR].
    The raw five-channel tensor is retained as the highest-resolution decoder
    skip, while encoder features are fused at the five ResNet feature scales.
    """

    def __init__(
        self,
        *,
        encoder_name: str = "resnet18",
        encoder_weights: str | None = "imagenet",
        in_channels: int = 5,
        out_channels: int = 3,
        decoder_channels: Sequence[int] = (256, 128, 64, 32, 16),
    ) -> None:
        super().__init__()
        if in_channels != 5:
            raise ValueError("DualStreamUNet requires the frozen five-channel RQ2 layout")
        if encoder_name != "resnet18":
            raise ValueError("RQ2-A freezes both modality encoders to resnet18")

        import segmentation_models_pytorch as smp
        from segmentation_models_pytorch.base import SegmentationHead
        from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder

        self.optical_encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=3,
            depth=5,
            weights=encoder_weights,
        )
        self.sar_encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=1,
            depth=5,
            weights=None,
        )
        optical_state = self.optical_encoder.state_dict()
        sar_state = self.sar_encoder.state_dict()
        adapted_state: dict[str, torch.Tensor] = {}
        for key, value in sar_state.items():
            if key == "conv1.weight":
                adapted_state[key] = optical_state[key].mean(dim=1, keepdim=True)
            else:
                source = optical_state.get(key)
                if source is None or source.shape != value.shape:
                    raise ValueError(f"cannot initialize SAR encoder parameter {key}")
                adapted_state[key] = source
        self.sar_encoder.load_state_dict(adapted_state, strict=True)

        optical_channels = tuple(int(value) for value in self.optical_encoder.out_channels)
        sar_channels = tuple(int(value) for value in self.sar_encoder.out_channels)
        if optical_channels[1:] != sar_channels[1:]:
            raise ValueError(
                "dual-stream encoder channels differ: "
                f"optical={optical_channels}, sar={sar_channels}"
            )
        fused_channels = (5, *optical_channels[1:])
        self.fusion_blocks = nn.ModuleList(
            FusionProjection(channels) for channels in optical_channels[1:]
        )
        self.decoder = UnetDecoder(
            encoder_channels=fused_channels,
            decoder_channels=tuple(int(value) for value in decoder_channels),
            n_blocks=5,
            use_norm="batchnorm",
            attention_type=None,
            add_center_block=False,
            interpolation_mode="nearest",
        )
        self.segmentation_head = SegmentationHead(
            in_channels=int(decoder_channels[-1]),
            out_channels=out_channels,
            kernel_size=3,
            activation=None,
        )

    @property
    def encoder(self) -> nn.Module:
        """Expose the optical branch for existing diagnostic hooks."""

        return self.optical_encoder

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        if image.ndim != 4 or image.shape[1] != 5:
            raise ValueError(f"expected [B,5,H,W] input, got {tuple(image.shape)}")
        optical = image[:, 0:3]
        prior = image[:, 3:4]
        sar = image[:, 4:5]
        optical_features = self.optical_encoder(optical)
        sar_features = self.sar_encoder(sar)
        fused_features = [image]
        for block, optical_feature, sar_feature in zip(
            self.fusion_blocks,
            optical_features[1:],
            sar_features[1:],
            strict=True,
        ):
            fused_features.append(block(optical_feature, sar_feature, prior))
        decoder_output = self.decoder(fused_features)
        return {"damage_logits": self.segmentation_head(decoder_output)}


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
