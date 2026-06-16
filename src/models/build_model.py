from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.enc4 = ConvBlock(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.center = ConvBlock(256, 512)
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = ConvBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32)
        self.out = nn.Conv2d(32, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        c = self.center(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(c), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


def _none_if_null(value: object) -> object:
    if value in {"null", "none", "None", ""}:
        return None
    return value


def build_model(cfg: dict, no_pretrained: bool = False) -> nn.Module:
    model_cfg = dict(cfg.get("model", {}))
    name = str(model_cfg.get("name", "unet")).lower()
    in_channels = int(model_cfg.get("in_channels", 3))
    out_channels = int(model_cfg.get("out_channels", 1))
    weights = None if no_pretrained else _none_if_null(model_cfg.get("encoder_weights", None))

    try:
        import segmentation_models_pytorch as smp
    except Exception:
        if name != "unet":
            raise
        return SimpleUNet(in_channels=in_channels, out_channels=out_channels)

    if name == "unet":
        return smp.Unet(
            encoder_name=str(model_cfg.get("encoder", "resnet34")),
            encoder_weights=weights,
            in_channels=in_channels,
            classes=out_channels,
        )
    if name in {"deeplabv3plus", "deeplabv3p"}:
        return smp.DeepLabV3Plus(
            encoder_name=str(model_cfg.get("encoder", "resnet50")),
            encoder_weights=weights,
            in_channels=in_channels,
            classes=out_channels,
        )
    if name == "segformer":
        encoder = str(model_cfg.get("backbone", model_cfg.get("encoder", "mit_b0")))
        try:
            return smp.Segformer(
                encoder_name=encoder,
                encoder_weights=weights,
                in_channels=in_channels,
                classes=out_channels,
            )
        except AttributeError as exc:
            raise RuntimeError(
                "This segmentation_models_pytorch version does not expose smp.Segformer. "
                "Upgrade segmentation-models-pytorch or skip the SegFormer run."
            ) from exc
    raise ValueError(f"unknown model name: {name}")
