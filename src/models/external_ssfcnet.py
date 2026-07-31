from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F


class SSFCNetInputAdapter(nn.Module):
    """Adapt the external two-image SSFCNet to the unified six-channel input.

    The unified Stage-2 tensor is
    ``[pre_R, pre_G, pre_B, SAR, predicted_prior, SAR*predicted_prior]``.
    SSFCNet receives the first and last three channels as its optical and
    post-event streams, respectively.
    """

    def __init__(
        self,
        core: nn.Module,
        stream_a_channels: Sequence[int] = (0, 1, 2),
        stream_b_channels: Sequence[int] = (3, 4, 5),
        normalization: str = "upstream_minus_one_one",
    ) -> None:
        super().__init__()
        if len(stream_a_channels) != 3 or len(stream_b_channels) != 3:
            raise ValueError("SSFCNet requires exactly three channels in each stream")
        self.core = core
        self.stream_a_channels = tuple(int(index) for index in stream_a_channels)
        self.stream_b_channels = tuple(int(index) for index in stream_b_channels)
        self.normalization = str(normalization).lower()
        if self.normalization not in {"none", "upstream_minus_one_one"}:
            raise ValueError(f"unsupported SSFCNet normalization: {normalization}")

    def _select_stream(self, image: torch.Tensor, channels: tuple[int, ...]) -> torch.Tensor:
        return image[:, list(channels), :, :]

    def _normalize(self, stream: torch.Tensor) -> torch.Tensor:
        if self.normalization == "upstream_minus_one_one":
            return stream.mul(2.0).sub(1.0)
        return stream

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(f"expected [B,C,H,W] input, got {tuple(image.shape)}")
        required_channels = max((*self.stream_a_channels, *self.stream_b_channels)) + 1
        if image.shape[1] < required_channels:
            raise ValueError(
                f"SSFCNet adapter needs at least {required_channels} input channels, "
                f"got {image.shape[1]}"
            )
        stream_a = self._normalize(self._select_stream(image, self.stream_a_channels))
        stream_b = self._normalize(self._select_stream(image, self.stream_b_channels))
        logits = self.core(stream_a, stream_b)
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"external SSFCNet returned {type(logits)!r}, expected Tensor")
        if logits.shape[-2:] != image.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return logits


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_external_root(value: str | os.PathLike[str]) -> Path:
    raw = Path(value).expanduser()
    candidates = (
        [raw]
        if raw.is_absolute()
        else [Path.cwd() / raw, Path(__file__).resolve().parents[2] / raw]
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "model" / "SSFCNet.py").is_file():
            return resolved
    rendered = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"external SSFCNet root was not found; checked: {rendered}")


def _load_external_module(external_root: Path) -> types.ModuleType:
    identity = hashlib.sha256(str(external_root).encode("utf-8")).hexdigest()[:12]
    module_name = f"_stage2_external_ssfcnet_{identity}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    source_path = external_root / "model" / "SSFCNet.py"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load external SSFCNet module from {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def build_external_ssfcnet(
    model_cfg: dict[str, Any],
    no_pretrained: bool = False,
) -> nn.Module:
    del no_pretrained  # The released historical model has no pretrained checkpoint path.
    external_value = model_cfg.get("external_root")
    if not external_value:
        raise ValueError("model.external_root is required for external_ssfcnet")
    external_root = _resolve_external_root(str(external_value))
    source_path = external_root / "model" / "SSFCNet.py"

    expected_source_sha = str(model_cfg.get("source_sha256", "")).strip().lower()
    actual_source_sha = _sha256(source_path)
    if expected_source_sha and actual_source_sha != expected_source_sha:
        raise RuntimeError(f"external SSFCNet source checksum mismatch: {source_path}")

    module = _load_external_module(external_root)
    model_class = getattr(module, "SSFCNet")
    core = model_class(
        n_classes=int(model_cfg.get("out_channels", 3)),
        shared_encoder=bool(model_cfg.get("shared_encoder", True)),
        use_iwt_decoder=bool(model_cfg.get("use_iwt_decoder", True)),
        use_diffcorr=bool(model_cfg.get("use_diffcorr", True)),
        norm=str(model_cfg.get("norm", "gn")),
        padding_mode=str(model_cfg.get("padding_mode", "reflect")),
        auto_pad=bool(model_cfg.get("auto_pad", True)),
    )
    adapter = SSFCNetInputAdapter(
        core=core,
        stream_a_channels=model_cfg.get("stream_a_channels", [0, 1, 2]),
        stream_b_channels=model_cfg.get("stream_b_channels", [3, 4, 5]),
        normalization=str(model_cfg.get("normalization", "upstream_minus_one_one")),
    )
    adapter.external_root = str(external_root)
    adapter.external_source = str(source_path)
    adapter.external_source_sha256 = actual_source_sha
    return adapter
