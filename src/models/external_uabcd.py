from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import types
from contextlib import contextmanager
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from torch import nn
from torch.nn import functional as F


class UABCDInputAdapter(nn.Module):
    """Adapt the external two-stream UABCD model to a unified CHW tensor.

    The unified Stage-2 input is
    ``[pre_R, pre_G, pre_B, SAR, predicted_prior, SAR*predicted_prior]``.
    UABCD receives the first and last three channels as its two streams.
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
            raise ValueError("UABCD requires exactly three channels in each stream")
        self.core = core
        self.stream_a_channels = tuple(int(index) for index in stream_a_channels)
        self.stream_b_channels = tuple(int(index) for index in stream_b_channels)
        self.normalization = str(normalization).lower()
        if self.normalization not in {"none", "upstream_minus_one_one"}:
            raise ValueError(f"unsupported UABCD normalization: {normalization}")

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
                f"UABCD adapter needs at least {required_channels} input channels, "
                f"got {image.shape[1]}"
            )
        stream_a = self._normalize(self._select_stream(image, self.stream_a_channels))
        stream_b = self._normalize(self._select_stream(image, self.stream_b_channels))
        logits = self.core(stream_a, stream_b)
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"external UABCD returned {type(logits)!r}, expected Tensor")
        if logits.shape[-2:] != image.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=True,
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
    candidates = [raw] if raw.is_absolute() else [Path.cwd() / raw, Path(__file__).resolve().parents[2] / raw]
    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "network" / "UABCD.py").is_file():
            return resolved
    rendered = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"external UABCD root was not found; checked: {rendered}")


def _namespace_package(name: str, path: Path) -> types.ModuleType:
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    module = types.ModuleType(name)
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    module.__package__ = name
    module.__spec__ = ModuleSpec(name=name, loader=None, is_package=True)
    module.__spec__.submodule_search_locations = [str(path)]
    sys.modules[name] = module
    return module


def _load_external_module(external_root: Path) -> types.ModuleType:
    identity = hashlib.sha256(str(external_root).encode("utf-8")).hexdigest()[:12]
    package_name = f"_stage2_external_uabcd_{identity}"
    network_package = f"{package_name}.network"
    module_name = f"{network_package}.UABCD"
    if module_name in sys.modules:
        return sys.modules[module_name]

    _namespace_package(package_name, external_root)
    _namespace_package(network_package, external_root / "network")
    source_path = external_root / "network" / "UABCD.py"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load external UABCD module from {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _instantiate_core(
    module: types.ModuleType,
    external_root: Path,
    out_channels: int,
    no_pretrained: bool,
) -> nn.Module:
    model_class = getattr(module, "UABCD")
    if not no_pretrained:
        with _working_directory(external_root):
            return model_class(num_classes=out_channels)

    # The upstream constructor always calls init_backbone_weights.  Preserve the
    # architecture while bypassing only its local torch.load for checkpoint eval
    # and explicit --no-pretrained diagnostics.
    original_initializer = model_class.init_backbone_weights

    def initialize_without_weights(instance: nn.Module) -> None:
        instance.backbone = module.pvt_v2_b2()

    model_class.init_backbone_weights = initialize_without_weights
    try:
        return model_class(num_classes=out_channels)
    finally:
        model_class.init_backbone_weights = original_initializer


def build_external_uabcd(model_cfg: dict[str, Any], no_pretrained: bool = False) -> nn.Module:
    external_value = model_cfg.get("external_root")
    if not external_value:
        raise ValueError("model.external_root is required for external_uabcd")
    external_root = _resolve_external_root(str(external_value))
    source_path = external_root / "network" / "UABCD.py"
    backbone_path = external_root / "pretrain_backbones" / "new_pvt_v2_b2.pth"

    expected_source_sha = str(model_cfg.get("source_sha256", "")).strip().lower()
    if expected_source_sha and _sha256(source_path) != expected_source_sha:
        raise RuntimeError(f"external UABCD source checksum mismatch: {source_path}")

    expected_backbone_sha = str(model_cfg.get("backbone_sha256", "")).strip().lower()
    if not no_pretrained:
        if not backbone_path.is_file():
            raise FileNotFoundError(f"missing external UABCD backbone: {backbone_path}")
        if expected_backbone_sha and _sha256(backbone_path) != expected_backbone_sha:
            raise RuntimeError(f"external UABCD backbone checksum mismatch: {backbone_path}")

    module = _load_external_module(external_root)
    core = _instantiate_core(
        module=module,
        external_root=external_root,
        out_channels=int(model_cfg.get("out_channels", 3)),
        no_pretrained=no_pretrained,
    )
    adapter = UABCDInputAdapter(
        core=core,
        stream_a_channels=model_cfg.get("stream_a_channels", [0, 1, 2]),
        stream_b_channels=model_cfg.get("stream_b_channels", [3, 4, 5]),
        normalization=str(model_cfg.get("normalization", "upstream_minus_one_one")),
    )
    adapter.external_root = str(external_root)
    adapter.external_source = str(source_path)
    adapter.external_backbone = str(backbone_path)
    return adapter
