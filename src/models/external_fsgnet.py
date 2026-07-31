from __future__ import annotations

import hashlib
import importlib
import os
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


class FSGNetInputAdapter(nn.Module):
    """Adapt the external two-image FSG-Net to the unified six-channel input.

    The unified Stage-2 tensor is
    ``[pre_R, pre_G, pre_B, SAR, predicted_prior, SAR*predicted_prior]``.
    FSG-Net receives the first and last three channels as its two streams.
    The upstream loader scales images to ``[0, 1]`` without mean/std
    normalization, which is already the range produced by Stage-2.
    """

    def __init__(
        self,
        core: nn.Module,
        stream_a_channels: Sequence[int] = (0, 1, 2),
        stream_b_channels: Sequence[int] = (3, 4, 5),
        normalization: str = "upstream_zero_one",
        inference_tile_size: int = 512,
        inference_tile_overlap: int = 0,
    ) -> None:
        super().__init__()
        if len(stream_a_channels) != 3 or len(stream_b_channels) != 3:
            raise ValueError("FSG-Net requires exactly three channels in each stream")
        self.core = core
        self.stream_a_channels = tuple(int(index) for index in stream_a_channels)
        self.stream_b_channels = tuple(int(index) for index in stream_b_channels)
        self.normalization = str(normalization).lower()
        if self.normalization not in {"none", "upstream_zero_one"}:
            raise ValueError(f"unsupported FSG-Net normalization: {normalization}")
        self.inference_tile_size = int(inference_tile_size)
        self.inference_tile_overlap = int(inference_tile_overlap)
        if self.inference_tile_size <= 0:
            raise ValueError("FSG-Net inference_tile_size must be positive")
        if not 0 <= self.inference_tile_overlap < self.inference_tile_size:
            raise ValueError(
                "FSG-Net inference_tile_overlap must be non-negative and smaller than tile size"
            )

    @staticmethod
    def _select_stream(image: torch.Tensor, channels: tuple[int, ...]) -> torch.Tensor:
        return image[:, list(channels), :, :]

    @staticmethod
    def _tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
        if length <= tile_size:
            return [0]
        starts = list(range(0, length - tile_size + 1, stride))
        final = length - tile_size
        if starts[-1] != final:
            starts.append(final)
        return starts

    def _forward_tiled(
        self,
        stream_a: torch.Tensor,
        stream_b: torch.Tensor,
    ) -> torch.Tensor:
        height, width = stream_a.shape[-2:]
        tile_size = self.inference_tile_size
        stride = tile_size - self.inference_tile_overlap
        y_starts = self._tile_starts(height, tile_size, stride)
        x_starts = self._tile_starts(width, tile_size, stride)
        output: torch.Tensor | None = None
        counts: torch.Tensor | None = None
        for top in y_starts:
            bottom = min(top + tile_size, height)
            for left in x_starts:
                right = min(left + tile_size, width)
                logits = self.core(
                    stream_a[:, :, top:bottom, left:right],
                    stream_b[:, :, top:bottom, left:right],
                )
                if not isinstance(logits, torch.Tensor):
                    raise TypeError(
                        f"external FSG-Net returned {type(logits)!r}, expected Tensor"
                    )
                if logits.shape[-2:] != (bottom - top, right - left):
                    logits = F.interpolate(
                        logits,
                        size=(bottom - top, right - left),
                        mode="bilinear",
                        align_corners=True,
                    )
                if output is None:
                    output = logits.new_zeros(
                        (logits.shape[0], logits.shape[1], height, width)
                    )
                    counts = logits.new_zeros((1, 1, height, width))
                output[:, :, top:bottom, left:right] += logits
                counts[:, :, top:bottom, left:right] += 1
        if output is None or counts is None:
            raise RuntimeError("FSG-Net tiled inference produced no tiles")
        return output / counts.clamp_min(1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(f"expected [B,C,H,W] input, got {tuple(image.shape)}")
        required_channels = max((*self.stream_a_channels, *self.stream_b_channels)) + 1
        if image.shape[1] < required_channels:
            raise ValueError(
                f"FSG-Net adapter needs at least {required_channels} input channels, "
                f"got {image.shape[1]}"
            )
        stream_a = self._select_stream(image, self.stream_a_channels)
        stream_b = self._select_stream(image, self.stream_b_channels)
        use_tiled_inference = (
            not self.training
            and (
                image.shape[-2] > self.inference_tile_size
                or image.shape[-1] > self.inference_tile_size
            )
        )
        logits = (
            self._forward_tiled(stream_a, stream_b)
            if use_tiled_inference
            else self.core(stream_a, stream_b)
        )
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"external FSG-Net returned {type(logits)!r}, expected Tensor")
        if logits.shape[-2:] != image.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
        return logits

    def step_epoch(self) -> None:
        """Advance the upstream DropBlock schedule once after a train epoch."""

        neck = getattr(self.core, "neck", None)
        drop = getattr(neck, "drop", None)
        step = getattr(drop, "step", None)
        if callable(step):
            step()


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
        if (resolved / "models" / "main_model.py").is_file():
            return resolved
    rendered = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"external FSG-Net root was not found; checked: {rendered}")


def _resolve_artifact(external_root: Path, value: str | os.PathLike[str]) -> Path:
    raw = Path(value).expanduser()
    return raw.resolve() if raw.is_absolute() else (external_root / raw).resolve()


def _verify_sources(external_root: Path, specs: Mapping[str, object]) -> None:
    if not specs:
        raise ValueError("model.source_artifacts is required for external_fsgnet")
    for relative, expected_value in specs.items():
        path = (external_root / str(relative)).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"missing external FSG-Net source: {path}")
        expected = str(expected_value).strip().lower()
        if not expected:
            raise ValueError(f"empty FSG-Net checksum for {relative}")
        if _sha256(path) != expected:
            raise RuntimeError(f"external FSG-Net source checksum mismatch: {path}")


def _load_external_modules(external_root: Path) -> tuple[Any, Any]:
    """Load upstream modules through the existing top-level ``models`` package.

    Upstream mixes relative imports with absolute imports such as
    ``models.block.Base``.  Extending the local package search path lets those
    imports resolve without copying or editing the unlicensed author source.
    """

    package = importlib.import_module("models")
    package_path = getattr(package, "__path__", None)
    if package_path is None:
        raise ImportError("the local models package has no package search path")
    external_models = str((external_root / "models").resolve())
    if external_models not in list(package_path):
        package_path.append(external_models)
    importlib.invalidate_caches()

    main_module = importlib.import_module("models.main_model")
    wavelet_module = importlib.import_module("models.block.torch_wavelets")
    expected_main = (external_root / "models" / "main_model.py").resolve()
    actual_main = Path(str(main_module.__file__)).resolve()
    if actual_main != expected_main:
        raise ImportError(
            f"models.main_model already resolves to another checkout: {actual_main}"
        )
    return main_module, wavelet_module


def _load_legacy_resnet18(backbone: nn.Module, checkpoint_path: Path) -> None:
    # The fixed torchvision checkpoint is a legacy tar archive.  Its SHA-256 is
    # verified before this function is called, so the legacy loader is only
    # applied to the audited local artifact.
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    extractor = getattr(backbone, "extract", None)
    if extractor is None:
        raise TypeError("upstream ResNet wrapper has no extract module")
    incompatible = extractor.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(
            f"legacy ResNet18 checkpoint has missing keys: {incompatible.missing_keys[:5]}"
        )
    if incompatible.unexpected_keys != ["fc.weight", "fc.bias"]:
        raise RuntimeError(
            "legacy ResNet18 checkpoint has unexpected keys beyond the classifier: "
            f"{incompatible.unexpected_keys[:5]}"
        )


def _patch_wavelet_amp(core: nn.Module, wavelet_module: Any) -> int:
    """Keep fixed Haar filters in the tensor dtype used by AMP.

    The released custom autograd functions save float32 filters while autocast
    produces float16 gradients, causing a dtype error in backward.  Casting the
    constant filters at the call boundary preserves the same DWT/IDWT equations
    and makes the official implementation compatible with the unified AMP loop.
    """

    def compute_dtype(value: torch.Tensor) -> torch.dtype:
        device_type = value.device.type
        if device_type in {"cpu", "cuda"} and torch.is_autocast_enabled(device_type):
            return torch.get_autocast_dtype(device_type)
        return value.dtype

    def dwt_forward(instance: nn.Module, value: torch.Tensor) -> torch.Tensor:
        dtype = compute_dtype(value)
        return wavelet_module.DWT_Function.apply(
            value,
            instance.w_ll.to(dtype=dtype),
            instance.w_lh.to(dtype=dtype),
            instance.w_hl.to(dtype=dtype),
            instance.w_hh.to(dtype=dtype),
        )

    def idwt_forward(instance: nn.Module, value: torch.Tensor) -> torch.Tensor:
        dtype = compute_dtype(value)
        return wavelet_module.IDWT_Function.apply(
            value,
            instance.filters.to(dtype=dtype),
        )

    patched = 0
    for module in core.modules():
        if isinstance(module, wavelet_module.DWT_2D):
            module.forward = MethodType(dwt_forward, module)
            patched += 1
        elif isinstance(module, wavelet_module.IDWT_2D):
            module.forward = MethodType(idwt_forward, module)
            patched += 1
    if patched != 6:
        raise RuntimeError(f"expected to patch six FSG-Net wavelet modules, patched {patched}")
    return patched


def build_external_fsgnet(
    model_cfg: dict[str, Any],
    no_pretrained: bool = False,
) -> nn.Module:
    external_value = model_cfg.get("external_root")
    if not external_value:
        raise ValueError("model.external_root is required for external_fsgnet")
    external_root = _resolve_external_root(str(external_value))
    source_specs = model_cfg.get("source_artifacts")
    if not isinstance(source_specs, Mapping):
        raise ValueError("model.source_artifacts must be a path-to-SHA mapping")
    _verify_sources(external_root, source_specs)

    backbone_value = model_cfg.get("backbone_path")
    if not backbone_value:
        raise ValueError("model.backbone_path is required for external_fsgnet")
    backbone_path = _resolve_artifact(external_root, str(backbone_value))
    expected_backbone_sha = str(model_cfg.get("backbone_sha256", "")).strip().lower()
    if not backbone_path.is_file():
        raise FileNotFoundError(f"missing external FSG-Net backbone: {backbone_path}")
    actual_backbone_sha = _sha256(backbone_path)
    if not expected_backbone_sha or actual_backbone_sha != expected_backbone_sha:
        raise RuntimeError(f"external FSG-Net backbone checksum mismatch: {backbone_path}")

    main_module, wavelet_module = _load_external_modules(external_root)
    original_backbone_class = main_module.ResNet_timm
    use_pretrained = bool(model_cfg.get("pretrained_backbone", True)) and not no_pretrained
    expected_backbone_name = str(model_cfg.get("backbone", "resnet18"))

    def fixed_backbone(name: str = "resnet18", pretrained: bool = True) -> nn.Module:
        del pretrained
        if name != expected_backbone_name:
            raise ValueError(
                f"FSG-Net requested backbone {name!r}, expected {expected_backbone_name!r}"
            )
        backbone = original_backbone_class(name=name, pretrained=False)
        if use_pretrained:
            _load_legacy_resnet18(backbone, backbone_path)
        return backbone

    main_module.ResNet_timm = fixed_backbone
    try:
        options = SimpleNamespace(
            dual_label=False,
            backbone=expected_backbone_name,
            neck=str(model_cfg.get("neck", "fpn+aspp+fuse+drop")),
            head=str(model_cfg.get("head", "fcn")),
            input_size=int(model_cfg.get("input_size", 512)),
            pretrain="",
        )
        core = main_module.ChangeDetection(options)
    finally:
        main_module.ResNet_timm = original_backbone_class

    out_channels = int(model_cfg.get("out_channels", 3))
    core.head1 = main_module.FCNHead(core.inplanes, out_channels)
    core.head2 = None
    patched_wavelet_modules = 0
    if bool(model_cfg.get("amp_wavelet_compat", True)):
        patched_wavelet_modules = _patch_wavelet_amp(core, wavelet_module)

    adapter = FSGNetInputAdapter(
        core=core,
        stream_a_channels=model_cfg.get("stream_a_channels", [0, 1, 2]),
        stream_b_channels=model_cfg.get("stream_b_channels", [3, 4, 5]),
        normalization=str(model_cfg.get("normalization", "upstream_zero_one")),
        inference_tile_size=int(model_cfg.get("inference_tile_size", 512)),
        inference_tile_overlap=int(model_cfg.get("inference_tile_overlap", 0)),
    )
    adapter.external_root = str(external_root)
    adapter.external_source = str(external_root / "models" / "main_model.py")
    adapter.external_backbone = str(backbone_path)
    adapter.external_backbone_sha256 = actual_backbone_sha
    adapter.external_pretrained_backbone = use_pretrained
    adapter.external_wavelet_amp_patches = patched_wavelet_modules
    return adapter
