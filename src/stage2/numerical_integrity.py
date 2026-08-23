from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch


NUMERICAL_FAILURE_EXIT_CODE = 86


@dataclass
class NumericalIntegrityError(RuntimeError):
    report: dict[str, Any]

    def __str__(self) -> str:
        return f"numerical integrity failure: {self.report.get('first_nonfinite_tensor')}"


def strict_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def named_tensors(value: Any, prefix: str = "") -> Iterator[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        yield prefix or "tensor", value
    elif isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            yield from named_tensors(child, name)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            name = f"{prefix}.{index}" if prefix else str(index)
            yield from named_tensors(child, name)


def _tensor_summary(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach()
    summary: dict[str, Any] = {
        "name": name,
        "dtype": str(detached.dtype),
        "shape": list(detached.shape),
        "device": str(detached.device),
        "numel": detached.numel(),
    }
    if detached.numel() == 0 or not (detached.is_floating_point() or detached.is_complex()):
        summary["finite"] = True
        return summary
    finite = torch.isfinite(detached)
    summary["finite"] = bool(finite.all().item())
    summary["finite_count"] = int(finite.sum().item())
    summary["nonfinite_count"] = int(detached.numel() - summary["finite_count"])
    if summary["finite_count"]:
        finite_values = detached[finite]
        if finite_values.is_complex():
            finite_values = torch.abs(finite_values)
        summary["finite_min"] = float(finite_values.min().item())
        summary["finite_max"] = float(finite_values.max().item())
    return summary


def scan_tensors(items: Iterable[tuple[str, torch.Tensor]]) -> dict[str, Any]:
    scanned = 0
    elements = 0
    for name, tensor in items:
        scanned += 1
        elements += tensor.numel()
        if tensor.numel() and (tensor.is_floating_point() or tensor.is_complex()):
            if not bool(torch.isfinite(tensor.detach()).all().item()):
                return {
                    "passed": False,
                    "tensor_count_scanned": scanned,
                    "element_count_scanned": elements,
                    "first_nonfinite": _tensor_summary(name, tensor),
                }
    return {
        "passed": True,
        "tensor_count_scanned": scanned,
        "element_count_scanned": elements,
        "first_nonfinite": None,
    }


def finite_python_values(value: Any, prefix: str = "value") -> tuple[bool, str | None]:
    if isinstance(value, float):
        return (math.isfinite(value), None if math.isfinite(value) else prefix)
    if isinstance(value, dict):
        for key, child in value.items():
            passed, name = finite_python_values(child, f"{prefix}.{key}")
            if not passed:
                return passed, name
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            passed, name = finite_python_values(child, f"{prefix}.{index}")
            if not passed:
                return passed, name
    return True, None


def fail_numerical(
    output_dir: Path,
    *,
    stage: str,
    first_nonfinite_tensor: str,
    step: int | None = None,
    batch_ids: list[str] | None = None,
    tensor_summary: dict[str, Any] | None = None,
    amp_scale: float | None = None,
    component_areas: dict[str, Any] | None = None,
) -> None:
    report = {
        "status": "failed_numerical",
        "exit_code": NUMERICAL_FAILURE_EXIT_CODE,
        "stage": stage,
        "step": step,
        "batch_ids": batch_ids or [],
        "first_nonfinite_tensor": first_nonfinite_tensor,
        "tensor": tensor_summary,
        "amp_scale": amp_scale,
        "component_areas": component_areas,
    }
    strict_write_json(output_dir / "numerical_failure.json", report)
    raise NumericalIntegrityError(report)


def guard_tensors(
    output_dir: Path,
    *,
    stage: str,
    items: Iterable[tuple[str, torch.Tensor]],
    step: int | None = None,
    batch_ids: list[str] | None = None,
    amp_scale: float | None = None,
    component_areas: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = scan_tensors(items)
    if not result["passed"]:
        summary = result["first_nonfinite"]
        fail_numerical(
            output_dir,
            stage=stage,
            first_nonfinite_tensor=str(summary["name"]),
            step=step,
            batch_ids=batch_ids,
            tensor_summary=summary,
            amp_scale=amp_scale,
            component_areas=component_areas,
        )
    return result


def component_area_summary(component_map: torch.Tensor | None) -> dict[str, Any] | None:
    if component_map is None:
        return None
    positive = component_map.detach().long().reshape(-1)
    positive = positive[positive > 0]
    if not positive.numel():
        return {"component_count": 0, "min": None, "max": None}
    counts = torch.bincount(positive)
    counts = counts[counts > 0]
    return {
        "component_count": int(counts.numel()),
        "min": int(counts.min().item()),
        "max": int(counts.max().item()),
    }


def append_integrity_snapshot(path: Path, snapshot: dict[str, Any]) -> None:
    existing: list[Any] = []
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            existing = loaded
    existing.append(snapshot)
    strict_write_json(path, existing)


def checkpoint_integrity_sidecar(path: Path, scan: dict[str, Any]) -> Path:
    sidecar = path.with_suffix(path.suffix + ".integrity.json")
    strict_write_json(
        sidecar,
        {
            "status": "passed",
            "checkpoint": path.name,
            "checkpoint_sha256": file_sha256(path),
            "scan": scan,
        },
    )
    return sidecar


def verify_checkpoint_sidecar(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".integrity.json")
    if not sidecar.is_file():
        raise ValueError(f"missing checkpoint integrity sidecar: {sidecar}")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    if payload.get("status") != "passed" or payload.get("checkpoint_sha256") != file_sha256(path):
        raise ValueError(f"checkpoint integrity sidecar mismatch: {sidecar}")
    return payload
