#!/usr/bin/env python3
"""Measure fixed-shape CUDA training throughput for F0 or an RQ3 factor stack."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.build_model import build_model  # noqa: E402
from models.metadata_multitask import unpack_model_output  # noqa: E402
from stage2.losses_v2 import BuildingOnlyGradeLoss  # noqa: E402
from stage2.rq3_runtime import stage2_damage_loss  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("F0", "RQ3"), required=True)
    parser.add_argument("--factors", nargs="*", default=[])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite benchmark: {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for throughput benchmarking")
    torch.manual_seed(20260820)
    model_cfg = {
        "name": "unet" if args.model == "F0" else "rq3_damage_evidence",
        "encoder": "resnet34", "encoder_weights": None,
        "in_channels": 5, "out_channels": 3,
    }
    if args.model == "RQ3":
        model_cfg["factors"] = args.factors
    model = build_model({"model": model_cfg}, no_pretrained=True).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    image = torch.rand(args.batch_size, 5, args.size, args.size, device="cuda")
    target = torch.randint(0, 3, (args.batch_size, args.size, args.size), device="cuda")
    component_map = torch.zeros(args.batch_size, args.size, args.size, dtype=torch.long, device="cuda")
    component_id = 1
    stride = max(args.size // 4, 1)
    for y in range(0, args.size, stride):
        for x in range(0, args.size, stride):
            component_map[:, y:y + stride, x:x + stride] = component_id
            component_id += 1
    valid = component_map > 0
    criterion = BuildingOnlyGradeLoss().cuda()

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        if bool(getattr(model, "requires_rq3_batch", False)):
            output = model(image, component_map=component_map, instance_valid_mask=valid)
        else:
            output = model(image)
        logits, _ = unpack_model_output(output)
        loss_target = target + 1
        loss = stage2_damage_loss(criterion, output, loss_target)
        loss.backward()
        optimizer.step()

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.steps):
        step()
    end.record()
    torch.cuda.synchronize()
    elapsed_seconds = float(start.elapsed_time(end) / 1000.0)
    result = {
        "schema": "rq3_cuda_throughput_v1", "model": args.model,
        "factors": args.factors, "steps": args.steps, "warmup": args.warmup,
        "batch_size": args.batch_size, "image_size": args.size,
        "elapsed_seconds": elapsed_seconds, "steps_per_second": args.steps / elapsed_seconds,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "device": torch.cuda.get_device_name(0),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
