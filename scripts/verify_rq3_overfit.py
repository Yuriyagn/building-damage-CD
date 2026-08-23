#!/usr/bin/env python3
"""Verify that an RQ3 config reduces loss by at least 50% on two frozen batches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.build_model import build_model  # noqa: E402
from stage2.rq3_runtime import forward_stage2_model, stage2_damage_loss  # noqa: E402
from stage2.train_stage2_v2 import build_stage2_v2_loss, compute_class_weights, make_dataset, set_seed  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite overfit verdict: {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(cfg.get("train", {}).get("seed", 42))
    set_seed(seed)
    device = torch.device("cuda")
    dataset = make_dataset(cfg, args.data_root, "train", True, seed, args.batch_size * 2, True)
    batches = list(DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0))
    weights, _ = compute_class_weights(dataset, cfg)
    criterion = build_stage2_v2_loss(cfg, weights).to(device)
    model = build_model(cfg, no_pretrained=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    def fail(stage: str, step: int, tensor: str) -> int:
        result = {
            "schema": "rq3_two_batch_overfit_v1",
            "status": "failed_numerical",
            "stage": stage,
            "step": step,
            "first_nonfinite_tensor": tensor,
            "amp_scale": float(scaler.get_scale()),
            "steps_requested": args.steps,
            "batch_count": len(batches),
            "required_reduction": 0.5,
            "passed": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, sort_keys=True), file=sys.stderr)
        return 86

    def losses(training: bool) -> list[torch.Tensor]:
        model.train(training)
        values = []
        for batch in batches:
            with torch.autocast("cuda", dtype=torch.float16):
                output = forward_stage2_model(model, batch["image"].to(device), batch, device)
                values.append(
                    stage2_damage_loss(criterion, output, batch["mask"].to(device))
                )
        return values

    with torch.no_grad():
        initial_loss = float(torch.stack(losses(False)).mean())
    model.train()
    for step in range(args.steps):
        batch = batches[step % len(batches)]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            output = forward_stage2_model(model, batch["image"].to(device), batch, device)
            loss = stage2_damage_loss(criterion, output, batch["mask"].to(device))
        if not torch.isfinite(loss):
            return fail("loss", step + 1, "loss.total")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                return fail("gradient_unscaled", step + 1, f"gradient.{name}")
        scaler.step(optimizer)
        scaler.update()
    with torch.no_grad():
        final_loss = float(torch.stack(losses(False)).mean())
    reduction = (initial_loss - final_loss) / max(initial_loss, 1e-12)
    result = {
        "schema": "rq3_two_batch_overfit_v1", "steps": args.steps,
        "batch_count": len(batches), "initial_loss": initial_loss,
        "final_loss": final_loss, "relative_reduction": reduction,
        "required_reduction": 0.5, "passed": reduction >= 0.5,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
