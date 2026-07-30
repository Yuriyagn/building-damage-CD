#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import load_config, train_params  # noqa: E402
from models.build_model import build_model  # noqa: E402
from stage2.common import write_csv, write_json  # noqa: E402
from stage2.datasets_v2 import Stage2V2Dataset  # noqa: E402


def code_state() -> dict[str, Any]:
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=SRC_ROOT.parent, check=False, capture_output=True, text=True
        )
        return result.stdout.strip()

    diff = git("diff", "--binary")
    source_hash = hashlib.sha256()
    source_files = []
    repo_root = SRC_ROOT.parent
    for pattern in ("src/**/*.py", "scripts/**/*.py", "tests/**/*.py", "configs/**/*.yaml", "*.sh"):
        for path in sorted(repo_root.glob(pattern)):
            if path.is_file():
                relative = path.relative_to(repo_root).as_posix()
                source_hash.update(relative.encode("utf-8") + b"\0")
                source_hash.update(path.read_bytes())
                source_files.append(relative)
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_status_short": git("status", "--short").splitlines(),
        "tracked_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        "tracked_diff_bytes": len(diff.encode("utf-8")),
        "source_tree_sha256": source_hash.hexdigest(),
        "source_file_count": len(source_files),
    }


def audit_state() -> dict[str, Any]:
    path = SRC_ROOT.parent / "outputs" / "stage2" / "v2_preflight" / "data_audit.json"
    if not path.exists():
        return {"path": str(path), "exists": False}
    content = path.read_bytes()
    payload = json.loads(content)
    return {
        "path": str(path.resolve()),
        "exists": True,
        "sha256": hashlib.sha256(content).hexdigest(),
        "status": payload.get("status"),
        "hard_error_count": payload.get("hard_error_count"),
        "warning_count": payload.get("warning_count"),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_dataset(
    cfg: dict[str, Any],
    data_root: Path,
    split: str,
    train: bool,
    seed: int,
    limit: int | None = None,
) -> Stage2V2Dataset:
    dcfg = dict(cfg.get("dataset", {}))
    acfg = dict(cfg.get("augmentation", {}))
    crop_probs = acfg.get("crop_probabilities") or {
        "damaged": 0.4,
        "destroyed": 0.4,
        "building": 0.2,
        "random": 0.0,
    }
    split_offset = {"train": 0, "val": 10_000, "test": 20_000}[split]
    permutation_seed = int(dcfg.get("sar_shuffle_seed", seed))
    return Stage2V2Dataset(
        data_root=data_root,
        manifest=str(dcfg[f"{split}_manifest"]),
        train=train,
        prior_type=str(dcfg.get("prior_type", "predicted")),
        input_mode=str(dcfg.get("input_mode", "pre_sar_texture_prior")),
        crop_size=int(dcfg.get("crop_size", 512)) if train else None,
        crop_probabilities={str(k): float(v) for k, v in dict(crop_probs).items()},
        hflip=bool(acfg.get("hflip", True)),
        vflip=bool(acfg.get("vflip", False)),
        rotate90=bool(acfg.get("rotate90", True)),
        sar_shuffle_mode=str(dcfg.get("sar_shuffle_mode", "paired")),
        sar_shuffle_seed=permutation_seed + split_offset,
        sar_singleton_policy=str(dcfg.get("sar_singleton_policy", "error")),
        limit=limit,
    )


def make_predictor_input(batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    pre = batch["pre_image"].to(device, non_blocking=True)
    prior = batch["prior"].to(device, non_blocking=True)
    return torch.cat([pre, prior], dim=1)


def make_predictor_target(batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    sar = batch["sar"].to(device, non_blocking=True)
    sar_grad = batch["sar_grad"].to(device, non_blocking=True)
    return torch.cat([sar, sar_grad], dim=1)


def forward_padded(model: torch.nn.Module, image: torch.Tensor, divisor: int = 16) -> torch.Tensor:
    h, w = image.shape[-2:]
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor
    if pad_h or pad_w:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
    logits = model(image)
    return logits[..., :h, :w]


def intact_weights(mask: torch.Tensor, boundary_weight: float) -> torch.Tensor:
    intact = (mask == 1).float().unsqueeze(1)
    building = (mask > 0).float().unsqueeze(1)
    eroded = 1.0 - F.max_pool2d(1.0 - building, kernel_size=3, stride=1, padding=1)
    boundary = (building - eroded).clamp(0.0, 1.0)
    weights = torch.where(boundary > 0, torch.full_like(boundary, boundary_weight), torch.ones_like(boundary))
    return weights * intact


def masked_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    boundary_weight: float,
) -> torch.Tensor:
    weights = intact_weights(mask, boundary_weight)
    denom = weights.sum()
    if float(denom.detach().cpu()) <= 0.0:
        return pred.sum() * 0.0
    loss_map = F.smooth_l1_loss(pred, target, reduction="none").mean(dim=1, keepdim=True)
    return (loss_map * weights).sum() / denom.clamp_min(1.0)


def regression_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, boundary_weight: float) -> dict[str, float]:
    weights = intact_weights(mask, boundary_weight)
    denom = weights.sum().clamp_min(1.0)
    abs_err = (pred - target).abs()
    mae = (abs_err.mean(dim=1, keepdim=True) * weights).sum() / denom
    sar_mae = (abs_err[:, 0:1] * weights).sum() / denom
    grad_mae = (abs_err[:, 1:2] * weights).sum() / denom
    return {
        "mae": float(mae.detach().cpu()),
        "sar_mae": float(sar_mae.detach().cpu()),
        "sar_grad_mae": float(grad_mae.detach().cpu()),
        "valid_weight": float(weights.sum().detach().cpu()),
    }


def merge_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    total_weight = sum(float(row["valid_weight"]) for row in rows)
    if total_weight <= 0.0:
        return {"mae": 0.0, "sar_mae": 0.0, "sar_grad_mae": 0.0, "valid_weight": 0.0}
    merged: dict[str, float] = {"valid_weight": total_weight}
    for key in ("mae", "sar_mae", "sar_grad_mae"):
        merged[key] = sum(float(row[key]) * float(row["valid_weight"]) for row in rows) / total_weight
    return merged


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp: bool,
    boundary_weight: float,
    max_batches: int | None = None,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    loss_sum = 0.0
    sample_count = 0
    metric_rows: list[dict[str, float]] = []
    desc = "train" if train else "val"
    for batch_index, batch in enumerate(tqdm(loader, desc=desc, leave=False)):
        if max_batches is not None and batch_index >= max_batches:
            break
        image = make_predictor_input(batch, device)
        target = make_predictor_target(batch, device)
        mask = batch["mask"].to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            pred = torch.sigmoid(forward_padded(model, image))
            loss = masked_smooth_l1(pred, target, mask, boundary_weight)
        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        batch_size = image.shape[0]
        loss_sum += float(loss.detach().cpu()) * batch_size
        sample_count += batch_size
        metric_rows.append(regression_metrics(pred.detach(), target, mask, boundary_weight))
    metrics = merge_metric_rows(metric_rows)
    metrics["loss"] = loss_sum / max(sample_count, 1)
    return metrics


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    cfg: dict[str, Any],
    epoch: int,
    seed: int,
    best_val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "seed": seed,
            "config": cfg,
            "best_val_loss": best_val_loss,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train OGSR-v0 expected SAR texture predictor")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--early-stopping-min-epochs", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = deepcopy(load_config(args.config))
    train_cfg = train_params(cfg)
    seed = int(args.seed if args.seed is not None else train_cfg.get("seed", cfg.get("seed", 42)))
    cfg.setdefault("train", {})["seed"] = seed
    set_seed(seed)

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config_resolved.json", cfg)

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is unavailable; pass --allow-cpu only for a diagnostic smoke test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(train_cfg.get("amp", True) if args.amp is None else args.amp) and device.type == "cuda"
    batch_size = int(args.batch_size or train_cfg.get("batch_size", 8))
    eval_batch_size = int(train_cfg.get("eval_batch_size", max(1, batch_size // 2)))
    num_workers = int(args.num_workers if args.num_workers is not None else train_cfg.get("num_workers", 8))
    epochs = int(args.epochs or train_cfg.get("epochs", 40))
    limit = None
    max_batches = None
    if args.smoke_test:
        limit = max(batch_size, 2)
        epochs = 1
        max_batches = 1
        eval_batch_size = 1

    data_root = Path(args.data_root)
    train_ds = make_dataset(cfg, data_root, "train", True, seed, limit)
    val_ds = make_dataset(cfg, data_root, "val", False, seed, limit if limit is not None else None)
    write_json(output_dir / "permutations" / "train.json", train_ds.permutation_info)
    write_json(output_dir / "permutations" / "val.json", val_ds.permutation_info)

    model = build_model(cfg, no_pretrained=args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = None
    if str(train_cfg.get("scheduler", "cosine")).lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    boundary_weight = float(dict(cfg.get("dataset", {})).get("intact_boundary_weight", 0.25))

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=max(0, min(num_workers, 4)),
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed + 1),
    )

    patience = int(
        args.early_stopping_patience
        if args.early_stopping_patience is not None
        else train_cfg.get("early_stopping_patience", 8)
    )
    min_epochs = int(
        args.early_stopping_min_epochs
        if args.early_stopping_min_epochs is not None
        else train_cfg.get("early_stopping_min_epochs", 10)
    )
    write_json(
        output_dir / "run_info.json",
        {
            "device": str(device),
            "cuda": torch.cuda.is_available(),
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
            "data_root": str(data_root.resolve()),
            "train_count": len(train_ds),
            "val_count": len(val_ds),
            "batch_size": batch_size,
            "eval_batch_size": eval_batch_size,
            "epochs": epochs,
            "amp": amp,
            "seed": seed,
            "boundary_weight": boundary_weight,
            "sar_shuffle_mode": train_ds.permutation_info["mode"],
            "early_stopping": {"patience": patience, "min_epochs": min_epochs},
            "code_state": code_state(),
            "data_audit": audit_state(),
            "cudnn": torch.backends.cudnn.version(),
        },
    )

    if args.smoke_test:
        train_metrics = run_epoch(
            model, train_loader, optimizer, scaler, device, amp, boundary_weight, max_batches=1
        )
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, None, scaler, device, amp, boundary_weight, max_batches=1)
        save_checkpoint(
            output_dir / "checkpoints" / "best_val_loss.pth",
            model,
            optimizer,
            scheduler,
            cfg,
            1,
            seed,
            float(val_metrics["loss"]),
        )
        payload = {"status": "ok", "train": train_metrics, "val": val_metrics}
        write_json(output_dir / "smoke_test.json", payload)
        print(json.dumps(payload, sort_keys=True), flush=True)
        return

    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    stopped_early = False
    min_delta = float(train_cfg.get("early_stopping_min_delta", 0.0))
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, optimizer, scaler, device, amp, boundary_weight, max_batches
        )
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, None, scaler, device, amp, boundary_weight, max_batches)
        if scheduler is not None:
            scheduler.step()
        row: dict[str, Any] = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        val_loss = float(val_metrics["loss"])
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint(
                output_dir / "checkpoints" / "best_val_loss.pth",
                model,
                optimizer,
                scheduler,
                cfg,
                epoch,
                seed,
                best_val_loss,
            )
        elif epoch > 1:
            epochs_without_improvement += 1
        row["best_val_loss"] = best_val_loss
        row["epochs_without_improvement"] = epochs_without_improvement
        history.append(row)
        write_csv(output_dir / "metrics_history.csv", history)
        write_json(output_dir / "latest_metrics.json", row)
        save_checkpoint(
            output_dir / "checkpoints" / "last.pth",
            model,
            optimizer,
            scheduler,
            cfg,
            epoch,
            seed,
            best_val_loss,
        )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "val_loss": val_loss,
                    "val_mae": val_metrics["mae"],
                    "best_val_loss": best_val_loss,
                    "epochs_without_improvement": epochs_without_improvement,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if patience > 0 and epoch >= min_epochs and epochs_without_improvement >= patience:
            stopped_early = True
            break

    completion = {
        "status": "completed",
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "best_val_loss": best_val_loss,
    }
    write_json(output_dir / "completed.json", completion)
    print(json.dumps(completion, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
