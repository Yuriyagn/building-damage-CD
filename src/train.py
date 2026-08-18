#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import augmentation_params, get_train_manifest, get_val_manifest, input_params, load_config, train_params
from datasets.disasterm3_stage1_dataset import DisasterM3Stage1Dataset
from losses import build_loss
from metrics import BinarySegmentationMeter
from models.build_model import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_train_dataset(cfg: dict[str, Any], data_root: Path, limit: int | None = None) -> DisasterM3Stage1Dataset:
    train_cfg = train_params(cfg)
    aug = augmentation_params(cfg)
    inp = input_params(cfg)
    return DisasterM3Stage1Dataset(
        data_root=data_root,
        manifest=get_train_manifest(cfg),
        train=True,
        crop_size=int(train_cfg.get("image_size", 512)),
        positive_crop_ratio=float(aug.get("positive_crop_ratio", 0.5)),
        hflip=bool(aug.get("hflip", False)),
        vflip=bool(aug.get("vflip", False)),
        rotate90=bool(aug.get("rotate90", False)),
        color_jitter=bool(aug.get("color_jitter", False)),
        normalize=str(inp.get("normalize", "imagenet")),
        limit=limit,
    )


def make_eval_dataset(cfg: dict[str, Any], data_root: Path, manifest: str | None = None, limit: int | None = None) -> DisasterM3Stage1Dataset:
    inp = input_params(cfg)
    return DisasterM3Stage1Dataset(
        data_root=data_root,
        manifest=manifest or get_val_manifest(cfg),
        train=False,
        crop_size=None,
        normalize=str(inp.get("normalize", "imagenet")),
        limit=limit,
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp: bool,
) -> dict[str, float]:
    model.train()
    meter = BinarySegmentationMeter()
    loss_sum = 0.0
    count = 0
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    for batch in tqdm(loader, desc="train", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(image)
            loss = criterion(logits, mask)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = image.shape[0]
        loss_sum += float(loss.item()) * bs
        count += bs
        meter.update(logits.detach(), mask)

    metrics = meter.compute()
    metrics["loss"] = loss_sum / max(count, 1)
    return metrics


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    amp: bool,
) -> dict[str, float]:
    model.eval()
    meter = BinarySegmentationMeter()
    loss_sum = 0.0
    count = 0
    for batch in tqdm(loader, desc="val", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(image)
            loss = criterion(logits, mask)
        bs = image.shape[0]
        loss_sum += float(loss.item()) * bs
        count += bs
        meter.update(logits, mask)
    metrics = meter.compute()
    metrics["loss"] = loss_sum / max(count, 1)
    return metrics


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    epoch: int,
    best_iou: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_iou": best_iou,
            "config": cfg,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )


def early_stopping_params(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    train_cfg = train_params(cfg)
    patience = int(train_cfg.get("early_stopping_patience", 0))
    min_delta = float(train_cfg.get("early_stopping_min_delta", 0.0))
    min_epochs = int(train_cfg.get("early_stopping_min_epochs", 0))

    if args.early_stopping_patience is not None:
        patience = int(args.early_stopping_patience)
    if args.early_stopping_min_delta is not None:
        min_delta = float(args.early_stopping_min_delta)
    if args.early_stopping_min_epochs is not None:
        min_epochs = int(args.early_stopping_min_epochs)

    patience = max(0, patience)
    min_delta = max(0.0, min_delta)
    min_epochs = max(0, min_epochs)
    return {
        "enabled": patience > 0,
        "patience": patience,
        "min_delta": min_delta,
        "min_epochs": min_epochs,
        "metric": "val_iou_building",
        "mode": "max",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--overfit-batches", type=int)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--early-stopping-min-delta", type=float)
    parser.add_argument("--early-stopping-min-epochs", type=int)
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--log-steps", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    train_cfg = train_params(cfg)
    seed = int(train_cfg.get("seed", cfg.get("seed", 42)))
    set_seed(seed)

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config_resolved.json", cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(args.amp and device.type == "cuda")
    model = build_model(cfg, no_pretrained=args.no_pretrained).to(device)
    criterion = build_loss(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    batch_size = int(args.batch_size or train_cfg.get("batch_size", 8))
    num_workers = int(args.num_workers if args.num_workers is not None else train_cfg.get("num_workers", 8))
    epochs = int(args.epochs or train_cfg.get("epochs", 100))
    early_stopping = early_stopping_params(cfg, args)
    limit = None
    if args.smoke_test:
        limit = max(batch_size, 2)
        epochs = 1
    if args.overfit_batches:
        limit = max(1, args.overfit_batches * batch_size)

    train_ds = make_train_dataset(cfg, Path(args.data_root), limit=limit)
    val_ds = make_eval_dataset(cfg, Path(args.data_root), limit=limit if args.overfit_batches else None)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(0, min(num_workers, 4)),
        pin_memory=device.type == "cuda",
    )

    write_json(
        output_dir / "run_info.json",
        {
            "device": str(device),
            "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
            "train_count": len(train_ds),
            "val_count": len(val_ds),
            "batch_size": batch_size,
            "epochs": epochs,
            "amp": amp,
            "early_stopping": early_stopping,
            "max_optimizer_steps": args.max_optimizer_steps,
        },
    )

    if args.smoke_test:
        batch = next(iter(train_loader))
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(image)
        loss = criterion(logits, mask)
        loss.backward()
        optimizer.step()
        metrics = evaluate(model, val_loader, criterion, device, amp=False)
        write_json(output_dir / "smoke_test.json", {"loss": float(loss.item()), "val": metrics})
        print(json.dumps({"smoke_test": "ok", "loss": float(loss.item()), "val_iou": metrics["iou_building"]}))
        return

    if args.max_optimizer_steps is not None:
        max_steps = int(args.max_optimizer_steps)
        if max_steps <= 0:
            raise ValueError("--max-optimizer-steps must be positive")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
        model.train()
        global_step = 0
        running_loss = 0.0
        running_count = 0
        history: list[dict[str, Any]] = []
        while global_step < max_steps:
            for batch in train_loader:
                image = batch["image"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=amp):
                    logits = model(image)
                    loss = criterion(logits, mask)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                global_step += 1
                running_loss += float(loss.item())
                running_count += 1
                if global_step % int(args.log_steps) == 0 or global_step == max_steps:
                    row = {
                        "optimizer_step": global_step,
                        "train_loss_window": running_loss / max(running_count, 1),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                    }
                    history.append(row)
                    write_csv(output_dir / "metrics_history.csv", history)
                    write_json(output_dir / "latest_metrics.json", row)
                    print(json.dumps(row, sort_keys=True), flush=True)
                    running_loss = 0.0
                    running_count = 0
                if global_step >= max_steps:
                    break
        final_metrics = evaluate(model, val_loader, criterion, device, amp)
        save_checkpoint(output_dir / "checkpoints" / "fixed_final.pth", model, optimizer, cfg, 0, -1.0)
        completion = {
            "status": "completed",
            "optimizer_steps": global_step,
            "selection_policy": "fixed_step_no_heldout_checkpoint_selection",
            "export_evaluation": final_metrics,
        }
        write_json(output_dir / "completed.json", completion)
        print(json.dumps(completion, sort_keys=True), flush=True)
        return

    history = []
    best_iou = -1.0
    early_stop_best_iou = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, amp)
        val_metrics = evaluate(model, val_loader, criterion, device, amp)
        val_iou = float(val_metrics["iou_building"])
        previous_best_iou = best_iou
        checkpoint_improved = val_iou > previous_best_iou
        significant_improved = val_iou > early_stop_best_iou + float(early_stopping["min_delta"])

        if checkpoint_improved:
            best_iou = val_iou
            best_epoch = epoch
            save_checkpoint(output_dir / "checkpoints" / "best_iou.pth", model, optimizer, cfg, epoch, best_iou)

        if significant_improved:
            early_stop_best_iou = val_iou
            epochs_without_improvement = 0
        elif epoch > 1:
            epochs_without_improvement += 1

        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "best_iou": best_iou,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
        }
        history.append(row)
        write_csv(output_dir / "metrics_history.csv", history)
        write_json(output_dir / "latest_metrics.json", row)

        save_checkpoint(output_dir / "checkpoints" / "last.pth", model, optimizer, cfg, epoch, best_iou)

        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "val_loss": val_metrics["loss"],
                    "val_iou": val_iou,
                    "best_iou": best_iou,
                    "best_epoch": best_epoch,
                    "epochs_without_improvement": epochs_without_improvement,
                },
                sort_keys=True,
            ),
            flush=True,
        )

        if (
            early_stopping["enabled"]
            and epoch >= int(early_stopping["min_epochs"])
            and epochs_without_improvement >= int(early_stopping["patience"])
        ):
            payload = {
                "early_stopped": True,
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_iou": best_iou,
                "patience": early_stopping["patience"],
                "min_delta": early_stopping["min_delta"],
                "min_epochs": early_stopping["min_epochs"],
            }
            write_json(output_dir / "early_stop.json", payload)
            print(json.dumps(payload, sort_keys=True), flush=True)
            break


if __name__ == "__main__":
    main()
