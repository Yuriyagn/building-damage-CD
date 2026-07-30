#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import load_config, train_params  # noqa: E402
from models.build_model import build_model  # noqa: E402
from stage2.common import write_csv, write_json  # noqa: E402
from stage2.datasets import Stage2DamageDataset  # noqa: E402
from stage2.losses import build_stage2_loss  # noqa: E402
from stage2.metrics import Stage2DamageMeter  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dataset_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(cfg.get("dataset", {}))


def aug_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(cfg.get("augmentation", {}))


def make_dataset(
    cfg: dict[str, Any],
    data_root: Path,
    split: str,
    train: bool,
    limit: int | None = None,
) -> Stage2DamageDataset:
    dcfg = dataset_cfg(cfg)
    acfg = aug_cfg(cfg)
    manifest = dcfg[f"{split}_manifest"]
    crop_probs = acfg.get("crop_probabilities") or {
        "damaged": 0.4,
        "destroyed": 0.4,
        "building": 0.2,
        "random": 0.0,
    }
    return Stage2DamageDataset(
        data_root=data_root,
        manifest=manifest,
        train=train,
        prior_type=str(dcfg.get("prior_type", "predicted")),
        input_mode=str(dcfg.get("input_mode", "sar_prior")),
        crop_size=int(dcfg.get("crop_size", 512)) if train else None,
        crop_probabilities={str(k): float(v) for k, v in dict(crop_probs).items()},
        hflip=bool(acfg.get("hflip", True)),
        vflip=bool(acfg.get("vflip", False)),
        rotate90=bool(acfg.get("rotate90", True)),
        limit=limit,
    )


def compute_class_weights(dataset: Stage2DamageDataset, cfg: dict[str, Any]) -> tuple[torch.Tensor | None, dict[str, Any]]:
    loss_cfg = dict(cfg.get("loss", {}))
    if "class_weights" in loss_cfg:
        weights = torch.tensor([float(x) for x in loss_cfg["class_weights"]], dtype=torch.float32)
        return weights, {"strategy": "explicit", "class_weights": weights.tolist()}
    if str(loss_cfg.get("class_weight_strategy", "median_frequency")).lower() in {"none", "null", "false"}:
        return None, {"strategy": "none", "class_weights": None}

    counts = np.zeros(4, dtype=np.int64)
    for row in tqdm(dataset.rows, desc="class_weights", leave=False):
        path = dataset.data_root / str(row["mask_multiclass"])
        if Path(str(row["mask_multiclass"])).is_absolute():
            path = Path(str(row["mask_multiclass"]))
        mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
        bincount = np.bincount(mask.reshape(-1), minlength=4)[:4]
        counts += bincount.astype(np.int64)

    frequencies = counts.astype(np.float64) / max(float(counts.sum()), 1.0)
    nonzero = frequencies[frequencies > 0]
    median = float(np.median(nonzero)) if len(nonzero) else 1.0
    weights_np = np.ones(4, dtype=np.float32)
    for idx, freq in enumerate(frequencies):
        weights_np[idx] = float(median / freq) if freq > 0 else 1.0
    max_weight = float(loss_cfg.get("max_class_weight", 8.0))
    weights_np = np.clip(weights_np, 0.0, max_weight)
    min_background = float(loss_cfg.get("min_background_weight", 0.05))
    weights_np[0] = max(float(weights_np[0]), min_background)
    weights = torch.tensor(weights_np, dtype=torch.float32)
    return weights, {
        "strategy": str(loss_cfg.get("class_weight_strategy", "median_frequency")),
        "pixel_counts": counts.astype(int).tolist(),
        "frequencies": frequencies.tolist(),
        "median_frequency": median,
        "max_class_weight": max_weight,
        "class_weights": weights.tolist(),
    }


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp: bool,
) -> dict[str, float]:
    model.train()
    meter = Stage2DamageMeter()
    loss_sum = 0.0
    count = 0
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    for batch in tqdm(loader, desc="train", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(image)
            loss = criterion(logits, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        bs = image.shape[0]
        loss_sum += float(loss.item()) * bs
        count += bs
        meter.update_logits(logits.detach(), target)
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
    meter = Stage2DamageMeter()
    loss_sum = 0.0
    count = 0
    for batch in tqdm(loader, desc="val", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(image)
            loss = criterion(logits, target)
        bs = image.shape[0]
        loss_sum += float(loss.item()) * bs
        count += bs
        meter.update_logits(logits, target)
    metrics = meter.compute()
    metrics["loss"] = loss_sum / max(count, 1)
    return metrics


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    cfg: dict[str, Any],
    epoch: int,
    best_metric: float,
    class_weight_info: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_metric": best_metric,
            "config": cfg,
            "class_weight_info": class_weight_info,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        },
        path,
    )


def early_stopping_params(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    train_cfg = train_params(cfg)
    patience = int(train_cfg.get("early_stopping_patience", 20))
    min_delta = float(train_cfg.get("early_stopping_min_delta", 0.0))
    min_epochs = int(train_cfg.get("early_stopping_min_epochs", 30))
    if args.early_stopping_patience is not None:
        patience = int(args.early_stopping_patience)
    if args.early_stopping_min_delta is not None:
        min_delta = float(args.early_stopping_min_delta)
    if args.early_stopping_min_epochs is not None:
        min_epochs = int(args.early_stopping_min_epochs)
    metric = str(train_cfg.get("checkpoint_metric", "val_building_only_damage_macro_f1"))
    return {
        "enabled": patience > 0,
        "patience": max(0, patience),
        "min_delta": max(0.0, min_delta),
        "min_epochs": max(0, min_epochs),
        "metric": metric,
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    train_cfg = train_params(cfg)
    seed = int(train_cfg.get("seed", cfg.get("seed", 42)))
    set_seed(seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config_resolved.json", cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(args.amp and device.type == "cuda")
    batch_size = int(args.batch_size or train_cfg.get("batch_size", 8))
    num_workers = int(args.num_workers if args.num_workers is not None else train_cfg.get("num_workers", 8))
    epochs = int(args.epochs or train_cfg.get("epochs", 100))
    limit = None
    if args.smoke_test:
        limit = max(batch_size, 2)
        epochs = 1
    if args.overfit_batches:
        limit = max(1, args.overfit_batches * batch_size)

    train_ds = make_dataset(cfg, Path(args.data_root), split="train", train=True, limit=limit)
    val_limit = limit if (args.smoke_test or args.overfit_batches) else None
    val_ds = make_dataset(cfg, Path(args.data_root), split="val", train=False, limit=val_limit)
    class_weights, class_weight_info = compute_class_weights(train_ds, cfg)
    write_json(output_dir / "class_weights.json", class_weight_info)

    model = build_model(cfg, no_pretrained=args.no_pretrained).to(device)
    criterion = build_stage2_loss(cfg, class_weights=class_weights).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = None
    if str(train_cfg.get("scheduler", "cosine")).lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

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
        batch_size=int(train_cfg.get("eval_batch_size", 1)),
        shuffle=False,
        num_workers=max(0, min(num_workers, 4)),
        pin_memory=device.type == "cuda",
    )
    early_stopping = early_stopping_params(cfg, args)
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
            "class_weight_info": class_weight_info,
        },
    )

    if args.smoke_test:
        batch = next(iter(train_loader))
        image = batch["image"].to(device)
        target = batch["mask"].to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(image)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        metrics = evaluate(model, val_loader, criterion, device, amp=False)
        write_json(output_dir / "smoke_test.json", {"loss": float(loss.item()), "val": metrics})
        print(json.dumps({"smoke_test": "ok", "loss": float(loss.item()), "val_metric": metrics}, sort_keys=True))
        return

    history: list[dict[str, Any]] = []
    best_metric = -1.0
    best_epoch = 0
    early_stop_best = -1.0
    epochs_without_improvement = 0
    metric_name = str(early_stopping["metric"])
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, amp)
        val_metrics = evaluate(model, val_loader, criterion, device, amp)
        if scheduler is not None:
            scheduler.step()
        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        current_metric = float(row.get(metric_name, row.get("val_miou_4class", 0.0)))
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            save_checkpoint(
                output_dir / "checkpoints" / "best_metric.pth",
                model,
                optimizer,
                scheduler,
                cfg,
                epoch,
                best_metric,
                class_weight_info,
            )
        if current_metric > early_stop_best + float(early_stopping["min_delta"]):
            early_stop_best = current_metric
            epochs_without_improvement = 0
        elif epoch > 1:
            epochs_without_improvement += 1

        row["best_metric"] = best_metric
        row["best_epoch"] = best_epoch
        row["checkpoint_metric"] = metric_name
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
            best_metric,
            class_weight_info,
        )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "val_loss": val_metrics["loss"],
                    "checkpoint_metric": metric_name,
                    "val_checkpoint_metric": current_metric,
                    "best_metric": best_metric,
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
                "best_metric": best_metric,
                "checkpoint_metric": metric_name,
                "patience": early_stopping["patience"],
                "min_delta": early_stopping["min_delta"],
                "min_epochs": early_stopping["min_epochs"],
            }
            write_json(output_dir / "early_stop.json", payload)
            print(json.dumps(payload, sort_keys=True), flush=True)
            break


if __name__ == "__main__":
    main()
