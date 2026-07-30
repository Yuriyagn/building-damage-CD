#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import load_config, train_params  # noqa: E402
from models.build_model import build_model  # noqa: E402
from stage2.common import write_csv, write_json  # noqa: E402
from stage2.datasets_v2 import Stage2V2Dataset  # noqa: E402
from stage2.losses_v2 import build_stage2_v2_loss  # noqa: E402
from stage2.metrics_v2 import Stage2V2MeterBundle  # noqa: E402


CHECKPOINT_METRICS = {
    "best_bo_grade_macro_f1.pth": "val_building_only_macro_f1_3class",
    "best_bo_damage_macro_f1.pth": "val_building_only_damage_macro_f1",
    "best_bo_damaged_f1.pth": "val_building_only_f1_damaged",
}


def checkpoint_metrics_for_policy(train_cfg: dict[str, Any]) -> dict[str, str]:
    policy = str(train_cfg.get("checkpoint_policy", "all_metrics")).lower()
    if policy in {"all", "all_metrics", "legacy"}:
        return dict(CHECKPOINT_METRICS)
    if policy in {"primary", "primary_only"}:
        return {
            "best_bo_grade_macro_f1.pth": CHECKPOINT_METRICS["best_bo_grade_macro_f1.pth"]
        }
    raise ValueError(f"unknown checkpoint policy: {policy}")


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
    cache_items: bool = False,
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
        input_mode=str(dcfg.get("input_mode", "sar_prior")),
        crop_size=int(dcfg.get("crop_size", 512)) if train else None,
        crop_probabilities={str(k): float(v) for k, v in dict(crop_probs).items()},
        hflip=bool(acfg.get("hflip", True)),
        vflip=bool(acfg.get("vflip", False)),
        rotate90=bool(acfg.get("rotate90", True)),
        ogsr_feature_root=dcfg.get("ogsr_feature_root"),
        ogsr_feature_keys=dcfg.get("ogsr_feature_keys"),
        sar_shuffle_mode=str(dcfg.get("sar_shuffle_mode", "paired")),
        sar_shuffle_seed=permutation_seed + split_offset,
        sar_singleton_policy=str(dcfg.get("sar_singleton_policy", "error")),
        cache_items=cache_items,
        limit=limit,
    )


def compute_class_weights(dataset: Stage2V2Dataset, cfg: dict[str, Any]) -> tuple[torch.Tensor | None, dict[str, Any]]:
    loss_cfg = dict(cfg.get("loss", {}))
    if "class_weights" in loss_cfg:
        weights = torch.tensor([float(x) for x in loss_cfg["class_weights"]], dtype=torch.float32)
        if weights.numel() != 3:
            raise ValueError("Stage-2 v2 requires three class weights")
        return weights, {"strategy": "explicit", "class_weights": weights.tolist()}
    strategy = str(loss_cfg.get("class_weight_strategy", "median_frequency")).lower()
    if strategy in {"none", "null", "false"}:
        return None, {"strategy": "none", "class_weights": None}

    counts = np.zeros(3, dtype=np.int64)
    for row in tqdm(dataset.rows, desc="class_weights", leave=False):
        path = Path(str(row["mask_multiclass"]))
        if not path.is_absolute():
            path = dataset.data_root / path
        mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
        bincount = np.bincount(mask.reshape(-1), minlength=4)[1:4]
        counts += bincount.astype(np.int64)
    frequencies = counts.astype(np.float64) / max(float(counts.sum()), 1.0)
    nonzero = frequencies[frequencies > 0]
    median = float(np.median(nonzero)) if len(nonzero) else 1.0
    weights_np = np.ones(3, dtype=np.float32)
    for index, frequency in enumerate(frequencies):
        weights_np[index] = float(median / frequency) if frequency > 0 else 1.0
    max_weight = float(loss_cfg.get("max_class_weight", 8.0))
    weights_np = np.clip(weights_np, 0.0, max_weight)
    weights = torch.tensor(weights_np, dtype=torch.float32)
    return weights, {
        "strategy": strategy,
        "building_pixel_counts": counts.astype(int).tolist(),
        "building_frequencies": frequencies.tolist(),
        "median_frequency": median,
        "max_class_weight": max_weight,
        "class_weights": weights.tolist(),
    }


def build_sampler(dataset: Stage2V2Dataset, alpha: float, seed: int) -> WeightedRandomSampler | None:
    if alpha <= 0:
        return None
    counts = Counter(str(row.get("disaster_type", "") or "unknown") for row in dataset.rows)
    weights = [counts[str(row.get("disaster_type", "") or "unknown")] ** (-alpha) for row in dataset.rows]
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True, generator=generator)


def _bundle_update(
    meter: Stage2V2MeterBundle,
    logits: torch.Tensor,
    target: torch.Tensor,
    prior: torch.Tensor,
    gate_threshold: float,
) -> None:
    grade = torch.argmax(logits.detach(), dim=1).cpu().numpy().astype(np.uint8) + 1
    true = target.detach().cpu().numpy().astype(np.uint8)
    support = prior[:, 0].detach().cpu().numpy() >= gate_threshold
    meter.update(grade, true, support)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp: bool,
    gate_threshold: float,
    max_batches: int | None = None,
    gradient_accumulation_steps: int = 1,
) -> dict[str, float]:
    model.train()
    meter = Stage2V2MeterBundle()
    loss_sum = 0.0
    sample_count = 0
    accumulation_steps = max(1, int(gradient_accumulation_steps))
    total_batches = len(loader) if max_batches is None else min(len(loader), int(max_batches))
    optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        if max_batches is not None and batch_index >= max_batches:
            break
        image = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(image)
            loss = criterion(logits, target)
        group_start = (batch_index // accumulation_steps) * accumulation_steps
        current_group_size = min(accumulation_steps, total_batches - group_start)
        scaler.scale(loss / max(current_group_size, 1)).backward()
        processed_batches = batch_index + 1
        if processed_batches % accumulation_steps == 0 or processed_batches == total_batches:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        batch_size = image.shape[0]
        loss_sum += float(loss.item()) * batch_size
        sample_count += batch_size
        _bundle_update(meter, logits, target, batch["prior"], gate_threshold)
    metrics = meter.compute()
    metrics["loss"] = loss_sum / max(sample_count, 1)
    return metrics


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    amp: bool,
    gate_threshold: float,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    meter = Stage2V2MeterBundle()
    loss_sum = 0.0
    sample_count = 0
    for batch_index, batch in enumerate(tqdm(loader, desc="val", leave=False)):
        if max_batches is not None and batch_index >= max_batches:
            break
        image = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(image)
            loss = criterion(logits, target)
        batch_size = image.shape[0]
        loss_sum += float(loss.item()) * batch_size
        sample_count += batch_size
        _bundle_update(meter, logits, target, batch["prior"], gate_threshold)
    metrics = meter.compute()
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
    best_metrics: dict[str, dict[str, float | int]],
    class_weight_info: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "seed": seed,
            "config": cfg,
            "best_metrics": best_metrics,
            "class_weight_info": class_weight_info,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-2 v2 building-only three-grade training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--overfit-batches", type=int)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--early-stopping-min-epochs", type=int)
    parser.add_argument("--ogsr-feature-root")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = deepcopy(load_config(args.config))
    train_cfg = train_params(cfg)
    checkpoint_metrics = checkpoint_metrics_for_policy(train_cfg)
    checkpoint_policy = str(train_cfg.get("checkpoint_policy", "all_metrics")).lower()
    save_last_checkpoint = bool(train_cfg.get("save_last_checkpoint", True))
    gradient_accumulation_steps = max(1, int(train_cfg.get("gradient_accumulation_steps", 1)))
    seed = int(args.seed if args.seed is not None else train_cfg.get("seed", cfg.get("seed", 42)))
    cfg.setdefault("train", {})["seed"] = seed
    if args.ogsr_feature_root:
        cfg.setdefault("dataset", {})["ogsr_feature_root"] = args.ogsr_feature_root
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
    num_workers = int(args.num_workers if args.num_workers is not None else train_cfg.get("num_workers", 8))
    epochs = int(args.epochs or train_cfg.get("epochs", 100))
    limit = None
    max_batches = None
    if args.smoke_test:
        limit = max(batch_size, 2)
        epochs = 1
        max_batches = 1
    elif args.overfit_batches:
        limit = max(1, args.overfit_batches * batch_size)
        max_batches = int(args.overfit_batches)

    data_root = Path(args.data_root)
    train_ds = make_dataset(
        cfg, data_root, "train", True, seed, limit, cache_items=bool(args.overfit_batches)
    )
    val_ds = make_dataset(cfg, data_root, "val", False, seed, limit if limit is not None else None)
    write_json(output_dir / "permutations" / "train.json", train_ds.permutation_info)
    write_json(output_dir / "permutations" / "val.json", val_ds.permutation_info)
    class_weights, class_weight_info = compute_class_weights(train_ds, cfg)
    write_json(output_dir / "class_weights.json", class_weight_info)

    model = build_model(cfg, no_pretrained=args.no_pretrained).to(device)
    criterion = build_stage2_v2_loss(cfg, class_weights).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = None
    if str(train_cfg.get("scheduler", "cosine")).lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    sampler = build_sampler(train_ds, float(train_cfg.get("disaster_sampling_alpha", 0.0)), seed)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=num_workers > 0,
    )
    eval_batch_size = int(train_cfg.get("eval_batch_size", 1))
    if args.smoke_test:
        eval_batch_size = 1
    val_loader = DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=max(0, min(num_workers, 4)),
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed + 1),
    )
    gate_threshold = float(dict(cfg.get("dataset", {})).get("gate_threshold", 0.6))
    patience = int(args.early_stopping_patience if args.early_stopping_patience is not None else train_cfg.get("early_stopping_patience", 30))
    min_epochs = int(args.early_stopping_min_epochs if args.early_stopping_min_epochs is not None else train_cfg.get("early_stopping_min_epochs", 40))
    early_window = int(train_cfg.get("early_stopping_window", 3))
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
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": batch_size * gradient_accumulation_steps,
            "eval_batch_size": eval_batch_size,
            "epochs": epochs,
            "amp": amp,
            "seed": seed,
            "gate_threshold": gate_threshold,
            "sar_shuffle_mode": train_ds.permutation_info["mode"],
            "disaster_sampling_alpha": float(train_cfg.get("disaster_sampling_alpha", 0.0)),
            "early_stopping": {"patience": patience, "min_epochs": min_epochs, "window": early_window},
            "checkpoint_policy": checkpoint_policy,
            "save_last_checkpoint": save_last_checkpoint,
            "code_state": code_state(),
            "data_audit": audit_state(),
            "cudnn": torch.backends.cudnn.version(),
        },
    )

    if args.smoke_test:
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            amp,
            gate_threshold,
            max_batches=1,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        val_metrics = evaluate(model, val_loader, criterion, device, amp, gate_threshold, max_batches=1)
        payload = {"status": "ok", "train": train_metrics, "val": val_metrics}
        write_json(output_dir / "smoke_test.json", payload)
        print(json.dumps(payload, sort_keys=True), flush=True)
        return

    history: list[dict[str, Any]] = []
    best_metrics = {
        name: {"metric": metric, "value": -1.0, "epoch": 0}
        for name, metric in checkpoint_metrics.items()
    }
    smoothed_best = -1.0
    epochs_without_improvement = 0
    primary_name = "val_building_only_macro_f1_3class"
    stopped_early = False
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            amp,
            gate_threshold,
            max_batches,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        val_metrics = evaluate(model, val_loader, criterion, device, amp, gate_threshold, max_batches)
        if scheduler is not None:
            scheduler.step()
        row: dict[str, Any] = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        for filename, metric_name in checkpoint_metrics.items():
            value = float(row[metric_name])
            if value > float(best_metrics[filename]["value"]):
                best_metrics[filename] = {"metric": metric_name, "value": value, "epoch": epoch}
                save_checkpoint(
                    output_dir / "checkpoints" / filename,
                    model,
                    optimizer,
                    scheduler,
                    cfg,
                    epoch,
                    seed,
                    best_metrics,
                    class_weight_info,
                )
        recent = history[-(early_window - 1) :] if early_window > 1 else []
        primary_values = [float(old[primary_name]) for old in recent] + [float(row[primary_name])]
        smoothed = float(np.mean(primary_values))
        if smoothed > smoothed_best + float(train_cfg.get("early_stopping_min_delta", 0.0)):
            smoothed_best = smoothed
            epochs_without_improvement = 0
        elif epoch > 1:
            epochs_without_improvement += 1
        row["early_stopping_smoothed_primary"] = smoothed
        row["epochs_without_improvement"] = epochs_without_improvement
        history.append(row)
        write_csv(output_dir / "metrics_history.csv", history)
        write_json(output_dir / "latest_metrics.json", row)
        if save_last_checkpoint:
            save_checkpoint(
                output_dir / "checkpoints" / "last.pth",
                model,
                optimizer,
                scheduler,
                cfg,
                epoch,
                seed,
                best_metrics,
                class_weight_info,
            )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "val_loss": val_metrics["loss"],
                    "val_bo_grade_macro_f1": row[primary_name],
                    "val_bo_damage_macro_f1": row["val_building_only_damage_macro_f1"],
                    "smoothed_primary": smoothed,
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
        "checkpoint_policy": checkpoint_policy,
        "save_last_checkpoint": save_last_checkpoint,
        "best_metrics": best_metrics,
    }
    write_json(output_dir / "completed.json", completion)
    print(json.dumps(completion, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
