#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler
from tqdm import tqdm

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import load_config, train_params  # noqa: E402
from models.build_model import build_model  # noqa: E402
from models.metadata_multitask import (  # noqa: E402
    diagnostic_shared_parameters,
    gradient_interaction,
    unpack_model_output,
)
from stage2.common import read_jsonl, resolve_manifest, write_csv, write_json  # noqa: E402
from stage2.datasets_v2 import Stage2V2Dataset  # noqa: E402
from stage2.losses_v2 import build_stage2_v2_loss  # noqa: E402
from stage2.metadata_multitask import (  # noqa: E402
    DISASTER_CLASSES,
    DisasterClassificationMeter,
    disaster_class_counts,
    encode_disaster_types,
    inverse_sqrt_class_weights,
    load_label_map,
    validate_disaster_classes,
)
from stage2.metrics_v2 import GroupedV2Meters, Stage2V2MeterBundle  # noqa: E402


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


def model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def audit_state() -> dict[str, Any]:
    configured_path = os.environ.get("STAGE2_DATA_AUDIT_PATH")
    if configured_path:
        path = Path(configured_path)
        if not path.is_absolute():
            path = SRC_ROOT.parent / path
    else:
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


def resolve_batch_limits(
    *,
    smoke_test: bool,
    overfit_batches: int | None,
    fixed_step_mode: bool,
    batch_size: int,
    gradient_accumulation_steps: int,
) -> tuple[int | None, int | None, int | None]:
    """Return dataset limit, train-batch limit, and validation-batch limit.

    Formal fixed-step training consumes one optimizer step worth of training
    micro-batches, but validation must cover the complete validation loader.
    The previous implementation reused the training limit for validation and
    silently evaluated only one validation batch at every selection step.
    """

    if smoke_test:
        return max(batch_size, 2), 1, 1
    if overfit_batches:
        batches = int(overfit_batches)
        return max(1, batches * batch_size), batches, batches
    if fixed_step_mode:
        return None, max(1, int(gradient_accumulation_steps)), None
    return None, None, None


def resolve_training_diagnostics(
    *,
    fixed_step_mode: bool,
    optimizer_step: int,
    eval_steps: set[int],
    progress_interval_steps: int,
) -> tuple[bool, bool]:
    """Return whether to collect full train metrics and the scalar loss.

    Full pixel/event diagnostics copy predictions from the GPU to the CPU. They
    are useful beside validation checkpoints, but performing them at every
    fixed optimizer step stalls the training pipeline without affecting model
    selection. A lightweight scalar loss is retained at progress heartbeats.
    """

    if not fixed_step_mode:
        return True, True
    collect_metrics = optimizer_step in eval_steps
    collect_loss = collect_metrics or optimizer_step % progress_interval_steps == 0
    return collect_metrics, collect_loss


class CyclingDataIterator:
    """Keep one DataLoader iterator alive and cycle only after a full pass.

    Fixed-step training previously called ``iter(loader)`` once per optimizer
    step and consumed only its first batch. Even with persistent workers that
    repeatedly rebuilt the sampler/prefetch pipeline and left the GPU waiting
    on CPU input. This iterator preserves the normal shuffled epoch traversal
    and only creates a new iterator after the current one is exhausted.
    """

    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader
        self._iterator = iter(loader)
        self.completed_cycles = 0

    def __iter__(self) -> "CyclingDataIterator":
        return self

    def __next__(self) -> dict[str, Any]:
        try:
            return next(self._iterator)
        except StopIteration:
            self.completed_cycles += 1
            self._iterator = iter(self.loader)
            return next(self._iterator)


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
        sar_permutation_file=dcfg.get(f"{split}_sar_permutation_file") or dcfg.get("sar_permutation_file"),
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


def build_capped_event_sampler(
    dataset: Stage2V2Dataset,
    *,
    alpha: float,
    min_weight: float,
    max_weight: float,
    seed: int,
) -> tuple[WeightedRandomSampler, dict[str, Any]]:
    """Build the frozen RQ2-C sqrt event sampler with auditable caps."""

    if alpha <= 0:
        raise ValueError("capped_event requires event_sampling_alpha > 0")
    if min_weight <= 0 or max_weight < min_weight:
        raise ValueError("invalid capped_event weight bounds")
    counts = Counter(str(row.get("event_id", "") or "unknown") for row in dataset.rows)
    if not counts:
        raise ValueError("capped_event requires a non-empty dataset")
    median_count = float(np.median(list(counts.values())))
    raw_by_event = {
        event_id: float((median_count / count) ** alpha)
        for event_id, count in counts.items()
    }
    clipped_by_event = {
        event_id: float(np.clip(weight, min_weight, max_weight))
        for event_id, weight in raw_by_event.items()
    }
    raw_sample_weights = np.asarray(
        [clipped_by_event[str(row.get("event_id", "") or "unknown")] for row in dataset.rows],
        dtype=np.float64,
    )
    normalized_sample_weights = raw_sample_weights / float(raw_sample_weights.mean())
    weight_tensor = torch.tensor(normalized_sample_weights, dtype=torch.double)
    audit_generator = torch.Generator().manual_seed(int(seed))
    first_epoch_indices = torch.multinomial(
        weight_tensor,
        num_samples=len(dataset),
        replacement=True,
        generator=audit_generator,
    ).tolist()
    first_epoch_counts = Counter(
        str(dataset.rows[index].get("event_id", "") or "unknown")
        for index in first_epoch_indices
    )
    sampler = WeightedRandomSampler(
        weight_tensor,
        num_samples=len(dataset),
        replacement=True,
        generator=torch.Generator().manual_seed(int(seed)),
    )
    event_rows = []
    total_weight = float(weight_tensor.sum())
    for event_id in sorted(counts):
        per_image_weight = float(
            normalized_sample_weights[
                next(
                    index
                    for index, row in enumerate(dataset.rows)
                    if str(row.get("event_id", "") or "unknown") == event_id
                )
            ]
        )
        expected_draws = len(dataset) * counts[event_id] * per_image_weight / total_weight
        event_rows.append(
            {
                "event_id": event_id,
                "image_count": int(counts[event_id]),
                "raw_per_image_weight": raw_by_event[event_id],
                "clipped_per_image_weight": clipped_by_event[event_id],
                "normalized_per_image_weight": per_image_weight,
                "expected_draw_count": float(expected_draws),
                "first_epoch_draw_count": int(first_epoch_counts[event_id]),
            }
        )
    return sampler, {
        "strategy": "capped_event",
        "sampling_unit": "canonical_event",
        "replacement": True,
        "num_samples_per_epoch": len(dataset),
        "seed": int(seed),
        "alpha": float(alpha),
        "min_weight": float(min_weight),
        "max_weight": float(max_weight),
        "median_event_count": median_count,
        "events": event_rows,
    }


class EventClassBalancedSampler(Sampler[tuple[int, int]]):
    """Sample an (event, class) group uniformly, then crop that class in one group image."""

    def __init__(
        self,
        groups: dict[tuple[str, int], list[int]],
        *,
        num_samples: int,
        seed: int,
    ) -> None:
        self.groups = {key: tuple(values) for key, values in sorted(groups.items())}
        if not self.groups:
            raise ValueError("event-class sampler has no eligible groups")
        self.keys = tuple(self.groups)
        self.num_samples = int(num_samples)
        self.generator = torch.Generator().manual_seed(int(seed))

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        group_choices = torch.randint(
            len(self.keys), (self.num_samples,), generator=self.generator
        ).tolist()
        for group_index in group_choices:
            key = self.keys[int(group_index)]
            members = self.groups[key]
            member_index = int(
                torch.randint(len(members), (1,), generator=self.generator).item()
            )
            yield int(members[member_index]), int(key[1])


def build_event_class_sampler(
    dataset: Stage2V2Dataset,
    *,
    seed: int,
    min_pixels_per_image: int,
    min_images_per_group: int,
    classes: list[int],
) -> tuple[EventClassBalancedSampler, dict[str, Any]]:
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    image_class_pixels: list[dict[str, Any]] = []
    for index, row in enumerate(tqdm(dataset.rows, desc="event_class_groups", leave=False)):
        target = dataset.load_target(row)
        event_id = str(row.get("event_id", "") or "unknown")
        counts = np.bincount(target.reshape(-1), minlength=4)
        record = {"index": index, "event_id": event_id}
        for target_class in classes:
            pixels = int(counts[target_class])
            record[f"class_{target_class}_pixels"] = pixels
            if pixels >= min_pixels_per_image:
                groups[(event_id, target_class)].append(index)
        image_class_pixels.append(record)

    eligible = {
        key: values for key, values in groups.items() if len(values) >= min_images_per_group
    }
    sampler = EventClassBalancedSampler(
        eligible, num_samples=len(dataset), seed=seed
    )
    info = {
        "strategy": "event_class_balanced",
        "sampling_unit": "event_class",
        "num_samples_per_epoch": len(dataset),
        "min_pixels_per_image": int(min_pixels_per_image),
        "min_images_per_group": int(min_images_per_group),
        "classes": classes,
        "eligible_group_count": len(eligible),
        "excluded_groups": [
            {"event_id": key[0], "target_class": key[1], "image_count": len(values)}
            for key, values in sorted(groups.items())
            if key not in eligible
        ],
        "eligible_groups": [
            {"event_id": key[0], "target_class": key[1], "image_count": len(values)}
            for key, values in sorted(eligible.items())
        ],
    }
    return sampler, info


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


def _disaster_loss_for_batch(
    disaster_logits: torch.Tensor | None,
    batch: dict[str, Any],
    criterion: torch.nn.Module | None,
    device: torch.device,
    label_map: dict[str, str] | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if disaster_logits is None:
        if criterion is not None:
            raise RuntimeError("metadata multitask config requires disaster_logits from the model")
        return None, None
    if criterion is None:
        raise RuntimeError("model returned disaster_logits without a configured disaster criterion")
    targets = encode_disaster_types(
        [str(value) for value in batch["disaster_type"]],
        sample_ids=[str(value) for value in batch["id"]],
        label_map=label_map,
        device=device,
    )
    return criterion(disaster_logits, targets), targets


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
    disaster_criterion: torch.nn.Module | None = None,
    disaster_weight: float = 0.0,
    disaster_label_map: dict[str, str] | None = None,
    batch_iterator: Iterator[dict[str, Any]] | None = None,
    collect_metrics: bool = True,
    collect_loss: bool = True,
) -> dict[str, float]:
    model.train()
    meter = Stage2V2MeterBundle() if collect_metrics else None
    loss_sum = 0.0
    sample_count = 0
    damage_loss_sum = 0.0
    disaster_loss_sum = 0.0
    disaster_meter = DisasterClassificationMeter() if collect_metrics else None
    disaster_sample_count = 0
    event_meters = GroupedV2Meters(["event_id"]) if collect_metrics else None
    gradient_metrics: dict[str, float] = {}
    accumulation_steps = max(1, int(gradient_accumulation_steps))
    if batch_iterator is not None:
        if max_batches is None:
            raise ValueError("batch_iterator requires an explicit max_batches")
        total_batches = int(max_batches)
        batches = itertools.islice(batch_iterator, total_batches)
    else:
        total_batches = len(loader) if max_batches is None else min(len(loader), int(max_batches))
        batches = itertools.islice(loader, total_batches)
    optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(
        tqdm(
            batches,
            total=total_batches,
            desc="train",
            leave=False,
            disable=batch_iterator is not None,
        )
    ):
        image = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            output = model(image)
            logits, disaster_logits = unpack_model_output(output)
            damage_loss = criterion(logits, target)
            disaster_loss, disaster_targets = _disaster_loss_for_batch(
                disaster_logits,
                batch,
                disaster_criterion,
                device,
                disaster_label_map,
            )
            loss = damage_loss
            if disaster_loss is not None:
                loss = loss + float(disaster_weight) * disaster_loss
        if collect_metrics and batch_index == 0 and disaster_loss is not None:
            gradient_metrics = gradient_interaction(
                damage_loss,
                disaster_loss,
                diagnostic_shared_parameters(model),
                disaster_weight=disaster_weight,
            )
        group_start = (batch_index // accumulation_steps) * accumulation_steps
        current_group_size = min(accumulation_steps, total_batches - group_start)
        scaler.scale(loss / max(current_group_size, 1)).backward()
        processed_batches = batch_index + 1
        if processed_batches % accumulation_steps == 0 or processed_batches == total_batches:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        if collect_loss or collect_metrics:
            batch_size = image.shape[0]
            loss_sum += float(loss.item()) * batch_size
            sample_count += batch_size
        if not collect_metrics:
            continue
        assert meter is not None and disaster_meter is not None and event_meters is not None
        damage_loss_sum += float(damage_loss.item()) * batch_size
        if disaster_loss is not None and disaster_logits is not None and disaster_targets is not None:
            disaster_loss_sum += float(disaster_loss.item()) * batch_size
            disaster_sample_count += batch_size
            disaster_meter.update(disaster_logits, disaster_targets)
        _bundle_update(meter, logits, target, batch["prior"], gate_threshold)
        grade_batch = logits.detach().argmax(dim=1).cpu().numpy().astype(np.uint8) + 1
        target_batch = target.detach().cpu().numpy().astype(np.uint8)
        support_batch = batch["prior"][:, 0].numpy() >= gate_threshold
        for sample_index in range(batch_size):
            event_meters.update(
                grade_batch[sample_index],
                target_batch[sample_index],
                support_batch[sample_index],
                {"event_id": str(batch["event_id"][sample_index])},
            )
    if not collect_metrics:
        return {"loss": loss_sum / max(sample_count, 1)} if collect_loss else {}
    assert meter is not None and disaster_meter is not None and event_meters is not None
    metrics = meter.compute()
    event_rows = event_meters.rows("event_id")
    if event_rows:
        metrics["event_macro_bo_f1"] = float(
            np.mean([row["building_only_macro_f1_3class"] for row in event_rows])
        )
        present_damage = []
        for row in event_rows:
            values = [
                row[f"building_only_f1_{name}"]
                for name in ("damaged", "destroyed")
                if row.get(f"building_only_support_{name}", 0) > 0
            ]
            if values:
                present_damage.append(float(np.mean(values)))
        metrics["event_macro_bo_damage_f1"] = float(np.mean(present_damage)) if present_damage else 0.0
    metrics["loss"] = loss_sum / max(sample_count, 1)
    metrics["damage_loss"] = damage_loss_sum / max(sample_count, 1)
    if disaster_sample_count:
        metrics["disaster_loss"] = disaster_loss_sum / disaster_sample_count
        metrics.update(
            {
                f"disaster_classification_{key}": value
                for key, value in disaster_meter.compute().items()
            }
        )
    metrics.update(gradient_metrics)
    return metrics


def step_model_epoch(model: torch.nn.Module) -> None:
    """Advance an optional architecture-owned epoch schedule."""

    hook = getattr(model, "step_epoch", None)
    if callable(hook):
        hook()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    amp: bool,
    gate_threshold: float,
    max_batches: int | None = None,
    disaster_criterion: torch.nn.Module | None = None,
    disaster_weight: float = 0.0,
) -> dict[str, float]:
    model.eval()
    meter = Stage2V2MeterBundle()
    loss_sum = 0.0
    sample_count = 0
    damage_loss_sum = 0.0
    disaster_loss_sum = 0.0
    disaster_meter = DisasterClassificationMeter()
    disaster_sample_count = 0
    event_meters = GroupedV2Meters(["event_id"])
    for batch_index, batch in enumerate(tqdm(loader, desc="val", leave=False)):
        if max_batches is not None and batch_index >= max_batches:
            break
        image = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            output = model(image)
            logits, disaster_logits = unpack_model_output(output)
            damage_loss = criterion(logits, target)
            disaster_loss, disaster_targets = _disaster_loss_for_batch(
                disaster_logits, batch, disaster_criterion, device, None
            )
            loss = damage_loss
            if disaster_loss is not None:
                loss = loss + float(disaster_weight) * disaster_loss
        batch_size = image.shape[0]
        loss_sum += float(loss.item()) * batch_size
        damage_loss_sum += float(damage_loss.item()) * batch_size
        if disaster_loss is not None and disaster_logits is not None and disaster_targets is not None:
            disaster_loss_sum += float(disaster_loss.item()) * batch_size
            disaster_sample_count += batch_size
            disaster_meter.update(disaster_logits, disaster_targets)
        sample_count += batch_size
        _bundle_update(meter, logits, target, batch["prior"], gate_threshold)
        grade_batch = logits.detach().argmax(dim=1).cpu().numpy().astype(np.uint8) + 1
        target_batch = target.detach().cpu().numpy().astype(np.uint8)
        support_batch = batch["prior"][:, 0].numpy() >= gate_threshold
        for sample_index in range(batch_size):
            event_meters.update(
                grade_batch[sample_index],
                target_batch[sample_index],
                support_batch[sample_index],
                {"event_id": str(batch["event_id"][sample_index])},
            )
    metrics = meter.compute()
    event_rows = event_meters.rows("event_id")
    if event_rows:
        metrics["event_macro_bo_f1"] = float(
            np.mean([row["building_only_macro_f1_3class"] for row in event_rows])
        )
        present_damage = []
        for row in event_rows:
            values = [
                row[f"building_only_f1_{name}"]
                for name in ("damaged", "destroyed")
                if row.get(f"building_only_support_{name}", 0) > 0
            ]
            if values:
                present_damage.append(float(np.mean(values)))
        metrics["event_macro_bo_damage_f1"] = float(np.mean(present_damage)) if present_damage else 0.0
    metrics["loss"] = loss_sum / max(sample_count, 1)
    metrics["damage_loss"] = damage_loss_sum / max(sample_count, 1)
    if disaster_sample_count:
        metrics["disaster_loss"] = disaster_loss_sum / disaster_sample_count
        metrics.update(
            {
                f"disaster_classification_{key}": value
                for key, value in disaster_meter.compute().items()
            }
        )
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
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--eval-steps", nargs="+", type=int)
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
    deterministic = bool(train_cfg.get("deterministic", False))
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)

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
    max_optimizer_steps_value = args.max_optimizer_steps or train_cfg.get("max_optimizer_steps") or train_cfg.get("final_step")
    fixed_step_mode = max_optimizer_steps_value is not None
    epochs = int(max_optimizer_steps_value) if fixed_step_mode else int(args.epochs or train_cfg.get("epochs", 100))
    configured_eval_steps = args.eval_steps or train_cfg.get("eval_steps", [])
    eval_steps = sorted({int(value) for value in configured_eval_steps if int(value) > 0})
    if fixed_step_mode:
        eval_steps = sorted(set(eval_steps) | {epochs})
    eval_step_set = set(eval_steps)
    limit, train_max_batches, eval_max_batches = resolve_batch_limits(
        smoke_test=bool(args.smoke_test),
        overfit_batches=args.overfit_batches,
        fixed_step_mode=fixed_step_mode,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    if args.smoke_test:
        epochs = 1

    data_root = Path(args.data_root)
    dataset_cfg = dict(cfg.get("dataset", {}))
    full_train_rows = read_jsonl(
        resolve_manifest(data_root, str(dataset_cfg["train_manifest"]))
    )
    full_val_rows = read_jsonl(resolve_manifest(data_root, str(dataset_cfg["val_manifest"])))
    train_ds = make_dataset(
        cfg, data_root, "train", True, seed, limit, cache_items=bool(args.overfit_batches)
    )
    val_ds = make_dataset(cfg, data_root, "val", False, seed, limit if limit is not None else None)
    write_json(output_dir / "permutations" / "train.json", train_ds.permutation_info)
    write_json(output_dir / "permutations" / "val.json", val_ds.permutation_info)
    class_weights, class_weight_info = compute_class_weights(train_ds, cfg)
    write_json(output_dir / "class_weights.json", class_weight_info)

    multitask_cfg = dict(cfg.get("multitask", {}))
    multitask_enabled = bool(multitask_cfg.get("enabled", False))
    disaster_weight = float(multitask_cfg.get("disaster_loss_weight", 0.0))
    disaster_criterion: torch.nn.Module | None = None
    disaster_label_map: dict[str, str] | None = None
    multitask_info: dict[str, Any] = {"enabled": multitask_enabled}
    if multitask_enabled:
        validate_disaster_classes(multitask_cfg.get("disaster_classes"))
        train_counts = disaster_class_counts(full_train_rows)
        computed_weights = inverse_sqrt_class_weights(train_counts)
        configured_weights = [float(value) for value in multitask_cfg.get("class_weights", [])]
        if len(configured_weights) != len(DISASTER_CLASSES):
            raise ValueError("multitask.class_weights must contain seven frozen values")
        if not np.allclose(configured_weights, computed_weights, rtol=0.0, atol=1e-4):
            raise ValueError(
                "configured disaster class weights do not match inverse-sqrt R4 train counts: "
                f"configured={configured_weights}, computed={computed_weights}"
            )
        encode_disaster_types(
            [str(row.get("disaster_type", "")) for row in full_val_rows]
        )
        label_map_value = multitask_cfg.get("train_label_map")
        if label_map_value:
            label_map_path = Path(str(label_map_value))
            if not label_map_path.is_absolute():
                label_map_path = SRC_ROOT.parent / label_map_path
        else:
            label_map_path = None
        disaster_label_map, label_map_info = load_label_map(label_map_path)
        if disaster_label_map is not None:
            true_by_id = {
                str(row["id"]): str(row.get("disaster_type", "")) for row in full_train_rows
            }
            if set(disaster_label_map) != set(true_by_id):
                missing = sorted(set(true_by_id) - set(disaster_label_map))
                extra = sorted(set(disaster_label_map) - set(true_by_id))
                raise ValueError(
                    f"shuffled label map sample mismatch: missing={missing[:5]}, extra={extra[:5]}"
                )
            if any(disaster_label_map[key] == value for key, value in true_by_id.items()):
                raise ValueError("shuffled disaster label map is not fully deranged")
            if Counter(disaster_label_map.values()) != Counter(true_by_id.values()):
                raise ValueError("shuffled disaster label map does not preserve class counts")
        disaster_criterion = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(configured_weights, dtype=torch.float32, device=device)
        )
        multitask_info = {
            "enabled": True,
            "disaster_classes": list(DISASTER_CLASSES),
            "disaster_loss_weight": disaster_weight,
            "class_weight_strategy": "inverse_sqrt_frequency_normalized_mean_1",
            "class_counts": train_counts,
            "computed_class_weights": computed_weights,
            "configured_class_weights": configured_weights,
            "train_label_map": label_map_info,
            "validation_uses_true_labels": True,
        }
    elif disaster_weight != 0.0:
        raise ValueError("multitask disaster_loss_weight requires multitask.enabled=true")
    write_json(output_dir / "multitask_info.json", multitask_info)

    model = build_model(cfg, no_pretrained=args.no_pretrained).to(device)
    initial_model_sha256 = model_state_sha256(model)
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

    sampling_strategy = str(train_cfg.get("sampling_strategy", "default")).lower()
    sampler_info: dict[str, Any] = {"strategy": sampling_strategy}
    if sampling_strategy == "event_class_balanced":
        if float(train_cfg.get("disaster_sampling_alpha", 0.0)) > 0:
            raise ValueError("event_class_balanced is incompatible with disaster_sampling_alpha > 0")
        sampler, sampler_info = build_event_class_sampler(
            train_ds,
            seed=seed,
            min_pixels_per_image=int(train_cfg.get("event_class_min_pixels_per_image", 64)),
            min_images_per_group=int(train_cfg.get("event_class_min_images_per_group", 2)),
            classes=[int(value) for value in train_cfg.get("event_class_classes", [1, 2, 3])],
        )
    elif sampling_strategy == "capped_event":
        if float(train_cfg.get("disaster_sampling_alpha", 0.0)) > 0:
            raise ValueError("capped_event is incompatible with disaster_sampling_alpha > 0")
        sampler, sampler_info = build_capped_event_sampler(
            train_ds,
            alpha=float(train_cfg.get("event_sampling_alpha", 0.5)),
            min_weight=float(train_cfg.get("event_sampling_min_weight", 0.5)),
            max_weight=float(train_cfg.get("event_sampling_max_weight", 4.0)),
            seed=seed,
        )
    elif sampling_strategy in {"default", "none", "disaster_balanced"}:
        sampler = build_sampler(train_ds, float(train_cfg.get("disaster_sampling_alpha", 0.0)), seed)
    else:
        raise ValueError(f"unknown sampling_strategy: {sampling_strategy}")
    write_json(output_dir / "sampler.json", sampler_info)
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
    fixed_batch_iterator = (
        CyclingDataIterator(train_loader)
        if fixed_step_mode and not args.smoke_test and not args.overfit_batches
        else None
    )
    gate_threshold = float(dict(cfg.get("dataset", {})).get("gate_threshold", 0.6))
    patience = int(args.early_stopping_patience if args.early_stopping_patience is not None else train_cfg.get("early_stopping_patience", 30))
    min_epochs = int(args.early_stopping_min_epochs if args.early_stopping_min_epochs is not None else train_cfg.get("early_stopping_min_epochs", 40))
    early_window = int(train_cfg.get("early_stopping_window", 3))
    progress_interval_steps = max(1, int(train_cfg.get("progress_interval_steps", 1000)))
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
            "fixed_step_mode": fixed_step_mode,
            "max_optimizer_steps": epochs if fixed_step_mode else None,
            "eval_steps": eval_steps,
            "fixed_step_iterator_policy": (
                "persistent_full_pass_cycle" if fixed_batch_iterator is not None else None
            ),
            "fixed_step_training_diagnostics_policy": (
                "full_at_validation_loss_at_heartbeat"
                if fixed_batch_iterator is not None
                else None
            ),
            "train_max_batches_per_step_or_epoch": train_max_batches,
            "validation_batch_limit": eval_max_batches,
            "amp": amp,
            "deterministic": deterministic,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "seed": seed,
            "gate_threshold": gate_threshold,
            "sar_shuffle_mode": train_ds.permutation_info["mode"],
            "disaster_sampling_alpha": float(train_cfg.get("disaster_sampling_alpha", 0.0)),
            "early_stopping": {"patience": patience, "min_epochs": min_epochs, "window": early_window},
            "checkpoint_policy": checkpoint_policy,
            "save_last_checkpoint": save_last_checkpoint,
            "progress_interval_steps": progress_interval_steps,
            "initial_model_state_sha256": initial_model_sha256,
            "multitask": multitask_info,
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
            max_batches=train_max_batches,
            gradient_accumulation_steps=gradient_accumulation_steps,
            disaster_criterion=disaster_criterion,
            disaster_weight=disaster_weight,
            disaster_label_map=disaster_label_map,
            batch_iterator=fixed_batch_iterator,
        )
        step_model_epoch(model)
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            amp,
            gate_threshold,
            max_batches=eval_max_batches,
            disaster_criterion=disaster_criterion,
            disaster_weight=disaster_weight,
        )
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
    run_loop_started = time.monotonic()

    def fixed_step_progress(step: int, train_loss: float) -> dict[str, float | int]:
        elapsed = max(time.monotonic() - run_loop_started, 1e-9)
        steps_per_second = step / elapsed
        return {
            "optimizer_step": step,
            "max_optimizer_steps": epochs,
            "train_loss": train_loss,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "elapsed_run_loop_seconds": elapsed,
            "average_optimizer_steps_per_second_including_validation": steps_per_second,
            "estimated_remaining_run_loop_seconds": max(
                0.0, (epochs - step) / steps_per_second
            ),
        }

    for epoch in range(1, epochs + 1):
        collect_train_metrics, collect_train_loss = resolve_training_diagnostics(
            fixed_step_mode=fixed_step_mode,
            optimizer_step=epoch,
            eval_steps=eval_step_set,
            progress_interval_steps=progress_interval_steps,
        )
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            amp,
            gate_threshold,
            train_max_batches,
            gradient_accumulation_steps=gradient_accumulation_steps,
            disaster_criterion=disaster_criterion,
            disaster_weight=disaster_weight,
            disaster_label_map=disaster_label_map,
            batch_iterator=fixed_batch_iterator,
            collect_metrics=collect_train_metrics,
            collect_loss=collect_train_loss,
        )
        step_model_epoch(model)
        if fixed_step_mode and epoch not in eval_step_set:
            if scheduler is not None:
                scheduler.step()
            if epoch % progress_interval_steps == 0:
                progress = fixed_step_progress(epoch, train_metrics["loss"])
                write_json(output_dir / "latest_progress.json", progress)
                print(json.dumps(progress, sort_keys=True), flush=True)
            continue
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            amp,
            gate_threshold,
            eval_max_batches,
            disaster_criterion=disaster_criterion,
            disaster_weight=disaster_weight,
        )
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
        if fixed_step_mode:
            save_checkpoint(
                output_dir / "checkpoints" / f"step_{epoch:06d}.pth",
                model,
                optimizer,
                scheduler,
                cfg,
                epoch,
                seed,
                best_metrics,
                class_weight_info,
            )
            progress = fixed_step_progress(epoch, train_metrics["loss"])
            write_json(output_dir / "latest_progress.json", progress)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "val_loss": val_metrics["loss"],
                    "val_bo_grade_macro_f1": row[primary_name],
                    "val_bo_damage_macro_f1": row["val_building_only_damage_macro_f1"],
                    "train_disaster_loss": train_metrics.get("disaster_loss"),
                    "val_disaster_present_macro_f1": val_metrics.get(
                        "disaster_classification_present_class_macro_f1"
                    ),
                    "task_grad_cosine": train_metrics.get("task_grad_cosine"),
                    "smoothed_primary": smoothed,
                    "epochs_without_improvement": epochs_without_improvement,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if (not fixed_step_mode) and patience > 0 and epoch >= min_epochs and epochs_without_improvement >= patience:
            stopped_early = True
            break

    completion = {
        "status": "completed",
        "epochs_completed": None if fixed_step_mode else len(history),
        "evaluation_count": len(history),
        "optimizer_steps_completed": epochs if fixed_step_mode else None,
        "selection_policy": "fixed_steps_inner_only" if fixed_step_mode else "validation_checkpoint",
        "stopped_early": stopped_early,
        "checkpoint_policy": checkpoint_policy,
        "save_last_checkpoint": save_last_checkpoint,
        "best_metrics": best_metrics,
    }
    write_json(output_dir / "completed.json", completion)
    print(json.dumps(completion, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
