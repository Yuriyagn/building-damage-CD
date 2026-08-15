#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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

from config import load_config  # noqa: E402
from models.build_model import build_model  # noqa: E402
from models.metadata_multitask import unpack_model_output  # noqa: E402
from stage2.common import read_jsonl, resolve_manifest, write_csv, write_json  # noqa: E402
from stage2.datasets_v2 import Stage2V2Dataset  # noqa: E402
from stage2.metadata_multitask import (  # noqa: E402
    DisasterClassificationMeter,
    encode_disaster_types,
)
from stage2.metrics_v2 import (  # noqa: E402
    CCSurrogateMeter,
    GroupedV2Meters,
    Stage2V2MeterBundle,
    summarize_event_generalization,
)


COLORS = {
    0: np.array([0, 0, 0], dtype=np.uint8),
    1: np.array([0, 170, 90], dtype=np.uint8),
    2: np.array([245, 190, 40], dtype=np.uint8),
    3: np.array([220, 50, 45], dtype=np.uint8),
}


def event_sets(cfg: dict[str, Any], data_root: Path) -> tuple[set[str], set[str]]:
    dcfg = dict(cfg.get("dataset", {}))
    train_rows = read_jsonl(resolve_manifest(data_root, str(dcfg["train_manifest"])))
    val_rows = read_jsonl(resolve_manifest(data_root, str(dcfg["val_manifest"])))
    return (
        {str(row.get("event_id", "") or "unknown") for row in train_rows},
        {str(row.get("event_id", "") or "unknown") for row in val_rows},
    )


def familiarity(event_id: str, train_events: set[str], val_events: set[str]) -> str:
    if event_id in train_events:
        return "train_seen"
    if event_id in val_events:
        return "val_seen_only"
    return "globally_unseen"


def make_dataset(
    cfg: dict[str, Any], data_root: Path, split: str, seed: int, limit: int | None
) -> Stage2V2Dataset:
    dcfg = dict(cfg.get("dataset", {}))
    split_offset = {"train": 0, "val": 10_000, "test": 20_000}[split]
    permutation_seed = int(dcfg.get("sar_shuffle_seed", seed))
    return Stage2V2Dataset(
        data_root=data_root,
        manifest=str(dcfg[f"{split}_manifest"]),
        train=False,
        prior_type=str(dcfg.get("prior_type", "predicted")),
        input_mode=str(dcfg.get("input_mode", "sar_prior")),
        crop_size=None,
        ogsr_feature_root=dcfg.get("ogsr_feature_root"),
        ogsr_feature_keys=dcfg.get("ogsr_feature_keys"),
        sar_shuffle_mode=str(dcfg.get("sar_shuffle_mode", "paired")),
        sar_shuffle_seed=permutation_seed + split_offset,
        sar_singleton_policy=str(dcfg.get("sar_singleton_policy", "error")),
        limit=limit,
    )


def colorize(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for value, color in COLORS.items():
        out[mask == value] = color
    return out


def save_preview(path: Path, sar: np.ndarray, prior: np.ndarray, pred: np.ndarray, target: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sar_rgb = np.repeat((np.clip(sar, 0, 1) * 255).astype(np.uint8)[..., None], 3, axis=2)
    prior_rgb = np.repeat((np.clip(prior, 0, 1) * 255).astype(np.uint8)[..., None], 3, axis=2)
    correct = np.zeros((*pred.shape, 3), dtype=np.uint8)
    correct[(pred == target) & (target > 0)] = np.array([0, 180, 80], dtype=np.uint8)
    correct[(pred != target) & (pred > 0)] = np.array([230, 60, 45], dtype=np.uint8)
    correct[(pred != target) & (target > 0)] = np.array([40, 110, 230], dtype=np.uint8)
    tile = np.concatenate([sar_rgb, prior_rgb, colorize(target), colorize(pred), correct], axis=1)
    Image.fromarray(tile).save(path)


def decode_grade_logits(logits: torch.Tensor, damage_margin_threshold: float | None = None) -> np.ndarray:
    if damage_margin_threshold is None:
        return torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8) + 1

    damage_logits = logits[:, 1:3].detach()
    damage_choice = torch.argmax(damage_logits, dim=1).cpu().numpy().astype(np.uint8) + 2
    damage_margin = (torch.max(damage_logits, dim=1).values - logits[:, 0].detach()).float().cpu().numpy()
    return np.where(damage_margin >= float(damage_margin_threshold), damage_choice, 1).astype(np.uint8)


def confusion_flat(pred: np.ndarray, target: np.ndarray, building_only: bool) -> str:
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    if building_only:
        keep = target_flat > 0
        encoded = 3 * (target_flat[keep] - 1).astype(np.int64) + (pred_flat[keep] - 1).astype(np.int64)
        confusion = np.bincount(encoded, minlength=9).reshape(3, 3)
    else:
        encoded = 4 * target_flat.astype(np.int64) + pred_flat.astype(np.int64)
        confusion = np.bincount(encoded, minlength=16).reshape(4, 4)
    return json.dumps(confusion.astype(int).reshape(-1).tolist(), separators=(",", ":"))


def sample_row(
    metadata: dict[str, Any], grade: np.ndarray, pred: np.ndarray, target: np.ndarray, support: np.ndarray
) -> dict[str, Any]:
    meter = Stage2V2MeterBundle()
    meter.update(grade, target, support)
    return {
        **metadata,
        **meter.compute(),
        "bo_confusion_3x3": confusion_flat(grade, target, building_only=True),
        "predicted_gate_confusion_4x4": confusion_flat(pred, target, building_only=False),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage-2 v2 building-only models")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-previews", type=int, default=32)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--damage-margin-threshold", type=float)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    seed = int(checkpoint.get("seed", dict(cfg.get("train", {})).get("seed", 42)))
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is unavailable; pass --allow-cpu only for a diagnostic test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, no_pretrained=True).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    data_root = Path(args.data_root)
    dataset = make_dataset(cfg, data_root, args.split, seed, args.limit)
    write_json(output_dir / "permutation.json", dataset.permutation_info)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    train_events, val_events = event_sets(cfg, data_root)
    gate_threshold = float(dict(cfg.get("dataset", {})).get("gate_threshold", 0.6))
    amp = bool(dict(cfg.get("train", {})).get("amp", True)) and device.type == "cuda"
    meter = Stage2V2MeterBundle()
    grouped = GroupedV2Meters(["disaster_type", "country_or_region", "event_id", "event_familiarity"])
    cc_surrogate = CCSurrogateMeter()
    disaster_meter = DisasterClassificationMeter()
    disaster_prediction_count = 0
    sample_rows: list[dict[str, Any]] = []
    preview_count = 0
    if args.save_predictions:
        (output_dir / "predictions").mkdir(parents=True, exist_ok=True)

    for batch in tqdm(loader, desc=f"test_{args.split}"):
        image = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            output = model(image)
            logits, disaster_logits = unpack_model_output(output)
        if disaster_logits is not None:
            disaster_targets = encode_disaster_types(
                [str(value) for value in batch["disaster_type"]], device=device
            )
            disaster_meter.update(disaster_logits, disaster_targets)
            disaster_prediction_count += int(disaster_targets.numel())
        grade_batch = decode_grade_logits(logits, args.damage_margin_threshold)
        target_batch = batch["mask"].numpy().astype(np.uint8)
        support_batch = batch["prior"][:, 0].numpy() >= gate_threshold
        pred_batch = np.where(support_batch, grade_batch, 0).astype(np.uint8)
        meter.update(grade_batch, target_batch, support_batch)
        for index in range(grade_batch.shape[0]):
            event_id = str(batch["event_id"][index])
            metadata: dict[str, Any] = {
                "id": str(batch["id"][index]),
                "split": str(batch["split"][index]),
                "event_id": event_id,
                "disaster_type": str(batch["disaster_type"][index]),
                "country_or_region": str(batch["country_or_region"][index]),
                "qc_label": str(batch["qc_label"][index]),
                "event_familiarity": familiarity(event_id, train_events, val_events),
                "sar_source_id": str(batch["sar_source_id"][index]),
                "sar_is_paired": bool(batch["sar_is_paired"][index]),
            }
            grade = grade_batch[index]
            target = target_batch[index]
            support = support_batch[index]
            pred = pred_batch[index]
            grouped.update(grade, target, support, metadata)
            if args.damage_margin_threshold is None:
                cc_surrogate.update(logits[index], target)
            else:
                cc_surrogate.update_prediction(grade, target)
            sample_rows.append(sample_row(metadata, grade, pred, target, support))
            safe_id = f"{args.split}__{metadata['id']}"
            if args.save_predictions:
                Image.fromarray(pred).save(output_dir / "predictions" / f"{safe_id}.png")
            if preview_count < args.max_previews:
                save_preview(
                    output_dir / "previews" / f"{safe_id}.png",
                    batch["sar"][index, 0].numpy(),
                    batch["prior"][index, 0].numpy(),
                    pred,
                    target,
                )
                preview_count += 1

    metrics: dict[str, Any] = meter.compute()
    metrics.update(cc_surrogate.compute())
    if disaster_prediction_count:
        metrics.update(
            {
                f"disaster_classification_{key}": value
                for key, value in disaster_meter.compute().items()
            }
        )
    metrics.update(
        {
            "sample_count": len(dataset),
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "split": args.split,
            "seed": seed,
            "gate_threshold": gate_threshold,
            "decode_mode": "argmax" if args.damage_margin_threshold is None else "damage_margin_threshold",
            "damage_margin_threshold": args.damage_margin_threshold,
            "cc_surrogate_decode": (
                "raw_logits_mean_argmax"
                if args.damage_margin_threshold is None
                else "thresholded_grade_majority"
            ),
            "sar_shuffle_mode": dataset.permutation_info["mode"],
        }
    )
    per_event_rows = grouped.rows("event_id")
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "event_generalization.json", summarize_event_generalization(per_event_rows))
    write_csv(output_dir / "sample_metrics.csv", sample_rows)
    write_csv(output_dir / "per_disaster_metrics.csv", grouped.rows("disaster_type"))
    write_csv(output_dir / "per_region_metrics.csv", grouped.rows("country_or_region"))
    write_csv(output_dir / "per_event_metrics.csv", per_event_rows)
    write_csv(output_dir / "per_event_familiarity_metrics.csv", grouped.rows("event_familiarity"))
    print(json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
