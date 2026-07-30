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
from stage2.common import CLASS_NAMES, write_csv, write_json  # noqa: E402
from stage2.datasets import Stage2DamageDataset  # noqa: E402
from stage2.metrics import GroupedStage2Meters, Stage2DamageMeter  # noqa: E402


COLORS = {
    0: np.array([0, 0, 0], dtype=np.uint8),
    1: np.array([0, 170, 90], dtype=np.uint8),
    2: np.array([245, 190, 40], dtype=np.uint8),
    3: np.array([220, 50, 45], dtype=np.uint8),
}


def dataset_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(cfg.get("dataset", {}))


def make_dataset(cfg: dict[str, Any], data_root: Path, manifest: str, limit: int | None = None) -> Stage2DamageDataset:
    dcfg = dataset_cfg(cfg)
    return Stage2DamageDataset(
        data_root=data_root,
        manifest=manifest,
        train=False,
        prior_type=str(dcfg.get("prior_type", "predicted")),
        input_mode=str(dcfg.get("input_mode", "sar_prior")),
        crop_size=None,
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
    pred_rgb = colorize(pred)
    target_rgb = colorize(target)
    error = np.zeros_like(pred_rgb)
    error[(pred == target) & (target > 0)] = np.array([0, 180, 80], dtype=np.uint8)
    error[(pred != target) & (pred > 0)] = np.array([230, 60, 45], dtype=np.uint8)
    error[(pred != target) & (target > 0)] = np.array([40, 110, 230], dtype=np.uint8)
    tile = np.concatenate([sar_rgb, prior_rgb, target_rgb, pred_rgb, error], axis=1)
    Image.fromarray(tile).save(path)


def sample_metric_row(metadata: dict[str, Any], pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    meter = Stage2DamageMeter()
    meter.update_prediction(pred, target)
    row = {
        "id": str(metadata.get("id", "")),
        "split": str(metadata.get("split", "")),
        "event_id": str(metadata.get("event_id", "")),
        "disaster_type": str(metadata.get("disaster_type", "")),
        "country_or_region": str(metadata.get("country_or_region", "")),
        "qc_label": str(metadata.get("qc_label", "")),
    }
    row.update(meter.compute())
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-previews", type=int, default=32)
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    dcfg = dataset_cfg(cfg)
    manifest = args.manifest or str(dcfg[f"{args.split}_manifest"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg, no_pretrained=True).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    ds = make_dataset(cfg, Path(args.data_root), manifest=manifest, limit=args.limit)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    meter = Stage2DamageMeter()
    grouped = GroupedStage2Meters(["disaster_type", "country_or_region", "event_id"])
    sample_rows: list[dict[str, Any]] = []
    preview_count = 0
    if args.save_predictions:
        (output_dir / "predictions").mkdir(parents=True, exist_ok=True)

    for batch in tqdm(loader, desc=f"test_{args.split}"):
        image = batch["image"].to(device, non_blocking=True)
        target_t = batch["mask"]
        logits = model(image)
        pred_batch = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.uint8)
        target_batch = target_t.numpy().astype(np.uint8)
        meter.update_prediction(pred_batch, target_batch)
        for i in range(pred_batch.shape[0]):
            metadata = {
                "id": batch["id"][i],
                "split": batch["split"][i],
                "event_id": batch["event_id"][i],
                "disaster_type": batch["disaster_type"][i],
                "country_or_region": batch["country_or_region"][i],
                "qc_label": batch["qc_label"][i],
            }
            pred = pred_batch[i]
            target = target_batch[i]
            grouped.update(pred, target, metadata)
            sample_rows.append(sample_metric_row(metadata, pred, target))
            if args.save_predictions:
                Image.fromarray(pred).save(output_dir / "predictions" / f"{metadata['id']}.png")
            if preview_count < args.max_previews:
                sar = batch["sar"][i, 0].numpy()
                prior = batch["prior"][i, 0].numpy()
                save_preview(output_dir / "previews" / f"{metadata['id']}.png", sar, prior, pred, target)
                preview_count += 1

    metrics = meter.compute()
    metrics.update(
        {
            "sample_count": len(ds),
            "manifest": str(manifest),
            "checkpoint": str(args.checkpoint),
            "split": args.split,
            "class_names": CLASS_NAMES,
        }
    )
    write_json(output_dir / "metrics.json", metrics)
    write_csv(output_dir / "sample_metrics.csv", sample_rows)
    write_csv(output_dir / "per_disaster_metrics.csv", grouped.rows("disaster_type"))
    write_csv(output_dir / "per_region_metrics.csv", grouped.rows("country_or_region"))
    write_csv(output_dir / "per_event_metrics.csv", grouped.rows("event_id"))
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
