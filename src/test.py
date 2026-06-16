#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import load_config
from datasets.disasterm3_stage1_dataset import DisasterM3Stage1Dataset
from metrics import BinarySegmentationMeter, boundary_f1, group_metrics
from models.build_model import build_model


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


def save_preview(path: Path, image_t: torch.Tensor, pred: np.ndarray, target: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = image_t.detach().cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = np.clip((image * std + mean) * 255.0, 0, 255).astype(np.uint8)
    overlay = image.copy()
    overlay[target.astype(bool)] = (0.5 * overlay[target.astype(bool)] + np.array([0, 220, 0]) * 0.5).astype(np.uint8)
    overlay[pred.astype(bool)] = (0.5 * overlay[pred.astype(bool)] + np.array([255, 80, 0]) * 0.5).astype(np.uint8)
    Image.fromarray(overlay).save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-previews", type=int, default=32)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg, no_pretrained=True).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    ds = DisasterM3Stage1Dataset(
        data_root=args.data_root,
        manifest=args.manifest,
        train=False,
        crop_size=None,
        normalize=str(cfg.get("input", {}).get("normalize", "imagenet")),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    meter = BinarySegmentationMeter(threshold=args.threshold)
    sample_rows = []
    boundary_scores = []

    for idx, batch in enumerate(tqdm(loader, desc="test")):
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        logits = model(image)
        pred = (torch.sigmoid(logits) >= args.threshold).detach().cpu().numpy()[0, 0].astype(np.uint8)
        true = mask.detach().cpu().numpy()[0, 0].astype(np.uint8)

        sample_meter = BinarySegmentationMeter(threshold=args.threshold)
        sample_meter.update_binary(pred, true)
        sample_metrics = sample_meter.compute()
        b_f1 = boundary_f1(pred, true)
        boundary_scores.append(b_f1)
        meter.update_binary(pred, true)

        row = {
            "id": batch["id"][0],
            "event_id": batch["event_id"][0],
            "disaster_type": batch["disaster_type"][0],
            "country_or_region": batch["country_or_region"][0],
            "qc_label": batch["qc_label"][0],
            "boundary_f1": b_f1,
            **sample_metrics,
        }
        sample_rows.append(row)
        if idx < args.max_previews:
            save_preview(output_dir / "previews" / f"{row['id']}.png", batch["image"][0], pred, true)

    metrics = meter.compute()
    metrics["boundary_f1"] = float(np.mean(boundary_scores)) if boundary_scores else 0.0
    metrics["sample_count"] = len(sample_rows)
    metrics["manifest"] = str(args.manifest)
    metrics["checkpoint"] = str(args.checkpoint)
    write_json(output_dir / "metrics.json", metrics)
    write_csv(output_dir / "sample_metrics.csv", sample_rows)
    write_csv(output_dir / "per_disaster_metrics.csv", group_metrics(sample_rows, "disaster_type"))
    write_csv(output_dir / "per_region_metrics.csv", group_metrics(sample_rows, "country_or_region"))
    write_csv(output_dir / "per_event_metrics.csv", group_metrics(sample_rows, "event_id"))
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
