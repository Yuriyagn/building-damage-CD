#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_ROOT))

from config import load_config  # noqa: E402
from datasets.disasterm3_stage1_dataset import DisasterM3Stage1Dataset  # noqa: E402
from metrics import BinarySegmentationMeter  # noqa: E402
from models.build_model import build_model  # noqa: E402
from stage1_closeout_utils import write_csv, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--tie-eps", type=float, default=1e-4)
    parser.add_argument("--min-recall", type=float, default=0.85)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    thresholds = sorted(set(float(x) for x in args.thresholds))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    meters = {threshold: BinarySegmentationMeter() for threshold in thresholds}

    for batch in tqdm(loader, desc="threshold_tune"):
        image = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].detach().cpu().numpy()[0, 0].astype("uint8")
        prob = torch.sigmoid(model(image)).detach().cpu().numpy()[0, 0]
        for threshold, meter in meters.items():
            pred = (prob >= threshold).astype("uint8")
            meter.update_binary(pred, target)

    rows = []
    for threshold in thresholds:
        metrics = meters[threshold].compute()
        rows.append({"threshold": threshold, **metrics})
    write_csv(out_dir / "threshold_tuning.csv", rows)

    max_iou = max(float(row["iou_building"]) for row in rows)
    candidates = [row for row in rows if max_iou - float(row["iou_building"]) <= args.tie_eps]
    best = sorted(
        candidates,
        key=lambda row: (
            float(row["f1_building"]),
            float(row["recall_building"]) >= args.min_recall,
            float(row["precision_building"]),
            -abs(float(row["threshold"]) - 0.5),
        ),
        reverse=True,
    )[0]
    official = {
        "model": "O1_unet_resnet34_freq",
        "selection_split": "val",
        "primary_metric": "IoU_building",
        "tie_eps": args.tie_eps,
        "threshold": float(best["threshold"]),
        "val_iou": float(best["iou_building"]),
        "val_f1": float(best["f1_building"]),
        "val_precision": float(best["precision_building"]),
        "val_recall": float(best["recall_building"]),
        "val_overall_accuracy": float(best["overall_accuracy"]),
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
    }
    write_json(out_dir / "official_threshold.json", official)
    print(json.dumps(official, sort_keys=True))


if __name__ == "__main__":
    main()
