#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from tqdm import tqdm

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))

from config import load_config  # noqa: E402
from datasets.disasterm3_stage1_dataset import DisasterM3Stage1Dataset  # noqa: E402
from metrics import BinarySegmentationMeter  # noqa: E402
from models.build_model import build_model  # noqa: E402


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def select_from_metrics(path: Path, samples_per_disaster: int) -> dict[str, str]:
    rows = read_csv_rows(path)
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["disaster_type"], []).append(row)

    selected: dict[str, str] = {}
    rank_names = ["worst", "median", "best"]
    for disaster, items in sorted(grouped.items()):
        items = sorted(items, key=lambda item: float(item["iou_building"]))
        if samples_per_disaster <= 1:
            indices = [len(items) // 2]
            labels = ["median"]
        elif samples_per_disaster == 2:
            indices = [0, len(items) - 1]
            labels = ["worst", "best"]
        else:
            indices = [0, len(items) // 2, len(items) - 1]
            labels = rank_names

        used = set()
        for label, index in zip(labels, indices):
            index = max(0, min(index, len(items) - 1))
            sample_id = items[index]["id"]
            if sample_id in used:
                continue
            used.add(sample_id)
            selected[sample_id] = f"{disaster}_{label}"
    return selected


def select_from_manifest(path: Path, samples_per_disaster: int) -> dict[str, str]:
    rows = read_jsonl(path)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("disaster_type", "unknown")), []).append(row)

    selected: dict[str, str] = {}
    for disaster, items in sorted(grouped.items()):
        for index, row in enumerate(items[:samples_per_disaster]):
            selected[str(row["id"])] = f"{disaster}_{index + 1:02d}"
    return selected


def denormalize(image_t: torch.Tensor) -> np.ndarray:
    image = image_t.detach().cpu().numpy().transpose(1, 2, 0)
    image = np.clip((image * IMAGENET_STD + IMAGENET_MEAN) * 255.0, 0, 255)
    return image.astype(np.uint8)


def apply_overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.55) -> np.ndarray:
    out = image.copy()
    active = mask.astype(bool)
    if active.any():
        color_arr = np.array(color, dtype=np.float32)
        out[active] = (out[active].astype(np.float32) * (1.0 - alpha) + color_arr * alpha).astype(np.uint8)
    return out


def error_overlay(image: np.ndarray, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred_b = pred.astype(bool)
    target_b = target.astype(bool)
    out = image.copy()
    out = apply_overlay(out, pred_b & target_b, (0, 200, 80), alpha=0.55)
    out = apply_overlay(out, pred_b & ~target_b, (235, 55, 55), alpha=0.65)
    out = apply_overlay(out, ~pred_b & target_b, (40, 110, 235), alpha=0.65)
    return out


def add_title(image: Image.Image, title: str, subtitle: str = "") -> Image.Image:
    font = ImageFont.load_default()
    title_h = 42 if subtitle else 28
    canvas = Image.new("RGB", (image.width, image.height + title_h), "white")
    canvas.paste(image, (0, title_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill=(20, 20, 20), font=font)
    if subtitle:
        draw.text((8, 22), subtitle, fill=(70, 70, 70), font=font)
    return canvas


def make_comparison(
    image: np.ndarray,
    pred: np.ndarray,
    target: np.ndarray,
    title: str,
    subtitle: str,
) -> Image.Image:
    input_panel = add_title(Image.fromarray(image), "Input RGB", title)
    gt_panel = add_title(Image.fromarray(apply_overlay(image, target, (0, 200, 80))), "Label mask", "green = building")
    pred_panel = add_title(Image.fromarray(apply_overlay(image, pred, (255, 125, 0))), "Prediction", "orange = predicted building")
    err_panel = add_title(
        Image.fromarray(error_overlay(image, pred, target)),
        "Error overlay",
        "green TP, red FP, blue FN",
    )

    panels = [input_panel, gt_panel, pred_panel, err_panel]
    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    return canvas


def save_contact_sheet(paths: list[Path], out: Path, thumb_width: int = 900) -> None:
    if not paths:
        return
    thumbs = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        scale = thumb_width / image.width
        thumb = image.resize((thumb_width, max(1, int(image.height * scale))), Image.Resampling.BILINEAR)
        thumbs.append(thumb)
    width = thumb_width
    height = sum(img.height for img in thumbs)
    sheet = Image.new("RGB", (width, height), "white")
    y = 0
    for img in thumbs:
        sheet.paste(img, (0, y))
        y += img.height
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-metrics")
    parser.add_argument("--samples-per-disaster", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    comparison_dir = output_dir / "comparisons"
    pred_dir = output_dir / "pred_masks"
    gt_dir = output_dir / "gt_masks"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.sample_metrics:
        selected = select_from_metrics(Path(args.sample_metrics), args.samples_per_disaster)
    else:
        selected = select_from_manifest(Path(args.manifest), args.samples_per_disaster)

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
    ds.rows = [row for row in ds.rows if str(row.get("id", "")) in selected]
    ds.rows.sort(key=lambda row: selected[str(row.get("id", ""))])
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    metadata_rows = []
    comparison_paths = []
    for batch in tqdm(loader, desc="visualize"):
        image_t = batch["image"].to(device, non_blocking=True)
        mask_t = batch["mask"].to(device, non_blocking=True)
        logits = model(image_t)
        pred = (torch.sigmoid(logits) >= args.threshold).detach().cpu().numpy()[0, 0].astype(np.uint8)
        target = mask_t.detach().cpu().numpy()[0, 0].astype(np.uint8)
        image = denormalize(batch["image"][0])

        meter = BinarySegmentationMeter()
        meter.update_binary(pred, target)
        metrics = meter.compute()

        sample_id = str(batch["id"][0])
        tag = selected[sample_id]
        safe_name = f"{tag}_{sample_id}".replace("/", "_")
        title = f"{sample_id} | {batch['disaster_type'][0]} | {batch['event_id'][0]}"
        subtitle = f"IoU={metrics['iou_building']:.3f}, F1={metrics['f1_building']:.3f}"

        pred_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray((pred * 255).astype(np.uint8)).save(pred_dir / f"{safe_name}_pred.png")
        Image.fromarray((target * 255).astype(np.uint8)).save(gt_dir / f"{safe_name}_gt.png")

        comparison = make_comparison(image, pred, target, title, subtitle)
        comparison_path = comparison_dir / f"{safe_name}_compare.jpg"
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        comparison.save(comparison_path, quality=92)
        comparison_paths.append(comparison_path)

        metadata_rows.append(
            {
                "tag": tag,
                "id": sample_id,
                "event_id": str(batch["event_id"][0]),
                "disaster_type": str(batch["disaster_type"][0]),
                "country_or_region": str(batch["country_or_region"][0]),
                "qc_label": str(batch["qc_label"][0]),
                **metrics,
                "comparison": str(comparison_path.relative_to(output_dir)),
                "pred_mask": str((pred_dir / f"{safe_name}_pred.png").relative_to(output_dir)),
                "gt_mask": str((gt_dir / f"{safe_name}_gt.png").relative_to(output_dir)),
            }
        )

    write_csv(output_dir / "visualization_samples.csv", metadata_rows)
    save_contact_sheet(comparison_paths, output_dir / "contact_sheet.jpg")
    print(json.dumps({"output_dir": str(output_dir), "sample_count": len(metadata_rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
