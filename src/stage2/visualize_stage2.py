#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import load_config  # noqa: E402
from stage2.datasets import Stage2DamageDataset  # noqa: E402
from stage2.test_stage2 import colorize  # noqa: E402


def make_dataset(cfg: dict, data_root: Path, split: str, limit: int | None) -> Stage2DamageDataset:
    dcfg = dict(cfg.get("dataset", {}))
    return Stage2DamageDataset(
        data_root=data_root,
        manifest=dcfg[f"{split}_manifest"],
        train=False,
        prior_type=str(dcfg.get("prior_type", "predicted")),
        input_mode=str(dcfg.get("input_mode", "sar_prior")),
        limit=limit,
    )


def to_rgb_gray(arr: np.ndarray) -> np.ndarray:
    gray = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def labeled_tile(image: np.ndarray, label: str, width: int = 256) -> Image.Image:
    resampling = getattr(Image, "Resampling", Image).NEAREST
    pil = Image.fromarray(image).resize((width, width), resampling)
    canvas = Image.new("RGB", (width, width + 24), (255, 255, 255))
    canvas.paste(pil, (0, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), label, fill=(0, 0, 0))
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--tile-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ds = make_dataset(cfg, Path(args.data_root), args.split, args.limit)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in range(len(ds)):
        item = ds[idx]
        sar = item["sar"][0].numpy()
        prior = item["prior"][0].numpy()
        image = item["image"].numpy()
        target = item["mask"].numpy().astype(np.uint8)
        masked = image[2]
        tiles = [
            labeled_tile(to_rgb_gray(sar), "post SAR", args.tile_size),
            labeled_tile(to_rgb_gray(prior), "building prior", args.tile_size),
            labeled_tile(to_rgb_gray(masked), "SAR x prior", args.tile_size),
            labeled_tile(colorize(target), "GT damage mask", args.tile_size),
        ]
        sample = Image.new("RGB", (args.tile_size * len(tiles), args.tile_size + 24), (255, 255, 255))
        for tile_idx, tile in enumerate(tiles):
            sample.paste(tile, (tile_idx * args.tile_size, 0))
        sample_id = str(item["id"])
        sample.save(out_dir / f"{idx:03d}_{sample_id}.png")
        rows.append(sample)

    if rows:
        sheet_cols = 1
        sheet = Image.new("RGB", (rows[0].width * sheet_cols, rows[0].height * len(rows)), (255, 255, 255))
        for idx, row in enumerate(rows):
            sheet.paste(row, (0, idx * row.height))
        sheet.save(out_dir / "contact_sheet.png")
    print(f"wrote {len(rows)} visualizations to {out_dir}")


if __name__ == "__main__":
    main()
