#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from overlap_audit_core import write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export contact sheets for overlap review.")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--risk", default="high,medium")
    parser.add_argument("--split-pair", default="train-test")
    parser.add_argument("--max-pairs", type=int, default=300)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def tile(path: str, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "#17191d")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def contact_sheet(row: dict[str, str]) -> Image.Image:
    width, cell_height, label_height = 420, 300, 30
    kinds = (("pre", "pre optical"), ("sar", "post SAR"), ("mask", "mask"))
    canvas = Image.new("RGB", (width * 2, label_height + len(kinds) * (cell_height + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    title = f"{row['pair_id']}  {row['split_pair']}  score={row['overlap_score']}  risk={row['risk_level']}"
    draw.text((8, 7), title, fill="black")
    y = label_height
    for kind, label in kinds:
        for column, side in enumerate(("a", "b")):
            caption = f"{side.upper()} {label}: {row[f'id_{side}']} / {row[f'event_{side}']}"
            draw.text((column * width + 8, y + 7), caption, fill="black")
            canvas.paste(tile(row[f"{kind}_{side}"], (width, cell_height)), (column * width, y + label_height))
        y += cell_height + label_height
    return canvas


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.force:
        raise FileExistsError(f"refusing to overwrite non-empty {args.out_dir}; pass --force")
    if args.force and args.out_dir.exists():
        for path in args.out_dir.glob("*.jpg"):
            path.unlink()
        index_path = args.out_dir / "index.csv"
        if index_path.exists():
            index_path.unlink()
    risks = set(args.risk.split(","))
    with args.candidates.open(encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["risk_level"] in risks and row["split_pair"] == args.split_pair
        ][: args.max_pairs]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, str]] = []
    for rank, row in enumerate(tqdm(rows, desc="contact_sheets"), 1):
        filename = f"{rank:04d}_{row['risk_level']}_{row['pair_id']}.jpg"
        contact_sheet(row).save(args.out_dir / filename, quality=90)
        index_rows.append(
            {
                "rank": str(rank),
                "pair_id": row["pair_id"],
                "risk_level": row["risk_level"],
                "overlap_score": row["overlap_score"],
                "contact_sheet": filename,
            }
        )
    write_csv(
        args.out_dir / "index.csv",
        index_rows,
        ["rank", "pair_id", "risk_level", "overlap_score", "contact_sheet"],
    )
    print(f"[OK] contact_sheets={len(rows)} out={args.out_dir}")


if __name__ == "__main__":
    main()
