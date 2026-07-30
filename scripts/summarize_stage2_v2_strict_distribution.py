#!/usr/bin/env python3
"""Summarize strict Stage-2 split/event class distributions and concentration."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


CLASS_NAMES = ("background", "intact", "damaged", "destroyed")
BUILDING_CLASSES = (1, 2, 3)
COLORS = ("#4caf50", "#ffc107", "#f44336")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    split_totals: dict[str, np.ndarray] = {}
    split_event_hists: dict[str, dict[str, np.ndarray]] = {}

    for split in ("train", "val", "test"):
        rows = read_jsonl(args.manifest_root / f"stage2_master_{split}.jsonl")
        event_hists: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(4, dtype=np.int64))
        event_images: dict[str, int] = defaultdict(int)
        event_presence: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(4, dtype=np.int64))
        event_names: dict[str, str] = {}
        for row in rows:
            event = str(row["event_id"])
            event_names[event] = str(row.get("canonical_event_name", event))
            mask = np.asarray(Image.open(args.data_root / str(row["mask_multiclass"])))
            hist = np.bincount(mask.reshape(-1), minlength=4)[:4]
            event_hists[event] += hist
            event_images[event] += 1
            event_presence[event] += (hist > 0).astype(np.int64)
        split_event_hists[split] = dict(event_hists)
        split_total = sum(event_hists.values(), np.zeros(4, dtype=np.int64))
        split_totals[split] = split_total
        building_total = int(split_total[list(BUILDING_CLASSES)].sum())
        for event in sorted(event_hists):
            hist = event_hists[event]
            building = int(hist[list(BUILDING_CLASSES)].sum())
            record: dict[str, object] = {
                "split": split,
                "event_id": event,
                "event_name": event_names[event],
                "images": event_images[event],
                "building_pixels": building,
            }
            for class_id, name in enumerate(CLASS_NAMES):
                record[f"{name}_pixels"] = int(hist[class_id])
                record[f"images_with_{name}"] = int(event_presence[event][class_id])
                if class_id in BUILDING_CLASSES:
                    record[f"{name}_share_within_event"] = float(hist[class_id] / building) if building else 0.0
                    denom = int(split_total[class_id])
                    record[f"{name}_share_of_split_class"] = float(hist[class_id] / denom) if denom else 0.0
            records.append(record)

    write_csv(args.out_dir / "event_class_distribution.csv", records)

    concentration: dict[str, dict[str, dict[str, float | str]]] = {}
    for split, events in split_event_hists.items():
        concentration[split] = {}
        for class_id in BUILDING_CLASSES:
            name = CLASS_NAMES[class_id]
            values = {event: int(hist[class_id]) for event, hist in events.items()}
            total = sum(values.values())
            shares = {event: value / total if total else 0.0 for event, value in values.items()}
            top_event, top_share = max(shares.items(), key=lambda item: item[1])
            hhi = sum(value * value for value in shares.values())
            concentration[split][name] = {
                "top_event": top_event,
                "top1_share": top_share,
                "hhi": hhi,
                "effective_event_count": 1 / hhi if hhi else 0.0,
            }

    summary = {
        "manifest_root": str(args.manifest_root.resolve()),
        "data_root": str(args.data_root.resolve()),
        "split_totals": {
            split: {
                "events": len(split_event_hists[split]),
                "images": sum(int(row["images"]) for row in records if row["split"] == split),
                **{f"{CLASS_NAMES[index]}_pixels": int(value) for index, value in enumerate(hist)},
                **{
                    f"{CLASS_NAMES[index]}_building_share": float(hist[index] / hist[list(BUILDING_CLASSES)].sum())
                    for index in BUILDING_CLASSES
                },
            }
            for split, hist in split_totals.items()
        },
        "concentration": concentration,
    }
    (args.out_dir / "distribution_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharex=False)
    for axis, split in zip(axes, ("train", "val", "test"), strict=True):
        split_rows = [row for row in records if row["split"] == split]
        labels = [str(row["event_id"]) for row in split_rows]
        left = np.zeros(len(split_rows))
        for class_id, color in zip(BUILDING_CLASSES, COLORS, strict=True):
            name = CLASS_NAMES[class_id]
            values = np.array([float(row[f"{name}_share_within_event"]) for row in split_rows])
            axis.barh(labels, values, left=left, color=color, label=name)
            left += values
        axis.set_title(f"{split.title()} ({len(split_rows)} events)")
        axis.set_xlim(0, 1)
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.2)
    axes[0].legend(loc="lower right")
    fig.suptitle("Strict-v1 building-pixel class distribution by event")
    fig.tight_layout()
    fig.savefig(args.out_dir / "event_class_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    test_rows = [row for row in records if row["split"] == "test"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for axis, class_id, color in zip(axes, BUILDING_CLASSES, COLORS, strict=True):
        name = CLASS_NAMES[class_id]
        labels = [str(row["event_id"]) for row in test_rows]
        values = [float(row[f"{name}_share_of_split_class"]) for row in test_rows]
        axis.barh(labels, values, color=color)
        axis.set_title(f"Test {name}: contribution by event")
        axis.set_xlim(0, 1)
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.2)
        for index, value in enumerate(values):
            axis.text(value + 0.01, index, pct(value), va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out_dir / "test_class_concentration.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    lines = ["# Strict-v1 event/class distribution", ""]
    for split in ("train", "val", "test"):
        lines.extend([f"## {split.title()}", "", "| event | images | intact | damaged | destroyed |", "|---|---:|---:|---:|---:|"])
        for row in [item for item in records if item["split"] == split]:
            lines.append(
                f"| {row['event_id']} | {row['images']} | {pct(float(row['intact_share_within_event']))} | "
                f"{pct(float(row['damaged_share_within_event']))} | {pct(float(row['destroyed_share_within_event']))} |"
            )
        lines.append("")
    lines.extend(["## Test concentration", "", "| class | top event | top-1 share | HHI | effective events |", "|---|---|---:|---:|---:|"])
    for name in ("intact", "damaged", "destroyed"):
        item = concentration["test"][name]
        lines.append(
            f"| {name} | {item['top_event']} | {pct(float(item['top1_share']))} | "
            f"{float(item['hhi']):.4f} | {float(item['effective_event_count']):.2f} |"
        )
    (args.out_dir / "distribution_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
