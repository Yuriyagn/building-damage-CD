#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_ROOT))

from config import load_config  # noqa: E402
from models.build_model import build_model  # noqa: E402
from stage1_closeout_utils import (  # noqa: E402
    default_data_root,
    read_json,
    read_jsonl,
    resolve_data_path,
    resolve_manifest,
    to_data_rel,
    write_json,
    write_jsonl,
)


DEFAULT_INPUT_MANIFESTS = [
    "splits/v0.2_qc/building_mask_train_sar_noempty_qc.jsonl",
    "splits/v0.2_qc/building_mask_val_sar_qc.jsonl",
    "splits/v0.2_qc/building_mask_test_sar_qc.jsonl",
]
DEFAULT_SPLIT_NAMES = ["train", "val", "test"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threshold-json", required=True)
    parser.add_argument("--practice-root", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--input-manifests", nargs="+", default=DEFAULT_INPUT_MANIFESTS)
    parser.add_argument("--split-names", nargs="+", default=DEFAULT_SPLIT_NAMES)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def load_image(path: Path, normalize: str) -> torch.Tensor:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    if normalize == "imagenet":
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
    return tensor.unsqueeze(0)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    practice_root = Path(args.practice_root)
    data_root = Path(args.data_root) if args.data_root else default_data_root(practice_root)
    out_dir = Path(args.out_dir)
    if len(args.input_manifests) != len(args.split_names):
        raise ValueError("--input-manifests and --split-names must have equal length")

    threshold_info = read_json(args.threshold_json)
    threshold = float(threshold_info["threshold"])
    cfg = load_config(args.config)
    normalize = str(cfg.get("input", {}).get("normalize", "imagenet"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, no_pretrained=True).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    index_rows = []
    split_summary: dict[str, dict[str, object]] = {}
    global_min = 1.0
    global_max = 0.0
    had_missing_inputs = False

    for split_name, manifest_value in zip(args.split_names, args.input_manifests):
        manifest_path = resolve_manifest(practice_root, manifest_value)
        rows = read_jsonl(manifest_path)
        split_min = 1.0
        split_max = 0.0
        count = 0
        missing_inputs: list[str] = []

        for row in tqdm(rows, desc=f"export_{split_name}"):
            sample_id = str(row["id"])
            image_path = resolve_data_path(data_root, row["pre_image"])
            if not image_path.exists():
                missing_inputs.append(str(image_path))
                continue

            image = load_image(image_path, normalize).to(device)
            prob = torch.sigmoid(model(image)).detach().cpu().numpy()[0, 0].astype(np.float32)
            prob = np.clip(prob, 0.0, 1.0)
            binary = (prob >= threshold).astype(np.uint8) * 255
            prob_uint8 = np.rint(prob * 255.0).astype(np.uint8)

            prob_png = out_dir / "prob_uint8" / split_name / f"{sample_id}.png"
            prob_npz = out_dir / "prob_float16_npz" / split_name / f"{sample_id}.npz"
            binary_png = out_dir / "binary_tuned" / split_name / f"{sample_id}.png"
            prob_png.parent.mkdir(parents=True, exist_ok=True)
            prob_npz.parent.mkdir(parents=True, exist_ok=True)
            binary_png.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(prob_uint8).save(prob_png)
            np.savez_compressed(prob_npz, prob=prob.astype(np.float16))
            Image.fromarray(binary).save(binary_png)

            pmin = float(prob.min())
            pmax = float(prob.max())
            split_min = min(split_min, pmin)
            split_max = max(split_max, pmax)
            global_min = min(global_min, pmin)
            global_max = max(global_max, pmax)
            count += 1

            index_rows.append(
                {
                    "id": sample_id,
                    "split": split_name,
                    "event_id": row.get("event_id", ""),
                    "disaster_type": row.get("disaster_type", ""),
                    "building_prob": to_data_rel(data_root, prob_npz),
                    "building_prob_uint8": to_data_rel(data_root, prob_png),
                    "building_binary": to_data_rel(data_root, binary_png),
                    "stage1_model": "O1_unet_resnet34_freq",
                    "stage1_threshold": threshold,
                }
            )

        split_summary[split_name] = {
            "manifest": str(manifest_path),
            "expected_count": len(rows),
            "exported_count": count,
            "missing_input_count": len(missing_inputs),
            "missing_inputs_preview": missing_inputs[:20],
            "prob_min": split_min if count else None,
            "prob_max": split_max if count else None,
        }
        had_missing_inputs = had_missing_inputs or bool(missing_inputs)

    write_jsonl(out_dir / "prior_manifest_index.jsonl", index_rows)
    summary = {
        "model": "O1_unet_resnet34_freq",
        "checkpoint": str(args.checkpoint),
        "threshold_json": str(args.threshold_json),
        "threshold": threshold,
        "practice_root": str(practice_root),
        "data_root": str(data_root),
        "out_dir": str(out_dir),
        "total_exported": len(index_rows),
        "prob_min": global_min if index_rows else None,
        "prob_max": global_max if index_rows else None,
        "splits": split_summary,
    }
    write_json(out_dir / "export_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    if had_missing_inputs:
        raise SystemExit("missing input images; copy the manifest-referenced images or set --data-root")


if __name__ == "__main__":
    main()
