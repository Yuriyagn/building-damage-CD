#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import load_config  # noqa: E402
from models.build_model import build_model  # noqa: E402
from stage2.common import write_csv, write_json  # noqa: E402
from stage2.datasets_v2 import Stage2V2Dataset  # noqa: E402
from stage2.train_ogsr_v0_predictor import forward_padded, make_predictor_input  # noqa: E402


def split_offset(split: str) -> int:
    return {"train": 0, "val": 10_000, "test": 20_000}[split]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_sample_id(sample_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("._")
    if safe:
        return safe
    return hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:16]


def make_dataset(
    cfg: dict[str, Any],
    data_root: Path,
    split: str,
    seed: int,
    sar_shuffle_mode: str,
    sar_shuffle_seed: int,
    limit: int | None,
) -> Stage2V2Dataset:
    dcfg = dict(cfg.get("dataset", {}))
    return Stage2V2Dataset(
        data_root=data_root,
        manifest=str(dcfg[f"{split}_manifest"]),
        train=False,
        prior_type=str(dcfg.get("prior_type", "predicted")),
        input_mode=str(dcfg.get("input_mode", "pre_sar_texture_prior")),
        crop_size=None,
        sar_shuffle_mode=sar_shuffle_mode,
        sar_shuffle_seed=sar_shuffle_seed + split_offset(split),
        sar_singleton_policy=str(dcfg.get("sar_singleton_policy", "error")),
        limit=limit,
    )


def as_numpy_2d(tensor: torch.Tensor) -> np.ndarray:
    return np.asarray(tensor.detach().cpu().numpy()[0, 0], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export OGSR-v0 residual texture features")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["train", "val"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sar-shuffle-mode", default="paired")
    parser.add_argument("--sar-shuffle-seed", type=int, default=20260627)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = deepcopy(load_config(args.config)) if args.config else deepcopy(checkpoint["config"])
    seed = int(args.seed if args.seed is not None else checkpoint.get("seed", 42))

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is unavailable; pass --allow-cpu only for a diagnostic export")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_default = bool(dict(cfg.get("train", {})).get("amp", True))
    amp = bool(amp_default if args.amp is None else args.amp) and device.type == "cuda"

    model = build_model(cfg, no_pretrained=args.no_pretrained).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    write_json(
        output_dir / "export_info.json",
        {
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "seed": seed,
            "data_root": str(Path(args.data_root).resolve()),
            "splits": list(args.splits),
            "sar_shuffle_mode": str(args.sar_shuffle_mode),
            "sar_shuffle_seed": int(args.sar_shuffle_seed),
            "feature_keys": [
                "expected_sar",
                "expected_sar_grad",
                "residual_sar",
                "abs_residual_sar",
                "residual_grad",
                "abs_residual_grad",
            ],
            "device": str(device),
            "cuda": torch.cuda.is_available(),
            "torch": torch.__version__,
        },
    )

    data_root = Path(args.data_root)
    all_rows: list[dict[str, Any]] = []
    for split in args.splits:
        dataset = make_dataset(
            cfg,
            data_root,
            split,
            seed,
            str(args.sar_shuffle_mode),
            int(args.sar_shuffle_seed),
            args.limit,
        )
        loader = DataLoader(
            dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=device.type == "cuda",
        )
        feature_dir = output_dir / "features" / split
        feature_dir.mkdir(parents=True, exist_ok=True)
        index: dict[str, str] = {}
        split_rows: list[dict[str, Any]] = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"export-{split}", leave=False):
                image = make_predictor_input(batch, device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    expected = torch.sigmoid(forward_padded(model, image))
                observed_sar = batch["sar"].to(device, non_blocking=True)
                observed_grad = batch["sar_grad"].to(device, non_blocking=True)
                residual_sar = observed_sar - expected[:, 0:1]
                residual_grad = observed_grad - expected[:, 1:2]
                ids = batch["id"]
                sar_source_ids = batch.get("sar_source_id", ids)
                sar_is_paired = batch.get("sar_is_paired", [True] * len(ids))
                for item_index, sample_id in enumerate(ids):
                    sample_id = str(sample_id)
                    filename = f"{safe_sample_id(sample_id)}.npz"
                    out_path = feature_dir / filename
                    rel_path = out_path.relative_to(output_dir).as_posix()
                    np.savez_compressed(
                        out_path,
                        expected_sar=as_numpy_2d(expected[item_index : item_index + 1, 0:1]).astype(np.float16),
                        expected_sar_grad=as_numpy_2d(expected[item_index : item_index + 1, 1:2]).astype(np.float16),
                        residual_sar=as_numpy_2d(residual_sar[item_index : item_index + 1]).astype(np.float16),
                        abs_residual_sar=as_numpy_2d(residual_sar[item_index : item_index + 1].abs()).astype(
                            np.float16
                        ),
                        residual_grad=as_numpy_2d(residual_grad[item_index : item_index + 1]).astype(np.float16),
                        abs_residual_grad=as_numpy_2d(residual_grad[item_index : item_index + 1].abs()).astype(
                            np.float16
                        ),
                    )
                    index[sample_id] = rel_path
                    paired_value = sar_is_paired[item_index]
                    if hasattr(paired_value, "item"):
                        paired_value = bool(paired_value.item())
                    split_rows.append(
                        {
                            "split": split,
                            "id": sample_id,
                            "feature_path": rel_path,
                            "sar_source_id": str(sar_source_ids[item_index]),
                            "sar_is_paired": bool(paired_value),
                        }
                    )
        write_json(
            output_dir / f"index_{split}.json",
            {
                "split": split,
                "count": len(index),
                "sar_shuffle_mode": dataset.permutation_info["mode"],
                "sar_shuffle_seed": int(args.sar_shuffle_seed) + split_offset(split),
                "permutation": dataset.permutation_info,
                "features": index,
            },
        )
        write_csv(output_dir / f"samples_{split}.csv", split_rows)
        all_rows.extend(split_rows)
    write_csv(output_dir / "samples_all.csv", all_rows)
    write_json(output_dir / "completed.json", {"status": "completed", "sample_count": len(all_rows)})
    print(json.dumps({"status": "completed", "sample_count": len(all_rows)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
