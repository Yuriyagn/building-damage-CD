#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from stage1_closeout_utils import (
    common_stage2_fields,
    default_data_root,
    read_json,
    read_jsonl,
    resolve_manifest,
    to_data_rel,
    write_json,
    write_jsonl,
)


DEFAULT_INPUT_MANIFESTS = {
    "train": "splits/v0.2_qc/building_mask_train_sar_noempty_qc.jsonl",
    "val": "splits/v0.2_qc/building_mask_val_sar_qc.jsonl",
    "test": "splits/v0.2_qc/building_mask_test_sar_qc.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--practice-root", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--stage1-prior-dir", required=True)
    parser.add_argument("--stage1-binary-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--threshold-json")
    return parser.parse_args()


def load_prior_index(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    index = {}
    for row in read_jsonl(path):
        index[(str(row["split"]), str(row["id"]))] = row
    return index


def get_post_sar(row: dict[str, Any]) -> str:
    post_images = row.get("post_images")
    if isinstance(post_images, dict) and post_images.get("SAR"):
        return str(post_images["SAR"])
    if row.get("post_modality") == "SAR" and row.get("post_image"):
        return str(row["post_image"])
    raise ValueError(f"missing SAR post image for id={row.get('id')}")


def make_common_row(
    row: dict[str, Any],
    split_name: str,
    data_root: Path,
    stage1_binary_dir: Path,
    prior_row: dict[str, Any],
    threshold: float | None,
) -> dict[str, Any]:
    sample_id = str(row["id"])
    oracle_path = stage1_binary_dir / split_name / f"{sample_id}.png"
    return {
        **common_stage2_fields(row),
        "split": split_name,
        "pre_image": row["pre_image"],
        "post_sar": get_post_sar(row),
        "mask_multiclass": row["mask_multiclass"],
        "oracle_building_mask": to_data_rel(data_root, oracle_path),
        "pred_building_prob": prior_row["building_prob"],
        "pred_building_prob_uint8": prior_row["building_prob_uint8"],
        "pred_building_binary": prior_row["building_binary"],
        "stage1_model": "O1_unet_resnet34_freq",
        "stage1_threshold": threshold,
        "target_num_classes": 4,
        "target_values": {"0": "background", "1": "intact", "2": "damaged", "3": "destroyed"},
    }


def main() -> None:
    args = parse_args()
    practice_root = Path(args.practice_root)
    data_root = Path(args.data_root) if args.data_root else default_data_root(practice_root)
    prior_dir = Path(args.stage1_prior_dir)
    stage1_binary_dir = Path(args.stage1_binary_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prior_index = load_prior_index(prior_dir / "prior_manifest_index.jsonl")
    threshold = None
    if args.threshold_json:
        threshold = float(read_json(args.threshold_json)["threshold"])
    elif (prior_dir / "export_summary.json").exists():
        threshold = float(read_json(prior_dir / "export_summary.json")["threshold"])

    summary: dict[str, Any] = {"splits": {}, "missing_priors": []}
    for split_name, manifest_rel in DEFAULT_INPUT_MANIFESTS.items():
        source_manifest = resolve_manifest(practice_root, manifest_rel)
        source_rows = read_jsonl(source_manifest)
        master_rows = []
        no_prior_rows = []
        oracle_rows = []
        predicted_rows = []

        for row in source_rows:
            key = (split_name, str(row["id"]))
            prior_row = prior_index.get(key)
            if prior_row is None:
                summary["missing_priors"].append({"split": split_name, "id": row["id"]})
                continue
            common = make_common_row(row, split_name, data_root, stage1_binary_dir, prior_row, threshold)
            master_rows.append(common)

            no_prior_rows.append(
                {
                    **common,
                    "prior_type": "none",
                    "building_prior": None,
                    "building_prior_uint8": None,
                    "building_prior_binary": None,
                }
            )
            oracle_rows.append(
                {
                    **common,
                    "prior_type": "oracle",
                    "building_prior": common["oracle_building_mask"],
                    "building_prior_uint8": common["oracle_building_mask"],
                    "building_prior_binary": common["oracle_building_mask"],
                }
            )
            predicted_rows.append(
                {
                    **common,
                    "prior_type": "predicted",
                    "building_prior": common["pred_building_prob"],
                    "building_prior_uint8": common["pred_building_prob_uint8"],
                    "building_prior_binary": common["pred_building_binary"],
                }
            )

        write_jsonl(out_dir / f"stage2_master_{split_name}.jsonl", master_rows)
        write_jsonl(out_dir / "no_prior" / f"{split_name}.jsonl", no_prior_rows)
        write_jsonl(out_dir / "oracle_prior" / f"{split_name}.jsonl", oracle_rows)
        write_jsonl(out_dir / "predicted_prior" / f"{split_name}.jsonl", predicted_rows)
        summary["splits"][split_name] = {
            "source_manifest": str(source_manifest),
            "source_count": len(source_rows),
            "master_count": len(master_rows),
            "no_prior_count": len(no_prior_rows),
            "oracle_prior_count": len(oracle_rows),
            "predicted_prior_count": len(predicted_rows),
        }

    summary["practice_root"] = str(practice_root)
    summary["data_root"] = str(data_root)
    summary["stage1_prior_dir"] = str(prior_dir)
    summary["stage1_binary_dir"] = str(stage1_binary_dir)
    summary["stage1_threshold"] = threshold
    write_json(out_dir / "stage2_manifest_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
