#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from stage1_closeout_utils import (
    common_stage2_fields,
    read_json,
    read_jsonl,
    resolve_data_path,
    resolve_manifest,
    to_data_rel,
    write_json,
    write_jsonl,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - keeps --help/lightweight runs dependency-free
    def tqdm(iterable, **_: Any):  # type: ignore[no-redef]
        return iterable


DEFAULT_INPUT_MANIFESTS = {
    "train": "splits/v0.2_qc/building_mask_train_sar_noempty_qc.jsonl",
    "val": "splits/v0.2_qc/building_mask_val_sar_qc.jsonl",
    "test": "splits/v0.2_qc/building_mask_test_sar_qc.jsonl",
}
EXPECTED_COUNTS = {"train": 1543, "val": 506, "test": 564}
MODEL_NAME = "O1_unet_resnet34_freq"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--practice-root", required=True)
    parser.add_argument("--source-data-root", required=True)
    parser.add_argument("--stage1-prior-dir", required=True)
    parser.add_argument("--stage1-binary-dir", required=True)
    parser.add_argument("--threshold-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--allow-missing-priors", action="store_true")
    return parser.parse_args()


def get_post_sar(row: dict[str, Any]) -> str:
    post_images = row.get("post_images")
    if isinstance(post_images, dict) and post_images.get("SAR"):
        return str(post_images["SAR"])
    if row.get("post_modality") == "SAR" and row.get("post_image"):
        return str(row["post_image"])
    raise ValueError(f"missing SAR post image for id={row.get('id')}")


def load_prior_index(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row["split"]), str(row["id"])): row for row in read_jsonl(path)}


def copy_file(src: Path, dst: Path) -> int:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return 0
    shutil.copy2(src, dst)
    return src.stat().st_size


def copy_optional(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    copy_file(src, dst)
    return True


def make_stage1_prior_input(row: dict[str, Any], split_name: str, pre_rel: str) -> dict[str, Any]:
    return {
        **common_stage2_fields(row),
        "id": str(row["id"]),
        "split": split_name,
        "pre_image": pre_rel,
    }


def make_stage2_common(
    row: dict[str, Any],
    split_name: str,
    pre_rel: str,
    post_sar_rel: str,
    mask_rel: str,
    oracle_rel: str,
    prior_paths: dict[str, str] | None,
    threshold: float,
) -> dict[str, Any]:
    return {
        **common_stage2_fields(row),
        "id": str(row["id"]),
        "split": split_name,
        "pre_image": pre_rel,
        "post_sar": post_sar_rel,
        "mask_multiclass": mask_rel,
        "oracle_building_mask": oracle_rel,
        "pred_building_prob": prior_paths["prob"] if prior_paths else None,
        "pred_building_prob_uint8": prior_paths["prob_uint8"] if prior_paths else None,
        "pred_building_binary": prior_paths["binary"] if prior_paths else None,
        "stage1_model": MODEL_NAME,
        "stage1_threshold": threshold,
        "target_num_classes": 4,
        "target_values": {"0": "background", "1": "intact", "2": "damaged", "3": "destroyed"},
    }


def write_needed_file(path: Path, values: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{value}\n" for value in sorted(values)), encoding="utf-8")


def main() -> None:
    args = parse_args()
    practice_root = Path(args.practice_root)
    source_data_root = Path(args.source_data_root)
    stage1_prior_dir = Path(args.stage1_prior_dir)
    stage1_binary_dir = Path(args.stage1_binary_dir)
    out_dir = Path(args.out_dir)
    manifest_dir = out_dir / "manifests"
    reports_dir = out_dir / "reports"
    threshold = float(read_json(args.threshold_json)["threshold"])

    prior_index_path = stage1_prior_dir / "prior_manifest_index.jsonl"
    prior_index: dict[tuple[str, str], dict[str, Any]] = {}
    if prior_index_path.exists():
        prior_index = load_prior_index(prior_index_path)
    elif not args.allow_missing_priors:
        raise FileNotFoundError(f"missing prior index: {prior_index_path}")

    needed_raw_files: set[str] = set()
    needed_practice_files: set[str] = set()
    needed_prior_files: set[str] = set()
    prior_manifest_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "package": out_dir.name,
        "source_data_root": str(source_data_root),
        "practice_root": str(practice_root),
        "stage1_prior_dir": str(stage1_prior_dir),
        "stage1_binary_dir": str(stage1_binary_dir),
        "stage1_threshold": threshold,
        "splits": {},
        "missing_priors": [],
        "copied_bytes": 0,
    }

    for split_name, manifest_rel in DEFAULT_INPUT_MANIFESTS.items():
        source_manifest = resolve_manifest(practice_root, manifest_rel)
        source_rows = read_jsonl(source_manifest)
        if len(source_rows) != EXPECTED_COUNTS[split_name]:
            raise ValueError(f"{split_name} source count {len(source_rows)} != {EXPECTED_COUNTS[split_name]}")

        stage1_prior_rows = []
        master_rows = []
        no_prior_rows = []
        oracle_rows = []
        predicted_rows = []

        for row in tqdm(source_rows, desc=f"build_minimal_{split_name}"):
            sample_id = str(row["id"])
            pre_src = resolve_data_path(source_data_root, row["pre_image"])
            post_sar_src = resolve_data_path(source_data_root, get_post_sar(row))
            mask_src = resolve_data_path(source_data_root, row["mask_multiclass"])
            oracle_src = stage1_binary_dir / split_name / f"{sample_id}.png"

            pre_rel = f"images/{split_name}/{sample_id}_pre.png"
            post_sar_rel = f"images/{split_name}/{sample_id}_sar_post.png"
            mask_rel = f"masks_4class/{split_name}/{sample_id}.png"
            oracle_rel = f"oracle_building_masks/{split_name}/{sample_id}.png"

            summary["copied_bytes"] += copy_file(pre_src, out_dir / pre_rel)
            summary["copied_bytes"] += copy_file(post_sar_src, out_dir / post_sar_rel)
            summary["copied_bytes"] += copy_file(mask_src, out_dir / mask_rel)
            summary["copied_bytes"] += copy_file(oracle_src, out_dir / oracle_rel)

            needed_raw_files.add(to_data_rel(source_data_root, pre_src))
            needed_raw_files.add(to_data_rel(source_data_root, post_sar_src))
            needed_practice_files.add(to_data_rel(source_data_root, mask_src))
            needed_practice_files.add(to_data_rel(source_data_root, oracle_src))

            prior_paths: dict[str, str] | None = None
            source_prior_row = prior_index.get((split_name, sample_id))
            if source_prior_row:
                prob_src = resolve_data_path(source_data_root, source_prior_row["building_prob"])
                prob_uint8_src = resolve_data_path(source_data_root, source_prior_row["building_prob_uint8"])
                binary_src = resolve_data_path(source_data_root, source_prior_row["building_binary"])
                prob_rel = f"building_priors/{MODEL_NAME}/prob_float16_npz/{split_name}/{sample_id}.npz"
                prob_uint8_rel = f"building_priors/{MODEL_NAME}/prob_uint8/{split_name}/{sample_id}.png"
                binary_rel = f"building_priors/{MODEL_NAME}/binary_tuned/{split_name}/{sample_id}.png"
                summary["copied_bytes"] += copy_file(prob_src, out_dir / prob_rel)
                summary["copied_bytes"] += copy_file(prob_uint8_src, out_dir / prob_uint8_rel)
                summary["copied_bytes"] += copy_file(binary_src, out_dir / binary_rel)
                needed_prior_files.update(
                    {
                        to_data_rel(source_data_root, prob_src),
                        to_data_rel(source_data_root, prob_uint8_src),
                        to_data_rel(source_data_root, binary_src),
                    }
                )
                prior_paths = {"prob": prob_rel, "prob_uint8": prob_uint8_rel, "binary": binary_rel}
                prior_manifest_rows.append(
                    {
                        "id": sample_id,
                        "split": split_name,
                        "event_id": row.get("event_id", ""),
                        "disaster_type": row.get("disaster_type", ""),
                        "building_prob": prob_rel,
                        "building_prob_uint8": prob_uint8_rel,
                        "building_binary": binary_rel,
                        "stage1_model": MODEL_NAME,
                        "stage1_threshold": threshold,
                    }
                )
            else:
                summary["missing_priors"].append({"split": split_name, "id": sample_id})
                if not args.allow_missing_priors:
                    raise FileNotFoundError(f"missing prior for {split_name}/{sample_id}")

            stage1_prior_rows.append(make_stage1_prior_input(row, split_name, pre_rel))
            common = make_stage2_common(
                row=row,
                split_name=split_name,
                pre_rel=pre_rel,
                post_sar_rel=post_sar_rel,
                mask_rel=mask_rel,
                oracle_rel=oracle_rel,
                prior_paths=prior_paths,
                threshold=threshold,
            )
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
                    "building_prior": oracle_rel,
                    "building_prior_uint8": oracle_rel,
                    "building_prior_binary": oracle_rel,
                }
            )
            if prior_paths:
                predicted_rows.append(
                    {
                        **common,
                        "prior_type": "predicted",
                        "building_prior": prior_paths["prob"],
                        "building_prior_uint8": prior_paths["prob_uint8"],
                        "building_prior_binary": prior_paths["binary"],
                    }
                )

        write_jsonl(manifest_dir / f"stage1_prior_input_{split_name}.jsonl", stage1_prior_rows)
        write_jsonl(manifest_dir / f"stage2_master_{split_name}.jsonl", master_rows)
        write_jsonl(manifest_dir / "no_prior" / f"{split_name}.jsonl", no_prior_rows)
        write_jsonl(manifest_dir / "oracle_prior" / f"{split_name}.jsonl", oracle_rows)
        write_jsonl(manifest_dir / "predicted_prior" / f"{split_name}.jsonl", predicted_rows)
        summary["splits"][split_name] = {
            "source_manifest": str(source_manifest),
            "source_count": len(source_rows),
            "stage1_prior_input_count": len(stage1_prior_rows),
            "master_count": len(master_rows),
            "no_prior_count": len(no_prior_rows),
            "oracle_prior_count": len(oracle_rows),
            "predicted_prior_count": len(predicted_rows),
        }
        needed_practice_files.add(to_data_rel(source_data_root, source_manifest))

    write_jsonl(out_dir / "building_priors" / MODEL_NAME / "prior_manifest_index.jsonl", prior_manifest_rows)
    if (stage1_prior_dir / "export_summary.json").exists():
        copy_optional(stage1_prior_dir / "export_summary.json", out_dir / "building_priors" / MODEL_NAME / "export_summary.json")

    for rel in [
        "DATA_QC_REPORT.md",
        "DATA_STAGE1_OPTICAL_BUILDING.md",
        "diagnostics/sar_qc_generation_summary.json",
        "diagnostics/stage1_optical_building_summary.json",
        "tables/qc_summary_by_manifest_sar.csv",
        "tables/qc_by_disaster_type_sar.csv",
        "tables/qc_by_region_sar.csv",
    ]:
        copied = copy_optional(practice_root / rel, reports_dir / "source_practice_reports" / rel)
        if copied:
            needed_practice_files.add(to_data_rel(source_data_root, practice_root / rel))

    write_needed_file(reports_dir / "needed_raw_files.txt", needed_raw_files)
    write_needed_file(reports_dir / "needed_practice_files.txt", needed_practice_files)
    write_needed_file(reports_dir / "needed_prior_files.txt", needed_prior_files)
    write_json(reports_dir / "minimal_package_summary.json", summary)
    readme = f"""# DisasterM3 Optical-SAR Building Damage Minimal Package

This package contains only the QC-clean optical-SAR building damage subset for the Stage-2 mainline.

- train: {EXPECTED_COUNTS["train"]}
- val: {EXPECTED_COUNTS["val"]}
- test: {EXPECTED_COUNTS["test"]}
- Stage-1 prior: {MODEL_NAME}
- Stage-1 binary threshold: {threshold}

Stage-2 manifests under `manifests/` use paths relative to this package root.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if summary["missing_priors"]:
        raise SystemExit("minimal package built without complete predicted priors")


if __name__ == "__main__":
    main()
