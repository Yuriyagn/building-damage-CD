#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
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
from stage2.metrics_v2 import Stage2V2MeterBundle  # noqa: E402
from stage2.test_stage2_v2 import make_dataset  # noqa: E402


SUMMARY_METRICS = [
    "building_only_macro_f1_3class",
    "building_only_damage_macro_f1",
    "building_only_damage_binary_f1",
    "building_only_f1_intact",
    "building_only_f1_damaged",
    "building_only_f1_destroyed",
    "predicted_gate_macro_f1_4class",
    "predicted_gate_damage_binary_f1",
]


def parse_thresholds(raw: str | None) -> list[float]:
    if raw:
        values = [float(item) for item in raw.split(",") if item.strip()]
    else:
        values = np.linspace(-1.5, 1.5, 25).astype(float).tolist()
    values.append(0.0)
    rounded = sorted({round(float(value), 6) for value in values})
    return rounded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep Stage-2 v2 intact-vs-damage logit thresholds")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--thresholds", help="Comma-separated thresholds. Default: 25 values from -1.5 to 1.5.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--bo-drop-tolerance", type=float, default=0.005)
    parser.add_argument("--pred-gate-drop-tolerance", type=float, default=0.005)
    return parser.parse_args()


def _row(threshold: float, metrics: dict[str, float], baseline: dict[str, float] | None) -> dict[str, Any]:
    row: dict[str, Any] = {"threshold": float(threshold)}
    for key in SUMMARY_METRICS:
        row[key] = metrics.get(key)
        if baseline is not None and key in baseline:
            row[f"delta_{key}"] = float(metrics[key]) - float(baseline[key])
    return row


def _best(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    return max(rows, key=lambda row: float(row.get(metric, -1.0)))


def _write_summary(output_dir: Path, rows: list[dict[str, Any]], best: dict[str, Any]) -> None:
    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    display_keys = [
        "threshold",
        "building_only_macro_f1_3class",
        "building_only_damage_macro_f1",
        "building_only_damage_binary_f1",
        "building_only_f1_intact",
        "building_only_f1_damaged",
        "building_only_f1_destroyed",
        "predicted_gate_macro_f1_4class",
    ]
    interesting = [
        best["baseline"]["threshold"],
        best["best_grade"]["threshold"],
        best["best_damage_constrained"]["threshold"],
        best["best_damage"]["threshold"],
        best["best_predicted_gate"]["threshold"],
    ]
    selected = []
    seen = set()
    for threshold in interesting:
        for row in rows:
            if abs(float(row["threshold"]) - float(threshold)) < 1e-9 and threshold not in seen:
                selected.append(row)
                seen.add(threshold)
                break

    lines = [
        "# Stage-2 v2 threshold sweep",
        "",
        "Split: validation-only diagnostic. Test set not evaluated.",
        "",
        "## Selected Thresholds",
        "| " + " | ".join(display_keys) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(display_keys) - 1)) + " |",
    ]
    for row in selected:
        lines.append("| " + " | ".join(fmt(row[key]) for key in display_keys) + " |")
    lines.extend(
        [
            "",
            "## Best Rows",
            f"- baseline threshold: {best['baseline']['threshold']}",
            f"- best grade threshold: {best['best_grade']['threshold']}",
            f"- best damage threshold with BO/pred-gate tolerance: {best['best_damage_constrained']['threshold']}",
            f"- unconstrained best damage threshold: {best['best_damage']['threshold']}",
            f"- best predicted-gate threshold: {best['best_predicted_gate']['threshold']}",
        ]
    )
    (output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    seed = int(checkpoint.get("seed", dict(cfg.get("train", {})).get("seed", 42)))
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is unavailable; pass --allow-cpu only for a diagnostic sweep")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, no_pretrained=True).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    thresholds = parse_thresholds(args.thresholds)
    meters = {threshold: Stage2V2MeterBundle() for threshold in thresholds}

    data_root = Path(args.data_root)
    dataset = make_dataset(cfg, data_root, args.split, seed, args.limit)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    gate_threshold = float(dict(cfg.get("dataset", {})).get("gate_threshold", 0.6))
    amp = bool(dict(cfg.get("train", {})).get("amp", True)) and device.type == "cuda"

    for batch in tqdm(loader, desc=f"sweep_{args.split}"):
        image = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(image)
        true = batch["mask"].numpy().astype(np.uint8)
        support = batch["prior"][:, 0].numpy() >= gate_threshold
        damage_logits = logits[:, 1:3].detach()
        damage_choice = torch.argmax(damage_logits, dim=1).cpu().numpy().astype(np.uint8) + 2
        damage_margin = (torch.max(damage_logits, dim=1).values - logits[:, 0].detach()).float().cpu().numpy()
        for threshold, meter in meters.items():
            grade = np.where(damage_margin >= threshold, damage_choice, 1).astype(np.uint8)
            meter.update(grade, true, support)

    rows: list[dict[str, Any]] = []
    metrics_by_threshold: dict[float, dict[str, float]] = {}
    baseline_metrics: dict[str, float] | None = None
    for threshold in thresholds:
        metrics = meters[threshold].compute()
        metrics_by_threshold[threshold] = metrics
        if abs(threshold) < 1e-9:
            baseline_metrics = metrics
    if baseline_metrics is None:
        raise AssertionError("threshold list must include 0.0")
    for threshold in thresholds:
        rows.append(_row(threshold, metrics_by_threshold[threshold], baseline_metrics))

    baseline = next(row for row in rows if abs(float(row["threshold"])) < 1e-9)
    min_bo = float(baseline["building_only_macro_f1_3class"]) - float(args.bo_drop_tolerance)
    min_pred = float(baseline["predicted_gate_macro_f1_4class"]) - float(args.pred_gate_drop_tolerance)
    constrained = [
        row
        for row in rows
        if float(row["building_only_macro_f1_3class"]) >= min_bo
        and float(row["predicted_gate_macro_f1_4class"]) >= min_pred
    ]
    if not constrained:
        constrained = [baseline]
    best = {
        "baseline": baseline,
        "best_grade": _best(rows, "building_only_macro_f1_3class"),
        "best_damage": _best(rows, "building_only_damage_macro_f1"),
        "best_damage_constrained": _best(constrained, "building_only_damage_macro_f1"),
        "best_predicted_gate": _best(rows, "predicted_gate_macro_f1_4class"),
        "bo_drop_tolerance": float(args.bo_drop_tolerance),
        "pred_gate_drop_tolerance": float(args.pred_gate_drop_tolerance),
        "threshold_count": len(thresholds),
        "sample_count": len(dataset),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "split": args.split,
        "seed": seed,
        "gate_threshold": gate_threshold,
    }

    write_csv(output_dir / "threshold_sweep.csv", rows)
    write_json(output_dir / "best_thresholds.json", best)
    write_json(output_dir / "permutation.json", dataset.permutation_info)
    _write_summary(output_dir, rows, best)
    print(json.dumps(best, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
