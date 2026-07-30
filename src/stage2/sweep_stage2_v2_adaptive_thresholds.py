#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Candidate:
    name: str
    mode: str
    damage_threshold: float
    evidence_stat: str
    evidence_gate: float


def parse_float_list(raw: str | None, default: list[float]) -> list[float]:
    if raw:
        values = [float(item) for item in raw.split(",") if item.strip()]
    else:
        values = default
    return sorted({round(float(value), 6) for value in values})


def parse_stats(raw: str | None) -> list[str]:
    values = [item.strip() for item in (raw or "p95,p99,max").split(",") if item.strip()]
    allowed = {"p90", "p95", "p99", "max"}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown evidence stats: {unknown}; allowed={sorted(allowed)}")
    return values


def label_float(value: float) -> str:
    text = f"{value:.6g}".replace("-", "m").replace(".", "p")
    return text


def make_candidates(damage_thresholds: list[float], evidence_stats: list[str], evidence_gates: list[float]) -> list[Candidate]:
    candidates = [Candidate("argmax", "argmax", 0.0, "", 0.0)]
    for threshold in damage_thresholds:
        candidates.append(
            Candidate(
                name=f"global_t{label_float(threshold)}",
                mode="global",
                damage_threshold=float(threshold),
                evidence_stat="",
                evidence_gate=0.0,
            )
        )
    for threshold in damage_thresholds:
        for stat in evidence_stats:
            for gate in evidence_gates:
                candidates.append(
                    Candidate(
                        name=f"adaptive_t{label_float(threshold)}_{stat}_g{label_float(gate)}",
                        mode="adaptive",
                        damage_threshold=float(threshold),
                        evidence_stat=stat,
                        evidence_gate=float(gate),
                    )
                )
    return candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep Stage-2 v2 sample-adaptive damage thresholds")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--damage-thresholds", help="Comma-separated pixel thresholds. Default: -0.25,-0.375,-0.5,-0.625")
    parser.add_argument(
        "--evidence-gates",
        help="Comma-separated sample-level gates. Default: -0.5,-0.25,0,0.25,0.5,0.75,1,1.25,1.5,2",
    )
    parser.add_argument("--evidence-stats", help="Comma-separated stats from p90,p95,p99,max. Default: p95,p99,max")
    parser.add_argument("--evidence-region", default="prior_support", choices=["prior_support", "all"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--bo-drop-tolerance", type=float, default=0.005)
    parser.add_argument("--pred-gate-drop-tolerance", type=float, default=0.005)
    return parser.parse_args()


def evidence_values(damage_margin: np.ndarray, support: np.ndarray, stats: list[str], region: str) -> dict[str, np.ndarray]:
    out = {stat: np.zeros((damage_margin.shape[0],), dtype=np.float32) for stat in stats}
    for index in range(damage_margin.shape[0]):
        if region == "prior_support":
            values = damage_margin[index][support[index]]
            if values.size == 0:
                values = damage_margin[index].reshape(-1)
        else:
            values = damage_margin[index].reshape(-1)
        for stat in stats:
            if stat == "max":
                out[stat][index] = float(np.max(values))
            else:
                percentile = float(stat[1:])
                out[stat][index] = float(np.percentile(values, percentile))
    return out


def row_from_metrics(
    candidate: Candidate,
    metrics: dict[str, float],
    baseline: dict[str, float] | None,
    active_count: int,
    sample_count: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": candidate.name,
        "mode": candidate.mode,
        "damage_threshold": candidate.damage_threshold,
        "evidence_stat": candidate.evidence_stat,
        "evidence_gate": candidate.evidence_gate,
        "active_sample_count": int(active_count),
        "active_sample_fraction": float(active_count / sample_count) if sample_count else 0.0,
    }
    for key in SUMMARY_METRICS:
        row[key] = metrics.get(key)
        if baseline is not None and key in baseline:
            row[f"delta_{key}"] = float(metrics[key]) - float(baseline[key])
    return row


def best_row(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    return max(rows, key=lambda row: float(row.get(metric, -1.0)))


def constrained_rows(
    rows: list[dict[str, Any]],
    baseline: dict[str, Any],
    bo_drop_tolerance: float,
    pred_gate_drop_tolerance: float,
) -> list[dict[str, Any]]:
    min_bo = float(baseline["building_only_macro_f1_3class"]) - float(bo_drop_tolerance)
    min_pred = float(baseline["predicted_gate_macro_f1_4class"]) - float(pred_gate_drop_tolerance)
    keep = [
        row
        for row in rows
        if float(row["building_only_macro_f1_3class"]) >= min_bo
        and float(row["predicted_gate_macro_f1_4class"]) >= min_pred
    ]
    return keep or [baseline]


def write_summary(output_dir: Path, best: dict[str, Any]) -> None:
    rows = [
        ("baseline", best["baseline"]),
        ("best_global_grade", best["best_global_grade"]),
        ("best_adaptive_grade", best["best_adaptive_grade"]),
        ("best_adaptive_damage_constrained", best["best_adaptive_damage_constrained"]),
        ("best_any_damage_constrained", best["best_any_damage_constrained"]),
    ]
    display_keys = [
        "name",
        "mode",
        "damage_threshold",
        "evidence_stat",
        "evidence_gate",
        "active_sample_fraction",
        "building_only_macro_f1_3class",
        "building_only_damage_macro_f1",
        "building_only_damage_binary_f1",
        "building_only_f1_intact",
        "predicted_gate_macro_f1_4class",
    ]

    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    lines = [
        "# Stage-2 v2 adaptive threshold sweep",
        "",
        "Split: validation-only diagnostic. Test set not evaluated.",
        "",
        f"Evidence region: `{best['evidence_region']}`",
        "",
        "## Selected Rows",
        "| label | " + " | ".join(display_keys) + " |",
        "| --- | " + " | ".join(["---"] * 5 + ["---:"] * (len(display_keys) - 5)) + " |",
    ]
    seen = set()
    for label, row in rows:
        if row["name"] in seen:
            continue
        seen.add(row["name"])
        lines.append("| " + " | ".join([label] + [fmt(row[key]) for key in display_keys]) + " |")
    lines.extend(
        [
            "",
            "## Best Names",
            f"- baseline: {best['baseline']['name']}",
            f"- best global grade: {best['best_global_grade']['name']}",
            f"- best adaptive grade: {best['best_adaptive_grade']['name']}",
            f"- best adaptive damage constrained: {best['best_adaptive_damage_constrained']['name']}",
            f"- best any damage constrained: {best['best_any_damage_constrained']['name']}",
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

    damage_thresholds = parse_float_list(args.damage_thresholds, [-0.25, -0.375, -0.5, -0.625])
    evidence_gates = parse_float_list(args.evidence_gates, [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    evidence_stats = parse_stats(args.evidence_stats)
    candidates = make_candidates(damage_thresholds, evidence_stats, evidence_gates)
    meters = {candidate.name: Stage2V2MeterBundle() for candidate in candidates}
    active_counts = {candidate.name: 0 for candidate in candidates}

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

    thresholds_for_grade = sorted({0.0, *damage_thresholds})
    candidate_by_name = {candidate.name: candidate for candidate in candidates}

    for batch in tqdm(loader, desc=f"adaptive_sweep_{args.split}"):
        image = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(image)
        true = batch["mask"].numpy().astype(np.uint8)
        support = batch["prior"][:, 0].numpy() >= gate_threshold
        damage_logits = logits[:, 1:3].detach()
        damage_choice = torch.argmax(damage_logits, dim=1).cpu().numpy().astype(np.uint8) + 2
        damage_margin = (torch.max(damage_logits, dim=1).values - logits[:, 0].detach()).float().cpu().numpy()
        grades = {
            threshold: np.where(damage_margin >= threshold, damage_choice, 1).astype(np.uint8)
            for threshold in thresholds_for_grade
        }
        evidences = evidence_values(damage_margin, support, evidence_stats, args.evidence_region)

        baseline_grade = grades[0.0]
        for name, candidate in candidate_by_name.items():
            if candidate.mode == "argmax":
                grade = baseline_grade
                active = np.zeros((baseline_grade.shape[0],), dtype=bool)
            elif candidate.mode == "global":
                grade = grades[candidate.damage_threshold]
                active = np.ones((baseline_grade.shape[0],), dtype=bool)
            else:
                active = evidences[candidate.evidence_stat] >= candidate.evidence_gate
                grade = np.where(active[:, None, None], grades[candidate.damage_threshold], baseline_grade)
            active_counts[name] += int(active.sum())
            meters[name].update(grade, true, support)

    baseline_metrics = meters["argmax"].compute()
    rows = [
        row_from_metrics(candidate, meters[candidate.name].compute(), baseline_metrics, active_counts[candidate.name], len(dataset))
        for candidate in candidates
    ]
    baseline = next(row for row in rows if row["name"] == "argmax")
    global_rows = [row for row in rows if row["mode"] == "global"]
    adaptive_rows = [row for row in rows if row["mode"] == "adaptive"]
    constrained = constrained_rows(rows, baseline, args.bo_drop_tolerance, args.pred_gate_drop_tolerance)
    constrained_adaptive = constrained_rows(adaptive_rows, baseline, args.bo_drop_tolerance, args.pred_gate_drop_tolerance)
    best = {
        "baseline": baseline,
        "best_global_grade": best_row(global_rows, "building_only_macro_f1_3class"),
        "best_adaptive_grade": best_row(adaptive_rows, "building_only_macro_f1_3class"),
        "best_adaptive_damage_constrained": best_row(constrained_adaptive, "building_only_damage_macro_f1"),
        "best_any_damage_constrained": best_row(constrained, "building_only_damage_macro_f1"),
        "bo_drop_tolerance": float(args.bo_drop_tolerance),
        "pred_gate_drop_tolerance": float(args.pred_gate_drop_tolerance),
        "candidate_count": len(candidates),
        "sample_count": len(dataset),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "split": args.split,
        "seed": seed,
        "gate_threshold": gate_threshold,
        "evidence_region": args.evidence_region,
        "damage_thresholds": damage_thresholds,
        "evidence_stats": evidence_stats,
        "evidence_gates": evidence_gates,
    }

    write_csv(output_dir / "adaptive_threshold_sweep.csv", rows)
    write_json(output_dir / "best_adaptive_thresholds.json", best)
    write_json(output_dir / "permutation.json", dataset.permutation_info)
    write_summary(output_dir, best)
    print(json.dumps(best, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
