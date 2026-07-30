#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/stage2_v2_prechange_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPO_ROOT / "outputs/stage2/v2_prechange_analysis_20260626"
ASSET_DIR = REPO_ROOT / "reports/stage2_v2/assets/prechange_results_20260626"
REPORT_PATH = REPO_ROOT / "reports/stage2_v2/PRECHANGE_EXPERIMENT_RESULTS_20260626.md"
STRICT_MEAN_STD = REPO_ROOT / "outputs/stage2/v2_strict_clean_analysis_20260625/mean_std.csv"


RUNS = {
    "O1": {
        "group": "oracle_baseline",
        "display": "O1 oracle prior only",
        "run_dir": "outputs/stage2/v2_oracle_explore/V2_ORACLE_O1_prior_only/seed_42/run_20260625_170524",
        "color": "#5B8FF9",
    },
    "O2": {
        "group": "oracle_baseline",
        "display": "O2 SAR + oracle prior",
        "run_dir": "outputs/stage2/v2_oracle_explore/V2_ORACLE_O2_sar_oracle_prior/seed_42/run_20260625_125918",
        "color": "#F6BD16",
    },
    "O3": {
        "group": "oracle_baseline",
        "display": "O3 shuffled SAR + oracle prior",
        "run_dir": "outputs/stage2/v2_oracle_explore/V2_ORACLE_O3_shuffled_sar_oracle_prior/seed_42/run_20260625_140145",
        "color": "#E8684A",
    },
    "C1": {
        "group": "prechange",
        "display": "C1 pre + prior, no SAR",
        "run_dir": "outputs/stage2/v2_prechange_explore/V2_PRE_C1_pre_oracle_prior_only/seed_42/run_20260626_085906",
        "color": "#7A77FF",
    },
    "C2": {
        "group": "prechange",
        "display": "C2 pre + paired SAR + prior",
        "run_dir": "outputs/stage2/v2_prechange_explore/V2_PRE_C2_pre_sar_oracle_prior/seed_42/run_20260626_085921",
        "color": "#00A870",
    },
    "C3": {
        "group": "prechange",
        "display": "C3 pre + shuffled SAR + prior",
        "run_dir": "outputs/stage2/v2_prechange_explore/V2_PRE_C3_pre_shuffled_sar_oracle_prior/seed_42/run_20260626_095628",
        "color": "#E8684A",
    },
    "C4": {
        "group": "prechange",
        "display": "C4 pre + SAR texture + prior",
        "run_dir": "outputs/stage2/v2_prechange_explore/V2_PRE_C4_pre_sartex_oracle_prior/seed_42/run_20260626_095637",
        "color": "#9270CA",
    },
    "C5": {
        "group": "prechange",
        "display": "C5 pre + shuffled SAR texture + prior",
        "run_dir": "outputs/stage2/v2_prechange_explore/V2_PRE_C5_pre_shuffled_sartex_oracle_prior/seed_42/run_20260626_121520",
        "color": "#269A99",
    },
}

METRICS = [
    "building_only_macro_f1_3class",
    "building_only_damage_macro_f1",
    "building_only_damage_binary_f1",
    "building_only_f1_intact",
    "building_only_f1_damaged",
    "building_only_f1_destroyed",
    "oracle_gate_macro_f1_4class",
]

METRIC_LABELS = {
    "building_only_macro_f1_3class": "BO macro",
    "building_only_damage_macro_f1": "damage macro",
    "building_only_damage_binary_f1": "binary damage",
    "building_only_f1_intact": "intact",
    "building_only_f1_damaged": "damaged",
    "building_only_f1_destroyed": "destroyed",
    "oracle_gate_macro_f1_4class": "oracle 4c macro",
}

KEY_DELTAS = [
    ("C2_minus_C1", "C2", "C1", "paired SAR gain over pre+prior"),
    ("C2_minus_C3", "C2", "C3", "paired SAR gain over shuffled SAR"),
    ("C2_minus_O2", "C2", "O2", "pre optical gain over O2"),
    ("C4_minus_C2", "C4", "C2", "SAR texture gain over raw SAR"),
    ("C4_minus_C5", "C4", "C5", "paired SAR texture gain over shuffled"),
    ("C4_minus_O2", "C4", "O2", "pre+texture gain over O2"),
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
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


def fmt(value: Any, ndigits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "NA"
    return f"{number:.{ndigits}f}"


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.22,
            "font.size": 9,
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def load_runs() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    run_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    completion_rows: list[dict[str, Any]] = []
    for exp, meta in RUNS.items():
        run_dir = REPO_ROOT / str(meta["run_dir"])
        completed = read_json(run_dir / "completed.json")
        cfg = read_json(run_dir / "config_resolved.json")
        best = completed.get("best_metrics", {}).get("best_bo_grade_macro_f1.pth", {})
        completion_rows.append(
            {
                "exp": exp,
                "display": meta["display"],
                "group": meta["group"],
                "run_dir": str(run_dir.relative_to(REPO_ROOT)),
                "input_mode": cfg.get("dataset", {}).get("input_mode"),
                "in_channels": cfg.get("model", {}).get("in_channels"),
                "sar_shuffle_mode": cfg.get("dataset", {}).get("sar_shuffle_mode"),
                "status": completed.get("status"),
                "epochs_completed": completed.get("epochs_completed"),
                "stopped_early": completed.get("stopped_early"),
                "best_epoch": best.get("epoch"),
                "best_val_bo_macro": best.get("value"),
            }
        )
        for split in ("val", "test"):
            metrics_path = run_dir / f"{split}_best_grade/metrics.json"
            metrics = read_json(metrics_path)
            row = {
                "split": split,
                "exp": exp,
                "display": meta["display"],
                "group": meta["group"],
                "run_dir": str(run_dir.relative_to(REPO_ROOT)),
                "sample_count": metrics.get("sample_count"),
                "checkpoint_epoch": metrics.get("checkpoint_epoch"),
                "sar_shuffle_mode": metrics.get("sar_shuffle_mode"),
            }
            for key in METRICS:
                row[key] = metrics.get(key)
            run_rows.append(row)
            for event_row in read_csv(run_dir / f"{split}_best_grade/per_event_metrics.csv"):
                out = {
                    "split": split,
                    "exp": exp,
                    "display": meta["display"],
                    "group": meta["group"],
                    "event_id": event_row["event_id"],
                }
                for key in METRICS:
                    if key in event_row:
                        out[key] = float(event_row[key])
                event_rows.append(out)
            for sample_row in read_csv(run_dir / f"{split}_best_grade/sample_metrics.csv"):
                out = {
                    "split": split,
                    "exp": exp,
                    "id": sample_row["id"],
                    "event_id": sample_row["event_id"],
                }
                for key in METRICS:
                    if key in sample_row:
                        out[key] = float(sample_row[key])
                sample_rows.append(out)
    return run_rows, event_rows, sample_rows, completion_rows


def compute_event_macro(event_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in event_rows:
        grouped[(row["split"], row["exp"])].append(row)
    rows: list[dict[str, Any]] = []
    for (split, exp), group in sorted(grouped.items()):
        out = {
            "split": split,
            "exp": exp,
            "display": RUNS[exp]["display"],
            "group": RUNS[exp]["group"],
            "event_count": len(group),
        }
        for key in METRICS:
            out[key] = float(np.mean([float(row[key]) for row in group]))
        rows.append(out)
    return rows


def compute_deltas(rows: list[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    by = {(row["split"], row["exp"]): row for row in rows}
    out_rows: list[dict[str, Any]] = []
    for split in ("val", "test"):
        for name, left, right, label in KEY_DELTAS:
            row = {"scope": scope, "split": split, "comparison": name, "label": label, "left": left, "right": right}
            for key in METRICS:
                row[key] = float(by[(split, left)][key]) - float(by[(split, right)][key])
            out_rows.append(row)
    return out_rows


def bootstrap_sample_deltas(sample_rows: list[dict[str, Any]], metric: str, reps: int = 5000) -> list[dict[str, Any]]:
    by: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    events: dict[str, str] = {}
    for row in sample_rows:
        by[(row["split"], row["exp"])][row["id"]] = float(row[metric])
        events[row["id"]] = str(row["event_id"])
    rng = np.random.default_rng(20260626)
    out: list[dict[str, Any]] = []
    for split in ("val", "test"):
        for name, left, right, label in KEY_DELTAS:
            ids = sorted(set(by[(split, left)]) & set(by[(split, right)]))
            diffs = np.asarray([by[(split, left)][sample_id] - by[(split, right)][sample_id] for sample_id in ids], dtype=np.float64)
            if len(diffs) == 0:
                continue
            boot = np.empty(reps, dtype=np.float64)
            for index in range(reps):
                sample = rng.integers(0, len(diffs), size=len(diffs))
                boot[index] = float(np.mean(diffs[sample]))
            out.append(
                {
                    "split": split,
                    "comparison": name,
                    "label": label,
                    "metric": metric,
                    "sample_count": len(diffs),
                    "mean_delta": float(np.mean(diffs)),
                    "median_delta": float(np.median(diffs)),
                    "ci95_low": float(np.percentile(boot, 2.5)),
                    "ci95_high": float(np.percentile(boot, 97.5)),
                    "positive_fraction": float(np.mean(diffs > 0)),
                    "event_count": len({events[sample_id] for sample_id in ids}),
                }
            )
    return out


def load_strict_reference() -> list[dict[str, Any]]:
    if not STRICT_MEAN_STD.is_file():
        return []
    rows = []
    for row in read_csv(STRICT_MEAN_STD):
        if row["split"] != "test":
            continue
        rows.append(
            {
                "split": "test",
                "exp": row["exp"],
                "display": row["display"],
                "seed_count": int(row["seed_count"]),
                "building_only_macro_f1_3class_mean": float(row["building_only_macro_f1_3class_mean"]),
                "building_only_damage_macro_f1_mean": float(row["building_only_damage_macro_f1_mean"]),
                "building_only_damage_binary_f1_mean": float(row["building_only_damage_binary_f1_mean"]),
                "building_only_f1_intact_mean": float(row["building_only_f1_intact_mean"]),
                "building_only_f1_damaged_mean": float(row["building_only_f1_damaged_mean"]),
                "building_only_f1_destroyed_mean": float(row["building_only_f1_destroyed_mean"]),
            }
        )
    return rows


def plot_global_metrics(run_rows: list[dict[str, Any]]) -> None:
    experiments = ["O1", "O2", "O3", "C1", "C2", "C3", "C4", "C5"]
    metrics = [
        "building_only_macro_f1_3class",
        "building_only_damage_macro_f1",
        "building_only_damage_binary_f1",
        "building_only_f1_damaged",
        "building_only_f1_destroyed",
    ]
    by = {(row["split"], row["exp"]): row for row in run_rows}
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharey=True)
    x = np.arange(len(metrics))
    width = 0.1
    for ax, split in zip(axes, ("val", "test"), strict=True):
        for idx, exp in enumerate(experiments):
            row = by[(split, exp)]
            values = [float(row[key]) for key in metrics]
            ax.bar(x + (idx - 3.5) * width, values, width, color=RUNS[exp]["color"], label=exp)
        ax.set_title(f"{split} global metrics")
        ax.set_xticks(x, [METRIC_LABELS[key] for key in metrics], rotation=20, ha="right")
        ax.set_ylim(0, 0.85)
        ax.set_ylabel("F1")
    axes[1].legend(ncol=4, fontsize=8)
    save_figure(fig, ASSET_DIR / "prechange_global_metrics.png")


def plot_event_macro(event_macro_rows: list[dict[str, Any]]) -> None:
    experiments = ["O1", "O2", "O3", "C1", "C2", "C3", "C4", "C5"]
    metrics = [
        "building_only_macro_f1_3class",
        "building_only_damage_macro_f1",
        "building_only_damage_binary_f1",
        "building_only_f1_damaged",
        "building_only_f1_destroyed",
    ]
    by = {(row["split"], row["exp"]): row for row in event_macro_rows}
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharey=True)
    x = np.arange(len(metrics))
    width = 0.1
    for ax, split in zip(axes, ("val", "test"), strict=True):
        for idx, exp in enumerate(experiments):
            row = by[(split, exp)]
            values = [float(row[key]) for key in metrics]
            ax.bar(x + (idx - 3.5) * width, values, width, color=RUNS[exp]["color"], label=exp)
        ax.set_title(f"{split} event-macro metrics")
        ax.set_xticks(x, [METRIC_LABELS[key] for key in metrics], rotation=20, ha="right")
        ax.set_ylim(0, 0.85)
        ax.set_ylabel("mean over events")
    axes[1].legend(ncol=4, fontsize=8)
    save_figure(fig, ASSET_DIR / "prechange_event_macro_metrics.png")


def plot_key_deltas(deltas: list[dict[str, Any]], scope: str) -> None:
    rows = [row for row in deltas if row["scope"] == scope and row["split"] == "test"]
    metrics = ["building_only_macro_f1_3class", "building_only_damage_macro_f1", "building_only_f1_damaged", "building_only_f1_destroyed"]
    x = np.arange(len(rows))
    width = 0.2
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    for idx, metric in enumerate(metrics):
        values = [float(row[metric]) for row in rows]
        ax.bar(x + (idx - 1.5) * width, values, width, label=METRIC_LABELS[metric])
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_xticks(x, [row["comparison"].replace("_minus_", " - ") for row in rows], rotation=25, ha="right")
    ax.set_ylabel("test delta")
    ax.set_title(f"Key test deltas ({scope})")
    ax.legend(ncol=4, fontsize=8)
    save_figure(fig, ASSET_DIR / f"prechange_key_deltas_{scope}.png")


def plot_event_heatmap(event_rows: list[dict[str, Any]]) -> None:
    experiments = ["O2", "C1", "C2", "C3", "C4", "C5"]
    rows = [row for row in event_rows if row["split"] == "test" and row["exp"] in experiments]
    events = sorted({row["event_id"] for row in rows})
    by = {(row["event_id"], row["exp"]): float(row["building_only_macro_f1_3class"]) for row in rows}
    data = np.asarray([[by[(event, exp)] for exp in experiments] for event in events], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    im = ax.imshow(data, vmin=0, vmax=max(0.65, float(data.max())), cmap="viridis")
    ax.set_xticks(np.arange(len(experiments)), experiments)
    ax.set_yticks(np.arange(len(events)), events)
    ax.set_title("Test per-event BO macro F1")
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            ax.text(x, y, f"{data[y, x]:.2f}", ha="center", va="center", color="white" if data[y, x] < 0.35 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.82)
    save_figure(fig, ASSET_DIR / "prechange_test_event_heatmap.png")


def make_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def write_report(
    run_rows: list[dict[str, Any]],
    event_macro_rows: list[dict[str, Any]],
    deltas: list[dict[str, Any]],
    event_deltas: list[dict[str, Any]],
    bootstrap_rows: list[dict[str, Any]],
    completion_rows: list[dict[str, Any]],
    strict_rows: list[dict[str, Any]],
) -> None:
    by = {(row["split"], row["exp"]): row for row in run_rows}
    event_by = {(row["split"], row["exp"]): row for row in event_macro_rows}
    delta_by = {(row["scope"], row["split"], row["comparison"]): row for row in deltas + event_deltas}
    boot_by = {(row["split"], row["comparison"]): row for row in bootstrap_rows if row["metric"] == "building_only_macro_f1_3class"}

    completion_table = make_table(
        ["ID", "input", "channels", "shuffle", "epochs", "best val BO", "run"],
        [
            [
                row["exp"],
                str(row["input_mode"]),
                str(row["in_channels"]),
                str(row["sar_shuffle_mode"]),
                str(row["epochs_completed"]),
                fmt(row["best_val_bo_macro"]),
                row["run_dir"],
            ]
            for row in completion_rows
            if row["exp"].startswith("C")
        ],
    )
    global_table = make_table(
        ["split", "ID", "BO macro", "damage macro", "binary damage", "intact", "damaged", "destroyed"],
        [
            [
                split,
                exp,
                fmt(by[(split, exp)]["building_only_macro_f1_3class"]),
                fmt(by[(split, exp)]["building_only_damage_macro_f1"]),
                fmt(by[(split, exp)]["building_only_damage_binary_f1"]),
                fmt(by[(split, exp)]["building_only_f1_intact"]),
                fmt(by[(split, exp)]["building_only_f1_damaged"]),
                fmt(by[(split, exp)]["building_only_f1_destroyed"]),
            ]
            for split in ("val", "test")
            for exp in ("O1", "O2", "O3", "C1", "C2", "C3", "C4", "C5")
        ],
    )
    event_table = make_table(
        ["split", "ID", "event BO macro", "event damage macro", "event binary", "event damaged", "event destroyed"],
        [
            [
                split,
                exp,
                fmt(event_by[(split, exp)]["building_only_macro_f1_3class"]),
                fmt(event_by[(split, exp)]["building_only_damage_macro_f1"]),
                fmt(event_by[(split, exp)]["building_only_damage_binary_f1"]),
                fmt(event_by[(split, exp)]["building_only_f1_damaged"]),
                fmt(event_by[(split, exp)]["building_only_f1_destroyed"]),
            ]
            for split in ("val", "test")
            for exp in ("O1", "O2", "O3", "C1", "C2", "C3", "C4", "C5")
        ],
    )
    delta_table_rows = []
    for comparison, *_ in KEY_DELTAS:
        g = delta_by[("global", "test", comparison)]
        e = delta_by[("event_macro", "test", comparison)]
        b = boot_by.get(("test", comparison), {})
        delta_table_rows.append(
            [
                comparison.replace("_minus_", " - "),
                fmt(g["building_only_macro_f1_3class"]),
                fmt(g["building_only_damage_macro_f1"]),
                fmt(g["building_only_f1_damaged"]),
                fmt(g["building_only_f1_destroyed"]),
                fmt(e["building_only_macro_f1_3class"]),
                f"{fmt(b.get('mean_delta'))} [{fmt(b.get('ci95_low'))}, {fmt(b.get('ci95_high'))}]" if b else "NA",
            ]
        )
    delta_table = make_table(
        ["comparison", "global ΔBO", "global Δdamage", "global Δdamaged", "global Δdestroyed", "event ΔBO", "sample mean ΔBO 95% CI"],
        delta_table_rows,
    )

    strict_table = ""
    if strict_rows:
        strict_table = "\n\n" + make_table(
            ["formal ID", "test BO macro mean", "damage macro", "binary damage", "damaged", "destroyed"],
            [
                [
                    row["display"],
                    fmt(row["building_only_macro_f1_3class_mean"]),
                    fmt(row["building_only_damage_macro_f1_mean"]),
                    fmt(row["building_only_damage_binary_f1_mean"]),
                    fmt(row["building_only_f1_damaged_mean"]),
                    fmt(row["building_only_f1_destroyed_mean"]),
                ]
                for row in strict_rows
            ],
        )

    text = f"""# Stage-2 v2 pre-change optical / SAR-enhancement results

日期：2026-06-26

本报告汇总 clean human-reviewed strict 数据集上的 pre-change optical / SAR-enhancement 探索实验。C1-C5 均为 seed 42，使用 oracle building support；它们是诊断实验，不等同于可部署 predicted-prior pipeline。正式 deployable reference 仍是 strict clean A1-A4 三种子结果。

## 1. 完成状态

{completion_table}

所有 C1-C5 训练、validation grade checkpoint 评估、test grade checkpoint 评估均已完成。

## 2. Global pixel metrics

{global_table}

## 3. Event-macro metrics

{event_table}

## 4. Key test deltas

{delta_table}

## 5. Formal strict clean reference

以下是同一 clean strict 数据集上的 predicted-prior formal reference（三种子 mean）。这些是可部署 pipeline 对照；C1-C5 是 oracle-support 诊断上界/机制实验。{strict_table}

## 6. 结论

1. Pre-event optical 明显改善 oracle-support 诊断。Test global BO macro 从 O2 `0.2645` 提升到 C2 `0.4646`，C2 也高于 C1 `0.3789` 和 C3 `0.3840`。这说明 pre optical 不只是替代 prior，它确实让 paired SAR 在 global test 上更可用。
2. SAR correspondence 有正向证据，但仍不能视为最终定论。C2-C3 的 test global BO macro 为 `+0.0806`，event-macro 为 `+0.0720`；但 sample-level BO macro mean delta 的 95% CI 为 `[-0.0076, 0.0178]`，跨 0。同时 C3 的 damaged F1 高于 C2，而 C2 主要提升 intact 和 destroyed。这说明 paired SAR 的增益主要体现在更好的等级平衡，尤其 destroyed，而不是简单提高 damaged recall。
3. SAR texture/gradient 有局部帮助，但不是 C2 的直接升级。C4 的 test event-macro BO macro `0.3748` 是本组最高，说明 texture 对跨事件平均更有价值；但 C4 global BO macro `0.4446` 低于 C2，且 damaged F1 低。它更偏向提高 destroyed / intact，牺牲 damaged。
4. Val 和 Test 排名不一致。Val 上 O2/C2/C4 排名不能稳定预测 test 排名，说明该 clean split 的 val 仍不能完全代表 test event/class composition。后续配方选择必须继续保留 shuffled controls 和 event-macro，而不是只看 val global。
5. 当前最好的 oracle-support diagnostic 是：global test 用 C2，event-macro/test destroyed 用 C4。若目标是可部署 pipeline，下一步应把 C2 的输入思想迁移到 predicted-prior support，并继续保留 C3/C5 这种 fixed shuffled-SAR control。

## 7. 产物

- CSV 汇总：`outputs/stage2/v2_prechange_analysis_20260626/`
- 图像：`reports/stage2_v2/assets/prechange_results_20260626/`
- 关键图：
  - `prechange_global_metrics.png`
  - `prechange_event_macro_metrics.png`
  - `prechange_key_deltas_global.png`
  - `prechange_key_deltas_event_macro.png`
  - `prechange_test_event_heatmap.png`
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    setup_plot_style()
    run_rows, event_rows, sample_rows, completion_rows = load_runs()
    event_macro_rows = compute_event_macro(event_rows)
    deltas = compute_deltas(run_rows, "global")
    event_deltas = compute_deltas(event_macro_rows, "event_macro")
    bootstrap_rows = bootstrap_sample_deltas(sample_rows, "building_only_macro_f1_3class")
    strict_rows = load_strict_reference()

    write_csv(ANALYSIS_DIR / "runs.csv", run_rows)
    write_csv(ANALYSIS_DIR / "per_event.csv", event_rows)
    write_csv(ANALYSIS_DIR / "event_macro.csv", event_macro_rows)
    write_csv(ANALYSIS_DIR / "deltas_global.csv", deltas)
    write_csv(ANALYSIS_DIR / "deltas_event_macro.csv", event_deltas)
    write_csv(ANALYSIS_DIR / "sample_bootstrap_deltas.csv", bootstrap_rows)
    write_csv(ANALYSIS_DIR / "completion.csv", completion_rows)
    if strict_rows:
        write_csv(ANALYSIS_DIR / "strict_formal_reference.csv", strict_rows)

    plot_global_metrics(run_rows)
    plot_event_macro(event_macro_rows)
    plot_key_deltas(deltas, "global")
    plot_key_deltas(event_deltas, "event_macro")
    plot_event_heatmap(event_rows)
    write_report(run_rows, event_macro_rows, deltas, event_deltas, bootstrap_rows, completion_rows, strict_rows)
    print(f"wrote {ANALYSIS_DIR.relative_to(REPO_ROOT)}")
    print(f"wrote {ASSET_DIR.relative_to(REPO_ROOT)}")
    print(f"wrote {REPORT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
