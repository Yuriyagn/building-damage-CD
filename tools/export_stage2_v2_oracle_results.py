#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/stage2_v2_oracle_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO_ROOT / "outputs/stage2/v2_oracle_explore"
ANALYSIS_DIR = REPO_ROOT / "outputs/stage2/v2_oracle_explore_analysis_20260625"
ASSET_DIR = REPO_ROOT / "reports/stage2_v2/assets/oracle_explore_20260625"
REPORT_PATH = REPO_ROOT / "reports/stage2_v2/ORACLE_PRIOR_EXPLORATION_RESULTS_20260625.md"
STRICT_ANALYSIS = REPO_ROOT / "outputs/stage2/v2_strict_clean_analysis_20260625/mean_std.csv"


EXPERIMENTS = {
    "O1": {
        "dir": "V2_ORACLE_O1_prior_only",
        "display": "O1 oracle-prior only",
        "color": "#5B8FF9",
    },
    "O2": {
        "dir": "V2_ORACLE_O2_sar_oracle_prior",
        "display": "O2 paired SAR + oracle prior",
        "color": "#F6BD16",
    },
    "O3": {
        "dir": "V2_ORACLE_O3_shuffled_sar_oracle_prior",
        "display": "O3 shuffled SAR + oracle prior",
        "color": "#E8684A",
    },
}

METRICS = [
    ("building_only_macro_f1_3class", "BO grade macro F1"),
    ("building_only_damage_macro_f1", "BO damage macro F1"),
    ("building_only_damage_binary_f1", "BO damage binary F1"),
    ("building_only_f1_intact", "Intact F1"),
    ("building_only_f1_damaged", "Damaged F1"),
    ("building_only_f1_destroyed", "Destroyed F1"),
    ("oracle_gate_macro_f1_4class", "Oracle-gated 4-class macro F1"),
    ("oracle_gate_miou_4class", "Oracle-gated 4-class mIoU"),
]

MAIN_METRIC_KEYS = [
    "building_only_macro_f1_3class",
    "building_only_damage_macro_f1",
    "building_only_damage_binary_f1",
    "building_only_f1_intact",
    "building_only_f1_damaged",
    "building_only_f1_destroyed",
    "oracle_gate_macro_f1_4class",
    "oracle_gate_miou_4class",
    "sample_count",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def find_run_dir(exp_dir: str) -> Path:
    markers = sorted((OUT_ROOT / exp_dir / "seed_42").glob("run_*/completed.json"))
    if len(markers) != 1:
        raise RuntimeError(f"expected one completed run for {exp_dir}, found {len(markers)}: {markers}")
    return markers[0].parent


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


def load_oracle_runs() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    run_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    completion_rows: list[dict[str, Any]] = []
    for exp, meta in EXPERIMENTS.items():
        run_dir = find_run_dir(meta["dir"])
        completed = read_json(run_dir / "completed.json")
        best_metrics = completed.get("best_metrics", {})
        best_grade = best_metrics.get("best_bo_grade_macro_f1.pth", {})
        best_damage = best_metrics.get("best_bo_damage_macro_f1.pth", {})
        completion_rows.append(
            {
                "exp": exp,
                "display": meta["display"],
                "run_dir": str(run_dir.relative_to(REPO_ROOT)),
                "status": completed.get("status"),
                "epochs_completed": completed.get("epochs_completed"),
                "stopped_early": completed.get("stopped_early"),
                "best_bo_grade_macro_f1_epoch": best_grade.get("epoch"),
                "best_bo_grade_macro_f1_val": best_grade.get("value"),
                "best_bo_damage_macro_f1_epoch": best_damage.get("epoch"),
                "best_bo_damage_macro_f1_val": best_damage.get("value"),
            }
        )
        for split in ("val", "test"):
            metrics_path = run_dir / f"{split}_best_grade/metrics.json"
            metrics = read_json(metrics_path)
            row: dict[str, Any] = {
                "split": split,
                "exp": exp,
                "display": meta["display"],
                "seed": metrics.get("seed", 42),
                "run_dir": str(run_dir.relative_to(REPO_ROOT)),
                "metrics_path": str(metrics_path.relative_to(REPO_ROOT)),
                "checkpoint_epoch": metrics.get("checkpoint_epoch"),
                "sar_shuffle_mode": metrics.get("sar_shuffle_mode"),
            }
            for key in MAIN_METRIC_KEYS:
                row[key] = metrics.get(key)
            run_rows.append(row)
            per_event_path = run_dir / f"{split}_best_grade/per_event_metrics.csv"
            for event_row in read_csv(per_event_path):
                out = {
                    "split": split,
                    "exp": exp,
                    "display": meta["display"],
                    "event_id": event_row["event_id"],
                }
                for key in MAIN_METRIC_KEYS:
                    if key in event_row and event_row[key] != "":
                        out[key] = float(event_row[key])
                event_rows.append(out)
    return run_rows, event_rows, completion_rows


def load_strict_seed42_and_mean() -> list[dict[str, Any]]:
    if not STRICT_ANALYSIS.is_file():
        return []
    rows = read_csv(STRICT_ANALYSIS)
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["split"] != "test":
            continue
        if row["exp"] != "V2S_A3_sar_predicted_prior":
            continue
        out.append(
            {
                "split": "test",
                "exp": "A3 strict predicted-prior mean",
                "display": "Strict A3 SAR+pred prior mean(3 seeds)",
                "building_only_macro_f1_3class": float(row["building_only_macro_f1_3class_mean"]),
                "building_only_damage_macro_f1": float(row["building_only_damage_macro_f1_mean"]),
                "building_only_damage_binary_f1": float(row["building_only_damage_binary_f1_mean"]),
                "building_only_f1_intact": float(row["building_only_f1_intact_mean"]),
                "building_only_f1_damaged": float(row["building_only_f1_damaged_mean"]),
                "building_only_f1_destroyed": float(row["building_only_f1_destroyed_mean"]),
                "oracle_gate_macro_f1_4class": None,
                "oracle_gate_miou_4class": None,
                "sample_count": float(row["sample_count_mean"]),
            }
        )
    return out


def compute_deltas(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by = {(row["split"], row["exp"]): row for row in run_rows}
    comparisons = [
        ("O2_minus_O1", "paired SAR gain over oracle-prior only", "O2", "O1"),
        ("O2_minus_O3", "paired SAR gain over shuffled SAR control", "O2", "O3"),
    ]
    rows: list[dict[str, Any]] = []
    for split in ("val", "test"):
        for name, label, left, right in comparisons:
            left_row = by[(split, left)]
            right_row = by[(split, right)]
            out: dict[str, Any] = {"split": split, "comparison": name, "label": label}
            for key in MAIN_METRIC_KEYS:
                if key == "sample_count":
                    continue
                lv = left_row.get(key)
                rv = right_row.get(key)
                out[key] = None if lv is None or rv is None else float(lv) - float(rv)
            rows.append(out)
    return rows


def compute_event_macro(event_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in event_rows:
        grouped[(row["split"], row["exp"])].append(row)
    rows: list[dict[str, Any]] = []
    for (split, exp), group in sorted(grouped.items()):
        out: dict[str, Any] = {
            "split": split,
            "exp": exp,
            "display": EXPERIMENTS[exp]["display"],
            "event_count": len(group),
        }
        for key in MAIN_METRIC_KEYS:
            if key == "sample_count":
                continue
            values = [float(row[key]) for row in group if key in row]
            out[key] = float(np.mean(values)) if values else None
        rows.append(out)
    return rows


def plot_val_test_metrics(run_rows: list[dict[str, Any]]) -> None:
    metrics = [
        ("building_only_macro_f1_3class", "Grade macro"),
        ("building_only_damage_macro_f1", "Damage macro"),
        ("building_only_f1_intact", "Intact"),
        ("building_only_f1_damaged", "Damaged"),
        ("building_only_f1_destroyed", "Destroyed"),
    ]
    by = {(row["split"], row["exp"]): row for row in run_rows}
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), sharey=True)
    x = np.arange(len(metrics))
    width = 0.24
    for ax, split in zip(axes, ("val", "test"), strict=True):
        for idx, (exp, meta) in enumerate(EXPERIMENTS.items()):
            row = by[(split, exp)]
            values = [float(row[key]) for key, _ in metrics]
            ax.bar(x + (idx - 1) * width, values, width, label=exp, color=meta["color"])
        ax.set_title(f"{split} best-grade checkpoint")
        ax.set_xticks(x, [label for _, label in metrics], rotation=20, ha="right")
        ax.set_ylim(0.0, 0.85)
        ax.set_ylabel("F1")
    axes[1].legend([meta["display"] for meta in EXPERIMENTS.values()], loc="upper right", fontsize=8)
    save_figure(fig, ASSET_DIR / "oracle_val_test_metrics.png")


def plot_event_macro(event_macro_rows: list[dict[str, Any]]) -> None:
    metrics = [
        ("building_only_macro_f1_3class", "Grade macro"),
        ("building_only_damage_macro_f1", "Damage macro"),
        ("building_only_damage_binary_f1", "Damage binary"),
    ]
    by = {(row["split"], row["exp"]): row for row in event_macro_rows}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
    x = np.arange(len(metrics))
    width = 0.24
    for ax, split in zip(axes, ("val", "test"), strict=True):
        for idx, (exp, meta) in enumerate(EXPERIMENTS.items()):
            row = by[(split, exp)]
            values = [float(row[key]) for key, _ in metrics]
            ax.bar(x + (idx - 1) * width, values, width, color=meta["color"], label=exp)
        ax.set_title(f"{split} event-macro")
        ax.set_xticks(x, [label for _, label in metrics])
        ax.set_ylim(0.0, 0.65)
        ax.set_ylabel("Mean over events")
    axes[1].legend([meta["display"] for meta in EXPERIMENTS.values()], loc="upper right", fontsize=8)
    save_figure(fig, ASSET_DIR / "oracle_event_macro.png")


def plot_test_per_event(event_rows: list[dict[str, Any]]) -> None:
    test_rows = [row for row in event_rows if row["split"] == "test"]
    events = sorted({row["event_id"] for row in test_rows})
    x = np.arange(len(events))
    width = 0.24
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    by = {(row["event_id"], row["exp"]): row for row in test_rows}
    for idx, (exp, meta) in enumerate(EXPERIMENTS.items()):
        values = [float(by[(event, exp)]["building_only_macro_f1_3class"]) for event in events]
        ax.bar(x + (idx - 1) * width, values, width, color=meta["color"], label=meta["display"])
    ax.set_title("Oracle exploration test per-event BO grade macro F1")
    ax.set_xticks(x, events, rotation=25, ha="right")
    ax.set_ylabel("BO grade macro F1")
    ax.set_ylim(0.0, 0.55)
    ax.legend(fontsize=8)
    save_figure(fig, ASSET_DIR / "oracle_test_per_event_grade_macro.png")


def make_markdown_table(rows: list[list[str]], headers: list[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_report(
    run_rows: list[dict[str, Any]],
    deltas: list[dict[str, Any]],
    event_macro_rows: list[dict[str, Any]],
    completion_rows: list[dict[str, Any]],
    strict_reference: list[dict[str, Any]],
) -> None:
    by = {(row["split"], row["exp"]): row for row in run_rows}
    delta_by = {(row["split"], row["comparison"]): row for row in deltas}
    event_by = {(row["split"], row["exp"]): row for row in event_macro_rows}

    completion_table = make_markdown_table(
        [
            [
                row["exp"],
                row["display"],
                str(row["status"]),
                str(row["epochs_completed"]),
                str(row["stopped_early"]),
                str(row["best_bo_grade_macro_f1_epoch"]),
                fmt(row["best_bo_grade_macro_f1_val"]),
                row["run_dir"],
            ]
            for row in completion_rows
        ],
        ["ID", "实验", "状态", "epochs", "early stop", "best epoch", "best val BO macro", "run dir"],
    )

    metric_table_rows: list[list[str]] = []
    for split in ("val", "test"):
        for exp, meta in EXPERIMENTS.items():
            row = by[(split, exp)]
            metric_table_rows.append(
                [
                    split,
                    exp,
                    fmt(row["building_only_macro_f1_3class"]),
                    fmt(row["building_only_damage_macro_f1"]),
                    fmt(row["building_only_damage_binary_f1"]),
                    fmt(row["building_only_f1_intact"]),
                    fmt(row["building_only_f1_damaged"]),
                    fmt(row["building_only_f1_destroyed"]),
                    fmt(row["oracle_gate_macro_f1_4class"]),
                ]
            )
    metric_table = make_markdown_table(
        metric_table_rows,
        ["split", "ID", "BO macro", "damage macro", "binary damage", "intact", "damaged", "destroyed", "oracle 4c macro"],
    )

    delta_rows: list[list[str]] = []
    for split in ("val", "test"):
        for comparison in ("O2_minus_O1", "O2_minus_O3"):
            row = delta_by[(split, comparison)]
            delta_rows.append(
                [
                    split,
                    row["label"],
                    fmt(row["building_only_macro_f1_3class"], 4),
                    fmt(row["building_only_damage_macro_f1"], 4),
                    fmt(row["building_only_damage_binary_f1"], 4),
                    fmt(row["building_only_f1_damaged"], 4),
                    fmt(row["building_only_f1_destroyed"], 4),
                ]
            )
    delta_table = make_markdown_table(
        delta_rows,
        ["split", "comparison", "Δ BO macro", "Δ damage macro", "Δ binary damage", "Δ damaged", "Δ destroyed"],
    )

    event_macro_table_rows: list[list[str]] = []
    for split in ("val", "test"):
        for exp in EXPERIMENTS:
            row = event_by[(split, exp)]
            event_macro_table_rows.append(
                [
                    split,
                    exp,
                    str(row["event_count"]),
                    fmt(row["building_only_macro_f1_3class"]),
                    fmt(row["building_only_damage_macro_f1"]),
                    fmt(row["building_only_damage_binary_f1"]),
                    fmt(row["building_only_f1_damaged"]),
                    fmt(row["building_only_f1_destroyed"]),
                ]
            )
    event_macro_table = make_markdown_table(
        event_macro_table_rows,
        ["split", "ID", "events", "event BO macro", "event damage macro", "event binary damage", "event damaged", "event destroyed"],
    )

    strict_text = ""
    if strict_reference:
        strict = strict_reference[0]
        o2_test = by[("test", "O2")]
        strict_text = (
            "\n和正式 clean A3（predicted prior，3 seeds test mean）相比，O2 使用真实建筑 mask 后 "
            f"BO macro 为 `{fmt(o2_test['building_only_macro_f1_3class'])}`，"
            f"strict A3 mean 为 `{fmt(strict['building_only_macro_f1_3class'])}`；"
            f"damage macro 为 `{fmt(o2_test['building_only_damage_macro_f1'])}` vs "
            f"`{fmt(strict['building_only_damage_macro_f1'])}`。这说明本轮 oracle mask 不是简单把测试性能抬高，"
            "主要瓶颈仍在建筑内损伤分级与跨事件泛化，而不只是 Stage-1 建筑支持域。"
        )

    text = f"""# Stage-2 v2 oracle-prior exploration results

日期：2026-06-25

本报告汇总单 seed 探索实验：用真实标签建筑 mask 作为 Stage-2 prior/support，检验建筑内三分类能力。该实验用于诊断，不是正式可部署结果；正式系统仍应看 predicted-prior pipeline。

## 1. 完成状态

{completion_table}

所有 O1/O2/O3 训练均已完成，val/test 的 `best_bo_grade_macro_f1.pth` 评估产物齐全。当前没有运行中的 tmux 训练会话。

## 2. 全局指标

{metric_table}

图：

- `assets/oracle_explore_20260625/oracle_val_test_metrics.png`
- `assets/oracle_explore_20260625/oracle_event_macro.png`
- `assets/oracle_explore_20260625/oracle_test_per_event_grade_macro.png`

## 3. 关键差值

{delta_table}

解释：

- Val 上，O2 明显高于 O1/O3：paired SAR 在 oracle mask 条件下对开发集的 BO macro 有增益。
- Test global 上，O2 低于 O1（`-0.0729`），只比 O3 高 `+0.0047`；同时 binary damage F1 明显低于 O1/O3。这不支持“真实 SAR 对当前 strict test 有稳定可泛化增益”的结论。
- O1 test 的 damaged F1 很高但 destroyed F1 很低，说明 prior-only/bias 可以命中 `Damaged` 主导事件，却不能可靠分开损伤等级。
- O2 test 的 intact 较高，但 damaged/destroyed 都低，说明模型偏向完整建筑，损伤召回/分级没有稳定起来。
{strict_text}

## 4. 事件宏平均

{event_macro_table}

事件宏平均避免 `mexico_hurricane` 的像素量过度支配。Test event-macro 下 O2 的 BO macro 和 damage macro 最高，但 binary damage 仍低于 O1/O3；结合 global 指标，结论只能定为“有局部迹象，但不稳健”。这进一步说明 oracle support 后，主要问题仍是建筑内损伤分级与跨事件泛化。

## 5. 结论

本轮探索回答了一个关键问题：即使把 Stage-1 prior 替换为真实建筑 mask，Stage-2 在 strict clean test 上也没有自然变成可靠的损伤分级器。当前证据更符合：

```text
建筑支持域误差不是唯一瓶颈；建筑内三分类本身仍不稳定。
单时相 post-SAR 在当前数据/模型/损失下，对跨事件 damage grading 的可泛化增益不足。
```

下一步不建议继续只堆 Stage-2 网络结构。更有价值的方向是：

1. 增加 pre/post temporal cue 或 SAR change feature；
2. 做 per-event 失败样本可视化，确认 O2 在哪些事件把 damage 预测成 intact；
3. 若继续 Stage-2，先从 loss/sampler 调 damaged/destroyed recall，但仍必须用 shuffled-SAR control 约束解释。

## 6. 产物

- 汇总表：`outputs/stage2/v2_oracle_explore_analysis_20260625/oracle_runs.csv`
- 差值表：`outputs/stage2/v2_oracle_explore_analysis_20260625/oracle_deltas.csv`
- 事件表：`outputs/stage2/v2_oracle_explore_analysis_20260625/oracle_per_event.csv`
- 事件宏平均：`outputs/stage2/v2_oracle_explore_analysis_20260625/oracle_event_macro.csv`
- 图像目录：`reports/stage2_v2/assets/oracle_explore_20260625/`
"""
    write_text(REPORT_PATH, text)


def main() -> None:
    setup_plot_style()
    run_rows, event_rows, completion_rows = load_oracle_runs()
    deltas = compute_deltas(run_rows)
    event_macro_rows = compute_event_macro(event_rows)
    strict_reference = load_strict_seed42_and_mean()

    run_fieldnames = [
        "split",
        "exp",
        "display",
        "seed",
        "run_dir",
        "metrics_path",
        "checkpoint_epoch",
        "sar_shuffle_mode",
        *MAIN_METRIC_KEYS,
    ]
    event_fieldnames = ["split", "exp", "display", "event_id", *MAIN_METRIC_KEYS]
    delta_fieldnames = [
        "split",
        "comparison",
        "label",
        *[key for key in MAIN_METRIC_KEYS if key != "sample_count"],
    ]
    event_macro_fieldnames = [
        "split",
        "exp",
        "display",
        "event_count",
        *[key for key in MAIN_METRIC_KEYS if key != "sample_count"],
    ]
    completion_fieldnames = [
        "exp",
        "display",
        "run_dir",
        "status",
        "epochs_completed",
        "stopped_early",
        "best_bo_grade_macro_f1_epoch",
        "best_bo_grade_macro_f1_val",
        "best_bo_damage_macro_f1_epoch",
        "best_bo_damage_macro_f1_val",
    ]
    write_csv(ANALYSIS_DIR / "oracle_runs.csv", run_rows, run_fieldnames)
    write_csv(ANALYSIS_DIR / "oracle_per_event.csv", event_rows, event_fieldnames)
    write_csv(ANALYSIS_DIR / "oracle_deltas.csv", deltas, delta_fieldnames)
    write_csv(ANALYSIS_DIR / "oracle_event_macro.csv", event_macro_rows, event_macro_fieldnames)
    write_csv(ANALYSIS_DIR / "oracle_completion.csv", completion_rows, completion_fieldnames)
    if strict_reference:
        write_csv(ANALYSIS_DIR / "strict_a3_reference.csv", strict_reference, list(strict_reference[0].keys()))

    plot_val_test_metrics(run_rows)
    plot_event_macro(event_macro_rows)
    plot_test_per_event(event_rows)
    write_report(run_rows, deltas, event_macro_rows, completion_rows, strict_reference)

    print(f"wrote {ANALYSIS_DIR.relative_to(REPO_ROOT)}")
    print(f"wrote {ASSET_DIR.relative_to(REPO_ROOT)}")
    print(f"wrote {REPORT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
