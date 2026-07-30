#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import load_config  # noqa: E402
from models.build_model import build_model  # noqa: E402
from stage2.common import write_csv  # noqa: E402
from stage2.test_stage2_v2 import COLORS, make_dataset  # noqa: E402


EXPERIMENTS = {
    "A1": "V2_A1_prior_only",
    "A2": "V2_A2_sar_only",
    "A3": "V2_A3_sar_predicted_prior",
    "A4": "V2_A4_shuffled_sar_predicted_prior",
}
DISPLAY = {
    "V2_A1_prior_only": "A1 prior-only",
    "V2_A2_sar_only": "A2 SAR-only",
    "V2_A3_sar_predicted_prior": "A3 SAR + prior",
    "V2_A4_shuffled_sar_predicted_prior": "A4 shuffled SAR",
    "V2_B2_focal_sar_predicted_prior": "B2 focal",
    "V2_B3_disaster_balanced_sar_predicted_prior": "B3 disaster-balanced",
}
PALETTE = {
    "A0": "#777777",
    "A1": "#5B8FF9",
    "A2": "#61DDAA",
    "A3": "#F6BD16",
    "A4": "#E8684A",
    "B2": "#9270CA",
    "B3": "#269A99",
}
SELECTED_IDS = [
    "beriut_explosion_24",
    "turkey_earthquake4_327",
    "mexico_hurricane_00000042",
    "la_palma_volcano_00000620",
    "libya_flood_89",
    "ukraine_conflict_00000423",
    "marshall_wildfire_00000035",
    "noto_earthquake_172",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def unique_glob(pattern: str) -> Path:
    paths = sorted(REPO_ROOT.glob(pattern))
    if len(paths) != 1:
        raise RuntimeError(f"expected one path for {pattern}, found {len(paths)}: {paths}")
    return paths[0]


def completed_run_dir(pattern: str) -> Path:
    markers = sorted(REPO_ROOT.glob(f"{pattern}/completed.json"))
    if len(markers) != 1:
        raise RuntimeError(f"expected one completed run for {pattern}, found {len(markers)}: {markers}")
    return markers[0].parent


def completed_run_file(pattern: str, relative_path: str) -> Path:
    path = completed_run_dir(pattern) / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


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


def metric_value(row: dict[str, str], metric: str, suffix: str = "mean") -> float:
    return float(row[f"{metric}_{suffix}"])


def plot_split_distribution(output_dir: Path) -> None:
    audit = read_json(REPO_ROOT / "outputs/stage2/v2_preflight/data_audit.json")
    splits = ["train", "val", "test"]
    classes = ["intact", "damaged", "destroyed"]
    colors = ["#00AA6C", "#F6BD16", "#E33332"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bottom = np.zeros(len(splits))
    for name, color in zip(classes, colors, strict=True):
        values = np.asarray([100.0 * audit["splits"][split]["building_class_share"][name] for split in splits])
        ax.bar(splits, values, bottom=bottom, label=name, color=color, width=0.62)
        for x, value, base in zip(range(len(splits)), values, bottom, strict=True):
            if value >= 2.0:
                ax.text(x, base + value / 2, f"{value:.1f}%", ha="center", va="center", fontsize=8)
        bottom += values
    ax.set_ylabel("Share within GT building pixels (%)")
    ax.set_title("Stage-2 v2 split-level building-class shift")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    save_figure(fig, output_dir / "split_building_class_share.png")


def plot_phase1_validation(output_dir: Path) -> None:
    rows = read_csv(REPO_ROOT / "outputs/stage2/v2_analysis/val_best_grade/mean_std.csv")
    by = {row["experiment"]: row for row in rows}
    metrics = [
        ("building_only_macro_f1_3class", "BO grade macro"),
        ("building_only_damage_macro_f1", "BO damage macro"),
        ("building_only_f1_intact", "Intact"),
        ("building_only_f1_damaged", "Damaged"),
        ("building_only_f1_destroyed", "Destroyed"),
    ]
    experiments = list(EXPERIMENTS.items())
    x = np.arange(len(metrics))
    width = 0.19
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    for idx, (short, experiment) in enumerate(experiments):
        row = by[experiment]
        means = [metric_value(row, metric) for metric, _ in metrics]
        stds = [metric_value(row, metric, "std") for metric, _ in metrics]
        ax.bar(x + (idx - 1.5) * width, means, width, yerr=stds, capsize=2, label=DISPLAY[experiment], color=PALETTE[short])
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylim(0, 0.78)
    ax.set_ylabel("Validation F1")
    ax.set_title("Phase-1 validation: three-seed mean ± SD")
    ax.legend(ncol=2)
    save_figure(fig, output_dir / "phase1_validation_metrics.png")


def plot_phase2_screen(output_dir: Path) -> None:
    paths = {
        "A3": completed_run_file("outputs/stage2/v2_phase1/V2_A3*/seed_42/run_*", "val_best_grade/metrics.json"),
        "B2": completed_run_file("outputs/stage2/v2_phase2/V2_B2*/seed_42/run_*", "val_best_grade/metrics.json"),
        "B3": completed_run_file("outputs/stage2/v2_phase2/V2_B3*/seed_42/run_*", "val_best_grade/metrics.json"),
    }
    values = {name: read_json(path) for name, path in paths.items()}
    metrics = [
        ("building_only_macro_f1_3class", "BO grade macro"),
        ("building_only_damage_macro_f1", "BO damage macro"),
        ("building_only_f1_intact", "Intact"),
        ("building_only_f1_damaged", "Damaged"),
        ("building_only_f1_destroyed", "Destroyed"),
    ]
    x = np.arange(len(metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    labels = {"A3": "A3 weighted CE", "B2": "B2 weighted focal", "B3": "B3 disaster-balanced"}
    for idx, name in enumerate(("A3", "B2", "B3")):
        ax.bar(x + (idx - 1) * width, [values[name][metric] for metric, _ in metrics], width, color=PALETTE[name], label=labels[name])
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylim(0, 0.78)
    ax.set_ylabel("Seed-42 validation F1")
    ax.set_title("Phase-2 screen: both alternatives rejected")
    ax.legend(ncol=3)
    save_figure(fig, output_dir / "phase2_seed42_screen.png")


def plot_training_curves(output_dir: Path) -> None:
    phase1 = {
        short: completed_run_file(
            f"outputs/stage2/v2_phase1/{experiment}/seed_42/run_*", "metrics_history.csv"
        )
        for short, experiment in EXPERIMENTS.items()
    }
    phase2 = {
        "B2": completed_run_file("outputs/stage2/v2_phase2/V2_B2*/seed_42/run_*", "metrics_history.csv"),
        "B3": completed_run_file("outputs/stage2/v2_phase2/V2_B3*/seed_42/run_*", "metrics_history.csv"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    for name, path in phase1.items():
        rows = read_csv(path)
        axes[0].plot([int(r["epoch"]) for r in rows], [float(r["val_building_only_macro_f1_3class"]) for r in rows], label=name, color=PALETTE[name], linewidth=1.35)
    axes[0].set_title("Phase-1 seed-42 validation curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BO grade macro F1")
    axes[0].legend(ncol=2)
    for name, path in {"A3": phase1["A3"], **phase2}.items():
        rows = read_csv(path)
        axes[1].plot([int(r["epoch"]) for r in rows], [float(r["val_building_only_macro_f1_3class"]) for r in rows], label=name, color=PALETTE[name], linewidth=1.35)
    axes[1].set_title("Phase-2 seed-42 validation curves")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(ncol=3)
    save_figure(fig, output_dir / "validation_training_curves_seed42.png")


def plot_test_primary(output_dir: Path) -> None:
    rows = read_csv(REPO_ROOT / "outputs/stage2/v2_analysis/test_best_grade/mean_std.csv")
    by = {row["experiment"]: row for row in rows}
    a0 = read_json(REPO_ROOT / "outputs/stage2/v2_phase1/V2_A0_all_intact/test/metrics.json")
    metrics = [
        ("building_only_macro_f1_3class", "BO grade macro"),
        ("building_only_damage_macro_f1", "BO damage macro"),
        ("building_only_f1_intact", "Intact"),
        ("building_only_f1_damaged", "Damaged"),
        ("building_only_f1_destroyed", "Destroyed"),
    ]
    series: list[tuple[str, list[float], list[float]]] = [("A0", [float(a0[m]) for m, _ in metrics], [0.0] * len(metrics))]
    for short, experiment in EXPERIMENTS.items():
        series.append((short, [metric_value(by[experiment], m) for m, _ in metrics], [metric_value(by[experiment], m, "std") for m, _ in metrics]))
    x = np.arange(len(metrics))
    width = 0.16
    fig, ax = plt.subplots(figsize=(10.4, 4.8))
    for idx, (name, means, stds) in enumerate(series):
        ax.bar(x + (idx - 2) * width, means, width, yerr=stds, capsize=2, label=name, color=PALETTE[name])
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("Test F1")
    ax.set_title("Frozen test results: three-seed mean ± SD")
    ax.legend(ncol=5)
    save_figure(fig, output_dir / "test_primary_metrics.png")


def plot_test_familiarity(output_dir: Path) -> None:
    rows = read_csv(REPO_ROOT / "outputs/stage2/v2_analysis/test_best_grade/per_event_familiarity_mean_std.csv")
    by = {(row["experiment"], row["event_familiarity"]): row for row in rows}
    groups = ["train_seen", "val_seen_only", "globally_unseen"]
    labels = ["Train-seen\n(n=341)", "Val-seen only\n(n=197)", "Globally unseen\n(n=26)"]
    x = np.arange(len(groups))
    width = 0.19
    fig, ax = plt.subplots(figsize=(9.0, 4.7))
    for idx, (short, experiment) in enumerate(EXPERIMENTS.items()):
        means = [float(by[(experiment, group)]["building_only_macro_f1_3class_mean"]) for group in groups]
        stds = [float(by[(experiment, group)]["building_only_macro_f1_3class_std"]) for group in groups]
        ax.bar(x + (idx - 1.5) * width, means, width, yerr=stds, capsize=2, label=short, color=PALETTE[short])
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 0.58)
    ax.set_ylabel("BO grade macro F1")
    ax.set_title("Test performance by event familiarity")
    ax.legend(ncol=4)
    save_figure(fig, output_dir / "test_event_familiarity.png")


def plot_a3_disaster_classes(output_dir: Path) -> None:
    rows = read_csv(REPO_ROOT / "outputs/stage2/v2_analysis/test_best_grade/per_disaster_mean_std.csv")
    rows = [row for row in rows if row["experiment"] == EXPERIMENTS["A3"]]
    rows.sort(key=lambda row: row["disaster_type"])
    disasters = [row["disaster_type"] for row in rows]
    metrics = [
        ("building_only_f1_intact_mean", "Intact", "#00AA6C"),
        ("building_only_f1_damaged_mean", "Damaged", "#F6BD16"),
        ("building_only_f1_destroyed_mean", "Destroyed", "#E33332"),
    ]
    x = np.arange(len(disasters))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    for idx, (key, label, color) in enumerate(metrics):
        ax.bar(x + (idx - 1) * width, [float(row[key]) for row in rows], width, label=label, color=color)
    ax.set_xticks(x, disasters, rotation=25, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("A3 test F1")
    ax.set_title("A3 class specialization by disaster type")
    ax.legend(ncol=3)
    save_figure(fig, output_dir / "test_a3_per_disaster_classes.png")


def plot_bootstrap(output_dir: Path) -> None:
    rows = read_csv(REPO_ROOT / "outputs/stage2/v2_analysis/test_best_grade/paired_bootstrap.csv")
    rows = [row for row in rows if row["metric"] in {"bo_grade_macro_f1", "bo_damage_macro_f1"}]
    label_metric = {"bo_grade_macro_f1": "BO grade macro", "bo_damage_macro_f1": "BO damage macro"}
    labels = [f"{row['comparison'].replace('V2_', '')}\n{label_metric[row['metric']]}" for row in rows]
    point = np.asarray([float(row["point_difference"]) for row in rows])
    low = np.asarray([float(row["ci95_low"]) for row in rows])
    high = np.asarray([float(row["ci95_high"]) for row in rows])
    y = np.arange(len(rows))[::-1]
    colors = [PALETTE["A3"] if row["metric"] == "bo_grade_macro_f1" else "#4E79A7" for row in rows]
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.axvline(0, color="black", linewidth=1)
    for yi, p, lo, hi, color in zip(y, point, low, high, colors, strict=True):
        ax.errorbar(p, yi, xerr=[[p - lo], [hi - p]], fmt="o", color=color, capsize=3)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Paired A3 minus control F1 (95% bootstrap CI)")
    ax.set_title("Frozen test paired effects")
    save_figure(fig, output_dir / "test_paired_bootstrap_effects.png")


def plot_v1_v2(output_dir: Path) -> None:
    v1 = read_json(REPO_ROOT / "outputs/stage2/B1_S2_predicted_prior_unet_resnet34/test/metrics.json")
    a3 = next(row for row in read_csv(REPO_ROOT / "outputs/stage2/v2_analysis/test_best_grade/mean_std.csv") if row["experiment"] == EXPERIMENTS["A3"])
    pairs = [
        ("building_only_macro_f1_3class", "building_only_macro_f1_3class_mean", "BO grade macro"),
        ("building_only_damage_macro_f1", "building_only_damage_macro_f1_mean", "BO damage macro"),
        ("building_only_f1_damaged", "building_only_f1_damaged_mean", "Damaged"),
        ("building_only_f1_destroyed", "building_only_f1_destroyed_mean", "Destroyed"),
        ("miou_4class", "predicted_gate_miou_4class_mean", "4-class mIoU"),
    ]
    x = np.arange(len(pairs))
    fig, ax = plt.subplots(figsize=(9.4, 4.5))
    ax.bar(x - 0.18, [float(v1[left]) for left, _, _ in pairs], 0.36, label="Stage-2 v1 S2", color="#8C8C8C")
    ax.bar(x + 0.18, [float(a3[right]) for _, right, _ in pairs], 0.36, label="Stage-2 v2 A3", color=PALETTE["A3"])
    ax.set_xticks(x, [label for _, _, label in pairs])
    ax.set_ylim(0, 0.56)
    ax.set_ylabel("Test score")
    ax.set_title("Contextual Stage-2 v1 vs v2 comparison")
    ax.legend()
    save_figure(fig, output_dir / "stage2_v1_v2_comparison.png")


def export_quantitative(output_dir: Path) -> None:
    setup_plot_style()
    plot_split_distribution(output_dir)
    plot_phase1_validation(output_dir)
    plot_phase2_screen(output_dir)
    plot_training_curves(output_dir)
    plot_test_primary(output_dir)
    plot_test_familiarity(output_dir)
    plot_a3_disaster_classes(output_dir)
    plot_bootstrap(output_dir)
    plot_v1_v2(output_dir)


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def contrast_gray(array: np.ndarray) -> Image.Image:
    values = np.asarray(array, dtype=np.float32)
    lo, hi = np.percentile(values, [2, 98])
    if hi <= lo:
        hi = lo + 1.0
    values = np.clip((values - lo) / (hi - lo), 0, 1)
    return Image.fromarray((values * 255).astype(np.uint8), mode="L").convert("RGB")


def prior_gray(array: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8), mode="L").convert("RGB")


def colorize(mask: np.ndarray) -> Image.Image:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for value, color in COLORS.items():
        out[mask == value] = color
    return Image.fromarray(out)


def resized(image: Image.Image, size: int, mask: bool) -> Image.Image:
    method = Image.Resampling.NEAREST if mask else Image.Resampling.BILINEAR
    return image.resize((size, size), method)


def load_sample_metrics() -> dict[str, dict[str, dict[str, str]]]:
    result = {}
    for short, experiment in EXPERIMENTS.items():
        path = completed_run_file(
            f"outputs/stage2/v2_phase1/{experiment}/seed_42/run_*", "test_best_grade/sample_metrics.csv"
        )
        result[short] = {row["id"]: row for row in read_csv(path)}
    return result


def export_saved_preview_qualitative(output_dir: Path) -> None:
    """Build a same-sample comparison from the previews saved by formal seed-42 tests."""
    run_dirs = {
        short: completed_run_dir(f"outputs/stage2/v2_phase1/{experiment}/seed_42/run_*")
        for short, experiment in EXPERIMENTS.items()
    }
    preview_dirs = {short: run_dir / "test_best_grade/previews" for short, run_dir in run_dirs.items()}
    preview_names = sorted(path.name for path in preview_dirs["A3"].glob("test__*.png"))
    shared_names = [
        name for name in preview_names if all((preview_dirs[short] / name).is_file() for short in EXPERIMENTS)
    ]
    if not shared_names:
        raise RuntimeError("no shared seed-42 formal-test previews found")

    sample_metrics = load_sample_metrics()
    comparison_dir = output_dir / "qualitative_saved_previews"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    tile_size = 184
    header_height = 51
    column_height = 24
    column_names = ["post SAR", "O1 prior", "GT", "A0", "A1", "A2", "A3", "A4 shuffled"]
    comparison_paths: list[Path] = []
    rows_for_csv: list[dict[str, Any]] = []

    for preview_name in shared_names:
        sample_id = preview_name.removeprefix("test__").removesuffix(".png")
        raw_tiles: dict[str, list[Image.Image]] = {}
        for short, preview_dir in preview_dirs.items():
            with Image.open(preview_dir / preview_name) as image:
                width, height = image.size
                if width != 5 * height:
                    raise ValueError(f"unexpected preview layout: {preview_dir / preview_name} ({image.size})")
                raw_tiles[short] = [
                    image.crop((idx * height, 0, (idx + 1) * height, height)).convert("RGB")
                    for idx in range(5)
                ]
        prior_array = np.asarray(raw_tiles["A3"][1])[:, :, 0]
        a0_mask = np.where(prior_array >= round(0.6 * 255), 1, 0).astype(np.uint8)
        tiles = [
            resized(raw_tiles["A3"][0], tile_size, False),
            resized(raw_tiles["A3"][1], tile_size, False),
            resized(raw_tiles["A3"][2], tile_size, True),
            resized(colorize(a0_mask), tile_size, True),
            *[resized(raw_tiles[short][3], tile_size, True) for short in ("A1", "A2", "A3", "A4")],
        ]
        canvas = Image.new(
            "RGB", (tile_size * len(tiles), header_height + column_height + tile_size), "white"
        )
        draw = ImageDraw.Draw(canvas)
        row = sample_metrics["A3"][sample_id]
        draw.rectangle((0, 0, canvas.width, header_height), fill=(242, 242, 242))
        draw.text(
            (7, 6),
            f"{sample_id} | {row['disaster_type']} | {row['event_familiarity']}",
            fill="black",
            font=font(15),
        )
        scores = ", ".join(
            f"{short}={float(sample_metrics[short][sample_id]['building_only_macro_f1_3class']):.3f}"
            for short in EXPERIMENTS
        )
        draw.text(
            (7, 27),
            f"sample BO macro F1: {scores}; A4 SAR source={sample_metrics['A4'][sample_id]['sar_source_id']}",
            fill=(60, 60, 60),
            font=font(12),
        )
        for idx, (name, tile) in enumerate(zip(column_names, tiles, strict=True)):
            x = idx * tile_size
            draw.text((x + 5, header_height + 4), name, fill="black", font=font(12))
            canvas.paste(tile, (x, header_height + column_height))
        path = comparison_dir / f"comparison_{sample_id}.png"
        canvas.save(path)
        comparison_paths.append(path)
        result: dict[str, Any] = {
            "id": sample_id,
            "event_id": row["event_id"],
            "disaster_type": row["disaster_type"],
            "event_familiarity": row["event_familiarity"],
            "a4_sar_source_id": sample_metrics["A4"][sample_id]["sar_source_id"],
            "comparison_path": str(path),
        }
        for short in EXPERIMENTS:
            metrics = sample_metrics[short][sample_id]
            result[f"{short.lower()}_bo_grade_macro_f1"] = float(
                metrics["building_only_macro_f1_3class"]
            )
            result[f"{short.lower()}_bo_damage_macro_f1"] = float(
                metrics["building_only_damage_macro_f1"]
            )
        rows_for_csv.append(result)

    with Image.open(comparison_paths[0]) as first:
        row_width, row_height = first.size
    legend_height = 42
    spacing = 5
    sheet = Image.new(
        "RGB",
        (row_width, legend_height + len(comparison_paths) * row_height + spacing * (len(comparison_paths) - 1)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((7, 10), "Legend:", fill="black", font=font(14))
    x = 70
    for value, name in ((0, "background"), (1, "intact"), (2, "damaged"), (3, "destroyed")):
        color = tuple(int(v) for v in COLORS[value])
        draw.rectangle((x, 9, x + 22, 31), fill=color)
        draw.text((x + 29, 11), name, fill="black", font=font(13))
        x += 145
    y = legend_height
    for path in comparison_paths:
        with Image.open(path) as image:
            sheet.paste(image.convert("RGB"), (0, y))
        y += row_height + spacing
    sheet.save(output_dir / "saved_preview_a0_a4_contact_sheet.png")
    write_csv(output_dir / "saved_preview_visualization_samples.csv", rows_for_csv)


@torch.no_grad()
def export_qualitative(data_root: Path, output_dir: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for qualitative model inference")
    device = torch.device("cuda")
    run_dirs = {
        short: completed_run_dir(f"outputs/stage2/v2_phase1/{experiment}/seed_42/run_*")
        for short, experiment in EXPERIMENTS.items()
    }
    datasets = {}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    sar_sources: dict[str, dict[str, str]] = {}
    for short, run_dir in run_dirs.items():
        cfg = load_config(run_dir / "config_resolved.json")
        checkpoint = torch.load(run_dir / "checkpoints/best_bo_grade_macro_f1.pth", map_location="cpu", weights_only=False)
        seed = int(checkpoint.get("seed", 42))
        dataset = make_dataset(cfg, data_root, "test", seed, None)
        datasets[short] = dataset
        index_by_id = {str(row["id"]): idx for idx, row in enumerate(dataset.rows)}
        missing = [sample_id for sample_id in SELECTED_IDS if sample_id not in index_by_id]
        if missing:
            raise KeyError(f"{short} dataset is missing selected IDs: {missing}")
        model = build_model(cfg, no_pretrained=True).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        threshold = float(cfg.get("dataset", {}).get("gate_threshold", 0.6))
        predictions[short] = {}
        sar_sources[short] = {}
        for start in range(0, len(SELECTED_IDS), 4):
            ids = SELECTED_IDS[start : start + 4]
            items = [dataset[index_by_id[sample_id]] for sample_id in ids]
            batch = torch.stack([item["image"] for item in items]).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                logits = model(batch)
            grades = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8) + 1
            for sample_id, item, grade in zip(ids, items, grades, strict=True):
                support = item["prior"][0].numpy() >= threshold
                predictions[short][sample_id] = np.where(support, grade, 0).astype(np.uint8)
                sar_sources[short][sample_id] = str(item.get("sar_source_id", sample_id))
        del model, checkpoint
        torch.cuda.empty_cache()

    base = datasets["A3"]
    base_index = {str(row["id"]): idx for idx, row in enumerate(base.rows)}
    sample_metrics = load_sample_metrics()
    comparison_dir = output_dir / "qualitative"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    tile_size = 184
    header_height = 48
    column_height = 24
    column_names = ["post SAR", "O1 prior", "GT", "A0", "A1", "A2", "A3", "A4 shuffled"]
    rows_for_csv: list[dict[str, Any]] = []
    comparison_paths: list[Path] = []
    for sample_id in SELECTED_IDS:
        item = base[base_index[sample_id]]
        sar = item["sar"][0].numpy()
        prior = item["prior"][0].numpy()
        target = item["mask"].numpy().astype(np.uint8)
        a0 = np.where(prior >= 0.6, 1, 0).astype(np.uint8)
        tiles = [
            resized(contrast_gray(sar), tile_size, False),
            resized(prior_gray(prior), tile_size, False),
            resized(colorize(target), tile_size, True),
            resized(colorize(a0), tile_size, True),
            *[resized(colorize(predictions[name][sample_id]), tile_size, True) for name in ("A1", "A2", "A3", "A4")],
        ]
        canvas = Image.new("RGB", (tile_size * len(tiles), header_height + column_height + tile_size), "white")
        draw = ImageDraw.Draw(canvas)
        row = next(row for row in base.rows if str(row["id"]) == sample_id)
        familiarity = sample_metrics["A3"][sample_id]["event_familiarity"]
        title = f"{sample_id} | {row.get('disaster_type', '')} | {familiarity} | {row.get('country_or_region', '')}"
        draw.rectangle((0, 0, canvas.width, header_height), fill=(242, 242, 242))
        draw.text((7, 7), title, fill="black", font=font(15))
        a3_metric = float(sample_metrics["A3"][sample_id]["building_only_macro_f1_3class"])
        a4_metric = float(sample_metrics["A4"][sample_id]["building_only_macro_f1_3class"])
        draw.text((7, 27), f"sample BO macro F1: A3={a3_metric:.3f}, A4={a4_metric:.3f}; A4 SAR source={sar_sources['A4'][sample_id]}", fill=(60, 60, 60), font=font(12))
        for idx, (name, tile) in enumerate(zip(column_names, tiles, strict=True)):
            x = idx * tile_size
            draw.text((x + 5, header_height + 4), name, fill="black", font=font(12))
            canvas.paste(tile, (x, header_height + column_height))
        path = comparison_dir / f"comparison_{sample_id}.png"
        canvas.save(path)
        comparison_paths.append(path)
        class_counts = np.bincount(target.reshape(-1), minlength=4)
        result: dict[str, Any] = {
            "id": sample_id,
            "event_id": row.get("event_id", ""),
            "disaster_type": row.get("disaster_type", ""),
            "event_familiarity": familiarity,
            "gt_intact_pixels": int(class_counts[1]),
            "gt_damaged_pixels": int(class_counts[2]),
            "gt_destroyed_pixels": int(class_counts[3]),
            "a4_sar_source_id": sar_sources["A4"][sample_id],
            "comparison_path": str(path),
        }
        for short in EXPERIMENTS:
            metrics = sample_metrics[short][sample_id]
            result[f"{short.lower()}_bo_grade_macro_f1"] = float(metrics["building_only_macro_f1_3class"])
            result[f"{short.lower()}_bo_damage_macro_f1"] = float(metrics["building_only_damage_macro_f1"])
        rows_for_csv.append(result)

    legend_height = 42
    spacing = 5
    comparison_sizes = []
    for path in comparison_paths:
        with Image.open(path) as image:
            comparison_sizes.append(image.size)
    sheet = Image.new(
        "RGB",
        (
            max(width for width, _ in comparison_sizes),
            legend_height + sum(height for _, height in comparison_sizes) + spacing * (len(comparison_paths) - 1),
        ),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((7, 10), "Legend:", fill="black", font=font(14))
    x = 70
    for value, name in ((0, "background"), (1, "intact"), (2, "damaged"), (3, "destroyed")):
        color = tuple(int(v) for v in COLORS[value])
        draw.rectangle((x, 9, x + 22, 31), fill=color)
        draw.text((x + 29, 11), name, fill="black", font=font(13))
        x += 145
    y = legend_height
    for path in comparison_paths:
        with Image.open(path) as image:
            sheet.paste(image.convert("RGB"), (0, y))
            y += image.height + spacing
    sheet.save(output_dir / "selected_a0_a4_contact_sheet.png")
    write_csv(output_dir / "selected_visualization_samples.csv", rows_for_csv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Stage-2 v2 report plots and selected qualitative comparisons")
    parser.add_argument(
        "--data-root",
        default=str(REPO_ROOT.parent / "datasets/DisasterM3_optical_sar_damage_minimal_v0.2"),
    )
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "reports/stage2_v2/assets"))
    parser.add_argument("--quantitative-only", action="store_true")
    parser.add_argument("--qualitative-only", action="store_true")
    parser.add_argument("--saved-preview-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modes = (args.quantitative_only, args.qualitative_only, args.saved_preview_only)
    if sum(bool(mode) for mode in modes) > 1:
        raise ValueError("choose at most one export-only mode")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.saved_preview_only:
        export_saved_preview_qualitative(output_dir)
        print(f"wrote Stage-2 v2 saved-preview assets to {output_dir}")
        return
    if not args.qualitative_only:
        export_quantitative(output_dir)
    if not args.quantitative_only:
        export_qualitative(Path(args.data_root), output_dir)
    print(f"wrote Stage-2 v2 report assets to {output_dir}")


if __name__ == "__main__":
    main()
