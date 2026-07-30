#!/usr/bin/env python3
from __future__ import annotations

import ast
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/stage2_v2_qualitative_matplotlib")

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

from models.build_model import build_model  # noqa: E402
from stage2.common import read_jsonl, resolve_manifest  # noqa: E402
from stage2.datasets_v2 import Stage2V2Dataset  # noqa: E402


WORKSPACE_ROOT = REPO_ROOT.parent
DATA_ROOT = WORKSPACE_ROOT / "datasets/DisasterM3_optical_sar_damage_minimal_v0.2"
OUT_DIR = REPO_ROOT / "reports/stage2_v2/assets/qualitative_sar_diagnosis_20260626"
ANALYSIS_DIR = REPO_ROOT / "outputs/stage2/v2_qualitative_sar_diagnosis_20260626"
REPORT_PATH = REPO_ROOT / "reports/stage2_v2/QUALITATIVE_SAR_DIAGNOSIS_20260626.md"
ORACLE_OUT = REPO_ROOT / "outputs/stage2/v2_oracle_explore"

RUNS = {
    "O1": {
        "label": "O1 prior-only",
        "run": ORACLE_OUT / "V2_ORACLE_O1_prior_only/seed_42/run_20260625_170524",
        "color": "#5B8FF9",
    },
    "O2": {
        "label": "O2 paired SAR+oracle",
        "run": ORACLE_OUT / "V2_ORACLE_O2_sar_oracle_prior/seed_42/run_20260625_125918",
        "color": "#F6BD16",
    },
    "O3": {
        "label": "O3 shuffled SAR+oracle",
        "run": ORACLE_OUT / "V2_ORACLE_O3_shuffled_sar_oracle_prior/seed_42/run_20260625_140145",
        "color": "#E8684A",
    },
}

SELECTED_IDS = [
    "strictv1__test__noto_earthquake_124",
    "strictv1__val__rwanda_volcano_00000093",
    "strictv1__test__turkey_earthquake3_272",
    "strictv1__test__mexico_hurricane_00000042",
    "strictv1__test__mexico_hurricane_00000060",
    "strictv1__test__rwanda_volcano_00000028",
]

CASE_NOTES = {
    "strictv1__test__noto_earthquake_124": "O2 positive: destroyed improves",
    "strictv1__val__rwanda_volcano_00000093": "O2 positive: strong destroyed cue",
    "strictv1__test__turkey_earthquake3_272": "O2 mild positive",
    "strictv1__test__mexico_hurricane_00000042": "O2 failure on damaged-heavy Mexico",
    "strictv1__test__mexico_hurricane_00000060": "O2 collapse; O1 bias works",
    "strictv1__test__rwanda_volcano_00000028": "O2 empty/low-confidence failure",
}

CLASS_COLORS = {
    0: np.array([0, 0, 0], dtype=np.uint8),
    1: np.array([0, 170, 90], dtype=np.uint8),
    2: np.array([245, 190, 40], dtype=np.uint8),
    3: np.array([220, 50, 45], dtype=np.uint8),
}
CLASS_NAMES = ["intact", "damaged", "destroyed"]
CLASS_PLOT_COLORS = ["#00AA6C", "#F6BD16", "#DC322D"]


@dataclass
class PreparedRun:
    key: str
    label: str
    cfg: dict[str, Any]
    dataset: Stage2V2Dataset
    id_to_index: dict[str, int]
    model: torch.nn.Module
    metrics: dict[str, dict[str, str]]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_image(path: Path, mode: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert(mode))


def data_path(row: dict[str, Any], key: str) -> Path:
    path = Path(str(row[key]))
    return path if path.is_absolute() else DATA_ROOT / path


def make_dataset_from_cfg(cfg: dict[str, Any], split: str = "test") -> Stage2V2Dataset:
    dcfg = dict(cfg.get("dataset", {}))
    seed = int(dict(cfg.get("train", {})).get("seed", 42))
    split_offset = {"train": 0, "val": 10_000, "test": 20_000}[split]
    manifest = Path(str(dcfg[f"{split}_manifest"]))
    if not manifest.is_absolute():
        manifest = REPO_ROOT / manifest
    return Stage2V2Dataset(
        data_root=DATA_ROOT,
        manifest=manifest,
        train=False,
        prior_type=str(dcfg.get("prior_type", "predicted")),
        input_mode=str(dcfg.get("input_mode", "sar_prior")),
        crop_size=None,
        sar_shuffle_mode=str(dcfg.get("sar_shuffle_mode", "paired")),
        sar_shuffle_seed=int(dcfg.get("sar_shuffle_seed", seed)) + split_offset,
        sar_singleton_policy=str(dcfg.get("sar_singleton_policy", "error")),
    )


def prepare_run(key: str, meta: dict[str, Any], device: torch.device) -> PreparedRun:
    run_dir = Path(meta["run"])
    cfg = read_json(run_dir / "config_resolved.json")
    dataset = make_dataset_from_cfg(cfg, "test")
    id_to_index = {str(row["id"]): index for index, row in enumerate(dataset.rows)}
    model = build_model(cfg, no_pretrained=True).to(device)
    checkpoint = torch.load(run_dir / "checkpoints/best_bo_grade_macro_f1.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    metrics = {
        row["id"]: row
        for row in read_csv_rows(run_dir / "test_best_grade/sample_metrics.csv")
    }
    return PreparedRun(key=key, label=str(meta["label"]), cfg=cfg, dataset=dataset, id_to_index=id_to_index, model=model, metrics=metrics)


def colorize(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for value, color in CLASS_COLORS.items():
        out[mask == value] = color
    return out


def stretch_gray(arr: np.ndarray) -> np.ndarray:
    arr_f = arr.astype(np.float32)
    lo, hi = np.percentile(arr_f, [2, 98])
    if hi <= lo:
        lo, hi = float(arr_f.min()), float(arr_f.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip((arr_f - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def crop_box(mask: np.ndarray, size: int = 512) -> tuple[int, int, int, int]:
    h, w = mask.shape
    ys, xs = np.where(mask >= 2)
    if len(ys) == 0:
        ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        cy, cx = h // 2, w // 2
    else:
        cy = int(np.median(ys))
        cx = int(np.median(xs))
    y0 = min(max(cy - size // 2, 0), max(h - size, 0))
    x0 = min(max(cx - size // 2, 0), max(w - size, 0))
    return y0, x0, min(y0 + size, h), min(x0 + size, w)


def resize_tile(arr: np.ndarray, tile: int = 224, nearest: bool = False) -> Image.Image:
    image = Image.fromarray(arr)
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    return image.resize((tile, tile), resample=resample)


def add_label(tile: Image.Image, title: str, subtitle: str = "") -> Image.Image:
    w, h = tile.size
    canvas = Image.new("RGB", (w, h + 42), "white")
    canvas.paste(tile, (0, 42))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((5, 5), title[:34], fill=(0, 0, 0), font=font)
    if subtitle:
        draw.text((5, 22), subtitle[:36], fill=(70, 70, 70), font=font)
    return canvas


def o2_error_map(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    out = np.zeros((*target.shape, 3), dtype=np.uint8)
    building = target > 0
    correct = (pred == target) & building
    missed_damage = (target >= 2) & (pred == 1)
    false_damage = (target == 1) & (pred >= 2)
    damage_confusion = (target >= 2) & (pred >= 2) & (pred != target)
    other_wrong = building & (pred != target) & ~(missed_damage | false_damage | damage_confusion)
    out[correct] = np.array([190, 190, 190], dtype=np.uint8)
    out[missed_damage] = np.array([40, 120, 230], dtype=np.uint8)
    out[false_damage] = np.array([190, 60, 190], dtype=np.uint8)
    out[damage_confusion] = np.array([255, 255, 255], dtype=np.uint8)
    out[other_wrong] = np.array([255, 120, 20], dtype=np.uint8)
    return out


def grade_from_logits(logits: torch.Tensor) -> np.ndarray:
    return (torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)[0] + 1)


@torch.no_grad()
def predict_crop(run: PreparedRun, sample_id: str, box: tuple[int, int, int, int], device: torch.device) -> np.ndarray:
    item = run.dataset[run.id_to_index[sample_id]]
    y0, x0, y1, x1 = box
    image = item["image"][:, y0:y1, x0:x1].unsqueeze(0).to(device)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        logits = run.model(image)
    return grade_from_logits(logits)


def full_image_metric(metrics: dict[str, str], key: str) -> str:
    return f"{float(metrics[key]):.3f}"


def make_case_visuals(runs: dict[str, PreparedRun], manifest_rows: dict[str, dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "cases").mkdir(parents=True, exist_ok=True)
    case_rows: list[dict[str, Any]] = []
    case_images: list[Image.Image] = []
    tile = 224
    for sample_id in SELECTED_IDS:
        row = manifest_rows[sample_id]
        target = load_image(data_path(row, "mask_multiclass"), "L").astype(np.uint8)
        pre = load_image(data_path(row, "pre_image"), "RGB")
        sar = load_image(data_path(row, "post_sar"), "L").astype(np.uint8)
        box = crop_box(target, size=512)
        y0, x0, y1, x1 = box
        target_c = target[y0:y1, x0:x1]
        pre_c = pre[y0:y1, x0:x1]
        sar_c = sar[y0:y1, x0:x1]
        preds = {key: predict_crop(run, sample_id, box, device) for key, run in runs.items()}
        panels = [
            add_label(resize_tile(pre_c, tile), "pre optical", str(row["event_id"])),
            add_label(resize_tile(np.repeat(stretch_gray(sar_c)[..., None], 3, axis=2), tile), "paired post-SAR", CASE_NOTES[sample_id]),
            add_label(resize_tile(colorize(target_c), tile, nearest=True), "GT", "green/yellow/red"),
        ]
        for key in ("O1", "O2", "O3"):
            m = runs[key].metrics[sample_id]
            subtitle = f"full BO={full_image_metric(m, 'building_only_macro_f1_3class')}"
            panels.append(add_label(resize_tile(colorize(preds[key]), tile, nearest=True), runs[key].label, subtitle))
        panels.append(add_label(resize_tile(o2_error_map(preds["O2"], target_c), tile, nearest=True), "O2 error", "blue=missed damage"))
        width = sum(panel.size[0] for panel in panels)
        height = max(panel.size[1] for panel in panels) + 44
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        title = f"{sample_id} | {row['event_id']} | {CASE_NOTES[sample_id]}"
        draw.text((5, 5), title[:180], fill=(0, 0, 0), font=ImageFont.load_default())
        draw.text(
            (5, 22),
            "mask colors: intact=green, damaged=yellow, destroyed=red; O2 error: gray=correct, blue=damage->intact, magenta=intact->damage, white=damage grade swap",
            fill=(70, 70, 70),
            font=ImageFont.load_default(),
        )
        x = 0
        for panel in panels:
            canvas.paste(panel, (x, 44))
            x += panel.size[0]
        safe = sample_id.replace("/", "_")
        canvas.save(OUT_DIR / "cases" / f"{safe}.png")
        case_images.append(canvas)
        out_row: dict[str, Any] = {
            "id": sample_id,
            "event_id": row["event_id"],
            "source_split": row.get("source_split", ""),
            "note": CASE_NOTES[sample_id],
            "crop_box_y0x0y1x1": json.dumps([y0, x0, y1, x1]),
        }
        for key in ("O1", "O2", "O3"):
            m = runs[key].metrics[sample_id]
            out_row[f"{key}_full_bo_macro"] = float(m["building_only_macro_f1_3class"])
            out_row[f"{key}_full_damaged_f1"] = float(m["building_only_f1_damaged"])
            out_row[f"{key}_full_destroyed_f1"] = float(m["building_only_f1_destroyed"])
        case_rows.append(out_row)
    grid_width = max(image.size[0] for image in case_images)
    grid_height = sum(image.size[1] for image in case_images)
    grid = Image.new("RGB", (grid_width, grid_height), "white")
    y = 0
    for image in case_images:
        grid.paste(image, (0, y))
        y += image.size[1]
    grid.save(OUT_DIR / "qualitative_cases_grid.png")
    write_csv_rows(ANALYSIS_DIR / "selected_case_metrics.csv", case_rows)
    return case_rows


def confusion_from_row(row: dict[str, str]) -> np.ndarray:
    return np.asarray(ast.literal_eval(row["bo_confusion_3x3"]), dtype=np.int64).reshape(3, 3)


def plot_event_grade_shares(runs: dict[str, PreparedRun]) -> None:
    events = sorted({row["event_id"] for row in runs["O2"].metrics.values()})
    rows: list[dict[str, Any]] = []
    fig, axes = plt.subplots(len(events), 1, figsize=(11.5, 2.0 * len(events)), sharex=True)
    if len(events) == 1:
        axes = [axes]
    for ax, event in zip(axes, events, strict=True):
        labels = []
        shares = []
        true_added = False
        for key in ("O1", "O2", "O3"):
            conf = np.zeros((3, 3), dtype=np.int64)
            for row in runs[key].metrics.values():
                if row["event_id"] == event:
                    conf += confusion_from_row(row)
            if not true_added:
                true_counts = conf.sum(axis=1)
                true_share = true_counts / max(int(true_counts.sum()), 1)
                labels.append("GT")
                shares.append(true_share)
                true_added = True
                rows.append({"event_id": event, "source": "GT", **{CLASS_NAMES[i]: float(true_share[i]) for i in range(3)}})
            pred_counts = conf.sum(axis=0)
            pred_share = pred_counts / max(int(pred_counts.sum()), 1)
            labels.append(key)
            shares.append(pred_share)
            rows.append({"event_id": event, "source": key, **{CLASS_NAMES[i]: float(pred_share[i]) for i in range(3)}})
        x = np.arange(len(labels))
        bottom = np.zeros(len(labels))
        share_arr = np.vstack(shares)
        for class_index, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_PLOT_COLORS, strict=True)):
            ax.bar(x, share_arr[:, class_index], bottom=bottom, color=color, label=name if event == events[0] else None)
            bottom += share_arr[:, class_index]
        ax.set_ylim(0, 1.0)
        ax.set_ylabel(event)
        ax.set_xticks(x, labels)
    axes[0].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.35))
    axes[-1].set_xlabel("Per-event building-pixel grade distribution: GT vs model predictions")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "event_grade_share_gt_vs_predictions.png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    write_csv_rows(ANALYSIS_DIR / "event_grade_share_gt_vs_predictions.csv", rows)


def plot_sample_delta_scatter(runs: dict[str, PreparedRun]) -> None:
    events = sorted({row["event_id"] for row in runs["O2"].metrics.values()})
    cmap = dict(zip(events, plt.cm.tab10(np.linspace(0, 1, len(events))), strict=True))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharex=True, sharey=True)
    for ax, x_key, title in [
        (axes[0], "O1", "O2 vs O1 prior-only"),
        (axes[1], "O3", "O2 vs O3 shuffled-control"),
    ]:
        for event in events:
            xs = []
            ys = []
            for sample_id, row in runs["O2"].metrics.items():
                if row["event_id"] != event:
                    continue
                xs.append(float(runs[x_key].metrics[sample_id]["building_only_macro_f1_3class"]))
                ys.append(float(runs["O2"].metrics[sample_id]["building_only_macro_f1_3class"]))
            ax.scatter(xs, ys, s=18, alpha=0.75, color=cmap[event], label=event)
        ax.plot([0, 0.75], [0, 0.75], color="black", linewidth=1.0, linestyle="--")
        for sample_id in SELECTED_IDS:
            x = float(runs[x_key].metrics[sample_id]["building_only_macro_f1_3class"])
            y = float(runs["O2"].metrics[sample_id]["building_only_macro_f1_3class"])
            ax.scatter([x], [y], s=80, facecolors="none", edgecolors="black", linewidths=1.5)
        ax.set_title(title)
        ax.set_xlabel(f"{x_key} full-image BO macro F1")
        ax.set_xlim(-0.02, 0.75)
        ax.set_ylim(-0.02, 0.75)
    axes[0].set_ylabel("O2 full-image BO macro F1")
    axes[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sample_delta_scatter.png", bbox_inches="tight", dpi=180)
    plt.close(fig)


def reservoir_append(existing: np.ndarray, values: np.ndarray, max_count: int, rng: np.random.Generator) -> np.ndarray:
    if len(values) == 0:
        return existing
    values = values.astype(np.float32)
    if len(existing) >= max_count:
        return existing
    remaining = max_count - len(existing)
    if len(values) > remaining:
        values = rng.choice(values, size=remaining, replace=False)
    if len(existing) == 0:
        return values
    return np.concatenate([existing, values])


def plot_sar_histograms(manifest_rows: dict[str, dict[str, Any]]) -> None:
    rng = np.random.default_rng(20260626)
    events = sorted({str(row["event_id"]) for row in manifest_rows.values()})
    max_per_class_event = 70_000
    values: dict[tuple[str, int], np.ndarray] = {(event, cls): np.asarray([], dtype=np.float32) for event in events for cls in (1, 2, 3)}
    for row in manifest_rows.values():
        event = str(row["event_id"])
        sar = load_image(data_path(row, "post_sar"), "L").astype(np.float32)[::4, ::4]
        mask = load_image(data_path(row, "mask_multiclass"), "L").astype(np.uint8)[::4, ::4]
        for cls in (1, 2, 3):
            key = (event, cls)
            if len(values[key]) >= max_per_class_event:
                continue
            class_values = sar[mask == cls] / 255.0
            values[key] = reservoir_append(values[key], class_values, max_per_class_event, rng)
    rows: list[dict[str, Any]] = []
    fig, axes = plt.subplots(len(events), 1, figsize=(10.5, 2.2 * len(events)), sharex=True, sharey=True)
    if len(events) == 1:
        axes = [axes]
    bins = np.linspace(0, 1, 60)
    for ax, event in zip(axes, events, strict=True):
        for cls, name, color in zip((1, 2, 3), CLASS_NAMES, CLASS_PLOT_COLORS, strict=True):
            arr = values[(event, cls)]
            if len(arr) == 0:
                continue
            ax.hist(arr, bins=bins, density=True, histtype="step", linewidth=1.5, color=color, label=name)
            rows.append(
                {
                    "event_id": event,
                    "class": name,
                    "sampled_pixels": len(arr),
                    "sar_mean": float(np.mean(arr)),
                    "sar_std": float(np.std(arr)),
                    "sar_p10": float(np.percentile(arr, 10)),
                    "sar_p50": float(np.percentile(arr, 50)),
                    "sar_p90": float(np.percentile(arr, 90)),
                }
            )
        ax.set_ylabel(event)
        ax.grid(alpha=0.2)
    axes[0].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.35))
    axes[-1].set_xlabel("Post-SAR grayscale intensity from data package, normalized to 0..1")
    fig.suptitle("Test-set SAR intensity distributions overlap strongly across damage grades", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sar_intensity_hist_by_event.png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    write_csv_rows(ANALYSIS_DIR / "sar_intensity_summary_by_event.csv", rows)


def make_report(case_rows: list[dict[str, Any]]) -> None:
    def fmt(value: Any) -> str:
        return f"{float(value):.3f}"

    case_lines = [
        "| id | event | note | O1 BO | O2 BO | O3 BO | O2 damaged | O2 destroyed |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in case_rows:
        case_lines.append(
            "| "
            + " | ".join(
                [
                    str(row["id"]),
                    str(row["event_id"]),
                    str(row["note"]),
                    fmt(row["O1_full_bo_macro"]),
                    fmt(row["O2_full_bo_macro"]),
                    fmt(row["O3_full_bo_macro"]),
                    fmt(row["O2_full_damaged_f1"]),
                    fmt(row["O2_full_destroyed_f1"]),
                ]
            )
            + " |"
        )
    text = f"""# Stage-2 v2 qualitative SAR diagnosis

日期：2026-06-26

本次只做小规模定性诊断，没有启动新的正式训练，也没有用 test 结果调参。图像使用 clean human-reviewed strict manifest 和已训练完成的 oracle-prior O1/O2/O3。样本面板中的预测是在 512×512 crop 上重新前向得到；表格中的 BO 指标仍引用完整 test image 的已保存评估结果。O3 是 shuffled-SAR control，其预测使用评估协议中的 shuffled SAR 输入，面板中展示的 SAR 图仍是 paired SAR 作为视觉参考。

## 1. 生成的图

- 样本面板：`assets/qualitative_sar_diagnosis_20260626/qualitative_cases_grid.png`
- 单样本图：`assets/qualitative_sar_diagnosis_20260626/cases/`
- 样本级 O2 vs O1/O3 散点：`assets/qualitative_sar_diagnosis_20260626/sample_delta_scatter.png`
- 事件级 GT/预测类别占比：`assets/qualitative_sar_diagnosis_20260626/event_grade_share_gt_vs_predictions.png`
- SAR 灰度按事件/类别分布：`assets/qualitative_sar_diagnosis_20260626/sar_intensity_hist_by_event.png`

## 2. 选取样本

{chr(10).join(case_lines)}

## 3. 定性判断

当前结果不能解释为“模型完全无法利用 SAR”。Noto earthquake、Rwanda volcano 和部分 Turkey earthquake3 样本中，O2 相对 O1/O3 有局部正向表现，说明模型确实可以在某些事件/纹理模式下利用 SAR。

但更关键的是：这种利用不稳定，不能跨事件可靠泛化。Mexico hurricane 是 test Damaged 像素的主要来源，O2 在大量 Mexico 样本中把 damaged/destroyed 压成 intact 或近似空损伤；prior-only O1 反而因为事件/标签偏置能拿到更高 damaged F1。O3 shuffled control 在一些样本上接近甚至超过 O2，也说明 paired SAR 对最终预测的支配力不足。

## 4. 问题位置

1. 单时相 post-SAR 的建筑损伤可分性弱。`sar_intensity_hist_by_event.png` 显示 intact/damaged/destroyed 的 SAR 灰度分布大量重叠，单靠后时相 SAR 强度/纹理很难稳定区分损伤等级。
2. 事件分布和类别分布漂移严重。模型在 Val/部分事件学到的 destroyed cue，到 strict test 的 Mexico damaged 主导场景中不成立。
3. 当前 loss/checkpoint 容易在 intact、damaged、destroyed 之间做错误折中。O2 Val BO macro 高，但 Test binary damage 低，说明模型更像在学习事件相关纹理与类别先验，而不是稳定 damage evidence。
4. Stage-1 建筑 mask 不是主瓶颈。本轮使用 oracle building mask 后，O2 仍没有在 strict test global 上超过 O1；问题主要在建筑内损伤分级。

## 5. 下一步建议

不要把结论写成“SAR 没用”。更准确的表述是：

```text
当前单时相 post-SAR + U-Net/CE 方案能在部分事件中提取有用线索，
但这些线索无法在 clean event-disjoint test 上稳定泛化；
瓶颈主要是 SAR 损伤信号弱、事件/标签分布漂移和类别决策不稳定。
```

下一步优先级：

1. 引入 pre/post change cue，而不是只用 post-SAR；
2. 做 event-aware 或 domain-robust 的损伤分类实验；
3. 若继续单时相 SAR，先把 binary damage recall 稳住，再谈 damaged vs destroyed 细分；
4. 保留 shuffled-SAR control，避免把事件先验误判为 SAR 贡献。
"""
    REPORT_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = {key: prepare_run(key, meta, device) for key, meta in RUNS.items()}
    manifest_path = resolve_manifest(REPO_ROOT, "manifests/stage2_v2_clean_human_reviewed_20260624/oracle_prior/test.jsonl")
    manifest_rows = {str(row["id"]): row for row in read_jsonl(manifest_path)}
    missing = [sample_id for sample_id in SELECTED_IDS if sample_id not in manifest_rows]
    if missing:
        raise KeyError(f"selected ids missing from test manifest: {missing}")

    case_rows = make_case_visuals(runs, manifest_rows, device)
    plot_event_grade_shares(runs)
    plot_sample_delta_scatter(runs)
    plot_sar_histograms(manifest_rows)
    make_report(case_rows)
    print(f"device={device}")
    print(f"wrote {OUT_DIR.relative_to(REPO_ROOT)}")
    print(f"wrote {ANALYSIS_DIR.relative_to(REPO_ROOT)}")
    print(f"wrote {REPORT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
