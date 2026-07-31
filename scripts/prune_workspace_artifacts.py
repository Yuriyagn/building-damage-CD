#!/usr/bin/env python3
"""Conservatively prune regenerable local experiment artifacts.

The default mode is a dry run.  Destructive changes require ``--apply``.
Datasets, source code, configs, manifests, reports, scalar metrics, and logs are
never selected by this script.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class Removal:
    path: str
    kind: str
    reason: str
    bytes: int
    files: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the reviewed removal plan. Without this flag nothing is deleted.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON report path (default: reports/workspace_hygiene/...).",
    )
    return parser.parse_args()


def path_size(path: Path) -> tuple[int, int]:
    if path.is_file() or path.is_symlink():
        return path.lstat().st_size, 1

    total_bytes = 0
    total_files = 0
    for root, _, files in os.walk(path, followlinks=False):
        root_path = Path(root)
        for name in files:
            candidate = root_path / name
            try:
                total_bytes += candidate.lstat().st_size
                total_files += 1
            except FileNotFoundError:
                continue
    return total_bytes, total_files


def is_within(path: Path, root: Path) -> bool:
    try:
        path.absolute().relative_to(root.absolute())
    except ValueError:
        return False
    return True


def git_tracked_files(git_root: Path) -> set[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=git_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return set()
    return {Path(line) for line in result.stdout.splitlines() if line}


def checkpoint_is_retained(relative_path: Path) -> bool:
    """Return True only for checkpoints that define the current reproducible core."""
    parts = relative_path.parts
    filename = relative_path.name

    # Stage-1 formal architecture baselines: one primary checkpoint per recipe.
    if parts and parts[0] in {
        "O1_unet_resnet34_freq",
        "O2_unet_resnet34_all",
        "O3_deeplabv3p_resnet50_freq",
        "O4_segformer_b0_freq",
    }:
        return filename == "best_iou.pth"

    relative_text = relative_path.as_posix()

    # Audited Stage-2 strict-v1 A1-A4, three seeds: retain primary BO checkpoint.
    if relative_text.startswith("stage2/v2_strict_v1_phase1/"):
        return filename == "best_bo_grade_macro_f1.pth"

    # Current validation-selected E2 candidate: retain its primary checkpoint.
    e2_prefix = (
        "stage2/v2_phase3_objective/"
        "V2_OBJ_E2_p2_binary_aux/seed_42/run_20260628_113741/"
    )
    if relative_text.startswith(e2_prefix):
        return filename == "best_bo_grade_macro_f1.pth"

    # Unified-v1 formal transfers: retain one primary checkpoint per run.
    unified_v1_prefixes = (
        "stage2/unified_v1/S2U1_UABCD_paired/",
        "stage2/unified_v1/S2U1_UABCD_shuffled/",
        "stage2/ssfcnet_unified_v1/S2SF1_SSFCNet_paired/",
        "stage2/ssfcnet_unified_v1/S2SF1_SSFCNet_shuffled/",
        "stage2/fsgnet_unified_v1/S2FG1_FSGNet_paired/",
        "stage2/fsgnet_unified_v1/S2FG1_FSGNet_shuffled/",
    )
    if relative_text.startswith(unified_v1_prefixes):
        return filename == "best_bo_grade_macro_f1.pth"

    return False


def collect_removals(repo_root: Path, workspace_root: Path) -> tuple[list[Removal], list[str]]:
    removals: dict[Path, Removal] = {}
    retained: list[str] = []

    def add(path: Path, kind: str, reason: str) -> None:
        if not path.exists() and not path.is_symlink():
            return
        if not is_within(path, workspace_root):
            raise RuntimeError(f"Refusing out-of-workspace path: {path}")
        # Avoid listing children when their parent is already selected.
        if any(parent in removals for parent in path.parents):
            return
        size_bytes, file_count = path_size(path)
        removals[path] = Removal(
            path=str(path.relative_to(workspace_root)),
            kind=kind,
            reason=reason,
            bytes=size_bytes,
            files=file_count,
        )

    outputs_root = repo_root / "outputs"
    if outputs_root.exists():
        add(
            outputs_root / "stage2" / "ogsr_v0_features",
            "derived_features",
            "Rejected OGSR branch; features are deterministically regenerable.",
        )

        for directory_name in ("previews", "predictions"):
            for directory in sorted(outputs_root.rglob(directory_name)):
                if directory.is_dir():
                    add(
                        directory,
                        "generated_visuals",
                        "Generated predictions/previews; curated report assets remain untouched.",
                    )

        for checkpoint in sorted(outputs_root.rglob("*.pth")):
            relative_path = checkpoint.relative_to(outputs_root)
            if checkpoint_is_retained(relative_path):
                retained.append(str(checkpoint.relative_to(workspace_root)))
            else:
                add(
                    checkpoint,
                    "checkpoint",
                    "Non-primary, legacy, debug, or rejected-branch checkpoint.",
                )

    reproduction_root = workspace_root / "cross_modal_bdm_reproduction_20260702"
    external_checkout = reproduction_root / "original_code"
    external_tracked = (
        git_tracked_files(external_checkout) if external_checkout.is_dir() else set()
    )

    def is_external_tracked(path: Path) -> bool:
        if not is_within(path, external_checkout):
            return False
        return path.relative_to(external_checkout) in external_tracked

    def contains_external_tracked_file(path: Path) -> bool:
        if not is_within(path, external_checkout):
            return False
        relative = path.relative_to(external_checkout)
        return any(relative == tracked or relative in tracked.parents for tracked in external_tracked)

    if reproduction_root.exists():
        for checkpoint in sorted(reproduction_root.rglob("Seg_epoch_*_Seg.pth")):
            add(
                checkpoint,
                "checkpoint",
                "Per-epoch upstream reproduction checkpoint; Seg_epoch_best.pth is retained.",
            )
        best_checkpoints = sorted(reproduction_root.rglob("Seg_epoch_best.pth"))
        retained.extend(str(path.relative_to(workspace_root)) for path in best_checkpoints)

    # Python/test caches are local build products. Limit traversal to code workspaces.
    ssfcnet_root = workspace_root / "ssfcnet_reproduction_20260731"
    fsgnet_root = workspace_root / "fsgnet_reproduction_20260731"
    for code_root in (repo_root, reproduction_root, ssfcnet_root, fsgnet_root):
        if not code_root.exists():
            continue
        for cache_name in ("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"):
            for cache_dir in sorted(code_root.rglob(cache_name)):
                if cache_dir.is_dir():
                    if contains_external_tracked_file(cache_dir):
                        continue
                    add(cache_dir, "cache", "Regenerable Python/tool cache.")
        for bytecode in sorted(code_root.rglob("*.pyc")):
            if is_external_tracked(bytecode):
                continue
            add(bytecode, "cache", "Regenerable Python bytecode.")

    ordered = sorted(removals.values(), key=lambda item: item.path)
    return ordered, sorted(set(retained))


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    workspace_root = repo_root.parent

    required_markers = (
        workspace_root / "PROJECT_CONTEXT.md",
        repo_root / "PROJECT_EXPERIENCE.md",
    )
    if not all(marker.is_file() for marker in required_markers):
        raise RuntimeError("Workspace markers do not match the expected project layout.")

    removals, retained = collect_removals(repo_root, workspace_root)
    mode = "applied" if args.apply else "dry_run"
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")

    if args.apply:
        for item in removals:
            remove_path(workspace_root / item.path)

    remaining_after_apply = [
        item.path
        for item in removals
        if (workspace_root / item.path).exists()
        or (workspace_root / item.path).is_symlink()
    ]
    total_bytes = sum(item.bytes for item in removals)
    total_files = sum(item.files for item in removals)

    report_path = args.report
    if report_path is None:
        suffix = "applied" if args.apply else "dry_run"
        report_path = (
            repo_root
            / "reports"
            / "workspace_hygiene"
            / f"workspace_cleanup_20260731_{suffix}.json"
        )
    elif not report_path.is_absolute():
        report_path = repo_root / report_path

    if not is_within(report_path, workspace_root):
        raise RuntimeError(f"Refusing report outside workspace: {report_path}")

    report = {
        "schema_version": 1,
        "generated_at": timestamp,
        "mode": mode,
        "workspace_root": str(workspace_root),
        "policy": {
            "never_remove": [
                "datasets",
                "source code",
                "configs",
                "manifests",
                "reports",
                "scalar metrics",
                "logs",
            ],
            "retained_checkpoint_policy": [
                "Stage-1 O1-O4 best_iou.pth",
                "strict-v1 A1-A4 three-seed best_bo_grade_macro_f1.pth",
                "current E2 best_bo_grade_macro_f1.pth",
                "unified-v1 UABCD/SSFCNet formal paired/shuffled best_bo_grade_macro_f1.pth",
                "upstream UABCD Seg_epoch_best.pth",
            ],
        },
        "summary": {
            "removal_entries": len(removals),
            "files": total_files,
            "bytes": total_bytes,
            "gib": round(total_bytes / (1024**3), 3),
            "remaining_selected_paths_after_apply": len(remaining_after_apply),
        },
        "retained_checkpoints": retained,
        "removals": [asdict(item) for item in removals],
        "remaining_selected_paths_after_apply": remaining_after_apply,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print(json.dumps(report["summary"], indent=2))
    print(f"report={report_path}")
    if not args.apply:
        print("dry-run only; pass --apply after reviewing the report")
    return 0 if not remaining_after_apply else 1


if __name__ == "__main__":
    raise SystemExit(main())
