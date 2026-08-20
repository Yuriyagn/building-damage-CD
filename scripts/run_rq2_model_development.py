#!/usr/bin/env python3
"""Run the development-only RQ2 pilot or selected-candidate extension queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from run_rq1_stage2_nested_queue import (  # noqa: E402
    evaluation_complete,
    run_tasks,
    shell_command,
    training_complete,
    write_json,
    write_yaml,
)
from validate_rq2_blind_evaluator_lock import load_and_validate  # noqa: E402


CANDIDATES: dict[str, dict[str, Any]] = {
    "R2_A": {
        "model": {
            "name": "dual_stream_unet_r18",
            "encoder": "resnet18",
            "encoder_weights": "imagenet",
            "in_channels": 5,
            "out_channels": 3,
        },
        "loss": {
            "name": "building_only_ce",
            "class_weight_strategy": "median_frequency",
            "max_class_weight": 8.0,
        },
        "sampling": {"sampling_strategy": "default"},
    },
    "R2_B": {
        "model": {
            "name": "hierarchical_unet_r34",
            "encoder": "resnet34",
            "encoder_weights": "imagenet",
            "in_channels": 5,
            "out_channels": 3,
        },
        "loss": {
            "name": "building_only_hierarchical_nll",
            "class_weight_strategy": "median_frequency",
            "max_class_weight": 8.0,
        },
        "sampling": {"sampling_strategy": "default"},
    },
    "R2_C": {
        "model": {
            "name": "unet",
            "encoder": "resnet34",
            "encoder_weights": "imagenet",
            "in_channels": 5,
            "out_channels": 3,
        },
        "loss": {
            "name": "building_only_ce",
            "class_weight_strategy": "median_frequency",
            "max_class_weight": 8.0,
        },
        "sampling": {
            "sampling_strategy": "capped_event",
            "event_sampling_alpha": 0.5,
            "event_sampling_min_weight": 0.5,
            "event_sampling_max_weight": 4.0,
        },
    },
}


def optional_blind_lock_summary(path: Path | None) -> dict[str, Any]:
    """Record an optional future blind contract without gating development."""

    if path is None:
        return {
            "status": "not_provided",
            "required_for_internal_development": False,
            "blind_confirmation_completed": False,
        }
    lock = load_and_validate(path.resolve())
    return {
        "status": "accepted",
        "required_for_internal_development": False,
        "blind_confirmation_completed": False,
        "evaluator": lock["evaluator"],
        "event_count": lock["event_count"],
        "lock_sha256": sha256_file(path.resolve()),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_rq1_anchors(
    manifest_root: Path,
    derangement_root: Path,
    selection_root: Path,
) -> dict[str, Any]:
    """Fail closed if a frozen RQ1 input or completeness anchor drifted."""

    rq1_root = manifest_root.parent
    frozen = {
        manifest_root / "FINALIZED.json": "b19791aa0d3fab30c8be30a74d593fc16a86bd744a4ab9223b0cdfa97c183d80",
        derangement_root / "AUDIT.json": "5a661ab322451a45f61de349fb5032abe3ffe6eb582bd88d71cdd15d013588ba",
        rq1_root / "stage1_oof" / "QUEUE_COMPLETED.json": "6df43b0412d258f5cbac2e1e897b0d32754f53d4ac0901fe0f56779c7106207b",
    }
    observed = {}
    for path, expected in frozen.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen RQ1 anchor: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"frozen RQ1 anchor drifted: {path}: observed={actual}, expected={expected}"
            )
        observed[str(path)] = actual

    required_manifest_names = (
        "final_train.jsonl",
        "inner_train.jsonl",
        "inner_val.jsonl",
        "outer_eval.jsonl",
        "building_threshold.json",
    )
    required_derangement_names = (
        "final_train.json",
        "inner_train.json",
        "inner_val.json",
        "outer_eval.json",
    )
    for outer in range(7):
        for name in required_manifest_names:
            path = manifest_root / f"outer_{outer}" / name
            if not path.is_file():
                raise FileNotFoundError(f"missing frozen RQ1 manifest: {path}")
        for name in required_derangement_names:
            path = derangement_root / f"outer_{outer}" / name
            if not path.is_file():
                raise FileNotFoundError(f"missing frozen RQ1 derangement: {path}")
        if not (selection_root / f"outer_{outer}.json").is_file():
            raise FileNotFoundError(f"missing frozen RQ1 selected step for outer {outer}")

    stage1_runs = [path for path in (rq1_root / "stage1_oof" / "runs").iterdir() if path.is_dir()]
    stage1_exports = [
        path
        for path in (rq1_root / "stage1_oof" / "priors").iterdir()
        if path.is_dir() and (path / "export_summary.json").is_file()
    ]
    if len(stage1_runs) != 56 or len(stage1_exports) != 56:
        raise ValueError(
            "RQ1 OOF completeness drifted: "
            f"runs={len(stage1_runs)}, exports={len(stage1_exports)}, expected=56"
        )
    return {
        "status": "verified",
        "frozen_sha256": observed,
        "stage1_oof_run_count": len(stage1_runs),
        "stage1_oof_export_count": len(stage1_exports),
        "outer_fold_count": 7,
        "derangement_mapping_count": 28,
        "artifact_copy_policy": "reuse_by_reference_no_bulk_copy",
    }


def read_selected_steps(selection_root: Path) -> dict[int, int]:
    selected = {}
    for outer in range(7):
        path = selection_root / f"outer_{outer}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if bool(payload.get("outer_metrics_used", True)):
            raise ValueError(f"selection used outer metrics: {path}")
        selected[outer] = int(payload["selected_step"])
    expected = {0: 14000, 1: 10000, 2: 14000, 3: 14000, 4: 14000, 5: 12000, 6: 16000}
    if selected != expected:
        raise ValueError(f"RQ1 selected-step anchor drifted: observed={selected}, expected={expected}")
    return selected


def candidate_config(
    *,
    manifest_root: Path,
    derangement_root: Path,
    outer: int,
    candidate: str,
    condition: str,
    seed: int,
    selected_step: int,
) -> dict[str, Any]:
    if candidate not in CANDIDATES:
        raise ValueError(f"unknown RQ2 candidate: {candidate}")
    if condition not in {"C2", "C3"}:
        raise ValueError(f"unknown RQ2 condition: {condition}")
    fold = manifest_root / f"outer_{outer}"
    dataset: dict[str, Any] = {
        "train_manifest": str((fold / "final_train.jsonl").resolve()),
        "val_manifest": str((fold / "outer_eval.jsonl").resolve()),
        "test_manifest": str((fold / "outer_eval.jsonl").resolve()),
        "prior_type": "predicted",
        "input_mode": "pre_prior_sar5",
        "sar_shuffle_mode": "paired",
        "crop_size": 512,
        "gate_threshold": float(
            json.loads((fold / "building_threshold.json").read_text(encoding="utf-8"))[
                "selected_threshold"
            ]
        ),
    }
    if condition == "C3":
        dataset["train_sar_permutation_file"] = str(
            (derangement_root / f"outer_{outer}" / "final_train.json").resolve()
        )
        dataset["val_sar_permutation_file"] = str(
            (derangement_root / f"outer_{outer}" / "outer_eval.json").resolve()
        )
    candidate_spec = CANDIDATES[candidate]
    train = {
        "batch_size": 8,
        "eval_batch_size": 4,
        "num_workers": 8,
        "lr": 0.0001,
        "weight_decay": 0.0001,
        "scheduler": "cosine",
        "amp": True,
        "deterministic": True,
        "seed": seed,
        "max_optimizer_steps": selected_step,
        "eval_steps": [selected_step],
        "progress_interval_steps": 250,
        "checkpoint_policy": "primary_only",
        "save_last_checkpoint": False,
        "early_stopping_patience": 0,
        "disaster_sampling_alpha": 0.0,
        **candidate_spec["sampling"],
    }
    return {
        "experiment_name": f"rq2_{candidate}_{condition}_o{outer}_s{seed}",
        "protocol_id": "rq2_model_development_v1.0",
        "dataset": dataset,
        "augmentation": {
            "hflip": True,
            "vflip": False,
            "rotate90": True,
            "crop_probabilities": {
                "damaged": 0.4,
                "destroyed": 0.4,
                "building": 0.2,
                "random": 0.0,
            },
        },
        "target": {
            "num_classes": 3,
            "class_names": ["intact", "damaged", "destroyed"],
            "loss_support": "gt_building",
        },
        "model": candidate_spec["model"],
        "loss": candidate_spec["loss"],
        "train": train,
    }


def require_selected_pilot(summary_path: Path, candidate: str) -> dict[str, Any]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if payload.get("phase") != "pilot":
        raise ValueError("full extension requires a pilot summary")
    if payload.get("selected_candidate") != candidate:
        raise ValueError(
            "requested candidate does not match frozen pilot selection: "
            f"requested={candidate}, selected={payload.get('selected_candidate')}"
        )
    if payload.get("status") != "candidate_selected":
        raise ValueError(f"pilot did not select an eligible candidate: {payload.get('status')}")
    return payload


def build_tasks(
    *,
    phase: str,
    candidates: list[str],
    seeds: tuple[int, ...],
    manifest_root: Path,
    derangement_root: Path,
    data_root: Path,
    selection: dict[int, int],
    output_root: Path,
    repo: Path,
) -> tuple[list[dict[str, Any]], int]:
    tasks = []
    completed = 0
    for candidate in candidates:
        for outer in range(7):
            selected_step = selection[outer]
            for condition in ("C2", "C3"):
                for seed in seeds:
                    task_id = f"{candidate}_o{outer}_{condition}_s{seed}"
                    config = output_root / "configs" / f"{task_id}.yaml"
                    run_dir = (
                        output_root
                        / "candidates"
                        / candidate
                        / "final"
                        / f"outer_{outer}"
                        / condition
                        / f"seed_{seed}"
                    )
                    eval_dir = (
                        output_root
                        / "candidates"
                        / candidate
                        / "evaluation"
                        / f"outer_{outer}"
                        / condition
                        / f"seed_{seed}"
                    )
                    train_ok = training_complete(run_dir, selected_step)
                    eval_ok = evaluation_complete(eval_dir, selected_step)
                    if train_ok and eval_ok:
                        completed += 1
                        continue
                    if not train_ok and run_dir.exists():
                        raise RuntimeError(f"partial or invalid run requires review: {run_dir}")
                    if eval_dir.exists() and not eval_ok:
                        raise RuntimeError(f"partial or invalid evaluation requires review: {eval_dir}")
                    write_yaml(
                        config,
                        candidate_config(
                            manifest_root=manifest_root,
                            derangement_root=derangement_root,
                            outer=outer,
                            candidate=candidate,
                            condition=condition,
                            seed=seed,
                            selected_step=selected_step,
                        ),
                    )
                    tasks.append(
                        {
                            "id": task_id,
                            "command": shell_command(
                                repo,
                                data_root,
                                config,
                                run_dir,
                                eval_dir,
                                selected_step,
                            ),
                        }
                    )
    expected = len(candidates) * 7 * 2 * len(seeds)
    if completed + len(tasks) != expected:
        raise RuntimeError(
            f"{phase} task accounting mismatch: completed={completed}, "
            f"pending={len(tasks)}, expected={expected}"
        )
    return tasks, completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "full"), required=True)
    parser.add_argument("--blind-evaluator-lock", type=Path)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--derangement-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--candidate", choices=tuple(CANDIDATES))
    parser.add_argument("--pilot-summary", type=Path)
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    blind_lock = optional_blind_lock_summary(args.blind_evaluator_lock)
    repo = Path(__file__).resolve().parents[1]
    output_root = args.output_root.resolve()
    if args.phase == "pilot":
        if args.candidate or args.pilot_summary:
            raise ValueError("pilot phase fixes all three candidates and seed 42")
        candidates = list(CANDIDATES)
        seeds = (42,)
    else:
        if args.candidate is None or args.pilot_summary is None:
            raise ValueError("full phase requires --candidate and --pilot-summary")
        require_selected_pilot(args.pilot_summary.resolve(), args.candidate)
        candidates = [args.candidate]
        seeds = (3407, 2026)

    anchor_audit = validate_rq1_anchors(
        args.manifest_root.resolve(),
        args.derangement_root.resolve(),
        args.selection_root.resolve(),
    )
    selection = read_selected_steps(args.selection_root.resolve())
    tasks, completed = build_tasks(
        phase=args.phase,
        candidates=candidates,
        seeds=seeds,
        manifest_root=args.manifest_root.resolve(),
        derangement_root=args.derangement_root.resolve(),
        data_root=args.data_root.resolve(),
        selection=selection,
        output_root=output_root,
        repo=repo,
    )
    plan = {
        "protocol_id": "rq2_model_development_v1.0",
        "phase": args.phase,
        "status": "dry_run" if args.dry_run else "running",
        "evidence_scope": "14_exposed_events_development_only",
        "unseen_event_generalization_claim_allowed": False,
        "blind_evaluator_lock": blind_lock,
        "candidates": candidates,
        "seeds": list(seeds),
        "expected_count": completed + len(tasks),
        "completed_count": completed,
        "pending_count": len(tasks),
        "task_ids": [task["id"] for task in tasks],
        "rq1_anchor_audit": anchor_audit,
    }
    write_json(output_root / f"{args.phase.upper()}_PLAN.json", plan)
    if args.dry_run:
        print(json.dumps(plan, sort_keys=True))
        return
    run_tasks(tasks, args.gpus, output_root / "logs", repo, f"rq2_{args.phase}")
    tasks_after, completed_after = build_tasks(
        phase=args.phase,
        candidates=candidates,
        seeds=seeds,
        manifest_root=args.manifest_root.resolve(),
        derangement_root=args.derangement_root.resolve(),
        data_root=args.data_root.resolve(),
        selection=selection,
        output_root=output_root,
        repo=repo,
    )
    if tasks_after:
        raise RuntimeError(f"RQ2 {args.phase} artifact verification failed")
    write_json(
        output_root / f"{args.phase.upper()}_COMPLETED.json",
        {
            **plan,
            "status": "completed",
            "completed_count": completed_after,
            "pending_count": 0,
        },
    )


if __name__ == "__main__":
    main()
