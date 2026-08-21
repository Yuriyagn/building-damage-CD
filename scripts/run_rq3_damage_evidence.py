#!/usr/bin/env python3
"""Generate or execute the fail-closed RQ3 single-factor experiment queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml


SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from run_rq1_stage2_nested_queue import (  # noqa: E402
    evaluation_complete, run_tasks, shell_command, training_complete, write_json, write_yaml,
)
from run_rq2_model_development import read_selected_steps, validate_rq1_anchors  # noqa: E402
from validate_rq3_blind_evaluator_lock import load_and_validate  # noqa: E402


EXPERIMENT_FACTOR = {
    "E1": "instance",
    "E2": "wavelet",
    "E3": "sensor",
    "E4": "reliability",
}
SEEDS = (42, 3407, 2026)
UNLOCKED_DEVELOPMENT_AMENDMENT_ID = "rq3_damage_evidence_decomposition_unlocked_dev_v1.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_unlocked_development_amendment(path: Path, phase: str) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"unlocked-development amendment unreadable: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("unlocked-development amendment must be a YAML object")
    expected = {
        "amendment_id": UNLOCKED_DEVELOPMENT_AMENDMENT_ID,
        "parent_protocol_id": "rq3_damage_evidence_decomposition_v1.0",
        "authority": "user_explicit_instruction",
        "decision": "waive_blind_evaluator_lock_for_exposed_development",
        "execution_class": "exposed_development_only",
        "blind_evaluator_lock_required": False,
    }
    errors = [
        f"{key}: expected {value!r}, got {payload.get(key)!r}"
        for key, value in expected.items() if payload.get(key) != value
    ]
    allowed_phases = payload.get("allowed_phases")
    if not isinstance(allowed_phases, list) or phase not in allowed_phases:
        errors.append(f"allowed_phases: {phase!r} is not authorized")
    claim_policy = payload.get("claim_policy")
    if not isinstance(claim_policy, dict):
        errors.append("claim_policy: expected an object")
    else:
        for key in ("independent_blind_confirmation", "unseen_event_robustness", "deployment_readiness"):
            if claim_policy.get(key) != "forbidden":
                errors.append(f"claim_policy.{key}: expected 'forbidden'")
    if errors:
        raise RuntimeError("unlocked-development amendment rejected:\n- " + "\n- ".join(errors))
    return payload


def resolve_evaluation_contract(
    *, phase: str, blind_evaluator_lock: Path | None,
    unlocked_development_amendment: Path | None,
) -> dict[str, Any]:
    if phase == "preflight":
        return {"status": "not_required_for_technical_preflight", "execution_class": "technical_preflight"}
    if blind_evaluator_lock is not None and unlocked_development_amendment is not None:
        raise RuntimeError("choose either a blind evaluator lock or an unlocked-development amendment, not both")
    if blind_evaluator_lock is not None:
        payload = load_and_validate(blind_evaluator_lock.resolve())
        return {
            "status": "accepted",
            "execution_class": "blind_locked",
            "event_count": payload["event_count"],
            "lock_sha256": sha256_file(blind_evaluator_lock.resolve()),
        }
    if unlocked_development_amendment is not None:
        path = unlocked_development_amendment.resolve()
        payload = load_unlocked_development_amendment(path, phase)
        return {
            "status": "waived_by_explicit_user_amendment",
            "execution_class": "exposed_development_only",
            "amendment_id": payload["amendment_id"],
            "amendment_sha256": sha256_file(path),
            "claim_policy": payload["claim_policy"],
        }
    raise RuntimeError(
        "RQ3 pilot/full requires either --blind-evaluator-lock or "
        "--unlocked-development-amendment"
    )


def require_anchor(summary: Path | None, experiment: str, phase: str) -> list[str]:
    if experiment == "E1" and phase != "full":
        if summary is not None:
            raise ValueError("E1 is anchored directly to frozen F0; do not provide an anchor summary")
        return []
    if summary is None:
        raise ValueError(f"{experiment} {phase} requires --anchor-summary")
    payload = json.loads(summary.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != "rq3_damage_evidence_decomposition_v1.0":
        raise ValueError("anchor summary has the wrong protocol")
    expected_phase = "pilot" if phase == "full" else "full"
    if payload.get("passed") is not True or payload.get("phase") != expected_phase:
        raise ValueError("anchor summary is not a passed RQ3 pilot/full verdict")
    factors = [str(value) for value in payload.get("selected_factors", [])]
    if not factors:
        raise ValueError("anchor summary does not freeze selected_factors")
    factor = EXPERIMENT_FACTOR[experiment]
    if phase == "full":
        if payload.get("experiment") != experiment or factors[-1] != factor:
            raise ValueError("full phase must use this experiment's passed pilot factor stack")
    elif factor in factors:
        raise ValueError(f"single-factor violation: {factor} is already present in anchor")
    return factors


def candidate_config(
    *, manifest_root: Path, derangement_root: Path, calibration_root: Path | None,
    outer: int, experiment: str, factors: list[str], condition: str,
    seed: int, selected_step: int, evaluation_contract: dict[str, Any],
) -> dict[str, Any]:
    fold = manifest_root / f"outer_{outer}"
    threshold = float(json.loads((fold / "building_threshold.json").read_text(encoding="utf-8"))["selected_threshold"])
    dataset: dict[str, Any] = {
        "train_manifest": str((fold / "final_train.jsonl").resolve()),
        "val_manifest": str((fold / "outer_eval.jsonl").resolve()),
        "test_manifest": str((fold / "outer_eval.jsonl").resolve()),
        "prior_type": "predicted",
        "input_mode": "pre_prior5" if condition == "C1" else "pre_prior_sar5",
        "sar_shuffle_mode": "paired",
        "crop_size": 512,
        "gate_threshold": threshold,
        "rq3_min_component_area": 4,
        "rq3_label_purity": 0.8,
    }
    if condition == "C3":
        dataset["train_sar_permutation_file"] = str((derangement_root / f"outer_{outer}" / "final_train.json").resolve())
        dataset["val_sar_permutation_file"] = str((derangement_root / f"outer_{outer}" / "outer_eval.json").resolve())
    if "reliability" in factors:
        dataset["train_rq3_reliability_permutation_file"] = str((derangement_root / f"outer_{outer}" / "final_train.json").resolve())
    if "sensor" in factors:
        if calibration_root is None:
            raise ValueError("sensor factor requires --calibration-root")
        calibration = calibration_root / f"outer_{outer}.json"
        if not calibration.is_file():
            raise FileNotFoundError(f"missing train-fold calibration: {calibration}")
        dataset["sar_calibration_file"] = str(calibration.resolve())
    return {
        "experiment_name": f"rq3_{experiment}_{condition}_o{outer}_s{seed}",
        "protocol_id": "rq3_damage_evidence_decomposition_v1.0",
        "evaluation_contract": evaluation_contract,
        "dataset": dataset,
        "augmentation": {
            "hflip": True, "vflip": False, "rotate90": True,
            "crop_probabilities": {"damaged": 0.4, "destroyed": 0.4, "building": 0.2, "random": 0.0},
        },
        "target": {"num_classes": 3, "class_names": ["intact", "damaged", "destroyed"], "loss_support": "predicted_component_valid_or_gt_building"},
        "model": {
            "name": "rq3_damage_evidence", "encoder": "resnet34",
            "encoder_weights": "imagenet", "in_channels": 5, "out_channels": 3,
            "factors": factors,
        },
        "rq3": {"reliability_loss_weight": 0.1},
        "loss": {"name": "building_only_ce", "class_weight_strategy": "median_frequency", "max_class_weight": 8.0},
        "train": {
            "batch_size": 8, "eval_batch_size": 4, "num_workers": 8,
            "lr": 0.0001, "weight_decay": 0.0001, "scheduler": "cosine",
            "amp": True, "deterministic": True, "seed": seed,
            "max_optimizer_steps": selected_step, "eval_steps": [selected_step],
            "progress_interval_steps": 250, "checkpoint_policy": "primary_only",
            "save_last_checkpoint": False, "early_stopping_patience": 0,
            "disaster_sampling_alpha": 0.0, "sampling_strategy": "default",
        },
    }


def build_tasks(*, phase: str, experiment: str, factors: list[str], manifest_root: Path,
                derangement_root: Path, calibration_root: Path | None, data_root: Path,
                selection: dict[int, int], output_root: Path, repo: Path,
                evaluation_contract: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    conditions = ("C2", "C3") if phase == "pilot" else ("C1", "C2", "C3")
    seeds = (42,) if phase == "pilot" else SEEDS
    tasks: list[dict[str, Any]] = []
    completed = 0
    for outer in range(7):
        step = selection[outer]
        for condition in conditions:
            for seed in seeds:
                task_id = f"{experiment}_o{outer}_{condition}_s{seed}"
                config = output_root / "configs" / f"{task_id}.yaml"
                run_dir = output_root / "runs" / experiment / f"outer_{outer}" / condition / f"seed_{seed}"
                eval_dir = output_root / "evaluation" / experiment / f"outer_{outer}" / condition / f"seed_{seed}"
                train_ok = training_complete(run_dir, step)
                eval_ok = evaluation_complete(eval_dir, step)
                if train_ok and eval_ok:
                    completed += 1
                    continue
                if run_dir.exists() or eval_dir.exists():
                    raise RuntimeError(f"partial RQ3 artifact requires review: {run_dir} or {eval_dir}")
                write_yaml(config, candidate_config(
                    manifest_root=manifest_root, derangement_root=derangement_root,
                    calibration_root=calibration_root, outer=outer, experiment=experiment,
                    factors=factors, condition=condition, seed=seed, selected_step=step,
                    evaluation_contract=evaluation_contract,
                ))
                tasks.append({"id": task_id, "command": shell_command(repo, data_root, config, run_dir, eval_dir, step)})
    return tasks, completed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "pilot", "full"), required=True)
    parser.add_argument("--experiment", choices=tuple(EXPERIMENT_FACTOR), required=True)
    parser.add_argument("--blind-evaluator-lock", type=Path)
    parser.add_argument("--unlocked-development-amendment", type=Path)
    parser.add_argument("--anchor-summary", type=Path)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--derangement-root", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path)
    parser.add_argument("--evidence-audit", type=Path)
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    anchor_factors = require_anchor(
        args.anchor_summary.resolve() if args.anchor_summary else None,
        args.experiment,
        args.phase,
    )
    factor = EXPERIMENT_FACTOR[args.experiment]
    factors = anchor_factors if args.phase == "full" else [*anchor_factors, factor]
    evaluation_contract = resolve_evaluation_contract(
        phase=args.phase,
        blind_evaluator_lock=args.blind_evaluator_lock,
        unlocked_development_amendment=args.unlocked_development_amendment,
    )
    if "sensor" in factors:
        if args.evidence_audit is None:
            raise RuntimeError("sensor factor requires --evidence-audit")
        audit = json.loads(args.evidence_audit.read_text(encoding="utf-8"))
        if audit.get("sensor_branch_admissible") is not True:
            raise RuntimeError("sensor factor is inadmissible: metadata coverage gate did not pass")

    manifest_root = args.manifest_root.resolve()
    derangement_root = args.derangement_root.resolve()
    selection_root = args.selection_root.resolve()
    anchors = validate_rq1_anchors(manifest_root, derangement_root, selection_root)
    selection = read_selected_steps(selection_root)
    output_root = args.output_root.resolve()
    if args.phase == "preflight":
        config = candidate_config(
            manifest_root=manifest_root, derangement_root=derangement_root,
            calibration_root=args.calibration_root.resolve() if args.calibration_root else None,
            outer=0, experiment=args.experiment, factors=factors, condition="C2", seed=42,
            selected_step=selection[0], evaluation_contract=evaluation_contract,
        )
        write_yaml(output_root / "configs" / f"{args.experiment}_preflight.yaml", config)
        write_json(output_root / "PREFLIGHT_PLAN.json", {
            "protocol_id": "rq3_damage_evidence_decomposition_v1.0", "phase": "preflight",
            "experiment": args.experiment, "factors": factors, "formal_training_started": False,
            "evaluation_contract": evaluation_contract, "rq1_anchor_audit": anchors,
        })
        print(json.dumps({"status": "preflight_config_generated", "factors": factors}, sort_keys=True))
        return 0

    repo = Path(__file__).resolve().parents[1]
    tasks, completed = build_tasks(
        phase=args.phase, experiment=args.experiment, factors=factors,
        manifest_root=manifest_root, derangement_root=derangement_root,
        calibration_root=args.calibration_root.resolve() if args.calibration_root else None,
        data_root=args.data_root.resolve(), selection=selection, output_root=output_root, repo=repo,
        evaluation_contract=evaluation_contract,
    )
    expected = 14 if args.phase == "pilot" else 63
    if completed + len(tasks) != expected:
        raise RuntimeError(f"task accounting mismatch: {completed}+{len(tasks)} != {expected}")
    plan = {
        "protocol_id": "rq3_damage_evidence_decomposition_v1.0", "phase": args.phase,
        "experiment": args.experiment, "selected_factors": factors,
        "expected_count": expected, "completed_count": completed,
        "pending_count": len(tasks), "evaluation_contract": evaluation_contract,
        "rq1_anchor_audit": anchors,
    }
    write_json(output_root / f"{args.experiment}_{args.phase.upper()}_PLAN.json", plan)
    if args.dry_run:
        print(json.dumps(plan, sort_keys=True))
        return 0
    run_tasks(tasks, args.gpus, output_root / "logs", repo, f"rq3_{args.experiment}_{args.phase}")
    remaining, completed_after = build_tasks(
        phase=args.phase, experiment=args.experiment, factors=factors,
        manifest_root=manifest_root, derangement_root=derangement_root,
        calibration_root=args.calibration_root.resolve() if args.calibration_root else None,
        data_root=args.data_root.resolve(), selection=selection, output_root=output_root, repo=repo,
        evaluation_contract=evaluation_contract,
    )
    if remaining or completed_after != expected:
        raise RuntimeError("RQ3 artifact verification failed")
    write_json(output_root / f"{args.experiment}_{args.phase.upper()}_COMPLETED.json", {**plan, "status": "completed", "completed_count": completed_after, "pending_count": 0})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
