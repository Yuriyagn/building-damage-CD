#!/usr/bin/env python3
"""Run RQ1 Stage-2 inner selection and final outer evaluation on two GPUs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import threading
import uuid
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import yaml


CONDITIONS = {
    "C0": {"input_mode": "prior_only5", "deranged": False},
    "C1": {"input_mode": "pre_prior5", "deranged": False},
    "C2": {"input_mode": "pre_prior_sar5", "deranged": False},
    "C3": {"input_mode": "pre_prior_sar5", "deranged": True},
}
SEEDS = (42, 3407, 2026)
INNER_STEPS = (4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _csv_row_count(path: Path) -> int | None:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return sum(1 for _ in csv.DictReader(handle))
    except (FileNotFoundError, OSError, csv.Error):
        return None


def training_complete(
    run_dir: Path, expected_step: int, *, require_numerical_integrity: bool = False
) -> bool:
    """Accept only complete runs produced by the corrected fixed-step trainer."""

    completion = _read_json_object(run_dir / "completed.json")
    run_info = _read_json_object(run_dir / "run_info.json")
    if completion is None or run_info is None:
        return False
    if completion.get("status") != "completed":
        return False
    if require_numerical_integrity:
        integrity = _read_json_object(run_dir / "numerical_integrity.json")
        if completion.get("numerical_integrity_passed") is not True:
            return False
        if integrity is None or integrity.get("status") != "passed":
            return False
    if int(completion.get("optimizer_steps_completed") or -1) != expected_step:
        return False
    if run_info.get("fixed_step_mode") is not True:
        return False
    if int(run_info.get("max_optimizer_steps") or -1) != expected_step:
        return False
    if run_info.get("fixed_step_iterator_policy") != "persistent_full_pass_cycle":
        return False
    if (
        run_info.get("fixed_step_training_diagnostics_policy")
        != "full_at_validation_loss_at_heartbeat"
    ):
        return False
    if run_info.get("validation_batch_limit", "missing") is not None:
        return False
    checkpoint = run_dir / "checkpoints" / f"step_{expected_step:06d}.pth"
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        return False
    if require_numerical_integrity:
        sidecar = _read_json_object(checkpoint.with_suffix(checkpoint.suffix + ".integrity.json"))
        if sidecar is None or sidecar.get("status") != "passed":
            return False
    try:
        with (run_dir / "metrics_history.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return bool(rows) and int(rows[-1]["epoch"]) == expected_step
    except (FileNotFoundError, OSError, csv.Error, KeyError, TypeError, ValueError):
        return False


def evaluation_complete(
    eval_dir: Path, expected_step: int, *, require_numerical_integrity: bool = False
) -> bool:
    """Validate the full outer-evaluation artifact set, not a marker alone."""

    summary = _read_json_object(eval_dir / "event_generalization.json")
    metrics = _read_json_object(eval_dir / "metrics.json")
    if summary is None or metrics is None:
        return False
    if require_numerical_integrity:
        integrity = _read_json_object(eval_dir / "numerical_integrity.json")
        if integrity is None or integrity.get("status") != "passed":
            return False
    try:
        event_count = int(summary["event_count"])
        sample_count = int(metrics["sample_count"])
        checkpoint_step = int(metrics["checkpoint_epoch"])
    except (KeyError, TypeError, ValueError):
        return False
    if event_count <= 0 or sample_count <= 0 or checkpoint_step != expected_step:
        return False
    if metrics.get("split") != "val":
        return False
    return (
        _csv_row_count(eval_dir / "per_event_metrics.csv") == event_count
        and _csv_row_count(eval_dir / "sample_metrics.csv") == sample_count
    )


def config_for(
    manifests: Path, derangements: Path, outer: int, condition: str, seed: int,
    phase: str, selected_step: int | None = None,
) -> dict[str, Any]:
    if phase == "inner":
        train_role, val_role = "inner_train", "inner_val"
        max_steps, eval_steps = 20000, list(INNER_STEPS)
    else:
        train_role, val_role = "final_train", "outer_eval"
        if selected_step is None:
            raise ValueError("final config requires selected_step")
        max_steps, eval_steps = selected_step, [selected_step]
    dataset: dict[str, Any] = {
        "train_manifest": str((manifests / f"outer_{outer}" / f"{train_role}.jsonl").resolve()),
        "val_manifest": str((manifests / f"outer_{outer}" / f"{val_role}.jsonl").resolve()),
        "test_manifest": str((manifests / f"outer_{outer}" / f"{val_role}.jsonl").resolve()),
        "prior_type": "predicted", "input_mode": CONDITIONS[condition]["input_mode"],
        "sar_shuffle_mode": "paired", "crop_size": 512,
        "gate_threshold": float(json.loads((manifests / f"outer_{outer}" / "building_threshold.json").read_text())["selected_threshold"]),
    }
    if CONDITIONS[condition]["deranged"]:
        dataset["train_sar_permutation_file"] = str((derangements / f"outer_{outer}" / f"{train_role}.json").resolve())
        dataset["val_sar_permutation_file"] = str((derangements / f"outer_{outer}" / f"{val_role}.json").resolve())
    return {
        "experiment_name": f"rq1_{phase}_o{outer}_{condition}_s{seed}",
        "dataset": dataset,
        "augmentation": {"hflip": True, "vflip": False, "rotate90": True, "crop_probabilities": {"damaged": 0.4, "destroyed": 0.4, "building": 0.2, "random": 0.0}},
        "target": {"num_classes": 3, "class_names": ["intact", "damaged", "destroyed"], "loss_support": "gt_building"},
        "model": {"name": "unet", "encoder": "resnet34", "encoder_weights": "imagenet", "in_channels": 5, "out_channels": 3},
        "loss": {"name": "building_only_ce", "class_weight_strategy": "median_frequency", "max_class_weight": 8.0},
        "train": {
            "batch_size": 8, "eval_batch_size": 4, "num_workers": 8, "lr": 0.0001,
            "weight_decay": 0.0001, "scheduler": "cosine", "amp": True,
            "sampling_strategy": "default", "deterministic": True, "seed": seed,
            "max_optimizer_steps": max_steps, "eval_steps": eval_steps,
            "progress_interval_steps": 250,
            "checkpoint_policy": "primary_only", "save_last_checkpoint": False,
            "early_stopping_patience": 0,
        },
    }


def _temporary_output(final_dir: Path) -> Path:
    return final_dir.with_name(f"{final_dir.name}.in_progress.{uuid.uuid4().hex[:12]}")


def _evaluation_parts(
    data_root: Path,
    config: Path,
    run_dir: Path,
    eval_dir: Path,
    final_step: int,
) -> list[str]:
    temporary_eval = _temporary_output(eval_dir)
    return [
        subprocess.list2cmdline([
            "python", "src/stage2/test_stage2_v2.py", "--config", str(config),
            "--data-root", str(data_root),
            "--checkpoint", str(run_dir / "checkpoints" / f"step_{final_step:06d}.pth"),
            "--split", "val", "--output-dir", str(temporary_eval), "--batch-size", "4",
            "--max-previews", "0",
        ]),
        subprocess.list2cmdline(["mv", "-T", "--", str(temporary_eval), str(eval_dir)]),
    ]


def shell_command(repo: Path, data_root: Path, config: Path, run_dir: Path, eval_dir: Path | None, final_step: int) -> list[str]:
    temporary_run = _temporary_output(run_dir)
    parts = [
        "source /home/yr/miniconda3/etc/profile.d/conda.sh", "conda activate sam3",
        f"cd {subprocess.list2cmdline([str(repo)])}",
        subprocess.list2cmdline([
            "python", "src/stage2/train_stage2_v2.py", "--config", str(config),
            "--data-root", str(data_root), "--output-dir", str(temporary_run),
        ]),
        subprocess.list2cmdline(["mv", "-T", "--", str(temporary_run), str(run_dir)]),
    ]
    if eval_dir is not None:
        parts.extend(_evaluation_parts(data_root, config, run_dir, eval_dir, final_step))
    return ["bash", "-lc", " && ".join(parts)]


def evaluation_command(
    repo: Path,
    data_root: Path,
    config: Path,
    run_dir: Path,
    eval_dir: Path,
    final_step: int,
) -> list[str]:
    parts = [
        "source /home/yr/miniconda3/etc/profile.d/conda.sh",
        "conda activate sam3",
        f"cd {subprocess.list2cmdline([str(repo)])}",
        *_evaluation_parts(data_root, config, run_dir, eval_dir, final_step),
    ]
    return ["bash", "-lc", " && ".join(parts)]


def run_tasks(
    tasks: list[dict[str, Any]],
    gpus: list[str],
    log_root: Path,
    repo: Path,
    phase: str,
) -> None:
    queue: Queue[dict[str, Any]] = Queue()
    for task in tasks:
        queue.put(task)
    failures: list[dict[str, Any]] = []
    stop = threading.Event()
    lock = threading.Lock()
    running: dict[str, str] = {}
    completed_ids: list[str] = []
    state_path = log_root.parent / f"{phase.upper()}_QUEUE_STATE.json"

    def write_state(status: str) -> None:
        write_json(
            state_path,
            {
                "status": status,
                "phase": phase,
                "task_count": len(tasks),
                "completed_count": len(completed_ids),
                "pending_count": queue.qsize(),
                "running": dict(sorted(running.items())),
                "failures": failures,
            },
        )

    write_state("running" if tasks else "completed")

    def worker(gpu: str) -> None:
        while not stop.is_set():
            try:
                task = queue.get_nowait()
            except Empty:
                return
            log = log_root / f"{task['id']}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ)
            env.update({"CUDA_VISIBLE_DEVICES": gpu, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
            with lock:
                running[task["id"]] = gpu
                write_state("running")
            with log.open("w", encoding="utf-8") as handle:
                result = subprocess.run(task["command"], cwd=repo, env=env, stdout=handle, stderr=subprocess.STDOUT)
            with lock:
                running.pop(task["id"], None)
                if result.returncode:
                    failures.append({
                        "id": task["id"],
                        "gpu": gpu,
                        "exit_code": result.returncode,
                        "failure_type": (
                            "failed_numerical" if result.returncode == 86 else "failed_runtime"
                        ),
                        "log": str(log),
                    })
                    stop.set()
                else:
                    completed_ids.append(task["id"])
                write_state("failed" if failures else "running")
            queue.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    if failures:
        with lock:
            write_state("failed")
        raise RuntimeError(f"queue failed without retry: {failures}")
    with lock:
        write_state("completed")


def selected_steps(run_root: Path) -> dict[int, int]:
    selected: dict[int, int] = {}
    for outer in range(7):
        by_step: dict[int, list[float]] = {step: [] for step in INNER_STEPS}
        for condition in ("C1", "C2", "C3"):
            path = run_root / "inner" / f"outer_{outer}" / condition / "seed_42" / "metrics_history.csv"
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    step = int(row["epoch"])
                    by_step[step].append(float(row["val_event_macro_bo_f1"]))
        means = {step: sum(values) / len(values) for step, values in by_step.items() if len(values) == 3}
        if len(means) != len(INNER_STEPS):
            raise ValueError(f"incomplete inner selection metrics for outer {outer}")
        selected[outer] = max(means, key=lambda step: (means[step], -step))
        write_json(run_root / "selection" / f"outer_{outer}.json", {"outer_fold": outer, "condition_mean_by_step": means, "selected_step": selected[outer], "outer_metrics_used": False})
    return selected


def build_final_tasks(
    output: Path,
    manifests: Path,
    derangements: Path,
    data_root: Path,
    repo: Path,
    selection: dict[int, int],
) -> tuple[list[dict[str, Any]], int]:
    tasks: list[dict[str, Any]] = []
    completed = 0
    for outer in range(7):
        for condition, seeds in (("C0", (42,)), ("C1", SEEDS), ("C2", SEEDS), ("C3", SEEDS)):
            for seed in seeds:
                task_id = f"final_o{outer}_{condition}_s{seed}"
                config = output / "configs" / f"{task_id}.yaml"
                run_dir = output / "final" / f"outer_{outer}" / condition / f"seed_{seed}"
                eval_dir = output / "evaluation" / f"outer_{outer}" / condition / f"seed_{seed}"
                selected_step = selection[outer]
                run_complete = training_complete(run_dir, selected_step)
                eval_complete = evaluation_complete(eval_dir, selected_step)
                if run_complete and eval_complete:
                    completed += 1
                    continue
                write_yaml(
                    config,
                    config_for(
                        manifests, derangements, outer, condition, seed, "final", selected_step
                    ),
                )
                if run_complete:
                    if eval_dir.exists():
                        raise RuntimeError(f"partial or invalid evaluation requires review: {eval_dir}")
                    tasks.append({
                        "id": f"{task_id}_eval_only",
                        "command": evaluation_command(
                            repo, data_root, config, run_dir, eval_dir, selected_step
                        ),
                    })
                    continue
                if run_dir.exists():
                    raise RuntimeError(f"partial or invalid run requires review: {run_dir}")
                if eval_complete:
                    raise RuntimeError(f"evaluation exists without valid training: {eval_dir}")
                if eval_dir.exists():
                    raise RuntimeError(f"partial or invalid evaluation requires review: {eval_dir}")
                tasks.append({
                    "id": task_id,
                    "command": shell_command(
                        repo, data_root, config, run_dir, eval_dir, selected_step
                    ),
                })
    if completed + len(tasks) != 70:
        raise RuntimeError(
            f"final task accounting mismatch: completed={completed}, pending={len(tasks)}, expected=70"
        )
    return tasks, completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--derangement-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    manifests, derangements, data_root, output = (args.manifest_root.resolve(), args.derangement_root.resolve(), args.data_root.resolve(), args.output_root.resolve())
    output.mkdir(parents=True, exist_ok=True)
    inner_tasks = []
    for outer in range(7):
        for condition in ("C1", "C2", "C3"):
            task_id = f"inner_o{outer}_{condition}_s42"
            config = output / "configs" / f"{task_id}.yaml"
            run_dir = output / "inner" / f"outer_{outer}" / condition / "seed_42"
            if training_complete(run_dir, 20000):
                continue
            if run_dir.exists():
                raise RuntimeError(f"partial or invalid run requires review: {run_dir}")
            write_yaml(config, config_for(manifests, derangements, outer, condition, 42, "inner"))
            inner_tasks.append({"id": task_id, "command": shell_command(repo, data_root, config, run_dir, None, 20000)})
    run_tasks(inner_tasks, args.gpus, output / "logs", repo, "inner")
    invalid_inner = [
        str(output / "inner" / f"outer_{outer}" / condition / "seed_42")
        for outer in range(7)
        for condition in ("C1", "C2", "C3")
        if not training_complete(
            output / "inner" / f"outer_{outer}" / condition / "seed_42", 20000
        )
    ]
    if invalid_inner:
        raise RuntimeError(f"inner artifact verification failed: {invalid_inner}")
    selection = selected_steps(output)
    final_tasks, previously_completed = build_final_tasks(
        output, manifests, derangements, data_root, repo, selection
    )
    completion_marker = output / "COMPLETED.json"
    if completion_marker.is_file() and previously_completed < 70:
        previous = json.loads(completion_marker.read_text(encoding="utf-8"))
        write_json(
            output / "COMPLETION_INVALIDATED.json",
            {
                "reason": "completion marker existed before all 70 final artifacts were present",
                "observed_final_completion_count": previously_completed,
                "previous_marker": previous,
            },
        )
        completion_marker.unlink()
    run_tasks(final_tasks, args.gpus, output / "logs", repo, "final")
    _, final_completed = build_final_tasks(
        output, manifests, derangements, data_root, repo, selection
    )
    if final_completed != 70:
        raise RuntimeError(f"final artifact verification failed: completed={final_completed}, expected=70")
    write_json(completion_marker, {"status": "completed", "inner_run_count": 21, "final_run_count": final_completed, "selected_steps": selection, "evidence_scope": "nested_cv_development_only"})
    print(json.dumps({"status": "completed", "selected_steps": selection}, sort_keys=True))


if __name__ == "__main__":
    main()
