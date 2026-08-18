#!/usr/bin/env python3
"""Prepare or run the fixed-step Stage-1 OOF lineage queue for RQ1.

The queue reuses completed fixed-step checkpoints and can resume a failed export
without retraining. Other partial model states are rejected for human review.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
from typing import Any

import yaml


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def config_for(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_name": f"rq1_oof_{model['prior_model_id']}",
        "dataset": {
            "name": "RQ1_14_event_development_union",
            "train_manifest": model["train_manifest"],
            "val_manifest": model["export_manifest"],
            "image_key": "image",
            "mask_key": "mask_binary",
            "input_modality": "pre_optical_rgb",
        },
        "model": {"name": "unet", "encoder": "resnet34", "encoder_weights": "imagenet", "in_channels": 3, "out_channels": 1},
        "train": {
            "image_size": 512, "batch_size": 8, "epochs": 100,
            "early_stopping_patience": 0, "optimizer": "adamw", "lr": 0.0001,
            "weight_decay": 0.0001, "scheduler": "cosine", "num_workers": 8, "seed": 42,
            "max_optimizer_steps": 20000,
        },
        "input": {"native_image_size": 1024, "normalize": "imagenet", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "target": {"num_classes": 1, "threshold": 0.5},
        "loss": {"name": "bce_dice", "bce_weight": 0.5, "dice_weight": 0.5, "pos_weight": 5.0, "max_pos_weight": 5.0},
        "augmentation": {"hflip": True, "vflip": True, "rotate90": True, "color_jitter": True, "gaussian_blur_light": True, "random_crop_size": 512, "positive_crop_ratio": 0.5},
    }


def command_for(
    repo: Path,
    data_root: Path,
    queue_root: Path,
    model: dict[str, Any],
    *,
    include_training: bool,
) -> list[str]:
    model_id = str(model["prior_model_id"])
    config = queue_root / "configs" / f"{model_id}.yaml"
    run_dir = queue_root / "runs" / model_id
    prior_dir = queue_root / "priors" / model_id
    threshold = queue_root / "threshold_0p5.json"
    parts = [
        "source /home/yr/miniconda3/etc/profile.d/conda.sh",
        "conda activate sam3",
        f"cd {subprocess.list2cmdline([str(repo)])}",
    ]
    if include_training:
        parts.append(
            subprocess.list2cmdline(["python", "src/train.py", "--config", str(config), "--data-root", str(data_root), "--output-dir", str(run_dir), "--amp", "--max-optimizer-steps", "20000"])
        )
    parts.append(
        subprocess.list2cmdline([
            "python", "scripts/export_stage1_o1_priors.py",
            "--config", str(config),
            "--checkpoint", str(run_dir / "checkpoints" / "fixed_final.pth"),
            "--threshold-json", str(threshold),
            "--practice-root", str(repo),
            "--data-root", str(data_root),
            "--input-manifests", str(model["export_manifest"]),
            "--split-names", "excluded",
            "--out-dir", str(prior_dir),
            "--model-id", model_id,
        ])
    )
    return ["bash", "-lc", " && ".join(parts)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    protocol_root = args.protocol_root.resolve()
    data_root = args.data_root.resolve()
    queue_root = args.queue_root.resolve()
    lineage = read_json(protocol_root / "prior_lineage.json")
    models = list(lineage["models"])
    queue_root.mkdir(parents=True, exist_ok=True)
    previous_completion = queue_root / "QUEUE_COMPLETED.json"
    previous_failure = queue_root / "QUEUE_PREVIOUS_FAILURE.json"
    if previous_completion.is_file():
        previous_status = read_json(previous_completion)
        if previous_status.get("status") == "failed" and not previous_failure.exists():
            write_json(previous_failure, previous_status)
    write_json(queue_root / "threshold_0p5.json", {"threshold": 0.5, "role": "export_preview_only_not_outer_fold_threshold"})
    commands: list[dict[str, Any]] = []
    pending: Queue[dict[str, Any]] = Queue()
    for model in models:
        model_id = str(model["prior_model_id"])
        config_path = queue_root / "configs" / f"{model_id}.yaml"
        write_yaml(config_path, config_for(model))
        completed = queue_root / "runs" / model_id / "completed.json"
        exported = queue_root / "priors" / model_id / "export_summary.json"
        run_dir = queue_root / "runs" / model_id
        prior_dir = queue_root / "priors" / model_id
        if completed.is_file() and exported.is_file():
            commands.append({"prior_model_id": model_id, "excluded_events": model["excluded_events"], "mode": "already_completed"})
            continue
        if completed.is_file():
            checkpoint = run_dir / "checkpoints" / "fixed_final.pth"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"completed model is missing fixed checkpoint: {model_id}")
            if prior_dir.exists() and any(prior_dir.iterdir()):
                raise RuntimeError(f"partial prior export requires human review: {model_id}")
            mode = "export_only"
            command = command_for(repo, data_root, queue_root, model, include_training=False)
        elif run_dir.exists() and any(run_dir.iterdir()):
            raise RuntimeError(f"partial model requires human review; refusing automatic retry: {model_id}")
        else:
            mode = "train_and_export"
            command = command_for(repo, data_root, queue_root, model, include_training=True)
        commands.append({"prior_model_id": model_id, "excluded_events": model["excluded_events"], "mode": mode, "command": command})
        pending.put({"model": model, "mode": mode, "command": command})
    write_json(queue_root / "commands.json", {"models": commands})
    write_json(queue_root / "QUEUE_PREPARED.json", {"model_count": len(models), "pending_count": pending.qsize(), "gpus": args.gpus})
    if not args.run:
        print(json.dumps({"status": "prepared", "model_count": len(models), "pending_count": pending.qsize()}, sort_keys=True))
        return
    running_marker = queue_root / "QUEUE_RUNNING.json"
    write_json(
        running_marker,
        {
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "pending_count": pending.qsize(),
            "resume_policy": "reuse_completed_training_and_resume_export_only",
        },
    )

    failures: list[dict[str, Any]] = []
    lock = threading.Lock()
    stop_event = threading.Event()

    def worker(gpu: str) -> None:
        while not stop_event.is_set() and not pending.empty():
            try:
                task = pending.get_nowait()
            except Exception:
                return
            model_id = str(task["model"]["prior_model_id"])
            suffix = ".resume_export.log" if task["mode"] == "export_only" else ".log"
            log_path = queue_root / "logs" / f"{model_id}{suffix}"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = gpu
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(task["command"], cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode != 0:
                with lock:
                    failures.append({"prior_model_id": model_id, "gpu": gpu, "exit_code": result.returncode, "log": str(log_path)})
                stop_event.set()
            pending.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in args.gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    status = {"status": "failed" if failures else "completed", "failures": failures}
    write_json(queue_root / "QUEUE_COMPLETED.json", status)
    running_marker.unlink(missing_ok=True)
    print(json.dumps(status, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
