#!/usr/bin/env python3
"""Wait for the RQ1 Stage-1 queue, then execute dependency-gated Stage-2 phases."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def run(command: list[str], cwd: Path) -> None:
    print(json.dumps({"command": command}, ensure_ascii=False), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    protocol = args.protocol_root.resolve()
    stage1 = args.stage1_root.resolve()
    data = args.data_root.resolve()
    experiment = args.experiment_root.resolve()
    completion = stage1 / "QUEUE_COMPLETED.json"
    running = stage1 / "QUEUE_RUNNING.json"
    while running.is_file() or not completion.is_file():
        if not (stage1 / "QUEUE_PREPARED.json").is_file():
            raise FileNotFoundError("Stage-1 queue preparation marker disappeared")
        print(json.dumps({"status": "waiting_for_stage1", "poll_seconds": args.poll_seconds}), flush=True)
        time.sleep(args.poll_seconds)
    stage1_status = json.loads(completion.read_text(encoding="utf-8"))
    if stage1_status.get("status") != "completed":
        raise RuntimeError(f"Stage-1 queue did not complete successfully: {stage1_status}")

    manifests = experiment / "finalized_manifests"
    derangements = experiment / "derangements"
    stage2 = experiment / "stage2_nested"
    if not (manifests / "FINALIZED.json").is_file():
        run(["python", "scripts/finalize_rq1_oof_manifests.py", "--protocol-root", str(protocol), "--stage1-queue-root", str(stage1), "--data-root", str(data), "--output-root", str(manifests)], repo)
    if not (derangements / "AUDIT.json").is_file():
        run(["python", "scripts/build_rq1_balanced_sar_derangements.py", "--manifest-root", str(manifests), "--data-root", str(data), "--output-root", str(derangements)], repo)
    run(["python", "scripts/run_rq1_stage2_nested_queue.py", "--manifest-root", str(manifests), "--derangement-root", str(derangements), "--data-root", str(data), "--output-root", str(stage2), "--gpus", "0", "1"], repo)
    run(["python", "scripts/summarize_rq1_nested_cv.py", "--stage2-root", str(stage2), "--output-root", str(experiment / "results"), "--bootstrap-iterations", "10000"], repo)


if __name__ == "__main__":
    main()
