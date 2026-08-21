## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-20T20:36:15+08:00
- Verification Status: IN_PROGRESS
- Version Label: rq3_e1_unlocked_development_launch_v1

## Experiment Launch Receipt

- **ID**: `rq3_e1_unlocked_pilot_20260820_203615`
- **Type**: training
- **Status**: running
- **Execution class**: `exposed_development_only`
- **Parent protocol**: `rq3_damage_evidence_decomposition_v1.0`
- **Amendment**: `rq3_damage_evidence_decomposition_unlocked_dev_v1.0`
- **Working directory**: `/home/yr/code/Building damage change detection/stage1_optical_building`
- **Started at**: `2026-08-20T20:36:15+08:00`
- **Task design**: `7 folds × {C2, C3} × seed 42 = 14`
- **Selected factor**: `instance`

The user explicitly waived the third-party blind-evaluator lock for this development experiment. The seven frozen RQ1 outer folds, 56 event-excluded OOF priors, selected training steps and fixed C3 derangements remain unchanged. This amendment allows development screening only; independent blind confirmation, unseen-event robustness and deployment-readiness claims remain forbidden.

### Command

```bash
bash scripts/launch_rq3_unlocked_e1_pilot.sh \
  "/home/yr/code/Building damage change detection/outputs/stage2/rq3_damage_evidence_decomposition_unlocked_dev_v1_20260820"
```

### Runtime

- tmux session: `rq3_e1_unlocked_pilot_20260820_203615`
- launcher log: `/home/yr/code/Building damage change detection/outputs/stage2/rq3_damage_evidence_decomposition_unlocked_dev_v1_20260820/launcher_logs/rq3_e1_unlocked_pilot_20260820_203615.log`
- output root: `/home/yr/code/Building damage change detection/outputs/stage2/rq3_damage_evidence_decomposition_unlocked_dev_v1_20260820`
- queue state: `RQ3_E1_PILOT_QUEUE_STATE.json`
- verified startup state: 2 running, 12 pending, 0 completed, 0 failures
- GPU assignment at startup: outer-0/C2 on GPU 0; outer-0/C3 on GPU 1

The launcher log can remain quiet while tasks are active because training output is written to `logs/<task_id>.log`. Both initial task logs showed active class-weight computation, and both training processes were alive.

### Monitoring

```bash
tmux attach -t rq3_e1_unlocked_pilot_20260820_203615
jq . "/home/yr/code/Building damage change detection/outputs/stage2/rq3_damage_evidence_decomposition_unlocked_dev_v1_20260820/RQ3_E1_PILOT_QUEUE_STATE.json"
tail -f "/home/yr/code/Building damage change detection/outputs/stage2/rq3_damage_evidence_decomposition_unlocked_dev_v1_20260820/logs/E1_o0_C2_s42.log"
```

To request a user-directed stop:

```bash
tmux send-keys -t rq3_e1_unlocked_pilot_20260820_203615 C-c
```

### Pre-launch verification

- Stage-2 runtime/CUDA gate: passed
- complete repository tests: `101 passed`
- RQ3-specific tests: `16 passed`
- RQ1 anchor dry-run: 7 folds, 56 OOF exports/runs and 28 derangement mappings verified
- task accounting: exactly 14 pending tasks
- available disk at launch: 733 GiB
- pre-launch GPUs: both idle

### Anomalies detected

- A registry validation failure caused by two absolute generated-artifact paths was detected before launch, corrected, and followed by a clean 101-test run.
- A manual validator invocation initially omitted its required registry argument; the corrected command returned `OK: 4 experiment entries validated`.
- No experiment-process anomaly was present at startup.
