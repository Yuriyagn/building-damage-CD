# Stage-2 v2 Next Experiment Plan

Updated: 2026-06-21

Current status, corrected 2026-06-22: the legacy test is retracted because full-content SHA-256 found exact train-test duplicates. The commands below remain execution history only. The authoritative status is `DATA_INTEGRITY_CORRECTION_20260622.md`; new formal work must use `manifests/stage2_v2_strict_v1/` and fresh training.

## 1. Seed-42 validation result

All values below use each run's `best_bo_grade_macro_f1.pth` checkpoint on the full validation split.

| Experiment | BO grade macro F1 | Intact F1 | Damaged F1 | Destroyed F1 | Damage macro F1 | Damage binary F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A1 prior-only | 0.2907 | 0.2950 | 0.4181 | 0.1590 | 0.2886 | 0.4376 |
| A2 SAR-only | 0.4219 | 0.5889 | 0.3194 | 0.3575 | 0.3384 | 0.3510 |
| A3 paired SAR + predicted prior | **0.5103** | **0.6140** | **0.5903** | 0.3264 | **0.4584** | **0.6215** |
| A4 fixed within-event shuffled SAR + predicted prior | 0.2736 | 0.3021 | 0.4308 | 0.0879 | 0.2593 | 0.4934 |

Seed 42 passes the provisional identifiability gate. A3−A1 BO grade macro F1 is +0.2196 and A3−A4 is +0.2367. This is not yet a final result because seed variation and per-event concentration have not been measured.

## 2. Complete Phase-1 replication

Do not change model, loss, crop policy, class weights, or shuffle seed. Complete all A1–A4 runs for seeds 3407 and 2026.

Use one training process per GPU. Start a new wave only after both jobs in the current wave contain `completed.json`.

### Seed 3407, wave 1

```bash
bash scripts/launch_stage2_v2_run.sh a1 3407 0
bash scripts/launch_stage2_v2_run.sh a2 3407 1
```

### Seed 3407, wave 2

```bash
bash scripts/launch_stage2_v2_run.sh a3 3407 0
bash scripts/launch_stage2_v2_run.sh a4 3407 1
```

### Seed 2026, wave 1

```bash
bash scripts/launch_stage2_v2_run.sh a1 2026 0
bash scripts/launch_stage2_v2_run.sh a2 2026 1
```

### Seed 2026, wave 2

```bash
bash scripts/launch_stage2_v2_run.sh a3 2026 0
bash scripts/launch_stage2_v2_run.sh a4 2026 1
```

The launcher now refuses an active duplicate experiment/seed and refuses a busy GPU unless `STAGE2_ALLOW_BUSY_GPU=1` is explicitly set. Do not set that override for formal runs.

## 3. Validation evaluation and gate

For every completed run, use the output path printed by its launcher:

```bash
bash scripts/evaluate_stage2_v2_run.sh RUN_DIR GPU_INDEX val grade
```

Evaluate the damage checkpoint as a secondary diagnostic:

```bash
bash scripts/evaluate_stage2_v2_run.sh RUN_DIR GPU_INDEX val damage
```

After all twelve `val_best_grade` evaluations exist:

```bash
/home/yr/miniconda3/envs/sam3/bin/python src/stage2/summarize_stage2_v2.py \
  --runs-root outputs/stage2/v2_phase1 \
  --evaluation-name val_best_grade \
  --output-dir outputs/stage2/v2_analysis/val_best_grade
```

The validation report must include:

- BO grade macro F1 mean ± standard deviation;
- BO intact, damaged, and destroyed F1 mean ± standard deviation;
- BO damage macro and binary F1;
- A3−A1 and A3−A4 paired bootstrap 95% intervals for macro and all three per-class F1 metrics;
- per-event and per-disaster results with sample counts;
- predicted-gate four-class metrics as deployment metrics.

The Phase-1 gate passes only if:

1. A3 has higher three-seed mean BO grade macro F1 than A1 and A4;
2. the paired-bootstrap lower 95% bound for A3−A1 and A3−A4 BO grade macro F1 is above zero;
3. A3 also improves damage macro F1 without a catastrophic intact F1 decline;
4. the gain is not produced by only one validation event.

## 4. Conditional Phase-2 decision

Default decision: if the four conditions above pass and all three per-class F1 values are usable, freeze the current A3 weighted-CE recipe. Do not add architectural variables merely because more experiments are possible.

Only if multi-seed validation shows persistently weak or unstable destroyed/damaged performance, run one-factor-at-a-time diagnostics:

1. A3 + class-balanced focal CE, natural disaster sampling;
2. A3 + weighted CE, disaster sampling alpha 0.5;
3. compare each against the unchanged A3 baseline on validation;
4. rerun the selected recipe with prior-only and shuffled-SAR controls before attributing its gain to SAR.

Do not change focal loss and sampler in the same first experiment.

## 5. Frozen test protocol

Do not run model test evaluation until the validation recipe is frozen. Then evaluate every formal seed's primary checkpoint:

```bash
bash scripts/evaluate_stage2_v2_run.sh RUN_DIR GPU_INDEX test grade
```

Evaluate the best-damage checkpoint only as a clearly labelled secondary result:

```bash
bash scripts/evaluate_stage2_v2_run.sh RUN_DIR GPU_INDEX test damage
```

Aggregate the primary test results:

```bash
/home/yr/miniconda3/envs/sam3/bin/python src/stage2/summarize_stage2_v2.py \
  --runs-root outputs/stage2/v2_phase1 \
  --evaluation-name test_best_grade \
  --output-dir outputs/stage2/v2_analysis/test_best_grade
```

The final claim must be based on three-seed mean ± standard deviation and paired bootstrap intervals. `cc_surrogate` remains a diagnostic and must not be called true building-instance performance.
