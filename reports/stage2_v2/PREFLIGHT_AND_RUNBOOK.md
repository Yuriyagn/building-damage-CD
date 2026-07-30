# Stage-2 v2 Preflight and Runbook

> Correction, 2026-06-22: the legacy path/ID audit was incomplete and the legacy test is retracted. Full-content SHA-256 found 661 duplicate groups. Do not reuse this legacy runbook for new formal work; use `DATA_INTEGRITY_CORRECTION_20260622.md` and `manifests/stage2_v2_strict_v1/`.

Updated: 2026-06-20

## Current status

Implementation and preflight are complete. Formal training is intentionally not running and is delegated to the user.

Passed checks:

- legacy path/ID audit reported 0 hard errors, but this was later invalidated by full-content hashing;
- seven synthetic/regression tests;
- A3 paired-SAR CUDA smoke;
- A4 fixed within-event shuffled-SAR CUDA smoke;
- 1024×1024 validation batch size 4 smoke;
- fixed two-batch, 30-epoch overfit: train loss 1.562 to 0.574, BO grade macro F1 0.297 to 0.775.

The incomplete directories below came from a deliberately stopped launch diagnostic. They have no `completed.json`, must not be resumed, compared, or reported:

```text
outputs/stage2/v2_phase1/V2_A1_prior_only/seed_42
outputs/stage2/v2_phase1/V2_A2_sar_only/seed_42
```

The safe launcher creates a new `run_YYYYMMDD_HHMMSS` child and never overwrites either directory.

## Formal Phase-1 training

From the repository root:

```bash
source /home/yr/miniconda3/etc/profile.d/conda.sh
conda activate sam3
cd '/home/yr/code/Building damage change detection/stage1_optical_building'
```

First run the four seed-42 experiments, two at a time:

```bash
bash scripts/launch_stage2_v2_run.sh a1 42 0
bash scripts/launch_stage2_v2_run.sh a2 42 1
```

After both runs contain `completed.json`:

```bash
bash scripts/launch_stage2_v2_run.sh a3 42 0
bash scripts/launch_stage2_v2_run.sh a4 42 1
```

Each launcher prints the unique tmux session, log, output directory, monitor command, and stop command. It also runs the mandatory CUDA/data check and displays existing tmux sessions, GPU occupancy, config, seed, and output path before launch.

## Validation gate

For each completed seed-42 run, evaluate the primary checkpoint on validation only:

```bash
bash scripts/evaluate_stage2_v2_run.sh RUN_DIR GPU_INDEX val grade
```

Also evaluate the damage checkpoint if the best-grade and best-damage epochs differ materially:

```bash
bash scripts/evaluate_stage2_v2_run.sh RUN_DIR GPU_INDEX val damage
```

Proceed to the remaining formal seeds only if paired A3 improves over both A1 prior-only and A4 shuffled-SAR beyond plausible single-run noise on validation. Do not inspect test to make this decision.

If the validation gate passes:

```bash
bash scripts/launch_stage2_v2_run.sh a1 3407 0
bash scripts/launch_stage2_v2_run.sh a2 3407 1
# then a3/a4; repeat for seed 2026
```

The formal seed set is exactly `42, 3407, 2026`.

## Frozen test evaluation

After the recipe and seed set are frozen:

```bash
bash scripts/evaluate_stage2_v2_run.sh RUN_DIR GPU_INDEX test grade
bash scripts/evaluate_stage2_v2_run.sh RUN_DIR GPU_INDEX test damage
```

The evaluator writes overall BO metrics, predicted/oracle-gate four-class metrics, per-disaster/region/event/event-familiarity tables, per-sample additive confusion matrices, previews, and explicitly named `cc_surrogate` metrics. A4 excludes the singleton `hawaii_wildfire_642`; paired analysis intersects `(split,id)` keys automatically.

Aggregate all completed `test_best_grade` evaluations with:

```bash
/home/yr/miniconda3/envs/sam3/bin/python src/stage2/summarize_stage2_v2.py \
  --runs-root outputs/stage2/v2_phase1 \
  --evaluation-name test_best_grade \
  --output-dir outputs/stage2/v2_analysis/test_best_grade
```

This produces seed-wise results and mean ± sample standard deviation including BO intact, damaged, and destroyed F1. It also produces 10,000-iteration paired bootstrap confidence intervals for A3−A1 and A3−A4 on BO grade macro F1, BO damage macro F1, and all three per-class F1 metrics.

## Interpretation boundary

- No `completed.json` means no valid run.
- Test is not a tuning split.
- A3 must beat both prior-only and fixed shuffled-SAR before attributing gains to paired SAR.
- Connected components are `cc_surrogate`, not true building instances.
- If A3 does not pass the validation gate, report the negative identifiability result and do not proceed to architecture expansion.
