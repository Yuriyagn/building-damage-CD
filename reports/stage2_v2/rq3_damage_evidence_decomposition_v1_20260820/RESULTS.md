# RQ3 E1 exposed-development pilot result

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-21
- Verification Status: ANALYZED
- Version Label: rq3_e1_technical_failure_v1
- Evidence Scope: 14 exposed development events; no blind confirmation

## Verdict

The `7 folds × {C2,C3} × seed 42 = 14` E1 queue completed mechanically, but
the experiment is a **technical numerical failure**, not a valid negative model
result. Every final checkpoint is dominated by NaN parameters. E1 must be
frozen and must not become the next RQ3 anchor.

This result does not show that instance-level supervision is intrinsically
ineffective and does not show that SAR lacks damage evidence. It shows that the
implemented E1 training path became non-finite before its frozen evaluation
step and that the existing completion validator did not reject it.

## Completion and provenance

- Launch: `2026-08-20T20:36:15+08:00`
- Mechanical completion: `2026-08-21T00:56:45+08:00`
- Expected/completed queue tasks: `14/14`
- Queue failures: `0`
- Evaluation exports: 14 `metrics.json` and 14 `per_event_metrics.csv`
- Run Git anchor: `31a1adc37087928eecdf6d973634bfbf7f7d408e`
- Run source-tree SHA-256: `224ca5ab349be44ae64c976f1b1298d2c3c97f6746ab6318926174dfe14969f3`
- Run tracked-diff SHA-256: `6aedaa9e3f80dadcaf64fbdc0fb912bfd9068c4222619c610d6d0b03dea59d3f`

The 7.7 GB local output tree, checkpoints, logs, datasets and generated
predictions are intentionally excluded from Git. Compact results and hashes are
stored in `summary.json` and `event_effects.csv`.

## Aggregate exposed-development metrics

All values below are macro-averaged across the 14 frozen outer-held-out events.

| Condition | Damage F1 | Grade F1 | Intact F1 | Damaged F1 | Destroyed F1 | Binary damage F1 |
|---|---:|---:|---:|---:|---:|---:|
| F0-C2 | 0.193499 | 0.350972 | 0.665917 | 0.144764 | 0.242235 | 0.335521 |
| E1-C2 | 0.000000 | 0.274362 | 0.823087 | 0.000000 | 0.000000 | 0.000000 |
| E1-C3 | 0.000000 | 0.274362 | 0.823087 | 0.000000 | 0.000000 | 0.000000 |

E1 predicted zero damage pixels over `99,417,491` true damage pixels. The
non-zero grade score is the Intact-only outcome divided into the three-class
macro average. All seven C2/C3 per-event CSV pairs are byte-identical.

## Frozen pilot gates

| Gate | Value | Threshold | Nominal result |
|---|---:|---:|---|
| C2 relative to F0 event-macro damage | -0.193499 | >= +0.010 | fail |
| Damaged delta | -0.144764 | >= +0.005 | fail |
| Destroyed delta | -0.242235 | >= -0.010 | fail |
| Improved events | 0/14 | >= 8/14 | fail |
| E1 C2-C3 damage | 0.000000 | >= +0.020 | fail |
| Worst event grade vs F0 | -0.379180 | >= -0.030 | fail |
| Worst event grade vs C3 | 0.000000 | >= -0.030 | nominal pass only |
| Overall damage vs F0 | -0.249887 | >= 0 | fail |

The last overall-damage value is the mean of the seven fold-level
`metrics.json` building-only damage macro F1 differences. The only nominally
passing gate is not interpretable because both E1 conditions collapsed.

## Numerical-integrity audit

- Final train loss was NaN in `14/14` runs.
- Final validation loss was NaN in `14/14` runs.
- First reported NaN appeared by step 250 in 2 runs, step 500 in 11 runs and
  step 750 in 1 run.
- All 14 final checkpoints contain non-finite model state.
- Per checkpoint, 232 of 280 state tensors are affected.
- Per checkpoint, `24,461,603 / 24,462,084` state elements are NaN
  (`99.9980%`).
- The evaluator did not reject NaN logits. Their argmax became grade index 0,
  yielding an artificial all-Intact prediction.

The exact numerical trigger is not yet localized. AMP is enabled, and the E1
path includes masked mean/max component pooling, an instance classifier and
per-component CE. These are root-cause candidates, not established causes.

## Why preflight did not catch it

The two-batch 30-step overfit test passed with loss reduction
`1.028579 -> 0.328123` (`68.10%`). The 500-step throughput benchmark also met
the wall-time ceiling. Neither gate checked long-horizon finite loss, gradients,
logits or checkpoint parameters on the formal data stream.

The shared queue completion functions check status markers, frozen step counts,
checkpoint/file existence and CSV row counts. They do not check numerical
finiteness. This allowed the queue to report `completed` despite unusable model
state.

## Claim verdicts

| Claim | Verdict | Basis |
|---|---|---|
| The 14-task E1 queue and evaluation export completed mechanically. | ALIGNED | Completion marker, queue state and all expected compact evaluation exports exist. |
| The produced E1 checkpoints are valid trained models. | NOT_SUPPORTED_BY_PROVENANCE | All 14 checkpoints are dominated by NaN model state. |
| Instance supervision harms damage grading. | NOT_SUPPORTED_BY_PROVENANCE | Numerical divergence prevents a valid intervention comparison. |
| Paired SAR carries no correspondence signal under E1. | NOT_SUPPORTED_BY_PROVENANCE | C2 and C3 became identical only after both paths collapsed. |
| RQ3 demonstrates independent-event robustness or deployment readiness. | PROVENANCE_INSUFFICIENT | The experiment uses 14 exposed events and no blind evaluation. |

## Statistical and methodological limits

- This is a single-seed pilot and cannot estimate multi-seed variation.
- No inferential p-value is used for the fail decision; the frozen deterministic
  gates and finite-value audit are decisive.
- All 14 event-level damage deltas are non-positive, so there is no aggregate
  versus subgroup direction reversal.
- Event-macro and per-class reporting prevent the Intact-only grade score from
  substituting for damage-class performance.
- The conceptual effect of instance supervision is not identifiable from a
  numerically invalid run.
- Evidence remains exposed-development-only.

Fallacy scan coverage: **11/11**. Simpson reversal was checked; ecological and
base-rate overreach are blocked by event/class reporting; selected exposed
events remain a Berkson/selection limitation; collider bias, regression to the
mean and reverse causality are not applicable; all runs completed, so there is
no run-level survivorship filter; the prespecified E1 factor and eight gates
limit look-elsewhere and forking-path risks; causal attribution to the E1 idea
is explicitly rejected because the implementation diverged.

## Required recovery before further score-bearing runs

1. Make training and evaluation fail closed on non-finite loss, gradients,
   logits, metrics and checkpoint tensors.
2. Run one non-score-bearing fold diagnostic for at least 1,000 steps, with
   finite checks and intermediate state at steps 100/250/500/750/1000.
3. Localize the fault with a controlled AMP-on versus AMP-off comparison and
   parameter/gradient anomaly reporting. Do not select a recipe by validation
   score during this diagnostic.
4. If the repair is numerically equivalent, rerun the unchanged 14-task E1
   pilot in a new output root. If LR, loss, supervision, pooling or another
   scientific factor changes, freeze a new protocol version first.
5. Do not run the 63-task E1 extension and do not stack E1 into E2-E4 unless a
   valid pilot clears every gate.

The machine-readable result is `summary.json`. A reusable external-model review
request is provided in `MODEL_AUDIT_PROMPT.md`.
