# Stage-2 v2 Phase-1 Validation Results

> Integrity note, 2026-06-22: legacy test results were retracted, but this train-versus-validation result is retained because the corrected full-content audit found no train-val exact duplicates and no train-val event overlap. It remains legacy validation evidence, not a strict-v1 test result.

Date: 2026-06-21

Status: Phase-1 identifiability gate passed; final recipe is not frozen because validation exposed strong event/class concentration. The test split remains unused for model selection.

## Evaluation protocol

- Formal seeds: 42, 3407, 2026.
- Checkpoint: `best_bo_grade_macro_f1.pth` from every completed run.
- Evaluation: complete 506-image validation split, which is event-held-out from train.
- Main metric: building-only (BO) intact/damaged/destroyed macro F1.
- Uncertainty: sample-paired bootstrap, 10,000 iterations, averaged across matched seeds.
- `cc_surrogate` uses connected components of the binary GT building mask. It is not a true footprint-instance metric.

## Three-seed validation result

| Experiment | BO grade macro F1 | Intact F1 | Damaged F1 | Destroyed F1 | Damage macro F1 | Binary damage F1 | Predicted-gate macro F1 | CC-surrogate macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A1 prior-only | 0.2927 ± 0.0044 | 0.2149 ± 0.0698 | 0.5095 ± 0.0797 | 0.1537 ± 0.0145 | 0.3316 ± 0.0373 | 0.5189 ± 0.0711 | 0.4171 ± 0.0039 | 0.2388 |
| A2 SAR-only | 0.4455 ± 0.0212 | 0.6477 ± 0.0560 | 0.3538 ± 0.0299 | 0.3349 ± 0.0429 | 0.3444 ± 0.0208 | 0.3786 ± 0.0245 | 0.5039 ± 0.0109 | 0.5253 |
| A3 paired SAR + predicted prior | **0.4751 ± 0.0341** | 0.5906 ± 0.0462 | **0.5292 ± 0.0530** | 0.3056 ± 0.0189 | **0.4174 ± 0.0355** | **0.5561 ± 0.0572** | **0.5183 ± 0.0217** | **0.5347** |
| A4 fixed within-event shuffled SAR + predicted prior | 0.2731 ± 0.0086 | 0.3414 ± 0.0740 | 0.3943 ± 0.0344 | 0.0836 ± 0.0265 | 0.2389 ± 0.0266 | 0.4859 ± 0.0336 | 0.4023 ± 0.0076 | 0.2596 |

CC-surrogate standard deviations and size-stratified metrics are retained in the machine-readable output; the short table shows only its three-seed mean.

## Paired bootstrap result

| Comparison | Metric | Mean delta | Paired 95% CI |
| --- | --- | ---: | ---: |
| A3 − A1 | BO grade macro F1 | +0.1824 | [0.1550, 0.2062] |
| A3 − A1 | BO damage macro F1 | +0.0858 | [0.0489, 0.1180] |
| A3 − A4 | BO grade macro F1 | +0.2020 | [0.1672, 0.2313] |
| A3 − A4 | BO damage macro F1 | +0.1785 | [0.1291, 0.2192] |
| A3 − A2 | BO grade macro F1 | +0.0296 | [0.0103, 0.0515] |
| A3 − A2 | BO damage macro F1 | +0.0730 | [0.0504, 0.1000] |

For A3 − A4, all three per-class intervals are positive: intact +0.2491 [0.2270, 0.2703], damaged +0.1348 [0.1093, 0.1604], and destroyed +0.2221 [0.1309, 0.2982]. This is the strongest evidence that the model uses sample-corresponding SAR rather than only the prior or an event-level SAR signature.

A3 is not uniformly better than A2 per class: it trades intact F1 (-0.0571 [−0.0763, −0.0382]) for damaged F1 (+0.1753 [0.1479, 0.2022]); the destroyed difference is inconclusive (-0.0293 [−0.0659, 0.0173]). Therefore the prior's benefit should be described as a change in class balance and higher damage macro F1, not a universal per-class gain.

## Event-held-out validation result

| Event | Images | A3 grade macro F1 | A3−A1 | A3−A2 | A3−A4 | A3 intact / damaged / destroyed F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bata_explosion | 9 | 0.3120 | +0.0564 | +0.1236 | +0.0202 | 0.5296 / 0.3711 / 0.0353 |
| mexico_hurricane | 86 | 0.3240 | +0.0345 | +0.0629 | +0.1474 | 0.1822 / 0.7751 / 0.0147 |
| rwanda_volcano | 71 | 0.5798 | +0.3232 | −0.0079 | +0.3137 | 0.8933 / 0.0632 / 0.7827 |
| ukraine_conflict | 340 | 0.3372 | +0.1266 | +0.0226 | +0.0868 | 0.5750 / 0.4288 / 0.0078 |

The A3 grade-macro gain over A1 and A4 is positive in all four validation events, so the identifiability result is not produced by only one event. However, class quality is strongly event-concentrated: destroyed performance is almost entirely supplied by Rwanda, while Rwanda damaged F1 is only 0.0632. The global per-class values therefore hide event-dependent majority-class behavior.

## Decision

Phase-1 passes the SAR identifiability gate:

1. A3 beats prior-only and fixed within-event shuffled SAR in three-seed mean.
2. Both primary paired-bootstrap lower bounds are above zero.
3. A3 improves damage macro F1 and does not collapse intact F1 globally.
4. Grade-macro gains occur in all four held-out validation events.

The current A3 weighted-CE recipe is retained as the Phase-2 baseline (`V2-B1`), but it is not yet frozen for test. The event/class concentration triggers the planned one-factor Phase-2 diagnostics. Test evaluation must wait until this choice is resolved.

## Phase-2 screening plan

Prepared candidates:

- `V2-B2`: A3 with class-weighted focal CE, gamma 2.0; all other settings unchanged.
- `V2-B3`: A3 with weighted CE and per-image disaster sampling weight `n_d^-0.5`; all other settings unchanged.

Both candidate configs passed a real one-train-batch/one-validation-batch CUDA smoke test on 2026-06-21. This verifies execution only; smoke metrics are not experimental results.

Run seed 42 only for screening:

```bash
cd '/home/yr/code/Building damage change detection/stage1_optical_building'
bash scripts/launch_stage2_v2_run.sh b2 42 0
bash scripts/launch_stage2_v2_run.sh b3 42 1
```

After both runs contain `completed.json`, evaluate their grade checkpoints on validation using the run directories printed by the launcher:

```bash
bash scripts/evaluate_stage2_v2_run.sh B2_RUN_DIR 0 val grade
bash scripts/evaluate_stage2_v2_run.sh B3_RUN_DIR 1 val grade
```

Shortlist on validation only. A candidate must preserve BO grade macro F1, improve BO damage macro/per-class balance, and improve more than one weak event rather than merely increasing Rwanda destroyed or Mexico damaged. Do not select from test.

If neither candidate improves the event/class trade-off, freeze A3 and proceed to the frozen test protocol. If one candidate passes, run seeds 3407 and 2026 for it. Before making a stronger SAR claim for the new recipe, repeat its prior-only and fixed shuffled-SAR controls with the same loss or sampler. Only then freeze and evaluate test.

## Artifacts

- `outputs/stage2/v2_analysis/val_best_grade/mean_std.csv`
- `outputs/stage2/v2_analysis/val_best_grade/paired_bootstrap.csv`
- `outputs/stage2/v2_analysis/val_best_grade/per_event_mean_std.csv`
- `outputs/stage2/v2_analysis/val_best_grade/per_disaster_mean_std.csv`
- `outputs/stage2/v2_analysis/val_best_grade/per_event_familiarity_mean_std.csv`
- Individual `val_best_grade/metrics.json`, sample metrics, grouped metrics, previews, and CC-surrogate diagnostics under each formal run.
