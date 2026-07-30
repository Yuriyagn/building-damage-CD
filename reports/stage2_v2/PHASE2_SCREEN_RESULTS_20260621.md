# Stage-2 v2 Phase-2 Seed-42 Screen

> Integrity note, 2026-06-22: legacy test results were retracted. This validation-only screen is preserved as historical recipe-selection evidence because corrected auditing found no train-val exact duplicates or event overlap. Strict-v1 requires fresh training.

Date: 2026-06-21

Status: B2 and B3 rejected on event-held-out validation. The original A3 weighted-CE recipe is frozen before any formal test evaluation.

## Protocol

- Same seed 42, model, data, crop policy, class weights, optimizer, and checkpoint rule as A3.
- B2 changes only weighted CE to weighted focal CE (`gamma=2.0`).
- B3 changes only natural image sampling to per-image disaster weight `n_d^-0.5`.
- All results use `best_bo_grade_macro_f1.pth` on all 506 validation images.
- Intervals use 10,000 sample-paired bootstrap resamples. They quantify sample uncertainty for this seed; they do not replace multi-seed replication.

## Global result

| Recipe | BO grade macro F1 | Intact F1 | Damaged F1 | Destroyed F1 | Damage macro F1 | Binary damage F1 | Predicted-gate macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A3 weighted CE | **0.5103** | 0.6140 | **0.5903** | **0.3264** | **0.4584** | **0.6215** | **0.5403** |
| B2 weighted focal | 0.4930 | **0.6974** | 0.4638 | 0.3178 | 0.3908 | 0.5381 | 0.5282 |
| B3 disaster-balanced | 0.4482 | 0.5720 | 0.4035 | 0.3692 | 0.3863 | 0.4425 | 0.4999 |

## Paired differences against A3

| Candidate | Metric | Delta | Paired 95% CI |
| --- | --- | ---: | ---: |
| B2 | BO grade macro F1 | -0.0173 | [-0.0426, 0.0110] |
| B2 | BO damage macro F1 | -0.0676 | [-0.1001, -0.0294] |
| B2 | Intact F1 | +0.0834 | [0.0596, 0.1084] |
| B2 | Damaged F1 | -0.1265 | [-0.1585, -0.0966] |
| B2 | Destroyed F1 | -0.0086 | [-0.0667, 0.0627] |
| B3 | BO grade macro F1 | -0.0621 | [-0.0974, -0.0290] |
| B3 | BO damage macro F1 | -0.0720 | [-0.1145, -0.0344] |
| B3 | Intact F1 | -0.0421 | [-0.0777, -0.0062] |
| B3 | Damaged F1 | -0.1868 | [-0.2386, -0.1374] |
| B3 | Destroyed F1 | +0.0427 | [-0.0115, 0.0916] |

## Per-event class behavior

| Event | Recipe | Grade macro | Intact | Damaged | Destroyed | Damage macro |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| bata_explosion | A3 / B2 / B3 | 0.3031 / 0.3203 / 0.2498 | 0.5129 / 0.5551 / 0.4665 | 0.3597 / 0.3682 / 0.2251 | 0.0368 / 0.0378 / 0.0577 | 0.1982 / 0.2030 / 0.1414 |
| mexico_hurricane | A3 / B2 / B3 | 0.3144 / 0.3432 / 0.1957 | 0.0494 / 0.0882 / 0.2696 | 0.8745 / 0.7709 / 0.3080 | 0.0193 / 0.1705 / 0.0095 | 0.4469 / 0.4707 / 0.1588 |
| rwanda_volcano | A3 / B2 / B3 | 0.6421 / 0.6661 / 0.4829 | 0.9569 / 0.9789 / 0.5418 | 0.0913 / 0.1097 / 0.0142 | 0.8781 / 0.9098 / 0.8926 | 0.4847 / 0.5097 / 0.4534 |
| ukraine_conflict | A3 / B2 / B3 | 0.3469 / 0.3262 / 0.3769 | 0.5782 / 0.6888 / 0.6348 | 0.4543 / 0.2711 / 0.4651 | 0.0081 / 0.0188 / 0.0308 | 0.2312 / 0.1449 / 0.2479 |

B2 does improve destroyed F1 in Mexico and slightly improves Bata/Rwanda, but it achieves this by substantially reducing damaged F1 globally and in the dominant 340-image Ukraine event. B3 slightly raises global destroyed F1, but its change is inconclusive and it significantly lowers grade macro, damage macro, intact, and damaged F1. Neither candidate resolves the event/class trade-off.

## Frozen decision

1. Reject B2 and B3; do not spend two additional seeds on them.
2. Freeze A3 weighted CE as the Stage-2 v2 primary recipe.
3. Preserve A1 prior-only, A2 SAR-only, and fixed within-event shuffled A4 as the formal controls.
4. Run test once for all three formal seeds' primary checkpoints. Do not tune any setting from test.
5. Report test by event familiarity and retain the name `cc_surrogate` for connected-component diagnostics.

The remaining event-dependent grade failure is a documented limitation. The next scientific remedy should add temporal pre/post information, improve alignment/labels, or introduce stronger event-generalization data—not continue tuning loss functions against test.
