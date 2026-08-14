# R0 vs R4 predicted-prior comparison (2026-08-04)

## Protocol

- R0 train/val/test: 1372/290/290.
- R4 train/val/test: 1634/290/290; the only data change is 262 BRIGHT train samples.
- Both variants use frozen O1 predicted building priors, identical model/optimization/augmentation, and seeds 42, 3407, 2026.
- Selection was completed on validation only using `best_bo_grade_macro_f1.pth`.
- Test was evaluated once after all six checkpoints were frozen, with argmax decoding and gate threshold 0.6. No test-time threshold or checkpoint selection was performed.

## Validation result

Values are mean +/- sample standard deviation across three seeds.

| Metric | R0 | R4 | R4 - R0 |
|---|---:|---:|---:|
| BO three-grade macro F1 (primary) | 0.3098 +/- 0.0035 | 0.3494 +/- 0.0238 | +0.0396 |
| BO damage macro F1 | 0.0573 +/- 0.0142 | 0.0801 +/- 0.0247 | +0.0229 |
| BO Damaged F1 | 0.0726 +/- 0.0268 | 0.0578 +/- 0.0376 | -0.0149 |
| BO Destroyed F1 | 0.0419 +/- 0.0212 | 0.1025 +/- 0.0647 | +0.0606 |
| BO binary-damage F1 | 0.0802 +/- 0.0177 | 0.1213 +/- 0.0354 | +0.0411 |
| Predicted-gate four-class macro F1 | 0.4222 +/- 0.0022 | 0.4408 +/- 0.0120 | +0.0186 |

Primary-metric paired deltas for seeds 42/3407/2026 are +0.0647, +0.0435, and +0.0106. Thus R4 wins the frozen primary comparison in all three seeds.

Full-image event aggregation gives:

| Event-level metric | R0 mean | R4 mean | R4 - R0 |
|---|---:|---:|---:|
| Event-macro BO F1 | 0.3183 | 0.3390 | +0.0208 |
| Worst-event BO F1 | 0.2418 | 0.3006 | +0.0588 |
| Damaged event-mean F1 | 0.0672 | 0.0667 | -0.0005 |
| Destroyed event-mean F1 | 0.0941 | 0.1039 | +0.0098 |

The event-macro paired deltas are +0.0417, +0.0215, and -0.0009. Event-level improvement is therefore positive on average but not unanimous across seeds.

## One-time diagnostic test

The diagnostic test contains historically observed events and is not a pristine unseen-event test.

| Metric | R0 | R4 | R4 - R0 |
|---|---:|---:|---:|
| BO three-grade macro F1 | 0.2719 +/- 0.0196 | 0.3217 +/- 0.0845 | +0.0498 |
| BO damage macro F1 | 0.1258 +/- 0.0322 | 0.1858 +/- 0.0996 | +0.0600 |
| BO Damaged F1 | 0.0313 +/- 0.0303 | 0.1459 +/- 0.2043 | +0.1147 |
| BO Destroyed F1 | 0.2203 +/- 0.0376 | 0.2256 +/- 0.0121 | +0.0053 |
| BO binary-damage F1 | 0.3559 +/- 0.0323 | 0.4529 +/- 0.2161 | +0.0971 |
| Predicted-gate four-class macro F1 | 0.4037 +/- 0.0117 | 0.4356 +/- 0.0535 | +0.0319 |

Primary-metric paired deltas for seeds 42/3407/2026 are -0.0185, +0.0194, and +0.1485. The positive mean is dominated by seed 2026 and is not stable across seeds.

Test event-macro BO F1 improves from 0.3396 to 0.3751 (+0.0355 mean) and is positive in all three seeds, but only three test events are available. The worst event is `mexico_hurricane` for every run; its mean BO F1 changes from 0.1127 to 0.1590, with one of three seeds regressing.

## Verdict

R4 passes the fixed R0 comparison: adding 262 BRIGHT samples with predicted O1 priors improves the validation primary metric in all three seeds and improves average event-level performance.

However, the data addition does not solve event-category shortcut learning:

- Validation Damaged F1 decreases, while most of the aggregate gain comes from Destroyed.
- Event-level gains are smaller than pixel-weighted gains and one validation seed is slightly negative.
- Diagnostic-test variance is substantially larger for R4; several means are driven by seed 2026.
- BRIGHT adds samples but not independent new disaster events, and the test events are historically observed.
- The split still lacks formal geospatial-footprint independence proof.

The defensible conclusion is: BRIGHT supplementation is useful and R4 should replace R0 as the stronger historical-diagnostic training set, but additional independently sourced events with Damaged/Destroyed support are still required before claiming robust cross-event generalization.
