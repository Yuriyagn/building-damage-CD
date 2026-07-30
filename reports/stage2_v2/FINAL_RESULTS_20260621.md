# [RETRACTED] Stage-2 v2 Legacy Test Results

Original date: 2026-06-21  
Retracted: 2026-06-22

Status: **retracted because the legacy test contains exact train-test content duplicates. Do not cite the tables below as valid test results.** A3 was frozen before test, but that procedural control cannot repair a contaminated split.

The authoritative correction is `DATA_INTEGRITY_CORRECTION_20260622.md`. The tables below are preserved only as an audit trail of the withdrawn result.

## Protocol integrity

- Primary checkpoint: `best_bo_grade_macro_f1.pth`.
- Test samples: 564 for A1-A3; 563 for A4 because the predeclared within-event singleton policy excludes one sample.
- Formal seeds: 42, 3407, 2026.
- Statistics: mean ± sample standard deviation and 10,000 sample-paired bootstrap resamples averaged across matched seeds.
- A3 was frozen before any formal test metric existed; no test-driven configuration change was made.
- `cc_surrogate` is a connected-component diagnostic, not a true building-instance result.

## Withdrawn legacy test table

| Experiment | BO grade macro F1 | Intact F1 | Damaged F1 | Destroyed F1 | Damage macro F1 | Binary damage F1 | Predicted-gate macro F1 | Predicted-gate mIoU | CC-surrogate macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 all-intact rule | 0.2744 | 0.8233 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.4074 | 0.3580 | — |
| A1 prior-only | 0.3042 ± 0.0075 | 0.3251 ± 0.1176 | 0.3664 ± 0.0822 | 0.2211 ± 0.0180 | 0.2938 ± 0.0497 | 0.4242 ± 0.0655 | 0.4308 ± 0.0025 | 0.3436 ± 0.0019 | 0.2590 ± 0.0239 |
| A2 SAR-only | 0.4009 ± 0.0314 | **0.5218 ± 0.0037** | 0.2837 ± 0.0255 | 0.3974 ± 0.0775 | 0.3405 ± 0.0467 | 0.3837 ± 0.0249 | 0.4993 ± 0.0215 | 0.3927 ± 0.0160 | 0.3118 ± 0.0369 |
| A3 paired SAR + predicted prior | **0.4442 ± 0.0188** | 0.4944 ± 0.0410 | **0.3728 ± 0.0407** | **0.4655 ± 0.0215** | **0.4191 ± 0.0258** | **0.4751 ± 0.0414** | **0.5322 ± 0.0112** | **0.4158 ± 0.0089** | **0.3485 ± 0.0262** |
| A4 fixed within-event shuffled SAR + predicted prior | 0.2507 ± 0.0425 | 0.2777 ± 0.1358 | 0.2598 ± 0.0262 | 0.2145 ± 0.0052 | 0.2372 ± 0.0150 | 0.4053 ± 0.0097 | 0.3999 ± 0.0267 | 0.3234 ± 0.0177 | 0.2143 ± 0.0662 |

Historical contaminated-split values were `0.4560 / 0.4225 / 0.4542` for seeds 42/3407/2026; their apparent stability does not make them valid test evidence.

## Withdrawn paired test evidence

| Comparison | Metric | Mean delta | Paired 95% CI |
| --- | --- | ---: | ---: |
| A3 − A1 | BO grade macro F1 | +0.1400 | [0.1127, 0.1618] |
| A3 − A1 | BO damage macro F1 | +0.1254 | [0.0954, 0.1450] |
| A3 − A2 | BO grade macro F1 | +0.0433 | [0.0266, 0.0576] |
| A3 − A2 | BO damage macro F1 | +0.0786 | [0.0582, 0.0962] |
| A3 − A4 | BO grade macro F1 | +0.1938 | [0.1646, 0.2161] |
| A3 − A4 | BO damage macro F1 | +0.1823 | [0.1448, 0.2084] |

The original per-class and control interpretations are withdrawn because the paired sample population contains exact train-test duplicates. See the corrected post-hoc diagnostic in `DATA_INTEGRITY_CORRECTION_20260622.md`.

## Event-familiarity generalization

| Test stratum | Images | A1 grade macro | A2 grade macro | A3 grade macro | A4 grade macro | A3 intact / damaged / destroyed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train-seen event | 341 | 0.2843 | 0.4095 | **0.4309** | 0.2536 | 0.4925 / 0.2199 / 0.5804 |
| val-seen-only event | 197 | 0.2703 | 0.3188 | **0.3860** | 0.2432 | 0.5322 / 0.6164 / 0.0094 |
| globally unseen event | 26 | 0.1179 | 0.1451 | 0.1541 | **0.1823** | 0.3226 / 0.0896 / 0.0500 |

This old event-familiarity partition was based only on event-ID strings and did not detect renamed duplicate content. It is withdrawn and must not be interpreted as a valid generalization analysis.

Class behavior remains event/disaster dependent:

- volcano: A3 intact/damaged/destroyed F1 `0.9361 / 0.1256 / 0.8401`;
- hurricane: `0.1196 / 0.6807 / 0.0033`;
- earthquake: `0.2250 / 0.1940 / 0.0102`.

These disaster-specific values are retained only to show how the contaminated aggregate was formed; they are not valid test estimates.

## Withdrawn deployment and surrogate diagnostics

- A3 predicted-prior end-to-end macro F1: `0.5322 ± 0.0112`.
- A3 predicted-prior end-to-end four-class mIoU: `0.4158 ± 0.0089`.
- A3 CC-surrogate macro F1: `0.3485 ± 0.0262`.
- A3 CC-surrogate intact/damaged/destroyed F1: `0.5307 / 0.2085 / 0.3063`.

All values in this section are withdrawn. Connected-component evaluation is also outside the current pixel-level project objective.

## Withdrawn relation to Stage-2 v1

On the same test benchmark, v2 A3 versus v1 S2 changes:

- BO grade macro F1: `0.4120 -> 0.4442` (`+0.0322`);
- BO damage macro F1: `0.3376 -> 0.4191` (`+0.0815`);
- damaged F1: `0.2127 -> 0.3728` (`+0.1601`);
- destroyed F1: `0.4626 -> 0.4655` (`+0.0029`);
- predicted-gate four-class mIoU: `0.3953 -> 0.4158` (`+0.0205`).

This comparison is withdrawn because its v2 side uses the contaminated legacy test.

## Corrected conclusion

The legacy test conclusion is withdrawn. Full-content SHA-256 found 661 duplicate groups in the legacy manifest union, including 105 exact train-test duplicates. After excluding those 105 test samples post hoc, A3−A1 BO grade macro F1 is only `+0.0089 [-0.0086, 0.0265]`, while A3−A4 remains `+0.0442 [0.0281, 0.0601]`. This diagnostic cannot replace a predeclared formal test.

Legacy validation remains useful because train and legacy val have no event overlap or exact content duplicates. A new globally deduplicated, canonical-event-disjoint strict-v1 split has been built. A1–A4 must be retrained from scratch on strict-v1 before any new formal test claim. The project target remains pixel-level four-class semantic output; connected-component analysis is not a required objective.

## Historical machine-readable artifacts

- `outputs/stage2/v2_analysis/test_best_grade/runs.csv`
- `outputs/stage2/v2_analysis/test_best_grade/mean_std.csv`
- `outputs/stage2/v2_analysis/test_best_grade/paired_bootstrap.csv`
- `outputs/stage2/v2_analysis/test_best_grade/per_event_familiarity_mean_std.csv`
- `outputs/stage2/v2_analysis/test_best_grade/per_event_mean_std.csv`
- `outputs/stage2/v2_analysis/test_best_grade/per_disaster_mean_std.csv`
- Individual `test_best_grade/metrics.json`, sample metrics, grouped metrics, previews, and CC-surrogate diagnostics under every formal run.

These artifacts belong to the withdrawn legacy test. Corrected integrity artifacts are under `outputs/stage2/v2_data_integrity_20260622/`, and strict manifests are under `manifests/stage2_v2_strict_v1/`.
