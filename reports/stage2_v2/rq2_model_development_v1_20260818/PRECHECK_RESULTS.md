# RQ2 Model Development v1 — technical preflight

Verdict: **technical preflight passed; internal development is authorized**.

This is implementation evidence only. It is not a model-selection result and
does not change the frozen RQ1 scientific failure verdict.

## Checks completed

- `sam3` runtime gate passed with two NVIDIA GeForce RTX 4090 D devices.
- Frozen RQ1 inputs were verified by digest: 56 Stage-1 OOF runs and exports,
  seven finalized outer folds, and 28 C3 mapping files. They are reused by
  reference; the historical artifact tree was not copied.
- All 84 repository unit tests and all 14 RQ2-focused tests passed.
- All three candidates passed real CUDA smoke tests.
- R2-A has 26,558,579 trainable parameters versus F0's 24,442,931, a ratio of
  1.0866 within the frozen `[0.85, 1.15]` budget.
- Thirty-epoch two-batch overfit reductions were 74.96% (R2-A), 84.12%
  (R2-B), and 74.33% (R2-C), each above the 50% technical threshold.
- On the same GPU and outer-0 C2 data, 500 steps plus full validation took
  217.55 seconds for F0 and 212.01 seconds for R2-A. The 0.9745 ratio passes
  the frozen `<=2.5` wall-time limit.
- The placeholder blind-evaluator lock is rejected fail-closed. The synthetic
  label-free manifest, fixed C3 derangement, and 12-tree uint8 prediction
  package completed format validation without computing scores.

The full machine-readable record is in `PRECHECK_RESULTS.json`. Metrics, logs,
resolved configs and a retention audit remain outside Git under the unique
preflight artifact root recorded there. The five reproducible preflight-only
checkpoint directories were removed after verification, reducing the artifact
tree from 2.0 GB to under 1 MB; no scientific result checkpoint was removed.

## Development authorization and remaining evidence limit

The user removed the third-party lock as a prerequisite on 2026-08-19, so the
42-run pilot may start. Until a later independent blind evaluation exists, all
results remain development-only evidence from 14 exposed events. The old
California/Jamaica BRIGHT events cannot substitute for independent evidence.
