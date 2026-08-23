# RQ3-E1 numerical recovery status

- Parent protocol: `rq3_damage_evidence_decomposition_v1.0`
- Amendment: `rq3_e1_numerical_integrity_amendment_v1.0`
- Evidence scope: 14 exposed development events; no independent blind evaluation.
- Historical E1: retained as a technical failure because all 14 completed runs contained non-finite losses/checkpoints. It is not treated as a scientific negative result.
- Current stage: frozen after a numerical preflight hard-gate failure.
- Formal pilot/full training: not started.

The recovery branch must stop without retry or recipe changes if AMP-on E1, the exact-loss overfit gate, the C2/C3 condition contract, numerical diagnostics, or the throughput gate fails.

## Execution result

- Repository regression tests: `107 passed`.
- Runtime gate: passed with two idle NVIDIA RTX 4090 D GPUs and the expected minimal Stage-2 data root.
- Outer-0 validation C2/C3 contract: passed on 680 samples; C3 source IDs were 100% deranged, C2/C3 aggregate tensor digests differed, no identical tensor pairs were observed, and SAR metadata synchronization had zero mismatches.
- One-update AMP CUDA smoke: passed with finite training and validation outputs.
- Required exact-loss two-batch AMP overfit: **failed numerically at optimizer step 2**. After `GradScaler.unscale_()`, the first detected non-finite gradient was `base_model.encoder.conv1.weight`; AMP scale was `65536`.

The 30-step 50% loss-reduction gate was therefore not reached. Under the frozen amendment, this failure stops the recovery funnel immediately: the seven-fold condition audit, three 1,000-step diagnostics, 500-step throughput comparison, E1 pilot and E1 full were not started. No LR, loss, sampler, gradient clipping, AMP, ROI, backbone or selected-step change was made.

## Verdict

The FP32 component pooling and `int64` voting repairs are covered by tests, but they are insufficient to establish AMP numerical recovery under the approved fail-fast gradient contract. This is a valid technical recovery failure, not evidence that instance supervision scientifically harms damage grading.
