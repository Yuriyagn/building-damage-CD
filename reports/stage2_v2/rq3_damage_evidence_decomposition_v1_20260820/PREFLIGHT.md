# RQ3 Damage Evidence Decomposition v1 — implementation preflight

Date: 2026-08-20
Protocol: `rq3_damage_evidence_decomposition_v1.0`
Verdict: **technical preflight passed; E1 exposed-development pilot launched under explicit user amendment**

## Material Passport

- Historical provenance: frozen RQ1 manifests, 56 event-excluded OOF priors, seven outer folds, selected steps `{0:14000, 1:10000, 2:14000, 3:14000, 4:14000, 5:12000, 6:16000}`, and fixed within-event derangements are referenced in place and hash-checked.
- Development evidence: all 2,214 samples and 14 exposed canonical events are represented in the SAR evidence sidecar. This evidence is descriptive only and is forbidden for causal claims or hyperparameter selection.
- Blind evidence: unavailable. No custodian lock has been supplied, no hidden labels have been accessed, and no scored package has been submitted.
- Execution amendment: the user explicitly waived the blind-lock startup requirement for exposed development experiments. The original protocol remains preserved; the amendment is `unlocked_development_amendment.yaml`.
- Claim boundary: E1 may be screened on the 14 exposed events, but no RQ3 factor has yet passed its pilot gate, and independent blind confirmation, unseen-event robustness and deployment-readiness claims remain forbidden.

## Implemented interfaces

- `E1 Instance`: 4-connected ROIs from the frozen event-excluded OOF building prior, minimum area 4 pixels, label purity 0.8, masked mean+max pooling, three-grade instance logits, and rasterized four-class-compatible semantic output.
- `E2 Wavelet`: deterministic two-level Haar LL/LH/HL/HH residuals, zero-initialized at two encoder scales. It adds 512 parameters (`0.002095%` of F0), below the 15% limit.
- `E3 Sensor`: train-fold-only background median/MAD with group → provider → global fallback and optional train-fit incidence-angle/GSD modulation. It is explicitly an 8-bit acquisition-conditioned normalization, not radiometric calibration.
- `E4 Reliability`: balanced paired/deranged training rows using the frozen C3 mapping, metadata moved with its SAR source, auxiliary reliability supervision, and `control + g * residual` fusion. The `g=0` endpoint is unit-tested for exact control-logit equality.
- Runner: pilot is exactly `7 × C2/C3 × seed 42 = 14`; full is exactly `7 × C1/C2/C3 × 3 seeds = 63`. It accepts either the original blind lock or the explicit unlocked-development amendment, never both. E2–E4 require a passed latest-anchor summary; failed factors cannot be silently carried forward.
- Blind lock remains available for a future independent evidence tier but is no longer a startup requirement for the explicitly amended development-only path.

## Evidence audit

- Sample/event count: `2214 / 14`.
- Official-polygon traceability: `1383` samples (`262` direct official instance records; `1121` decoded-identical pairs whose source points to the official BRIGHT instance JSON).
- Per-sample hashes: present for SAR, OOF prior and four-class target in the final sidecar.
- Official BRIGHT pixel/georeference recovery: `2189` unique matches, `3` ambiguous matches and `22` unmatched samples. At least one exact decoded-pixel match exists for `2192 / 2214` samples (`99.0063%`); only unique matches are assigned a geographic footprint (`98.8708%`).
- Binary-container identity is deliberately separate: `262 / 2214` files are byte-identical to an official GeoTIFF. The canonical recovery audit is `bright_georeference_recovery_pixels_v1_20260820/AUDIT.json`; the earlier file-SHA-only directory is diagnostic and noncanonical.
- High-confidence STAC mapping: `0 / 2214`. No local Capella/Umbra STAC or extended metadata files were found.
- Official catalog follow-up found plausible Capella candidates for six events and four official Umbra candidates covering Noto, but candidate multiplicity prevents exact per-sample attribution. Details and source hashes are in `SAR_PROVENANCE_RECOVERY.md` and the generated Noto candidate audit.
- E3 admission: **failed** (`0% < 90%` overall and `0% < 80%` in every event). Geographic footprint recovery must not be substituted for collect-level metadata.

Artifacts live outside Git under:

`/home/yr/code/Building damage change detection/outputs/stage2/rq3_damage_evidence_decomposition_v1_20260820/`

The canonical evidence audit is `evidence_sidecar_final/AUDIT.json`; earlier `retry*` directories are retained as noncanonical diagnostic attempts and must not be used for claims.

## Verification

- Runtime gate: passed; two NVIDIA RTX 4090 D GPUs, CUDA PyTorch 2.7.0+cu126, minimal Stage-2 data present, 734 GB free at check time.
- Syntax: passed for all new and modified Python entry points.
- Tests: the complete repository suite passed (`101 passed`), including 16 RQ3-specific tests. The unlocked path is tested to require an explicit amendment, reject conflicting contracts and forbid blind/generalization/deployment claims.
- CUDA E1 smoke: passed with ImageNet initialization and finite train/validation losses.
- Two-batch overfit with one CE contribution per valid component and background-inclusive purity: passed; loss `1.028579 → 0.328123`, reduction `68.10%` (required at least 50%).
- 500-step CUDA throughput using the actual per-instance loss: F0 `15.8882 s`; E1 `18.2922 s`; E1/F0 wall-time ratio `1.15131`, below the `2×` ceiling. Peak allocated memory was 1.171 GB (F0) and 1.302 GB (E1) at batch 2, 512².
- A no-pretrained diagnostic smoke produced non-finite validation logits because one update leaves random BatchNorm statistics unsuitable for FP16 evaluation. The formal ImageNet-initialized path passed; the failed diagnostic outputs are not evidence.

## Execution amendment and current action

The E1 pilot started at `2026-08-20T20:36:15+08:00` in tmux session `rq3_e1_unlocked_pilot_20260820_203615`. Its output root is `/home/yr/code/Building damage change detection/outputs/stage2/rq3_damage_evidence_decomposition_unlocked_dev_v1_20260820`. Startup verification found two tasks running, twelve pending and no failures. See `UNLOCKED_DEVELOPMENT_LAUNCH.md` for the command and monitoring receipt.

The pilot verdict remains pending until all 14 training/evaluation artifacts pass completion checks and the preregistered metrics are summarized. E3 retains its independent metadata-coverage blocker. E2–E4 remain conditional on the preceding factor's frozen passing verdict.
