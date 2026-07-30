# Codex Notes For This Training Repo

Read `PROJECT_EXPERIENCE.md`, `STAGE2_BASELINE_IMPLEMENTATION.md`, and the parent workspace's `stage2-v2.md` when it is available.

This Git repository contains both Stage-1 DisasterM3 optical building extraction and the Stage-2 SAR damage baseline. Do not commit data packages, model weights, checkpoints, logs, generated predictions, or `__pycache__`. Preserve the existing dirty/untracked Stage-2 work.

Use Miniconda environment `sam3`. Stage-2 mainline must use `${STAGE2_DATA_ROOT}`. `instruction.sh` resolves either the legacy `/home/yr/code/datasets` deployment or the current workspace's sibling `datasets/` directory.

Do not start long GPU training unless the user explicitly asks. Verify the host, data path, `tmux ls`, and `nvidia-smi` first. Formal training and full inference must run in a named tmux session with a unique output directory and log; never hold the Codex foreground.

For Stage-2 v2, run the full-content hash audit with the event-disjoint gate before training, select recipes on validation rather than test, keep prior-only and fixed shuffled-SAR controls, and use three formal seeds. The task is pixel-level semantic damage mapping; connected-component metrics are not required.

The 2026-06-21 legacy test is retracted: 661 complete-content duplicate groups were found, including 105 exact train-test duplicates. Never cite the legacy test tables. The formal training split is now `manifests/stage2_v2_clean_human_reviewed_20260624/`: it is globally deduplicated and canonical-event-disjoint (`1207/357/388`), incorporates user overlap-review decisions, and passed the strict audit. Fresh training is still required. Read `reports/stage2_v2/DATA_INTEGRITY_CORRECTION_20260622.md` and `reports/stage2_v2/HUMAN_REVIEWED_CLEAN_DATASET_20260624.md` before any new Stage-2 work.

The `relaxed-overlap-v1` automatic audit and user review are complete. Strict-v1 train-test medium/low were recorded as `not_overlap`; the formal clean set removes 0 strict-v1 samples. A legacy diagnostic export removes 107 train-side samples but still has 556 within-split full duplicates, so it is not a formal protocol. This image-based audit still does not prove formal geospatial footprint independence.
