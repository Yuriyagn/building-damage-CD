# Workspace cleanup — 2026-07-31

## Outcome

- Applied removal plan: `99.023 GiB`, `10,337` files, `424` exact targets
- `stage1_optical_building/outputs`: about `100 GiB -> 4.7 GiB`
- isolated author reproduction: about `4.6 GiB -> 650 MiB`
- datasets: unchanged (`86 GiB`, read-only)
- selected paths remaining after apply: `0`

Machine-readable evidence:

- `workspace_cleanup_20260731_dry_run.json`
- `workspace_cleanup_20260731_applied.json`

## Removed

- non-primary, legacy, debug, and rejected-branch checkpoints
- 40 per-epoch UABCD reproduction checkpoints
- rejected OGSR derived features
- generated predictions and previews under `outputs/`
- Python and test caches

## Retained

- all source, configs, manifests, reports, scalar metrics, permutations, and logs
- Stage-1 O1–O4 primary checkpoints
- strict-v1 A1–A4 primary checkpoints for seeds 42/3407/2026
- current E2 primary checkpoint
- upstream UABCD best checkpoint
- unified-v1 formal primary checkpoints
- external PVTv2 backbone weights

## Repeatable policy

Dry run:

```bash
python scripts/prune_workspace_artifacts.py
```

Apply only after reviewing the generated JSON:

```bash
python scripts/prune_workspace_artifacts.py --apply
```

The script refuses paths outside this workspace and never selects datasets,
source, configs, manifests, reports, scalar metrics, or logs.
