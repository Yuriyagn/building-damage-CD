# Experiment registry

`registry.json` is the canonical machine-readable index for experiment status,
provenance, test exposure, gate decisions, and claim alignment.

Validate it with:

```bash
python scripts/validate_experiment_registry.py experiments/registry.json
```

Rules:

- register planned experiments before formal training;
- never promote an exposed test to a blind evaluation by renaming it;
- keep failed, superseded, and retracted entries instead of deleting them;
- use repository-relative artifact paths; checkpoints, logs, outputs, and data
  remain untracked;
- a completed experiment needs run records, a gate decision, and at least one
  claim-provenance verdict.
