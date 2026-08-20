## Material Passport

| Material | Source / date | Role | Current status |
|---|---|---|---|
| RQ1 frozen result | `rq1_paired_sar_nested_cv_v1_20260818` | Development baseline and failed worst-event anchor | Complete; 14 events are exposed development data |
| RQ1 finalized manifests / OOF priors / C3 maps | Local RQ1 artifact tree, 2026-08-18 | Reused inputs for RQ2; not copied | Complete; hashes must match before launch |
| RQ2 protocol | `configs/rq2_model_development_v1/protocol.yaml` | Candidates, gates, stopping rule and claim boundary | Internal development authorized on 2026-08-19 |
| Third-party blind evaluator lock | Future external custodian | Optional post-development confirmation | Not arranged; does not block internal development |
| Existing BRIGHT CVPRW26 test | California wildfire 2025 and Jamaica hurricane 2025 | Explicit exclusion from future confirmation | Exposed through prior submissions |

# RQ2 Model Development v1 — internal development authorization

Status: **`completed_stopped_no_eligible_candidate`**
Evidence scope: **14 exposed events, development only**.

The authorized 42-run pilot completed on 2026-08-19. No candidate passed all
seven frozen gates, so the branch stopped without multi-seed extension. See
[RESULTS.md](RESULTS.md) and [summary.json](summary.json) for the frozen result.

On 2026-08-19 the user removed the third-party blind-evaluator lock as a
prerequisite for the internal development funnel. The 42-run pilot was therefore
allowed to run before an external blind evaluator was arranged.

This change affects execution order, not claim strength. Development results
may support statements about nested event-held-out behavior on the 14 exposed
events. They cannot establish independent unseen-event generalization,
deployment readiness, or external replication.

## Frozen development funnel

- R2-A: parameter-matched dual ResNet18 optical/SAR encoders with fixed multi-scale concat/absolute-difference/prior fusion.
- R2-B: affected then damaged/destroyed factorized probabilities with building-only hierarchical NLL.
- R2-C: canonical-event sqrt-frequency sampling capped to `[0.5, 4.0]`.
- Pilot: `3 candidates × 7 folds × C2/C3 × seed42 = 42` runs.
- Exactly one eligible pilot candidate may receive seeds 3407 and 2026, adding 28 runs.
- Candidate advancement, full-development gates and failure-implies-stop rules are unchanged.
- Existing RQ1 fold-specific steps, OOF priors, manifests, C3 mappings and baseline evaluations are reused by reference.

## Claim boundary

Before an independent blind evaluation exists, every result artifact must state:

- `evidence_scope = 14_exposed_events_development_only`;
- `unseen_event_generalization_claim_allowed = false`;
- old California/Jamaica BRIGHT scores are not confirmation evidence;
- a passed development gate identifies a candidate for future confirmation,
  not a confirmed robust model.

## Future blind confirmation

The lock schema, label-free derangement builder, prediction-package validator
and unsent request draft are retained for an optional later confirmation. A
future blind submission still requires the real lock, at least three new
canonical events, hidden labels and the one-shot scoring contract. None of
those conditions are required to begin the internal development runs.
