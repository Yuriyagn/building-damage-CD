# RQ2 third-party blind evaluation request

Status: **optional post-development draft — not sent**
Preferred recipient: BRIGHT Challenge organizers, public contact
`Qschrx@gmail.com` listed on the [official challenge page](https://chrx97.com/challenge.html).
This request is no longer a prerequisite for the internal RQ2 development
funnel. It may be used later only if independent blind confirmation is pursued.

## Email draft

Subject: Request for one-shot private unseen-event evaluation for pixel-level BRIGHT damage research

Dear BRIGHT Challenge organizers,

We are studying pixel-level building damage grading from pre-event optical RGB
and post-event SAR. Our current development benchmark contains 14 exposed
events and uses event-excluded building priors, event-grouped cross-validation,
and a fixed within-event deranged-SAR negative control.

We would like to ask whether your team could act as an independent evaluation
custodian for a one-shot unseen-event confirmation. The previously released
CVPR 2026 test events (California wildfire 2025 and Jamaica hurricane 2025)
cannot be used because our machine has already generated multiple submissions
for them. We therefore require at least three genuinely new canonical events,
ideally five, that have not been available to our development process.

We propose the following isolation protocol:

1. Your team freezes the event set, hidden pixel labels, event mapping,
   geospatial/sensor metadata and metric implementation before our formal model
   development starts.
2. Labels remain entirely with your team. Either unlabeled inputs are released
   only after our model/checkpoint hashes are frozen, or your team runs our
   deterministic inference bundle.
3. We perform one format-only synthetic dry run and one scored submission.
4. You return per-event class support and confusion matrices plus pooled and
   event-macro pixel metrics, without returning labels.
5. No model change or resubmission is allowed after scores are disclosed.

Our required semantic labels are background, intact, damaged and destroyed.
The primary metric is event-macro building-only damage F1; we also report
three-grade macro F1, Damaged, Destroyed, binary damage and end-to-end
four-class metrics. We can provide the exact evaluator schema and prediction
format before any data transfer.

Please let us know whether a fresh private event set and this one-shot pixel
evaluation are feasible, and what data-use or publication conditions would
apply.

Sincerely,

[Researcher name and affiliation]

## Required written confirmation

The response must provide or authorize the fields in
`configs/rq2_model_development_v1/blind_evaluator_lock.schema.json`:

- real custodian identity and acceptance timestamp;
- SHA-256 of the custodian's written confirmation or signed protocol record;
- event count and frozen event-list digest;
- input manifest, hidden-label and metric-script SHA-256 values;
- confirmation that labels are withheld and pixel semantic metrics are supported;
- Damaged/Destroyed event coverage;
- geospatial non-overlap verdict;
- one dry-run / one scored-submission limits and no-score resubmission rule.

An informal statement that an old leaderboard remains open is not sufficient.
