# RQ2 Model Development v1 — frozen internal-development result

Status: **`stopped_no_eligible_candidate`**

Completed: **2026-08-19**

Evidence scope: **14 exposed canonical events, development only**

Unseen-event generalization claim allowed: **false**

## Executive verdict

The complete pilot finished without runtime failures:

```text
3 candidates x 7 outer event folds x 2 conditions x seed 42 = 42/42 runs
failed runs = 0
selected candidate = null
```

All three candidates preserved a positive aggregate paired-SAR versus deranged-SAR
signal, but none improved event-macro damage grading over F0 while satisfying the
class and worst-event safety floors. The frozen funnel therefore stops after the
pilot. Seeds 3407 and 2026, full-development training, and blind evaluation were
not run.

This is a valid negative development result, not an execution failure.

## Material Passport

| Material | Role | Integrity state |
|---|---|---|
| RQ1 finalized 7-fold manifests | outer event partition and frozen inner-selected steps | verified by frozen RQ1 SHA-256 anchors |
| 56 RQ1 Stage-1 OOF prior runs/exports | event-excluded building priors | verified, reused by reference |
| 28 RQ1 C3 mappings | fixed within-event deranged-SAR control | verified, no bulk artifact copy |
| RQ2-A | dual ResNet18 optical/SAR streams with fixed five-scale fusion | 14 C2/C3 fold runs completed |
| RQ2-B | hierarchical affected / destroyed-given-affected head and NLL | 14 C2/C3 fold runs completed |
| RQ2-C | capped inverse-event-frequency sampling | 14 C2/C3 fold runs completed |
| Third-party blind evaluator | optional future confirmation | not arranged and not used |

Training provenance was frozen at Git anchor
`004f86a8dd24e02fa04cd49d5832307ba5c24a01` plus:

```text
source_tree_sha256  fc0ca2a353e2be996d8b3f95f1552a290c2ad7cd996ed2d0199f30912eee7d9a
tracked_diff_sha256 4a56f493412e85cec8ec55a29f14bee2a7112bfdf1ecf2958a9018e5bce2f079
data_audit_sha256   ff801bba052969dc95122ac01244853300b8324e9b397f8e63dc3139bec5022b
```

All 42 C2/C3 comparisons share the same recorded initialization provenance.

## Candidate definitions

- **R2-A dual stream:** independent ImageNet ResNet18 encoders for pre-event RGB
  and post-event SAR, fixed optical/SAR/absolute-difference/prior fusion, and a
  standard U-Net decoder.
- **R2-B hierarchical grade head:** F0 encoder/decoder with affected and
  destroyed-given-affected logits converted to normalized three-grade
  log-probabilities and optimized with building-only weighted NLL.
- **R2-C capped event sampling:** unchanged F0 model/loss with per-image event
  weight `clip((median_event_count / event_count)^0.5, 0.5, 4.0)`, normalized
  replacement sampling, and unchanged epoch size.

`C2` uses sample-paired SAR. `C3` uses the frozen within-event deranged SAR.
F0-C2 is the frozen seed-42 RQ1 baseline.

## Aggregate development metrics

All values are macro-averaged over the 14 outer-held-out events. `damage` is the
mean of Damaged and Destroyed F1; `grade` is the Intact/Damaged/Destroyed macro F1.

| Model-condition | Damage F1 | Grade F1 | Damaged F1 | Destroyed F1 |
|---|---:|---:|---:|---:|
| F0-C2 | 0.193499 | 0.350972 | 0.144764 | 0.242235 |
| F0-C3 | 0.116251 | 0.268985 | 0.126573 | 0.105929 |
| R2-A-C2 | 0.169288 | 0.363491 | 0.147180 | 0.191397 |
| R2-A-C3 | 0.108748 | 0.300567 | 0.147475 | 0.070020 |
| R2-B-C2 | 0.178981 | 0.344575 | 0.141581 | 0.216380 |
| R2-B-C3 | 0.116582 | 0.278977 | 0.136240 | 0.096924 |
| R2-C-C2 | 0.167985 | 0.331891 | 0.128672 | 0.207298 |
| R2-C-C3 | 0.124750 | 0.283178 | 0.169466 | 0.080033 |

## Frozen pilot gates

| Candidate | C2-F0 damage | Positive events | Damaged delta | Destroyed delta | C2-C3 damage | Worst grade delta vs F0 | Worst grade delta vs C3 | Passed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R2-A | -0.024211 | 4/14 | +0.002416 | -0.050839 | +0.060541 | -0.176157 | -0.035474 | 2/7 |
| R2-B | -0.014519 | 6/14 | -0.003182 | -0.025855 | +0.062398 | -0.100161 | -0.026421 | 3/7 |
| R2-C | -0.025515 | 6/14 | -0.016092 | -0.034937 | +0.043235 | -0.136433 | -0.187959 | 1/7 |

No candidate is eligible because advancement requires all seven gates.

## Interpretation

### R2-A: better Intact separation, worse damage severity

R2-A is the only candidate with a positive three-grade macro delta
(`+0.012519`). That gain is driven by Intact F1 (`+0.085979`), while Destroyed
F1 falls by `-0.050839`. Its primary damage metric falls by `-0.024211`, only
4/14 events improve, and Hawaii wildfire has the worst F0-relative grade delta
of `-0.176157`. The apparent grade gain therefore does not satisfy the intended
damage-grading objective.

### R2-B: least harmful, still ineligible

R2-B is the closest candidate to F0 and passes three gates, but damage F1 still
falls by `-0.014519`, Destroyed falls by `-0.025855`, and only 6/14 events are
positive. Nyiragongo is `-0.142176` in damage F1 and `-0.100161` in grade F1
relative to F0. It must not receive the two additional seeds under the frozen
stopping rule.

### R2-C: event balancing does not produce event robustness

R2-C reduces both damage-class F1 values and has the weakest aggregate paired
correspondence among the candidates. Its C2-C3 worst-event grade delta is
`-0.187959` on Myanmar hurricane; Mexico hurricane also has a C2-C3 damage delta
of `-0.235127`. The capped sampler does not stabilize the target behavior.

### Correspondence is informative but not sufficient

F0 itself has a seed-42 C2-C3 damage delta of `+0.077248`, larger than R2-A
(`+0.060541`), R2-B (`+0.062398`), and R2-C (`+0.043235`). The new interventions
preserve some SAR correspondence but do not strengthen it over F0.

The candidates' C2-C3 signal is also concentrated in Destroyed. Their Damaged
C2-C3 deltas are approximately `-0.0003`, `+0.0053`, and `-0.0408`, respectively.
For example, R2-B on Nyiragongo is `+0.281795` versus C3 while remaining
`-0.142176` versus F0. Sample correspondence can therefore be real without the
candidate being a better damage grader.

## Event-level direction

- R2-A improves over F0 on Marshall wildfire, Mexico hurricane, Myanmar
  hurricane, and Noto earthquake.
- R2-B improves on Libya flood, Mexico hurricane, Morocco earthquake, Myanmar
  hurricane, Noto earthquake, and Turkey earthquake 2023.
- R2-C improves on Bata explosion, La Palma volcano, Libya flood, Mexico
  hurricane, Morocco earthquake, and Turkey earthquake 2023.

The direction changes substantially by event, which is why pooled metrics alone
are not accepted and the worst-event gates are binding.

## Claim verdicts

| Claim | Verdict | Basis |
|---|---|---|
| All three RQ2 candidates are technically runnable under the frozen interface. | `ALIGNED` | 42/42 pilot runs and evaluations completed with no queue failures. |
| Paired SAR remains distinguishable from fixed within-event deranged SAR for all three candidates in aggregate. | `ALIGNED` | C2-C3 event-macro damage deltas are +0.060541, +0.062398, and +0.043235. |
| One candidate improves event-robust damage grading over F0. | `NOT_SUPPORTED_BY_EVIDENCE` | Every candidate has a negative F0-relative damage delta and fails multiple frozen gates. |
| RQ2 demonstrates independent unseen-event generalization. | `PROVENANCE_INSUFFICIENT` | All 14 events are exposed development data and no third-party blind evaluation occurred. |

## Statistical and reproducibility limits

- This is a deliberately single-seed pilot. It is sufficient to execute the
  preregistered stop decision, but it is not a multi-seed effect estimate.
- No score-bearing blind set, California/Jamaica reuse, or test-label tuning is
  part of this result.
- The result is `ANALYZED`; independent rerun reproducibility remains
  `CANNOT_VERIFY` even though artifact completeness and provenance are verified.
- The source tree was dirty at launch but was frozen by both tracked-diff and
  source-tree hashes. Reproduction needs those hashes, not only the Git anchor.

## Statistical fallacy scan

Coverage: **11/11**. Aggregate failure agrees with the low positive-event counts,
so no Simpson reversal is observed. Event-macro and per-class reporting avoid
pixel-to-event ecological extrapolation and expose class base rates. The 14-event
development selection remains a Berkson/selection limitation. Collider bias,
regression to the mean, and reverse causality are not applicable to this fixed
comparison. All 42 runs completed, so there is no run-level survivorship filter.
All three prespecified candidates and all frozen gates are reported, limiting
look-elsewhere and garden-of-forking risks. Causal language is restricted to the
controlled development interventions and is not extended to unseen events.

## Final action

Freeze RQ2 Model Development v1 as a negative internal-development result. Do
not run the 28 additional seed jobs, do not train a full-development candidate,
and do not claim unseen-event robustness from this branch.
