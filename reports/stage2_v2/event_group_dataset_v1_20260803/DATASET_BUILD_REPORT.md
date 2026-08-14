# Stage-2 event-group v1 dataset build

> **Status update (2026-08-14):** this file freezes the state immediately after dataset
> construction. Frozen R4 priors and all six R0/R4 validation runs were subsequently
> completed. Use
> [R0_R4_PREDICTED_PRIOR_COMPARISON_20260804.md](R0_R4_PREDICTED_PRIOR_COMPARISON_20260804.md)
> for the completed comparison and
> [the project audit](../../../docs/PROJECT_AUDIT_20260814.md) for current limitations.

Status: the event-identity rebuild, BRIGHT conversion, and image-overlap audits are complete. No training was started. R0 is ready only as a historically observed diagnostic baseline; R4 remains blocked until frozen O1 predicted priors exist for all supplement samples.

## Frozen historical baseline

- Historical directory split: `manifests/stage2_v2_clean_human_reviewed_20260624`
- Historical label: `strict_v1_directory_split`
- Event mapping: `configs/stage2_v2/event_group_mapping_v1.csv`
- Event-group split plan: `configs/stage2_v2/event_group_split_v1.json`
- The historical manifests, source DisasterM3 package, and external BRIGHT data root were read only and were not modified.

Every rebuilt row retains `source_dataset`, `source_event_name`, `canonical_event_name`, `event_group_id`, `subevent_id`, `tile_id`, and `geographic_footprint`. The former split is retained as `directory_split_v1` for traceability and is not used for the new assignment.

## Canonical real-event split

| split | samples | event groups |
|---|---:|---|
| train | 1,372 | `beirut_explosion_2020`, `hawaii_wildfire`, `la_palma_volcano`, `libya_flood`, `myanmar_hurricane`, `turkey_eq_2023`, `ukraine_conflict` |
| val | 290 | `bata_explosion`, `haiti_earthquake`, `marshall_wildfire`, `morocco_earthquake` |
| test | 290 | `mexico_hurricane`, `noto_earthquake`, `nyiragongo_2021` |

Turkey earthquake 1/3/4/5 are one `turkey_eq_2023` train group. Current `rwanda_volcano` and BRIGHT `congo-volcano` are one `nyiragongo_2021` group. Current `spain_volcano` and `la_palma_volcano` are one `la_palma_volcano` group. No `event_group_id` crosses a split.

The rebuilt test contains events inspected in earlier work. It is a diagnostic event-group-held-out test, not pristine unseen-disaster evidence.

## BRIGHT_supplement_v1

- Eligible source events: Beirut, Hawaii, and Libya only.
- Initial eligible candidates: 281.
- Removed: 8 decoded-identical matches to the current strict union and 11 samples with empty building annotations.
- Final supplement: **262** samples: Beirut 116, Hawaii 56, Libya 90.
- Derived output: 262 four-class semantic masks and 262 oracle building masks (524 files, 8.7 MiB).
- Combined R4-data split: **1,634 / 290 / 290** train/val/test.
- No La Palma or other disallowed BRIGHT event was added.

The semantic conversion was checked on 1,121 decoded-identical current/BRIGHT pairs. Building IoU was 0.9767; class IoU was 0.9749 intact, 0.9766 damaged, and 0.9801 destroyed. The shared-foreground class-conflict rate was 0.0469%, supporting the COCO-to-semantic label mapping.

## Event x damage-class matrix

The following is the train split, in labeled pixels. All 21 real-event x damage-class cells are nonempty in both versions.

### R0: current data after real-event splitting

| event_group_id | images | intact | damaged | destroyed |
|---|---:|---:|---:|---:|
| `beirut_explosion_2020` | 6 | 348,187 | 735,113 | 402,037 |
| `hawaii_wildfire` | 3 | 75,815 | 29,678 | 423,199 |
| `la_palma_volcano` | 667 | 58,808,269 | 258,020 | 21,134,234 |
| `libya_flood` | 8 | 1,183,930 | 992,398 | 106,141 |
| `myanmar_hurricane` | 78 | 6,256,667 | 612,029 | 43,430 |
| `turkey_eq_2023` | 184 | 58,092,503 | 8,586,193 | 5,990,453 |
| `ukraine_conflict` | 426 | 44,930,468 | 21,226,596 | 713,969 |

### R4-data: R0 plus final BRIGHT supplement

| event_group_id | images | intact | damaged | destroyed |
|---|---:|---:|---:|---:|
| `beirut_explosion_2020` | 122 | 37,002,845 | 2,173,587 | 640,923 |
| `hawaii_wildfire` | 59 | 2,401,340 | 825,094 | 4,587,687 |
| `la_palma_volcano` | 667 | 58,808,269 | 258,020 | 21,134,234 |
| `libya_flood` | 98 | 14,468,575 | 6,382,659 | 981,792 |
| `myanmar_hurricane` | 78 | 6,256,667 | 612,029 | 43,430 |
| `turkey_eq_2023` | 184 | 58,092,503 | 8,586,193 | 5,990,453 |
| `ukraine_conflict` | 426 | 44,930,468 | 21,226,596 | 713,969 |

| class | R0 top event/share | R0 HHI | R0 effective events | R4-data top event/share | R4-data HHI | R4-data effective events |
|---|---|---:|---:|---|---:|---:|
| intact | La Palma / 34.66% | 0.3088 | 3.24 | La Palma / 26.49% | 0.2126 | 4.70 |
| damaged | Ukraine / 65.43% | 0.5001 | 2.00 | Ukraine / 52.98% | 0.3557 | 2.81 |
| destroyed | La Palma / 73.35% | 0.5823 | 1.72 | La Palma / 61.99% | 0.4349 | 2.30 |

The split correction itself materially improves the old directory-based concentration estimate because the complete Turkey event now belongs to train. BRIGHT then further reduces concentration, but adds no new real event group. It is therefore an event-internal class-coverage repair set, not a cross-event expansion set.

## Integrity and overlap gates

| gate | result | evidence |
|---|---|---|
| canonical event disjointness | pass | zero train/val/test `event_id` or `event_group_id` overlap |
| complete-file SHA-256 audit | pass | zero duplicate full-sample groups in both R0 and R4 manifests |
| decoded exact image-pair audit | pass | all eligible exact matches excluded from the supplement |
| semantic masks | pass | values restricted to 0/1/2/3; all 262 masks nonempty |
| near-duplicate image audit | pass after human review | 0 high-risk pairs; 3 medium candidates; all 3 were different scenes and retained |
| projected footprint audit | partial pass | supplement coverage 100%; current val/test coverage 350/580 (60.34%); zero overlap among known comparable footprints |
| manifest audit | pass with warnings | R0 and R4 each have 0 hard errors; warnings are real class-distribution shift and missing historical `qc_label` values |
| real dataloader smoke | pass | `Stage2V2Dataset` loaded BRIGHT and current samples from train/val/test with no-prior and oracle-prior, yielding 6x1024x1024 inputs and valid masks |

The footprint limitation remains substantive: 230 current val/test PNG rows have no recoverable source georeferencing. The all-image near-duplicate audit reduces the practical leakage risk but does not prove formal geospatial-footprint independence for those rows.

## Verification

- `py_compile` passed for the builder and its focused tests.
- All 4 event-group/BRIGHT conversion unit tests passed.
- All 28 existing Stage-2 v2 core tests passed.
- A final manifest invariant check parsed every generated JSON/JSONL row, found all seven identity fields on every sample, and confirmed 14 event groups with zero cross-split assignment in both R0 and R4-data.

## Readiness decision

- **R0 data gate:** pass for diagnostic training/evaluation only; no run was started.
- **R4 data gate:** blocked because frozen O1 predicted building priors have not been generated for the final 262 BRIGHT samples.
- **Formal unseen-event claim:** blocked because the rebuilt test was historically observed.
- **Formal geospatial-independence claim:** not proven because footprint coverage is incomplete.

Before any R4 run, generate the 262 frozen predicted priors without val/test selection, build and audit the `predicted_prior` manifests, and retain the test-status limitation. A genuinely final unseen-event evaluation requires newly held-out real disasters.
