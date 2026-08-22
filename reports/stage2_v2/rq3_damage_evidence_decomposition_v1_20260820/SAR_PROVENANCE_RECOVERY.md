# RQ3 SAR provenance recovery audit

Date: 2026-08-20
Protocol: `rq3_damage_evidence_decomposition_v1.0`
Verdict: **BRIGHT tile identity and georeferencing are largely recovered; per-sample sensor provenance is not**

## What is now proven

The frozen RQ3 sidecar was compared with all 4,672 official BRIGHT post-event GeoTIFFs under `/home/yr/code/cvprw26/data/BRIGHT/post-event`. The comparison uses two distinct hashes:

- file SHA-256 proves byte-for-byte container identity;
- canonical pixel SHA-256 proves identical decoded uint8 grayscale dimensions and pixels while allowing the TIFF container or tags to differ.

Of 2,214 exposed development samples, 2,189 have a unique exact pixel match, 3 have multiple exact-pixel candidates and remain ambiguous, and 22 are unmatched. Thus 2,192 samples (`99.0063%`) have at least one exact-pixel BRIGHT match, while 2,189 (`98.8708%`) receive a unique official tile and geographic footprint. Only 262 are byte-identical files. The unresolved samples are all 13 Bata samples, 6 Beirut samples and 3 Hawaii samples.

This result is reproducible with `scripts/recover_rq3_bright_georeference.py`. The canonical generated audit is:

`/home/yr/code/Building damage change detection/outputs/stage2/rq3_damage_evidence_decomposition_v1_20260820/bright_georeference_recovery_pixels_v1_20260820/AUDIT.json`

The earlier directory `bright_georeference_recovery_20260820/` is a diagnostic file-SHA-only attempt and is not canonical.

## What remains only event-level context

The [BRIGHT paper](https://essd.copernicus.org/articles/17/6217/2025/) identifies Capella and Umbra as the SAR providers and reports the following event-level source/GSD context. It also describes Spotlight/Stripmap, single-polarization VV/HH, provider or author 8-bit scaling, and manual optical–SAR registration. These statements do not identify the collect used for an individual tile.

| Local canonical event | Published SAR provider | Published SAR GSD (m) |
|---|---:|---:|
| `bata_explosion` | Capella | 0.50 |
| `beirut_explosion_2020` | Capella | 1.00 |
| `nyiragongo_2021` | Capella | 0.33 |
| `haiti_earthquake` | Capella | 0.48 |
| `la_palma_volcano` | Capella | 0.30–0.35 |
| `marshall_wildfire` (Boulder/Marshall) | Capella | 0.60 |
| `ukraine_conflict` | Capella | 0.60 |
| `turkey_eq_2023` | Capella and Umbra | 0.30–0.35 |
| `myanmar_hurricane` | Capella | 0.60 |
| `hawaii_wildfire` | Capella | 0.60 |
| `morocco_earthquake` | Capella | 0.35–0.40 |
| `libya_flood` | Capella | 0.35 |
| `mexico_hurricane` | Capella | 0.35–0.80 |
| `noto_earthquake` | Umbra | 0.50 |

The official [BRIGHT repository](https://github.com/ChenHongruixuan/BRIGHT) and [Zenodo release](https://zenodo.org/records/20072020) expose image/label archives but no frozen per-tile STAC or extended-metadata manifest. Consequently, event-level provider/GSD values remain `published_event_context`, not `stac_exact` sample metadata.

## Official catalog recovery

[Capella's STAC documentation](https://docs.capellaspace.com/accessing-data/searching-for-data/) confirms that collect metadata can include platform, mode, polarization, time, incidence angle, orbit and product information. Searches of the public catalog found plausible date/footprint candidates for Haiti (3), Hawaii (3), Libya (1), Marshall (2), Turkey (2) and La Palma (7). Multiple collects or GEO/GEC products remain possible; therefore none is attached to a sample as high-confidence metadata. The catalog search did not establish a candidate for Bata, Beirut, Mexico, Morocco, Myanmar, Nyiragongo or Ukraine. A cold `Other` catalog object timed out during read, so this is not evidence that those collections are absent.

The [Umbra open-data catalog](https://umbra.space/open-data/) and [metadata specification](https://docs.canopy.umbra.space/docs/collect-metadata) expose four official Noto candidates whose footprints jointly cover all 79 BRIGHT Noto tiles. Coverage multiplicity is 33 tiles with one candidate, 12 with two, 25 with three and 9 with four. The four source URLs, task/collect IDs, acquisition fields and source SHA-256 values are frozen in:

`/home/yr/code/Building damage change detection/outputs/stage2/rq3_damage_evidence_decomposition_v1_20260820/noto_umbra_candidates_20260820/CANDIDATE_AUDIT.json`

A remote COG pixel-disambiguation probe produced a read error and then stalled. The exact probe process was terminated; no files were changed and no confidence level was promoted. Geometry alone therefore remains a candidate relation, not a per-tile source mapping.

## Gate consequence

The georeference recovery materially enables geographic overlap auditing and future catalog matching, but it does **not** satisfy E3's admission rule. High-confidence per-sample collect metadata remains `0 / 2214`, so the required overall `≥90%` and per-event `≥80%` coverage is still unmet. It would be invalid to fill mode, polarization, acquisition time or incidence angle from event-level averages and label that operation physical/radiometric calibration.

E1 is independently ready at the technical level and is blocked only by the mandatory third-party blind-evaluator lock. E3 additionally requires an original tile-to-collect manifest or deterministic local source-product matching with unambiguous results.
