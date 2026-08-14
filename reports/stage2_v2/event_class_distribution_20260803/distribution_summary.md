# Strict-v1 event × damage-class distribution

The tables use building pixels only. `within-event composition` answers what each event contains; `contribution to split class` answers how strongly each class is tied to a particular event.

## Train

### Within-event composition

| event | images | intact | damaged | destroyed |
|---|---:|---:|---:|---:|
| beriut_explosion | 6 | 23.44% | 49.49% | 27.07% |
| hawaii_wildfire | 3 | 14.34% | 5.61% | 80.05% |
| libya_flood | 8 | 51.87% | 43.48% | 4.65% |
| myanmar_hurricane | 78 | 90.52% | 8.85% | 0.63% |
| spain_la_palma_volcano | 667 | 73.33% | 0.32% | 26.35% |
| turkey_earthquake5 | 19 | 95.28% | 1.79% | 2.92% |
| ukraine_conflict | 426 | 67.19% | 31.74% | 1.07% |

### Contribution to split class

| event | intact contribution | damaged contribution | destroyed contribution |
|---|---:|---:|---:|
| beriut_explosion | 0.29% | 3.07% | 1.75% |
| hawaii_wildfire | 0.06% | 0.12% | 1.84% |
| libya_flood | 1.00% | 4.14% | 0.46% |
| myanmar_hurricane | 5.28% | 2.55% | 0.19% |
| spain_la_palma_volcano | 49.67% | 1.08% | 91.76% |
| turkey_earthquake5 | 5.73% | 0.53% | 0.90% |
| ukraine_conflict | 37.95% | 88.51% | 3.10% |

### Class concentration

| class | top event | top-1 share | HHI | effective events |
|---|---|---:|---:|---:|
| intact | spain_la_palma_volcano | 49.67% | 0.3970 | 2.52 |
| damaged | ukraine_conflict | 88.51% | 0.7869 | 1.27 |
| destroyed | spain_la_palma_volcano | 91.76% | 0.8438 | 1.19 |

## Val

### Within-event composition

| event | images | intact | damaged | destroyed |
|---|---:|---:|---:|---:|
| bata_explosion | 13 | 60.03% | 22.74% | 17.24% |
| haiti_earthquake | 67 | 98.38% | 1.48% | 0.14% |
| marshall_wildfire | 67 | 93.09% | 0.69% | 6.22% |
| morocco_earthquake | 143 | 99.09% | 0.83% | 0.08% |
| turkey_earthquake4 | 67 | 69.65% | 18.79% | 11.56% |

### Contribution to split class

| event | intact contribution | damaged contribution | destroyed contribution |
|---|---:|---:|---:|
| bata_explosion | 1.73% | 4.99% | 5.76% |
| haiti_earthquake | 22.67% | 2.61% | 0.36% |
| marshall_wildfire | 11.56% | 0.65% | 8.95% |
| morocco_earthquake | 19.99% | 1.28% | 0.19% |
| turkey_earthquake4 | 44.04% | 90.47% | 84.73% |

### Class concentration

| class | top event | top-1 share | HHI | effective events |
|---|---|---:|---:|---:|
| intact | turkey_earthquake4 | 44.04% | 0.2990 | 3.34 |
| damaged | turkey_earthquake4 | 90.47% | 0.8219 | 1.22 |
| destroyed | turkey_earthquake4 | 84.73% | 0.7293 | 1.37 |

## Test

### Within-event composition

| event | images | intact | damaged | destroyed |
|---|---:|---:|---:|---:|
| mexico_hurricane | 172 | 14.23% | 78.71% | 7.06% |
| noto_earthquake | 26 | 83.96% | 4.74% | 11.29% |
| rwanda_volcano | 92 | 88.75% | 0.65% | 10.61% |
| turkey_earthquake1 | 72 | 87.15% | 6.20% | 6.65% |
| turkey_earthquake3 | 26 | 78.67% | 13.93% | 7.40% |

### Contribution to split class

| event | intact contribution | damaged contribution | destroyed contribution |
|---|---:|---:|---:|
| mexico_hurricane | 7.97% | 85.09% | 31.75% |
| noto_earthquake | 7.48% | 0.81% | 8.07% |
| rwanda_volcano | 15.42% | 0.22% | 14.79% |
| turkey_earthquake1 | 47.64% | 6.54% | 29.17% |
| turkey_earthquake3 | 21.49% | 7.35% | 16.22% |

### Class concentration

| class | top event | top-1 share | HHI | effective events |
|---|---|---:|---:|---:|
| intact | turkey_earthquake1 | 47.64% | 0.3089 | 3.24 |
| damaged | mexico_hurricane | 85.09% | 0.7337 | 1.36 |
| destroyed | mexico_hurricane | 31.75% | 0.2406 | 4.16 |
