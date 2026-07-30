# Stage-2 Baseline Implementation

This repo now implements the Stage-2 diagnostic baseline suite for:

```text
post-event SAR + building prior -> background / intact / damaged / destroyed
```

The goal is not only to train MG-SAR-UNet, but to measure what comes from SAR, what comes from the building prior, and what is explained by class or event bias.

## Data Boundary

Use only:

```bash
STAGE2_DATA_ROOT=/home/yr/code/datasets/DisasterM3_optical_sar_damage_minimal_v0.2
```

Do not train Stage-2 against the full `DisasterM3_aria2` source tree.

## Implemented Experiments

| ID | Config | Input | Purpose |
| --- | --- | --- | --- |
| S0 | `b1_s0_no_prior_unet_resnet34.yaml` | `[SAR, 0, 0]` | no-prior control |
| S1 | `b1_s1_oracle_prior_unet_resnet34.yaml` | `[SAR, oracle prior, SAR*prior]` | upper bound with perfect building mask |
| S2 | `b1_s2_predicted_prior_unet_resnet34.yaml` | `[SAR, O1 prior, SAR*prior]` | full two-stage pipeline |
| S3-oracle | `b1_s3_oracle_prior_only_unet_resnet34.yaml` | `[0, oracle prior, 0]` | checks whether prior alone explains damage labels |
| S3-predicted | `b1_s3_predicted_prior_only_unet_resnet34.yaml` | `[0, O1 prior, 0]` | predicted-prior-only diagnostic |

Rule baselines are also implemented:

| Rule | Meaning |
| --- | --- |
| `R0_all_background` | predicts background everywhere |
| `R1_oracle_all_intact` | oracle building prior -> intact, outside -> background |
| `R2_predicted_all_intact` | predicted binary prior -> intact, outside -> background |
| `*_global_majority_inside_prior` | predicts the train-set majority building class inside prior |
| `*_disaster_majority_inside_prior` | predicts disaster-specific train majority building class inside prior |

## Recommended Execution Order

Run the lightweight checks first:

```bash
bash instruction.sh check_stage2_manifests
bash instruction.sh smoke_stage2_minimal_dataloader
bash instruction.sh stage2_visualize_dataloader
bash instruction.sh stage2_rule_baselines
bash instruction.sh stage2_smoke_s1
```

Then run the long training jobs on the server:

```bash
CUDA_VISIBLE_DEVICES=0 bash instruction.sh stage2_train_s3_oracle
CUDA_VISIBLE_DEVICES=0 bash instruction.sh stage2_train_s3_predicted
CUDA_VISIBLE_DEVICES=0 bash instruction.sh stage2_train_s1
CUDA_VISIBLE_DEVICES=0 bash instruction.sh stage2_train_s2
CUDA_VISIBLE_DEVICES=0 bash instruction.sh stage2_train_s0
```

After training, test all runs:

```bash
CUDA_VISIBLE_DEVICES=0 bash instruction.sh stage2_test_s3_oracle
CUDA_VISIBLE_DEVICES=0 bash instruction.sh stage2_test_s3_predicted
CUDA_VISIBLE_DEVICES=0 bash instruction.sh stage2_test_s1
CUDA_VISIBLE_DEVICES=0 bash instruction.sh stage2_test_s2
CUDA_VISIBLE_DEVICES=0 bash instruction.sh stage2_test_s0
bash instruction.sh stage2_analyze_prior_contribution
```

Outputs go under:

```text
${STAGE2_OUT_ROOT:-outputs/stage2}
```

## Metrics

Each trained and rule baseline reports:

```text
final 4-class metrics:
  background / intact / damaged / destroyed IoU and F1

damage metrics:
  damage_binary_F1 = damaged + destroyed
  damage_macro_F1 = mean(F1_damaged, F1_destroyed)

building-only metrics:
  metrics restricted to GT building pixels for intact / damaged / destroyed
```

Group metrics are written for:

```text
per-disaster
per-region
per-event
```

## Interpretation Rules

| Result pattern | Interpretation |
| --- | --- |
| S1 clearly above S0 | building prior helps |
| S2 close to S1 | O1 Stage-1 prior is good enough |
| S2 far below S1 | Stage-1 prior error is a major bottleneck |
| S1 close to S3-oracle | SAR adds little beyond prior shape / dataset bias |
| S1 close to all-intact | single post SAR is weak for damaged/destroyed discrimination |
| S0 close to S1/S2 | building prior contributes little, or SAR/data bias dominates |
| high mIoU but low damaged/destroyed F1 | background/intact dominate; damage grading failed |

The primary scientific readout should emphasize building-only damaged/destroyed F1 and the S1-S3 gap, not just 4-class mIoU.

