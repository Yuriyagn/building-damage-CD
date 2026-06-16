# Stage-1 Training Repo Experience

This repository is the Git-managed training code for DisasterM3 Stage-1.

Git state:

```text
repo: stage1_optical_building/
branch: main
initial pipeline commit: d7621e5 Add Stage-1 optical building training pipeline
```

Task:

```text
pre-event optical RGB -> binary building mask
```

Server deployment:

```text
server: yr@115.156.98.196
code:   /home/yr/code/stage1_optical_building
data:   /home/yr/code/datasets/DisasterM3_stage1_optical_building
env:    conda activate sam3
gpu:    NVIDIA GeForce RTX 4090 D
```

Run checks on server:

```bash
source /home/yr/miniconda3/etc/profile.d/conda.sh
conda activate sam3
cd /home/yr/code/stage1_optical_building
bash instruction.sh check_env
bash instruction.sh check_data
```

Formal training entrypoint:

```bash
CUDA_VISIBLE_DEVICES=0 bash train.sh o1
CUDA_VISIBLE_DEVICES=0 bash train.sh o2
CUDA_VISIBLE_DEVICES=0 bash train.sh o3
CUDA_VISIBLE_DEVICES=0 bash train.sh o4
```

Early stopping is implemented for fast exploration but disabled in the fixed baseline configs:

```text
train.early_stopping_patience: 0
```

To enable it temporarily on the server:

```bash
CUDA_VISIBLE_DEVICES=0 EARLY_STOPPING_PATIENCE=20 EARLY_STOPPING_MIN_EPOCHS=30 bash train.sh o4
```

`best_iou.pth` still saves the best validation-IoU checkpoint; early stopping only decides when to stop spending time on later epochs.

Completed as of 2026-06-16:

```text
O1 U-Net ResNet34 freq: train + test completed
O2 U-Net ResNet34 all:  train + test completed
O3 DeepLabV3+ ResNet50 freq: train + test completed
O4 SegFormer-B0 freq:    train + test completed
```

Stage-1 baseline report:

```text
reports/stage1_baseline/stage1_baseline_report.md
best test-all baseline: O1 U-Net ResNet34 freq, IoU 0.6871, F1 0.8145
visualizations: reports/stage1_baseline/visualizations/O1_unet_resnet34_freq_test_all/
```

O4 SegFormer-B0 failed initially because SMP tried to download `mit_b0.imagenet` weights while server network was unavailable; this was fixed by caching the weight. Local weight:

```text
pretrained/mit_b0.pth
sha256 df468f7f13c4186f25bd3e2caf09e4f927b5b5ac0abccac84011dae747d4c49c
```

Server cache target:

```text
/home/yr/.cache/torch/hub/checkpoints/mit_b0.pth
```

Keep `configs/stage1_optical_building_segformer_b0_freq.yaml` as:

```yaml
encoder_weights: imagenet
```

unless the user explicitly accepts random initialization.
