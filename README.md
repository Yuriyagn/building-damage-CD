# Stage-1 Optical Building Extraction

Task:

```text
pre-event optical RGB -> binary building mask
```

This project expects a self-contained data package at:

```text
/home/yr/code/datasets/DisasterM3_stage1_optical_building
```

Run on the server:

```bash
conda activate sam3
cd /home/yr/code/stage1_optical_building
bash instruction.sh check_env
bash instruction.sh check_data
bash instruction.sh smoke_unet_freq
bash instruction.sh overfit_unet_freq
bash instruction.sh train_unet_freq
bash instruction.sh test_unet_freq
```

For faster exploration after the fixed baselines, enable early stopping without editing configs:

```bash
CUDA_VISIBLE_DEVICES=0 EARLY_STOPPING_PATIENCE=20 EARLY_STOPPING_MIN_EPOCHS=30 bash train.sh o4
```

The fixed baseline configs keep `early_stopping_patience: 0`, so rerunning them without these environment variables preserves the original 100-epoch protocol.

Do not run the training commands on the local machine.

Stage-1 closeout and Stage-2 manifest preparation are documented in:

```text
STAGE1_CLOSEOUT_INSTRUCTIONS.md
```
