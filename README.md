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

Do not run the training commands on the local machine.
