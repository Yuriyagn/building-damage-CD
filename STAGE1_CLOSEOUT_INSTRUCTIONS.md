# Stage-1 Closeout Instructions

Goal:

```text
Freeze O1 -> tune official threshold on val -> build a task-level minimal Stage-2 package
-> check Stage-2 manifests -> run a dataloader smoke test
```

Server paths:

```text
code:          /home/yr/code/stage1_optical_building
stage1 data:   /home/yr/code/datasets/DisasterM3_stage1_optical_building
practice root: /home/yr/code/datasets/DisasterM3_aria2_building_damage_practice
minimal root:  /home/yr/code/datasets/DisasterM3_optical_sar_damage_minimal_v0.2
```

The Stage-2 mainline should use only the minimal root. Full source data is needed only when
constructing or rebuilding that package from the SAR-QC manifests.

Run on the server:

```bash
source /home/yr/miniconda3/etc/profile.d/conda.sh
conda activate sam3
cd /home/yr/code/stage1_optical_building

bash instruction.sh freeze_stage1_o1
bash instruction.sh tune_stage1_o1_threshold

# One-time source-root export, only if the source O1 prior directory does not exist yet.
bash instruction.sh export_stage1_o1_priors_source

bash instruction.sh build_minimal_stage2_package
bash instruction.sh check_stage2_manifests
bash instruction.sh smoke_stage2_minimal_dataloader
bash instruction.sh verify_no_full_disasterm3_dependency
```

Expected outputs:

```text
/home/yr/code/datasets/DisasterM3_aria2_building_damage_practice/
└── stage1_official/O1_unet_resnet34_freq/

/home/yr/code/datasets/DisasterM3_optical_sar_damage_minimal_v0.2/
├── images/{train,val,test}/
├── masks_4class/{train,val,test}/
├── oracle_building_masks/{train,val,test}/
├── building_priors/O1_unet_resnet34_freq/
├── manifests/
└── reports/
```

All Stage-2 manifests under `minimal root/manifests/` use paths relative to the minimal
package root. The prior export is inference-only, not training. It processes 1543 train,
506 val, and 564 test samples with the frozen O1 checkpoint.
