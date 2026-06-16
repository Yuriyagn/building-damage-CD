# DisasterM3 Optical-SAR Damage Minimal Package Report

生成日期：2026-06-16
服务器数据包：`/home/yr/code/datasets/DisasterM3_optical_sar_damage_minimal_v0.2`
本次研究主线：`pre-event optical + post-event SAR + 4-class building damage mask + O1 building prior`

## 1. 数据边界

当前最小包只服务“光学-SAR 建筑灾损”第二阶段实验，不再依赖完整 DisasterM3 多任务目录。

保留内容：

- pre-event optical image
- post-event SAR image
- 四分类建筑灾损 mask
- oracle building mask
- Stage-1 O1 building prior
- Stage-2 相对路径 manifest
- QC 与构建报告

不包含内容：

- 完整 `DisasterM3_Bench`
- 完整 `DisasterM3_Instruct`
- 文本任务数据
- 非当前主线图像与 mask
- 原始 zip 文件

完整原始目录已经在服务器上改名为：

```text
/home/yr/code/datasets/DisasterM3_aria2_BACKUP_FULL
```

该目录暂时只作为回滚备份，不建议现在删除。等若干轮 Stage-2 训练、测试都稳定后，再决定删除或转移。

## 2. 样本规模

| split | samples |
| --- | ---: |
| train | 1543 |
| val | 506 |
| test | 564 |
| total | 2613 |

这些样本来自 SAR-QC-clean 主线子集。

## 3. 文件构成

| item | train | val | test | total |
| --- | ---: | ---: | ---: | ---: |
| pre optical + post SAR raw images | 3086 | 1012 | 1128 | 5226 |
| 4-class damage masks | 1543 | 506 | 564 | 2613 |
| oracle building masks | 1543 | 506 | 564 | 2613 |
| O1 `prob_float16_npz` priors | - | - | - | 2613 |
| O1 `prob_uint8` priors | - | - | - | 2613 |
| O1 `binary_tuned` priors | - | - | - | 2613 |

需要文件清单计数：

| list | count | meaning |
| --- | ---: | --- |
| `needed_raw_files.txt` | 5226 | pre optical 与 post SAR 原始图像 |
| `needed_practice_files.txt` | 5236 | 四分类 mask、stage1 binary/oracle mask、QC manifest 与报告 |
| `needed_prior_files.txt` | 7839 | O1 三种 prior 文件 |

包体积：

```text
10G
copied_bytes = 10646919253
```

## 4. 目录结构

```text
DisasterM3_optical_sar_damage_minimal_v0.2/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── masks_4class/
│   ├── train/
│   ├── val/
│   └── test/
├── oracle_building_masks/
│   ├── train/
│   ├── val/
│   └── test/
├── building_priors/
│   └── O1_unet_resnet34_freq/
│       ├── prob_float16_npz/
│       ├── prob_uint8/
│       └── binary_tuned/
├── manifests/
│   ├── stage1_prior_input_{train,val,test}.jsonl
│   ├── stage2_master_{train,val,test}.jsonl
│   ├── no_prior/{train,val,test}.jsonl
│   ├── oracle_prior/{train,val,test}.jsonl
│   └── predicted_prior/{train,val,test}.jsonl
└── reports/
```

Manifest JSONL 文件数：15。

所有 Stage-2 manifest 均使用相对路径。例如：

```json
{
  "pre_image": "images/train/xxx_pre.png",
  "post_sar": "images/train/xxx_sar_post.png",
  "mask_multiclass": "masks_4class/train/xxx.png"
}
```

后续训练只需要设置：

```bash
DATA_ROOT=/home/yr/code/datasets/DisasterM3_optical_sar_damage_minimal_v0.2
```

## 5. 生成方式

生成脚本：

```text
scripts/build_minimal_optical_sar_damage_package.py
```

入口命令：

```bash
bash instruction.sh build_minimal_stage2_package
```

关键输入：

```text
/home/yr/code/datasets/DisasterM3_aria2
/home/yr/code/datasets/DisasterM3_aria2_building_damage_practice
/home/yr/code/datasets/DisasterM3_aria2_building_damage_practice/derived_priors/O1_unet_resnet34_freq
```

构建摘要：

```text
/home/yr/code/datasets/DisasterM3_optical_sar_damage_minimal_v0.2/reports/minimal_package_summary.json
mtime: 2026-06-16 21:28:08 +0800
missing_priors: []
stage1_threshold: 0.6
```

## 6. 校验结果

在校验前，完整原始数据目录已被改名：

```bash
mv /home/yr/code/datasets/DisasterM3_aria2 \
  /home/yr/code/datasets/DisasterM3_aria2_BACKUP_FULL
```

随后执行：

```bash
bash instruction.sh check_stage2_manifests
bash instruction.sh smoke_stage2_minimal_dataloader
bash instruction.sh verify_no_full_disasterm3_dependency
```

结果：

| check | result | detail |
| --- | --- | --- |
| `check_stage2_manifests` | pass | `error_count=0`，master/no_prior/oracle_prior/predicted_prior 的 train/val/test 计数全部匹配 |
| `smoke_stage2_minimal_dataloader` | pass | `error_count=0`，三种 prior setting 的 dataloader 均可读取 |
| `verify_no_full_disasterm3_dependency` | pass | `scripts/`、`src/`、`tools/`、`instruction.sh` 未发现完整 `DisasterM3_Instruct`/`DisasterM3_Bench` 路径依赖 |

校验报告：

```text
/home/yr/code/datasets/DisasterM3_optical_sar_damage_minimal_v0.2/reports/stage2_manifest_check.json
mtime: 2026-06-16 22:05:28 +0800

/home/yr/code/datasets/DisasterM3_optical_sar_damage_minimal_v0.2/reports/stage2_dataloader_smoke.json
mtime: 2026-06-16 21:54:28 +0800
```

## 7. 当前结论

这次收尾已经把工程数据边界收敛为：

```text
任务级最小数据包 + 相对路径 manifest + 单一 DATA_ROOT
```

第二阶段光学-SAR 建筑灾损实验现在不需要完整 `DisasterM3_aria2` 原始目录即可完成 manifest 检查和 dataloader smoke test。完整目录仍保留为 `DisasterM3_aria2_BACKUP_FULL`，等待后续正式训练确认稳定后再清理。
