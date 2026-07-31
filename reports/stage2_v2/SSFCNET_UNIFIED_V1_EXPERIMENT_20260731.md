# SSFCNet 统一标准迁移实验

日期：2026-07-31
协议：`stage2_ssfcnet_unified_v1`

论文：*Fine-grained heterogeneous change detection in complex disaster
response with wavelet-based spatial-frequency coupled learning*

DOI：<https://doi.org/10.1016/j.isprsjprs.2026.06.001>

作者仓库：<https://github.com/yangyang12318/SSFCNet>

## 1. 可复现性边界

作者仓库当前 `master` 为
`a18709098736dfe3b1984bd7c51bd7da9afeb624`。该提交删除了
`model/SSFCNet.py`，当前树也没有训练入口、测试入口、requirements、已训练
checkpoint 或正式 LICENSE 文件。模型源码仍存在于历史提交：

```text
commit:
284d0bf608eb4302cd47e3959e36deb4e300cf6f

model/SSFCNet.py:
8d6230e545257def938072b0ce6bf2002d3b315a57c86b7b9fccc689b7562e55

utils/loss.py:
450ffe3408493a3c77405a9f629bb4b3b31f2a014b9f631ca9fb730b7b4787b1
```

历史源码固定在主项目外的隔离 worktree：

```text
../ssfcnet_reproduction_20260731/historical_284d0bf
```

主仓库只保留动态加载适配器，不复制作者源码。由于缺少完整训练工程和权重，
本实验定义为 **作者架构迁移**，不能写成论文分数或作者训练配方的精确复现。

## 2. 迁移内容

已验证历史模型包含并能执行：

- 共享权重的双流 Siamese 编码；
- 基于小波分解的空间/频率特征编码；
- 方向高频增强；
- 显式差分与相关性投影；
- 三维跨通道融合；
- 逆小波解码。

三分类模型参数量为 `31,593,363`，已验证随机输入可产生有限的
`[B, 3, H, W]` logits。

统一六通道输入映射为：

```text
stream A = [pre_R, pre_G, pre_B]
stream B = [SAR, predicted_building_prior, SAR * prior]
```

两流都按作者残存 dataloader 的规则从 `[0,1]` 归一化到 `[-1,1]`。

作者仓库中的 Boundary-Aware Loss 是二分类边界损失，而本项目主任务是建筑区
三等级分类。为单独检验架构迁移是否有效，seed 42 主门槛先冻结本项目 E2 的
三等级 loss。BAL 只允许在架构主门槛通过后作为单独消融，禁止把架构、loss
和评价协议同时改变。

## 3. 冻结实验标准

- manifest：
  `manifests/stage2_v2_clean_human_reviewed_20260624/predicted_prior/`
- train / val / test：`1207 / 357 / 388`
- 模型开发和选择只使用 train/val；test 封存
- crop：`512`
- seeds：`42 / 3407 / 2026`
- AdamW，LR / weight decay：`1e-4 / 1e-4`
- cosine scheduler
- micro-batch：`8`
- gradient accumulation：`1`
- effective batch：`8`
- 最多 `40` epochs
- early stop：min `12`，patience `10`，window `3`
- loss：building-only weighted CE + binary auxiliary `0.3`
- 主指标：validation `building_only_macro_f1_3class`
- 每个正式 run 只保留 `best_bo_grade_macro_f1.pth`

micro-batch 8 已在 RTX 4090 D 48 GB 上通过 512×512 训练和验证探测，峰值
进程显存约 `28.8 GiB`。paired 与 shuffled 除实验名和 SAR 配对方式外保持
一致。

## 4. 负对照与晋级门槛

正式 seed 42 必须并行运行：

- `S2SF1_SSFCNet_paired`
- `S2SF1_SSFCNet_shuffled`

shuffled 使用固定 seed `20260731` 的事件内 derangement，必须保存完整置换、
保证无 self-pair。只有下面两项同时通过才扩展到 seeds 3407/2026：

1. paired BO macro F1 `>= 0.3586137203`（E2 argmax reference）
2. paired - shuffled BO macro F1 `>= 0.01`

门槛失败则停止 SSFCNet 分支，不继续调参，不运行 test。

## 5. 技术门禁

| 检查 | 结果 |
| --- | --- |
| 主线单测 | pass |
| Python / shell 语法与协议校验 | pass |
| 严格数据审计 | pass，0 hard error |
| exact-content duplicates | 0 |
| paired 512×512 smoke | pass |
| shuffled 512×512 smoke | pass |
| 固定两 batch overfit | pass |
| overfit 首轮 / 最低 loss | 1.107359 / 0.589771 |
| overfit 最佳相对下降 | 46.74%，门槛 30% |
| batch-8 train / eval 探测 | pass |

技术门禁输出：

```text
outputs/stage2/ssfcnet_unified_v1_overfit/
  paired_seed42_2batch_20260731_r1/overfit_gate.json
```

## 6. 可执行入口

```bash
bash scripts/launch_stage2_ssfcnet_unified_v1.sh paired 42 0
bash scripts/launch_stage2_ssfcnet_unified_v1.sh shuffled 42 1
```

正式训练完成后，只在全量 val 上独立复核各自最佳 checkpoint，再运行：

```bash
python scripts/evaluate_stage2_unified_v1_gate.py \
  --protocol configs/stage2_ssfcnet_unified_v1/protocol.yaml \
  --paired-run <paired-run> \
  --shuffled-run <shuffled-run> \
  --out outputs/stage2/ssfcnet_unified_v1/gates/seed42_gate.json
```

## 7. 正式训练与复核

正式训练固定在主仓库提交：

```text
6153d6145fc253c4a4113f654678a49284907a0f
source_tree_sha256:
aec3cfa9986de3e11a09107f6e27d85fc1148c781417e46ea8c9cfd867cda416
```

两个 run 启动时工作树均无差异，实际 `run_info.json` 均记录了本次严格数据
审计路径、SHA-256、0 hard error 和 3 个 `qc_label=unknown` 元数据 warning。

shuffled 的完整置换审计结果：

| split | records | self-pair | cross-event | singleton |
| --- | ---: | ---: | ---: | ---: |
| train | 1207 | 0 | 0 | 0 |
| val | 357 | 0 | 0 | 0 |

训练完成情况：

| 分支 | 完成 epochs | 结束方式 | 最佳 epoch | 训练内最佳 BO macro F1 |
| --- | ---: | --- | ---: | ---: |
| paired | 24 | early stop | 6 | 0.329494 |
| shuffled | 28 | early stop | 6 | 0.340489 |

随后分别加载唯一最佳 checkpoint，在全部 357 个 val 样本上独立复核。独立
复核值与训练内最佳值完全一致，且生成了逐样本、per-event、per-disaster 和
per-region 标量统计；未生成 predictions 或 previews。

## 8. seed 42 全量 val 结果

| 方法 | BO 3-grade macro F1 | BO damage macro F1 | BO binary damage F1 | intact F1 | damaged F1 | destroyed F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E2 argmax reference | **0.358614** | **0.112705** | **0.199500** | 0.850430 | **0.126116** | 0.099295 |
| SSFCNet paired | 0.329494 | 0.056961 | 0.125865 | **0.874558** | 0.020642 | 0.093281 |
| SSFCNet shuffled | 0.340489 | 0.081564 | 0.150487 | 0.858340 | 0.056028 | **0.107099** |

统一 predicted-building gate 下的端到端指标：

| 方法 | 4-class macro F1 | damage macro F1 | binary damage F1 |
| --- | ---: | ---: | ---: |
| E2 argmax reference | **0.469296** | **0.092199** | **0.161159** |
| SSFCNet paired | 0.451849 | 0.050837 | 0.107752 |
| SSFCNet shuffled | 0.459267 | 0.072440 | 0.126847 |

核心差值：

- paired 比 E2 主指标低 `0.029120`
- paired 比 shuffled 主指标低 `0.010996`
- paired 的 intact F1 比 E2 高 `0.024128`，但 damaged F1 低 `0.105474`
- paired 的 damage macro / binary damage F1 分别比 E2 低
  `0.055744 / 0.073635`
- shuffled 在主指标、两类损伤 F1、damage macro、binary damage 和
  predicted-gate 指标上均高于 paired

因此 paired 的分数主要由 intact 支撑。训练中偶尔会在 damaged 或 destroyed
其中一类上升，但另一类同步退化，无法形成稳定三等级分离。真实配对 SAR 并未
优于事件内打乱 SAR，不能声称模型学到了可靠的样本级跨模态对应关系。

## 9. 事件级检查

| val event | paired BO macro | shuffled BO macro | paired-shuffled | E2 BO macro |
| --- | ---: | ---: | ---: | ---: |
| bata_explosion | 0.267386 | 0.338049 | -0.070663 | 0.361908 |
| haiti_earthquake | 0.320961 | 0.330365 | -0.009404 | 0.308206 |
| marshall_wildfire | 0.313855 | 0.277669 | +0.036187 | 0.339084 |
| morocco_earthquake | 0.333127 | 0.316656 | +0.016472 | 0.329740 |
| turkey_earthquake4 | 0.310565 | 0.325926 | -0.015361 | 0.349356 |

paired 只在 2/5 个事件上优于 shuffled，在 bata_explosion 上落后
`0.070663`。它在 Haiti 和 Morocco 上略高于 E2，但在其余 3 个事件上低于
E2，且没有形成稳定的跨事件 paired 优势。

## 10. 预注册门槛裁决

| 检查 | 观测值 | 门槛 | 结果 |
| --- | ---: | ---: | --- |
| paired >= E2 | 0.329494 | 0.358614 | fail |
| paired - shuffled | -0.010996 | +0.010000 | fail |

自动门槛：

```text
outputs/stage2/ssfcnet_unified_v1/gates/seed42_gate.json
status: fail
decision: stop_after_seed42_and_record_as_rejected_transfer
```

两项要求必须同时通过，本次两项均失败。因此：

- 不运行 seeds 3407/2026
- 不运行 test
- 不追加 Boundary-Aware Loss 或其他调参分支
- 将本次结论记录为作者架构在本项目统一标准下的 rejected transfer

这不否定论文在其原始数据和完整未公开训练配方下的结论；它只说明可获得的历史
SSFCNet 架构，在本项目严格数据、三等级目标和 paired/shuffled 负对照下没有
超过当前 E2，也没有证明真实 SAR 配对增益。

## 11. 正式产物

Paired：

```text
outputs/stage2/ssfcnet_unified_v1/S2SF1_SSFCNet_paired/
  seed_42/run_20260731_103052/
```

Shuffled：

```text
outputs/stage2/ssfcnet_unified_v1/S2SF1_SSFCNet_shuffled/
  seed_42/run_20260731_103113/
```

每个正式 run 只保留：

- `checkpoints/best_bo_grade_macro_f1.pth`
- `completed.json`、`run_info.json`、resolved config
- 完整置换记录与 class weights
- 训练标量历史
- `val_best_grade/metrics.json`
- per-event / per-disaster / per-region / per-sample 标量 CSV

共同协议、审计和门槛证据位于：

```text
configs/stage2_ssfcnet_unified_v1/
outputs/stage2/ssfcnet_unified_v1_preflight/
outputs/stage2/ssfcnet_unified_v1/gates/
```

## 12. 工作区收尾

结果冻结后执行保守清理：

- 删除两个 batch-probe / overfit 诊断 checkpoint
- 删除主仓库新生成的 Python/test 缓存
- 删除 SSFCNet 外部 worktree 中本次生成的 Python 3.12 bytecode
- 两次主仓库清理共删除约 `0.708 GiB`、`184` 个文件；另删除外部
  worktree 中本次生成的 2 个 bytecode 文件
- 正式 paired / shuffled 最佳 checkpoint 均通过白名单复核并保留
- 外部当前 checkout 与历史 detached worktree 均恢复 clean

清理证据：

```text
reports/workspace_hygiene/
  workspace_cleanup_ssfcnet_20260731_dry_run.json
  workspace_cleanup_ssfcnet_20260731_applied.json
  workspace_cleanup_ssfcnet_20260731_final_applied.json
```
