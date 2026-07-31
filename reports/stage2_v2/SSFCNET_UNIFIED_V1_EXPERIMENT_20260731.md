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

本文件后续追加 seed 42 正式结果、事件级检查和最终门槛裁决。
