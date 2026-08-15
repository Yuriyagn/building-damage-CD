# ARS 实验方法学审查：跨事件 paired SAR 可识别性

> 审查日期：2026-08-15
> 使用方法：ARS-Codex `v0.1.25`，按 `deep-research/lit-review`、
> `deep-research/three-way-scan`、`deep-research/fact-check`、
> `academic-paper-reviewer/methodology-focus` 和 `experiment-agent/plan` 执行。
> 本文件只冻结研究问题、证据边界与实验协议；没有启动新训练，也没有读取 test。

## 1. 审查结论

拟议研究问题有价值，但当前版本不能直接进入正式训练。审查判定为：

**Major Revision / protocol blocked until P0 prerequisites are closed。**

最关键的三项前置条件是：

1. 为所有 development 样本生成真正 event-aware OOF/frozen Stage-1 prior；
2. 将 DisasterM3、BRIGHT 和历史派生样本按 canonical event、原始图像内容和地理 scene 去重，避免把同一事件或派生图像当作独立证据；
3. 冻结新的外部事件盲测机制。现有 legacy、clean strict 和 event-group test 都已暴露，只能作 `historical_diagnostic_only`。

在这些条件关闭前，可以实现数据审计、permutation 生成器、registry 和 CPU/CUDA smoke，但不应启动能被写成正式结论的 C0-C3 队列。

## 2. 精炼后的研究问题与 estimand

### RQ1

> 在 canonical-event-disjoint 的条件下，样本配对的 post-event SAR 是否包含超出 event-aware OOF building prior 与 pre-event optical RGB 的、可迁移到未见灾害事件的损伤等级信息？

### 主要 estimand

主要效应不是“某个 SAR 模型的绝对分数”，而是同 seed、同 outer fold 下：

```text
Delta_pair = event_macro_damage_F1(C2 paired SAR)
           - event_macro_damage_F1(C3 within-event deranged SAR)
```

它检验在尽可能保留事件、传感器和 SAR 边际分布时，逐样本对应关系是否提供信息。

次要效应：

```text
Delta_add = event_macro_damage_F1(C2 paired SAR)
          - event_macro_damage_F1(C1 pre + OOF prior)
```

它检验 SAR 是否在 pre optical 与建筑先验之外提供增量价值。C0 prior-only 是 shortcut 诊断，不是主要候选。

### 明确不研究

- 不在本轮加入 disaster/location/sensor metadata 输入或辅助头；
- 不更换 foundation encoder，不引入 LoRA/adapter；
- 不比较新架构；
- 不使用当前任何历史 test 做 checkpoint、阈值、超参数或模型选择；
- 不把 tile 数量当作独立事件数量。

## 3. 文献检索记录（lit-review）

### 检索策略

- 最后检索日期：2026-08-15；
- 来源：论文正式页面、出版社页面、CVF/PMLR/NeurIPS、作者官方代码仓库和 arXiv 原文；
- 关键词族：`building damage assessment cross-event`、`BRIGHT optical SAR transfer`、
  `DisasterM3 multi-task multi-sensor`、`xBD disaster metadata`、
  `transferable building damage assessment`、`multitask gradient conflict`；
- 纳入：直接定义数据、split、任务、迁移协议或多任务优化问题的一手资料；
- 排除：只有二手摘要、无法确认存在、没有足够方法描述或与遥感灾损无关的材料。

这不是 PRISMA systematic review；它是面向当前协议冻结的定向 narrative review。搜索覆盖和遗漏风险必须保留。

### 证据矩阵

| 来源 | 直接支持什么 | 不能支持什么 | 证据状态 |
|---|---|---|---|
| [xBD, CVPRW 2019](https://openaccess.thecvf.com/content_CVPRW_2019/html/cv4gc/Gupta_Creating_xBD_A_Dataset_for_Assessing_Building_Damage_from_Satellite_CVPRW_2019_paper.html) | 多事件 pre/post optical、building polygon、ordinal damage 与 satellite metadata 是 BDA 的基础资源 | 不能证明 metadata 输入或辅助监督提升 cross-event 泛化 | VERIFIED_PRIMARY |
| [Bouchard et al., 2022](https://doi.org/10.3390/rs14112532) | 灾害事件间图像与损伤机制不同；跨事件迁移是部署问题 | 不能为本项目的具体 paired-shuffle estimand 提供阈值 | VERIFIED_PRIMARY |
| [BRIGHT, ESSD 2025](https://doi.org/10.5194/essd-17-6217-2025) | pre optical + post SAR；明确区分 standard split 与 zero/one-shot cross-event transfer | standard split 不是 event-disjoint；不能把它误称为未见事件验证 | VERIFIED_PRIMARY |
| [BRIGHT 官方代码](https://github.com/ChenHongruixuan/BRIGHT) | 发布 standard ML、cross-event transfer、UDA/SSL/UMCD/UMIM 设置 | 仓库协议不自动证明本项目数据适配正确 | VERIFIED_AUTHOR_SOURCE |
| [DisasterM3](https://arxiv.org/abs/2505.21089) | 36 events、10 hazards、多传感器、9 类感知/推理任务；灾种表现存在差异 | 它是 VLM benchmark；不能证明 7 类灾种 CE 辅助头改善本项目 U-Net | VERIFIED_PREPRINT |
| [Zheng et al., RSE 2024](https://doi.org/10.1016/j.rse.2024.114416) | 未见事件/区域的 domain shift 是 BDA 的核心问题；提出 target adaptation | 需要 target-domain 数据的 adaptation 与纯 zero-shot generalization 不是同一 estimand | VERIFIED_PRIMARY |
| [Harrison et al., IJDRR 2026](https://doi.org/10.1016/j.ijdrr.2026.106048) | 11 个事件的 event-based CV 显示 unseen-event macro F1 下降且事件差异很大 | 中分辨率 Sentinel-1 + ML 结果不能直接外推到本项目 VHR 像素分割 | VERIFIED_PRIMARY |
| [Singh et al., 2025](https://doi.org/10.1177/87552930251377778) | 地震单灾种场景中，ground motion/soil/SAR proxy 等 metadata 可与图像融合 | 直接输入 metadata 的单灾种区域迁移不等于灾种辅助监督的跨灾种收益 | VERIFIED_PRIMARY_LIMITED_TRANSFER |
| [GradNorm](https://proceedings.mlr.press/v80/chen18a.html) | 多任务损失可能需要动态尺度平衡 | 不代表 GradNorm 必然改善本项目 | VERIFIED_PRIMARY |
| [PCGrad](https://papers.nips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html) | 多任务梯度可能冲突，可用投影缓解 | 不代表当前 M1 的失败仅由梯度冲突造成 | VERIFIED_PRIMARY |

## 4. WHY / HOW / WHAT 三路扫描

| 文献 | WHY | HOW | WHAT | 对 RQ1 的作用 |
|---|---|---|---|---|
| xBD | 灾损数据规模小、地点和灾种多样 | pre/post optical + building/damage annotation + metadata | 建立多事件 BDA 基准 | 定义任务背景；不提供严格 cross-event 因果对照 |
| BRIGHT | 光学受云、烟和夜间限制，跨事件迁移困难 | pre optical + post VHR SAR；standard 与 cross-event 两套协议 | 支持 unseen-event、UDA、SSL 和 multimodal change research | 最接近本项目输入和 RQ1；应复用其 cross-event 思路而非只复用数据 |
| DisasterM3 | 灾害理解同时受 hazard、sensor、region 和任务复杂度影响 | 多事件、多传感器、VLM instruction 多任务 | 不同灾种/传感器性能差异明显 | 说明研究 metadata/多任务合理；不构成辅助头有效性的直接证据 |
| STCA | 新事件的 post-event 标注不可及时获得，source-target shift 大 | 用 target pre-image 和 source damage 构造适配信号 | 改善 domain-adaptive BDA | 说明 domain adaptation 是独立分支；不能与本轮 zero-shot 控制混合 |
| SAR transferability 2026 | 同事件分数会高估部署到新事件的表现 | 11 个事件的 event-based CV | unseen-event median macro F1 明显降低且方差大 | 支持把 event 作为统计独立单位 |
| Metadata earthquake study | 视觉证据不足以解释结构脆弱性 | 融合图像、ground motion、soil 与 SAR proxies | 在同一地震的跨区域测试中报告提升 | 支持未来 metadata enrichment，但外部效度不足以决定本轮结构 |

### 跨论文综合

- 共同 WHY：跨事件 domain shift、灾种相关损伤机制、传感器差异和稀缺标签会让随机 tile split 高估泛化。
- 分歧 HOW：benchmark 用 event holdout；adaptation 方法使用 target data；metadata 方法直接融合物理上下文；DisasterM3 使用 VLM 多任务。它们回答的不是同一个问题。
- 最强 WHAT：未见事件的性能损失在多个工作中出现，event-level evaluation 有直接依据。
- 未解决缺口：尚无检索到的一手工作证明“灾种分类作为 auxiliary supervision、且 metadata 不进入 forward”能在 VHR optical-SAR 像素级 BDA 上改善 cross-event damage F1。

## 5. 逐 claim 事实核查（fact-check）

| Claim | 判定 | 证据与修正 |
|---|---|---|
| “event-disjoint split 是所有 BDA 实验的必要条件” | OVERSTATED | BRIGHT 同时使用 standard per-event 7:1:2 split 和独立 cross-event setup。只有声称 unseen-event generalization 时，event-disjoint/LOEO 才是必要条件。 |
| “metadata 有助于泛化” | PARTIALLY_SUPPORTED | 地震专用研究有正向证据，但多为直接融合、单灾种和区域迁移；不能泛化为任意 metadata 或本项目灾种辅助头都有效。 |
| “DisasterM3 证明多任务会提升本项目 damage segmentation” | NOT_SUPPORTED | DisasterM3 证明的是数据集/任务体系和 VLM benchmark/fine-tuning；任务、模型、指标均不同。 |
| “within-event shuffled SAR 是合理负对照” | METHODOLOGICALLY_JUSTIFIED, NOT_BENCHMARK_STANDARD | 它能保留事件域并破坏样本配对，适合检验 correspondence；但没有找到它已成为 BDA 标准协议的直接证据，必须公开 permutation 和残余混杂。 |
| “paired SAR 优于 shuffled 就证明 SAR 学到损伤机制” | OVERSTATED | 差值只说明样本配对信息被利用；还可能利用配准、building count、边缘或标签密度线索。需同时超过 C1，并做 event×class 与 nuisance 审计。 |
| “固定 3 seeds + 4 events 的 bootstrap 可以证明显著性” | NOT_SUPPORTED | 独立单位太少；CI 只能作不确定性描述，不能包装为稳定总体推断。 |
| “灾种辅助头 M1 已证明灾害知识提升泛化” | CONTRADICTED_BY_PROJECT_PROVENANCE | train disaster macro 约 0.999，validation 仅 0.239 且低于 majority 0.28；现有 provenance 只支持一般正则化/优化信号。 |

## 6. Methodology-focus 审稿

### Reviewer identity

跨模态遥感变化检测、grouped generalization、机器学习实验设计与统计复现审稿人。未绑定具体期刊标准。

### Overall recommendation

**Major Revision**。RQ 可回答且对部署有意义，但旧 test 暴露、非 OOF prior、数据源重叠和独立事件数不足会使正式结论失效。

### Strengths

1. 主要对比是 paired vs within-event deranged，能把“有 SAR”与“使用配对关系”分开。
   Evidence anchor: `text: docs/PROJECT_AUDIT_20260814.md "within-event shuffled SAR 是项目目前最有价值的设计之一"`
2. 已有 test embargo、三种子、event-macro/worst-event 和层级 bootstrap 的纪律。
   Evidence anchor: `text: reports/stage2_v2/metadata_multitask_v1_20260814/RESULTS.md "未读取、评估或用于选择 test"`
3. M1 在语义门槛失败后停止，没有用 damage 均值上涨覆盖灾种泛化失败。
   Evidence anchor: `text: reports/stage2_v2/metadata_multitask_v1_20260814/RESULTS.md "M1 未通过预注册首轮筛选"`

### Weaknesses

#### W1: Stage-1 prior 污染会破坏 C0-C3 的共同基线

- Severity: Critical
- Evidence anchor: `text: docs/PROJECT_AUDIT_20260814.md "latest val 119/290 灾前图被 O1 训练见过"`
- Confidence: 5 — 直接项目审计证据
- 问题：prior 参与输入和模型选择时，上游见过 validation 图像会偏高估所有含 prior 条件；不能因所有条件共享污染就认为差值一定无偏。
- 最小修复：在 outer event 外训练 Stage-1，并为 development 样本生成 event-aware OOF 概率 prior；记录模型 commit、fold 和输出哈希。

#### W2: 现有 test 已暴露，无法承担最终确认

- Severity: Critical
- Evidence anchor: `text: docs/PROJECT_AUDIT_20260814.md "clean strict test 和 event-group test 都已被多轮评估暴露"`
- Confidence: 5 — 直接 test-exposure ledger 事实
- 问题：任何基于这些 test 的最终陈述都混入历史选择反馈。
- 最小修复：预注册后一次性评估新的封存外部事件，或只把 nested event CV 明确称为 development evidence。

#### W3: DisasterM3 与 BRIGHT 不是天然独立来源

- Severity: Major
- Evidence anchor: `text: DisasterM3 paper §3 "There are 26 events from the xBD and BRIGHT dataset"`
- Confidence: 5 — 数据集原文
- 问题：简单拼接会重复事件、派生图像或标注来源，夸大 event 数与样本独立性。
- 最小修复：建立 canonical event map；对 pre/post/mask 做 exact 与近重复审计；同一 geographic scene 只能属于一个 outer fold。

#### W4: C3 只按 event shuffle 仍可能改变可见 nuisance 分布

- Severity: Major
- Evidence anchor: `text: docs/PROJECT_AUDIT_20260814.md "尽可能 deranged，不允许样本映射到自身"`
- Confidence: 4 — 设计推断，有项目审计支持
- 问题：若 source/target 在 sensor、acquisition、building count、SAR histogram 或 crop damage density 上差异明显，C2-C3 不只代表 correspondence。
- 最小修复：在 event 内按 sensor/acquisition bin、图像尺寸、building-pixel quantile 和 SAR histogram quantile 分层 derange；保存映射并报告匹配平衡。

#### W5: outer evaluation 与 checkpoint selection 尚未完全嵌套

- Severity: Major
- Evidence anchor: `text: docs/PROJECT_AUDIT_20260814.md "held-out event 同时用于选 epoch"`
- Confidence: 5 — 直接项目审计证据
- 问题：用 outer event 选 epoch 会乐观偏置 LOEO 分数。
- 最小修复：inner grouped validation 选择 epoch/阈值，outer event 只前向一次；所有条件共享 inner fold 和 selection rule。

#### W6: event×class 缺失类与宏平均分母没有完全冻结

- Severity: Major
- Evidence anchor: `text: docs/PROJECT_AUDIT_20260814.md "不同实验/划分的 event-class macro 可能具有不同分母"`
- Confidence: 5 — 代码/报告审计结论
- 问题：不同 outer folds 的可计算类别不同，简单平均会改变 estimand。
- 最小修复：每个 event×class cell 报 support；GT absent 记 NA，不进入该 cell 的类别均值；present-in-GT 而无预测必须记 0；同时报告固定三类 BO 和 present-class damage 两套指标。

#### W7: 三 seed 不能补偿少量独立事件

- Severity: Major
- Evidence anchor: `text: reports/stage2_v2/metadata_multitask_v1_20260814/RESULTS.md "validation 只有 4 个事件"`
- Confidence: 5 — 直接样本结构
- 问题：seed 是算法随机性重复，不是新的部署域。四个事件上的层级 CI 不具备强总体推断力。
- 最小修复：优先增加 canonical events；在不足时报告全体 outer-event paired differences 和描述性 CI，不作显著性语言。

#### W8: C0-C3 的训练预算与缺失模态处理需要统一

- Severity: Minor
- Evidence anchor: `absence: proposed RQ1 matrix — expected fixed optimizer-step and missing-modality implementation; checked user protocol and current project reports`
- Confidence: 4 — 协议缺项
- 问题：按 epoch 训练时输入条件可能产生不同 early-stop 长度；不同输入通道的 first-conv 初始化也引入结构差异。
- 最小修复：固定 optimizer steps 与 eval cadence；C1-C3 使用同一五通道结构，缺失 SAR 置零并固定相同初始化。C0 作为诊断单列，不参与 paired 因果差值。

### Reproducibility verdict

当前协议具备成为可复现实验的骨架，但在 OOF prior、blind test、canonical dedup、permutation balance 和 nested selection 关闭前，不能生成可支持核心 claim 的 provenance。

## 7. 冻结候选协议 v1.1

### 7.1 数据门槛

正式启动前必须全部 PASS：

```text
G0 exact full-sample cross-fold duplicates = 0
G1 canonical event overlap across outer folds = 0
G2 known/available geospatial scene overlap = 0
G3 every development prior produced by a Stage-1 model that did not see that outer event
G4 each outer event has non-zero damaged or destroyed support; missing cells disclosed
G5 test labels/checkpoints unavailable to training and selection process
G6 C3 permutation fixed, saved, deranged where feasible, and balance-audited
```

若 G2 因元数据缺失只能部分审计，必须标记 `geospatial_independence_unresolved`，不能写成已证明地理独立。

### 7.2 条件矩阵

| ID | 输入 | 角色 | 主要比较 |
|---|---|---|---|
| C0 | OOF prior only | shortcut diagnostic | C2−C0 只作解释 |
| C1 | pre RGB + OOF prior + zero SAR | additive baseline | C2−C1 |
| C2 | pre RGB + OOF prior + paired post SAR | candidate | C2−C3、C2−C1 |
| C3 | pre RGB + OOF prior + fixed within-event deranged SAR | correspondence negative control | C2−C3 |

C1-C3 使用完全相同的模型结构、初始化、optimizer steps、batch order、scheduler、augmentation、checkpoint rule 和评价代码。C0 若因输入语义无法保持同结构，必须明确为 diagnostic，不把结构差异归因于 SAR。

### 7.3 C3 permutation 规范

1. train、inner-val、outer-eval 分开生成，绝不跨 split 取 SAR；
2. 首先按 `canonical_event × sensor × acquisition/bin` 分层；
3. 在 strata 内用固定 seed 生成循环 derangement；任何 self-map 视为 hard error；
4. singleton strata 逐级放宽 sensor/acquisition bin，但不跨 event；无法 derange 的样本从 C2-C3 paired estimand 排除并单独计数；
5. 生成前后比较 SAR intensity histogram、image size、building-pixel quantile 和 source frequency；
6. 映射文件、seed、生成脚本 commit 和 SHA-256 进入 registry。

### 7.4 分层评估与选择

```text
Outer loop: canonical-event holdout
Inner loop: grouped folds over remaining events, select epoch only here
Stage-1: retrain without outer event; produce inner/outer OOF priors
Stage-2: train C0-C3 with seeds 42, 3407, 2026
Outer event: exactly one evaluation per frozen run
Final blind set: one post-freeze evaluation, if a genuinely sealed set exists
```

若事件数量和计算预算不足以对每个 outer fold 重训完整 Stage-1/Stage-2，则先减少 outer folds 的覆盖声明，不允许复用见过 outer event 的 prior。

### 7.5 指标

Primary：

```text
event_macro_damage_F1 = mean_event(mean(F1_damaged, F1_destroyed over GT-present classes))
```

Primary contrast：`C2−C3`。Key secondary：`C2−C1`、BO 3-grade macro、Damaged、Destroyed、binary damage、worst-event、每个 event×class confusion/support。

同时固定命名：

- `pooled_pixel_*`：所有像素合并；
- `sample_macro_*`：先按样本再平均；
- `event_macro_*`：先按事件再等权平均；
- `blind_external_*`：只用于新封存外部事件。

### 7.6 统计与判定

- 每个 outer event 和 seed 产生 paired C2-C3、C2-C1 差值；
- 层级 bootstrap 首先重采样 outer events，再在事件内重采样样本；seed 作为同一设计重复，在 event 内配对保留；
- 事件少于 8 时，CI 只描述不确定性；报告完整 event-level 差值，不使用“统计证明”；
- 不按 p-value 单独决策，主看 effect size、方向一致性、worst-event 和两类 damage trade-off。

建议筛选门：

```text
mean C2-C3 event_macro_damage_F1 >= +0.02
positive C2-C3 on >= 2/3 seeds and >= 60% eligible outer events
mean C2-C1 event_macro_damage_F1 >= +0.01
neither pooled Damaged nor Destroyed mean delta < -0.01
worst-event BO delta versus C3 >= -0.01
all data/provenance gates PASS
```

这些是筛选阈值，不是自然规律；必须在运行前冻结，不能按观测结果回改。

## 8. 与 Metadata M0/M1 的关系

Metadata v1 已完成且失败，不能被 RQ1 重写：

- damage 指标存在正向筛选信号；
- disaster 分类在 train 近乎完美、validation 低于 majority；
- 因灾种与训练事件一一绑定，现有证据不能区分灾害语义和 event shortcut；
- 按原门槛停止 M2 是正确的。

RQ1 不复活 metadata 分支。只有在每个灾种拥有多个独立训练事件、并能做 within-disaster leave-one-event-out 后，才重新审查 disaster auxiliary supervision。未来若继续，还必须加入固定乱序标签或等参数非语义辅助任务，以区分语义收益和一般 regularization。

## 9. Material Passport / provenance 最小字段

每个实验必须登记：

```text
experiment_id, RQ, hypothesis, status
dataset/source versions, canonical event map
manifest paths and SHA-256, split unit, audit verdict
Stage-1 prior model/fold/input exposure/output hash
config and resolved config hashes
code commit, dirty patch hash, environment
condition, seed, initialization hash, optimizer steps
selection split, checkpoint rule, threshold source
test exposure status
primary/secondary metrics and exact estimand
run/checkpoint/report artifact pointers and hashes
gate decision and reason
claim -> ALIGNED / OVERSTATED / NOT_SUPPORTED_BY_PROVENANCE / PROVENANCE_INSUFFICIENT
```

本仓库的机器可读入口为 `experiments/registry.json`；校验命令为：

```bash
python scripts/validate_experiment_registry.py experiments/registry.json
```

## 10. 下一步执行顺序

1. 完成 canonical event/source/content/scene 去重表；
2. 重训 Stage-1 group folds 并生成 OOF probability priors；
3. 实现并冻结 C3 derangement 与 balance audit；
4. 运行 registry、manifest、unit、CPU/CUDA smoke 和两批次 overfit；
5. 先做一个 outer fold 的 C1/C2/C3 seed-42 pilot，只验证管线，不用于改阈值；
6. pilot 无协议错误后，按冻结队列执行所有 outer events × 3 seeds；
7. 汇总 development evidence；只有存在真正封存的新事件时才进行一次 final blind evaluation。

## 11. 审查边界

- 文献搜索不是穷尽式 systematic review；未来论文写作前应扩展数据库与人工全文阅读。
- 本文件不把 web 摘要等同于全文复核；正式写作时应保存并人工阅读所引用原文。
- 研究设计审查不能证明未来实验实现正确；实现仍需 manifest、代码、运行时和结果层的独立审计。
- ARS 用于方法学与证据边界，不替代 GPU 训练、数据质量审计或外部盲测。
