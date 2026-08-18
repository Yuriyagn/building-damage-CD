# RQ1：配对 SAR 跨事件对应性 nested-CV 结果

> 冻结判定：**程序完整完成；科学验收失败；分支按预注册规则关闭。**
> 证据等级：**development-only nested CV**，没有独立盲测确认，不支持部署或未知事件泛化声明。
> 结果日期：2026-08-18；协议：`rq1_paired_sar_nested_cv_v1.2`。

## 1. 结论先行

配对的灾后 SAR（C2）在 14 个开发事件、3 个种子的 nested event CV 中，平均优于同事件固定错配 SAR（C3）：

- 主效应 `C2−C3 event-macro BO damage F1 = +0.06198`；
- 描述性层级 paired bootstrap 95% 区间为 `[+0.01206, +0.11543]`；
- 相对无 SAR 基线（C1）为 `C2−C1 = +0.06508`；
- 3/3 seeds 为正，14 个事件中 9 个为正；
- 平均收益主要来自 `Destroyed F1 +0.11368`，`Damaged F1` 只增加 `+0.01028`。

但是冻结的 7 项复合门槛只通过 6 项。`Myanmar Hurricane` 的 `C2−C3 BO grade macro F1 = -0.07386`，低于最差事件下限 `-0.01`。因此，本轮只能支持“在当前开发事件集合上，样本级 SAR 对应关系产生了平均正信号”，不能支持“该信号在跨事件上稳健”或“已能泛化到新的灾害事件”。

这不是运行错误：所有队列和结果汇总都已结束，失败的是预注册科学门槛。

## 2. 研究问题与受控条件

研究问题是：在 canonical-event-disjoint 评价下，样本对齐的 post-event SAR 是否能在 pre-event RGB 与 event-excluded OOF building prior 之外提供可迁移的损伤等级信息？

| 条件 | 五通道输入 | 作用 |
|---|---|---|
| C0 | `[0, 0, 0, OOF prior, 0]` | prior-only 捷径诊断；仅 seed 42，不参与主门槛 |
| C1 | `[pre RGB, OOF prior, 0]` | 无 post-event SAR 的加性基线 |
| C2 | `[pre RGB, OOF prior, paired SAR]` | 候选方法 |
| C3 | `[pre RGB, OOF prior, fixed within-event deranged SAR]` | 保留事件域、破坏样本对应关系的负对照 |

协议使用 7 个 outer folds，每个 fold 留出 2 个 canonical events；inner validation 使用循环相邻的另 2 个事件。Stage-1 prior 对相应 outer/inner 事件做排除训练，避免旧方案中 prior 生成器看过下游 holdout pre-image 的污染。C1–C3 共用 inner-fold 选择出的 optimizer step，tie 时取更早 step；outer events 不参与 checkpoint 选择。

## 3. 完整性与运行状态

| 阶段 | 预期 | 完成 | 失败 | 判定 |
|---|---:|---:|---:|---|
| Stage-1 event-excluded OOF models | 56 | 56 | 0 | 完整 |
| Stage-2 inner selection runs | 21 | 21 | 0 | 完整 |
| Stage-2 final outer runs | 70 | 70 | 0 | 完整 |
| Final outer evaluations | 70 | 70 | 0 | 完整 |

Stage-2 修正版 v3 从 2026-08-17 23:43:07 运行到 2026-08-18 16:53:27（Asia/Shanghai），约 `17 h 10 min`。完成时无 pending/failed task、无 `.in_progress` 标记；结果整理时也没有残留训练进程。

7 个 outer folds 选定的公共 step 为：

| outer fold | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| selected step | 14000 | 10000 | 14000 | 14000 | 14000 | 12000 | 16000 |

## 4. 主要结果

### 4.1 冻结 estimands

| estimand | 结果 | 解释 |
|---|---:|---|
| Primary: C2−C3 event-macro damage F1 | **+0.06198** | 配对 SAR 相对同事件错配 SAR 的平均对应性信号 |
| Descriptive hierarchical bootstrap 95% CI | `[+0.01206, +0.11543]` | 10,000 次；先重采样事件，再重采样 paired samples；保留事件内 seeds |
| Secondary: C2−C1 event-macro damage F1 | **+0.06508** | 配对 SAR 相对零 SAR 输入的平均增益 |
| C2−C3 Damaged F1 | +0.01028 | 正，但幅度很小 |
| C2−C3 Destroyed F1 | **+0.11368** | 主增益来源 |
| Worst-event C2−C3 grade macro F1 | **−0.07386** | Myanmar Hurricane；触发停止门 |

bootstrap 区间只作为开发集不确定性描述，不是独立确认性检验。协议没有预注册显著性检验，也没有新的 blind event，因此本文不报告或暗示 confirmatory p-value。

### 4.2 各条件绝对表现（描述性）

下表是 outer-event × seed 的等权宏平均。C0 只有 seed 42；C1–C3 各有 42 个 event-seed 单元。它用于理解效应来源，不替代预注册的 paired estimand。

| 条件 | BO damage macro | BO grade macro | Damaged | Destroyed | Binary damage | Predicted-gate 4-class macro |
|---|---:|---:|---:|---:|---:|---:|
| C0 prior-only | 0.14207 | 0.24550 | **0.19623** | 0.08791 | 0.30904 | 0.38326 |
| C1 pre + OOF prior | 0.12289 | 0.27506 | 0.15724 | 0.08853 | 0.24965 | 0.40135 |
| C2 paired SAR | **0.18797** | **0.34408** | 0.14620 | **0.22974** | **0.33129** | **0.44156** |
| C3 deranged SAR | 0.12599 | 0.26760 | 0.13593 | 0.11606 | 0.25744 | 0.39668 |

C2 的平均优势是真实存在于这批开发结果中的，但不能忽略两个结构性信号：C0 的 Damaged 绝对值最高，说明 building prior/事件类别结构仍可制造损伤分数；C2 的改善主要集中在 Destroyed，而不是两个损伤等级同时稳定改善。

### 4.3 种子稳定性

| seed | C2−C3 event-macro damage F1 |
|---:|---:|
| 42 | +0.07725 |
| 3407 | +0.05351 |
| 2026 | +0.05517 |

3/3 seeds 均为正，因此均值不是由单一随机种子制造；但 event 层异质性仍然很大。

### 4.4 事件异质性

| outer event | C2−C3 damage F1 | C2−C3 grade F1 | 方向 |
|---|---:|---:|---|
| Nyiragongo 2021 | **+0.34701** | +0.24947 | 正 |
| Hawaii Wildfire | +0.15546 | +0.14579 | 正 |
| Turkey EQ 2023 | +0.10031 | +0.04967 | 正 |
| Bata Explosion | +0.09325 | +0.19997 | 正 |
| La Palma Volcano | +0.09311 | +0.09285 | 正 |
| Libya Flood | +0.07624 | +0.04800 | 正 |
| Ukraine Conflict | +0.06625 | +0.02656 | 正 |
| Marshall Wildfire | +0.04556 | +0.16032 | 正 |
| Beirut Explosion 2020 | +0.03783 | +0.01335 | 正 |
| Morocco Earthquake | −0.00044 | +0.03818 | 非正 |
| Haiti Earthquake | −0.00193 | +0.00769 | 非正 |
| Myanmar Hurricane | −0.00436 | **−0.07386** | 非正；最差 grade |
| Mexico Hurricane | −0.06785 | −0.03663 | 非正 |
| Noto Earthquake | **−0.07277** | +0.14934 | 非正；最差 damage |

完整的事件级 damage/grade/Damaged/Destroyed/C2−C1 数值见 [`event_effects.csv`](event_effects.csv)。Noto 的 grade 正而 damage 负，Myanmar 的 damage 接近 0 而 grade 明显负，进一步说明单一聚合指标会掩盖类别间取舍。

## 5. 冻结门槛判定

| 预注册门槛 | 阈值 | 观测 | 结果 |
|---|---:|---:|---|
| mean C2−C3 damage | ≥ +0.02 | +0.06198 | PASS |
| positive seeds | ≥ 2/3 | 3/3 | PASS |
| positive events | ≥ 9/14 | 9/14 | PASS（刚好达线） |
| mean C2−C1 damage | ≥ +0.01 | +0.06508 | PASS |
| Damaged delta | ≥ −0.01 | +0.01028 | PASS |
| Destroyed delta | ≥ −0.01 | +0.11368 | PASS |
| worst-event grade delta | ≥ −0.01 | **−0.07386** | **FAIL** |

复合门槛要求全部通过，因此总判定为 `failed`。按照冻结协议，不再用同一开发事件追加超参数、架构或事后排除事件来“修复”门槛；这样做会把一次受控检验变成重复搜索。

## 6. 科学解释

### 可以说什么

- 在这 14 个已暴露的开发事件上，保留样本级 post-SAR 对应关系平均优于同事件错配和无 SAR 对照。
- 三个种子的主效应方向一致，平均信号不是单一种子异常。
- 收益有明显类别结构，主要来自 Destroyed；这比笼统地说“SAR 提升损伤识别”更准确。
- event-excluded OOF prior 和固定、同事件、无自配对的 C3 对照，使“只是 prior 泄漏”或“只是事件域纹理”成为较弱的解释。

### 不能说什么

- 不能说 paired SAR 已在所有或大多数灾害机制上稳健；5/14 事件没有正 damage 收益。
- 不能说模型已泛化到真正未见事件；全部 14 个事件都被纳入开发 nested CV，且历史上相关 splits 已被查看。
- 不能把 bootstrap CI 当作外部确认；重采样无法创造新的独立事件。
- 不能声称 geospatial footprint 独立；当前审计只有内容、事件和图像层面的证据。
- 不能把平均效应解释为灾害物理因果效应；它是受控输入对应性效应。

## 7. 运行缺陷修正与剩余风险

本轮正式 v3 是在终止错误的慢速/不完整尝试后，从头生成的新结果树。为使长任务在可接受时间内完成，训练路径增加了 fixed-step 快照、可恢复队列、低开销训练诊断以及完整 validation iterator 的复用。工程等价性曾用以下检查约束：

- 20-step 双 GPU 重跑得到相同模型 hash；
- 优化前后 100-step checkpoint 与 metrics 精确一致；
- 单任务与并发 500-step checkpoint 与 metrics 精确一致；
- 汇总时严格检查 21/21 inner、70/70 final training 和 70/70 evaluations，均无 invalid artifact。

这些检查支持“加速没有改变被检查路径的数值结果”，但不等价于在另一台机器上独立重跑全部 147 个训练任务。当前复现状态应标为 `ANALYZED / artifact-verified`，不是 `independently replicated`。

剩余工程风险包括：运行产物依赖未随 Git 发布的数据与 checkpoint；manifest 中缺少 SAR sensor/acquisition metadata，C3 无法按传感器与采集时相进一步平衡；不同 CUDA/cuDNN 环境可能不提供逐位一致性。

## 8. Material Passport

| 材料 | 来源与时间 | SHA-256 / 规模 | 本实验角色 | 权利与处理 |
|---|---|---|---|---|
| R4 event-group source manifests | 本地已审计 DisasterM3 + BRIGHT 派生包；协议生成于 2026-08-15 | train `07101572…`; val `5c2c8d7c…`; historical test `bf5c9201…`; 合并去重后 2,214 samples / 14 events | nested-CV 的开发事件池 | 原图与标签不随 Git 分发；使用者须按源数据许可自行取得。历史 test 被重新分类为 development data |
| Frozen protocol | `configs/rq1_paired_sar_nested_cv_v1/protocol.yaml` | `55378b43…` | estimands、fold、seed、step 选择和 7 项门槛 | 仓库内可公开复核；未在 outer evaluation 后改门槛 |
| Prepared lineage audit | 2026-08-15 生成 | `PREPARED.json` `32e49fed…`; 14 events, 2,214 samples, 56 Stage-1 models, 0 hard errors | 事件排除 prior 谱系与预检 | 大型逐样本 manifests 可由脚本重建，不提交 Git |
| Finalized fold audit | Stage-1 OOF 完成后生成 | `FINALIZED.json` `b19791aa…`; 7 folds, 0 hard errors | 确认 inner/final/outer roles 与 OOF prior 已落盘 | 只报告摘要和 digest；逐样本路径留在本地 |
| Fixed derangement audit | Stage-2 前生成 | `AUDIT.json` `5a661ab3…`; 28 role-fold maps; 0 self/cross-event/cross-shape/cross-source pairs | C3 negative control | 全部为 bijection；prior-area 邻近比例最低 0.9894；sensor balance 因源 metadata 缺失不可审计 |
| Final source summary | 2026-08-18 生成 | `summary.json` `b8bc506a…`; 42 paired event-seed rows | 本报告所有冻结效应与 gate 的权威数值源 | 只导出派生指标，不包含数据、预测或权重 |

排除项：旧 legacy test 分数、Codabench feedback、连通域 `cc_surrogate`、训练 loss 最优点、事后事件剔除和任何基于本轮 outer 结果的新 checkpoint 选择，均不参与本轮科学判定。

## 9. Statistical Fallacy Scan（11/11）

| 谬误 | 状态 | 审查结果 |
|---|---|---|
| Simpson's paradox | CAUTION | 总体为正但 5/14 events 非正；已同时报告 event strata，禁止只引用 pooled mean |
| Ecological fallacy | PASS | 推断单位保持在 event/event-seed，不把事件均值外推为单栋建筑效果 |
| Berkson's paradox | CAUTION | 事件池来自可取得且已有标注的开发数据，可能与真实部署事件选择机制不同 |
| Collider bias | PASS | 没有按模型输出或 post-treatment covariate 筛样/调整；未发现明确 collider 路径 |
| Base-rate neglect | CAUTION | 等级与事件支撑不均；报告 Damaged/Destroyed 和 event×class，而不是只报总体分数 |
| Regression to the mean | PASS | 不是按极端预分数挑选后的前后比较；未发现典型回归均值设计 |
| Survivorship bias | PASS | 56/56 Stage-1、21/21 inner、70/70 final 全部纳入；没有丢弃失败 run |
| Look-elsewhere effect | CAUTION | 7 项 gate 预先冻结；事件诊断为多重描述，不能把其中最好事件当确认性证据 |
| Garden of forking paths | CAUTION | v3 在 fresh output tree 上按冻结协议重跑；但此前工程修复和同一开发事件历史暴露限制确认性 |
| Correlation vs causation | CAUTION | paired/deranged 干预支持输入对应性归因，不支持灾害机制或外部世界因果声明 |
| Reverse causality | PASS | 输入时序固定为 pre-optical/post-SAR；但仍不能从单次观测推断物理损伤形成方向 |

**Overall Confidence: RED_FLAG for the robust cross-event claim; CAUTION for the average development-set correspondence effect.** 红旗来自冻结 composite gate 失败，而不是运行产物缺失。

## 10. 决策与后续

RQ1 当前分支到此冻结，不继续在同一 14 个事件上做方法调参。下一项有科学价值的工作不是再换模型，而是建立更好的确认数据：

1. 获取并封存新的 canonical disaster events，训练和选择期间不可见；
2. 补足 georeference 与 SAR sensor/acquisition metadata，做 footprint、传感器和时相独立性审计；
3. 在新事件上原样运行冻结的 C1/C2/C3 recipe 与 gate，不重选阈值或 steps；
4. 只有独立确认通过后，才升级“开发集平均对应性信号”为“跨事件稳健证据”。

在新盲事件可用前，最诚实的项目结论是：**paired SAR 有平均正信号，但跨事件稳健性不成立，方法分支停止。**

## 11. 可复核文件

- [`summary.json`](summary.json)：精简机器可读结论、门槛、完整性与 provenance；
- [`event_effects.csv`](event_effects.csv)：14 个事件的成对效应；
- [`configs/rq1_paired_sar_nested_cv_v1/protocol.yaml`](../../../configs/rq1_paired_sar_nested_cv_v1/protocol.yaml)：冻结协议；
- [`scripts/prepare_rq1_paired_sar_nested_cv.py`](../../../scripts/prepare_rq1_paired_sar_nested_cv.py)：fold 与 Stage-1 lineage 构建；
- [`scripts/run_rq1_stage1_oof_queue.py`](../../../scripts/run_rq1_stage1_oof_queue.py)：Stage-1 OOF 队列；
- [`scripts/finalize_rq1_oof_manifests.py`](../../../scripts/finalize_rq1_oof_manifests.py)：OOF manifest 最终化；
- [`scripts/build_rq1_balanced_sar_derangements.py`](../../../scripts/build_rq1_balanced_sar_derangements.py)：固定 C3 derangement；
- [`scripts/run_rq1_stage2_nested_queue.py`](../../../scripts/run_rq1_stage2_nested_queue.py)：inner/final 队列与完整性检查；
- [`scripts/summarize_rq1_nested_cv.py`](../../../scripts/summarize_rq1_nested_cv.py)：冻结 estimand、bootstrap 与 gate 汇总。
