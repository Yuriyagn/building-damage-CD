# D4 Leave-One-Event-Out 诊断结论

> **状态更新（2026-08-14）：** 本文件的 D4 数字保持有效，但文末“继续 E1”已过时。
> E1 event×class-balanced sampler 后续已经完成：BO macro-F1 `0.327085`、
> event macro `0.289261`、Damaged F1 `0.0704`、Destroyed F1 `0.1884`，均未形成
> 可接受的总体改进。该实现会对极小 event×class cells 进行极端重复采样，因此只能
> 判定“当前 E1 实现失败”，不能推出所有平衡采样都无效。当前结论见
> [项目审计](../../../docs/PROJECT_AUDIT_20260814.md)。

日期：2026-08-03<br>
协议：`stage2_event_shortcut_v1`<br>
数据来源：严格清洗训练清单，1207 张，7 个事件<br>
测试集：未使用

## 主要结果

| 留出事件 | 样本数 | BO macro-F1 | Intact F1 | Damaged F1 | Destroyed F1 |
|---|---:|---:|---:|---:|---:|
| Beirut explosion | 6 | 0.3757 | 0.3892 | 0.2363 | 0.5015 |
| Hawaii wildfire | 3 | 0.3524 | 0.1725 | 0.0077 | 0.8770 |
| Libya flood | 8 | 0.3430 | 0.6846 | 0.0115 | 0.3328 |
| Myanmar hurricane | 78 | 0.3835 | 0.8611 | 0.2331 | 0.0563 |
| Spain/La Palma volcano | 667 | 0.3002 | 0.5200 | 0.0034 | 0.3773 |
| Turkey earthquake5 | 19 | 0.3016 | 0.8598 | 0.0449 | 0.0000 |
| Ukraine conflict | 426 | 0.3104 | 0.5335 | 0.3708 | 0.0270 |

- D4 事件宏平均：**0.3381**。
- 事件 bootstrap 95% CI：**[0.3147, 0.3617]**。
- 最差事件：Spain/La Palma volcano，**0.3002**。
- Damaged 的事件均值/标准差：**0.1297 / 0.1374**。
- Destroyed 的事件均值/标准差：**0.3103 / 0.2937**。

## 与 E0 的同口径比较

E0 在严格验证集上的总体 BO macro-F1 为 0.3586，事件宏平均为 0.3377，事件 bootstrap 95% CI 为 [0.3207, 0.3523]。D4 的事件宏平均 0.3381 与 E0 几乎相同，因此只看三分类事件宏平均无法发现主要问题。

真正显著的是类别级不稳定性：多个留出事件上某一损伤等级接近完全失效。例如 Spain 和 Libya 的 Damaged F1 分别为 0.0034 和 0.0115；Turkey 的 Destroyed F1 为 0；Ukraine 的 Destroyed F1 为 0.0270。总体 macro-F1 被 Intact 或另一个损伤等级补偿。

## 科研结论

D4 提供了模型实际受事件—类别耦合影响的直接证据。更严谨的表述是：

> Leave-One-Event-Out 实验显示，模型的总体事件宏平均相对稳定，但损伤等级在不同未见事件间表现出极高方差，并在多个事件上发生单类别近乎完全失效。这说明当前监督能够维持粗粒度平均性能，却不足以支持事件不变的三等级损伤判别。该现象与训练集中 Damaged 和 Destroyed 分别高度集中于 Ukraine conflict 与 Spain/La Palma volcano 的分布统计一致。

这些结果不支持“移除主导事件后对应类别必然在所有事件上统一崩溃”这一过强命题。例如留出 Spain 后 Destroyed F1 仍为 0.3773，但 Damaged 几乎归零。事件监督缺失会改变整个三分类决策边界，错误可能转移到相邻等级，而不只体现在被集中类别本身。

## 局限

- 7 个事件的样本量从 3 到 667，单事件分数方差很大。
- Hawaii、Beirut、Libya 等极小折只能作为诊断，不宜单独形成强结论。
- 每折用留出事件进行 checkpoint 选择，因此这是 LOEO 验证诊断，不是完全嵌套的无偏泛化估计。
- 本实验没有使用正式 test，也不改变现有 test embargo。

## 后续实验决策

历史计划是继续 E1 事件×类别联合采样，并将主要准则设为事件—类别 macro、最差事件及 Damaged/Destroyed 事件标准差。E1 已按该方向运行但没有通过，因此此分支已停止；下一步应优先修复 OOF prior、独立事件覆盖和 capped/effective-number 采样设计。
