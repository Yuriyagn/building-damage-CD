# Prompt for independent model audit

将下面整段提示词连同 GitHub 仓库链接交给具备代码阅读能力的大模型。优先让模型直接读取仓库；如果模型不能访问 GitHub，则至少上传本目录、`configs/rq3_damage_evidence_decomposition_v1/`、相关 `scripts/`、`src/models/rq3_damage_evidence.py`、`src/stage2/rq3_*.py`、`src/stage2/train_stage2_v2.py`、数据集适配代码和 `tests/test_rq3_damage_evidence.py`。

```text
你是一名严格的计算机视觉、遥感灾损评估、PyTorch 数值稳定性和实验方法学审查员。请审查 GitHub 仓库 Yuriyagn/building-damage-CD 中的 RQ3 Damage Evidence Decomposition 实验，重点目录为：

- reports/stage2_v2/rq3_damage_evidence_decomposition_v1_20260820/
- configs/rq3_damage_evidence_decomposition_v1/
- scripts/run_rq3_damage_evidence.py
- scripts/launch_rq3_unlocked_e1_pilot.sh
- scripts/summarize_rq3_damage_evidence.py
- src/models/rq3_damage_evidence.py
- src/stage2/rq3_components.py
- src/stage2/rq3_runtime.py
- src/stage2/train_stage2_v2.py
- src/stage2/test_stage2_v2.py
- src/stage2/datasets.py
- src/stage2/datasets_v2.py
- tests/test_rq3_damage_evidence.py
- experiments/registry.json

研究问题不是“SAR 是否有用”，而是能否通过 Damage Evidence Decomposition 抑制 SAR 的事件级失效。E1 是唯一已经运行的候选：把 F0 像素等级监督改成基于事件外 OOF building prior 组件的建筑级监督，使用 masked mean+max pooling、三分类 median-frequency CE，并把实例等级重新栅格化为像素输出。C2 使用配对 SAR，C3 使用同事件固定 deranged SAR。

已知事实：

1. 14/14 个训练和评估任务机械完成，队列无进程级失败。
2. 14/14 最终 train/val loss 为 NaN。
3. 每个 checkpoint 的 24,462,084 个状态元素中有 24,461,603 个 NaN，232/280 个状态张量受影响。
4. NaN 最早在 250–750 optimizer steps 之间出现。
5. E1-C2 与 E1-C3 在所有事件上都只预测 Intact，Damaged/Destroyed/binary damage F1 都为 0；七组 per-event CSV 两两逐字节相同。
6. 30-step 两批次 overfit 曾通过，500-step throughput 也通过，但现有 completion checker 不检查非有限 loss、logits、metrics 或 checkpoint tensors。
7. 本实验只有 14 个已暴露开发事件，没有独立盲测。不要提出任何 unseen-event robustness 或 deployment claim。

请进行证据驱动、逐文件逐行的审查。不要仅根据上述摘要猜测；每个代码问题都要给出文件、行号、触发路径和可验证依据。不要把“实现发生 NaN”偷换为“实例级监督理论无效”，也不要建议无约束地扫 backbone、loss 权重或大量超参数。

请重点检查：

A. 数值失效根因
- AMP/autocast 和 GradScaler 的使用是否正确；scheduler 是否可能在 optimizer step 被跳过时仍推进。
- pool_component_features 中 scatter_reduce_(amax)、mean/max 拼接、FP16 累加、空组件和重复最大值的前向/反向行为。
- instance logits 的形状变换、标签范围、BuildingOnlyGradeLoss 的 target-1 逻辑和 class weight 应用。
- rasterize_instance_logits 对背景/无组件区域的零 logits，以及 NaN logits 进入 argmax 的行为。
- BatchNorm running statistics、梯度爆炸、非有限 loss 后 optimizer/scaler 状态传播的可能路径。
- 为什么短 overfit 有限，但正式数据流在 250–750 step 系统性发散。

B. 实现正确性
- OOF prior 组件是否严格使用 4 连通、最小面积 4、纯度 0.8。
- 组件图缩放到 decoder feature resolution 后，component ID、pooled feature、instance logits 和回填 pixel index 是否一一对应。
- 歧义组件和 false-positive prior 组件是否被正确排除出实例损失，同时仍按协议保留在像素评估中。
- 训练、验证、C2/C3 permutation 和 SAR metadata 是否存在泄漏、错位或标签污染。
- 每个组件是否恰好贡献一次 CE，还是由于形状/掩膜造成像素或组件重复加权。

C. 实验编排与 fail-closed 条件
- training_complete、evaluation_complete 和总队列完成标记为何会接受 NaN 模型。
- 应在哪些位置加入 torch.isfinite 检查，才能在首次异常时保留诊断信息并安全终止。
- evaluator 是否应在 argmax 前拒绝非有限 logits。
- 如何设计 1000-step 非计分 stability smoke，既能定位故障，又不利用外层验证结果调参。
- 当前 summarizer 是否真正从原始评估文件聚合指标，还是只对人工准备的 payload 判门槛；这是否可能引入报告错误。

D. 指标和实验方法学
- event-macro damage、Damaged、Destroyed、grade、fold-pooled overall damage 的聚合定义是否一致、可复现且无 Simpson/pooled-score 掩盖。
- C2-C3 为 0 是否只是共同崩溃的结果，而不是 SAR correspondence 缺失的证据。
- 14 个已暴露事件、单 seed、无盲测条件下，哪些主张可以成立，哪些必须禁止。
- 当前 E1 应定性为 valid negative result、technical failure 还是二者之一，并说明理由。
- 现有 30-step overfit 和 500-step throughput 前置检查为什么不足。

E. 可复现性与仓库证据
- 检查 resolved config、seed、selected steps、C3 mapping、OOF prior、代码/dirty diff hash、数据审计和结果报告是否足以复核。
- 检查 Git 中是否错误包含数据、checkpoint、日志、绝对路径、秘密或不可复现的生成物。
- 检查报告与 experiments/registry.json 是否一致。

输出必须采用以下结构：

1. Executive Verdict
   - 用一句话给出结论。
   - 分别评价“运行完成性”“数值有效性”“科学可解释性”“主张边界”。

2. Ranked Findings
   - 按 P0/P1/P2/P3 排序。
   - 每项包含：标题、文件与精确行号、证据、影响、最小修复、修复后测试。
   - P0/P1 必须区分“已证实根因”和“高可信候选根因”。

3. Numerical Failure Causal Tree
   - 从首个非有限 loss/gradient/logit/parameter 到 checkpoint 污染和 all-Intact 指标的完整传播链。
   - 对每个分支给出置信度和能证伪它的最小实验。

4. Experimental Design Audit
   - 逐项检查单因素、锚点、C2/C3、OOF、事件宏平均、最坏事件门槛、单 seed 和 exposed-development 边界。
   - 明确 11 类统计/方法学谬误是否存在，覆盖必须为 11/11。

5. Claim Audit
   - 以表格列出每个可能主张及 ALIGNED / OVERSTATED / NOT_SUPPORTED_BY_PROVENANCE / PROVENANCE_INSUFFICIENT。

6. Minimal Recovery Plan
   - 给出不超过 6 步的恢复方案。
   - 第一步必须是 fail-closed finite checking。
   - 在定位前不得启动 14-run 重跑或 63-run full。
   - 如果修复改变 LR/loss/pooling/supervision 等科学配方，要求新协议版本；数值等价修复才允许沿用原 E1 协议。

7. Required Tests
   - 列出具体单元测试、CUDA smoke、1000-step stability test 和 checkpoint finite audit。
   - 每个测试给出明确 pass/fail 条件。

8. Uncertainties
   - 明确哪些问题仅凭仓库无法确定，需要哪些最小额外 artifact。

请避免泛泛建议。若证据不足，请写“不确定”，并给出最小验证方法，不要编造根因。
```
