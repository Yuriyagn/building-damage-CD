# Stage-2 relaxed spatial-overlap audit

日期：2026-06-22  
协议：`relaxed-overlap-v1`  
状态：legacy 与 strict-v1 自动审计已完成；strict-v1 唯一 train-test medium 必审候选已判定为 no-data 误报；overlap-reviewed manifest 已生成并通过完整内容/event-disjoint 复审。

## 1. 目的与约束

在完整 SHA-256 和 canonical-event-disjoint 之外，检测平移 tile、部分裁剪、轻微重采样和同一大图局部重叠。

本轮按用户指定的 relaxed policy：

- train-test confirmed/likely overlap 必须排除一侧；默认保留 test、删除 train 侧；
- train-val 和 val-test 疑似重叠允许保留，但必须审核记录风险；
- 不得将结果表述成三个 split 形式化空间独立。

## 2. 方法

自动候选由以下无外部模型、可复现的证据联合生成：

1. pre optical、post SAR、mask 的 pHash/dHash；
2. 64×64 灰度低频 DCT 全局描述子；
3. 4×4 patch DCT 描述子和 FLANN patch 检索；
4. pre optical ORB 局部描述子全局检索；
5. ORB-Hamming 匹配与 affine RANSAC 几何内点验证。

综合风险阈值为 high ≥ 0.85、medium ≥ 0.70、low ≥ 0.60。几何一致内点可以覆盖全局相似度因平移/裁剪下降造成的漏检。自动等级只决定人工复核优先级。

## 3. 方法校准

从 legacy manifest 固定 seed `20260622` 抽取 12 张图，生成72个正例：中心裁剪、平移128/256像素、缩放0.8、亮度变化和JPEG压缩；另生成12个左右不相交半图负例。

| 指标 | 结果 |
| --- | ---: |
| global retrieval recall@20 | 0.9167 |
| global retrieval recall@8 | 0.8194 |
| high-risk recall（进入候选后的评分） | 1.0000 |
| medium-or-high recall | 1.0000 |
| adjacent non-overlap medium/high rate | 0.0000 |

记录：`outputs/stage2/overlap_audit_20260622/validation/synthetic_validation.json`。样本量有限，该校准证明实现能找回预设变换，但不能证明真实数据中无漏检。

## 4. 自动审计结果

### 4.1 legacy

| Pair | High | Medium | Low | 合计 |
| --- | ---: | ---: | ---: | ---: |
| train-test | 106 | 6 | 0 | 112 |
| train-val | 0 | 1 | 0 | 1 |
| val-test | 0 | 1 | 1 | 2 |

legacy train-test 的106个 high 中，105对是完整 SHA-256 审计已确认的 exact duplicate；剩余1个 high 和6个 medium 是部分重叠人工候选。该结果复现并扩展了旧 split 污染证据，不恢复任何旧 test 结论。

### 4.2 strict-v1

| Pair | High | Medium | Low | 合计 |
| --- | ---: | ---: | ---: | ---: |
| train-test | 0 | 1 | 1 | 2 |
| train-val | 0 | 4 | 0 | 4 |
| val-test | 0 | 0 | 0 | 0 |
| train-train | 1 | 8 | 0 | 9 |
| val-val | 1 | 1 | 0 | 2 |
| test-test | 0 | 4 | 2 | 6 |

strict-v1 没有 train-test high 候选。唯一必审 medium 候选为：

```text
pair_id: cd4d2f3d4f3875f18b00
train: strictv1__train__myanmar_hurricane_00000116
test:  strictv1__val__rwanda_volcano_00000122
score: 0.7325
```

该候选由大面积旋转 no-data 区域造成较高全局/patch 相似度，ORB affine 在尺度/条件数校验后没有有效几何内点。用户确认该候选为误报后，已记录 `not_overlap + keep_both_record_only`：两侧样本均保留，不删除任何样本。

首轮自动评分曾产生4个额外 strict train-test medium 假候选。检查发现其 RANSAC affine 退化为接近零尺度；实现已增加 singular-value 和 condition-number 校验，并完整重跑 legacy/strict-v1。上表是修复后的最终机器结果。

机器结果分别位于：

- `outputs/stage2/overlap_audit_20260622/legacy/`
- `outputs/stage2/overlap_audit_20260622/strict_v1/`

## 5. 人工复核与 manifest

FastAPI 应用：`apps/overlap_review/`。界面支持 legacy/strict-v1 切换、左侧 High/Medium/Low 与审核状态统计、中央连续 Train/Test 对照、`1`–`6` 快捷判定和显式导出 reviewed manifest。

strict-v1 train-test 必审复核进度：`2/2`（medium 与 low 均为 `not_overlap + keep_both_record_only`）。2026-06-24 整合用户 legacy 复核后，正式训练 manifest 统一为 `manifests/stage2_v2_clean_human_reviewed_20260624/`，`removed_unique_sample_count=0`。完整 SHA-256/event-disjoint 复审为 0 hard errors、0 完整重复组；3 个 warning 仅来自缺失 `qc_label`。复审记录：`outputs/stage2/clean_human_reviewed_20260624/full_audit.json`。严格 contact sheet：

`outputs/stage2/overlap_audit_20260622/strict_v1/contact_sheets/0001_medium_cd4d2f3d4f3875f18b00.jpg`

## 6. 限制

minimal package 为无地理坐标的 PNG。基于图像的候选检索不是空间 footprint 相交证明；“未检出候选”只能表示在本方法和阈值下未检出。若恢复原始 GeoTIFF、scene ID 或 tile offset，应优先追加基于地理 footprint 的确定性审计。

synthetic 全局检索 recall@20 为0.9167而非1.0，这也是不能宣称形式化 spatial-disjoint 的直接证据。ORB/patch 候选补充了全局检索，但真实数据的漏检率无法在无地理真值时精确估计。
