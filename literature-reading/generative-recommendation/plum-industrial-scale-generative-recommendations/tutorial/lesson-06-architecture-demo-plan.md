# 第 6 课：架构 Checklist 与 Demo 计划

## 1. 本课定位

最后一课把 PLUM 转成你可以和团队讨论的方案语言：要做什么、先验证什么、哪些链路要准备、哪些风险要提前问。

建议时长：2 小时。

## 2. 原文短句与翻译

> "industrial-scale"

中文翻译：工业级规模。

PLUM 的价值不是只在公开小数据集上验证生成式推荐，而是讨论如何在 YouTube 这种大规模推荐系统里训练、服务和上线。

## 3. 第一版 demo 不要贪大

一个合理的第一版 demo 目标是：

```text
验证生成式召回链路能跑通，
并判断 SID / 多模态 / CPT 哪个模块最值得继续投入。
```

不要一开始就试图复刻完整 YouTube PLUM。

## 4. 最小 PoC 路线

建议分四阶段：

### 阶段 1：样本和目标验证

- 用现有 U 聚合样本构造 prompt。
- 定义 next-click / next-watch / long-watch target。
- 把 target item 映射到 SID。
- 先离线跑通训练样本生成。

验收：

- prompt 字段稳定。
- target SID 可生成。
- 样本量足够。
- 没有明显 label leakage。

### 阶段 2：简化 SID baseline

- 先不完整复现 SID-v2。
- 可以先用内容 embedding + 简化量化。
- 记录 collision 和 coverage。

验收：

- item -> SID 和 SID -> item 可反查。
- uniqueness 可接受。
- 新 item 处理路径清楚。

### 阶段 3：小模型 generative retrieval

- 用小模型训练 SID prediction。
- beam search 生成 top-K。
- 计算 Recall@K、invalid SID、collision。

验收：

- 生成链路可跑。
- Recall 相对 baseline 有初步信号。
- latency 有估算。

### 阶段 4：补强模块

按瓶颈加入：

- 多模态 embedding。
- co-occurrence contrastive。
- CPT。
- 更大模型。
- 线上候选融合。

## 5. 数据 readiness checklist

| 数据/特征 | 要问的问题 |
| --- | --- |
| 用户行为序列 | 粒度是什么？是否能构造 next-item target？ |
| 多模态 embedding | 哪些 ready？刷新频率？缺失率？ |
| item metadata | title、ASR、channel、topics 是否齐全？ |
| label/reward | click、watch、satisfaction 怎么取舍？ |
| SID mapping | 如何版本化？如何处理 collision？ |
| 训练语料 | 是否能支持 CPT？是否能复用 checkpoint？ |

## 6. 服务 checklist

| 模块 | 关键问题 |
| --- | --- |
| prompt builder | 线上字段是否齐全？顺序和训练一致吗？ |
| decoder | 模型大小和 latency budget 是否匹配？ |
| beam search | beam size 多大？候选数多少？ |
| SID validation | invalid SID 如何处理？ |
| SID lookup | collision 如何决策？mapping 如何更新？ |
| candidate merge | 和现有召回源如何去重、配额、归因？ |
| fallback | 生成失败或超时时回退到什么？ |
| monitoring | 监控 recall proxy、latency、invalid、coverage、CTR？ |

## 7. 团队讨论问题

1. 当前阶段最想验证的是 SID、多模态、CPT，还是 generative retrieval 效果？
2. 第一版 demo 是否需要 CPT，还是先做 SFT baseline？
3. 多模态 embedding 是否已经稳定可用？
4. prompt schema 是否能从现有样本直接构造？
5. target label 是 click、watch、long watch，还是 satisfaction-weighted？
6. SID collision 和 invalid SID 的容忍度是多少？
7. 生成式召回是补充召回源还是替换召回源？
8. 如果线上不挂，离线 demo 成功标准是什么？

## 8. 一页方案模板

```text
标题：
  PLUM-like Generative Retrieval Demo

目标：
  验证用 Semantic ID + LLM 生成式召回是否能在当前业务数据上产生增量候选。

范围：
  先选一个业务面/一个 corpus/一个行为目标。

数据：
  用户行为序列、item metadata、多模态 embedding、label/reward。

方法：
  item tokenization -> prompt construction -> SID prediction -> beam decoding -> SID lookup。

指标：
  Recall@K、invalid SID、collision、coverage、latency、与现有召回互补率。

风险：
  SID 质量、CPT 成本、服务延迟、mapping 版本、样本目标定义。

下一步：
  如果离线可行，再加多模态增强、CPT 和候选融合实验。
```

## 9. 常见误区

误区 1：一开始就追求完整 PLUM。

完整 PLUM 涉及 SID-v2、CPT、MoE、beam serving、线上融合。第一版更应该验证关键假设。

误区 2：只让算法同学推进。

PLUM 会改动数据、样本、特征、服务、监控和评估，架构同学必须参与。

误区 3：demo 只看 Recall。

生成式召回还要看 invalid SID、collision、latency、coverage 和候选互补。

## 10. 本课检查点

1. 第一版 demo 应该验证哪个最小闭环？
2. 哪些数据是 PLUM-like 方案启动前必须 ready 的？
3. 为什么 SID mapping 要版本化？
4. 为什么生成式召回适合先做补充源？
5. 如何把论文实验指标转成你们业务的 PoC 指标？

## 11. 课后练习

写一页 demo proposal，包含：

- 目标。
- 数据范围。
- SID 方案。
- prompt/target schema。
- 模型训练路径。
- 离线指标。
- 服务风险。
- 下一步计划。

## 12. 拓展阅读

1. 论文 Conclusion and Future Work。
2. 论文 Section 3.1 的 live experiment 设计。
3. 当前业务的召回链路文档，用来做 gap analysis。

