# 第 4 课：Generative Retrieval 的训练、解码与服务

## 1. 本课定位

这一课讲 PLUM 如何真正做召回：把用户上下文组织成 prompt，让模型自回归生成下一个视频的 SID。

建议时长：2 到 3 小时。

## 2. 原文短句与翻译

> "beam search"

中文翻译：束搜索。

在 PLUM 里，beam search 用来生成多条候选 SID 序列，这些序列再反查成真实视频，形成候选集。

## 3. 训练样本长什么样

论文中的 prompt 近似是：

```text
watch history | user features | context video features
```

watch history 不是原始 item id 列表，而是 SID tokens 和其他行为特征 token 的组合。

目标是：

```text
ground-truth clicked / next-watch video 的 SID tokens
```

也就是说，训练任务是 next-token prediction，只不过预测的是 item SID。

## 4. Reward-weighted objective

论文里 SFT 目标包含 handcrafted reward signal。直觉是，不是所有点击都一样重要，样本权重要反映用户体验目标。

实践中因为训练成本高，论文采用：

```text
先按 reward 采样训练样本
再对采样后的样本等权训练
```

对业务落地来说，这意味着 label 定义非常重要：

- click
- watch time
- watch ratio
- satisfaction
- long watch
- negative feedback

不同目标会让生成式召回学到不同偏好。

## 5. 推理链路

PLUM 推理大致是：

```text
build prompt
  -> decoder-only LLM
  -> beam search generate K SID sequences
  -> validate SID
  -> SID-to-video lookup
  -> candidate set
  -> merge with other recall sources
```

服务上新增了几个核心问题：

| 问题 | 解释 |
| --- | --- |
| invalid SID | 模型生成了不存在的 SID |
| collision | 一个 SID 映射到多个 videos |
| latency | 自回归解码比一次向量检索更重 |
| diversity | beam search 容易偏向高概率相似候选 |
| fallback | 生成失败时要回退到其他召回源 |
| versioning | SID mapping 必须和模型版本匹配 |

论文提到 SFT 后 hallucination rate 很低，小于 5%，但生产系统仍必须监控。

## 6. Beam size 的取舍

Beam size 越大，可能覆盖更多候选，但延迟和计算成本也更高。

```text
larger beam:
  higher candidate coverage
  more decoding cost
  possible lower diversity

smaller beam:
  lower latency
  fewer candidates
  easier to serve
```

这不是纯算法选择，而是服务预算选择。

## 7. 与现有召回源融合

论文 live experiment 的方式不是直接替换 LEM，而是把 PLUM 推荐加入 candidate pool，并与增加 LEM+ quota 做对比。

这对 demo 很有启发：

```text
第一阶段不要想着替换主召回。
先作为补充召回源验证 unique value。
```

需要设计：

- 候选配额。
- 去重。
- 归因。
- 与排序模型的兼容。
- 回退策略。

## 8. 常见误区

误区 1：只把 prompt 当成字符串拼接。

Prompt schema 是训练/服务契约，字段顺序、token 类型、缺失值、版本都要稳定。

误区 2：只看生成 top-1。

召回需要候选集，top-K SID、beam diversity 和覆盖率都重要。

误区 3：忽略 SID mapping 版本。

模型生成的 SID 必须和线上 mapping 表同版本，否则会产生不可控候选。

## 9. 本课检查点

1. Generative retrieval 的输入和输出分别是什么？
2. 为什么 label/reward 定义会影响召回候选？
3. Beam search 为什么进入推荐服务链路？
4. Invalid SID 和 collision 分别是什么？
5. 为什么先作为补充召回源更稳？

## 10. 课后练习

设计一个 PLUM-like serving checklist：

| 项目 | 你的设计 |
| --- | --- |
| prompt 字段 | |
| beam size | |
| 返回候选数 | |
| invalid SID 处理 | |
| collision 处理 | |
| fallback | |
| 与现有召回融合 | |
| 监控指标 | |

## 11. 拓展阅读

1. 论文 Section 2.3：Generative Retrieval。
2. 论文 Section 3.1：Performance of Generative Retrieval。
3. YouTube 推荐链路相关论文，用来理解 candidate generation 位置。

