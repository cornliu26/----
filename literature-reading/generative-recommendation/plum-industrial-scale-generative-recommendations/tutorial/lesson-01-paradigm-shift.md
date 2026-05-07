# 第 1 课：从 Embedding Retrieval 到 Generative Retrieval

## 1. 本课定位

这一课先建立全局图：PLUM 不是在传统推荐模型旁边加一个 LLM，而是把召回从 embedding matching 改写成 token generation。

建议时长：1 到 2 小时。

## 2. 原文短句与翻译

> "generative retrieval"

中文翻译：生成式召回。

这里的生成不是生成自然语言答案，而是生成目标 item 的 Semantic ID token 序列，再把 SID 反查成真实视频。

## 3. 传统 LEM 范式

工业推荐长期依赖 Large Embedding Models, LEM。

典型流程是：

```text
user features / item id / context
  -> embedding tables
  -> neural network
  -> user/item representation
  -> dot product or ANN retrieval
```

它的优势是成熟、稳定、吞吐高。它的问题是大量参数集中在 embedding table，对 item ID 的记忆很强，但对语义泛化、冷启动、多模态理解和更深网络 scaling 不够友好。

论文里一个很关键的对比是：LEM 的 neural network 只占总参数很小比例，而 PLUM 把大部分参数放在 neural network 本身。

## 4. PLUM 的范式变化

PLUM 的核心流程是：

```text
item content and behavior signals
  -> Semantic ID tokenization

user history + context features
  -> prompt

decoder-only LLM
  -> autoregressively generate target SID

SID-to-video mapping
  -> candidate videos
```

最大的变化是：

```text
以前：
  在 item 库里找相似向量

现在：
  让模型生成可能被推荐的 item token
```

这会同时影响训练、服务和评估。

## 5. 为什么这不是简单替换召回模型

如果只是换模型，数据和服务接口可以基本不动。但 PLUM 会新增一批系统组件：

| 组件 | 为什么需要 |
| --- | --- |
| SID tokenizer | 把 item 变成 LLM 可处理 token |
| SID vocabulary | 扩展 LLM 词表 |
| CPT corpus builder | 构造推荐域预训练数据 |
| prompt builder | 把用户历史和上下文组织成序列 |
| target SID builder | 把 label item 转成训练目标 |
| beam search service | 线上生成多个候选 SID |
| SID mapping service | 把 SID 反查成真实 item |
| monitoring | 监控 invalid SID、collision、latency、coverage |

因此它是推荐召回范式的迁移，不只是一个模型替换 PR。

## 6. 和推荐架构的关系

你可以把经典系统组件这样映射到 PLUM：

| 经典推荐 | PLUM |
| --- | --- |
| item id embedding | Semantic ID token |
| user tower / retrieval DNN | decoder-only LLM |
| ANN index | beam search + SID lookup |
| 样本 join | prompt / target SID builder |
| embedding refresh | SID mapping / vocab refresh |
| 召回源 | 生成式召回源 |

这个映射非常重要。它能帮助架构同学把论文里的算法词汇放回熟悉的服务链路里。

## 7. 常见误区

误区 1：以为 PLUM 是排序模型。

论文重点是 retrieval，尤其是 candidate generation。它未来可以扩展到 ranking 或 search，但本文核心不是排序替换。

误区 2：以为生成式召回一定更慢、成本更高。

单样本计算更重，但论文展示了 sample efficiency：900M MoE 每天训练约 250M 样本，而传统 LEM 每天训练 several billion 样本，总训练 FLOPs 反而小于 LEM 的 0.55x。

误区 3：只看 Recall。

生成式召回还必须看 invalid SID、collision、beam latency、候选覆盖率、和现有召回源的互补价值。

## 8. 本课检查点

1. LEM 的主要瓶颈是什么？
2. PLUM 为什么说召回从 matching 变成 generation？
3. 为什么 item tokenization 是 PLUM 的第一层系统组件？
4. Beam search 在召回链路里扮演什么角色？
5. 为什么生成式召回会新增服务监控指标？

## 9. 课后练习

画一张你们系统的对照图：

```text
current retrieval path
  vs
PLUM-like retrieval path
```

至少标出：

- 样本构造差异。
- item 表征差异。
- 线上服务差异。
- 监控指标差异。

## 10. 拓展阅读

1. 论文 Abstract 和 Introduction。
2. 论文 Section 1.1：Related Work。
3. 论文 Section 3.1：PLUM 与 LEM 的生产对比。

