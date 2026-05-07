# 第 3 课：Continued Pre-training 与推荐域适配

## 1. 本课定位

SID 解决了 item 如何变成 token。CPT 解决的是 LLM 如何理解这些新 token，以及如何学习推荐域里的用户行为语言。

建议时长：2 小时。

## 2. 原文短句与翻译

> "continued pre-training"

中文翻译：继续预训练。

CPT 不是最终推荐任务的 SFT，而是先把基础 LLM 适配到推荐域，让 SID、用户行为和自然语言进入同一个表示空间。

## 3. 为什么不能直接 SFT

预训练 LLM 的知识主要来自自然语言和通用多模态数据。它不天然懂：

- SID token 是什么。
- `<sid_1> <channel_name> <watch_ratio>` 这样的序列语法。
- YouTube 视频标题、ASR、频道、topics 和 SID 的关系。
- 用户 watch history 中的兴趣迁移模式。

如果直接 SFT，模型既要学推荐任务，又要学推荐域 token 语言，训练压力很大。

## 4. CPT 的两类语料

论文使用两类主要数据，各占 50%：

```text
User behavior data:
  watch history
  channel name
  watch ratio
  watch time
  time since watch
  SID sequence

Video metadata corpus:
  SID
  title
  description
  ASR captions
  channel name
  topics
  synthetic data
```

这让模型同时学会：

- SID 和文本的对应关系。
- 用户行为序列的模式。
- SID 作为一种新 modality 如何和自然语言共存。

## 5. CPT 为什么是平台底座

CPT 的输出不是一个单任务模型，而是一个推荐域 base checkpoint。

它可以被多个下游任务复用：

```text
CPT checkpoint
  -> generative retrieval SFT
  -> ranking SFT
  -> personalized search
  -> metadata understanding
```

因此架构讨论里不要只问“CPT 一次训练贵不贵”，还要问：

```text
这个 checkpoint 能不能被多个任务摊销？
```

## 6. 论文结果怎么证明 CPT 有用

Table 5 做了 2x2 ablation：

| Model | Pre-trained LLM | CPT | Recall@10 |
| --- | --- | --- | ---: |
| R1 | No | No | 0.19 |
| R2 | Yes | No | 0.23 |
| CR1 | No | Yes | 0.27 |
| CR2 | Yes | Yes | 0.28 |

结论：

- CPT 从 0.19/0.23 提升到 0.27/0.28，是主要增益来源。
- LLM init 也稳定有帮助。
- 最完整路径是 LLM init + CPT + SFT。

论文还展示 CPT 后模型收敛更快，这意味着 CPT 不只提升最终效果，也提升下游 SFT 训练效率。

## 7. CPT 的工程要求

要落地 CPT，需要准备：

- SID vocab expansion。
- 训练样本 mixture。
- 用户行为序列构造。
- 视频元数据语料构造。
- synthetic data 生成策略。
- 通用文本能力回归评估。
- checkpoint 版本管理。
- 下游任务复用计划。

这不是算法同学单独能完成的事情。数据平台、样本链路、特征生产和训练平台都要参与。

## 8. 常见误区

误区 1：把 CPT 当作可选优化。

在 PLUM 结果里，CPT 是核心组件，不是锦上添花。

误区 2：只构造用户行为，不构造 metadata。

Metadata 语料让 SID 和自然语言对齐。没有它，LLM 的语言知识很难进入推荐 item 表示。

误区 3：只看最终 Recall，不看收敛速度。

CPT 的价值包括让多个下游 SFT 更快收敛。

## 9. 本课检查点

1. CPT 和 retrieval SFT 有什么区别？
2. 为什么 SID token 加入词表后还需要训练对齐？
3. User behavior data 和 video metadata corpus 各自解决什么问题？
4. Table 5 如何证明 CPT 有价值？
5. 为什么 CPT checkpoint 适合做平台底座？

## 10. 课后练习

写一份 CPT 数据 readiness memo：

| 数据源 | 是否已有 | 样本粒度 | 质量风险 | 可否进入 CPT |
| --- | --- | --- | --- | --- |
| watch history | | | | |
| video title | | | | |
| ASR caption | | | | |
| channel name | | | | |
| topics | | | | |
| synthetic metadata | | | | |

## 11. 拓展阅读

1. 论文 Section 2.2：Continued Pre-training。
2. 论文 Section 3.3：Impact of Continued Pre-training。
3. Appendix A.2：CPT 后的 in-context learning 示例。

