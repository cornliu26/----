# 第 1 课：Fat Row 与长序列数据墙

## 1. 本课定位

这一课回答一个最基础的问题：

为什么推荐模型想继续扩长用户历史时，训练数据平台会先成为瓶颈？

论文不是从模型结构开始讲，而是从训练数据样本的物理组织开始讲。这一点非常重要。很多长序列讨论默认“数据已经在那里”，但工业系统里，数据能不能以可承受的成本被写入、存储、读取、预处理，直接决定模型能不能继续 scaling。

建议时长：1 到 2 小时。

## 2. 原文短句与翻译

> "storage and I/O wall"

中文翻译：存储和 I/O 墙。

这里的“墙”指的是一个系统扩展边界：序列继续变长时，数据支撑服务的资源消耗增长到足以压过 GPU 训练本身，导致模型侧还有收益，数据侧却已经撑不住。

## 3. 先理解 UIH 为什么越来越长

UIH 是 User Interaction History，也就是用户历史行为序列。推荐模型用 UIH 来理解用户长期兴趣、短期意图和行为模式。

早期推荐模型可能只看几十个行为。后来 candidate-dependent search、DIN、SIM、ETA 等方法让历史窗口变长。再往后，HSTU 一类序列推荐模型把推荐问题改写成序列建模问题，让模型对更完整的用户历史做建模。

这背后的直觉是：

```text
更长历史
  -> 更多兴趣证据
  -> 更好地区分长期偏好和短期意图
  -> 更可能提升推荐质量
```

但这条链路只讲了模型收益，没有讲数据代价。

## 4. 什么是 Fat Row

Fat Row 是一种训练样本组织方式：每条训练样本里都预先物化完整特征。

可以把一条样本想象成：

```text
training_example:
  sample_id
  request_id
  user_id
  candidate_id
  request_context
  label
  user_profile_features
  candidate_features
  complete_uih_sequence
```

它的优点是训练时简单。Trainer 或 data loader 只要顺序读训练样本，就能拿到所有输入特征。

它的缺点也很直接：如果 `complete_uih_sequence` 很长，每条样本都复制一份，就会产生巨大重复。

## 5. K-fold amplification 是怎么来的

假设：

- 用户 U 一天有 K 次推荐请求。
- 模型需要最近 N 天历史。
- 用户前 N-1 天的大部分历史，在今天这 K 次请求中几乎不变。

Fat Row 下，这段几乎相同的历史会被复制进 K 条训练样本。

```text
request_1 -> copy long UIH
request_2 -> copy long UIH
request_3 -> copy long UIH
...
request_K -> copy long UIH
```

这就是 K-fold amplification。

这里的 K 不是一个抽象常数。推荐产品里高活用户请求频次很高，训练样本又来自大量用户和大量请求，这种重复会被全平台放大。

## 6. 为什么压缩不能根治问题

很多人会第一反应说：那把 UIH 压缩一下不就好了？

压缩当然有用，但它只能减少每一份 UIH 的大小，不能消除跨样本重复。

对比一下：

```text
压缩：
  每份 UIH 从 100 KB 降到 30 KB
  但仍然复制 K 份

消除重复：
  UIH 只存一份
  每条样本只存版本元数据
```

长序列时代，问题的核心不再是单条记录太宽，而是同一段历史被重复物理写入太多次。

## 7. 数据墙不只是存储墙

论文说的是 storage and I/O wall，不只是 storage wall。

因为 Fat Row 放大会影响整条链路：

| 环节 | Fat Row 带来的问题 |
| --- | --- |
| 样本写入 | 每条样本都写完整 UIH，写带宽放大 |
| 样本存储 | 长历史重复落盘，存储体积放大 |
| 训练读取 | 每个训练任务都读完整 UIH，读带宽放大 |
| 数据预处理 | 更大的样本需要更多 CPU decode 和搬运 |
| 网络传输 | 从存储到 worker 的数据量放大 |
| 多租户平台 | 短序列模型也可能被迫读长序列 payload |

所以这不是“多买点存储”就能解决的问题。它会让训练数据支撑服务的资源消耗超过 GPU 训练本身。

## 8. 为什么多租户会放大问题

成熟推荐平台通常会维护 union training dataset，供多个模型租户共用，例如：

- retrieval model
- pre-ranking model
- ranking model
- late-stage ranking model

不同模型需要的 UIH 长度不同。召回模型可能只需要最近 100 个行为，精排模型可能希望使用 10K 或 16K 行为。

Fat Row 下，union dataset 往往要按最大需求物化。结果是：

```text
长序列模型需要 16K
union dataset 写入 16K
短序列模型只需要 100
但仍然可能读取或搬运 16K payload
```

这就是 multi-tenant penalty。

## 9. 和 partial dump 的直接关系

如果你们内部说 partial dump，它要解决的第一类问题很可能就是：

```text
哪些重特征不应该继续完整 dump 到每条样本？
```

在这篇论文里，答案是 long-term UIH。

更一般地说，适合 partial dump 的字段通常有几个特征：

1. 字段很大。
2. 跨样本重复度高。
3. 可以通过 user_id、时间边界、版本号等信息重建。
4. 不需要每条样本都物理保存完整内容。

不适合 partial dump 的字段通常是：

1. label。
2. request-time candidate set。
3. request context。
4. 极易变化且难以按版本重建的 recent features。

## 10. 本课检查点

读完这一课，你应该能回答：

1. Fat Row 为什么在短序列时代合理？
2. K-fold amplification 是怎么产生的？
3. 为什么长 UIH 会让数据服务资源超过 GPU 训练资源？
4. 为什么压缩不能根治跨样本重复？
5. 多租户 union dataset 为什么会放大 Fat Row 问题？

## 11. 课后练习

请拿你们自己的训练样本字段做一次盘点：

| 字段 | 平均大小 | P99 大小 | 是否跨样本重复 | 是否可以按版本重建 | 是否适合 partial dump |
| --- | ---: | ---: | --- | --- | --- |
| UIH item id | | | | | |
| UIH timestamp | | | | | |
| 行为类型 | | | | | |
| item side info | | | | | |
| request context | | | | | |
| label | | | | | |

然后写一段结论：如果要先做一个 partial dump PoC，你会优先拆哪个字段，为什么？

## 12. 拓展阅读

1. 论文 Section 1：看作者如何从长序列 scaling 引出数据基础设施瓶颈。
2. 论文 Section 2.2：重点看 Storage and I/O Wall。
3. 论文 Section 2.3：重点看 Multi-Tenant Penalty。
4. 如果想补模型背景，可以再看 HSTU / ULTRA-HSTU 的长序列动机，但本课重点仍然是数据链路。

