# 第 2 课：Semantic ID 与 Item Tokenization

## 1. 本课定位

这一课讲 PLUM 最核心的中间层：Semantic ID。没有 SID，LLM 无法直接生成真实 item；SID 质量不好，生成式召回上限也会被锁死。

建议时长：2 到 3 小时。

## 2. 原文短句与翻译

> "Semantic IDs"

中文翻译：语义 ID。

它不是随机 item id，也不是普通 embedding，而是由 item 内容特征量化得到的一组离散 codeword token。

## 3. SID 是什么

可以把 SID 理解成：

```text
video/item
  -> content embeddings
  -> dense semantic embedding
  -> residual quantization
  -> <sid_1, sid_2, ..., sid_L>
```

它有三个目标：

1. 让 item 能进入 LLM 词表。
2. 让模型能生成 item。
3. 保留足够推荐语义，支持召回和泛化。

## 4. 为什么不能直接用 item id

直接把原始 item id 加进词表会遇到几个问题：

- ID 本身无语义，LLM 无法利用内容相似性。
- 长尾 item 学不到足够 embedding。
- 新 item 冷启动困难。
- 词表和输出空间巨大。
- 生成式模型难以从 ID 前缀中获得层次语义。

SID 的作用是把 item 从“任意编号”变成“有语义结构的 token 序列”。

## 5. SID-v2：多模态内容融合

论文认为单一内容 embedding 不够。视频的语义来自标题、视觉、音频、ASR、频道、topics 等多个来源。

PLUM 的做法是：

```text
embedding_1 -> encoder_1 -> z_1
embedding_2 -> encoder_2 -> z_2
...
concat(z_1, z_2, ...)
  -> projection
  -> unified item representation
  -> RQ-VAE
```

工程含义：

- 需要确认有哪些模态已经 ready。
- 需要处理缺失模态。
- 需要管理 embedding 版本和刷新频率。
- tokenization pipeline 依赖上游多模态特征平台。

## 6. SID-v2：Multi-resolution codebooks

传统 RQ-VAE 如果每层 codebook 大小一样，SID 组合空间可能很稀疏。

PLUM 使用 multi-resolution codebooks：

```text
前层 codebook：分辨率高，负责强区分
后层 codebook：分辨率低，负责残差细化
```

直觉上，SID 前缀越能表达大类语义，生成式解码越稳定。

## 7. SID-v2：Progressive masking

Progressive masking 训练时随机只保留前 r 层 SID code。它迫使前缀 code 学到有意义的层次信息。

如果没有这个机制，模型可能把太多信息塞到后面 code，导致前缀不稳定、层次不可解释。

对 beam search 来说，SID 前缀质量很重要，因为解码是自回归的。

## 8. SID-v2：Co-occurrence contrastive regularization

内容相似不等于推荐相关。用户经常连续观看的视频，即使内容表面不同，也可能在推荐语境里相关。

PLUM 把行为共现信号加到 SID 训练中：

```text
经常共现的视频
  -> SID 表示更接近

不共现的视频
  -> SID 表示更远
```

这个设计非常关键。Table 4 显示，去掉 co-occurrence 后 SID uniqueness 从 96.7% 降到 91.8%，VID Recall@10 从 14.4% 降到 12.6%。

## 9. 生产系统需要什么

一个 SID 平台至少包括：

- 多模态 embedding 输入。
- SID tokenizer 训练。
- SID vocabulary。
- item -> SID 映射。
- SID -> item 映射。
- collision 处理。
- 新 item / 下架 item 处理。
- SID 版本管理。
- 下游模型和服务的兼容策略。

## 10. 常见误区

误区 1：把 SID 当成简单 ID 压缩。

SID 是推荐语义表示，不只是缩短 ID。

误区 2：只用内容，不用行为。

内容信号适合冷启动和语义理解，但推荐相关性还需要行为共现。

误区 3：忽略 SID-to-item collision。

如果多个 item 映射到同一个 SID，生成模型预测 SID 后还要决定返回哪个 item。论文用 SID uniqueness 和 VID Recall 来评估这个问题。

## 11. 本课检查点

1. SID 和 item id embedding 有什么不同？
2. 多模态内容融合解决什么问题？
3. Multi-resolution codebooks 的直觉是什么？
4. Progressive masking 为什么有助于层次结构？
5. Co-occurrence contrastive 为什么重要？

## 12. 课后练习

设计你们业务的 SID readiness 表：

| 模态/信号 | 是否已有 | 刷新频率 | 缺失率 | 是否进入 SID | 风险 |
| --- | --- | --- | ---: | --- | --- |
| 标题/文本 | | | | | |
| 图片/视频 embedding | | | | | |
| ASR/OCR | | | | | |
| 行为共现 | | | | | |
| 作者/频道 | | | | | |

## 13. 拓展阅读

1. 论文 Section 2.1：Semantic IDs。
2. 论文 Table 4：SID-v2 ablation。
3. RQ-VAE / residual quantization 的基础介绍。

