# PLUM 论文精读大纲与学习路径

论文：PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations

本地文件：[`paper.pdf`](./paper.pdf)

一句话总结：

PLUM 不是简单地把 LLM 接到推荐系统上，而是把推荐召回改造成一个可由 LLM 处理的 token generation 问题：先把 item 变成 Semantic ID，再用推荐域数据做 continued pre-training，最后通过 generative retrieval 让模型直接生成目标 item 的 SID 序列。

## 1. 这篇文章到底在写什么

这篇文章来自 Google DeepMind / YouTube，目标是解释如何把预训练 LLM 适配到工业级推荐场景，重点落在 YouTube 的生成式召回。

它回答的是一个系统级问题：

```text
如果推荐系统不再主要依赖巨大 item embedding table，
而是让 LLM 根据用户上下文生成 item token，
那么 item 如何 token 化、LLM 如何理解推荐域、训练样本如何组织、线上如何解码和反查？
```

论文提出的框架叫 PLUM，由三阶段组成：

1. Item tokenization：把 item 编成 Semantic ID。
2. Continued pre-training, CPT：把 SID 和推荐域语料对齐到预训练 LLM。
3. Task-specific fine-tuning：针对推荐任务做 SFT，本文重点是 generative retrieval。

## 2. 为什么这篇文章重要

工业推荐长期依赖 Large Embedding Models, LEM。它们用巨大 embedding table 表示高基数离散特征，如 item ID、user ID、channel ID。

这种范式很有效，但有几个限制：

- 参数主要堆在 embedding table，而不是更深的神经网络。
- item ID embedding 更像记忆表，对冷启动和语义泛化不友好。
- 扩大 embedding table 需要大量训练样本支撑。
- 召回通常受 dot-product 检索形式限制。

PLUM 的方向是把复杂度从大 embedding table 转移到 LLM 的神经网络能力上：

```text
旧范式：
  item/user id -> embedding table -> dot product retrieval

PLUM：
  item -> Semantic ID tokens
  user context -> prompt
  LLM -> generate target SID tokens
```

这意味着推荐召回从“匹配”变成“生成”。

## 3. 这篇文章解决的核心挑战

### 挑战 1：LLM 不天然懂推荐域

预训练 LLM 懂自然语言，但没有在目标推荐域的用户行为序列、item corpus、视频元数据上训练过。

直接拿 off-the-shelf LLM 做推荐，会遇到 domain gap：

- 不懂 SID 是什么。
- 不懂用户 watch history 的模式。
- 不懂特定平台里 item 的细粒度质量差异。
- 不知道如何把视频标题、ASR、channel、行为信号和推荐目标对齐。

### 挑战 2：item 不是自然语言 token

LLM 处理 token，但推荐系统里的 item 是真实 corpus object。必须先解决：

```text
video/item -> token sequence -> model can generate -> map back to real video/item
```

这就是 Semantic ID 的位置。

### 挑战 3：Semantic ID 质量决定召回上限

如果 SID 不唯一、不稳定、不包含推荐语义，生成模型再强也会受限。

论文提出 SID-v2 来增强 SID：

- 多模态内容融合。
- multi-resolution codebooks。
- progressive masking。
- co-occurrence contrastive regularization。

### 挑战 4：生成式召回会引入服务问题

传统召回通常是 ANN 或 embedding dot product。PLUM 推理时要 beam search 生成 SID，再反查真实 item。

新问题包括：

- invalid SID / hallucination。
- SID-to-video collision。
- beam size 与延迟、候选多样性 tradeoff。
- SID 映射版本管理。
- 生成式召回和现有召回源如何融合。

### 挑战 5：工业级价值必须用线上和系统指标证明

论文不仅做离线 Recall，也给出 production comparison 和 live experiment：

- PLUM 对比高度优化的 LEM。
- PLUM 作为新增候选源叠加到现有系统。
- 评估 vocab size、CTR、watch metrics、satisfaction、sample efficiency。

## 4. PLUM 方法总览

PLUM 可以理解为一个三阶段训练和服务框架。

```text
阶段 1：Semantic ID tokenization
  多模态 item embeddings
    -> fusion encoder
    -> RQ-VAE / residual quantization
    -> SID token tuple

阶段 2：Continued Pre-training
  expanded vocab with SID tokens
    -> user behavior corpus
    -> video metadata corpus
    -> SID and text aligned in LLM checkpoint

阶段 3：Generative Retrieval SFT
  user context + history + features
    -> prompt
    -> autoregressive SID prediction
    -> beam search
    -> SID-to-video mapping
    -> candidate set
```

## 5. Semantic ID：item tokenization 的核心

论文里的短句：

> "Semantic IDs"

中文理解：语义 ID，也就是把 item 表示成一组离散 codeword token。

SID 的关键不是“给 item 一个新 ID”，而是让 item 进入 LLM 的 token 世界。

一个 SID 可以理解为：

```text
video -> dense semantic embedding -> hierarchical codewords -> <sid_1, sid_2, ..., sid_L>
```

相比传统 item ID，SID 有几个目标：

- 可被 LLM 输入和生成。
- 包含内容语义。
- 尽量和用户行为里的相关性对齐。
- 尽量减少 collision。
- 对冷启动和长尾 item 更友好。

## 6. SID-v2 的四个增强点

### 6.1 多模态内容融合

视频 item 的语义来自多种模态：

- 标题、描述、topics。
- ASR captions。
- 视觉/封面/视频内容 embedding。
- 音频或其他内容 embedding。

论文选择把多个 embedding 编码后拼接，再投影到统一表示，然后送入 RQ-VAE。

工程含义：

- item tokenization 不只是算法模块，也依赖多模态特征生产链路。
- demo 阶段要先确认哪些 embedding 已经 ready。
- 缺失模态、刷新频率、embedding 版本都会影响 SID 质量。

### 6.2 Multi-resolution codebooks

传统 RQ-VAE 如果每层 codebook 分辨率一样，可能造成 SID 空间稀疏、组合利用率低。

PLUM 的思路是：

- 前几层 codebook 分辨率高，用来做粗粒度强区分。
- 后面层数分辨率低，用来编码残差信息。

工程直觉：

```text
前层 SID 决定大类语义
后层 SID 做细粒度修正
```

### 6.3 Progressive masking

Progressive masking 训练时随机只使用前 r 层 codebook，让 SID 层次更稳定。

直觉是：

```text
不能让所有信息都依赖最后一层 code。
前面的层级也必须有语义承载能力。
```

这对生成式召回重要，因为自回归模型会按 SID token 顺序生成。前缀越有意义，beam search 越稳定。

### 6.4 Co-occurrence contrastive regularization

纯内容相似不等于推荐相关。

两个视频内容可能不同，但用户经常连续观看，说明它们在推荐语境里相关。论文把 co-occurrence 作为 contrastive signal 注入 SID 训练，让经常共现的 item 在 SID 空间更接近。

工程含义：

- item tokenization 需要内容信号，也需要行为信号。
- 行为共现可以提高 SID uniqueness 和 downstream recall。
- 但行为信号太动态，不能简单把动态 CF embedding 直接混入 SID，否则会导致 quantizer 和下游模型频繁重训。

## 7. Continued Pre-training：让 LLM 进入推荐域

论文里的短句：

> "continued pre-training"

中文理解：继续预训练，用推荐域语料把基础 LLM 适配到 SID 和用户行为世界。

CPT 的目标不是直接完成召回任务，而是训练一个更适合推荐域的 base checkpoint。

语料主要两类：

1. User behavior data：用户 watch history、watch ratio、watch time、time gap、channel 等。
2. Video metadata corpus：SID、title、description、ASR captions、channel name、synthetic data。

论文里的 CPT mixture 是两类数据各 50%，总训练约 1M steps、约 260B tokens。

系统含义：

- 需要构造推荐域预训练语料。
- 需要扩展 LLM vocab 加入 SID tokens。
- 需要管理 CPT checkpoint。
- 如果多个下游任务共用 CPT checkpoint，CPT 成本可以摊薄。

## 8. Generative Retrieval：把召回改写成生成任务

论文里的短句：

> "generative retrieval"

中文理解：生成式召回，让模型自回归生成推荐 item 的 SID。

SFT 阶段输入：

```text
watch history | user features | context video features
```

其中 watch history 由 SID token 和其他特征 token 组成。

训练目标：

```text
predict SID tokens of the clicked / next-watch video
```

论文使用带 handcrafted reward signal 的目标。实践中由于训练成本高，会按 reward 采样样本，然后对采样后的样本等权训练。

推理阶段：

```text
prompt
  -> beam search
  -> multiple SID sequences
  -> SID-to-video lookup
  -> retrieved candidates
```

## 9. 主要实验结果

### 9.1 PLUM vs LEM

论文比较 900M activated-param PLUM MoE 和高度优化的生产 LEM。

关键差异：

- LEM 绝大多数参数在 embedding table。
- PLUM 约 90% 参数在 neural network。
- PLUM 不依赖大 item embedding table 做召回。

Table 2 结果：

| Metric | LFV | Shorts |
| --- | ---: | ---: |
| Effective Vocab Size | 2.60x | 13.24x |
| CTR | 1.42x | 1.33x |
| WT/View | 0.72x | 1.13x |
| WF/View | 1.32x | 1.03x |

直觉解释：

- Effective vocab size 大幅提升，说明 PLUM 更能覆盖长尾和个性化内容。
- CTR 和部分观看指标有优势。
- LFV 的 WT/View 低于 LEM，说明生成式召回并不是所有指标都无条件胜出，仍需看业务权衡。

### 9.2 Live experiment

PLUM 作为额外候选源加入现有 candidate pool，与增加 LEM+ quota 做公平比较。

Table 3 结果：

| Metric | LFV | Shorts |
| --- | ---: | ---: |
| Engaged Users | +0.07% | +0.28% |
| Panel CTR | +0.76% | +4.96% |
| Views | +0.80% | +0.39% |
| Satisfaction | +0.06% | +0.39% |

这说明 PLUM 不是只在离线 Recall 上好看，而是能给已有生产系统带来增量候选价值。

### 9.3 Sample efficiency

论文强调 PLUM 很 sample efficient：

- 900M MoE 每天训练约 250M examples。
- 传统 LEM 每天训练 several billion examples。
- 虽然 PLUM 单样本训练成本更高，但总训练 FLOPs 小于 LEM 的 0.55x。

这是架构讨论里很重要的一点：大模型不一定总训练成本更高，关键要看收敛速度和样本效率。

### 9.4 SID-v2 ablation

Table 4：

| SID Model | SID Uniqueness | VID Recall@10 |
| --- | ---: | ---: |
| SIDv1 baseline | 94.0% | 12.3% |
| SIDv2 | 96.7% | 14.4% |
| Ablate multi-resolution | 94.8% | 13.2% |
| Ablate multi-embedding | 96.9% | 12.8% |
| Ablate co-occurrence | 91.8% | 12.6% |

结论：

- SID-v2 整体优于 SID-v1。
- Co-occurrence 对 uniqueness 和 recall 都很关键。
- 多模态和层次量化设计都影响 downstream retrieval。

### 9.5 CPT ablation

Table 5：

| Model | Pre-trained LLM | CPT | Recall@10 |
| --- | --- | --- | ---: |
| R1 | No | No | 0.19 |
| R2 | Yes | No | 0.23 |
| CR1 | No | Yes | 0.27 |
| CR2 | Yes | Yes | 0.28 |

结论：

- CPT 明显提升 retrieval SFT。
- 预训练 LLM 初始化也有增益。
- 完整路径是 LLM init + CPT + SFT。

### 9.6 Scaling study

论文用 MoE-110M、370M、900M、3B 做 scaling。

关键观察：

- loss 和 Recall@10 随 Iso-FLOPS 呈现 scaling 趋势。
- compute frontier 会随预算向更大模型移动。
- MoE-3B 在当前预算下没超过 900M，原因可能是 batch size、训练 examples 和 compute budget 没同步扩。
- 推荐任务的 compute-optimal training 需要同步 scaling model size 和 training examples。

## 10. 和推荐架构的映射

| 经典推荐组件 | PLUM 中的对应物 |
| --- | --- |
| item id embedding | Semantic ID token sequence |
| item tower / content feature pipeline | multi-modal embedding + SID quantizer |
| user behavior features | prompt 中的 watch history / user features |
| retrieval model | decoder-only LLM / MoE |
| ANN search | beam search + SID-to-video lookup |
| item index | SID vocabulary and mapping table |
| sample builder | prompt and target SID builder |
| online serving | autoregressive decoding service |
| retrieval monitoring | invalid SID, collision, latency, coverage, engagement |

## 11. 建议课程组织

这篇适合拆成 6 课：

1. 从 embedding retrieval 到 generative retrieval。
2. Semantic ID 与 item tokenization。
3. Continued pre-training 与推荐域适配。
4. Generative retrieval 的训练、解码与服务。
5. 实验、ablation 与 scaling。
6. 架构 checklist 与 demo/PoC 设计。

## 12. 两周学习路径

第 1 周：建立系统图。

- Day 1：读摘要、Introduction、PLUM Framework。
- Day 2：画出 LEM vs PLUM 的差异。
- Day 3：读 Semantic ID，整理 SID-v2 四个增强。
- Day 4：读 CPT，列出需要的语料和数据表。
- Day 5：读 generative retrieval，写 prompt/target schema。

第 2 周：转成方案能力。

- Day 6：读实验结果，整理效果指标。
- Day 7：读 ablation，判断哪些模块必须做。
- Day 8：读 scaling，理解模型大小、数据、算力的取舍。
- Day 9：映射到现有推荐系统，做 gap analysis。
- Day 10：写一页 demo proposal。

## 13. 团队讨论问题

1. 我们要验证的是生成式召回本身，还是多模态 SID，还是 CPT？
2. item tokenization 第一版是否必须完整复现 SID-v2？
3. 当前有哪些多模态 embedding 已经 ready？
4. 当前样本能否构造 next-watch / next-click 的 prompt-target 对？
5. SID-to-item mapping 如何版本化？
6. 生成 invalid SID 时如何回退？
7. beam size 的延迟预算是多少？
8. 生成式召回是替换现有召回，还是先作为补充召回源？
9. 除了 Recall@K，还看哪些线上/系统指标？
10. 如果 CPT 成本很高，是否有多个下游任务可以复用 checkpoint？

## 14. 最小 PoC 建议

第一阶段不建议一口气复刻完整 PLUM。更可控的 PoC 是：

1. 先选一个小 corpus 或业务子集。
2. 构造简化 SID baseline。
3. 用现有 U 聚合样本改造成 next-item prompt。
4. 小模型训练 SID 预测。
5. beam search 解码 top-K SID。
6. 做 SID-to-item lookup。
7. 与现有召回做离线 Recall / coverage 对比。
8. 同时记录 invalid SID rate、collision、latency。

如果第一版可行，再逐步加入：

- 多模态融合。
- co-occurrence contrastive。
- CPT。
- 更大模型和更长序列。
- 在线候选融合。

## 15. 局限与风险

1. SID 质量决定上限，tokenization 不是小模块。
2. CPT 成本高，语料构造和 checkpoint 复用需要提前规划。
3. Beam search 带来延迟和多样性 tradeoff。
4. SID collision 和 invalid SID 需要生产监控。
5. PLUM 当前重点是 retrieval，不能自动等价到 ranking。
6. 大模型 scaling 需要同时扩数据、batch、算力，不是单纯加参数。

## 16. 最后的理解

PLUM 的核心价值是把推荐系统里的 item、用户行为和多模态内容转成 LLM 可以学习和生成的 token 语言。

它真正改变的是推荐系统的表示层和召回范式：

```text
item 不再只是 embedding table 里的 row
而是一个可以被 LLM 理解、预测、生成、反查的 Semantic ID 序列
```

对架构同学来说，最重要的不是先推公式，而是先看清这件事会改动哪些系统边界：多模态特征、item tokenization、CPT 语料、样本构造、解码服务、SID 映射、监控和实验评估。

