# Versioned Late Materialization 教学大纲

论文：Versioned Late Materialization for Ultra-Long Sequence Training in Recommendation Systems at Scale

本地文件：[`paper.pdf`](./paper.pdf)

一句话总结：

这篇文章不是主要讲一个新的长序列模型结构，而是讲 Meta 如何把推荐训练样本里的超长用户行为序列从 "Fat Row" 里拆出来，用带版本的 late materialization 在训练时按需重建，从而突破长序列训练的数据存储和 I/O 墙。

## 1. 这篇文章适合谁读

这篇文章最适合推荐系统、训练数据平台、特征平台、样本生产链路、长序列建模工程化相关同学读。

如果只从算法视角看，它可能不如 HSTU / ULTRA-HSTU 那样显眼；但如果从生产系统视角看，它回答的是一个更底层的问题：

模型想吃 4K、10K、16K、64K 甚至更长的用户行为序列时，训练数据平台怎么不被拖垮？

这也就是它和 "partial dump" 可能相似的地方：核心都不是简单地存一份大样本，而是思考哪些内容必须物理 dump，哪些内容可以只 dump 指针、版本、时间边界，然后在训练读路径上按需补齐。

## 2. 先建立问题背景

推荐模型正在往更长的 User Interaction History, UIH 扩展。

早期方法可能只看几十到几百个行为；之后 DIN、SIM、ETA 等候选相关方法把历史扩到更大窗口；HSTU 一类序列推荐模型进一步把推荐改写成序列转导问题，对完整交互序列做因果注意力。论文强调一个趋势：更长序列通常带来更好的推荐质量，业界已经在往 10^5+ 行为长度推进。

问题在于，模型侧想扩长序列，数据侧会先爆。

传统推荐训练数据常用 "Fat Row" 范式：每条训练样本里都预先物化完整特征，包括长 UIH。这样做的好处是 O2O 一致性强，训练时看到的特征状态和线上请求时看到的特征状态一致；坏处是同一个用户的长历史会被重复写入大量样本。

可以把传统样本想成这样：

```text
training example =
  label
  request context
  candidate features
  user features
  complete UIH sequence copied here
```

如果一个用户一天有 K 次请求，要给模型提供 N 天历史，那么前 N-1 天那段几乎不变的历史会被重复写 K 次、读 K 次、传 K 次。这就是论文说的 K-fold amplification。

## 3. 论文要解决的核心挑战

### 挑战 1：长序列带来的存储和 I/O 墙

长 UIH 在训练数据里占了主要体积。序列越长，Fat Row 的写入量、存储量、训练读取带宽都会同步放大。

论文的 Figure 2 给出的判断很关键：当开启 ultra-long sequence 后，训练数据支撑服务的资源消耗会超过 GPU 训练本身。也就是说，瓶颈不是模型算不动，而是数据基础设施先撑不住。

### 挑战 2：O2O 一致性和 future leakage

推荐训练通常是异步的。线上请求发生在 `T_request`，用户反馈 label 发生在 `T_label`，训练发生在更晚的 `T_train`。

训练样本必须复现 `T_request` 时模型能看到的特征状态。如果训练时不小心读到了 `T_request` 之后发生的行为，离线指标会虚高，线上效果不转化。这就是 future leakage。

Fat Row 通过线上请求时直接 snapshot 全量特征来避免这个问题，但代价是巨大重复。

### 挑战 3：流式训练和批训练要共用同一套机制

生产系统里通常同时有两类训练：

- streaming training：秒级或分钟级消费实时样本，用于新鲜生产模型。
- batch training：从数据仓库回放历史样本，用于实验、回填和 warm-up。

一个 naive 的归一化方案是：训练样本只存主表，UIH 存 side table，离线 ETL 再 join。论文认为这不够，因为 streaming training 不能做重型离线 join，而且如果 join 出来的 UIH 快照和线上请求时不同，就会破坏 O2O 一致性。

### 挑战 4：多租户 union dataset 的浪费

同一份 union training dataset 会服务多个模型租户，比如召回、粗排、精排。

但这些模型需要的序列长度差异很大：

- 某些模型只需要最近 100 个行为。
- 某些 late-stage ranking 模型可能需要 100 倍更多行为。

Fat Row 下 union dataset 必须按最大模型需求物化全量长序列，短序列模型也被迫读全量 UIH。这就是 multi-tenant penalty。

## 4. 论文的核心洞察

这篇文章最重要的洞察是：

O2O 一致性是必须的，但 Fat Row 物理预物化不是必须的。

原因是 UIH 有一个特殊结构：

```text
UIH = append-only + temporally ordered + immutable
```

也就是说，用户历史不是随便覆盖更新的普通 feature。某个时间点 t 的历史状态可以由一个时间谓词确定：

```text
events where timestamp <= t
```

因此，理论上不需要在每条训练样本里复制完整 UIH。只要记录足够的版本信息，就能在训练时重新构造出当时的 UIH。

论文把这件事抽象成：

```text
Fat Row:
  每条样本存完整 UIH
  O(seq_length) logging per sample

Versioned Late Materialization:
  样本只存 mutable recent UIH + immutable UIH version metadata
  O(1) metadata logging per sample
  训练读路径按版本重建完整 UIH
```

这就是它和数据库里的 MVCC、late materialization 思想的关系：规范化存一份 canonical history，读的时候根据版本和时间边界 time-travel 回去。

## 5. 方法总览

论文方案可以拆成三层：

```text
线上请求阶段
  -> 读取 mutable UIH + immutable UIH 做在线推理
  -> 训练样本只记录 recent mutable sequence 和 immutable version metadata

离线/流式训练阶段
  -> 从训练样本读出 version metadata
  -> 到 immutable UIH storage 做 bounded range scan
  -> 拼接 mutable recent UIH 和 immutable historical UIH
  -> 得到 O2O consistent complete sequence

系统优化阶段
  -> read-optimized immutable storage
  -> projection pushdown
  -> disaggregated preprocessing
  -> pipelined I/O prefetching
  -> data-affinity sharding
```

## 6. 核心设计 1：把 UIH 拆成 mutable 和 immutable

论文把 UIH 分成两部分：

- Mutable UIH：最近发生的行为，要求秒级新鲜，在线服务会读。
- Immutable UIH：长期历史，占绝大多数体积，通过周期性 ETL / compaction 生成，主要给训练重建读取。

为什么要这样拆？

最近行为还在高速变化，适合继续 snapshot 到训练样本里，保证实时性和一致性。长期历史一旦进入 immutable store，就可以认为是静态、只读、有序的，可以被训练时反复按范围扫描，不需要复制到每个样本里。

训练样本里需要记录的不是全量 immutable UIH，而是：

```text
start_ts
end_ts
sequence_length
optional checksum
mutable recent sequence
```

其中 `start_ts` 和 `end_ts` 就是训练时重建 immutable 历史的版本边界。

## 7. 核心设计 2：训练时做 versioned late materialization

训练数据预处理时，每条样本会执行一次 time-travel reconstruction：

1. 从训练样本中取出 version metadata 和 recent mutable sequence。
2. 用 `start_ts`、`end_ts`、`sequence_length` 向 immutable UIH storage 发起 bounded range scan。
3. 读回长期历史片段。
4. 拼接 immutable history 和 mutable recent sequence。
5. 得到和线上 `T_request` 时一致的完整 UIH。

正确性来自两个地方：

- Mutable 部分在 `T_request` 已经被物理 snapshot，因此不会混入晚到行为。
- Immutable 部分用训练样本里记录的时间边界查询，而且 immutable store 中历史数据不再修改，因此训练时读到的范围和线上请求时等价。

论文还提到生产中会用 checksum 对比 Fat Row baseline 和 late-materialized sequence，验证重建一致性。

## 8. 核心设计 3：为训练读取优化 immutable UIH storage

如果只把 Fat Row 拆开，但训练时每条样本都随机查长历史，GPU 会饿死。所以论文的系统优化很重。

Immutable UIH store 的目标是支持大量训练任务并发做范围扫描。它的关键设计包括：

### 8.1 Offloaded compaction 和 single-level layout

每天 ETL 把新行为和已有历史 merge 成完整的、按时间排序的用户序列，再 bulk-load 到 immutable store。

因为数据只读，存储层不需要像 LSM 那样为写入优化，也不需要多层 compaction。对于每个用户的时间范围查询，理想情况是：

```text
one disk seek + sequential I/O
```

这比通用 KV / LSM 存储更适合训练时的长序列 range scan。

### 8.2 Multi-dimensional projection

Immutable store 的 key 不是只按 user_id 存一条巨大序列，而是组织成：

```text
user_id
feature_group
subsequence_timestamp
```

这样可以做两类 pushdown：

- sequence length projection：短序列模型只扫需要的 temporal stripes。
- feature group projection：不同模型只读自己需要的特征组。

这就是解决 multi-tenant penalty 的关键。

### 8.3 Trait-aware columnar encoding

UIH 里的 event trait 密度不同：

- post ID、timestamp 通常是 dense。
- like、comment、share 等 engagement signal 可能是 sparse。

因此 immutable store 用列式和 trait-aware encoding，比如 timestamp delta encoding、稀疏信号 bitmap、类别特征 dictionary compression。训练 materialization 时只解码模型需要的列，减少字节级 I/O 和 CPU decode。

## 9. 核心设计 4：训练读路径不拖慢 GPU

Late materialization 把一部分工作从写路径挪到读路径，必须补上训练时的吞吐优化。

### 9.1 Disaggregated Data Preprocessing, DPP

论文采用解耦的数据预处理架构，把数据加载、UIH 重建、解码从 trainer 主循环里拆出去，放到可弹性扩缩的 DPP worker 集群。

系统实时监控：

- GPU starvation percentage：GPU 等数据的空闲比例。
- worker waste percentage：DPP worker CPU 空闲比例。

当某个训练任务因为复杂 UIH materialization 导致 GPU 等数据时，系统可以加 DPP workers，让训练继续保持 GPU-bound。

论文提到，调节 DPP base batch size，在内存压力和线程并发之间取平衡，可带来 15% 的单 worker 预处理吞吐提升。

### 9.2 Pipelined I/O prefetching

DPP worker 上的查询引擎把 primary training table 作为 probe side，把 immutable UIH store 作为 build side。

对于 batch N，它先抽取 join key，发起 immutable UIH multi-range scan；同时不等返回，继续预取 batch N+1 的 primary training table。这样把 UIH lookup latency 和主训练数据读取重叠。

论文给出的收益是：在重 UIH lookup 模型上，单 worker 数据预处理吞吐额外提升 10%。

### 9.3 Data-affinity optimization

Batch training 访问历史分布更散，cache locality 差。论文用了两类亲和性优化：

- 按 user 聚类训练样本，让同一用户临近时间窗口的多个样本可以复用一次 UIH lookup。
- primary training data 和 immutable UIH storage 使用相同 hash partition key，减少网络 fanout。

论文给出的收益是：batch training 的 overall read bandwidth 降低 60%，单 worker 数据处理吞吐提升 28%。

## 10. 它到底解决了什么问题

这篇文章解决的不是“如何让 attention 复杂度更低”，而是：

当长序列模型已经证明有效时，怎么让训练数据基础设施允许模型继续扩序列长度？

可以从四个层面理解它的贡献：

1. 把长 UIH 从每条训练样本的物理 payload 中拆出来。
2. 用版本元数据保证训练时重建的 UIH 和线上请求时一致。
3. 用 immutable storage + projection pushdown 解决多租户和长序列 I/O 浪费。
4. 用 DPP、prefetch、data-affinity 让训练时重建不拖慢 GPU。

所以它真正卖的是一套训练数据基础设施范式：

```text
from pre-materialized fat rows
to versioned, normalized, just-in-time sequence materialization
```

## 11. 收益如何

论文的收益分成系统效率和模型效果两类。

### 11.1 系统效率收益

Table 1 以 Fat Row 为 baseline，给出三个共享 union dataset 的模型租户结果。

共享主训练数据写带宽：

- Primary write bandwidth 降低 46.2%。

每个训练任务主训练数据读带宽：

- Model A, long sequence：降低 70.3%。
- Model B, mid sequence：降低 50.9%。
- Model C, short sequence：降低 47.7%。

新增 immutable UIH lookup 带宽：

| Model | Streaming lookup | Batch lookup | Data loading latency |
| --- | ---: | ---: | ---: |
| A, long | +62.7% | +24.6% | +9.7% |
| B, mid | +16.2% | +6.5% | -26.4% |
| C, short | +8.7% | +3.4% | -36.2% |

这里要注意一个读法：Model A 的 lookup 带宽看起来抵消了很多 primary read savings，但 immutable store 的单位 host resource 读吞吐是 append-only primary training data storage 的 3.4x，所以实际资源占用仍显著更低。Model B/C 因为 projection pushdown，延迟还下降了。

### 11.2 模型效果收益

论文用两个生产推荐平台验证序列长度 scaling 的效果。

离线 NE 结果：

- 256 到 4K 区间，late materialization 和 Fat Row 效果持平，说明训练时重建没有质量劣化。
- Fat Row Wall 大约在 4K：当数据支撑服务资源 / GPU training power 超过 0.75 后，继续扩序列变得困难。
- Platform A 从 4K 扩到 64K 额外获得 1.2% cumulative NE improvement，总收益超过 5%。
- Platform B 从 4K 扩到 10K 获得 0.65% cumulative NE improvement。

线上 A/B：

| Product | Seq length | Metric | Relative gain |
| --- | --- | --- | ---: |
| Platform A | 4K -> 16K | Metric-Topline | +0.22% |
| Platform A | 4K -> 16K | Metrics-C | +4.1% |
| Platform A | 4K -> 16K | Metrics-E-1 | +2.3% |
| Platform A | 4K -> 16K | Metrics-E-2 | +4.3% |
| Platform B | 4K -> 10K | Metric-Topline | +0.14% |
| Platform B | 4K -> 10K | Metrics-C | +0.79% |
| Platform B | 4K -> 10K | Metrics-E-3 | +1.4% |
| Platform B | 4K -> 10K | Metrics-E-4 | +1.7% |

论文特别强调，在这种生产规模下，1% NE improvement 已经是成功，因此这些收益有实际业务意义。

## 12. 和 partial dump 的关系可以怎么理解

如果把 partial dump 抽象成“训练样本里不再 dump 所有重特征，而是只 dump 必要的快照信息和可回放索引”，那这篇文章就是一个很完整的工业案例。

可以对应成下面这张表：

| 传统 Fat Row | Versioned late materialization | Partial dump 类比 |
| --- | --- | --- |
| 每条样本复制完整 UIH | UIH 规范化存一份 | 重 payload 不重复 dump |
| 用完整 snapshot 保证 O2O | 用时间边界和版本保证 O2O | dump 版本/指针/边界 |
| 训练读样本即可 | 训练时按版本重建 | reader 端补齐特征 |
| 简单但 I/O 爆炸 | 系统复杂但可扩展 | 写路径省，读路径复杂 |
| 多租户被最大序列拖累 | projection pushdown | 不同模型只取需要部分 |

这篇文章真正值得借鉴的不是某个单点组件，而是它的边界划分：

- 哪些东西必须在 request time 固化？recent mutable sequence、version metadata、label join 所需信息。
- 哪些东西可以在 training time 重建？长期 immutable sequence。
- 重建需要什么正确性条件？append-only、temporal order、immutable、time boundary、checksum。
- 为了不拖慢训练，要把哪些系统能力补齐？高吞吐 immutable store、projection、DPP、prefetch、affinity。

## 13. 推荐教学安排

这份教程可以拆成 6 个模块，每个模块都围绕一个工程问题展开。

## 模块 1：为什么长序列推荐会撞上数据墙

### 学习目标

理解长 UIH 对推荐质量有价值，但 Fat Row 会把长序列收益变成数据基础设施成本。

### 重点内容

- UIH sequence length scaling 的背景。
- HSTU / ULTRA-HSTU 和数据基础设施的关系。
- Fat Row 是什么。
- K-fold amplification 是什么。
- 为什么 data supporting service 会超过 GPU training resource。

### 学完后应该能回答

- 为什么同一个用户历史会被重复写入训练样本？
- 为什么 4K 可能成为 Fat Row Wall？
- 长序列训练的瓶颈为什么不一定在 GPU？

## 模块 2：O2O 一致性、future leakage 和为什么不能简单归一化

### 学习目标

理解训练样本不是随便少存一点就行，少存之后必须仍然复现线上请求时的特征状态。

### 重点内容

- `T_request`、`T_label`、`T_train` 的时间关系。
- future leakage 如何造成离线指标虚高。
- 为什么 Fat Row 是 sufficient but not necessary。
- 为什么 naive side-table offline join 不适合统一流式和批训练。

### 学完后应该能回答

- O2O consistency 的真正约束是什么？
- 训练时读到了请求之后的行为会发生什么？
- 为什么只做离线 join 不能替代这篇论文的协议？

## 模块 3：Versioned late materialization 协议

### 学习目标

掌握论文的核心协议：线上只记录版本信息，训练时按版本 time-travel 重建。

### 重点内容

- UIH 的 append-only / temporal / immutable 特性。
- Mutable UIH 和 Immutable UIH 的拆分。
- `start_ts`、`end_ts`、`sequence_length`、checksum 的作用。
- Training-time bounded range scan。
- Mutable snapshot + Immutable reconstruction 如何拼接。

### 学完后应该能画出

```text
Ranking request
  -> fetch mutable UIH + immutable UIH
  -> log mutable sequence + immutable version metadata
  -> label arrives
  -> training data loader reads example
  -> range scan immutable UIH
  -> reconstruct complete UIH
```

## 模块 4：Immutable UIH store 如何支撑高并发训练

### 学习目标

理解为什么这个系统不能只靠“多加一个 KV 表”，而需要专门的 read-optimized immutable store。

### 重点内容

- Daily ETL / compaction。
- Chronologically ordered user sequences。
- Single-level layout。
- Multi-range scan。
- Sequence length projection。
- Feature group projection。
- Trait-aware columnar encoding。

### 学完后应该能回答

- 为什么 immutable/read-only 让存储布局可以更激进？
- 为什么 projection pushdown 能解决多租户浪费？
- 列式编码和 selective decoding 在这里省了什么？

## 模块 5：训练时 materialization 如何不拖慢 GPU

### 学习目标

理解 late materialization 的代价从写路径转移到了读路径，必须通过数据预处理系统吸收掉。

### 重点内容

- Disaggregated Data Preprocessing。
- GPU starvation 和 worker waste。
- Trainer-side rebatching。
- Pipelined I/O prefetching。
- Batch training 的 data-affinity optimization。
- Symmetric sharding 降低网络 fanout。

### 学完后应该能回答

- 为什么训练数据 reader 变复杂以后 GPU 可能饿死？
- DPP worker 和 trainer 之间的责任边界是什么？
- 论文如何把 sequence lookup latency 藏起来？

## 模块 6：如何判断收益和取舍

### 学习目标

学会用系统指标和模型指标共同评价这类 partial dump / late materialization 方案。

### 重点内容

- Primary write bandwidth。
- Primary read bandwidth。
- Immutable sequence lookup bandwidth。
- Per-batch data loading latency。
- GPU starvation。
- NE gain。
- Online A/B topline 和 consumption/engagement metrics。

### 学完后应该能回答

- 为什么不能只看 read bandwidth raw number？
- 新增 lookup 带宽什么时候值得？
- 为什么 Platform A/B 的线上收益说明基础设施本身是模型质量杠杆？

## 14. 逐课精讲

说明：下面每课都包含“原文短摘句”和“中文翻译”。为了避免大段搬运论文正文，这里只保留很短的原文片段；真正需要完整吸收的内容，我会用中文详细转述、解释和拓展。这样既能贴近原文，又能减少来回跳 PDF 的成本。

## 课 1：为什么长序列推荐会撞上数据墙

### 本课核心问题

这一课要讲清楚一个看似反直觉的事实：

长序列推荐的第一道墙，未必是模型算力墙，而是训练数据基础设施墙。

模型架构论文通常会强调 attention 怎么优化、序列怎么建模、参数量怎么扩展。但这篇文章的切入点更底层：当模型已经证明“吃更长历史会更好”以后，训练样本怎么生产、怎么存、怎么读，反而成了是否能继续 scaling 的决定因素。

### 原文短摘句

> "storage and I/O wall"

中文翻译：存储和 I/O 墙。

这里的 wall 不是普通的性能瓶颈，而是一个会改变研发投入回报的边界。序列继续加长时，收益还在增加，但数据支撑系统的成本上升得更快，导致继续扩序列不再划算。

### 论文在说什么

传统推荐训练样本是 Fat Row：每条样本都已经把模型训练需要的特征准备好，包括完整用户历史。它的优点是训练 reader 很简单，训练时不需要复杂回查；缺点是同一个用户历史会在很多请求样本里重复出现。

举一个直觉例子：

```text
用户 U 今天发起 20 次推荐请求
模型需要最近 30 天历史
其中前 29 天历史对这 20 次请求几乎一样
Fat Row 会把这段历史重复写入 20 条样本
```

如果历史序列很短，这种重复还能忍；一旦从 256 扩到 4K、16K、64K，重复 payload 就会变成训练数据体积的主导因素。

论文把这个问题写成 K-fold amplification：请求次数 K 越大，同一份历史被重复 materialize 的次数越多。

### 更深一层的理解

Fat Row 的本质是用“写路径冗余”换“读路径简单”：

```text
写路径：
  每条样本写完整 UIH，成本高

读路径：
  trainer 顺序读样本即可，逻辑简单
```

这在短序列时代是合理的，因为硬盘和网络多传一点历史，比训练时动态 join 更简单。但长序列时代，序列长度变成主因，Fat Row 开始把数据系统推向不可持续。

这里有一个很重要的工程判断：

```text
如果每条样本里的大字段是高度重复的，
那么继续压缩单条样本格式，只能缓解，不能根治。
真正要做的是消除跨样本重复。
```

这就是论文为什么不只做 encoding/compression，而是改变数据组织范式。

### 和 partial dump 的关系

如果你们说的 partial dump 是“样本里不 dump 全量重特征，只 dump 必要信息”，那课 1 的重点就是先识别：

- 哪些字段是大 payload？
- 哪些字段跨样本重复？
- 哪些字段重复来自同一用户多次请求？
- 哪些字段只是为了 O2O 一致性才被 dump？

在这个框架下，partial dump 不是简单少存一些列，而是要把样本拆成两类信息：

```text
必须随样本固化的信息：
  label、request context、候选集合、request-time recent snapshot

可以被共享和回查的信息：
  long-term UIH、稳定 SideInfo、可版本化的历史特征
```

### 课堂讲解建议

建议先让听众画自己系统里的训练样本结构，把字段按大小排序。然后问三个问题：

1. 最大的字段是不是 UIH 或历史特征？
2. 这个字段是不是在同一用户的多条样本中重复？
3. 如果只保留一个指针和时间边界，能不能重建它？

如果三个答案都是“是”，就已经进入这篇论文的问题域。

### 常见误区

误区 1：以为数据墙只是存储成本问题。

实际不是。它同时影响写带宽、读带宽、预处理 CPU、网络传输、训练 worker 数量和 GPU 等数据时间。

误区 2：以为压缩就能解决。

压缩能降低单份 payload 的大小，但不能消除 K 份重复。长序列重复放大到一定程度后，需要 normalization，而不是只靠 codec。

误区 3：只看旗舰模型。

在 union dataset 场景下，长序列旗舰模型会迫使所有租户一起承担最大序列成本。短序列模型被动 over-fetch，这才是多租户平台最痛的地方。

### 课后作业

拿你们当前训练样本做一次字段体积盘点，输出下面这张表：

| 字段 | 平均字节 | P99 字节 | 是否跨样本重复 | 是否 request-time 必须固化 | 是否可版本化重建 |
| --- | ---: | ---: | --- | --- | --- |
| UIH item id | | | | | |
| UIH timestamp | | | | | |
| 行为类型 | | | | | |
| item side info | | | | | |
| user profile | | | | | |

这张表比空谈 partial dump 有用得多，因为它能直接告诉你应该先拆哪个字段。

## 课 2：O2O 一致性、future leakage 和为什么不能简单归一化

### 本课核心问题

这一课要讲清楚：

训练样本不能因为省空间就随便少存。少存之后，仍然必须保证训练时看到的特征，等价于线上请求发生时模型能看到的特征。

推荐训练比普通监督学习麻烦的地方在于，label 通常晚于请求到达。请求发生时，模型做了一次排序；用户之后点击、观看、点赞、评论，label 才形成；训练又在更晚的时候发生。

时间线是：

```text
T_request:
  线上模型做推荐，此时只能看到 request 前的用户历史

T_label:
  用户反馈到达，比如点击、完播、点赞

T_train:
  训练系统消费样本，此时用户历史已经继续增长
```

### 原文短摘句

> "future leakage"

中文翻译：未来信息泄露。

意思是训练样本包含了请求发生之后才出现的信息。模型训练时学到了线上推理时不可能知道的东西，离线指标会看起来更好，但线上不会兑现。

### 论文在说什么

Fat Row 的一个重要价值是保证 O2O 一致性。线上请求时直接把当时的特征快照存下来，之后 label 到了，再和这个快照 join 成训练样本。这样训练时不需要重新查特征，也不担心查到未来状态。

但论文指出，O2O 一致性是硬要求，Fat Row 只是实现它的一种方式。换句话说：

```text
必须保证：
  training feature state == online inference feature state

不一定必须：
  把完整 feature state 物理复制进每条样本
```

这个区分非常重要。很多系统因为害怕训练服务不一致，于是把所有东西都 snapshot；短期看最稳，长期看会形成巨大的数据债务。

### 为什么 naive normalization 不够

最朴素的想法是：训练样本主表只存 sample_id 和 label，UIH 单独存在一张历史表，训练前离线 join 一下。

这个方案有两个问题：

```text
问题 1：streaming training 不适合重型 join
  实时训练要求秒级或分钟级消费样本，不可能每条都跑复杂离线 ETL。

问题 2：join 的时间语义可能错
  如果 join 时按当前最新 UIH 查，就可能读到 T_request 之后的行为。
```

所以论文真正解决的不是“UIH 放 side table”，而是“如何带时间语义地重建 request-time UIH”。

### 更深一层的理解

O2O 一致性不是一个数据格式问题，而是一个时间语义问题。

可以把特征分成三类：

| 特征类型 | 举例 | O2O 风险 | 处理建议 |
| --- | --- | --- | --- |
| 请求当下状态 | session context、candidate set | 高 | 直接 snapshot |
| 最近变化历史 | recent UIH | 高 | snapshot 或保留短窗口 |
| 长期不可变历史 | old UIH events | 中/低 | 记录版本边界，训练时重建 |

越靠近请求发生时刻、越容易变化的东西，越应该物理固化；越稳定、越可重放的东西，越适合 late materialization。

### 和 partial dump 的关系

partial dump 最危险的地方不是少存，而是少存后没有明确的时间版本。

一个合格的 partial dump 记录，至少要能回答：

```text
这个样本对应哪个 request_ts？
训练时最多能读到哪个 event_ts？
如果特征表后来回填或修正，应该读哪个版本？
如果数据删除或合规 scrub，如何保证重建语义一致？
```

如果这些问题没有答案，省下来的存储很可能会变成未来的训练偏差。

### 课堂讲解建议

建议用一个具体用户时间线讲：

```text
10:00 用户打开 App，请求推荐
10:01 用户点击视频 A
10:02 用户观看视频 B
10:05 label join 完成
10:10 训练样本被消费
```

然后让大家判断：训练这条 10:00 请求样本时，10:01 和 10:02 的行为能不能进入 UIH？答案通常是不能，因为这些行为在 10:00 线上排序时不可见。

### 常见误区

误区 1：把 label 行为也当作历史特征。

label 是训练目标，不应该泄露进同一条样本的输入历史。尤其在 next-session 或 next-item 任务中，这一点非常容易出错。

误区 2：以为 batch training 可以晚点 join。

batch 训练晚点 join 没问题，但 join 必须按 request-time version，而不是按 train-time latest state。

误区 3：只验证样本字段存在，不验证时间边界。

字段都齐不代表样本正确。必须检查 `event_ts <= request_ts` 或更严格的业务边界。

### 课后作业

设计一个 future leakage 检测 SQL 或伪代码：

```text
for each training_example:
  for each event in reconstructed_uih:
    assert event.event_ts <= training_example.request_ts
```

再加一个更真实的版本：

```text
assert event.feature_version <= training_example.logged_feature_version
assert event.ingestion_ts <= allowed_watermark
```

这里的第二个版本更贴近生产，因为有些系统的 event_ts、ingestion_ts、feature_version 不是同一个时间。

## 课 3：Versioned late materialization 协议

### 本课核心问题

这一课是整篇论文的中心：如何不在样本里复制完整 UIH，却仍然在训练时重建出线上请求时看到的 UIH？

答案是：记录轻量版本元数据，把 UIH 拆成 recent mutable 部分和 long-term immutable 部分。

### 原文短摘句

> "append-only, temporally ordered, immutable sequence"

中文翻译：只追加、按时间有序、不可变的序列。

这是论文方案成立的根基。UIH 之所以能 late materialize，不是因为任何特征都能这么做，而是因为历史行为序列天然接近事件日志：新事件追加进来，旧事件原则上不被原地覆盖。

### 论文在说什么

论文把 UIH 拆成两个存储层：

```text
Mutable UIH:
  最近行为
  高频写入
  秒级新鲜
  在线请求路径读取
  样本里保留 snapshot

Immutable UIH:
  长期历史
  低频批量生成
  只读、有序、可 range scan
  样本里只保留版本边界
```

训练样本不再存完整 long-term UIH，而是存：

```text
request_ts
mutable_sequence_snapshot
immutable_start_ts
immutable_end_ts
target_sequence_length
feature_groups
checksum
```

训练时 data loader 用这些信息去 immutable store 查历史片段，再和 mutable snapshot 拼起来。

### 为什么 mutable 部分还要 snapshot

很多人第一次读会问：既然 long-term UIH 可以 late materialize，recent UIH 为什么不也回查？

原因是 recent 部分最容易变化，也最容易造成 future leakage。比如请求后的几秒内，用户可能连续点击、滑走、停留，这些行为对 label 很相关，但在请求时不可见。如果训练时从一个实时 mutable store 查，很容易查到请求后的状态。

所以论文选择保守处理：

```text
recent, fast-changing, high-leakage-risk:
  request-time snapshot

old, compacted, immutable, low-change-risk:
  versioned reconstruction
```

这是一种非常工程化的折中：不是所有东西都 late materialize，而是只对适合的部分 late materialize。

### 正确性怎么理解

完整重建过程可以写成：

```text
reconstructed_uih(example):
  immutable_part = scan(
    user_id = example.user_id,
    start_ts = example.immutable_start_ts,
    end_ts = example.immutable_end_ts,
    max_len = example.target_sequence_length,
    feature_groups = example.feature_groups
  )

  mutable_part = example.mutable_sequence_snapshot

  return concat(immutable_part, mutable_part)
```

正确性依赖三个条件：

1. immutable store 里的历史不会被原地改写。
2. 训练样本记录的时间边界来自 request time。
3. 拼接逻辑和线上构造 UIH 的逻辑一致。

如果这三个条件成立，训练时晚一点重建也不会读到未来信息。

### 和 MVCC 的关系

数据库里的 MVCC 解决的是：读事务如何看到某个一致版本，而不被并发写影响。

这里的类比是：

```text
数据库 MVCC:
  transaction timestamp -> visible rows

推荐 UIH 重建:
  request timestamp/version boundary -> visible events
```

不同的是，推荐训练还要同时满足 streaming/batch 两类消费方式，并且面对的是超长序列、多租户投影和训练吞吐问题。所以这不是把数据库概念直接套过来，而是把版本化读取思想应用到推荐训练数据。

### 和 partial dump 的关系

partial dump 如果只 dump 一个 user_id，是不够的；如果 dump user_id + request_ts，也可能不够。因为真实生产里还会有：

- compaction 周期。
- event ingestion 延迟。
- feature schema 版本。
- feature group 选择。
- sequence length 选择。
- 删除和合规 scrub。

更稳妥的 partial dump metadata 应该像一个“可重建契约”：

```text
PartialDumpContract:
  sample identity:
    sample_id
    user_id
    request_ts

  visibility boundary:
    immutable_start_ts
    immutable_end_ts
    mutable_snapshot_end_ts

  projection:
    sequence_length
    feature_groups
    trait_columns

  validation:
    checksum
    schema_version
    materializer_version
```

这个 contract 的价值是：以后换 reader、换存储、换模型，都还能按同一个语义重建样本。

### 常见误区

误区 1：认为所有历史都 immutable。

如果你们有反作弊回填、内容删除、用户隐私删除、行为纠错，历史就不是简单 immutable。可以仍然做 late materialization，但需要把 compaction 产物视为带版本的 immutable snapshot。

误区 2：只记录 end_ts，不记录 start_ts。

不同模型需要不同 lookback 或不同 sequence length。只记录 end_ts 可能无法复现当时模型实际使用的窗口。

误区 3：checksum 只用于调试。

在迁移期，checksum 是建立信任的核心工具。没有 checksum，很难证明 reconstructed UIH 和 Fat Row baseline 等价。

### 课后作业

写一个最小 materializer 伪代码，并显式处理三类异常：

```text
materialize(example):
  if schema_version unsupported:
    return error("schema mismatch")

  immutable = scan_immutable_store(example.version_boundary)
  full = concat(immutable, example.mutable_snapshot)

  if has_future_event(full, example.request_ts):
    return error("future leakage")

  if example.checksum and checksum(full) != example.checksum:
    return error("reconstruction mismatch")

  return full
```

再思考：这些 error 是丢样本、降级用 Fat Row，还是阻断训练？

## 课 4：Immutable UIH store 如何支撑高并发训练

### 本课核心问题

这一课要讲清楚：

把 UIH 从 Fat Row 拆出来之后，压力不是消失了，而是集中到了 immutable UIH store 和训练 materializer 上。

所以系统成败取决于 immutable store 是否能支撑大量训练任务并发 range scan。

### 原文短摘句

> "read-optimized immutable storage"

中文翻译：面向读取优化的不可变存储。

这个短语很关键。它说明论文不是随便找一个 KV 存历史，而是围绕训练读取模式专门设计存储。

### 论文在说什么

Immutable UIH 的查询模式非常稳定：

```text
给定 user_id
给定时间范围
给定 sequence length
给定 feature group
读取一段连续历史
```

既然查询模式固定，存储就可以针对它做极致优化。论文的思路是：

- 每天通过离线 ETL 把新增行为和已有历史合并。
- 为每个用户生成按时间排序的完整历史。
- 按存储引擎拓扑预排序、预分片。
- bulk-load 成只读 single-level layout。
- 查询时尽量变成连续 I/O。

这和在线写优化的 LSM/KV 设计目标不一样。在线 KV 需要高频写入和点查，immutable UIH store 则主要服务大规模范围读。

### 为什么 single-level layout 重要

LSM 类存储通常为了写入吞吐，把数据写到多层结构，再通过 compaction 整理。读一个用户的历史范围时，可能要跨多个层级、多个文件查。

论文的 immutable store 通过离线 compaction 一次性生成有序只读文件，避免在线 compaction 和多层读放大。理想访问模式是：

```text
定位到用户序列所在位置
顺序扫描需要的 subsequence stripes
只解码模型需要的 feature groups / traits
```

这就是为什么它可以在单位 host resource 上达到更高读取吞吐。

### Projection pushdown 为什么是核心

如果 immutable store 只解决“存一份”，但每个模型读取时仍然读全量历史，那 multi-tenant penalty 还在。

论文的 projection 有三层：

| Projection | 解决什么浪费 | 示例 |
| --- | --- | --- |
| sequence length | 短序列模型不读长历史 | retrieval 只读最近 100 |
| feature group | 简单模型不读复杂特征组 | 只读 item id，不读 engagement traits |
| trait column | 只解码需要的列 | 读 timestamp，不解码 comment/share |

这三层叠加起来，才让 union dataset 可以同时服务不同模型，而不让所有模型被最大模型拖累。

### 和 partial dump 的关系

partial dump 方案如果只改写样本格式，但没有对应的 projection-aware storage，收益会很快被 reader 端吃掉。

一个更完整的设计应该包括：

```text
样本侧：
  只记录 version metadata 和 projection requirement

存储侧：
  支持按 sequence length / feature group / trait column 读取

reader 侧：
  把模型的 feature spec 下推到 immutable store
```

也就是说，partial dump 不只是“少写”，还必须让“少读”成立。

### 课堂讲解建议

可以画一个用户历史矩阵：

```text
rows    = chronological events
columns = traits

event_1: item_id, ts, event_type, watch_time, like, comment
event_2: item_id, ts, event_type, watch_time, like, comment
...
```

然后让不同模型选择不同视图：

- 召回模型：最近 100 个 event 的 item_id。
- 粗排模型：最近 1K event 的 item_id + event_type。
- 精排模型：最近 16K event 的多组 dense/sparse traits。

这时大家就会直观看到 projection pushdown 的价值。

### 常见误区

误区 1：认为 normalized store 只要能查到数据就行。

不够。训练是高吞吐批量读取，不是后台 debug 查询。存储必须面向模式化 range scan 优化。

误区 2：只做行级切分，不做列级投影。

长 UIH 不只是 event 数多，event 内 trait 也多。只按时间切 stripe，仍然可能解码大量无用列。

误区 3：忽略 schema evolution。

模型会不断尝试新 SideInfo 或丢弃旧特征。immutable compaction 如果能整窗口重建，会显著降低特征迭代成本。

### 课后作业

设计一个 immutable UIH key：

```text
key = hash_bucket(user_id) / user_id / feature_group / subsequence_start_ts
value = columnar_encoded_events
```

然后回答：

1. 如果模型只要最近 512 个行为，需要扫几个 stripe？
2. 如果模型不需要 comment/share，能不能不解码这些列？
3. 如果新增 creator_category 特征，需要全量回填多久？
4. 如果用户请求删除数据，下一次 compaction 如何 scrub？

## 课 5：训练时 materialization 如何不拖慢 GPU

### 本课核心问题

这一课要讲清楚一个 tradeoff：

late materialization 省了写路径和主训练数据读取，但会把复杂性转移到训练读路径。

如果处理不好，训练数据 reader 会成为新瓶颈，GPU 等数据，整体训练吞吐下降。论文的生产化部分，就是在解决这个问题。

### 原文短摘句

> "pipelined I/O prefetching"

中文翻译：流水线式 I/O 预取。

这是论文隐藏训练时 lookup 延迟的关键技术之一。它不是让 lookup 消失，而是把 lookup 和下一批主训练数据读取重叠起来。

### 论文在说什么

训练时，DPP worker 需要做三件事：

```text
读取 primary training example
根据 version metadata 查询 immutable UIH
解码并拼接完整 UIH
```

这些工作如果直接串行做，会增加 per-batch data loading latency。论文通过几个办法减少影响：

1. DPP 解耦：把数据预处理从 trainer 主循环拆出去，独立扩容。
2. base batch tuning：DPP worker 用较小 base batch 控制内存，再由 trainer-side rebatching 合并成训练大 batch。
3. I/O prefetch：batch N 的 UIH lookup 和 batch N+1 的 primary read 重叠。
4. partial projection：只查只解码模型需要的数据。
5. data affinity：把同一用户相邻样本聚在一起，减少重复 lookup。
6. symmetric sharding：让主训练数据和 UIH store 分片对齐，减少跨 shard fanout。

### 更深一层的理解

训练吞吐可以简单拆成：

```text
step_time = max(gpu_compute_time, data_ready_time)
```

如果 `data_ready_time < gpu_compute_time`，训练是 GPU-bound，数据系统的额外复杂度被隐藏了。

如果 `data_ready_time > gpu_compute_time`，训练变成 data-bound，GPU 开始 idle。

论文所有优化的目标都是让 late materialization 之后仍然满足：

```text
data_ready_time <= gpu_compute_time
```

这就是为什么它强调 GPU starvation percentage。不是因为 data loader 延迟本身绝对不能涨，而是不能涨到让 GPU 等数据。

### DPP worker 的责任边界

可以把 trainer 和 DPP 的边界理解成：

```text
DPP worker 负责：
  读取样本
  查 immutable UIH
  做 projection
  解码 traits
  拼接完整特征
  输出训练可消费 batch

Trainer 负责：
  接收已经 materialized 的 batch
  rebatch / shuffle
  forward / backward
  optimizer step
```

这条边界很重要。它让训练框架不需要理解太多存储细节，也让数据系统可以独立扩缩容。

### 和 partial dump 的关系

partial dump 上线时最容易被质疑的问题就是：

“你省了存储，但训练是不是变慢了？”

所以 PoC 不能只看样本体积下降，必须同时验证：

| 指标 | 为什么重要 |
| --- | --- |
| per-batch data loading latency | 直接影响训练 step 是否等数据 |
| GPU starvation | 判断训练是否从 GPU-bound 变成 data-bound |
| DPP worker CPU utilization | 判断 worker 是否扩太多或太少 |
| immutable lookup p50/p99 | 判断存储尾延迟风险 |
| network fanout | 判断 sharding 是否合理 |
| cache hit rate | 判断 batch/streaming 访问局部性 |

如果 partial dump 没有这些指标，就很难证明它不是把成本从一边挪到另一边。

### 常见误区

误区 1：只看平均 data loading latency。

p99 和 GPU starvation 更重要。训练集群里少量长尾 lookup 就可能造成同步训练等待。

误区 2：认为多加 worker 就能解决。

DPP worker 能缓解 CPU 和并发，但如果 immutable store 或网络 fanout 是瓶颈，只加 worker 反而会放大压力。

误区 3：忽略 batch training 和 streaming training 的访问差异。

Streaming training 更容易有时间局部性；batch training 可能打散历史分区，cache locality 更差，需要额外 data-affinity。

### 课后作业

为 partial dump reader 设计一个压测计划：

```text
case 1: short sequence, high concurrency
case 2: mid sequence, high concurrency
case 3: long sequence, low concurrency
case 4: batch backfill, low locality
case 5: streaming, high temporal locality
```

每个 case 输出：

- primary read bytes。
- immutable lookup bytes。
- materialization latency p50/p90/p99。
- GPU starvation。
- DPP worker 数量。
- storage QPS 和 network fanout。

## 课 6：如何判断收益和取舍

### 本课核心问题

这一课要讲清楚：

这类系统的收益不能只看“省了多少主表读写”，还要看它是否释放了模型扩长序列的空间，并最终转化成模型质量收益。

### 原文短摘句

> "model quality gains"

中文翻译：模型质量收益。

论文的重点不是 late materialization 本身让模型变聪明，而是它让更长 UIH 训练变得可承受；真正的质量提升来自序列长度继续 scaling。

### 论文在说什么

系统效率部分，论文看了这些指标：

- 主训练数据写带宽。
- 主训练数据读带宽。
- 新增 immutable UIH lookup 带宽。
- batch/streaming lookup 差异。
- per-batch data loading latency。

模型效果部分，论文看了：

- 离线 NE 随 sequence length 的变化。
- Fat Row 和 late materialization 在可重叠序列长度上的效果是否一致。
- 超过 Fat Row Wall 后继续扩长的 NE 收益。
- 线上 A/B 的 topline、consumption、explicit engagement 指标。

这套评估很完整，因为它同时证明两件事：

```text
正确性：
  late materialized sample 没有让模型质量变差

价值：
  它让原来不可承受的长序列训练变得可行，并带来线上收益
```

### 怎么读 Table 1

很多人看到 Model A long sequence 的新增 lookup 带宽比较高，会误以为收益不明显。但论文给出的解释是：immutable store 的单位资源读取吞吐更高，因此 raw bandwidth 不能直接和 primary training storage 的 raw bandwidth 等价比较。

更合理的阅读方式是：

```text
净收益 = 主训练数据读写资源下降
       - immutable lookup 资源新增
       - DPP/materializer 资源新增
```

而不是简单算：

```text
primary read savings - lookup bandwidth
```

因为不同存储层的单位 bandwidth 成本不一样。

### 怎么读模型收益

论文 Figure 4 的关键不是“4K 到 64K 有多少 NE”，而是这个逻辑：

```text
Fat Row 能支持到约 4K
late materialization 支持继续扩到 10K/16K/64K
更长序列继续带来 NE 改善
线上 A/B 也看到 engagement/topline 改善
```

所以基础设施收益最终体现在：

```text
同样资源 envelope 下，模型可以吃更长历史
```

这是比“省了多少 TB”更强的论点。省资源只是第一层，释放模型 scaling 空间才是第二层。

### 和 partial dump 的关系

评估 partial dump 时建议分三张表：

第一张表：系统资源表。

| 模型 | 主表写带宽 | 主表读带宽 | lookup 带宽 | materialization 延迟 | GPU starvation |
| --- | ---: | ---: | ---: | ---: | ---: |
| short | | | | | |
| mid | | | | | |
| long | | | | | |

第二张表：正确性表。

| 样本类型 | checksum match | future event rate | missing trait rate | schema mismatch |
| --- | ---: | ---: | ---: | ---: |
| streaming | | | | |
| batch | | | | |
| backfill | | | | |

第三张表：模型收益表。

| 序列长度 | 离线指标 | 训练成本 | 数据成本 | 是否可线上 |
| --- | ---: | ---: | ---: | --- |
| baseline | | | | |
| 2x | | | | |
| 4x | | | | |
| 8x | | | | |

这三张表能避免只讲一个角度。partial dump 不是单纯平台优化，也不是单纯算法优化，它夹在数据成本和模型质量之间。

### 常见误区

误区 1：只报存储下降。

老板和业务更关心的是：省下来的资源能不能换来更长序列、更快实验、更高指标。

误区 2：只报离线收益。

如果 late materialization 引入 subtle O2O skew，离线收益可能不可靠。必须先证明和 Fat Row baseline 对齐。

误区 3：忽略中短序列租户。

如果短序列模型训练变慢或资源上涨，union dataset 平台可能无法接受。多租户收益要按租户拆开看。

### 课后作业

写一份一页纸评估 memo，结构如下：

1. 当前 baseline 是什么。
2. partial dump 改动了哪些字段。
3. 正确性如何证明。
4. 系统资源节省在哪里。
5. 新增资源开销在哪里。
6. 是否释放了更长序列训练空间。
7. 下一步是继续省成本，还是拿更长序列做模型收益。

## 15. 两周快速学习路径

## 第 1 周：把问题和协议吃透

### Day 1：读摘要、Introduction、Section 2

目标：

- 能复述 Fat Row 问题。
- 能解释为什么长序列训练先撞的是数据墙。

产出：

- 写一页纸说明：如果你们当前 partial dump 仍然 dump 完整 UIH，会在哪些维度爆？

### Day 2：读 Section 2.1 和 3.1

目标：

- 搞清 O2O consistency 和 future leakage。
- 搞清为什么 UIH 可以不物理 snapshot 全量。

产出：

- 画出 `T_request -> T_label -> T_train` 时间线。
- 标出哪些事件可以进样本，哪些事件不能进样本。

### Day 3：读 Section 3.2 和 3.3

目标：

- 掌握 mutable/immutable 拆分。
- 掌握 version metadata 的最小字段。

产出：

- 设计一个你们系统里的 `partial_dump_record` schema 草稿：

```text
sample_id
user_id
request_ts
label_ts
mutable_sequence_snapshot
immutable_start_ts
immutable_end_ts
target_sequence_length
feature_group_list
checksum
```

### Day 4：读 Section 4.1

目标：

- 理解 immutable store 为什么要按训练 range scan 优化。

产出：

- 画一张存储 key 设计图：

```text
(user_id, feature_group, subsequence_timestamp) -> encoded event stripe
```

并说明 sequence length projection 和 feature group projection 怎么工作。

### Day 5：读 Section 4.2

目标：

- 理解训练读路径如何隐藏 sequence lookup latency。

产出：

- 画出 DPP worker 的流水线：

```text
read primary batch N
extract UIH version keys
issue immutable UIH lookup for N
prefetch primary batch N+1
decode and concatenate UIH for N
send rebatch-ready data to trainer
```

## 第 2 周：把论文映射到你们的 partial dump

### Day 6：对照你们现有样本字段

目标：

- 区分哪些字段必须 dump，哪些字段可以 late materialize。

产出：

| 字段 | 当前是否 dump | 是否可重建 | 重建依赖 | 风险 |
| --- | --- | --- | --- | --- |
| recent UIH | 是 | 不建议 | 请求时状态 | future leakage |
| long-term UIH | 是/否 | 可以 | immutable store + version metadata | range scan latency |
| item side info | 视情况 | 可能可以 | feature version | schema evolution |
| label | 是 | 不可重建 | event logger | late arrival |

### Day 7：定义正确性验收

目标：

- 不先谈性能，先保证 reconstructed sample 等价。

产出：

- Fat Row baseline vs reconstructed sequence checksum 对比方案。
- 抽样验证口径：用户、请求时间、序列长度、feature group、缺失值。
- 未来泄露检测：任何 `event_ts > request_ts` 都应为错误。

### Day 8：定义训练吞吐验收

目标：

- 判断 late materialization 是否拖慢训练。

产出：

- 指标清单：
  - data loading latency
  - DPP CPU utilization
  - GPU starvation
  - immutable store QPS
  - lookup bandwidth
  - cache hit rate
  - network fanout

### Day 9：定义多租户收益

目标：

- 不只看旗舰长序列模型，也看中短序列模型是否被拖累。

产出：

- 三类租户评估：
  - long sequence ranking
  - mid sequence pre-ranking
  - short sequence retrieval

并分别评估它们的 read bandwidth 和 data loading latency。

### Day 10：写一份方案判断 memo

目标：

- 能向老板解释这篇文章对你们 partial dump 的启发。

建议 memo 结构：

1. 我们当前的 Fat Row / dump 问题在哪里。
2. 哪些 UIH 或 SideInfo 具有 append-only / immutable / versioned 特性。
3. 哪些字段必须保留 request-time snapshot。
4. late materialization 需要补哪些系统组件。
5. 预期收益怎么量化。
6. 最大风险是什么。

## 16. 读论文时要特别关注的判断点

### 判断点 1：你们的特征是否真的 immutable

论文方案成立的前提是 UIH 事件一旦写入，不再 retroactively 修改。你们系统里如果有会回填、纠错、删除、重算的字段，就需要额外 versioning 或 compaction 机制。

### 判断点 2：mutable window 要留多长

Mutable store retention 必须覆盖 immutable compaction cadence。论文举例：如果 daily compaction，那么 mutable tier 至少要保留一天。

### 判断点 3：training reader 是否能承受复杂性

Fat Row 简单，reader 轻；late materialization 省存储和写入，但 reader 重。是否值得取决于序列体积、训练任务数量、DPP 能力和 immutable store 吞吐。

### 判断点 4：多租户是否真的有 projection 差异

如果所有模型都读同样的全量序列，那么 projection pushdown 收益有限。它在 union dataset + heterogeneous model tenants 下最有价值。

### 判断点 5：线上收益来自“能扩长”，不是重建本身

Late materialization 本身不直接提升模型质量；它释放资源，让模型可以从 4K 扩到 10K/16K/64K。质量收益来自更长 UIH，而基础设施让这件事可行。

## 17. 建议你和团队讨论的 10 个问题

1. 我们现在训练样本中最大的重复 payload 是不是 UIH？
2. 重复 payload 的重复倍数大概是多少？是否接近论文里的 K-fold amplification？
3. 当前样本是否服务多个模型租户？不同租户序列长度差异有多大？
4. 哪些 UIH 字段是 append-only immutable，哪些字段会被回填或修正？
5. request-time 必须 snapshot 的 recent window 是多长？
6. 如果训练时按版本重建，最小 metadata schema 是什么？
7. 有没有 Fat Row baseline 可用于 checksum 对比？
8. immutable storage 的 key 和 stripe 该怎么设计？
9. DPP / dataloader 是否能做 prefetch、rebatch、affinity sharding？
10. 我们的成功指标是省资源，还是为了把序列长度推到更长后拿模型收益？

## 18. 最小 PoC 设计

如果要做一个小规模 PoC，不建议一开始就复刻论文的完整系统。可以按下面顺序推进。

### 阶段 1：正确性 PoC

目标：

- 用已有 Fat Row 样本作为 ground truth。
- 从 normalized UIH store 按 `request_ts` 重建序列。
- 比较重建结果和 Fat Row 中的 UIH 是否一致。

验收：

- sequence item id 完全一致。
- event timestamp 不超过 request timestamp。
- feature group 缺失行为符合预期。
- checksum mismatch 有可解释原因。

### 阶段 2：读放大 PoC

目标：

- 对比 Fat Row 读取和 late materialized 读取的字节量。
- 按 short/mid/long 三类模型分别测试 projection。

验收：

- 主训练数据 read bandwidth 明显下降。
- lookup bandwidth 不把总资源吃回去。
- 短序列租户能拿到显著 projection 收益。

### 阶段 3：训练吞吐 PoC

目标：

- 把 reconstruction 接入 dataloader 或 DPP。
- 看 GPU starvation 是否可控。

验收：

- data loading latency 可接受。
- GPU idle 不显著恶化。
- prefetch 和 batch grouping 能带来可测收益。

### 阶段 4：模型收益 PoC

目标：

- 在相同资源预算下，把 sequence length 从 baseline 扩到更长。

验收：

- 离线指标不因 reconstruction 劣化。
- 更长序列带来稳定离线收益。
- 如果有线上小流量，验证 consumption/engagement/topline。

## 19. 这篇文章的局限

1. 它默认 UIH 的 append-only/immutable 属性比较强；对于频繁修正的特征，需要更复杂的版本管理。
2. 它没有详细公开真实资源规模和绝对成本，只给相对指标。
3. 它的收益依赖强工程基础设施：immutable storage、DPP、prefetch、projection、sharding 缺一不可。
4. 它不解决模型侧 attention 复杂度问题，仍然需要 ULTRA-HSTU/VISTA 等模型效率方案配合。
5. 它更适合 UIH 主导样本体积的大型推荐系统，小规模系统未必值得引入完整复杂度。

## 20. 结论

这篇文章的核心价值是把“训练数据基础设施”提升成推荐模型 scaling 的一等公民。

它的逻辑链路非常清晰：

```text
长 UIH 能提升推荐质量
  -> Fat Row 让长 UIH 造成 K-fold 存储和 I/O 放大
  -> O2O 一致性必须保留，但全量预物化不是唯一办法
  -> UIH 的 append-only/immutable 特性允许按版本重建
  -> 训练样本只 dump mutable recent sequence + version metadata
  -> immutable long history 在训练时按需 materialize
  -> projection/DPP/prefetch/affinity 保证吞吐
  -> 基础设施资源下降，并让模型扩到更长序列
  -> 更长序列带来可观线上质量收益
```

如果要用一句工程语言概括：

它把长序列从“每条样本里的重复 payload”变成“一个可版本化、可投影、可按需读取的共享数据资产”。
