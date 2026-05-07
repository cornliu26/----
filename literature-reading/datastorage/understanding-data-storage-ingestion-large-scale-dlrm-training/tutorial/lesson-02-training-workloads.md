# 第 2 课：Meta 推荐训练工作负载长什么样

## 1. 本课定位

这一课解释论文为什么要先讲 coordinated training at scale。

如果只看单个训练 job，可能会以为 DSI 只需要满足一个模型的一次读取。但生产推荐训练不是这样。它是很多工程师、很多模型、很多实验、很多 release candidate、多个 region 和多个 datacenter 共同构成的持续训练过程。

这带来一个关键事实：

```text
DSI capacity 要按训练峰值和数据位置规划，而不是只按平均流量规划。
```

## 2. 原文短句与翻译

原文短句："combo jobs"

中文翻译：组合实验训练任务。

解释：多个工程师会把各自的模型改动、feature 改动和实验想法合并到一组候选训练任务中，用来决定下一版生产模型。这类 job 往往集中出现，造成训练算力和 DSI 资源峰值。

## 3. 推荐模型训练不是一个 job，而是一条 release pipeline

论文描述的训练过程大致包含三类 job。

第一类，exploratory job。

工程师为了验证一个 feature、一个 transformation、一个模型结构或一个训练策略，会持续启动探索性训练。它们数量多、生命周期不稳定，效果不好就会被停掉。

第二类，combo job。

当进入模型 release 阶段，多个想法会被组合在一起训练。combo job 在 release critical path 上，因为它决定哪些改动进入候选版本。

第三类，release candidate job。

候选模型需要更稳定、更完整的训练和评估。如果成功，就可能进入下一版生产模型。

这三类 job 会形成一个特征：

```text
持续训练 + 周期性峰值 + 大量异步任务
```

这对 DSI 很麻烦。因为系统不能只服务一个稳定、同步、可预测的训练任务，而要服务大量异步且需求不同的训练任务。

## 4. 为什么 combo job 会制造 DSI 峰值

combo job 在同一时间窗口里集中启动，通常对应一轮模型 release 的关键阶段。它的系统含义是：

```text
many models or variants start training
  -> many datasets are read
  -> many preprocessing workers are needed
  -> many trainer nodes demand tensors
  -> DSI peak rises
```

如果数据中心只按全年平均利用率配置 DSI，combo window 到来时就会出现资源不足。资源不足不一定表现为 job 直接失败，更常见的是：

- trainer 等数据。
- preprocessing worker 排队。
- storage read latency 上升。
- cross-region 数据读取变多。
- release candidate 训练延迟。

这会直接影响模型迭代速度。

## 5. 为什么数据位置很重要

论文强调全球训练基础设施是 geo-distributed 的。训练任务分布在多个 region 和 datacenter，但数据并不能像计算任务一样随便跨 region 读取。

原因有三个。

第一，跨 region 带宽有限。

训练任务读的是 PB 级数据。如果一个 region 的 trainer 频繁跨 region 拉数据，网络会很快成为瓶颈。

第二，跨 region 数据读取会增加延迟和不确定性。

训练需要稳定持续的 batch ingestion。即使平均带宽足够，跨 region 抖动也可能造成 data stall。

第三，数据复制本身也有成本。

如果每个 region 都复制所有模型的数据，存储成本很高。如果只复制部分数据，调度器就必须理解哪个 job 可以放到哪里跑。

因此，训练调度不能只看 GPU 空闲情况，还要看数据是否 co-located。

## 6. feature engineering 为什么改变存储设计

论文里 Table 2 展示了 RM1 在 6 个月内有大量 feature 被提出、实验、上线或废弃。这说明推荐训练数据表不是稳定 schema。

这对系统设计有两层影响。

第一，逻辑 schema 必须支持快速 feature 演化。

如果每新增一个 feature 都要改固定 schema、重写大量表定义、调整下游 reader，feature engineering 会变慢。所以论文使用 map column 来承载大量 dense 和 sparse feature。

第二，物理存储不能因为逻辑灵活而牺牲读取效率。

map schema 对写入和演化友好，但训练任务通常只读一小部分 feature。如果物理层把整个 map 当成一个 blob，训练就会 over-read 大量无用 bytes。

这就是后面 feature flattening 的前置动机：

```text
logical schema should be flexible
physical layout should be selectively readable
```

## 7. 多模型共享基础设施带来的调度问题

生产环境有很多 RM，论文用 RM1、RM2、RM3 代表不同类型的推荐模型。它们的计算需求、数据需求、预处理需求都不一样。

这意味着 DSI scheduler 面临三类不均衡。

第一，模型之间不均衡。

有的模型样本更宽，有的模型 transformation 更重，有的模型 tensor ingestion 更高。

第二，region 之间不均衡。

某些模型的训练需求集中在某些 region。如果全局调度只追求计算均衡，就可能导致数据跨 region 读取。

第三，时间上不均衡。

combo window 会让某些时间段需求暴涨。探索任务又会异步启动和停止，难以用固定配额精确匹配。

因此，DSI 设计不能假设训练任务同步、同质、稳定。

## 8. 工程化拆解：如何给自己的训练平台建 workload profile

如果要借鉴这篇论文，第一件事不是马上做 DPP 或改文件格式，而是建立 workload profile。

建议从下面几张表开始。

### 8.1 job profile

| 字段 | 含义 |
| --- | --- |
| job_id | 训练任务标识 |
| model_name | 模型或业务线 |
| job_type | exploratory / combo / release candidate / backfill |
| start_time | 启动时间 |
| end_time | 结束时间 |
| trainer_nodes | 使用多少 trainer |
| target_dataset | 读取哪张表 |
| partitions | 读取哪些 partition |
| feature_projection | 读取哪些 feature |

### 8.2 peak window profile

| 指标 | 目的 |
| --- | --- |
| hourly trainer demand | 找训练峰值 |
| hourly storage throughput | 找存储峰值 |
| hourly DPP worker demand | 找预处理峰值 |
| per-region dataset reads | 判断是否跨 region 拉数 |
| failed or killed jobs | 判断探索任务浪费和资源抖动 |

### 8.3 feature lifecycle profile

| 指标 | 目的 |
| --- | --- |
| new features per week | 衡量 schema 演化速度 |
| active features | 衡量生产稳定特征 |
| experimental features | 衡量实验压力 |
| deprecated features | 衡量清理机制 |
| bytes per feature | 衡量存储成本 |
| read frequency per feature | 衡量热度和 cache 价值 |

有了这些 profile，才能判断数据系统最应该先优化哪里。

## 9. 本课检查点

读完这一课，你应该能回答：

1. exploratory job、combo job、release candidate job 分别是什么？
2. combo job 为什么会造成 DSI 峰值？
3. 为什么训练任务调度不能只看 GPU 空闲？
4. feature 高频演化为什么会影响训练数据 schema？
5. 为什么逻辑 schema 灵活性和物理读取效率之间存在张力？

## 10. 课后练习

请设计一个简单的训练任务调度规则，输入包括：

- 每个 region 的 GPU 空闲量。
- 每个 region 是否有目标 dataset 副本。
- 每个 region 的 storage read pressure。
- 每个 region 的 DPP worker pressure。
- job 类型是否是 release critical path。

要求输出：

```text
job -> chosen region
```

并解释：什么时候你愿意牺牲 GPU 利用率，换取数据本地性？
