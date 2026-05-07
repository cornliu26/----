# Data Storage and Ingestion for Large-Scale DLRM Training 教学大纲

论文：Understanding Data Storage and Ingestion for Large-Scale Deep Recommendation Model Training

本地文件：`paper.pdf`

会议：ISCA 2022 Industry Product

一句话总结：

这篇文章不是在介绍一个单点存储系统，而是在解释 Meta 如何把推荐模型训练中的数据生成、数据仓库、分布式存储、在线预处理、trainer 数据加载和数据中心容量规划视为同一条 Data Storage and Ingestion, DSI 链路来设计。它的核心观点是：当 GPU/DSA 越来越强时，真正限制大规模推荐训练的可能不是训练算力，而是训练数据如何被存、被筛、被预处理、被持续喂给 GPU。

如果老板说“这是她想做的 datastorage”，这里的 datastorage 大概率不是单纯做一个文件系统，而是做训练数据基础设施：

```text
raw logs / feature streams
  -> offline ETL / labeling
  -> warehouse table
  -> columnar distributed storage
  -> online preprocessing service
  -> tensors
  -> GPU trainer
```

这篇文章的教学主线可以概括为：

```text
GPU 集群越来越强
  -> DSI 变成瓶颈
  -> 推荐训练数据和公开 benchmark 很不一样
  -> 需要把存储和在线预处理从 trainer 中拆出来
  -> 用 DPP + DWRF/Tectonic + 多维系统优化来消除 data stalls
  -> 最终用吞吐、功耗、容量和调度效率衡量收益
```

## 1. 这篇文章适合谁读

这篇文章适合下面几类同学：

- 推荐系统训练平台同学：关心训练样本、特征、label、数据读取和 GPU 利用率。
- 特征平台和数据仓库同学：关心 PB/EB 级训练数据如何组织成表、列、文件、stripe 和 stream。
- 分布式存储同学：关心 HDD/SSD、IOPS、吞吐、cache、feature filtering 和小 I/O。
- 训练框架同学：关心 PyTorch Dataset、数据加载、预处理、tensor buffer 和 data stall。
- 数据中心架构同学：关心 power budget、跨机房数据复制、训练任务调度和 DSI 资源规划。

不建议把它当作一篇“推荐模型结构论文”来读。它不重点讨论 DLRM 的网络结构，而是讨论 DLRM 训练背后的数据系统。

## 2. 论文在回答什么问题

核心问题：

当推荐模型训练已经跑在几千甚至上万张 GPU/DSA 上时，训练数据系统如何不拖后腿？

这个问题可以拆成四个更具体的问题：

1. 数据到底有多大？
   - 论文中的生产训练数据已经是 PB 到 EB 级。
   - 模型训练需要从中心化数据仓库中持续读取，而不是把数据提前放在 trainer 本地。

2. 为什么传统训练数据加载不够？
   - 公开 benchmark 常常假设数据集相对稳定，可以多 epoch 反复读取。
   - 工业推荐训练往往是少于一个 epoch，但每次读取的数据极大，并且需要 row-wise 和 feature-wise filtering。

3. 为什么不能直接在 trainer CPU 上做预处理？
   - 在线预处理在训练关键路径上。
   - 如果 CPU、内存带宽或前端网络跟不上，GPU 会等待数据。

4. 如何从系统层面解决？
   - 把在线预处理做成 disaggregated service。
   - 让存储格式支持 feature filtering。
   - 用 workload profile 指导 feature flattening、coalesced reads、feature reordering、large stripes 和 in-memory flatmaps。
   - 从数据中心层面做容量规划和任务调度。

## 3. 原文关键词与中文解释

下面只保留极短原文片段，用来让后续教程能和论文原文对齐。

| 原文短句 | 中文解释 |
| --- | --- |
| "data storage and ingestion" | 不只是存储，还包括从原始样本读出、预处理并送到训练框架的整条链路。 |
| "eliminate data stalls" | 目标不是让数据系统单独跑得快，而是让 GPU 不因为等数据而空转。 |
| "feature flattening" | 把 map 里的 feature 拆成文件层面的逻辑列，让训练任务只读需要的 feature。 |
| "coalesced reads" | 把多个小读合并成较大的连续读，减少 HDD seek 带来的 IOPS 问题。 |
| "feature reordering" | 按近期训练任务的 feature 热度重排存储布局，让常一起读的 feature 更接近。 |

## 4. 挑战一：DSI 已经成为训练容量瓶颈

论文开篇的判断很直接：训练加速器越来越强，但训练数据系统没有同步变强。推荐训练中的 DSI 包括：

- offline data generation：从线上日志和 feature stream 生成训练样本。
- dataset storage：把样本存成 Hive 表，再落到分布式文件系统。
- online preprocessing：训练时把原始行、压缩列和 feature map 转成模型要吃的 tensor。
- data loading：把 tensor 从 preprocessing worker 或 trainer host 搬到 GPU。

论文给出的重要证据包括：

- 某些模型下，storage + preprocessing 的功耗可以超过 GPU trainer 本身。
- 推荐训练数据存储规模已经到 exabyte 量级。
- 在线数据摄入带宽达到 hundreds of Tbps 量级。
- 过去两年内，数据集规模增长超过 2x，在线 ingestion 带宽增长超过 4x。

教学时要强调：这不是“数据平台有点贵”的问题，而是固定数据中心 power budget 下，DSI 会直接挤占 trainer 能用的功耗和容量。

## 5. 挑战二：工业推荐训练不是公开 benchmark

很多训练系统优化默认数据集可以被重复读取，典型模式是：

```text
small or medium dataset
  -> many epochs
  -> cache / shuffle / augment
  -> train until convergence
```

但论文描述的 Meta 推荐训练更接近：

```text
huge evolving dataset
  -> less than one epoch
  -> read selected partitions
  -> read selected features
  -> heavy online preprocessing
  -> continuous model release pipeline
```

这带来几个差异：

- 数据是持续生成的，不是静态 benchmark。
- 数据表持续演化，feature 会新增、实验、上线、废弃。
- 单个训练任务只读一部分 partition，但这部分仍然是 PB 级。
- 单个训练任务只读少量 feature，但这些 feature 往往覆盖率高、字节占比高、对模型质量重要。
- 多个训练任务之间存在热门 feature 和热门 bytes 的复用。

论文中的关键数据：

| 观察点 | 论文结果 | 教学解释 |
| --- | --- | --- |
| 使用的 partition 规模 | RM1 约 11.95 PB，RM2 约 25.94 PB，RM3 约 1.95 PB | 单个训练任务也已经远超 trainer 本地盘容量。 |
| 每个任务使用的 feature 比例 | 约 9% 到 11% | 训练任务不会读完整表，需要强 feature filtering。 |
| 每个任务使用的 bytes 比例 | 约 21% 到 37% | 虽然 feature 数少，但热门 feature 通常更大、更密。 |
| 热门 bytes 复用 | 约 18% 到 39% 的 bytes 可贡献 80% storage traffic | 给 hot feature cache、SSD tier 和 feature reordering 提供依据。 |
| RM1 新 feature | 6 个月内提出 14614 个新 feature | 数据格式必须支持频繁 feature 演化。 |

这里的教学重点是：工业推荐数据系统不能只优化“顺序扫一整个文件”，而要优化“从巨大的、不断变化的表里，按任务需求高效读一小部分 feature 和 row”。

## 6. 挑战三：在线预处理在训练关键路径上

推荐训练样本不是从磁盘读出来就能直接给模型。读路径需要做：

- 解密、解压、重建存储 chunk。
- DWRF/ORC 类列式文件解码。
- 按 feature projection 过滤。
- 稠密 feature 归一化。
- 稀疏 feature hash、clip、sort、intersection、NGram 等。
- 派生新 feature。
- batch 成 tensor。
- 送到 trainer/GPU。

如果这些工作放在 trainer CPU 上，GPU 会等数据。

论文中的实验证据：

- 在 RM1 上，trainer host 负责预处理时，GPU stall time 达到 56%。
- 此时 CPU 利用率约 92%，memory bandwidth 利用率约 54%。
- 每个 8-GPU trainer node 的 tensor ingestion 需求差异很大：RM1 约 16.50 GB/s，RM2 约 4.69 GB/s，RM3 约 12.00 GB/s。
- 论文预计未来两年 online preprocessing throughput 需求还会增加约 3.5x。

教学时可以把这个挑战讲成一句话：

```text
GPU 越快，数据预处理越容易变成训练主瓶颈。
```

## 7. 解决方案一：端到端 DSI 架构

论文展示的 Meta DSI 架构可以分成四层。

### 7.1 数据生成层

线上服务产生 raw feature logs 和 event logs，进入 Scribe / LogDevice 一类日志系统。Streaming 或 batch ETL 负责 join、label、filter，最终生成结构化训练样本。

这部分不是训练 critical path，但它决定了训练数据的正确性、可复用性和 feature 演化速度。

### 7.2 数据仓库与存储层

训练样本存成 Hive table。每行包含 feature 和 label，feature 占绝大多数字节。为了兼容大量模型和大量 feature，论文把 feature 存成 map column：

```text
dense feature:
  map<feature_id, float>

sparse feature:
  map<feature_id, list<int>>

weighted sparse feature:
  map<feature_id, map<int, float>>
```

底层文件格式是 DWRF，类似 ORC，是列式格式。文件写入 Tectonic，这是 Meta 的 append-only 分布式文件系统。

教学重点：

- Hive/map schema 让 feature 工程更灵活。
- 但 map schema 如果直接存，会导致训练读取时 over-read。
- 所以存储层要做 feature flattening，让 map 里的 feature 在文件层面成为可单独读取的逻辑列。

### 7.3 在线预处理层

论文提出 Data PreProcessing Service, DPP。DPP 是一个 disaggregated online preprocessing service。

DPP 的目的不是跑一次离线 ETL，而是跟着训练任务持续生产 mini-batch tensor：

```text
DPP Worker:
  read raw bytes from storage
  -> decompress / decrypt / reconstruct
  -> decode rows
  -> filter features
  -> apply transforms
  -> batch tensors
  -> serve tensors to trainer
```

### 7.4 Trainer 数据加载层

每个 trainer node 上有 DPP Client。PyTorch runtime 调用 client 获取预处理后的 tensor。参数同步走 backend network，数据加载走 frontend network，二者逻辑上分开。

这意味着训练框架不需要自己管理成百上千个 preprocessing worker，但仍然可以持续得到 tensor。

## 8. 解决方案二：DPP 如何消除 data stalls

DPP 的核心设计可以分成 control plane 和 data plane。

### 8.1 DPP Master

DPP Master 接收训练任务的 session specification，包括：

- 要读哪张表。
- 要读哪些 partition。
- 要读哪些 feature。
- 每个 feature 要做哪些 transformation。

然后它把 PB 级 preprocessing workload 切成 split。每个 split 是独立 work item，DPP Worker 可以并行处理。

DPP Master 还负责：

- checkpoint reader state。
- 监控 worker health。
- 重启失败 worker。
- 根据 CPU、内存、网络和 buffered tensor 数量做 autoscaling。

autoscaling 的目标不是 worker 越多越好，而是在不浪费资源的情况下保持 trainer 不缺 tensor。

### 8.2 DPP Worker

DPP Worker 是 stateless 的，因此可以水平扩展。它只需要：

- 从 Master 拉 split。
- 从存储读数据。
- 执行 extraction 和 transformation。
- 产出 tensor buffer。
- 给有限数量的 DPP Client 提供 tensor。

stateless 设计让 worker 失败恢复简单，也让训练任务可以按吞吐需求右置资源。

### 8.3 DPP Client

DPP Client 跑在 trainer node 上，对 PyTorch 暴露数据读取 hook。它通过 RPC 从 worker tensor buffer 拉 batch。

教学重点：

```text
Trainer 不再负责重型 preprocessing。
Trainer 只负责稳定、高速地加载已经预处理好的 tensor。
```

## 9. 解决方案三：面向 feature filtering 的存储优化

论文最有工程味的一部分是 Section 7.5。它说明只做 feature flattening 还不够，因为存储和 preprocessing 是一个端到端系统。

### 9.1 Feature flattening

原始 Hive 表为了支持 feature 演化，把 feature 放在 map 里。这对工程师友好，但读取时可能必须读整行。

Feature flattening 把 map key，也就是 feature ID，在文件层变成单独 stream。训练任务只读自己需要的 feature。

收益：

- DPP Worker throughput 提升到 2.00x。
- 代价是存储容量增加约 12%。

新的问题：

- 小 I/O 变多。
- HDD 上 seek 过多。
- storage throughput 反而下降到 0.03x。

这里非常适合教学：一个局部优化会把瓶颈从 CPU/preprocessing 转移到 HDD IOPS。

### 9.2 Coalesced reads

为了解决大量小读，论文把相邻 feature stream 合并成较大的读。这样会读入一些不需要的 feature，但能摊薄 seek 成本。

收益：

- storage throughput 从 0.03x 恢复到约 0.99x。

代价：

- 会 over-read 一些不需要的 feature。

### 9.3 Feature reordering

Coalesced reads 的 over-read 取决于 feature 在文件中的顺序。如果常一起被读的 feature 相隔很远，中间会夹着很多不需要的 feature。

论文用近期训练任务中的 feature popularity 来重排 feature stream，使热门且常用的 feature 更接近。

收益：

- storage throughput 在 coalesced reads 基础上继续提升约 84%。

### 9.4 Large stripes

更大的 stripe 能增加平均 I/O size，进一步减少 seek 压力。

收益：

- storage throughput 再提升约 31%。

### 9.5 In-memory flatmaps

原先数据提取路径可能在列式 DWRF、行式 map 和 tensor 格式之间反复转换。论文把 DPP Worker 内存表示改成更贴近 DWRF 和 tensor 的 flatmap。

收益：

- DPP Worker throughput 提升约 15%。
- 主要原因是减少格式转换和内存拷贝，降低 memory bandwidth 压力。

### 9.6 Localized optimizations

包括去掉不必要的 null check、使用 LTO、AutoFDO 等编译优化。

收益：

- DPP Worker throughput 再提升约 28%。

教学重点：

```text
这篇论文真正想传达的是 top-to-bottom + end-to-end co-design。

top-to-bottom:
  从模型和 feature 需求一路考虑到底层 HDD/SSD/CPU/NIC/memory bandwidth。

end-to-end:
  从数据生成、文件布局、读取方式、DPP 内存格式到 trainer 数据加载一起优化。
```

## 10. 收益如何

论文给出的收益不是单一指标，而是覆盖吞吐、功耗、GPU 利用率和数据中心容量。

| 优化或系统设计 | 论文中的收益 | 应该如何理解 |
| --- | --- | --- |
| DPP disaggregation | 用外部 worker 承担 extraction 和 transformation，目标是消除 trainer data stalls | 让 GPU 不等数据，trainer CPU 不再被重型预处理压垮。 |
| Feature flattening | DPP throughput 达到 2.00x，但 storage throughput 暂时跌到 0.03x | 说明 selective read 有用，但会暴露 HDD 小 I/O 瓶颈。 |
| Coalesced reads | storage throughput 恢复到约 0.99x | 用较大读摊薄 seek 成本。 |
| Feature reordering | storage throughput 继续提升约 84% | 利用 feature 热度，让 coalesced read 少读无用字节。 |
| Large stripes | storage throughput 继续提升约 31% | 用更大 stripe 提高平均 I/O size。 |
| In-memory flatmaps | DPP throughput 提升约 15% | 减少列式、行式、tensor 表示之间的转换和拷贝。 |
| Localized optimizations | DPP throughput 提升约 28% | 低层代码优化在规模化系统中也能转化为容量收益。 |
| 总体 co-design | DPP throughput 2.94x，storage throughput 2.41x，DSI power requirement 降低 2.59x | 这才是 datacenter 视角最重要的收益：同样功耗预算下，可以留更多资源给 trainer。 |

## 11. 可以如何组织成 7 课教程

### Lesson 01：为什么推荐训练需要单独研究 DSI

本课定位：

先把读者从“训练就是 GPU 算矩阵”拉到“训练是 GPU + 数据系统 + 数据中心资源”的视角。

要讲清楚：

- DSI 包括什么。
- 为什么 DSI 会消耗大量 power。
- 为什么 DSI 会限制 trainer capacity。
- 为什么大规模推荐训练比 CV/NLP benchmark 更依赖数据链路。

关键图表：

- Figure 1：storage/preprocessing/training power。
- Figure 2：dataset size 和 online data ingestion bandwidth 增长。

检查点：

- 能否解释 DSI 为什么会和 GPU trainer 竞争数据中心功耗？
- 能否用一句话说明 data stall 为什么昂贵？

### Lesson 02：Meta 推荐训练工作负载长什么样

本课定位：

讲 training job 不是孤立运行，而是在多人协作、周期性 release、全球数据中心中连续运行。

要讲清楚：

- exploratory job、combo job、release candidate job 的关系。
- combo job 为什么造成训练和 DSI 峰值。
- 为什么跨 region 带宽约束会要求 dataset 和 trainer co-locate。
- feature 为什么会高速演化。

关键图表：

- Figure 4：combo job duration 和 status。
- Figure 5：一年内训练 compute peak。
- Figure 6：不同模型在不同 region 的需求。
- Table 2：6 个月内 RM1 feature 演化。

讨论题：

- 如果一个 region 没有某模型的数据副本，训练调度会遇到什么问题？
- 训练 job scheduler 是否应该同时理解模型算力需求和数据位置？

### Lesson 03：工业推荐数据如何存

本课定位：

讲 Hive table、map feature schema、DWRF、Tectonic，以及为什么 feature filtering 是第一性需求。

要讲清楚：

- dense feature 和 sparse feature 如何用 map column 表达。
- 为什么 map schema 对 feature engineering 友好。
- 为什么每个模型只读一小部分 feature。
- 为什么 PB 级数据不能假设在 trainer local storage。

关键图表：

- Table 3：不同 RM 的 PB 级 partition size。
- Table 4：模型实际需要的 feature 数。
- Table 5：数据表记录的 feature 数、coverage、used bytes。
- Table 6：feature filtering 带来的小 I/O size。

作业：

设计一个训练数据表 schema，同时满足：

- feature ID 可频繁新增。
- 训练任务可按 feature projection 读取。
- 文件层能支持列式压缩。
- 读路径能统计每个 feature 的热度。

### Lesson 04：为什么要有 DPP

本课定位：

讲在线预处理为什么不能简单放在 trainer CPU 上。

要讲清楚：

- online preprocessing 和 offline ETL 的区别。
- 为什么在线预处理在训练 critical path 上。
- trainer host CPU、memory bandwidth、network 分别会怎样成为瓶颈。
- DPP Master、Worker、Client 的职责。

关键图表：

- Table 7：RM1 的 GPU stall time。
- Table 8：每个 8-GPU trainer node 的 throughput。
- Figure 8：trainer frontend CPU 和 memory bandwidth。

作业：

给一个训练任务设计 DPP autoscaling 指标：

- worker CPU。
- worker memory。
- worker network RX/TX。
- tensor buffer depth。
- trainer data wait time。

要求说明每个指标对应什么扩缩容决策。

### Lesson 05：预处理到底慢在哪里

本课定位：

把 preprocessing 拆成 extraction、transformation、loading 三类成本，解释为什么 memory bandwidth 会成为长期瓶颈。

要讲清楚：

- DPP Worker 需要多少台机器才能喂饱一个 trainer node。
- 为什么 extraction 端网络比 tensor loading 端网络更重。
- 为什么 feature generation 通常比 normalization 更贵。
- 为什么 GPU/FPGA/SmartNIC 加速并不简单。

关键图表：

- Table 9：每个 DPP Worker 的吞吐和每个 trainer 所需 worker 数。
- Figure 9：DPP CPU、memory、memory bandwidth 利用率。
- Table 10：compute node 硬件规格变化。
- Table 11：常见 preprocessing transformations。

重要结论：

- RM1/RM2/RM3 的瓶颈不同，所以 preprocessing resource 必须按模型 right-size。
- feature generation 约占 transformation cycles 的 75%。
- 后续硬件中 compute 和 NIC 可能比 memory bandwidth 增长更快，所以 memory bandwidth 会更关键。

### Lesson 06：存储格式和读取路径如何协同优化

本课定位：

以 Section 7.5 为核心，讲从一个 naive map schema 到高效 selective read 的优化链路。

要讲清楚：

- Feature flattening 如何减少 over-read。
- 为什么 feature flattening 会制造小 I/O 和 seek 压力。
- Coalesced reads 如何在 selective read 和 HDD IOPS 之间折中。
- Feature reordering 如何利用训练任务热度。
- Large stripes、in-memory flatmaps 和 localized optimization 如何继续补收益。

关键图表：

- Figure 10：regular map、feature flattening、coalesced read、feature reordering 的读路径对比。
- Table 12：逐步优化后的 DPP throughput 和 storage throughput。

课堂练习：

给定一组模型的 feature projection 和每个 feature 的大小、热度，设计一种 feature ordering 策略，让 coalesced reads 的 over-read 最小。

### Lesson 07：如果我们要做这个 datastorage，应该从哪里开始

本课定位：

把论文抽象成可落地的项目路线，而不是停留在“Meta 做得很大”。

建议路线：

1. Workload characterization
   - 统计每个训练任务读取哪些 partition。
   - 统计每个训练任务读取哪些 feature。
   - 统计 feature bytes、coverage、sparse length。
   - 统计 trainer data wait time。
   - 统计 preprocessing CPU、memory bandwidth、network。

2. 数据表示和文件布局
   - 明确 raw row schema。
   - 支持 feature map 的灵活演化。
   - 在文件层支持 feature flattening。
   - 输出 per-feature stream metadata。

3. 读路径优化
   - 实现 feature projection pushdown。
   - 统计小 I/O 分布。
   - 实现 coalesced reads。
   - 用近期训练任务热度做 feature reordering。

4. 在线预处理服务
   - Master 切 split。
   - Worker stateless 执行 extract + transform。
   - Client 给训练框架提供 tensor。
   - 用 tensor buffer depth 和 trainer wait time 做 autoscaling。

5. 性能闭环
   - 以 GPU stall time 为最终指标。
   - 以 DPP throughput、storage throughput、DSI power、worker 数量为中间指标。
   - 每次优化都要检查是否把瓶颈从一层转移到了另一层。

## 12. 这篇论文对“老板想做的 datastorage”的启发

如果要把它映射到一个真实团队的方向，可以拆成五个问题。

### 12.1 我们是否知道训练数据读路径的真实画像

需要回答：

- 每个模型读多少数据？
- 每个模型读哪些 feature？
- 哪些 feature 是热的？
- 单次 I/O size 分布是什么？
- 训练期间 GPU 等数据多久？
- preprocessing 主要耗在 CPU、内存带宽还是网络？

如果这些都没有量化，直接做存储系统很容易优化错方向。

### 12.2 我们的数据格式是否支持 feature-level selective read

如果训练样本是一个大 JSON、大 protobuf 或大 map blob，模型只要 10% feature，也可能被迫读完整 row。

论文的启发是：逻辑 schema 可以对 feature engineering 友好，但物理格式必须对读取友好。

### 12.3 我们是否需要 disaggregated preprocessing

判断标准：

- trainer CPU 是否被 data loading/preprocessing 打满？
- GPU 是否经常 data stall？
- 不同模型的 preprocessing 需求是否差异很大？
- 是否需要按训练任务动态扩缩 preprocessing worker？

如果答案是肯定的，DPP 这种架构就很有参考价值。

### 12.4 我们是否能接受端到端协同优化

这篇文章的优化不是单点式的。Feature flattening 提升 DPP，却伤害 HDD throughput；coalesced reads 恢复 storage throughput，却引入 over-read；feature reordering 又要修改数据生成路径。

所以团队要能跨越：

- 训练框架。
- 预处理服务。
- 文件格式。
- 离线数据生成。
- 分布式存储。
- 数据中心资源规划。

### 12.5 我们应该用什么指标定义成功

不要只看存储 QPS。更合理的指标是：

- GPU stall time 是否下降。
- 每个 trainer node 的 tensor ingestion 是否稳定。
- DPP worker 数是否下降。
- storage throughput 是否提高。
- small I/O seek 问题是否缓解。
- DSI power per training job 是否降低。
- 同样 power budget 下 trainer capacity 是否增加。

## 13. 教学版结论

这篇文章的核心不是“Meta 有很大的数据”，而是：

```text
推荐训练的数据基础设施已经变成训练系统的一等公民。
```

它解决的问题是：

- 大规模推荐训练中的数据存储和摄入瓶颈没有被充分研究。
- 工业推荐数据和公开 benchmark 差异巨大。
- Trainer 本地 CPU 预处理会导致 GPU 等数据。
- 灵活 feature schema 和高效 selective read 之间存在冲突。
- DSI 的瓶颈会跨 storage、network、CPU、memory bandwidth 和 datacenter power 迁移。

它的解决方式是：

- 用 Hive + DWRF + Tectonic 存放结构化训练样本。
- 用 DPP 把在线预处理从 trainer 中拆出来，并支持 autoscaling。
- 用 feature flattening 支持 feature-level filtering。
- 用 coalesced reads、feature reordering、large stripes 解决小 I/O 和 HDD seek。
- 用 in-memory flatmaps 和 localized optimizations 降低 DPP 内部开销。
- 用 workload profiling 做数据中心容量规划和系统优化。

它的收益是：

- 让 GPU 更少等待数据。
- 让 preprocessing 资源可以按模型需求 right-size。
- 把 DPP throughput 提高到 2.94x。
- 把 storage throughput 提高到 2.41x。
- 把 DSI power requirement 降低 2.59x。

如果后续要扩成完整教程，可以按上面的 7 课结构展开。每课都应该保留一个原则：不要只讲系统组件，要把它和“哪个训练瓶颈被解决、瓶颈如何迁移、收益如何量化”绑定起来。
