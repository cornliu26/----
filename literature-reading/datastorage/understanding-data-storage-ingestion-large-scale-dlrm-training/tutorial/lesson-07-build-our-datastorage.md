# 第 7 课：如果我们要做这个 datastorage，应该从哪里开始

## 1. 本课定位

前 6 课讲了论文中的问题、架构和优化。这一课把它转成一个团队可以执行的路线图。

如果老板说“想做这个 datastorage”，不要急着理解成“做一个文件系统”或“做一个缓存”。这篇论文里的 datastorage 更接近训练数据基础设施：

```text
training data platform
  + warehouse table format
  + distributed storage layout
  + online preprocessing service
  + trainer data loading
  + workload-aware scheduling
  + power and capacity model
```

本课目标是给出一个从 0 到 1 的项目拆解。

## 2. 原文短句与翻译

原文短句："DSI power requirements"

中文翻译：DSI 的功耗需求。

解释：论文最后衡量收益时，不只看吞吐，也看 DSI 对数据中心功耗的占用。因为在大规模训练中，省下来的 DSI power 可以转化为更多 trainer capacity。

## 3. 第一阶段：先做 workload characterization

最容易犯的错是先选技术方案：

```text
先做 DPP？
先改文件格式？
先加 SSD cache？
先做 feature flattening？
```

论文给我们的经验是：先量化 workload。

你需要回答三组问题。

### 3.1 训练任务读什么

| 问题 | 指标 |
| --- | --- |
| 每个 job 读哪张表 | table name |
| 读哪些 partition | partition list and bytes |
| 读哪些 feature | feature projection |
| 读多少样本 | row count |
| 是否少于一个 epoch | epoch coverage |
| 哪些 job 属于 release critical path | job type |

### 3.2 数据系统花在哪里

| 问题 | 指标 |
| --- | --- |
| storage 是否小 I/O | I/O size distribution |
| storage 是否 seek-bound | read latency and IOPS |
| DPP 是否 CPU-bound | worker CPU profile |
| DPP 是否 memory-bound | memory bandwidth and LLC miss |
| DPP 是否 network-bound | worker NIC RX/TX |
| trainer 是否 loading-bound | host CPU, memory bandwidth, NIC |
| GPU 是否等数据 | data wait time, GPU utilization |

### 3.3 feature 是否有优化空间

| 问题 | 指标 |
| --- | --- |
| feature 新增速度 | new features per week |
| feature 使用比例 | selected feature count / stored feature count |
| feature 字节占比 | selected bytes / stored bytes |
| hot feature 分布 | traffic by feature |
| co-read 模式 | feature pair access matrix |
| sparse feature 长度 | avg and p99 list length |

没有这些数据，后续方案都是拍脑袋。

## 4. 第二阶段：定义成功指标

这类项目不能只用 storage QPS 定义成功。

建议分四层指标。

### 4.1 最终业务指标

- 模型训练 wall time 是否下降。
- release candidate 训练是否更稳定。
- combo window 是否缩短。
- 同样资源下能否支持更多训练 job。

### 4.2 trainer 指标

- GPU utilization 是否提高。
- GPU data wait time 是否下降。
- step time 是否更稳定。
- trainer host CPU 和 memory bandwidth 是否可控。

### 4.3 DPP 指标

- DPP throughput 是否提高。
- 单 trainer node 所需 worker 数是否下降。
- tensor buffer depth 是否稳定非零。
- worker CPU、memory bandwidth、NIC 是否不再单点饱和。

### 4.4 storage 指标

- storage throughput 是否提高。
- small I/O 比例是否下降。
- IOPS per useful byte 是否改善。
- over-read bytes 是否下降。
- hot feature cache hit rate 是否提高。

### 4.5 datacenter 指标

- DSI power per training job 是否下降。
- storage capacity over-provision 是否下降。
- 跨 region 数据读取是否下降。
- trainer capacity 是否能增加。

最重要的思想是：

```text
storage optimization should be judged by trainer impact
trainer impact should be judged by datacenter capacity
```

## 5. 第三阶段：做最小可行架构

一个 PoC 不需要一开始复制 Meta 全套系统。可以按下面顺序做。

### 5.1 建一个训练读取 profile collector

先让每个训练任务输出：

```text
job_id
model_name
table
partitions
features
bytes_read
samples_read
preprocessing_time
trainer_wait_time
```

这个 collector 是所有后续优化的地基。

### 5.2 建一个 feature metadata catalog

为每个 feature 记录：

```text
feature_id
type
owner
coverage
avg_bytes
avg_sparse_length
read_count
read_bytes
co_read_features
lifecycle_state
```

这一步会让 feature governance、cache、reordering、projection pushdown 都有数据依据。

### 5.3 做 feature projection pushdown

如果当前训练读取会读完整 row 或完整 map，第一步要支持：

```text
model config selected features
  -> reader projection
  -> file-level selected streams
```

注意：如果物理格式不支持 feature-level stream，只做 reader 逻辑过滤无法减少 storage I/O，只能减少后续 CPU。

### 5.4 做一个简化 DPP

最小 DPP 不必一开始很复杂，但要具备三件事：

- worker 可以从 storage 读 split。
- worker 可以执行 feature transform。
- trainer 可以从 worker 拉 tensor buffer。

第一版可以先不做复杂 autoscaling，但必须打通指标：

```text
buffer depth
trainer wait time
worker resource utilization
storage read latency
```

### 5.5 做存储读路径优化

当 feature projection pushdown 后，要立刻观察小 I/O。

如果小 I/O 伤害 storage throughput，就按第 6 课的顺序尝试：

1. coalesced reads。
2. feature reordering。
3. large stripes。
4. hot feature cache。
5. in-memory flatmaps。

不要只看单个优化后的局部指标，要看端到端 GPU wait 和 DSI power。

## 6. 第四阶段：把系统做成闭环

论文的成熟点在于它不是一次性优化，而是持续 profiling 和 co-design。

一个闭环系统应该像这样：

```text
training jobs emit read profiles
  -> feature catalog updates hotness and co-read stats
  -> offline data generation writes workload-aware layout
  -> storage reader uses projection and coalesced reads
  -> DPP reports resource bottlenecks
  -> scheduler places jobs near data and DPP capacity
  -> datacenter planner updates capacity model
```

这个闭环里，每个组件都会影响其他组件。

例如：

- feature reordering 需要修改 offline data generation。
- coalesced reads 需要 storage reader 理解 feature stream offset。
- in-memory flatmaps 需要 DPP transform runtime 改数据表示。
- autoscaling 需要 trainer wait time 反馈。
- scheduler 需要知道 dataset 在哪些 region。

这就是这篇论文反复强调的端到端协同。

## 7. 项目里程碑建议

### Milestone 1：profile 可见

目标：

- 知道每个训练 job 读什么。
- 知道 GPU 是否等数据。
- 知道 DPP 或 data loader 主要资源瓶颈。

交付物：

- job read profile 表。
- feature hotness 表。
- trainer data wait dashboard。
- storage I/O size dashboard。

### Milestone 2：projection 生效

目标：

- 训练任务只读取需要的 feature。
- 能统计 selected bytes 和 over-read bytes。

交付物：

- feature projection reader。
- file-level feature metadata。
- selected feature bytes 报告。

### Milestone 3：DPP 最小闭环

目标：

- trainer 不直接做重型 preprocessing。
- worker buffer 能稳定喂 trainer。
- 能按 worker 资源和 buffer 指标扩缩容。

交付物：

- DPP Master/Worker/Client PoC。
- tensor buffer depth 指标。
- trainer wait time 对比报告。

### Milestone 4：读路径协同优化

目标：

- feature flattening 不再打爆 HDD。
- coalesced reads 和 feature reordering 降低小 I/O 和 over-read。

交付物：

- small I/O 优化报告。
- feature ordering 策略。
- storage throughput 对比。
- DPP throughput 对比。

### Milestone 5：容量和调度模型

目标：

- 能预测新增模型或新增 feature 对 DSI 的压力。
- 能把 job 调度到数据和 DPP capacity 合适的位置。

交付物：

- per-model DSI resource model。
- per-region capacity model。
- job placement 策略。
- DSI power savings 报告。

## 8. 风险和反模式

### 8.1 只做存储，不看 trainer

如果 storage QPS 提升了，但 GPU data wait 没下降，说明优化没有打到最终瓶颈。

### 8.2 只做 DPP，不改文件格式

DPP 可以从 trainer 拆走预处理，但如果文件格式导致大量 over-read，worker 仍然会浪费 CPU、网络和内存带宽。

### 8.3 只做 feature flattening，不处理小 I/O

feature flattening 很可能把 DPP 瓶颈转移到 HDD seek。没有 coalesced reads 和 layout 优化，端到端收益可能很差。

### 8.4 只看平均流量，不看 combo peak

训练 release 的峰值窗口决定资源规划。平均流量好看，不代表关键训练窗口不 stall。

### 8.5 只看单模型，不看多租户

不同模型的 feature projection、transform cost、tensor throughput 都不同。统一资源模板通常会同时造成浪费和瓶颈。

## 9. 本课检查点

读完这一课，你应该能回答：

1. 为什么做 datastorage 前必须先做 workload characterization？
2. 为什么 storage QPS 不是最终成功指标？
3. 最小 DPP PoC 应该具备哪些能力？
4. feature projection pushdown 为什么必须和物理文件格式绑定？
5. 一个成熟 DSI 系统的反馈闭环包含哪些组件？

## 10. 课后练习

请为你们自己的训练平台写一个 4 周 PoC 计划。

要求包括：

1. 每周目标。
2. 要采集的指标。
3. 要改动的组件。
4. 成功标准。
5. 最大风险。

可以参考下面模板：

| 周次 | 目标 | 指标 | 改动 | 成功标准 | 风险 |
| --- | --- | --- | --- | --- | --- |
| Week 1 | workload profile | job, feature, wait time | logging | profile 覆盖 80% job | 埋点不全 |
| Week 2 | projection reader | selected bytes | reader | over-read 可量化 | 文件格式不支持 |
| Week 3 | DPP PoC | buffer, wait time | worker/client | GPU wait 下降 | trainer 接入复杂 |
| Week 4 | read path optimize | I/O size, throughput | coalesced read | storage throughput 恢复 | HDD seek 仍瓶颈 |

最后写一句你对项目的判断：

```text
我们当前最应该先解决的是 ______，因为 ______。
```
