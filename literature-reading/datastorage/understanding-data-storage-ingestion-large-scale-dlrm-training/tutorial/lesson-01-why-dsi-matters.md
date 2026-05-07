# 第 1 课：为什么推荐训练需要单独研究 DSI

## 1. 本课定位

这一课先回答最基础的问题：

为什么一篇推荐系统训练论文会把 Data Storage and Ingestion, DSI 单独拿出来讲？

传统理解里，大模型训练的关键瓶颈通常是 GPU 算力、通信、模型并行、embedding 参数规模。但这篇论文提醒我们：当训练集群已经有几千张 GPU 或专用加速器时，数据链路会变成同等重要的系统瓶颈。GPU 能算得更快，意味着每秒需要更多训练样本、更高 tensor ingestion throughput、更大的 preprocessing capacity，也意味着存储层要承受更高读取压力。

本课要建立一个核心视角：

```text
训练系统不是 GPU alone
训练系统 = GPU trainer + data storage + online preprocessing + network + datacenter power
```

## 2. 原文短句与翻译

原文短句："data stalls"

中文翻译：数据等待导致的训练停顿。

解释：这里不是说数据系统自己慢一点而已，而是 GPU 已经准备好执行下一步训练，却因为 batch tensor 还没有准备好而空转。GPU 是昂贵且功耗敏感的资源，让它等数据，本质上是在浪费训练容量。

## 3. DSI 到底包括什么

论文里的 DSI 不只是存储。它是一条从训练样本生成到 GPU 消费的链路。

可以拆成五段：

```text
1. raw logs and feature streams
   线上业务产生行为日志、特征日志、曝光和反馈事件。

2. offline or streaming ETL
   对日志做 join、label、filter，生成结构化训练样本。

3. dataset storage
   把训练样本存成 Hive table，再落到列式文件和分布式文件系统。

4. online preprocessing
   训练时把原始样本解码、过滤、归一化、派生特征、组 batch。

5. data loading into trainers
   把预处理后的 tensor 送到 trainer host 和 GPU device memory。
```

很多系统讨论只关注第 3 段，也就是文件系统或表格式。但这篇论文的重要性在于它把第 3、4、5 段连起来看：存储格式会影响读取模式，读取模式会影响 preprocessing worker，preprocessing worker 会影响 GPU 是否等待，GPU 等待又会影响整个数据中心的训练容量。

## 4. 为什么 DSI 会成为容量瓶颈

论文给出的判断是：训练加速器一直在提高 performance per watt，但 DSI 如果没有同步优化，就会开始主导训练系统的资源开销。

它有三个层面的后果。

第一，训练吞吐被拖慢。

如果在线预处理跟不上 trainer，每个 iteration 之间就会出现等待：

```text
GPU finishes step N
  -> waits for next tensor batch
  -> DPP or trainer CPU still preprocessing
  -> GPU utilization drops
```

第二，DSI 和 trainer 竞争功耗。

数据中心 power budget 是固定的。存储节点、DPP worker、trainer host CPU、网络设备都会耗电。如果 storage + preprocessing 的 power 占比变高，可留给 GPU trainer 的 power 就下降。

第三，DSI 会限制未来 scaling。

GPU 越快，每秒需要喂入的数据越多。模型样本越宽、feature 越多、训练 job 越多，DSI 的增长速度可能比 trainer 还快。论文中提到的趋势是：训练数据规模和在线 ingestion 带宽在快速增长，DSI 不再是背景基础设施，而是训练容量的第一等约束。

## 5. 推荐训练为什么尤其容易遇到 DSI 问题

推荐训练和很多公开 benchmark 不一样。

公开 benchmark 往往像这样：

```text
fixed dataset
  -> fit on local or nearby storage
  -> many epochs
  -> repeat read
  -> optimize cache and shuffle
```

推荐训练更像这样：

```text
continuously generated logs
  -> huge warehouse tables
  -> evolving feature set
  -> less than one epoch
  -> heavy row and feature filtering
  -> online preprocessing on critical path
```

这意味着两个常见优化假设会失效。

第一个假设：数据集可以完整缓存。

生产推荐数据是 PB 到 EB 级，单个训练任务读到的 partition 也可能是 PB 级。把它们放到每台 trainer 本地不可行。

第二个假设：训练会多 epoch 重复读取相同样本。

推荐训练常常数据量很大，模型不一定需要多次扫同一份数据才能达到目标质量。它更依赖一次性读取大量新鲜样本和特征，而不是围绕同一小数据集做多轮缓存优化。

## 6. data stall 为什么是系统级问题

data stall 表面上是 trainer 的等待时间，实际会影响整条链路。

| 层级 | data stall 的含义 |
| --- | --- |
| GPU | 昂贵加速器空转，训练吞吐下降。 |
| trainer host | CPU、内存、网络可能被数据加载占满，影响训练控制路径。 |
| preprocessing worker | worker 数不足或资源瓶颈会直接传递给 trainer。 |
| storage | 小 I/O、seek、带宽不足会让 worker 无法稳定取数。 |
| datacenter | 为了弥补低效率，需要更多 storage 和 DPP capacity，挤占 power。 |

所以 DSI 的目标不是让某一个存储指标好看，而是让 GPU 持续有数据可训。

## 7. 论文给出的关键证据如何读

读 Figure 1 时，不要只看柱状图谁高谁低。要看它背后的系统含义：

```text
storage power + preprocessing power can exceed training power
  -> DSI is not a small overhead
  -> optimizing DSI can release trainer capacity
```

读 Figure 2 时，要注意两个增长方向：

```text
dataset size grows
  -> storage capacity pressure

online ingestion bandwidth grows
  -> read throughput and preprocessing pressure
```

如果只解决容量，不解决 throughput，GPU 仍然会等数据。如果只解决 throughput，不解决容量和功耗，数据中心仍然不可持续。

## 8. 工程化拆解：一个团队该如何确认自己是否有 DSI 问题

可以从四组指标开始。

第一组，trainer 侧：

- GPU utilization。
- GPU step time。
- data wait time。
- host CPU utilization。
- host memory bandwidth。
- frontend NIC throughput。

第二组，preprocessing 侧：

- worker CPU utilization。
- worker memory capacity。
- worker memory bandwidth。
- worker network RX/TX。
- tensor buffer depth。
- per-feature transform cost。

第三组，storage 侧：

- read throughput。
- IOPS。
- I/O size distribution。
- seek-heavy read 比例。
- per-feature bytes。
- hot feature bytes。

第四组，任务和数据侧：

- 每个 job 读哪些 partition。
- 每个 job 读哪些 feature。
- feature coverage。
- sparse feature length。
- feature 新增、实验、上线、废弃速度。

这些 profile 不是锦上添花。没有它们，就很难判断问题到底在 trainer host、DPP worker、storage HDD IOPS，还是文件布局。

## 9. 本课检查点

读完这一课，你应该能回答：

1. DSI 为什么不等于单纯的数据存储？
2. data stall 为什么会直接影响 GPU 训练容量？
3. 为什么推荐训练的数据假设和公开 benchmark 不一样？
4. 为什么 DSI 会和 trainer 竞争数据中心 power budget？
5. 判断一个团队是否需要投入 DSI 优化，最先应该采哪些指标？

## 10. 课后练习

请用你熟悉的一条训练链路画出下面的表。

| 环节 | 当前系统组件 | 主要资源 | 可能瓶颈 | 已有监控 |
| --- | --- | --- | --- | --- |
| 样本生成 |  | CPU / shuffle / queue |  |  |
| 数据存储 |  | capacity / IOPS / bandwidth |  |  |
| 数据读取 |  | network / decode / seek |  |  |
| 在线预处理 |  | CPU / memory bandwidth |  |  |
| tensor 加载 |  | host CPU / NIC / PCIe |  |  |
| GPU 训练 |  | GPU compute / communication |  |  |

完成后回答：你们最像论文里的哪个问题，storage、preprocessing、data loading，还是 datacenter power？
