# 第 4 课：为什么要有 DPP

## 1. 本课定位

这一课讲论文的核心系统组件：Data PreProcessing Service, DPP。

在很多训练系统里，数据读取和预处理由 trainer host CPU 完成。对于小模型或中等数据集，这样做简单直接。但在大规模推荐训练里，样本宽、feature 多、transform 重、GPU throughput 高，trainer CPU 很容易成为瓶颈。

DPP 的核心思想是：

```text
把重型在线预处理从 trainer 中拆出去
让 trainer 只负责稳定加载已经预处理好的 tensor
```

## 2. 原文短句与翻译

原文短句："online preprocessing"

中文翻译：训练时在线预处理。

解释：这里的预处理不是离线生成数据集时的一次性 ETL，而是在训练 critical path 上持续把 raw samples 转成 tensor mini-batch。它必须跟上 GPU 消费速度。

## 3. online preprocessing 和 offline ETL 的区别

offline ETL 负责从原始日志生成训练样本。它可以使用 Spark、streaming engine 等传统数据处理系统。

online preprocessing 负责在训练过程中把存储里的样本转成模型输入 tensor。

两者差异很大。

| 维度 | offline ETL | online preprocessing |
| --- | --- | --- |
| 时间位置 | 训练前或数据生成阶段 | 训练 critical path |
| 目标 | 生成结构化样本 | 持续喂 trainer |
| 工作粒度 | 大批量数据处理 | mini-batch 局部处理 |
| 延迟敏感 | 相对较低 | 很高 |
| 扩缩容目标 | 尽快完成任务 | 正好匹配 GPU throughput |
| 失败影响 | 数据生成延迟 | GPU 直接等待 |

把 online preprocessing 当成普通 ETL，会忽略一个关键约束：它不是越快越好，而是要稳定、持续、刚好足够快，并且资源不要浪费。

## 4. trainer CPU 本地预处理为什么不够

论文用 RM1 做实验，展示 trainer host 负责预处理时会导致大量 GPU stall。原因可以拆成几类。

第一，CPU 算力不足。

推荐特征转换包含 hash、bucketize、NGram、list clipping、intersection、dense normalization 等操作。它们不是简单 memcpy，也不都是大矩阵运算，很多是小而杂的 CPU-heavy 操作。

第二，memory bandwidth 压力大。

预处理要反复解码、构造 map、生成中间结果、组 tensor。格式转换和内存拷贝会消耗大量内存带宽。

第三，frontend network 压力大。

如果 trainer 自己从 storage 拉 raw compressed bytes，再执行 extraction 和 transform，前端网络和 host memory 都会被占用。GPU 训练本身还需要 host 参与调度和加载，资源冲突会变得更严重。

第四，模型之间需求差异大。

一个固定 trainer host 配置无法同时适配所有模型。为了最重模型过度配置会浪费资源；按轻模型配置又会让重模型 data stall。

## 5. DPP 的总体架构

DPP 可以分成三类角色：

```text
DPP Master
  -> control plane
  -> split assignment
  -> checkpoint
  -> fault tolerance
  -> autoscaling

DPP Worker
  -> data plane
  -> read raw bytes
  -> extract and transform
  -> create tensor buffer

DPP Client
  -> trainer side
  -> PyTorch hook
  -> fetch tensor batches
```

这个架构的关键不是“多几个 worker”，而是把数据预处理服务化、弹性化、无状态化。

## 6. DPP Master 做什么

训练任务开始时，DPP Master 接收 session specification。可以把它理解成一个训练数据读取计划：

```text
session_spec:
  table
  partitions
  selected_features
  transformations
  batch_size
  trainer topology
```

Master 把整个数据读取和预处理任务切成 split。每个 split 是独立的 work item，worker 拉取 split 后就能处理。

Master 还要负责 autoscaling。关键指标不是单纯 CPU，而是 tensor buffer 是否足够：

```text
buffer depth > 0
  -> trainer usually has data

buffer depth == 0
  -> trainer may stall
```

理想状态是：

```text
keep a non-zero tensor buffer
while maximizing worker resource utilization
```

这比固定 worker 数更有效，因为不同模型、不同训练阶段、不同数据 partition 的预处理成本都可能变化。

## 7. DPP Worker 做什么

DPP Worker 是真正执行数据平面工作的节点。

一个 worker 的路径大致是：

```text
fetch split
  -> read storage chunks
  -> decrypt and decompress
  -> reconstruct streams
  -> decode rows
  -> filter unused features
  -> apply feature transforms
  -> batch into tensors
  -> buffer tensors
  -> serve tensors to clients
```

worker 设计成 stateless，这一点很重要。

stateless 带来三个好处：

1. 失败恢复简单：worker 挂了，Master 重新分配 split。
2. 水平扩展简单：吞吐不够就加 worker。
3. autoscaling 简单：不需要迁移复杂状态。

但是 stateless 不代表没有成本。worker 需要足够的 CPU、内存、网络和内存带宽。第 5 课会详细分析这些瓶颈。

## 8. DPP Client 做什么

DPP Client 跑在 trainer node 上，负责把 PyTorch 的数据读取请求转成 RPC，从 DPP Worker 获取 tensor。

它的目标是让训练框架看到的是一个普通 dataset 或 data loader 接口，但背后实际是一个分布式预处理服务。

这有两个好处。

第一，训练代码不需要理解 worker 拓扑。

ML engineer 只配置训练任务、数据表和 feature transform，不需要手动管理 DPP worker 集群。

第二，trainer host 不做重型 extraction 和 transformation。

trainer host 仍然要承担 tensor loading、RPC、反序列化、安全通信等成本，但负担比自己从 raw storage 做全量预处理小得多。

## 9. DPP 解决了什么，没解决什么

DPP 解决的是：

- trainer CPU 本地预处理不足。
- 模型间 preprocessing demand 差异大。
- worker 需要按任务动态扩缩。
- GPU 因预处理跟不上而等待。

DPP 没有自动解决的是：

- storage 小 I/O 和 seek 问题。
- 文件格式不支持 feature projection。
- DPP worker 自己的 memory bandwidth 瓶颈。
- trainer frontend network 和 data loading 成本。
- 数据中心级别的 power 和容量规划。

所以 DPP 是必要但不充分的。它把瓶颈从 trainer host 转移到 disaggregated preprocessing 和 storage read path，后面还需要存储格式和读取路径协同优化。

## 10. 工程化拆解：DPP autoscaling 应该怎么看

一个 DPP autoscaler 至少要观察下面几类信号。

| 指标 | 含义 | 扩容信号 | 缩容信号 |
| --- | --- | --- | --- |
| tensor buffer depth | worker 侧 tensor 缓冲 | 持续为 0 | 长期过高 |
| trainer wait time | trainer 是否等数据 | 上升 | 接近 0 且资源闲置 |
| worker CPU | transform 是否 CPU-bound | 高且 buffer 低 | 低且 buffer 高 |
| worker memory bandwidth | 是否内存带宽受限 | 高且 CPU 不满 | 低 |
| worker NIC RX | storage 拉数是否受限 | 接近上限 | 低 |
| worker memory capacity | 是否 OOM 风险 | 接近上限 | 低 |
| storage latency | storage 是否成为瓶颈 | 上升 | 稳定 |

关键是要区分“加 worker 有用”和“加 worker 没用”。

如果 worker CPU 满、buffer 低，加 worker 可能有用。

如果 storage 已经被小 I/O 打爆，加 worker 只会增加 storage 压力。

如果 trainer frontend NIC 已经接近饱和，加 worker 也无法让 trainer 更快加载 tensor。

## 11. 本课检查点

读完这一课，你应该能回答：

1. online preprocessing 和 offline ETL 的本质区别是什么？
2. 为什么 trainer CPU 本地预处理会导致 GPU stall？
3. DPP Master、Worker、Client 各自负责什么？
4. 为什么 DPP Worker 要设计成 stateless？
5. 为什么 DPP 不能单独解决所有 DSI 问题？

## 12. 课后练习

请为一个推荐训练任务设计 DPP 配置：

```text
model: ranking_v1
trainer_nodes: 64
features: 1200 dense, 300 sparse
observed_gpu_wait: high
worker_cpu: 90%
worker_nic_rx: 40%
worker_memory_bw: 65%
tensor_buffer_depth: often 0
```

回答：

1. 你会先扩 worker 吗？
2. 你会怀疑 CPU、memory bandwidth、storage，还是 trainer loading？
3. 你还需要采哪些指标才能做决定？
