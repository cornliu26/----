# 第 5 课：预处理到底慢在哪里

## 1. 本课定位

上一课讲了为什么需要 DPP。这一课深入 DPP 和 trainer 的资源瓶颈，回答一个更细的问题：

在线预处理到底慢在哪里？

论文把路径拆成三段：

```text
storage extraction
  -> feature transformation
  -> tensor loading
```

这三段分别消耗不同资源。理解它们，才能避免一句“数据读取慢”把所有问题混在一起。

## 2. 原文短句与翻译

原文短句："memory bandwidth"

中文翻译：内存带宽。

解释：在线预处理大量时间不是花在复杂计算公式上，而是花在解码、转换、拷贝、构造变长 sparse feature 和生成 tensor 的内存访问上。随着 CPU core 和 NIC 变强，内存带宽可能更早成为瓶颈。

## 3. trainer 侧的 tensor loading 成本

DPP 把重型 extraction 和 transformation 拆走之后，trainer 仍然不是零成本。trainer 还要做：

- 从 DPP Worker 拉 tensor。
- 网络协议处理。
- TLS、Thrift 等生产环境必要操作。
- 内存分配和拷贝。
- 把 tensor 交给 PyTorch runtime。
- 再送到 GPU device memory。

论文里 Figure 8 的意义是：即使 tensor 已经预处理好了，trainer host 的 CPU、内存带宽和 frontend NIC 仍然会被数据加载消耗。

所以 trainer node 设计不能只看 GPU。必须给 host 侧留足资源：

```text
GPU compute is not enough
trainer host must feed GPU
```

## 4. DPP Worker 侧的 extraction 成本

extraction 指从存储读取 raw bytes 并解码为样本。

典型步骤包括：

```text
read compressed storage chunks
  -> decrypt
  -> decompress
  -> reconstruct streams
  -> decode columnar data
  -> apply feature projection
```

这一步会消耗：

- storage read throughput。
- worker ingress network。
- CPU cycles。
- memory bandwidth。

论文指出，storage 端读取的 raw bytes 可能比输出给 trainer 的 tensor bytes 更多。原因是：

- raw bytes 可能包含 over-read。
- 压缩数据要解压。
- 某些 feature 经过 transform 后变小。
- 某些不需要的 feature 在较粗粒度读取中被一起读入。

这解释了为什么把 extraction 放到 trainer 上会放大 trainer frontend network 压力。

## 5. DPP Worker 侧的 transformation 成本

transformation 是推荐训练很特别的一段。

CV 预处理常见操作是 crop、resize、color jitter 等。推荐预处理则大量围绕 dense 和 sparse feature：

- Bucketize：把连续值按边界分桶。
- SigridHash：把 sparse ID hash 到固定空间。
- FirstX：截断 sparse list。
- NGram：组合多个 sparse feature。
- MapId：把原始 ID 映射到新 ID。
- BoxCox / Logit：做 dense normalization。
- sampling：采样训练样本。

论文把 transformation 粗略分为三类：

| 类型 | 例子 | 成本特点 |
| --- | --- | --- |
| dense normalization | Logit, BoxCox, Onehot | 通常不是最大头 |
| sparse normalization | SigridHash, FirstX | 受 sparse list 长度影响 |
| feature generation | Bucketize, NGram, MapId | 往往最贵 |

一个重要结论是：feature generation 通常占 transformation cycles 的大头，论文给出的量级约为 75%。

## 6. 为什么 GPU 加速预处理不简单

看到 CPU 预处理很重，很自然会想：能不能把预处理放到 GPU、FPGA 或 SmartNIC？

论文的态度是：有机会，但不简单。

原因有四个。

第一，操作很碎。

推荐 feature transform 往往是大量小操作，不像训练中的大矩阵乘法那样天然适合 GPU。

第二，操作图很复杂。

一个 feature 可能来自多个中间 transform：

```text
raw feature A
raw feature B
  -> Bucketize
  -> FirstX
  -> NGram
  -> SigridHash
  -> generated feature X
```

第三，变长 sparse feature 难处理。

sparse list 长度不固定，中间结果也可能是变长的。GPU 更擅长规则张量，复杂变长内存管理会降低收益。

第四，和训练 GPU 竞争资源。

如果把 preprocessing 放到 trainer GPU 上，可能提高预处理速度，但会抢训练算力、HBM、PCIe 和 kernel scheduling。

因此，accelerator placement 是一个系统问题，不是“把 CPU 代码搬到 GPU”。

## 7. 为什么 memory bandwidth 会越来越重要

论文比较了不同 compute node 的趋势：CPU core 数和 NIC 带宽增长很快，但每 core 可用 memory bandwidth 不一定同步增长。

这会带来一个现象：

```text
more cores
  -> more parallel preprocessing threads
  -> more memory traffic
  -> memory bandwidth saturates first
```

推荐 preprocessing 里有大量 format conversion：

```text
columnar storage format
  -> row-like feature map
  -> transformed intermediate values
  -> tensor layout
```

如果每一步都产生拷贝和随机访问，内存带宽会被迅速耗尽。第 6 课的 in-memory flatmaps 就是在减少这种格式转换。

## 8. 模型之间为什么需要 right-size

论文中 RM1、RM2、RM3 的 bottleneck 不一样。

有的模型 transformation 更重，有的模型 network 更重，有的模型 memory capacity 更紧张。每个 trainer node 需要多少 DPP Worker 也差异很大。

这说明 preprocessing 资源不能按统一模板分配：

```text
same worker count for every model
  -> light models waste resources
  -> heavy models still stall
```

更合理的做法是：

```text
measure per-model demand
  -> estimate worker throughput
  -> allocate enough workers
  -> monitor tensor buffer
  -> autoscale during training
```

right-size 的目标不是让 worker 很闲，而是在 GPU 不 stall 的前提下尽量少用 DPP power。

## 9. 工程化拆解：如何定位 preprocessing 瓶颈

定位时不要只问“数据慢不慢”，要按路径拆。

### 9.1 判断 trainer loading 是否瓶颈

看：

- trainer frontend NIC utilization。
- trainer host CPU utilization。
- trainer memory bandwidth。
- PyTorch data wait time。
- GPU utilization。

如果 DPP worker buffer 很足，但 trainer 仍然 data wait，问题可能在 trainer loading。

### 9.2 判断 worker extraction 是否瓶颈

看：

- worker NIC RX。
- storage latency。
- storage read size。
- compressed bytes read。
- decompression CPU。
- feature projection hit ratio。

如果 worker NIC RX 接近上限，或 storage latency 上升，可能是 extraction 或 storage read path 瓶颈。

### 9.3 判断 worker transformation 是否瓶颈

看：

- per-transform CPU time。
- LLC miss。
- memory bandwidth。
- sparse feature length。
- generated feature 数量。
- intermediate allocation bytes。

如果 CPU 和 memory bandwidth 高，而 NIC 不高，往往是 transformation 瓶颈。

### 9.4 判断 memory capacity 是否瓶颈

看：

- worker RSS。
- OOM 或 thread pool 限制。
- tensor buffer memory。
- intermediate values memory。
- variable-length sparse feature 分布。

如果为了避免 OOM 必须降低 worker 并发，memory capacity 会限制吞吐。

## 10. 本课检查点

读完这一课，你应该能回答：

1. extraction、transformation、loading 分别指什么？
2. 为什么 DPP 拆走 preprocessing 后，trainer host 仍然有数据加载成本？
3. 为什么 feature generation 比 dense normalization 更可能成为瓶颈？
4. 为什么 GPU 加速推荐 preprocessing 不一定简单？
5. 为什么 memory bandwidth 可能成为下一代 DPP worker 的主瓶颈？

## 11. 课后练习

给定下面观测，判断瓶颈最可能在哪：

```text
GPU utilization: 58%
trainer data wait: high
DPP tensor buffer: often empty
worker CPU: 45%
worker memory bandwidth: 40%
worker NIC RX: 95%
storage read latency: high
average I/O size: 18 KB
```

请回答：

1. 这是 transformation 瓶颈吗？
2. 加 DPP worker 一定有用吗？
3. 你会优先检查第 6 课里的哪个存储优化？
