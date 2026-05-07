# 第 6 课：存储格式和读取路径如何协同优化

## 1. 本课定位

这一课是整篇论文最工程化的一课。它解释为什么 DSI 优化不能只做一个局部改动。

论文中最精彩的链路是：

```text
map schema causes over-read
  -> feature flattening reduces DPP work
  -> small I/O hurts HDD throughput
  -> coalesced reads recover storage throughput
  -> feature reordering reduces over-read
  -> large stripes improve I/O size
  -> in-memory flatmaps reduce format conversion
```

本课要学会看懂瓶颈迁移：一个优化解决 A 层问题，可能把压力转移到 B 层。真正的收益来自端到端协同。

## 2. 原文短句与翻译

原文短句："feature flattening", "coalesced reads", "feature reordering"

中文翻译：特征展开、合并读取、特征重排。

解释：这是论文中三组关键读路径优化。它们分别解决过度读取、小 I/O seek、合并读取中的无用字节问题。

## 3. baseline 问题：map schema 导致 over-read

逻辑上，map schema 很适合推荐 feature 演化：

```text
dense_features:
  feature_id -> value

sparse_features:
  feature_id -> list
```

但如果物理文件层把整个 map 当成一个整体，训练任务只需要 feature A 和 D 时，也可能读入 A、B、C、D、E、F。

这叫 over-read。

over-read 的代价包括：

- storage 读更多 bytes。
- DPP 解压和解码更多 bytes。
- CPU 花在无用 feature 上。
- memory bandwidth 花在无用转换上。
- network 搬运更多 raw data。

所以第一步优化是让存储层能按 feature_id 读。

## 4. feature flattening：让 map key 变成文件层逻辑列

feature flattening 的核心思想是：

```text
logical map:
  feature_id -> values

physical streams:
  feature_A_stream
  feature_B_stream
  feature_C_stream
```

这样训练任务只需要 A 和 D 时，reader 可以只读取 A stream 和 D stream。

收益很明显：

- 减少无用 feature 解码。
- 减少 DPP extraction 成本。
- 减少 CPU 和 memory bandwidth。
- 论文中 DPP throughput 提升到 2.00x。

但它引入一个新问题：小 I/O。

每个 feature stream 可能很小。训练任务读取很多分散 feature 时，会产生大量小读。对于 HDD，这意味着大量 seek。论文中 feature flattening 后 storage throughput 曾大幅下降，这说明 DPP 侧变快了，但 storage 侧被打爆了。

这就是典型瓶颈迁移。

## 5. coalesced reads：在 selective read 和 HDD seek 之间折中

coalesced reads 的思路是：不要每个 feature stream 都单独发一个小读，而是把相近范围内的多个 stream 合成一次较大的 I/O。

例如需要读 A 和 D：

```text
file order:
  A B C D E F

need:
  A D

coalesced read:
  read A B C D together
```

这样会 over-read B 和 C，但减少了 seek 次数。

这个优化本质上是在两个成本之间做平衡：

```text
small random I/O cost
vs
extra bytes over-read cost
```

如果底层是 HDD，seek 成本高，适度 over-read 是值得的。如果底层是 SSD 或 cache，平衡点可能不同。

## 6. feature reordering：让常一起读的 feature 物理上靠近

coalesced reads 的缺点是会读入中间不需要的 feature。这个缺点取决于 feature 的物理顺序。

如果常一起读的 feature 离得很近：

```text
A D C B E F
need A D
read A D
```

over-read 就少。

如果常一起读的 feature 离得很远：

```text
A B C E F D
need A D
read A B C E F D
```

over-read 就多。

论文用近期训练任务的 feature popularity 和访问模式来重排 feature stream。也就是说，文件布局不是按 feature_id 固定排序，而是按训练工作负载优化。

这背后的原则很重要：

```text
storage layout should be workload-aware
```

## 7. large stripes：让每次读更接近存储硬件喜欢的形态

列式文件通常按 stripe 组织。stripe 太小，会导致读请求更碎。增加 stripe size 可以提高平均 I/O size，减少 seek 或 metadata 开销。

但 large stripes 也不是无限大越好。它可能影响：

- writer flush latency。
- reader skip granularity。
- memory footprint。
- failure recovery 粒度。

论文的工程点在于：stripe size 不是纯文件格式参数，而是 storage hardware、feature projection、training throughput 共同决定的参数。

## 8. in-memory flatmaps：减少 DPP 内部格式转换

即使存储层读得很快，DPP 内部也可能因为格式转换浪费大量内存带宽。

一个低效路径可能是：

```text
columnar file stream
  -> row-wise map
  -> transform intermediate map
  -> tensor layout
```

如果每一步都做拷贝、hash lookup 和对象构造，memory bandwidth 会非常重。

in-memory flatmaps 的思路是让 DPP 内部表示更接近输入和输出：

```text
DWRF columnar stream
  -> flat feature buffers
  -> tensor-friendly layout
```

这样可以减少：

- map reconstruction。
- per-row object allocation。
- column-to-row-to-column 来回转换。
- intermediate copy。

论文中这个优化提升了 DPP Worker throughput，说明很多瓶颈不是算法复杂度，而是数据表示不匹配。

## 9. localized optimizations：小优化在规模下也很大

论文还提到去掉不必要 null check、LTO、AutoFDO 等局部优化。

单看这些优化可能不显眼，但在 DPP 这种高 QPS、持续训练路径上，小的 per-sample 开销会被放大：

```text
tiny cost per feature
  x thousands of features
  x billions of samples
  x many training jobs
  -> datacenter-level power
```

这也是工业系统和实验系统的差异：在规模足够大时，底层代码路径的微小浪费会转化成 MW 级容量问题。

## 10. 如何读 Table 12

Table 12 不应该只读最终 2.94x 和 2.41x。更应该读优化之间的因果关系。

第一步，feature flattening：

```text
DPP throughput improves
storage throughput collapses
```

说明减少 over-read 有用，但小 I/O 伤害 HDD。

第二步，coalesced reads：

```text
storage throughput recovers
but introduces controlled over-read
```

说明存储硬件形态决定读粒度。

第三步，feature reordering 和 large stripes：

```text
reduce unnecessary over-read
increase average I/O size
```

说明文件布局要结合 workload。

第四步，in-memory flatmaps 和 localized optimizations：

```text
reduce DPP internal overhead
```

说明读到数据之后，内存表示和 CPU 路径也要优化。

## 11. 工程化拆解：如何设计一个 feature ordering 策略

可以把 feature ordering 看成一个简化的布局优化问题。

输入：

- feature size。
- feature read frequency。
- feature pair co-read frequency。
- storage medium seek cost。
- coalesced read window。
- feature lifecycle state。

目标：

```text
minimize expected over-read bytes
while keeping hot co-read features close
```

一个简单启发式：

1. 计算每个 feature 的 hotness。
2. 计算 feature pair 的 co-read score。
3. 把最热 feature 放在一起。
4. 把经常一起读的 feature 相邻。
5. 把很冷或实验 feature 放到后面。
6. 定期按最近 7 天或 14 天 workload 更新 ordering。

需要注意：

- 更新 ordering 会影响离线数据生成路径。
- 历史文件是否重写要看成本。
- 新旧 ordering 共存时，reader metadata 必须能识别。
- 过度追求最近热度可能造成布局抖动。

## 12. 本课检查点

读完这一课，你应该能回答：

1. feature flattening 解决了什么问题？
2. 为什么 feature flattening 可能让 HDD storage throughput 下降？
3. coalesced reads 为什么要接受一定 over-read？
4. feature reordering 为什么必须依赖训练任务访问热度？
5. in-memory flatmaps 为什么能缓解 memory bandwidth 压力？

## 13. 课后练习

给定下面 feature 访问模式：

| feature | size | read frequency | often read with |
| --- | --- | --- | --- |
| A | large | high | D |
| B | small | low | E |
| C | medium | medium | A |
| D | large | high | A |
| E | small | low | B |
| F | medium | high | C |

假设底层是 HDD，coalesced read window 会把物理相邻的 3 个 feature 一起读。

请设计一个 feature order，并回答：

1. 你为什么这样排？
2. 哪些 feature 可能被 over-read？
3. 如果底层换成 SSD，你的排序会不会变？
4. 如果 B 明天变成热门实验 feature，你如何更新布局？
