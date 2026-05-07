# 第 5 课：训练时 Materialization 与吞吐优化

## 1. 本课定位

前几课已经把写路径和存储讲清楚了。这一课讲训练读路径。

Late materialization 的本质是把一部分成本从 request-time 写路径挪到 train-time 读路径。如果训练 reader 处理不好，GPU 会等数据，整体训练吞吐会下降。

建议时长：2 到 3 小时。

## 2. 原文短句与翻译

> "pipelined I/O prefetching"

中文翻译：流水线式 I/O 预取。

它的目标不是消灭 immutable UIH lookup，而是让 lookup 延迟和其他读取工作重叠，从而尽量不阻塞训练。

## 3. 训练时多了哪些工作

Fat Row 下，训练 reader 只需要顺序读样本：

```text
read fat row
parse features
send batch to trainer
```

Late materialization 下，reader 变成：

```text
read primary training example
parse version metadata
query immutable UIH store
decode selected feature groups
concat immutable UIH and mutable snapshot
validate checksum
send batch to trainer
```

这显然更复杂。

## 4. GPU starvation 是核心风险

训练 step 可以简单理解成：

```text
step_time = max(gpu_compute_time, data_ready_time)
```

如果 `data_ready_time` 小于 `gpu_compute_time`，训练仍然是 GPU-bound。

如果 `data_ready_time` 大于 `gpu_compute_time`，GPU 就会等数据，训练变成 data-bound。

论文所有训练读路径优化，本质上都是为了让 late materialization 后仍然接近 GPU-bound。

## 5. Disaggregated Data Preprocessing

论文使用 DPP，也就是解耦的数据预处理。

可以把责任边界理解成：

```text
DPP worker:
  读取 primary training data
  查询 immutable UIH
  做 projection
  解码 columnar traits
  拼接完整 UIH
  输出可训练 batch

Trainer:
  接收 batch
  rebatch / shuffle
  forward
  backward
  optimizer step
```

这种拆分的好处是，数据预处理可以独立扩容。某个模型需要更复杂的 UIH materialization，就给它更多 DPP worker，而不是让 trainer 主循环自己扛。

## 6. Base batch 与 rebatching

超长序列会让单个 batch 占用大量内存。DPP worker 如果直接处理最终训练大 batch，可能内存压力很高，并发度也上不去。

论文的思路是：

```text
DPP worker:
  处理较小 base batch

Trainer-side DPP client:
  异步 buffer
  merge
  reshuffle
  形成训练需要的大 batch
```

这样可以在 DPP worker 侧保持更高线程并发，同时满足 GPU 训练对大 batch 的需求。

## 7. Pipelined I/O prefetching

训练 reader 需要读两类数据：

1. primary training table。
2. immutable UIH store。

如果串行执行：

```text
read primary batch N
lookup UIH for batch N
decode batch N
read primary batch N+1
lookup UIH for batch N+1
```

延迟会叠加。

Pipelined prefetch 的思路是：

```text
read primary batch N
issue UIH lookup for batch N
while waiting:
  prefetch primary batch N+1
when lookup N returns:
  decode and output batch N
```

这样把 immutable lookup latency 和下一批主数据读取重叠。

## 8. Partial projection 在 reader 里怎么发生

Reader 不应该盲目读取完整 UIH，而应该读取模型需要的 projection：

```text
model_feature_spec:
  sequence_length = 1024
  feature_groups = [item, action]
  trait_columns = [item_id, event_ts, event_type]
```

Reader 把这个 spec 下推到 immutable store：

```text
scan_request:
  user_id
  start_ts
  end_ts
  num_stripes
  selected_feature_groups
  selected_trait_columns
```

这一步决定了 partial dump 是不是真的能少读。

## 9. Data-affinity optimization

Batch training 的访问模式可能很散。不同样本来自不同用户、不同历史时间段，cache locality 差。

论文用了两个亲和性思路：

1. 按 user 对训练样本聚类，让同一用户相邻时间窗口的多个样本复用一次 UIH lookup。
2. 主训练数据和 immutable UIH store 使用相同 hash partition key，让同一 batch 的 lookup 尽量打到相同 shard，减少网络 fanout。

这背后的原则是：

```text
让训练 batch 的数据组织方式，
尽量贴近 immutable UIH store 的物理组织方式。
```

## 10. 如何压测 partial dump reader

上线前需要至少覆盖下面场景：

| 场景 | 目的 |
| --- | --- |
| short sequence, high concurrency | 验证短序列租户 projection 收益 |
| mid sequence, high concurrency | 验证常规模型吞吐 |
| long sequence, low concurrency | 验证旗舰长序列模型尾延迟 |
| batch backfill, low locality | 验证历史回放压力 |
| streaming, high temporal locality | 验证实时训练路径 |

每个场景至少记录：

- primary read bytes。
- immutable lookup bytes。
- lookup p50/p90/p99。
- materialization p50/p90/p99。
- DPP CPU utilization。
- GPU starvation。
- storage QPS。
- network fanout。

## 11. 常见误区

误区 1：只看平均 data loading latency。

训练系统更怕长尾。同步训练或大规模分布式训练里，一个 worker 慢可能拖住整个 step。

误区 2：以为加 DPP worker 就一定能解决。

如果瓶颈在 immutable store、网络 fanout 或热点 shard，加 worker 只会制造更多请求。

误区 3：忽略 batch 和 streaming 的差异。

Streaming training 通常时间局部性更好；batch training 可能访问历史更散，需要额外 data-affinity。

## 12. 本课检查点

1. Late materialization 为什么会增加训练 reader 复杂度？
2. GPU starvation 指标说明什么？
3. DPP worker 和 trainer 的责任边界是什么？
4. Pipelined prefetch 如何隐藏 lookup latency？
5. Data-affinity optimization 解决什么问题？

## 13. 课后练习

画出你们自己的 partial dump reader 流水线：

```text
primary read
metadata parse
immutable lookup
decode
concat
validate
batch output
```

然后标出每一步的可观测指标：

1. 耗时。
2. 字节数。
3. QPS。
4. p99。
5. 错误率。
6. 是否会造成 GPU starvation。

## 14. 拓展阅读

1. 论文 Section 4.2：Training-Time Materialization。
2. 论文 Section 4.2.2：Masking Latency。
3. 论文 Section 4.2.3：Data-Affinity I/O Optimization。

