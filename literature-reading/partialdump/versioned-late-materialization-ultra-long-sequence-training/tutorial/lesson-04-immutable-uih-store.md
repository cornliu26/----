# 第 4 课：Immutable UIH Store 与 Projection Pushdown

## 1. 本课定位

第 3 课讲了协议。这一课讲协议背后的存储系统。

把 long-term UIH 从 Fat Row 里拆出去以后，压力并不会凭空消失。训练 reader 会集中访问 immutable UIH store。如果这个 store 只是普通 KV 或普通离线表，很可能会成为新瓶颈。

建议时长：2 到 3 小时。

## 2. 原文短句与翻译

> "read-optimized immutable storage"

中文翻译：面向读取优化的不可变存储。

这个短句说明论文的存储设计不是泛用存储，而是针对训练时大规模 range scan 做了专门优化。

## 3. Immutable UIH 的查询模式

训练时对 long-term UIH 的查询相对固定：

```text
输入：
  user_id
  start_ts
  end_ts
  sequence_length
  feature_groups

输出：
  一段按时间排序的用户历史
```

这个模式比普通业务查询简单得多，也稳定得多。既然查询模式稳定，存储就可以围绕它做深度优化。

## 4. 为什么要 offloaded compaction

论文使用周期性 ETL/compaction 来生成 immutable UIH。

大致流程是：

```text
实时行为流
  -> 落到数据仓库
  -> 每日 ETL 合并新增行为和历史行为
  -> 按 user_id 和时间排序
  -> 生成预排序文件
  -> bulk-load 到 immutable UIH store
```

这样做的价值是：

1. 线上写路径不承担长历史整理成本。
2. immutable store 可以保持只读和有序。
3. 每次 compaction 可以顺手处理删除、回填、schema 更新。
4. 训练读取时可以尽量顺序 I/O。

## 5. Single-level layout 为什么重要

通用 LSM/KV 系统通常为写入优化。数据先写入新层，再通过后台 compaction 合并到旧层。读一个用户长历史时，可能跨多个层级和多个文件。

Immutable UIH store 不需要服务高频随机写。它可以一次性 bulk-load 成只读布局。

理想读取路径是：

```text
定位 user_id 的历史范围
顺序读取需要的 subsequence stripes
只解码模型需要的列
```

这就是论文强调 read-optimized immutable storage 的原因。

## 6. 存储 key 如何设计

论文提到多维组合 key，可以抽象成：

```text
key:
  user_id
  feature_group
  subsequence_timestamp

value:
  encoded event stripe
```

这不是随便拆 key，而是为 projection pushdown 服务。

`subsequence_timestamp` 让一段长历史被切成时间 stripe。模型只需要最近一段时，不必读取完整 lifelong history。

`feature_group` 让不同模型只读自己需要的特征组。

## 7. Sequence length projection

假设一个用户有 16K 个历史事件，按每个 stripe 512 个事件切分。

```text
stripe_01: oldest events
...
stripe_31
stripe_32: newest events
```

如果某个模型只需要最近 1K 行为，它只需要读最近 2 个 stripe。

如果另一个模型需要最近 16K 行为，它才需要读 32 个 stripe。

这就避免了短序列租户被长序列租户拖累。

## 8. Feature group projection

同一个 event 可以有很多 trait：

```text
item_id
event_ts
event_type
watch_time
like
comment
share
creator_id
content_category
```

不同模型需要的 feature group 不一样：

| 模型 | 可能需要的 feature group |
| --- | --- |
| retrieval | item_id, event_ts |
| pre-ranking | item_id, event_type, watch_time |
| ranking | 多行为特征、多内容特征、多 engagement signal |

如果 store 支持 feature group projection，简单模型就不用读取复杂模型的全部特征。

## 9. Trait-aware columnar encoding

UIH event 里的 trait 密度不同：

- `item_id`、`timestamp` 通常每个 event 都有。
- `like`、`comment`、`share` 可能很稀疏。
- `event_type` 可能适合字典压缩。
- `timestamp` 可能适合 delta encoding。

列式编码的好处是：

```text
按列存储
  -> 只读需要的 trait
  -> 只解码需要的 trait
  -> 稀疏列可以用更合适的编码
```

这对 partial dump 很重要。否则你虽然从样本主表里拆掉了 UIH，但训练时仍然把整段 UIH 的所有 trait 都读回来，收益会被吃掉。

## 10. 和 partial dump 的关系

一个完整 partial dump 方案至少有三层：

```text
样本层：
  dump version metadata 和 projection requirement

存储层：
  支持按时间、长度、feature group、trait column 读取

reader 层：
  把模型 feature spec 下推成 storage scan
```

如果只做样本层，不做存储层和 reader 层，系统可能只是从“写得贵”变成“读得贵”。

## 11. Schema evolution 的价值

论文还提到 offloaded compaction 对特征迭代有价值。

如果研究同学新增一个 SideInfo，比如 `creator_category`，传统 append-only 历史可能需要漫长回填。Immutable compaction 每次重建 lookback window，就可以在一次 pipeline run 中生成带新 schema 的历史窗口。

这对模型迭代很重要：

```text
长序列模型不断演进
  -> 需要不断尝试新 UIH feature
  -> 数据平台必须支持较快 schema iteration
```

## 12. 本课检查点

1. 为什么 immutable UIH store 不能只是普通 side table？
2. Offloaded compaction 解决了什么问题？
3. Single-level layout 为什么适合 range scan？
4. Sequence length projection 和 feature group projection 分别省什么？
5. Trait-aware columnar encoding 为什么比整行编码更适合 UIH？

## 13. 课后练习

设计一个简化 immutable UIH store：

```text
key = hash_bucket(user_id) / user_id / feature_group / stripe_start_ts
value = columnar_encoded_events
```

回答：

1. 如果 stripe 大小是 512，模型要最近 2048 个行为，需要读取几个 stripe？
2. 如果模型只要 item_id 和 timestamp，如何避免解码 like/comment/share？
3. 如果新增一个 trait，需要改动样本、store、reader 的哪些部分？
4. 如果用户请求删除历史，下一次 compaction 应该如何处理？

## 14. 拓展阅读

1. 论文 Section 4.1：Scalable UIH Storage。
2. 论文 Section 4.1.2：Read-Optimized Immutable Storage。
3. 列式存储和 projection pushdown 的基础材料。

