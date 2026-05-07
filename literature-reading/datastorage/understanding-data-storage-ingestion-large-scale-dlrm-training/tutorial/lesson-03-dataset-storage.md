# 第 3 课：工业推荐数据如何存

## 1. 本课定位

这一课进入论文的数据存储层，回答三个问题：

1. 推荐训练样本在数据仓库里长什么样？
2. 为什么要用 map column 来承载 dense 和 sparse feature？
3. 为什么只把数据存成列式文件还不够，还要支持 feature-level filtering？

本课的核心结论是：

```text
推荐训练数据格式必须同时满足两件事：
  feature engineering 要灵活
  training read path 要能选择性读取
```

这两件事天然有冲突。逻辑上越灵活，物理上越容易难读。论文的很多存储优化，都是在解决这个冲突。

## 2. 原文短句与翻译

原文短句："feature filtering"

中文翻译：按特征过滤读取。

解释：训练任务不会读取样本里的全部 feature，而是根据模型配置只读取需要的 feature。存储层如果不能理解这种 projection，就会读入大量无用数据。

## 3. 推荐训练样本的逻辑结构

一条推荐训练样本可以简化成：

```text
training_row:
  label
  dense_features
  sparse_features
  weighted_sparse_features
```

论文中 dense feature 和 sparse feature 都被放进 map column。

dense feature 类似：

```text
map<feature_id, float>
```

例如：

```text
current_time -> 13.5
user_age_bucket -> 4
```

sparse feature 类似：

```text
map<feature_id, list<int>>
```

例如：

```text
recent_page_ids -> [12, 98, 301]
clicked_ad_ids -> [7, 9]
```

weighted sparse feature 类似：

```text
map<feature_id, map<int, float>>
```

例如：

```text
page_id_to_score -> {
  12: 0.8,
  98: 0.2
}
```

这类 schema 的好处是：新增 feature 时，不一定需要为每个 feature 新增一个物理表字段。

## 4. 为什么 map schema 对推荐很重要

推荐系统的 feature 工程速度非常快。一个成熟业务中，feature 可能处在不同状态：

- beta：还没稳定上线，可能只在实验中注入。
- experimental：进入 combo 或 release candidate job。
- active：被当前生产模型使用。
- deprecated：不再推荐使用，等待清理。

如果每一个 feature 都是固定 schema 的一个 column，新增、回填、废弃都会带来很高元数据和兼容成本。

map schema 的价值在于：

```text
feature_id is data
not necessarily schema
```

工程师可以更快地做 feature engineering。但代价是：reader 如果只看到一个大 map，很难只读其中几个 key。

## 5. 列式文件格式解决了什么

论文使用 DWRF，这是 Meta 内部类似 ORC 的列式文件格式。列式格式的基本优势是：

- 同一列的数据连续存储，压缩效果好。
- reader 可以跳过不需要的列。
- 大批量 scan 时吞吐高。
- stripe 和 stream metadata 可以支持更细粒度读取。

但问题在于：如果逻辑列本身是一个 map，那么“列式”默认只能帮你跳过整个 map column，不能自动跳过 map 里的某些 feature_id。

也就是说，普通列式格式可能只能做到：

```text
read dense_features map
skip sparse_features map
```

但训练真正需要的是：

```text
read feature_id A, D, F
skip feature_id B, C, E
```

这就是 feature flattening 的原因。

## 6. 工业推荐训练的读取方式

论文说单个训练任务会从两个维度过滤数据。

第一，row filter。

训练任务只读某些 partition。例如按日期分区，release candidate 可能读最近若干天或若干周的 partition。

第二，column or feature filter。

训练任务只读模型配置需要的 feature。不同模型、不同实验、不同 release candidate 的 feature projection 不一样。

因此，一次训练读取可以描述成：

```text
read table T
  where partition in selected_partitions
  project selected_features
```

这个查询看起来像数据库 scan，但它在训练系统里有几个特殊要求：

- 数据量是 PB 级。
- online preprocessing 在训练 critical path。
- 读取不是为了返回 SQL 结果，而是为了持续喂 GPU。
- 每个 batch 之后还要执行 feature transforms。

## 7. 论文中的关键数据如何理解

Table 3 展示了使用 partition 的规模。RM1、RM2、RM3 的 representative training job 使用的数据量仍然是 PB 级。这说明：

```text
local storage assumption does not hold
```

不能假设 trainer 节点本地盘就能放下训练数据。

Table 4 和 Table 5 一起看，能看到另一个事实：

```text
dataset stores many features
model reads a small subset
```

论文中的模型只读取大约 9% 到 11% 的 feature，但读取 bytes 比例更高，大约 21% 到 37%。原因是被模型选中的 feature 通常更有信号，coverage 更高，sparse list 更长，所以字节占比更大。

Table 6 展示了小 I/O 问题。feature filtering 让每次读取变得更小，HDD 上 seek 成本会被放大。这为第 6 课里的 coalesced reads 埋下伏笔。

## 8. feature 热度为什么重要

论文还观察到训练任务之间存在数据复用。不同 job 不会完全读取同一组 feature，但会围绕生产 baseline 共享大量热门 feature。

这带来一个优化机会：

```text
if some bytes account for most storage traffic
  -> cache them
  -> place them on faster media
  -> order them closer in files
```

注意这里的 hotness 不是传统 Web cache 的 URL 热度，而是训练任务的 feature projection 热度。

例如：

```text
feature A appears in most release candidate jobs
feature D often appears together with feature A
feature X is experimental and rarely read
```

那么物理布局就应该让 A 和 D 更容易一起读，而不是按 feature_id 随机顺序落盘。

## 9. 工程化拆解：一个训练数据存储格式需要哪些元数据

如果自己设计类似系统，至少需要下面几类元数据。

### 9.1 table and partition metadata

| 元数据 | 用途 |
| --- | --- |
| table name | 定位数据集 |
| partition key | 支持 row filter |
| partition size | 做容量规划 |
| creation time | 判断 freshness |
| retention policy | 控制历史数据成本 |

### 9.2 feature metadata

| 元数据 | 用途 |
| --- | --- |
| feature_id | 训练配置引用 |
| feature type | dense / sparse / weighted sparse |
| coverage | 估算读取 bytes |
| avg sparse length | 估算 transform 成本 |
| lifecycle state | beta / experimental / active / deprecated |
| owner | 支持治理和清理 |

### 9.3 file-level metadata

| 元数据 | 用途 |
| --- | --- |
| stripe offset | 支持跳读 |
| feature stream offset | 支持 feature-level read |
| compressed size | 估算 I/O |
| encoding | 支持解码 |
| min/max row or timestamp | 支持 pruning |
| hotness score | 支持 feature reordering |

没有这些元数据，feature filtering 就只能停留在逻辑层，无法真正降低物理读取成本。

## 10. 本课检查点

读完这一课，你应该能回答：

1. 为什么推荐训练样本适合用 map column 承载 feature？
2. map schema 为什么会造成 over-read 风险？
3. row filter 和 feature filter 分别对应什么？
4. 为什么只做列式存储还不一定支持 feature-level selective read？
5. feature 热度如何影响 cache、SSD tier 和文件布局？

## 11. 课后练习

请设计一个简化版训练数据文件布局，输入如下：

```text
features:
  A dense, hot
  B dense, cold
  C sparse, hot
  D sparse, often read with A
  E sparse, experimental
```

要求回答：

1. 你会如何把这些 feature 组织成 file streams？
2. 如果底层是 HDD，你会如何减少 seek？
3. 如果底层有少量 SSD cache，你会优先放哪些 feature？
4. 如果明天新增 feature F，你的 schema 是否需要重写历史文件？
