# 第 2 课：O2O 一致性与 Future Leakage

## 1. 本课定位

第 1 课讲了 Fat Row 为什么贵。这一课要讲为什么不能简单把 Fat Row 拆掉。

推荐训练数据有一个硬约束：训练时看到的特征，必须等价于线上请求发生时模型能看到的特征。这就是 Online-to-Offline consistency，简称 O2O 一致性。

建议时长：1 到 2 小时。

## 2. 原文短句与翻译

> "future leakage"

中文翻译：未来信息泄露。

它指训练样本输入包含了线上请求发生之后才出现的信息。模型训练时学到了线上推理时不可能看到的信息，离线指标可能虚高，线上效果却无法兑现。

## 3. 推荐训练为什么天然异步

推荐样本通常不是请求发生时立刻完整生成的。一个简化时间线是：

```text
T_request:
  用户打开 App 或刷新页面
  ranking service 发起推荐请求
  模型使用当前特征完成排序

T_label:
  用户产生反馈
  例如点击、观看、完播、点赞、评论

T_train:
  训练系统消费样本
  此时用户历史已经继续增长
```

关键点是 `T_train > T_label > T_request`。

训练发生时，系统里已经有很多新行为。如果训练 reader 直接读取“当前最新用户历史”，就可能把 `T_request` 之后发生的行为放进输入特征。

## 4. Fat Row 为什么能保证 O2O

Fat Row 的传统做法是：

1. 在线请求时，读取当时的特征。
2. 把特征快照写入高吞吐 feature store 或训练样本缓存。
3. label 到达后，把 label 和当时的 feature snapshot join。
4. 生成完整训练样本。

这个方案很重，但语义简单：

```text
训练样本里的 feature state
  = request time 的 feature state
```

这也是为什么工业系统愿意长期使用 Fat Row。它虽然贵，但正确性边界清楚。

## 5. 论文的关键区分

论文最重要的判断之一是：

```text
O2O 一致性是必须的。
完整物理预物化不是必须的。
```

也就是说，Fat Row 是 sufficient，但不是 necessary。

这个区分对 partial dump 非常关键。你不能为了省存储牺牲 O2O；但你可以寻找另一种更便宜的方式来实现 O2O。

## 6. 为什么 naive side table join 不够

一个看似自然的方案是：

```text
主训练样本：
  sample_id, user_id, request_ts, label

UIH side table：
  user_id, event_ts, event_features

训练前按 user_id join
```

这个方案的问题在于，它没有自动保证 request-time 语义。

如果 join 时不带时间边界，就可能读到未来行为。如果 join 只在 batch ETL 里做，streaming training 又难以使用同一套逻辑。

更准确地说，论文反对的不是 normalization，而是“没有版本协议的 normalization”。

## 7. 时间语义比表结构更重要

可以用下面这条规则判断样本是否正确：

```text
对于某条 request 样本，
输入特征中任何用户行为都不应该晚于 request 可见边界。
```

生产中这个边界不一定只是 `event_ts <= request_ts`，因为可能还有：

- event_ts：用户行为发生时间。
- ingestion_ts：行为进入日志系统时间。
- feature_version：特征生产版本。
- compaction_version：离线压缩或回填版本。
- watermark：流式系统可见水位。

所以一个严肃的 partial dump 设计，需要把时间语义写成契约。

## 8. 哪些特征该 snapshot，哪些可以重建

可以按 O2O 风险把特征分层：

| 特征 | 示例 | 变化速度 | Future leakage 风险 | 建议 |
| --- | --- | --- | --- | --- |
| request context | 入口、设备、场景 | 每请求变化 | 高 | 直接 snapshot |
| candidate set | 当次召回候选 | 每请求变化 | 高 | 直接 snapshot |
| label | 点击、播放、互动 | 请求后产生 | 极高 | 作为目标，不进输入 |
| recent UIH | 请求前短窗口行为 | 高频变化 | 高 | snapshot 或短窗口固化 |
| long-term UIH | 较久历史行为 | 稳定 | 中低 | 按版本重建 |
| stable side info | 内容类目、作者属性 | 中低 | 取决于版本 | 可版本化重建 |

这张表体现了论文的核心工程折中：不是所有特征都 late materialize，而是只对适合的部分使用。

## 9. 和 partial dump 的映射

一个不安全的 partial dump 记录可能长这样：

```text
sample_id
user_id
label
```

它太弱，因为 reader 不知道该读哪个时间点的历史。

一个更安全的记录应该至少包含：

```text
sample_id
user_id
request_ts
label_ts
visible_event_end_ts
feature_schema_version
materializer_version
```

如果涉及 long-term UIH，还应该包含：

```text
immutable_start_ts
immutable_end_ts
target_sequence_length
feature_group_list
checksum
```

这些字段的意义不是“为了多存一点元数据”，而是让训练时重建有明确边界。

## 10. 一个具体例子

假设：

```text
10:00:00  用户请求推荐
10:00:02  系统返回视频 A、B、C
10:00:10  用户点击视频 B
10:00:30  用户又看了视频 D
10:05:00  label join 完成
10:10:00  训练消费样本
```

训练这条 10:00 的请求样本时，视频 B 的点击可以作为 label，但不能作为输入 UIH。视频 D 更不能进入输入 UIH。

如果训练输入里包含了视频 B 或 D，模型就可能学会“用户已经点击过 B，所以推荐 B 是好样本”这种线上不可能发生的关系。

## 11. 本课检查点

1. 为什么推荐训练样本天然存在 `T_request`、`T_label`、`T_train`？
2. Future leakage 会如何导致离线指标虚高？
3. 为什么 Fat Row 正确但昂贵？
4. 为什么没有版本协议的 side table join 不可靠？
5. 一个 partial dump record 至少应该包含哪些时间边界字段？

## 12. 课后练习

写一个 future leakage 检测伪代码：

```text
for example in training_examples:
  full_uih = materialize(example)
  for event in full_uih:
    if event.event_ts > example.request_ts:
      report_error(example.sample_id, event)
```

然后继续思考三个生产问题：

1. 如果 event_ts 可能乱序，应该用什么时间字段辅助判断？
2. 如果 feature 是后处理生成的，如何判断 feature_version 是否可见？
3. 如果发现泄露，是丢样本、回退 Fat Row，还是阻断训练？

## 13. 拓展阅读

1. 论文 Section 2.1：Online-to-Offline Consistency。
2. 论文 Section 3.1：The False Necessity of Pre-Materialization。
3. 特征平台相关材料：training-serving skew、point-in-time correctness。

