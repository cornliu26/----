# 第 3 课：Versioned Late Materialization 协议

## 1. 本课定位

这一课进入论文核心方案。

目标是理解：如何让训练样本不再保存完整 long-term UIH，却仍然能在训练时重建出线上请求时模型看到的完整 UIH。

建议时长：2 到 3 小时。

## 2. 原文短句与翻译

> "append-only, temporally ordered, immutable sequence"

中文翻译：只追加、按时间有序、不可变的序列。

这是论文能做 versioned late materialization 的前提。UIH 不是普通会被随意覆盖的 feature，而是事件日志式的历史序列。

## 3. Late materialization 是什么

Materialization 是把最终需要的宽数据提前组装好。

Late materialization 就是延迟组装：先不把完整结果提前写进样本，等真正读取时再按需组装。

对这篇论文来说：

```text
提前 materialization:
  request time 把完整 UIH 写进每条训练样本

late materialization:
  request time 只写版本元数据
  train time 再根据版本元数据读取 immutable UIH 并拼接
```

## 4. 为什么 UIH 适合 late materialization

UIH 的特殊性在于，它可以被时间边界定义。

某个请求时刻 `t` 可见的用户历史，可以近似表示成：

```text
visible_uih(user, t) =
  events of user where event_ts <= t
```

如果历史事件不被原地修改，那么训练时即使晚一点查询，只要查询条件还是 request-time 边界，就可以得到同一份历史。

这就是论文说 Fat Row 是 sufficient but not necessary 的原因。

## 5. Mutable UIH 与 Immutable UIH

论文没有把所有 UIH 都回查，而是拆成两段：

```text
Mutable UIH:
  最近窗口
  高频更新
  需要秒级新鲜
  request time snapshot 到训练样本

Immutable UIH:
  长期历史
  通过周期性 ETL/compaction 生成
  只读、按时间有序
  train time 按版本重建
```

这个拆分非常务实。最近行为最容易泄露未来信息，也最难保证回查时状态一致，所以继续 snapshot。长期历史占体积最大，但相对稳定，适合 late materialize。

## 6. 样本里到底存什么

一条 partial dump 风格的训练样本可以包含：

```text
sample_id
user_id
request_ts
label
request_context
candidate_features_or_ids

mutable_sequence_snapshot

immutable_start_ts
immutable_end_ts
target_sequence_length
feature_group_list
checksum
schema_version
materializer_version
```

注意，`immutable_start_ts` 和 `immutable_end_ts` 不是可有可无的优化字段，而是 O2O 语义的一部分。

`checksum` 也很关键。迁移阶段可以用它对比 Fat Row baseline 和 reconstructed UIH，建立正确性信任。

## 7. 训练时重建流程

训练 reader 或 DPP worker 执行：

```text
materialize(example):
  metadata = read_version_metadata(example)

  immutable_part = range_scan(
    user_id = example.user_id,
    start_ts = metadata.immutable_start_ts,
    end_ts = metadata.immutable_end_ts,
    max_len = metadata.target_sequence_length,
    feature_groups = metadata.feature_group_list
  )

  mutable_part = example.mutable_sequence_snapshot

  full_uih = concat(immutable_part, mutable_part)

  validate(full_uih, metadata.checksum)

  return full_uih
```

这一段就是 versioned late materialization 的核心。

## 8. 正确性为什么成立

正确性依赖四个条件：

1. Mutable 部分在 request time 已经 snapshot。
2. Immutable 部分来自只读历史，不被原地修改。
3. 训练样本记录的时间边界来自 request time。
4. 训练重建逻辑和线上构造 UIH 的逻辑一致。

如果这些条件成立，训练时即使在 `T_train` 查询，也不会读到 `T_request` 之后的行为。

可以把它理解成：

```text
Fat Row:
  用物理复制保存 request-time state

Versioned late materialization:
  用版本边界定义 request-time state
```

## 9. 和 MVCC 的关系

数据库 MVCC 让读事务看到一个一致快照。

这篇论文借用的是同一个思想：

```text
数据库：
  transaction timestamp -> visible rows

推荐训练：
  request timestamp/version boundary -> visible UIH events
```

不同点是，推荐训练的挑战还包括：

- streaming training 和 batch training 共用。
- future leakage 检测。
- 超长 sequence 的 range scan。
- 多租户 projection。
- 训练吞吐和 GPU starvation。

所以它不是简单把数据库 MVCC 拿来用，而是把 versioned read 的思想嵌入推荐训练数据链路。

## 10. PartialDumpContract 设计

如果要把这篇论文映射到你们内部 partial dump，可以把样本元数据抽象成一个 contract：

```text
PartialDumpContract:
  identity:
    sample_id
    user_id
    request_id

  time_boundary:
    request_ts
    visible_event_end_ts
    immutable_start_ts
    immutable_end_ts

  snapshot_payload:
    mutable_recent_sequence
    request_context
    label_join_keys

  projection:
    sequence_length
    feature_groups
    trait_columns

  validation:
    checksum
    schema_version
    materializer_version
```

这个 contract 的价值在于，样本不再是“一堆字段”，而是一份可重建承诺。

## 11. 迁移期如何验证

最稳的方式是双轨验证：

```text
Fat Row path:
  读原始完整 UIH

Late materialization path:
  根据 metadata 重建 UIH

对比：
  item_id 序列
  event_ts 序列
  feature group
  trait value
  checksum
```

迁移初期不要只看模型指标。先证明样本重建完全一致或差异可解释，再谈训练资源收益。

## 12. 本课检查点

1. Late materialization 和 pre-materialization 的区别是什么？
2. 为什么 UIH 的 append-only 属性很重要？
3. 为什么 recent mutable UIH 仍然需要 snapshot？
4. `start_ts`、`end_ts`、`sequence_length` 分别解决什么问题？
5. checksum 在迁移期有什么价值？

## 13. 课后练习

请写一个你们业务里的 `PartialDumpContract` 草稿。至少包含：

1. 样本身份字段。
2. 时间边界字段。
3. 必须 snapshot 的字段。
4. 可以重建的字段。
5. 校验字段。
6. schema/version 字段。

然后回答：如果某个字段没有版本边界，它是否真的适合 partial dump？

## 14. 拓展阅读

1. 论文 Section 3.1：为什么 UIH 不必须物理预物化。
2. 论文 Section 3.3：Versioned Late Materialization Protocol。
3. 数据库 MVCC 的基础概念，只需要理解“按版本读取一致快照”的直觉即可。

