# 第 6 课：收益评估、PoC 设计与团队落地

## 1. 本课定位

前五课讲了问题、正确性、协议、存储和训练读路径。这一课回答最后的问题：

如何判断这类方案真的值得做？

建议时长：2 到 3 小时。

## 2. 原文短句与翻译

> "model quality gains"

中文翻译：模型质量收益。

论文的重点不是 late materialization 本身直接提升模型，而是它释放了序列长度 scaling 的空间。基础设施让模型可以吃更长 UIH，模型质量收益来自更长历史。

## 3. 评价不能只看存储下降

Partial dump 或 late materialization 很容易只报：

```text
主训练样本体积下降多少
主表写带宽下降多少
```

这些指标重要，但不够。

完整评估至少要覆盖三件事：

1. 正确性：重建样本是否等价于 Fat Row baseline？
2. 系统效率：总资源是否真的下降，而不是换了一个地方消耗？
3. 模型收益：释放出来的资源是否能支持更长序列，并带来质量提升？

## 4. 正确性评估

第一步必须证明 reconstructed sample 是可信的。

建议表格：

| 指标 | 含义 | 目标 |
| --- | --- | --- |
| checksum match rate | 重建 UIH 和 baseline 是否一致 | 越接近 100% 越好 |
| future event rate | 是否存在请求后事件混入 | 必须接近 0 |
| missing trait rate | trait 是否缺失 | 可解释且可控 |
| schema mismatch rate | schema/materializer 是否不兼容 | 必须可监控 |
| sequence length mismatch | 长度是否和模型 spec 一致 | 必须可解释 |

正确性不过关时，不要直接看模型收益。否则你不知道收益来自更长序列，还是来自泄露或 skew。

## 5. 系统效率评估

论文 Table 1 的思路可以迁移成下面这张表：

| 模型租户 | 序列长度 | 并发训练任务 | 主表写带宽 | 主表读带宽 | lookup 带宽 | data loading latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| long ranking | long | low | | | | |
| mid ranking | mid | high | | | | |
| short retrieval | short | high | | | | |

注意不要只做一个平均值。多租户场景下，长序列模型和短序列模型的收益完全不同。

更合理的净收益公式是：

```text
净资源收益 =
  Fat Row 主表读写资源下降
  - immutable lookup 新增资源
  - DPP/materializer 新增资源
  - 运维复杂度成本
```

其中 immutable store 的单位 bandwidth 成本可能不同于原始训练数据存储，所以 raw bandwidth 不能直接相减。

## 6. 模型收益评估

基础设施优化最终要回答：

```text
省下来的资源能不能换来更长序列？
更长序列能不能换来模型收益？
```

建议做 sequence length ladder：

| 序列长度 | 数据成本 | GPU 成本 | 离线指标 | 是否可稳定训练 | 是否值得线上 |
| --- | ---: | ---: | ---: | --- | --- |
| baseline | | | | | |
| 2x | | | | | |
| 4x | | | | | |
| 8x | | | | | |

如果只省了资源，但没有把模型推到更长序列，也是一种收益；但它和论文里的主张不同。论文更强的主张是：数据基础设施本身是模型 scaling lever。

## 7. PoC 阶段 1：正确性 PoC

目标：

```text
证明 reconstructed UIH 与 Fat Row UIH 一致。
```

做法：

1. 选一批已有 Fat Row 样本。
2. 从样本中抽取 user_id、request_ts、sequence spec。
3. 从 normalized UIH store 按版本重建。
4. 对比 Fat Row 中的 UIH。

验收：

- item_id 序列一致。
- event_ts 不超过 request_ts。
- feature group 完整。
- checksum mismatch 有清晰原因。

这一阶段不要追求吞吐。先证明语义。

## 8. PoC 阶段 2：读放大 PoC

目标：

```text
证明 partial dump 不只是少写，也能少读。
```

做法：

1. 准备 short/mid/long 三类模型 spec。
2. 对比 Fat Row read bytes 和 late materialized read bytes。
3. 单独记录 immutable lookup bytes。
4. 测试 sequence length projection 和 feature group projection。

验收：

- 主训练数据 read bandwidth 明显下降。
- 短序列租户有显著 projection 收益。
- lookup bandwidth 没有把资源吃回去。

## 9. PoC 阶段 3：训练吞吐 PoC

目标：

```text
证明 late materialization 不会让 GPU 明显等数据。
```

做法：

1. 将 materializer 接入 dataloader 或 DPP。
2. 开启 prefetch。
3. 分别压测 streaming 和 batch。
4. 记录 GPU starvation 和 p99 latency。

验收：

- data loading latency 可接受。
- GPU starvation 不显著恶化。
- DPP worker 数量在可控范围内。
- batch affinity 对低 locality 场景有效。

## 10. PoC 阶段 4：模型收益 PoC

目标：

```text
在相近资源 envelope 下，把 sequence length 推长，并验证模型收益。
```

做法：

1. baseline 序列长度训练一版。
2. 2x/4x/8x 序列长度各训练一版。
3. 确保 reconstructed sample 不引入 skew。
4. 对比离线指标。
5. 条件允许时做小流量线上验证。

验收：

- 相同序列长度下，late materialization 不劣于 Fat Row。
- 更长序列带来稳定收益。
- 成本增长小于收益预期。

## 11. 团队方案 memo 模板

你可以直接用下面结构写给老板或团队：

```text
标题：
  Partial Dump / Versioned Late Materialization for Long UIH Training

1. 背景
  当前长 UIH 在训练样本中造成重复存储和读写放大。

2. 问题
  Fat Row 保证 O2O，但随着序列长度扩展，数据服务资源成为瓶颈。

3. 核心方案
  样本只保存 recent snapshot 和 immutable UIH version metadata，
  long-term UIH 在训练时按版本重建。

4. 正确性设计
  request_ts、start_ts、end_ts、checksum、schema_version。

5. 系统设计
  immutable store、projection pushdown、DPP、prefetch、data affinity。

6. 评估指标
  正确性、系统资源、训练吞吐、模型收益。

7. PoC 计划
  先正确性，再读放大，再训练吞吐，最后模型收益。

8. 风险
  历史是否 truly immutable、reader 复杂度、lookup 尾延迟、多租户收益是否足够。
```

## 12. 什么时候不值得做

这类方案不一定永远值得。下面情况要谨慎：

1. UIH 不是训练数据体积主因。
2. 样本跨请求重复度不高。
3. 模型租户之间没有明显序列长度差异。
4. 没有稳定的 normalized history store。
5. 训练 reader/DPP 基础设施很弱。
6. 当前目标只是小规模 demo，不需要扩长序列。

如果这些条件成立，先做简单压缩、采样、字段裁剪可能更划算。

## 13. 本课检查点

1. 为什么不能只报主表存储下降？
2. 正确性评估应该先于模型收益评估吗？为什么？
3. 系统效率应该按哪些模型租户拆开看？
4. 为什么 sequence length ladder 能证明基础设施价值？
5. 一个最小 PoC 应该先验证什么？

## 14. 课后练习

请写一页自己的方案 memo，回答：

1. 你们当前最大重复 payload 是什么？
2. 哪些字段可以 partial dump？
3. 最小 metadata schema 是什么？
4. 用什么 baseline 证明正确性？
5. 第一阶段 PoC 的成功标准是什么？
6. 如果 PoC 成功，下一步是省资源还是扩序列拿收益？

## 15. 拓展阅读

1. 论文 Section 5：Evaluation。
2. 论文 Table 1：System Efficiency。
3. 论文 Table 2：Online A/B testing results。
4. 论文 Conclusion：重点看作者如何把 data infrastructure 定义成 recommendation quality scaling lever。

