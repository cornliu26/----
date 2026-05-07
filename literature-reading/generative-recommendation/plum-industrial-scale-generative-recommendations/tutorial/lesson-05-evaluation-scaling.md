# 第 5 课：实验、Ablation 与 Scaling 怎么读

## 1. 本课定位

这一课把论文实验读成架构判断：哪些模块必须做，哪些指标能说明 PLUM 有增量价值，scaling 结果又该怎么理解。

建议时长：2 小时。

## 2. 原文短句与翻译

> "sample efficient"

中文翻译：样本效率高。

PLUM 单样本计算更重，但收敛更快、需要样本更少，所以总训练成本不一定更高。

## 3. PLUM vs LEM 怎么读

论文比较 900M activated-param PLUM MoE 和高度优化的生产 LEM。

Table 2：

| Metric | LFV | Shorts |
| --- | ---: | ---: |
| Effective Vocab Size | 2.60x | 13.24x |
| CTR | 1.42x | 1.33x |
| WT/View | 0.72x | 1.13x |
| WF/View | 1.32x | 1.03x |

解读：

- Effective vocab size 大幅提升，说明 PLUM 更能覆盖多样内容和长尾。
- CTR 提升明显。
- LFV 的 WT/View 低于 LEM，说明生成式召回需要结合业务目标看，不是单指标全胜。

## 4. Live experiment 怎么读

PLUM 不是直接替换 LEM，而是加到 candidate pool，对比增加 LEM+ quota。

Table 3：

| Metric | LFV | Shorts |
| --- | ---: | ---: |
| Engaged Users | +0.07% | +0.28% |
| Panel CTR | +0.76% | +4.96% |
| Views | +0.80% | +0.39% |
| Satisfaction | +0.06% | +0.39% |

解读：

```text
PLUM 的价值不是只证明自己能单独召回，
而是证明它能给已有生产系统增加独特候选价值。
```

这是 demo 和上线策略的关键。

## 5. SID-v2 ablation 怎么读

Table 4：

| SID Model | SID Uniqueness | VID Recall@10 |
| --- | ---: | ---: |
| SIDv1 baseline | 94.0% | 12.3% |
| SIDv2 | 96.7% | 14.4% |
| Ablate multi-resolution | 94.8% | 13.2% |
| Ablate multi-embedding | 96.9% | 12.8% |
| Ablate co-occurrence | 91.8% | 12.6% |

工程结论：

- SID 不是前处理小事，而是影响 downstream retrieval 的核心模块。
- Co-occurrence 对 uniqueness 特别重要。
- 多模态融合对 Recall 有明显价值。
- Multi-resolution 对 SID 空间效率和召回都有贡献。

## 6. CPT ablation 怎么读

Table 5：

| Model | Pre-trained LLM | CPT | Recall@10 |
| --- | --- | --- | ---: |
| R1 | No | No | 0.19 |
| R2 | Yes | No | 0.23 |
| CR1 | No | Yes | 0.27 |
| CR2 | Yes | Yes | 0.28 |

工程结论：

- CPT 是强增益。
- LLM init 稳定有帮助。
- 如果资源有限，优先保证 CPT 语料和 SID 对齐，而不是只调 SFT。

## 7. Scaling study 怎么读

论文测试 MoE-110M、370M、900M、3B。

关键观察：

- Loss 和 Recall@10 随 compute 增长有 scaling 趋势。
- 最优模型大小会随 compute budget 向更大模型移动。
- 3B 在当前预算下没有超过 900M，不代表大模型无效，而是训练 examples、batch size、compute budget 没有同步扩。

架构含义：

```text
模型变大不是单独按钮。
必须同步考虑数据量、batch、训练预算、服务成本。
```

## 8. 常见误区

误区 1：只看离线 Recall。

生产价值还要看 live engagement、candidate pool 增量、服务成本和覆盖率。

误区 2：看到 3B 没赢就否定 scaling。

论文明确讨论了 compute budget 和训练样本不足的问题。

误区 3：忽略样本效率。

PLUM 每个 example 更贵，但 examples 少很多，总 FLOPs 反而可以低于 LEM。

## 9. 本课检查点

1. Effective vocab size 说明什么？
2. 为什么 live experiment 比单纯离线指标更能说明价值？
3. SID-v2 哪个 ablation 最值得关注？
4. CPT 和 LLM init 谁的增益更大？
5. Scaling study 对 demo 选模型大小有什么启发？

## 10. 课后练习

设计一张你们自己的评估表：

| 指标类型 | 指标 | 为什么看 |
| --- | --- | --- |
| 离线召回 | Recall@K | |
| 生成质量 | invalid SID rate | |
| 映射质量 | collision rate | |
| 服务性能 | decoding latency | |
| 候选价值 | unique candidates | |
| 线上效果 | CTR / watch / satisfaction | |

## 11. 拓展阅读

1. 论文 Section 3.1：PLUM vs LEM。
2. 论文 Section 3.2：SID ablation。
3. 论文 Section 3.3：CPT ablation。
4. 论文 Section 3.4：Scaling study。

