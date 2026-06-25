# 第 2 课：Merging、RDA 与几个容易卡住的问题

## 1. 本课定位

第 1 课讲系统怎么跑。这一课补齐读 D.2 和系统设计时最容易卡住的几个概念：

- 范数是什么，为什么直接平均会让更新变小。
- outer optimizer 到底在优化什么。
- learner / syncer / fragment / quorum 分别是什么角色。
- 有没有公开源码，论文是否说明 Google 内部使用。
- 它和普通大模型预训练有什么区别。

## 2. 范数是什么

范数可以先理解成“向量的长度”。模型参数更新也是一个很长的向量：

```text
update = new_parameters - old_parameters
```

二维例子：

```text
update = [3, 4]
norm = sqrt(3^2 + 4^2) = 5
```

在训练里，范数大致表示这次更新把参数推了多远。范数变小，就是 optimizer 实际迈出的步子变短。

## 3. 为什么直接平均会让更新变小

假设两个 learner 的更新方向互相垂直：

```text
learner A: [1, 0]
learner B: [0, 1]
direct average: [0.5, 0.5]
```

A 和 B 的更新长度都是 1，但平均后的长度是：

```text
sqrt(0.5^2 + 0.5^2) = 0.707
```

如果有 `M` 个 learner，并且它们的更新方向接近正交，直接平均后的长度大约会从 `R` 变成：

```text
R / sqrt(M)
```

这就是 D.2 要解决的问题：learner 数越多，直接平均越容易让 outer optimizer 看到一个过小的更新。

## 4. D.2 的 Direct Averaging

在 Decoupled DiLoCo 里，syncer 收到每个 learner 对某个 parameter fragment 的更新。最简单的合并方法是直接平均。

可以把每个 learner 的 outer gradient 理解成：

```text
outer_gradient = syncer_old_fragment - learner_current_fragment
```

不加权时：

```text
merged_update = average(outer_gradient_1, ..., outer_gradient_M)
```

加权时：

```text
merged_update = sum(weight_i * outer_gradient_i) / sum(weight_i)
```

权重通常和 learner 的训练进度或处理 token 数有关。这样跑得更多、贡献更多 token 的 learner 对合并结果影响更大。

## 5. D.2 的 RDA

RDA 的全名是 Radial-Directional Averaging，可以翻成“径向-方向平均”。

它的核心思想是：

```text
不要直接平均完整 update。
先拆成长度和方向，再分别平均。
```

具体流程：

```text
1. 计算每个 update 的长度。
2. 把每个 update 归一化成单位方向。
3. 对长度做平均。
4. 对单位方向做平均，再归一化。
5. 最终更新 = 平均长度 * 平均方向。
```

这样做的好处是：如果所有 learner 的更新长度都差不多是 `R`，RDA 合并后的更新长度也接近 `R`，不会因为 learner 数量增加而自然缩小成 `R / sqrt(M)`。

## 6. 为什么 embedding 反而用 Avg

论文发现，embedding 部分的 outer gradients 不太符合“近似正交”这个观察。因此 RDA 对 embedding 不一定更好。

最终主实验采用的是：

```text
embedding: direct averaging
non-embedding model: RDA
```

这说明 RDA 不是无脑替换 direct averaging，而是针对非 embedding 主体模型里更明显的方向抵消问题。

## 7. 什么是 outer optimizer

普通大模型训练里，optimizer 通常直接看 batch gradient，然后更新模型参数。

DiLoCo 多了一层：

```text
learner 内部：
  用 AdamW 等 inner optimizer 做本地训练

syncer 层：
  看多个 learner 的参数变化
  把这些变化当作 outer gradient
  用 outer optimizer 合并全局参数
```

所以 outer optimizer 优化的不是单个 batch loss，而是多个 learner 本地训练轨迹之间的差异。

它回答的问题是：

```text
这些 learner 各自训练了一段后，
我们应该怎样更新全局模型，
让它吸收多个 learner 的进展？
```

## 8. 系统里有哪些模块和角色

| 角色 / 模块 | 职责 |
| --- | --- |
| learner worker | 跑本地训练循环，执行 forward/backward/inner optimizer |
| syncer worker | 维护全局 parameter fragments 和 outer optimizer 状态 |
| parameter fragment | 模型参数切片，是 learner 与 syncer 通信和合并的单位 |
| quorum logic | 决定收到多少 learner 更新后可以合并 |
| adaptive grace window | 在不明显影响吞吐时多等一点，尽量收更多 learner 更新 |
| token/progress tracker | 记录 learner 进度，用于加权合并 |
| vector clock | 记录异步 worker 之间的状态进度 |
| checkpoint/recovery | learner 或系统失败后恢复 |
| event tape/replay | 记录异步事件，支持复现和调试 |

这套系统的关键不是某一个公式，而是把一个大同步故障域拆成多个较小 learner，再用 syncer 协调。

## 9. 有没有源码实现

截至本文整理时，没有看到论文附带官方开源代码仓库。arXiv 页面和论文正文给出了算法、架构、实验结果和基础设施描述，但没有提供可直接运行的公开实现链接。

论文说明实验系统由 Pathways 编排，用它来分配 accelerator、构建设备 mesh，并管理 worker 间数据流。因此可以判断论文实验是在 Google 内部训练基础设施上实现和验证的。

但公开论文没有声称它已经用于 Gemini 或 Gemma 的正式生产训练主线。更稳妥的说法是：

```text
它是 Google DeepMind / Google Research 在内部基础设施上验证过的系统设计；
公开资料没有确认其生产部署范围，也没有公开源码。
```

## 10. 和普通大模型预训练有什么区别

模型和训练目标本身没有本质区别：

- 仍然是大模型预训练。
- 仍然处理 token 数据。
- 仍然做 forward/backward。
- learner 内部仍然可以用 AdamW 等 optimizer。

区别在分布式训练控制方式。

普通 data-parallel / SPMD：

```text
所有 worker 同步算一个 step
  -> all-reduce
  -> 全员拿到同一版参数
  -> 下一 step
```

Decoupled DiLoCo：

```text
多个 learner 各自本地训练
  -> 后台发送 parameter fragment
  -> syncer 异步合并
  -> 返回更新后的 fragment
  -> learner 继续训练
```

普通训练偏向强同步和强一致；Decoupled DiLoCo 偏向 availability-first，允许短期不完全同步，换取故障容忍、异构硬件利用和临时算力接入。

## 11. 一句话总结 D.2

D.2 想解决的是：learner 多了以后，直接平均 outer gradients 会让更新范数变小，导致 outer optimizer 行为不稳定。RDA 把长度和方向分开平均，让合并后的更新幅度更稳定；主实验里 embedding 用 Avg，其他模型部分用 RDA。

## 12. 读完检查点

1. 你能用二维向量解释为什么平均会让范数变小吗？
2. 你能说清 inner optimizer 和 outer optimizer 分别在优化什么吗？
3. 为什么 RDA 不直接用在所有参数上？
4. 为什么 Decoupled DiLoCo 的核心收益在系统层，而不是单纯模型结构？
5. 为什么公开论文不能等同于“已经确认在生产主线部署”？

## 13. Sources

- Paper PDF: [../paper.pdf](../paper.pdf)
- arXiv HTML: <https://arxiv.org/html/2604.21428v1>
