# Understanding Data Storage and Ingestion for Large-Scale DLRM Training

论文：Understanding Data Storage and Ingestion for Large-Scale Deep Recommendation Model Training

会议：ISCA 2022 Industry Product

本目录是这篇 Data Storage and Ingestion 文章的独立学习区。它按照仓库 `agent.md` 的论文阅读格式组织：PDF、中文深度大纲和可顺序阅读的教程分开保存。

## 文件导航

1. [论文 PDF](./paper.pdf)
2. [中文教学大纲](./course-outline-zh.md)
3. [详细课程](./tutorial/README.md)

## 一句话总结

这篇文章不是在介绍一个单点存储系统，而是在解释 Meta 如何把推荐模型训练中的数据生成、数据仓库、分布式存储、在线预处理、trainer 数据加载和数据中心容量规划视为同一条 Data Storage and Ingestion, DSI 链路来设计。

它的核心观点是：当 GPU/DSA 越来越强时，真正限制大规模推荐训练的可能不是训练算力，而是训练数据如何被存、被筛、被预处理、被持续喂给 GPU。

## 推荐阅读顺序

1. 先读 [中文教学大纲](./course-outline-zh.md)，建立整体问题框架。
2. 再按顺序读 [详细课程](./tutorial/README.md) 中的 7 课。
3. 每课读完完成检查点和练习，把论文内容映射到自己的训练数据链路。

## 学习主线

```text
GPU 集群越来越强
  -> DSI 变成瓶颈
  -> 推荐训练数据和公开 benchmark 很不一样
  -> 需要把存储和在线预处理从 trainer 中拆出来
  -> 用 DPP + DWRF/Tectonic + 多维系统优化来消除 data stalls
  -> 最终用吞吐、功耗、容量和调度效率衡量收益
```

读这篇文章时要一直追问四件事：

1. 挑战是什么：为什么 DSI 会成为训练瓶颈？
2. 解决什么问题：它到底在消除 storage bottleneck、preprocessing bottleneck，还是 GPU data stall？
3. 如何解决：DPP、DWRF/Tectonic、feature flattening、coalesced reads、feature reordering 分别在哪里起作用？
4. 收益如何：吞吐、功耗、GPU 利用率和数据中心容量分别怎么改善？
