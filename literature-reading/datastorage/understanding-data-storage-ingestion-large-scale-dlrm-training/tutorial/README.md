# Data Storage and Ingestion for Large-Scale DLRM Training 课程

本课程基于论文 Understanding Data Storage and Ingestion for Large-Scale Deep Recommendation Model Training，目标是把论文从“系统介绍”拆成一套可以学习、讨论和落地复盘的课程。

课程主线：

```text
训练加速器越来越强
  -> 数据存储和摄入链路开始限制训练容量
  -> 推荐训练数据和公开 benchmark 的数据假设不同
  -> trainer CPU 本地预处理会造成 GPU 等数据
  -> 需要 disaggregated preprocessing service
  -> 需要支持 feature-level selective read 的物理存储格式
  -> 需要从文件布局、读取方式、内存格式、调度和功耗一起优化
```

## 课程目录

1. [第 1 课：为什么推荐训练需要单独研究 DSI](./lesson-01-why-dsi-matters.md)
2. [第 2 课：Meta 推荐训练工作负载长什么样](./lesson-02-training-workloads.md)
3. [第 3 课：工业推荐数据如何存](./lesson-03-dataset-storage.md)
4. [第 4 课：为什么要有 DPP](./lesson-04-why-dpp.md)
5. [第 5 课：预处理到底慢在哪里](./lesson-05-preprocessing-bottlenecks.md)
6. [第 6 课：存储格式和读取路径如何协同优化](./lesson-06-storage-readpath-codesign.md)
7. [第 7 课：如果我们要做这个 datastorage，应该从哪里开始](./lesson-07-build-our-datastorage.md)

## 学完以后应该能回答的问题

1. 为什么 DSI 会和 GPU trainer 竞争数据中心 power budget？
2. 为什么工业推荐训练不能照搬 CV/NLP benchmark 的数据加载假设？
3. 为什么推荐训练数据格式需要同时支持 feature engineering 灵活性和 feature-level selective read？
4. 为什么 DPP 这种在线预处理服务要和 trainer 解耦？
5. feature flattening 为什么既能提升 DPP，又可能伤害 HDD storage throughput？
6. coalesced reads、feature reordering、large stripes、in-memory flatmaps 各自解决什么瓶颈？
7. 如果自己要做类似 datastorage，第一阶段应该先采哪些 profile 和指标？

## 课程读法

每课建议按下面顺序读：

1. 先读“本课定位”，明确这一课解决哪个问题。
2. 再读“核心概念详解”，把论文中的系统组件拆开。
3. 重点看“工程化拆解”，这里会把论文内容转成可落地的系统设计问题。
4. 最后完成“检查点”和“练习”，把 Meta 的经验迁移到自己的业务。

## 和大纲的关系

[教学大纲](../course-outline-zh.md) 是课程地图，本目录是按地图展开后的正式课程。后续如果继续深入，可以把每课再扩成讲义、读书会材料或 PoC 设计文档。
