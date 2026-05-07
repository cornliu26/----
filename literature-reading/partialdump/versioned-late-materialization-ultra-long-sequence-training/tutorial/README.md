# Versioned Late Materialization：超长序列训练数据基础设施教程

这套教程根据论文 `Versioned Late Materialization for Ultra-Long Sequence Training in Recommendation Systems at Scale` 扩写而来。

对应材料：

- 原始 PDF：[paper.pdf](../paper.pdf)
- 精读大纲：[course-outline-zh.md](../course-outline-zh.md)

## 课程目标

学完后你应该能回答四个问题：

1. 超长 UIH 为什么会让推荐训练数据平台先撞墙？
2. 为什么 O2O 一致性必须保留，但 Fat Row 不是唯一解？
3. Versioned late materialization 如何在训练时重建 request-time UIH？
4. 如何评价这类 partial dump / late materialization 方案是否真的值得做？

## 这套课的定位

这不是一套长序列模型结构课。它不重点讲 HSTU 的 attention 细节，也不重点讲推荐模型如何设计 loss。

它是一套训练数据基础设施课，核心是：

```text
当模型已经证明更长用户历史有收益时，
数据平台如何让模型吃到更长历史，
同时不被存储、I/O、训练 reader 和多租户成本拖垮。
```

你可以把它当作 partial dump 方向的一个工业案例精读。

## 课程目录

1. [第 1 课：Fat Row 与长序列数据墙](./lesson-01-fat-row-data-wall.md)
2. [第 2 课：O2O 一致性与 Future Leakage](./lesson-02-o2o-future-leakage.md)
3. [第 3 课：Versioned Late Materialization 协议](./lesson-03-versioned-late-materialization.md)
4. [第 4 课：Immutable UIH Store 与 Projection Pushdown](./lesson-04-immutable-uih-store.md)
5. [第 5 课：训练时 Materialization 与吞吐优化](./lesson-05-training-time-materialization.md)
6. [第 6 课：收益评估、PoC 设计与团队落地](./lesson-06-evaluation-and-poc.md)

## 推荐学习节奏

快速版：

1. 第一天读第 1、2 课，搞清问题和正确性约束。
2. 第二天读第 3、4 课，搞清系统协议和存储设计。
3. 第三天读第 5、6 课，整理成你们团队自己的 partial dump memo。

扎实版：

1. 每天一课。
2. 每课完成检查点和练习。
3. 第 6 天输出一份方案判断 memo。
4. 第 7 天和你们现有样本链路做字段级对照。

## 读前准备

建议你先准备下面三类背景：

1. 推荐训练样本链路：request、label、join、training example。
2. 长序列推荐：UIH、sequence length、HSTU/ULTRA-HSTU 的基本动机。
3. 数据平台概念：snapshot、version、range scan、projection pushdown、preprocessing worker。

不需要你先懂数据库 MVCC 或列式存储的所有细节，教程会在对应章节解释它们为什么在这里有用。
