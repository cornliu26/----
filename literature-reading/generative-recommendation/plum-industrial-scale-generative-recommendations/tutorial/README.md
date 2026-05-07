# PLUM 教程：给推荐架构开发看的生成式推荐课程

这套教程根据论文 `PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations` 和已有中文学习笔记重新整理。

对应材料：

- 原始论文：[paper.pdf](../paper.pdf)
- 精读大纲：[course-outline-zh.md](../course-outline-zh.md)

## 课程目标

学完后你应该能回答：

1. PLUM 为什么不是简单地“把 LLM 接到推荐系统”？
2. Semantic ID 如何把 item 变成可生成 token？
3. CPT 为什么是推荐域适配的关键阶段？
4. Generative retrieval 如何训练、推理和服务？
5. 论文中的实验结果如何转化成架构判断和 demo 计划？

## 课程目录

1. [第 1 课：从 Embedding Retrieval 到 Generative Retrieval](./lesson-01-paradigm-shift.md)
2. [第 2 课：Semantic ID 与 Item Tokenization](./lesson-02-semantic-id-tokenization.md)
3. [第 3 课：Continued Pre-training 与推荐域适配](./lesson-03-continued-pretraining.md)
4. [第 4 课：Generative Retrieval 的训练、解码与服务](./lesson-04-generative-retrieval.md)
5. [第 5 课：实验、Ablation 与 Scaling 怎么读](./lesson-05-evaluation-scaling.md)
6. [第 6 课：架构 Checklist 与 Demo 计划](./lesson-06-architecture-demo-plan.md)

## 推荐学习方式

快速版：

1. 第一天读第 1、2 课，建立范式和 SID 心智图。
2. 第二天读第 3、4 课，理解训练和服务链路。
3. 第三天读第 5、6 课，输出你们自己的 demo memo。

扎实版：

1. 每天一课。
2. 每课完成检查点和练习。
3. 最后一课输出架构 gap analysis 和 PoC 设计。

