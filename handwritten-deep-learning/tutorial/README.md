# Handwritten Deep Learning Tutorial

这套教程是根据 [手写深度学习教学大纲](../手写深度学习教学大纲.md) 拆分出的 12 节课时讲义，目标是带你从 `Python + NumPy` 起步，逐步写出自己的深度学习代码，再和 D2L / PyTorch 的实现对照学习。

## 建议使用方式

1. 每周先读一节教程，再打开对应 D2L 章节。
2. 跟着“详细操作过程”动手写代码，不要只读不写。
3. 完成“课后小作业”后，再看拓展阅读。
4. 每完成 3 节课，回头整理一次自己的代码结构和笔记。

## 课程目录

1. [第 1 课：环境准备与 NumPy 基础](./lesson-01-env-and-numpy.md)
2. [第 2 课：数据、张量与自动求导思想](./lesson-02-data-and-autograd.md)
3. [第 3 课：线性回归从零实现](./lesson-03-linear-regression.md)
4. [第 4 课：Softmax 回归与分类入门](./lesson-04-softmax-regression.md)
5. [第 5 课：多层感知机与反向传播](./lesson-05-mlp-and-backprop.md)
6. [第 6 课：训练稳定性、正则化与初始化](./lesson-06-regularization-and-stability.md)
7. [第 7 课：把脚本整理成迷你深度学习框架](./lesson-07-build-mini-framework.md)
8. [第 8 课：卷积神经网络入门](./lesson-08-cnn-basics.md)
9. [第 9 课：优化器实验与训练对比](./lesson-09-optimizers.md)
10. [第 10 课：序列建模与 RNN 手写实现](./lesson-10-rnn.md)
11. [第 11 课：GRU、LSTM 与现代序列模型](./lesson-11-gru-lstm.md)
12. [第 12 课：注意力机制与结课项目](./lesson-12-attention-and-project.md)

## 统一学习节奏

每节课都尽量按这个节奏来：

1. 预习 20 到 40 分钟：读教程里的背景部分。
2. 主学习 60 到 120 分钟：照着步骤写代码和做实验。
3. 复盘 20 分钟：回答“本课检查点”。
4. 作业 30 到 90 分钟：完成课后题和一个小扩展。

## 当前项目结构

这套课现在已经被整理到独立项目目录 `handwritten-deep-learning/` 下，推荐结构如下：

```text
handwritten-deep-learning/
├── README.md
├── tutorial/
├── practice/
└── 手写深度学习教学大纲.md
```

## 如何配合 D2L 仓库使用

- 本地书籍仓库如果仍放在仓库根目录旁边，可从这里按 `../../d2l-zh-book` 理解相对位置
- 你可以优先看 D2L 的 `scratch` 章节
- 如果某一节觉得抽象，就先完成教程里的操作，再去看 D2L 原文

## 学完后的能力目标

完成前 8 课后，你应该已经能：

1. 用 NumPy 写出线性回归、Softmax 回归和 MLP。
2. 自己解释前向传播、损失函数和反向传播。
3. 看懂 PyTorch 中 `nn.Linear`、`Module`、`optimizer.step()` 在帮你做什么。

完成全部 12 课后，你应该已经能：

1. 手写一个小型深度学习实验框架。
2. 理解 CNN、RNN、GRU/LSTM、Attention 的核心机制。
3. 独立完成一个小型课程项目。
