# 第 10 课：序列建模与 RNN 手写实现

## 1. 本课定位

- 对应 D2L：
  - `chapter_recurrent-neural-networks/text-preprocessing.md`
  - `chapter_recurrent-neural-networks/language-models-and-dataset.md`
  - `chapter_recurrent-neural-networks/sequence.md`
  - `chapter_recurrent-neural-networks/rnn.md`
  - `chapter_recurrent-neural-networks/rnn-scratch.md`
  - `chapter_recurrent-neural-networks/bptt.md`
- 建议时长：3 到 5 小时
- 课时目标：
  - 理解序列数据和隐藏状态
  - 用 `NumPy` 手写一个最简 RNN
  - 学会做字符级文本生成

## 2. 背景与原理介绍

前面的模型都默认样本之间彼此独立，但文本、时间序列、语音并不是这样。它们的当前时刻往往和过去的内容有关。

RNN 的核心思想，就是让模型维护一个“隐藏状态”，把过去的信息压缩进这个状态里，再和当前输入一起参与下一步计算。这样，模型就有了记忆。

## 3. 详细操作过程

### 步骤 1：做文本预处理

在 `practice/lesson10/text_preprocess.py` 中完成：

1. 读取文本
2. 清洗文本
3. 建立字符表或词表
4. 把文本映射成整数序列

第一次建议做字符级模型，比词级更容易上手。

### 步骤 2：构造时序批量数据

你需要把长序列切成很多短片段，例如长度为 `num_steps` 的子序列。每个训练样本通常是：

1. 输入序列 `x[t:t+num_steps]`
2. 目标序列 `x[t+1:t+num_steps+1]`

这一步会让你理解语言模型到底在预测什么。

### 步骤 3：写出 RNN 前向公式

最简 RNN 的核心计算：

1. `H_t = tanh(X_t W_xh + H_{t-1} W_hh + b_h)`
2. `Y_t = H_t W_hq + b_q`

请你在代码里清楚地区分：

1. 输入到隐藏层的权重
2. 隐藏层到隐藏层的权重
3. 隐藏层到输出层的权重

### 步骤 4：实现前向传播和采样

先不要急着训练。先随机初始化参数，然后写一个“从种子字符开始，往后采样若干字符”的函数，看看模型在随机参数下会输出什么。

### 步骤 5：理解 BPTT

反向传播穿过时间，本质上就是把同一个循环单元在时间维度展开后，再按链式法则回传梯度。这里你至少要搞清楚：

1. 为什么时间步越长，梯度越可能爆炸或消失
2. 为什么实践中常做梯度裁剪

### 步骤 6：训练最简字符级模型

训练时重点观察：

1. 困惑度或平均损失是否下降
2. 采样出来的文本是否越来越像真实文本
3. 隐藏状态是否需要在 batch 之间重置

## 4. 本课检查点

1. RNN 和普通 MLP 的最大差异是什么？
2. 隐藏状态在模型里扮演什么角色？
3. 为什么序列长度一长，训练就容易出问题？
4. 字符级建模和词级建模各有什么优缺点？

## 5. 课后小作业

1. 实现梯度裁剪。
2. 比较不同 `num_steps` 下的训练难度。
3. 用不同长度的种子文本做采样，观察生成风格变化。

## 6. 拓展阅读

1. D2L 的 RNN `scratch` 和 BPTT 章节
2. The Unreasonable Effectiveness of Recurrent Neural Networks
3. 有关梯度消失与爆炸的经典解释资料
