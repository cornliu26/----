# 第 8 课：卷积神经网络入门

## 1. 本课定位

- 对应 D2L：
  - `chapter_convolutional-neural-networks/why-conv.md`
  - `chapter_convolutional-neural-networks/conv-layer.md`
  - `chapter_convolutional-neural-networks/padding-and-strides.md`
  - `chapter_convolutional-neural-networks/pooling.md`
  - `chapter_convolutional-neural-networks/channels.md`
  - `chapter_convolutional-neural-networks/lenet.md`
- 建议时长：3 到 5 小时
- 课时目标：
  - 理解卷积的局部感受野与参数共享
  - 手写二维卷积前向传播
  - 完成一个最小图像分类模型

## 2. 背景与原理介绍

全连接网络处理图像时有两个明显问题：

1. 参数量巨大
2. 很难利用图像的局部空间结构

卷积神经网络通过“局部连接 + 权重共享”解决了这两个问题。卷积核会在图像上滑动，用同一组参数提取不同位置的局部模式，因此非常适合图像任务。

## 3. 详细操作过程

### 步骤 1：先用最小例子理解卷积

拿一个 `5x5` 输入和一个 `3x3` 卷积核，手算一次输出。你需要看清楚：

1. 卷积核每一步覆盖了哪些元素
2. 每个输出值为什么是一个局部加权和
3. 输出尺寸如何随核大小变化

### 步骤 2：手写 `conv2d` 前向传播

在 `practice/lesson08/conv2d_scratch.py` 里，从最朴素的双重循环开始写：

1. 遍历输出位置
2. 截取输入局部窗口
3. 与卷积核逐元素相乘再求和
4. 加上偏置

先不考虑 batch 和多通道，单通道版本写通最重要。

### 步骤 3：加入 padding 和 stride

扩展你的函数，支持：

1. 零填充
2. 步幅

每加一个功能，都用手工例子验证输出尺寸是否正确。

### 步骤 4：实现池化

再写一个最简版 `max_pool2d`：

1. 指定窗口大小
2. 指定步幅
3. 在每个局部窗口取最大值

这一步有助于你理解为什么池化可以压缩空间分辨率。

### 步骤 5：搭一个最小 CNN

建议结构：

`Conv -> ReLU -> Pool -> Flatten -> Linear`

如果你暂时不想手写卷积反向传播，可以先把重点放在前向结构和数据流理解上，再把训练部分作为选做提升。

### 步骤 6：对照 LeNet

打开 D2L 的 LeNet 章节，观察：

1. 卷积层如何逐步提取特征
2. 为什么最后还需要全连接层
3. 现代 CNN 和早期 CNN 的差异大概在哪里

## 4. 本课检查点

1. 卷积为什么比全连接更适合图像？
2. 参数共享具体节省了什么？
3. padding 和 stride 分别控制什么？
4. 池化层的作用是什么？

## 5. 课后小作业

1. 把单通道卷积扩展到多通道输入。
2. 写一个函数，给定输入尺寸、卷积核、步幅、填充，自动计算输出尺寸。
3. 选做：尝试推导卷积层对卷积核参数的梯度。

## 6. 拓展阅读

1. D2L 的卷积神经网络章节
2. CS231n 中关于 CNN 的经典讲义
3. LeCun 早期关于 LeNet 的论文或介绍资料
