# 第 7 课：把脚本整理成迷你深度学习框架

## 1. 本课定位

- 对应 D2L：
  - `chapter_deep-learning-computation/model-construction.md`
  - `chapter_deep-learning-computation/parameters.md`
  - `chapter_deep-learning-computation/read-write.md`
  - `chapter_deep-learning-computation/custom-layer.md`
- 建议时长：3 到 5 小时
- 课时目标：
  - 把前几课的零散脚本整理成可复用模块
  - 理解层、参数、优化器、训练器的职责
  - 形成自己的最小实验框架

## 2. 背景与原理介绍

当你已经写过几次模型后，会很快发现很多代码都在重复：

1. 参数初始化
2. 前向传播
3. 反向传播
4. 参数更新
5. 保存和加载模型

框架的价值不是“让你少写代码”那么简单，更重要的是帮你把职责拆清楚，让实验更稳定、可维护、可复用。理解这一层之后，你再去看 PyTorch 的 `Module`、`Parameter`、`state_dict`，就会轻松很多。

## 3. 详细操作过程

### 步骤 1：设计目录结构

建议你新建：

```text
practice/mydl/
├── modules.py
├── layers.py
├── losses.py
├── optim.py
├── utils.py
└── io.py
```

### 步骤 2：先定义 Parameter 和 Module

你可以从最小定义开始：

1. `Parameter`：持有数据和梯度
2. `Module`：提供 `forward()`、`parameters()`、`train()`、`eval()`

一开始不要追求太像 PyTorch，只要结构清晰就行。

### 步骤 3：把常用层封装起来

至少完成：

1. `Linear`
2. `ReLU`
3. `Dropout`
4. `Sequential`

这里的关键不是代码多高级，而是接口尽量统一。

### 步骤 4：把损失和优化器独立出来

你应该把这些职责分离：

1. `losses.py` 负责 `MSELoss`、`CrossEntropyLoss`
2. `optim.py` 负责 `SGD`
3. 训练脚本只负责调用

如果所有逻辑都还写在一个文件里，后面扩展会非常累。

### 步骤 5：写一个训练器脚本

新建 `practice/train.py`，让它完成：

1. 读入数据
2. 创建模型
3. 创建损失函数
4. 创建优化器
5. 训练若干轮
6. 输出指标

你会发现这时模型代码和训练代码已经可以解耦了。

### 步骤 6：支持保存和加载参数

至少实现两个函数：

1. `save_parameters(model, path)`
2. `load_parameters(model, path)`

哪怕只是用 `numpy.savez` 做一个最简单版本，也很值得。

## 4. 本课检查点

1. 为什么要区分“层”和“训练脚本”？
2. 为什么参数应该统一管理，而不是散落在各个局部变量里？
3. `train()` 和 `eval()` 模式为什么重要？
4. 你的小框架最薄弱的地方是什么？

## 5. 课后小作业

1. 为你的 `Sequential` 增加自动收集参数功能。
2. 支持打印模型摘要，包括每层名称和参数形状。
3. 写一个最简版配置文件，让学习率、批大小、轮数可配置。

## 6. 拓展阅读

1. D2L 的模型构造与参数管理章节
2. PyTorch 官方教程中 `nn.Module` 的使用方式
3. 一个轻量级深度学习框架的开源实现，例如 micrograd 或 minitorch
