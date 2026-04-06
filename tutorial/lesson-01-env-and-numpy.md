# 第 1 课：环境准备与 NumPy 基础

## 1. 本课定位

- 对应 D2L：
  - `chapter_installation/index`
  - `chapter_preliminaries/ndarray.md`
  - `chapter_preliminaries/linear-algebra.md`
  - `chapter_preliminaries/calculus.md`
  - `chapter_preliminaries/probability.md`
- 建议时长：2 到 3 小时
- 课时目标：
  - 配好后续学习环境
  - 熟悉 `NumPy` 数组、广播、矩阵乘法
  - 建立“导数是变化率、梯度是多变量方向导数”的直觉

## 2. 背景与原理介绍

深度学习表面上看是“堆模型”，本质上却是“对数组做大量可微分计算”。如果你还没把数组运算、矩阵乘法、广播规则和简单微积分直觉打牢，后面每一节都会感觉像在背 API。

这一课的核心，不是把所有数学都学完，而是先把后面会不断复用的基础能力建立起来：

1. 把数据看成张量，也就是多维数组。
2. 把模型计算看成一连串矩阵和逐元素运算。
3. 把训练看成“让损失下降”的数值优化过程。

## 3. 详细操作过程

### 步骤 1：准备环境

建议你准备一个最小环境：

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib jupyter
```

如果你不想现在装太多库，第一课只装 `numpy` 和 `matplotlib` 也足够。

### 步骤 2：新建自己的练习目录

建议在当前目录下建立：

```text
practice/lesson01/
```

本课你至少准备两个文件：

1. `numpy_basics.ipynb`
2. `numerical_grad.py`

### 步骤 3：熟悉 NumPy 数组

在 notebook 中依次练习下面几件事：

1. 创建一维、二维数组。
2. 查看 `shape`、`ndim`、`dtype`。
3. 做切片、索引、reshape。
4. 练习广播，例如列向量加行向量。
5. 使用 `@` 完成矩阵乘法。

你应该特别留意：

1. `*` 是逐元素乘法，不是矩阵乘法。
2. `@` 或 `np.matmul` 才是矩阵乘法。
3. 广播是 NumPy 最常见也最容易出错的地方。

### 步骤 4：把线性代数和代码对应起来

请自己实现并验证下面这些计算：

1. 向量点积
2. 矩阵乘向量
3. 矩阵乘矩阵
4. 按行求和、按列求和
5. L2 范数

建议你不用现成函数时，先手写最朴素版本，再用 NumPy 内置函数验证结果。

### 步骤 5：建立导数与梯度直觉

在 `numerical_grad.py` 里先写一个数值微分函数：

```python
def numerical_grad(f, x, eps=1e-5):
    return (f(x + eps) - f(x - eps)) / (2 * eps)
```

然后分别做三组实验：

1. `f(x) = x**2`
2. `f(x) = sin(x)`
3. `f(x) = 3*x**2 + 2*x + 1`

观察数值导数和你手算出的导数是否一致。

### 步骤 6：做一个小结

把今天的内容用自己的话总结成三句话：

1. 张量在代码里是什么。
2. 梯度在训练里意味着什么。
3. 为什么矩阵乘法是深度学习的基本操作。

## 4. 本课检查点

学完后，你应该能回答：

1. `shape=(32, 784)` 的数组表示什么？
2. 为什么 `A * B` 和 `A @ B` 完全不是一回事？
3. 中心差分为什么比单边差分更常用？
4. 广播在方便的同时，最容易带来什么错误？

## 5. 课后小作业

1. 用最朴素的双重循环手写矩阵乘法，再和 `np.matmul` 对比速度。
2. 自己生成一组一元函数曲线，同时画出它的数值导数曲线。
3. 写一个函数，输入二维数组，返回按行均值、按列均值、总体标准差。

## 6. 拓展阅读

1. D2L 中的 `ndarray` 与 `linear-algebra` 章节
2. 3Blue1Brown 的线性代数和微积分可视化视频
3. `NumPy` 官方文档里关于 broadcasting 的章节
