# Handwritten Deep Learning

这个目录是仓库中的一个独立教程项目，用于配合“手写深度学习”课程内容学习、练习和提交作业。

## Project Structure

- `tutorial/`: 12 节课程讲义
- `practice/`: 每节课的代码脚手架
- `手写深度学习教学大纲.md`: 总体教学大纲

## Git 提交约定

- 课程初始化内容使用：`[init]`
- 你的作业提交使用：`[homework]`

例如：

```bash
git commit -m "[homework] finish lesson03 linear regression"
```

## 推荐作业流程

1. 先阅读 `tutorial/` 中对应课时。
2. 在 `practice/lessonXX/` 中完成 `TODO(core)`。
3. 用 `git diff` 检查你改了什么。
4. 用 `[homework]` 前缀提交。
5. 之后我就可以直接基于你的改动来帮你 review。

## 说明

这个项目现在被收纳在仓库子目录 `handwritten-deep-learning/` 下，方便与其他教程项目并列管理。
