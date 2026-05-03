# Practice Workspace

This directory is the coding companion for the tutorials in
`../tutorial`.

The goal is simple:

1. make it easy to start each lesson
2. keep the core ideas for you to implement
3. give you small sanity checks so you know when you are on track

## How to use this folder

1. Read the matching lesson in `tutorial/`.
2. Open the corresponding directory in `practice/`.
3. Implement the items marked with `TODO(core)`.
4. Run the file directly and use the quick checks.
5. If you get stuck, compare your work with D2L `scratch` sections.

## What is already prepared

- lesson directories for lessons 1 to 12
- a lightweight `mydl/` package for lesson 7 onward
- starter scripts with function signatures and small examples
- a final project template

## What is intentionally not finished

The core learning parts are left for you:

- manual gradients
- model forward passes
- backprop logic
- optimizers beyond the simplest baseline
- sequence and attention internals

This way you can learn with low setup friction, but still do real coding.

## Suggested commands

```bash
cd handwritten-deep-learning/practice
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then run, for example:

```bash
python lesson01/numerical_grad.py
python lesson03/linear_regression_scratch.py
python lesson12/attention_scratch.py
```

## TODO conventions

- `TODO(core)`: important learning task, do this yourself
- `TODO(optional)`: useful extension if you have time
- `CHECKPOINT`: quick self-test after implementing

## Recommended order

1. lesson01 to lesson06: learn the training pipeline
2. lesson07 to lesson09: organize and compare implementations
3. lesson10 to lesson12: sequence models, attention, project

## Directory map

```text
practice/
├── README.md
├── requirements.txt
├── lesson01/
├── lesson02/
├── lesson03/
├── lesson04/
├── lesson05/
├── lesson06/
├── lesson07/
├── lesson08/
├── lesson09/
├── lesson10/
├── lesson11/
├── lesson12/
├── mydl/
└── final_project/
```
