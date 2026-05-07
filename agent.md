# Agent Instructions

This repository is a tutorial and literature-reading workspace. Treat it as a long-lived study repository, not a scratchpad.

## Repository Structure

Top-level projects:

```text
.
├── README.md
├── handwritten-deep-learning/
└── literature-reading/
```

Current responsibilities:

- `handwritten-deep-learning/`: a standalone handwritten deep learning course with tutorials and practice code.
- `literature-reading/`: paper-reading notes, course outlines, PDFs, and tutorial-style writeups.

Do not mix these two projects. When working on paper reading, stay under `literature-reading/` unless the root `README.md` needs a project index update.

## Literature Reading Layout

Use English directory and file names. Chinese tutorial content is fine when it is intended for the reader.

Recommended layout:

```text
literature-reading/
├── README.md
└── <category>/
    ├── README.md
    └── <paper-slug>/
        ├── README.md
        ├── paper.pdf
        ├── course-outline-zh.md
        └── tutorial/
            ├── README.md
            ├── lesson-01-xxx.md
            ├── lesson-02-xxx.md
            └── ...
```

Naming rules:

- Category names should be lowercase ASCII, for example `partialdump`.
- Paper folder names should be descriptive English slugs, for example `versioned-late-materialization-ultra-long-sequence-training`.
- Tutorial lessons should be numbered with two digits: `lesson-01-...md`, `lesson-02-...md`.
- Use relative Markdown links so GitHub renders the material correctly.

## How To Add A New Paper

When adding a paper, create or update these files:

1. `literature-reading/README.md`: add the category if it is new.
2. `literature-reading/<category>/README.md`: add the paper entry and explain why the category matters.
3. `literature-reading/<category>/<paper-slug>/README.md`: summarize the paper, list materials, and link lessons.
4. `paper.pdf`: include the source PDF when available and appropriate.
5. `course-outline-zh.md`: keep the deep outline or reading notes.
6. `tutorial/README.md`: explain the course goal, audience, and reading order.
7. `tutorial/lesson-*.md`: expand the outline into ordered lessons.

Keep the outline and the tutorial separate:

- `course-outline-zh.md` is the deep paper-reading or preparation note.
- `tutorial/` is the polished, sequential course that a reader can follow.

## How To Write Literature Tutorials

A good paper tutorial should explain the paper through the reader's engineering problem, not just summarize sections.

Prefer this narrative order:

```text
problem / challenge
  -> why the naive or existing solution breaks
  -> core insight
  -> method / protocol / system design
  -> productionization details
  -> evaluation and benefits
  -> limitations
  -> what to do in our own system
```

For each lesson, prefer this structure:

```text
# 第 N 课：...

## 1. 本课定位
## 2. 原文短句与翻译
## 3. 核心概念详解
## 4. 和实际工程/本仓库主题的关系
## 5. 常见误区
## 6. 本课检查点
## 7. 课后练习
## 8. 拓展阅读
```

The section titles can vary, but the lesson should still include:

- background
- detailed explanation
- engineering mapping
- review questions
- exercises or PoC tasks
- reading suggestions

## Quoting Papers

Use short original quotes only, then explain in Chinese. Do not paste long copyrighted passages from papers.

Recommended pattern:

```markdown
> "future leakage"

中文翻译：未来信息泄露。

解释：...
```

The tutorial should be self-contained enough that the reader does not need to jump back to the PDF often, but it should rely on paraphrase and explanation rather than long verbatim copying.

## Partial Dump Category Guidance

For `literature-reading/partialdump/`, always make the engineering angle explicit.

A partial dump writeup should usually discuss:

- Fat Row or full materialization baseline
- Online-to-Offline consistency
- future leakage
- request-time snapshot fields
- version metadata
- reconstructable historical features
- mutable vs immutable storage
- reader/materializer cost
- projection pushdown
- training throughput and GPU starvation
- evaluation, PoC design, and rollout risks

For the current Versioned Late Materialization paper, the reusable teaching axis is:

```text
Fat Row
  -> O2O correctness
  -> future leakage
  -> versioned late materialization
  -> immutable UIH storage
  -> projection pushdown
  -> DPP / prefetch / data affinity
  -> system efficiency and model-quality evaluation
```

## Generative Recommendation Category Guidance

For `literature-reading/generative-recommendation/`, explain the recommendation-system shift, not only the model architecture.

A generative recommendation writeup should usually discuss:

- the baseline retrieval paradigm, such as large embedding models or ANN retrieval
- what is being generated, such as item IDs, Semantic IDs, dense vectors, or natural language
- item tokenization and SID-to-item mapping
- user behavior sequence construction
- multimodal item representation
- continued pre-training and task-specific SFT
- decoding strategy, such as beam search
- invalid generated IDs, collisions, latency, candidate diversity, and fallback
- offline metrics, online metrics, ablations, and scaling results

For the current PLUM paper, the reusable teaching axis is:

```text
Large Embedding Models
  -> Semantic ID tokenization
  -> SID-v2 with multimodal and behavior alignment
  -> continued pre-training for domain adaptation
  -> generative retrieval SFT
  -> beam search and SID-to-video lookup
  -> production comparison, ablation, and scaling study
  -> architecture checklist and demo plan
```

## Editing Rules

- Read the relevant README files before changing a project.
- Inspect `git status --short` before editing.
- Keep changes scoped to the requested project or paper.
- Do not rewrite existing course material unless the user asks.
- Do not move or rename existing files casually; preserve links.
- Prefer adding small index updates when new folders are introduced.
- Keep Markdown readable on GitHub: clear headings, relative links, fenced code blocks, and simple tables.
- Use ASCII for file and directory names.
- Chinese prose is welcome inside tutorial content.

## Git And Commit Conventions

Before committing:

1. Run `git status --short`.
2. Check the staged diff or at least `git diff --stat`.
3. Confirm new README links are relative and valid.

Commit prefixes:

- `[paper] ...` for adding or substantially expanding paper-reading material.
- `[docs] ...` for repository guidance, index pages, or documentation-only changes.
- `[fix] ...` for corrections to existing material.
- `[homework] ...` is reserved for user homework in learning projects.

Keep commits focused. Do not bundle unrelated paper work with unrelated course edits.

Push only when the user asks to publish to GitHub or when the current task clearly includes publishing.

## Current Paper Reading Entries

The current partial dump paper is here:

```text
literature-reading/partialdump/versioned-late-materialization-ultra-long-sequence-training/
```

Important files:

- `README.md`: paper entry page.
- `paper.pdf`: source paper.
- `course-outline-zh.md`: deep Chinese outline.
- `tutorial/README.md`: course entry.
- `tutorial/lesson-01-fat-row-data-wall.md`
- `tutorial/lesson-02-o2o-future-leakage.md`
- `tutorial/lesson-03-versioned-late-materialization.md`
- `tutorial/lesson-04-immutable-uih-store.md`
- `tutorial/lesson-05-training-time-materialization.md`
- `tutorial/lesson-06-evaluation-and-poc.md`

The current generative recommendation paper is here:

```text
literature-reading/generative-recommendation/plum-industrial-scale-generative-recommendations/
```

Important files:

- `README.md`: paper entry page.
- `paper.pdf`: source paper.
- `course-outline-zh.md`: deep Chinese outline.
- `tutorial/README.md`: course entry.
- `tutorial/lesson-01-paradigm-shift.md`
- `tutorial/lesson-02-semantic-id-tokenization.md`
- `tutorial/lesson-03-continued-pretraining.md`
- `tutorial/lesson-04-generative-retrieval.md`
- `tutorial/lesson-05-evaluation-scaling.md`
- `tutorial/lesson-06-architecture-demo-plan.md`
