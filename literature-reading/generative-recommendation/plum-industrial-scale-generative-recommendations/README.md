# PLUM: Industrial-scale Generative Recommendations

Paper: `PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations`

Source: [paper.pdf](./paper.pdf)

## What This Paper Is About

PLUM is a framework for adapting pre-trained LLMs to industrial-scale recommendation, especially generative retrieval at YouTube scale.

The paper's core pipeline is:

```text
item content and behavior signals
  -> Semantic ID tokenization
  -> continued pre-training on domain data
  -> task-specific SFT for generative retrieval
  -> beam search produces SID candidates
  -> SID-to-video mapping returns retrieval candidates
```

## Materials

1. [Chinese course outline](./course-outline-zh.md)
2. [Six-lesson tutorial](./tutorial/README.md)
3. [Source paper PDF](./paper.pdf)

## Tutorial Lessons

1. [From embedding retrieval to generative retrieval](./tutorial/lesson-01-paradigm-shift.md)
2. [Semantic IDs and item tokenization](./tutorial/lesson-02-semantic-id-tokenization.md)
3. [Continued pre-training for recommendation domain adaptation](./tutorial/lesson-03-continued-pretraining.md)
4. [Generative retrieval training and serving](./tutorial/lesson-04-generative-retrieval.md)
5. [Evaluation, ablation, and scaling lessons](./tutorial/lesson-05-evaluation-scaling.md)
6. [Architecture checklist and demo plan](./tutorial/lesson-06-architecture-demo-plan.md)

## Key Takeaway

PLUM is not just "use an LLM for recommendation." It changes item representation, training data construction, retrieval training, and serving. The main engineering question is how to turn item corpora and user behavior into a token language that an LLM can learn, generate, and serve reliably.

