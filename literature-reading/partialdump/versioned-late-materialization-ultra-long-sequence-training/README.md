# Versioned Late Materialization for Ultra-Long Sequence Training

Paper: `Versioned Late Materialization for Ultra-Long Sequence Training in Recommendation Systems at Scale`

Source: [paper.pdf](./paper.pdf)

## What This Paper Is About

This paper studies how to support ultra-long User Interaction History (UIH) training in industrial recommendation systems without duplicating the full history into every training example.

The core idea is to move from Fat Row pre-materialization to versioned late materialization:

```text
Fat Row:
  each training example stores complete UIH

Versioned late materialization:
  each example stores recent mutable sequence plus lightweight version metadata
  long-term immutable UIH is reconstructed during training
```

## Materials

1. [Chinese course outline](./course-outline-zh.md)
2. [Six-lesson tutorial](./tutorial/README.md)
3. [Source paper PDF](./paper.pdf)

## Tutorial Lessons

1. [Fat Row and the long-sequence data wall](./tutorial/lesson-01-fat-row-data-wall.md)
2. [O2O consistency and future leakage](./tutorial/lesson-02-o2o-future-leakage.md)
3. [Versioned late materialization protocol](./tutorial/lesson-03-versioned-late-materialization.md)
4. [Immutable UIH store and projection pushdown](./tutorial/lesson-04-immutable-uih-store.md)
5. [Training-time materialization and throughput optimization](./tutorial/lesson-05-training-time-materialization.md)
6. [Evaluation, PoC design, and rollout checklist](./tutorial/lesson-06-evaluation-and-poc.md)

## Key Takeaway

The paper treats data infrastructure as a first-class scaling lever for recommendation quality. The quality gain does not come from late materialization itself; it comes from making longer UIH training feasible within a practical storage and I/O budget.

