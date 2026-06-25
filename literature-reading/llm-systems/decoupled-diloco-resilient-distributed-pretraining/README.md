# Decoupled DiLoCo for Resilient Distributed Pre-training

This folder contains Chinese reading notes for **Decoupled DiLoCo for Resilient Distributed Pre-training**.

## Materials

- [paper.pdf](./paper.pdf): source paper.
- [course-outline-zh.md](./course-outline-zh.md): compact Chinese reading outline.
- [tutorial/lesson-01-system-design-flow.md](./tutorial/lesson-01-system-design-flow.md): contrast-style Chinese explanation of the system design and end-to-end flow.

## Reading Focus

The paper studies how to make large-scale pre-training resilient when hardware failures, slow learners, heterogeneous accelerators, and opportunistic compute make fully synchronous SPMD training fragile.

The key idea is to decompose monolithic data-parallel training into independent learners coordinated by a lightweight synchronizer. The system relaxes strict step-by-step global synchronization, but keeps model quality close to data-parallel training while improving goodput and availability.
