# Partial Dump

This category collects papers and tutorials about partial dump, late materialization, point-in-time feature reconstruction, and training data infrastructure for large-scale recommendation systems.

## Papers

1. [Versioned Late Materialization for Ultra-Long Sequence Training in Recommendation Systems at Scale](./versioned-late-materialization-ultra-long-sequence-training/README.md)

## Why This Category Matters

Partial dump is not just a storage optimization. In recommendation training systems, it is usually about deciding:

1. Which request-time fields must be physically snapshotted.
2. Which heavy historical features can be reconstructed later.
3. Which version metadata is required to preserve Online-to-Offline consistency.
4. How to avoid shifting all cost from the write path to the training read path.

