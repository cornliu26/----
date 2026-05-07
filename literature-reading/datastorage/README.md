# Data Storage

This category collects papers and tutorials about data storage, data ingestion, online preprocessing, and training-data infrastructure for large-scale recommendation systems.

## Papers

1. [Understanding Data Storage and Ingestion for Large-Scale Deep Recommendation Model Training](./understanding-data-storage-ingestion-large-scale-dlrm-training/README.md)

## Why This Category Matters

Large-scale recommendation training is constrained not only by GPU compute, but also by the full data path:

```text
raw logs
  -> ETL and labeling
  -> warehouse table
  -> distributed columnar storage
  -> online preprocessing
  -> tensors
  -> GPU trainer
```

For this category, focus on:

1. How training samples are stored and filtered.
2. How online preprocessing avoids GPU data stalls.
3. How file layout, feature projection, and storage hardware interact.
4. How DSI power and capacity affect training throughput.
5. How to turn paper insights into workload profiles and PoC plans.
