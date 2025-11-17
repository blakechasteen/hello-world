# arXiv Recall Accuracy Benchmark

**Date:** 2025-11-17T04:58:49.897110

**Dataset:** arxiv_synthetic
**Papers:** 20
**Queries:** 15
**Domains:** ML, NLP, CV, RL

## Results

| Metric | Value |
|--------|-------|
| Precision@1 | 0.600 |
| Precision@5 | 0.333 |
| Precision@10 | 0.180 |
| Recall@5 | 0.911 |
| Recall@10 | 0.967 |
| MRR | 0.761 |
| Latency p95 | 0.2ms |
| Throughput | 6246.5 queries/sec |

## Analysis

**Dataset Characteristics:**
- 20 scientific papers across ML/AI domains
- Topics: Transformers, RL, Vision-Language, Optimization
- Average abstract length: ~150 words

**Performance Notes:**
- ✅ High recall (91.1%) - retrieves relevant papers reliably
- ✅ Sub-millisecond latency - suitable for real-time applications
