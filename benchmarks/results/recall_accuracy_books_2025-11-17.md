# Books Recall Accuracy Benchmark

**Date:** 2025-11-17T04:58:49.989296

**Dataset:** books_synthetic
**Books:** 15
**Queries:** 12
**Genres:** Fiction, Non-fiction, Fantasy, Science

## Results

| Metric | Value |
|--------|-------|
| Precision@1 | 0.917 |
| Precision@5 | 0.217 |
| Precision@10 | 0.125 |
| Recall@5 | 0.875 |
| Recall@10 | 0.958 |
| MRR | 0.931 |
| Latency p95 | 0.2ms |
| Throughput | 8614.0 queries/sec |

## Analysis

**Dataset Characteristics:**
- 15 books from classic and modern literature
- Mix of fiction (novels) and non-fiction (science, business)
- Long-form text (~100 word summaries)
- Thematic queries (not keyword-based)

**Performance Notes:**
- ✅ Strong MRR (0.931) - relevant books rank high
- ✅ Sub-millisecond latency - suitable for interactive applications
