# HoloLoom Benchmark Suite - Complete Results

**Date:** 2025-11-17

This report aggregates results from all HoloLoom benchmarks.

---

## Wikipedia Recall Accuracy


**Dataset:** wikipedia_synthetic
**Memories:** 10
**Queries:** 10

## Results

| Metric | Value |
|--------|-------|
| Precision@1 | 0.700 |
| Precision@5 | 0.340 |
| Precision@10 | 0.180 |
| Recall@5 | 0.950 |
| Recall@10 | 1.000 |
| MRR | 0.817 |
| Latency p95 | 0.1ms |
| Throughput | 30023.7 queries/sec |

---

## arXiv Recall Accuracy


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

---

## Books Recall Accuracy


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

---

## Cross-Dataset Summary

| Dataset | Precision@5 | Recall@5 | MRR | Latency p95 |
|---------|-------------|----------|-----|-------------|
| Wikipedia | 0.340 | 0.950 | 0.817 | 0.1ms |
| Arxiv | 0.333 | 0.911 | 0.761 | 0.2ms |
| Books | 0.217 | 0.875 | 0.931 | 0.2ms |

**Key Insights:**

- All datasets show high recall (>85%) - HoloLoom reliably retrieves relevant memories
- MRR consistently high (>0.75) - relevant items rank near the top
- Sub-millisecond latency across all datasets - suitable for real-time applications
- Precision varies by dataset complexity - room for improvement via reranking

