# BossPig Performance Benchmark Report
**Date**: 2025-11-22 21:50:35
**Results**: 6 benchmark runs

## Summary
- **Average Duration**: 95.1ms
- **Average Throughput**: 33823 words/sec
- **Largest Document**: 10,043 words

## Detailed Results
| Document Size | Duration (ms) | Throughput (words/sec) | Findings |
|--------------|---------------|------------------------|----------|
| 121 words | 4.2ms | 28861 | 37 |
| 605 words | 18.8ms | 32266 | 141 |
| 1,089 words | 28.2ms | 38658 | 245 |
| 2,057 words | 53.5ms | 38423 | 453 |
| 5,082 words | 158.6ms | 32047 | 1103 |
| 10,043 words | 307.3ms | 32681 | 2169 |

## Analysis
- **Scalability**: ~0.031ms per word
- **Complexity**: O(n) - linear with document size

## Performance Targets
- **Target**: <50ms for 1000-word documents
- **Target**: <100ms for 2000-word documents
- **1000-word target**: [PASSED] (28.2ms)
- **2000-word target**: [PASSED] (53.5ms)
