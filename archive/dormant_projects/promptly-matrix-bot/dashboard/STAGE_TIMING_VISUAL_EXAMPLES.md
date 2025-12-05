# Stage Timing Chart - Visual Examples

## Example 1: Balanced Query (Optimal)

```
Total Latency:    135.2ms
Avg per Stage:    15.0ms
Slowest Stage:    Convergence Engine (22.5ms)
Stages Completed: 9/9

█░░░░░░░░░░░░░░░░░░░░░░░░░░░ 1. Loom Command          12.1ms  9% ✓
░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 2. Chrono Trigger        14.3ms 11% ✓
░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 3. Yarn Graph            13.8ms 10% ✓
░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 4. Resonance Shed        16.2ms 12% ✓
░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 5. Warp Space            14.9ms 11% ✓
███░░░░░░░░░░░░░░░░░░░░░░░░░░ 6. Convergence Engine    22.5ms 17% ✓
░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 7. Tool Execution        15.1ms 11% ✓
░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 8. Spacetime Fabric      14.7ms 11% ✓
░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 9. Reflection Buffer     11.6ms  9% ✓

NO BOTTLENECKS DETECTED ✓

Performance Summary:
├─ Fastest Stage: Reflection Buffer (11.6ms)
├─ Median Duration: 14.7ms
└─ Optimization Potential: 0% (System is well-balanced)

Status: OPTIMAL ✓ Ready for production
```

**Color Legend**: All green (< 20% each)

---

## Example 2: Retrieval Bottleneck

```
Total Latency:    201.5ms
Avg per Stage:    22.4ms
Slowest Stage:    Yarn Graph (87.3ms)
Stages Completed: 9/9

████████████████████░░░░░░░░░░░ 1. Loom Command          10.2ms  5% ✓
░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 2. Chrono Trigger        12.1ms  6% ✓
██████████████████████████████████░░░ 3. Yarn Graph            87.3ms 43% ⚠ BOTTLENECK ✓
████░░░░░░░░░░░░░░░░░░░░░░░░░░░ 4. Resonance Shed        18.5ms  9% ✓
░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 5. Warp Space            11.3ms  6% ✓
████░░░░░░░░░░░░░░░░░░░░░░░░░░░ 6. Convergence Engine    19.8ms 10% ✓
░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 7. Tool Execution        14.2ms  7% ✓
░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 8. Spacetime Fabric      13.7ms  7% ✓
░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 9. Reflection Buffer      9.2ms  5% ✓

🚨 BOTTLENECK ALERT: 1 Bottleneck Detected
   └─ Yarn Graph exceeding 40% of total time

Performance Summary:
├─ Fastest Stage: Reflection Buffer (9.2ms)
├─ Median Duration: 13.7ms
└─ Optimization Potential: 43% reduction possible if optimized

Recommendations:
1. Yarn Graph (memory retrieval) is too slow (87.3ms)
2. Try reducing memory shard count (-25% to start)
3. Or switch to BARE mode for faster retrieval
4. Profile memory backend (Neo4j/Qdrant?)
5. Could reduce total latency from 201.5ms to ~115ms

Status: NEEDS OPTIMIZATION ⚠
```

**Color Legend**:
- 🟢 Green (< 20%): Loom, Chrono, Warp, Tool, Spacetime, Reflection
- 🟡 Yellow (20-40%): Resonance, Convergence
- 🔴 Red (> 40%): Yarn Graph (BOTTLENECK!)

---

## Example 3: Feature Extraction Bottleneck

```
Total Latency:    178.9ms
Avg per Stage:    19.9ms
Slowest Stage:    Resonance Shed (75.2ms)
Stages Completed: 9/9

███░░░░░░░░░░░░░░░░░░░░░░░░░░░ 1. Loom Command          13.4ms  7% ✓
███░░░░░░░░░░░░░░░░░░░░░░░░░░░ 2. Chrono Trigger        13.9ms  8% ✓
███░░░░░░░░░░░░░░░░░░░░░░░░░░░ 3. Yarn Graph            15.6ms  9% ✓
██████████████████████████████░░░ 4. Resonance Shed        75.2ms 42% ⚠ BOTTLENECK ✓
████░░░░░░░░░░░░░░░░░░░░░░░░░░░ 5. Warp Space            17.8ms 10% ✓
████░░░░░░░░░░░░░░░░░░░░░░░░░░░ 6. Convergence Engine    18.3ms 10% ✓
░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 7. Tool Execution        11.2ms  6% ✓
░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 8. Spacetime Fabric      10.8ms  6% ✓
░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 9. Reflection Buffer      8.5ms  5% ✓

🚨 BOTTLENECK ALERT: 1 Bottleneck Detected
   └─ Resonance Shed exceeding 40% of total time

Performance Summary:
├─ Fastest Stage: Reflection Buffer (8.5ms)
├─ Median Duration: 14.6ms
└─ Optimization Potential: 42% reduction possible if optimized

Recommendations:
1. Resonance Shed (feature extraction) is slow (75.2ms)
2. Extracting embeddings for 100+ memory shards?
3. Try BARE mode (regex motifs only, skip embeddings)
4. Or reduce memory shard count
5. Consider caching embeddings between queries

Status: NEEDS OPTIMIZATION ⚠
```

**Color Legend**:
- 🟢 Green: Loom, Chrono, Yarn, Warp, Tool, Spacetime, Reflection
- 🟡 Yellow: Convergence
- 🔴 Red: Resonance Shed (BOTTLENECK!)

---

## Example 4: Multiple Bottlenecks

```
Total Latency:    287.3ms
Avg per Stage:    31.9ms
Slowest Stage:    Yarn Graph (115.2ms)
Stages Completed: 9/9

██████░░░░░░░░░░░░░░░░░░░░░░░░ 1. Loom Command          18.5ms  6% ✓
█████░░░░░░░░░░░░░░░░░░░░░░░░░░ 2. Chrono Trigger        16.2ms  6% ✓
███████████████████████████████░░░ 3. Yarn Graph            115.2ms 40% ⚠ BOTTLENECK ✓
██████░░░░░░░░░░░░░░░░░░░░░░░░ 4. Resonance Shed        51.8ms 18% ✓
█████░░░░░░░░░░░░░░░░░░░░░░░░░░ 5. Warp Space            22.3ms  8% ✓
███████░░░░░░░░░░░░░░░░░░░░░░░░░░ 6. Convergence Engine    28.7ms 10% ✓
████░░░░░░░░░░░░░░░░░░░░░░░░░░░ 7. Tool Execution        19.4ms  7% ✓
████░░░░░░░░░░░░░░░░░░░░░░░░░░░ 8. Spacetime Fabric      10.7ms  4% ✓
███░░░░░░░░░░░░░░░░░░░░░░░░░░░ 9. Reflection Buffer      4.5ms  2% ✓

🚨 BOTTLENECK ALERT: 1 Bottleneck Detected
   └─ Yarn Graph exceeding 40% of total time

⚠️ WARNING: Additional caution-level stages (20-40%):
   └─ Resonance Shed (18%)

Performance Summary:
├─ Fastest Stage: Reflection Buffer (4.5ms)
├─ Median Duration: 18.5ms
└─ Optimization Potential: 40% reduction possible if optimized

Recommendations:
1. PRIMARY: Yarn Graph bottleneck (115.2ms)
   - Reduce memory shards from 1000 to 300
   - Use BARE mode to skip embeddings first
   - Profile Neo4j performance
2. SECONDARY: Resonance Shed warning (51.8ms)
   - If still slow after Yarn Graph fix, optimize embeddings
   - Consider Matryoshka multi-scale reduction

Status: CRITICAL OPTIMIZATION NEEDED 🚨
```

**Color Legend**:
- 🟡 Yellow (20-40%): Resonance
- 🔴 Red (> 40%): Yarn Graph (PRIMARY BOTTLENECK!)

---

## Example 5: Fast Query (BARE Mode)

```
Total Latency:    65.3ms
Avg per Stage:    7.3ms
Slowest Stage:    Resonance Shed (12.1ms)
Stages Completed: 9/9

████░░░░░░░░░░░░░░░░░░░░░░░░░░░ 1. Loom Command          5.2ms  8% ✓
██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 2. Chrono Trigger        3.1ms  5% ✓
███░░░░░░░░░░░░░░░░░░░░░░░░░░░ 3. Yarn Graph            4.8ms  7% ✓
████████░░░░░░░░░░░░░░░░░░░░░░░ 4. Resonance Shed        12.1ms 19% ✓
██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 5. Warp Space            2.9ms  4% ✓
███░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 6. Convergence Engine    8.3ms 13% ✓
██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 7. Tool Execution        3.4ms  5% ✓
██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 8. Spacetime Fabric      3.8ms  6% ✓
█░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 9. Reflection Buffer     1.7ms  3% ✓

NO BOTTLENECKS DETECTED ✓

Performance Summary:
├─ Fastest Stage: Reflection Buffer (1.7ms)
├─ Median Duration: 4.8ms
└─ Optimization Potential: 0% (All stages well-optimized)

Notes:
- BARE mode execution (minimal processing)
- Very fast due to: regex-only motifs, no embeddings, simple policy
- Good for repeated/cached queries
- Trade-off: Lower accuracy (confidence ~0.75 vs 0.92 in FUSED)

Status: EXCELLENT - PRODUCTION READY ✓
Mode: BARE
Confidence: 0.75 (acceptable for simple queries)
```

**Color Legend**: All green (< 20% each)

---

## Summary Table

| Scenario | Total Latency | Bottleneck? | Status | Recommendation |
|----------|---------------|-------------|--------|-----------------|
| **Example 1** | 135.2ms | None | ✓ Optimal | Deploy as-is |
| **Example 2** | 201.5ms | Yarn Graph (43%) | ⚠ Warning | Reduce memory shards |
| **Example 3** | 178.9ms | Resonance (42%) | ⚠ Warning | Use BARE mode |
| **Example 4** | 287.3ms | Yarn Graph (40%) | 🚨 Critical | Urgent optimization |
| **Example 5** | 65.3ms | None | ✓ Excellent | BARE mode best practice |

---

## How to Identify Your Stage Timing

When you see your dashboard:

1. **Look at Total Latency** - top left number
2. **Find the red bar** - that's your bottleneck (if any)
3. **Check Optimization Potential** - bottom right number
4. **Read the recommendation** - follow what it suggests

## Performance Standards by Mode

| Mode | Expected Total | Slowest Stage | Bottleneck? |
|------|---|---|---|
| BARE | 50-80ms | <15ms | Rarely |
| FAST | 100-150ms | <40ms | Occasionally |
| FUSED | 150-250ms | <80ms | More likely |
| RESEARCH | 200-500ms | <150ms | Expected |

---

**Created**: November 2025
**Version**: 1.0
