# Phase 5 Cache Tuning Guide
## Optimizing Compositional Cache Performance

**Audience**: DevOps, Performance Engineers, Production Deployments
**Prerequisites**: Phase 5 activated (see [PHASE_5_ACTIVATION_GUIDE.md](PHASE_5_ACTIVATION_GUIDE.md))

---

## Quick Reference

### Recommended Configurations

| Environment | Parse Cache | Merge Cache | Memory | Expected Hit Rate |
|-------------|-------------|-------------|--------|-------------------|
| **Development** | 1,000 | 5,000 | ~1MB | 40-50% |
| **Staging** | 10,000 | 50,000 | ~10MB | 60-70% |
| **Production** | 100,000 | 500,000 | ~100MB | 70-80% |
| **High-Volume** | 500,000 | 2,000,000 | ~500MB | 80-90% |

### Configuration Template

```python
from hololoom.config import Config

# Production configuration (recommended)
config = Config.fused()
config.enable_linguistic_gate = True
config.use_compositional_cache = True
config.linguistic_mode = "disabled"  # Cache only
config.parse_cache_size = 100000     # 100k phrases
config.merge_cache_size = 500000     # 500k compositions
```

---

## Understanding Cache Behavior

### Three-Tier Architecture

Phase 5 uses three caching tiers:

1. **Tier 1: Parse Cache** (X-bar structures)
   - Caches: Phrase parse trees
   - Speedup: 10-50×
   - Size: ~10 bytes/entry (parse tree reference)

2. **Tier 2: Merge Cache** (Compositional embeddings)
   - Caches: Composed embeddings (building blocks)
   - Speedup: 5-10×
   - Size: ~3KB/entry (768d float32 embedding)

3. **Tier 3: Semantic Cache** (244D projections)
   - Caches: Semantic dimension scores
   - Speedup: 3-10×
   - Size: ~2KB/entry (244d float32 vector)

**Key insight**: Tier 2 (merge cache) provides the most compositional reuse!

### Cache States

Queries can be in three states:

- **Cold** (0 hits): Full computation (~15-25ms)
- **Warm** (partial hits): Some reuse (~3-8ms, 3-5× faster)
- **Hot** (full hits): Maximum reuse (~0.01-0.1ms, 100-1000× faster)

**Distribution in production**: ~20% cold, ~30% warm, ~50% hot

---

## Tuning Strategy

### Step 1: Establish Baseline

Run the benchmark suite to understand your workload:

```bash
python benchmarks/phase5_performance_benchmark.py
```

This generates:
- `benchmarks/results/phase5_benchmark.json` - Raw data
- `benchmarks/results/phase5_benchmark.md` - Analysis report

**Key metrics to track**:
- Overall hit rate (target: >65%)
- Cold→hot speedup (target: >50×)
- P95 latency (target: <10ms)
- Cache utilization (target: 60-80% full)

### Step 2: Identify Bottlenecks

Check cache utilization:

```python
# In production code
stats = cache.get_statistics()

parse_util = stats['parse_cache']['size'] / stats['parse_cache']['capacity']
merge_util = stats['merge_cache']['size'] / stats['merge_cache']['capacity']

print(f"Parse cache: {parse_util:.1%} full")
print(f"Merge cache: {merge_util:.1%} full")
```

**Interpretation**:
- **<50% full**: Cache too large (wasted memory)
- **50-80% full**: Optimal (room for growth)
- **80-95% full**: Near capacity (consider increasing)
- **>95% full**: Thrashing (increase immediately!)

### Step 3: Adjust Cache Sizes

Based on utilization and hit rates:

#### Scenario A: Low Hit Rate (<50%), Low Utilization (<50%)

**Diagnosis**: Query diversity is high, cache not retaining useful entries

**Solution**: Increase cache sizes proportionally
```python
config.parse_cache_size *= 2   # Double both
config.merge_cache_size *= 2
```

#### Scenario B: Good Hit Rate (>65%), High Utilization (>90%)

**Diagnosis**: Cache is working but near capacity

**Solution**: Increase cache sizes moderately
```python
config.parse_cache_size = int(config.parse_cache_size * 1.5)   # 50% increase
config.merge_cache_size = int(config.merge_cache_size * 1.5)
```

#### Scenario C: Low Hit Rate (<50%), High Utilization (>90%)

**Diagnosis**: Eviction happening too frequently, need more capacity

**Solution**: Significant increase
```python
config.parse_cache_size *= 3   # Triple cache sizes
config.merge_cache_size *= 3
```

#### Scenario D: Good Hit Rate (>70%), Low Utilization (<40%)

**Diagnosis**: Cache oversized, wasting memory

**Solution**: Reduce cache sizes
```python
config.parse_cache_size = int(config.parse_cache_size * 0.6)   # 40% reduction
config.merge_cache_size = int(config.merge_cache_size * 0.6)
```

### Step 4: Monitor and Iterate

After adjusting:
1. Deploy changes
2. Monitor for 24-48 hours
3. Re-run benchmark
4. Compare metrics
5. Iterate if needed

---

## Memory Considerations

### Memory Usage Estimates

**Parse cache** (Tier 1):
```
Memory = entries × 10 bytes
10,000 entries ≈ 100KB
100,000 entries ≈ 1MB
1,000,000 entries ≈ 10MB
```

**Merge cache** (Tier 2):
```
Memory = entries × 3KB (768d float32 embeddings)
1,000 entries ≈ 3MB
10,000 entries ≈ 30MB
100,000 entries ≈ 300MB
500,000 entries ≈ 1.5GB
```

**Total memory**:
```
Small (1k/5k):     ~15MB
Medium (10k/50k):  ~150MB
Large (100k/500k): ~1.5GB
XLarge (1M/2M):    ~6GB
```

### Memory Limits

Set absolute limits based on container/instance size:

```python
# For 4GB container
max_memory_mb = 4000 * 0.25  # Use 25% for cache (1GB)
merge_entry_size_kb = 3

max_merge_entries = int((max_memory_mb * 1000) / merge_entry_size_kb)
config.merge_cache_size = max_merge_entries  # ~333k entries

# Parse cache is much smaller, set proportionally
config.parse_cache_size = max_merge_entries // 5  # ~66k entries
```

---

## Production Monitoring

### Key Metrics to Track

1. **Hit Rates** (gauge)
   - Parse cache hit rate
   - Merge cache hit rate
   - Overall hit rate
   - **Target**: >65% overall

2. **Latency** (histogram)
   - P50, P95, P99 query latency
   - Cold/warm/hot path distributions
   - **Target**: P95 <10ms

3. **Cache Utilization** (gauge)
   - Cache size / capacity
   - **Target**: 60-80%

4. **Speedup** (gauge)
   - Cold→hot speedup factor
   - **Target**: >50×

### Prometheus Integration

```python
from hololoom.performance.cache_metrics import CacheMetricsCollector

# Initialize collector
metrics = CacheMetricsCollector(cache)

# Track queries
with metrics.track_query() as tracker:
    embedding, trace = cache.get_compositional_embedding(query)
    tracker.record_result(trace)

# Expose /metrics endpoint (Flask example)
@app.route('/metrics')
def metrics():
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return generate_latest(metrics.registry), 200, {'Content-Type': CONTENT_TYPE_LATEST}
```

### Grafana Dashboard

Recommended panels:

1. **Cache Hit Rates** (line graph)
   - Parse, merge, overall hit rates over time
   - Alerts: Overall <50%

2. **Query Latency** (heatmap)
   - Distribution of cold/warm/hot queries
   - Alerts: P95 >20ms

3. **Cache Utilization** (gauge)
   - Parse and merge cache % full
   - Alerts: >90% full

4. **Speedup Factor** (gauge)
   - Real-time cold→hot speedup
   - Alerts: <20×

---

## Advanced Tuning

### Cache Ratio Tuning

The parse:merge ratio affects performance:

**Default ratio**: 1:5 (parse:merge)
```python
parse = 100000
merge = 500000  # 5× larger
```

**Adjust based on query patterns**:

- **High phrase diversity** → Increase parse cache
  ```python
  parse = 200000  # 2× increase
  merge = 500000  # Keep same
  # Ratio: 1:2.5
  ```

- **High compositional reuse** → Increase merge cache
  ```python
  parse = 100000  # Keep same
  merge = 1000000  # 2× increase
  # Ratio: 1:10
  ```

### Eviction Strategy

Current implementation: **Simple FIFO** (first in, first out)

For advanced use cases, consider:

- **LRU** (Least Recently Used): Better for temporal locality
- **LFU** (Least Frequently Used): Better for popularity-based
- **Weighted**: Combine frequency + recency + importance

**Note**: These require custom cache implementation.

### Cache Warming

Pre-load common queries at startup:

```python
# At application startup
common_queries = [
    "What is machine learning?",
    "How does neural network work?",
    "Explain reinforcement learning",
    # ... top 100 queries
]

for query in common_queries:
    cache.get_compositional_embedding(query)

print(f"Cache warmed with {len(common_queries)} queries")
```

### Regional Caching

For multi-region deployments:

1. **Shared cache** (Redis): Share parse/merge caches across regions
2. **Per-region cache**: Optimize for regional query patterns
3. **Hybrid**: Share parse cache, region-specific merge caches

---

## Troubleshooting

### Problem: Low Hit Rates (<40%)

**Possible causes**:
1. Query diversity too high (expected for some workloads)
2. Cache sizes too small (check utilization)
3. Eviction happening too frequently

**Solutions**:
- Increase cache sizes (2-3×)
- Check query patterns (are they truly diverse?)
- Consider cache warming for common queries

### Problem: High Memory Usage

**Possible causes**:
1. Cache sizes set too large
2. Memory leak (unlikely, but check)

**Solutions**:
- Reduce cache sizes based on utilization
- Monitor memory growth over time
- Restart if necessary (caches rebuild quickly)

### Problem: Cache Not Being Used

**Symptoms**:
- Hit rates at 0%
- All queries take same time

**Diagnosis**:
```python
# Check if Phase 5 is enabled
if shuttle.linguistic_gate:
    print("✅ Linguistic gate enabled")
    if shuttle.linguistic_gate.compositional_cache:
        print("✅ Compositional cache enabled")
    else:
        print("❌ Compositional cache NOT enabled")
else:
    print("❌ Linguistic gate NOT enabled")
```

**Solutions**:
- Ensure `config.enable_linguistic_gate = True`
- Ensure `config.use_compositional_cache = True`
- Check spaCy installation: `python -m spacy download en_core_web_sm`

### Problem: Inconsistent Performance

**Symptoms**:
- Latency varies wildly
- Hit rates fluctuate

**Possible causes**:
1. Cache thrashing (eviction too frequent)
2. External resource contention
3. GC pauses

**Solutions**:
- Increase cache sizes
- Check system resources (CPU, memory)
- Monitor GC activity
- Profile with `cProfile` or `py-spy`

---

## Benchmarking Checklist

Before production deployment:

- [ ] Run benchmark suite on production hardware
- [ ] Measure hit rates with production query patterns
- [ ] Validate memory usage within limits
- [ ] Test cache behavior under load (concurrent queries)
- [ ] Verify Prometheus metrics are being collected
- [ ] Set up Grafana dashboards
- [ ] Configure alerts for low hit rates, high latency
- [ ] Document cache sizes in deployment config
- [ ] Create runbook for cache tuning

---

## Reference: Cache Size Recommendations by Workload

### Low-Volume (<1000 q/day)
```python
config.parse_cache_size = 1000
config.merge_cache_size = 5000
# Memory: ~15MB, Hit rate: 40-50%
```

### Medium-Volume (1k-10k q/day)
```python
config.parse_cache_size = 10000
config.merge_cache_size = 50000
# Memory: ~150MB, Hit rate: 60-70%
```

### High-Volume (10k-100k q/day)
```python
config.parse_cache_size = 100000
config.merge_cache_size = 500000
# Memory: ~1.5GB, Hit rate: 70-80%
```

### Very High-Volume (>100k q/day)
```python
config.parse_cache_size = 500000
config.merge_cache_size = 2000000
# Memory: ~6GB, Hit rate: 80-90%
```

---

## Summary

**Key takeaways**:
1. Start with recommended production config (100k/500k)
2. Monitor hit rates and utilization for 24-48 hours
3. Adjust based on metrics (see decision tree above)
4. Iterate until hit rate >65% and utilization 60-80%
5. Set up continuous monitoring with Prometheus + Grafana

**Most important metric**: **Overall cache hit rate**
- <50%: Needs tuning
- 50-65%: Acceptable
- 65-80%: Optimal
- >80%: Excellent

**Remember**: Phase 5 gracefully degrades without spaCy. If cache is unavailable, system falls back to standard embeddings with no impact on functionality.

---

## Further Reading

- [PHASE_5_ACTIVATION_GUIDE.md](PHASE_5_ACTIVATION_GUIDE.md) - Activation instructions
- [phase5_performance_benchmark.py](../../benchmarks/phase5_performance_benchmark.py) - Benchmark suite
- [cache_metrics.py](../../hololoom/performance/cache_metrics.py) - Prometheus metrics
- [PHASE_5_SHIPPED.md](../../PHASE_5_SHIPPED.md) - Technical summary

---

**Last Updated**: October 30, 2025
**Status**: Production Ready