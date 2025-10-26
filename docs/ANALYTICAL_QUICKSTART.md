# Analytical Orchestrator - Quick Start Guide

**Get started with mathematically rigorous semantic processing in 5 minutes.**

---

## Installation

The analytical orchestrator requires the warp math modules:

```bash
# Already installed if you have HoloLoom
cd mythRL
python -m pip install -e .
```

---

## Your First Analytical Query

```python
import asyncio
from HoloLoom.analytical_orchestrator import create_analytical_orchestrator

async def main():
    # Create orchestrator with all analysis enabled
    weaver = create_analytical_orchestrator(
        pattern="fast",           # Use "fast" for quick results
        enable_all_analysis=True  # Enable all mathematical checks
    )
    
    # Execute query
    spacetime = await weaver.weave("What is machine learning?")
    
    # View results
    print(f"Result: {spacetime.response[:200]}...")
    print(f"Tool: {spacetime.tool_used}")
    print(f"Confidence: {spacetime.confidence:.2%}")
    
    # View analytical metrics
    if hasattr(spacetime.trace, 'analytical_metrics'):
        metrics = spacetime.trace.analytical_metrics
        
        print("\nMathematical Guarantees:")
        if 'metric_space' in metrics:
            print(f"  ✓ Valid metric: {metrics['metric_space']['is_valid_metric']}")
        
        if 'spectral_stability' in metrics:
            print(f"  ✓ Stable: {metrics['spectral_stability']['is_stable']}")
    
    weaver.stop()

asyncio.run(main())
```

---

## What You Get

### 1. Metric Space Verification

**Checks:** Non-negativity, symmetry, triangle inequality

```python
metrics['metric_space']
# {
#   'is_valid_metric': True,     # ✓ Distances are mathematically valid
#   'is_complete': True,          # ✓ Cauchy sequences converge
#   'num_elements': 10,           # Number of threads
#   'dimension': 384              # Embedding dimension
# }
```

### 2. Hilbert Space Orthogonalization

**Checks:** Information diversity via Gram-Schmidt

```python
metrics['orthogonalization']
# {
#   'threads_processed': 10,      # Threads orthogonalized
#   'diversity_score': 1.234,     # Mean pairwise angle (higher = more diverse)
#   'orthogonality_achieved': True # ✓ All threads orthogonal
# }
```

### 3. Gradient-Guided Attention

**Checks:** Optimal query positioning

```python
metrics['gradient_optimization']
# {
#   'gradient_norm': 0.0524,      # Magnitude of gradient
#   'improvement': 0.0123,        # Relevance improvement
#   'original_relevance': 0.456,  # Before optimization
#   'optimized_relevance': 0.468  # After optimization
# }
```

### 4. Spectral Stability Analysis

**Checks:** Non-explosive attention dynamics

```python
metrics['spectral_stability']
# {
#   'spectral_radius': 0.845,     # ✓ < 1 means stable
#   'is_stable': True,            # ✓ Won't explode
#   'largest_eigenvalue': 0.832,  # Dominant eigenvalue
#   'condition_number': 12.3      # Matrix conditioning
# }
```

### 5. Continuity Verification

**Checks:** Bounded sensitivity to input changes

```python
metrics['continuity']
# {
#   'lipschitz_constant': 45.2,   # Max rate of change
#   'is_smooth': True,            # ✓ Small inputs → small outputs
#   'samples_tested': 10          # Number of samples
# }
```

### 6. Convergence Tracking

**Checks:** Learning over time (after 10+ queries)

```python
metrics['convergence']
# {
#   'is_converging': True,        # ✓ Stabilizing
#   'limit': 0.85,                # Limiting confidence
#   'is_monotone': True,          # Monotonic improvement
#   'direction': 'increasing'     # Performance improving
# }
```

---

## Common Use Cases

### Production Retrieval

```python
# High-quality, mathematically sound retrieval
weaver = create_analytical_orchestrator(
    pattern="fused",
    enable_metric_verification=True,    # Always verify metrics
    enable_gradient_optimization=True,  # Optimize relevance
    enable_spectral_analysis=True       # Ensure stability
)
```

### Research & Debugging

```python
# Full analytical suite for rigorous experiments
weaver = create_analytical_orchestrator(
    pattern="fused",
    enable_all_analysis=True  # Everything enabled
)

# Run multiple queries and track convergence
for query in research_queries:
    spacetime = await weaver.weave(query)
    log_metrics(spacetime.trace.analytical_metrics)
```

### Performance-Critical

```python
# Minimal overhead, essential checks only
weaver = create_analytical_orchestrator(
    pattern="fast",
    enable_metric_verification=True,    # Low overhead, high value
    enable_gradient_optimization=False,
    enable_hilbert_orthogonalization=False,
    enable_spectral_analysis=False
)
```

---

## Interpreting Results

### Metric Verification

✅ **Valid metric = True** → Distances are mathematically sound  
❌ **Valid metric = False** → Check embedding function

### Diversity Score

📊 **High (>1.0 rad)** → Threads are informationally independent  
📊 **Medium (0.5-1.0)** → Some information overlap  
📊 **Low (<0.5)** → Redundant context

### Gradient Improvement

⬆️ **Positive** → Query moved toward optimal position  
➡️ **Zero** → Already at critical point  
⬇️ **Negative** → Moved away from optimum (rare)

### Spectral Radius

✅ **< 1.0** → Stable (safe for iterative use)  
⚠️ **≈ 1.0** → Critical (borderline stable)  
❌ **> 1.0** → Unstable (will explode)

### Lipschitz Constant

✅ **< 100** → Smooth, bounded sensitivity  
⚠️ **100-1000** → Moderate sensitivity  
❌ **> 1000** → High sensitivity (check embeddings)

### Convergence

✅ **Converging + Increasing** → Learning and improving  
✅ **Converging + Constant** → Stable performance  
⚠️ **Converging + Decreasing** → Performance degrading  
❌ **Not converging** → Unstable learning

---

## Tips & Tricks

### Enable Based on Pattern

```python
# Fast: Minimal analysis
if pattern == "fast":
    enable_metric_verification=True
    enable_others=False

# Fused: Full analysis
elif pattern == "fused":
    enable_all_analysis=True
```

### Track Over Time

```python
metrics_history = []

for query in queries:
    spacetime = await weaver.weave(query)
    metrics_history.append(spacetime.trace.analytical_metrics)

# Analyze trends
convergence_trend = [m.get('convergence', {}) for m in metrics_history]
```

### Custom Thresholds

```python
# Check custom conditions
metrics = spacetime.trace.analytical_metrics

if metrics['spectral_stability']['spectral_radius'] > 0.95:
    logger.warning("Approaching instability!")

if metrics['continuity']['lipschitz_constant'] > 500:
    logger.warning("High sensitivity detected!")
```

---

## Next Steps

1. **Read full documentation:** [ANALYTICAL_ORCHESTRATOR.md](ANALYTICAL_ORCHESTRATOR.md)
2. **Run demos:** `python demos/analytical_weaving_demo.py`
3. **Explore math modules:** [warp/math/analysis/](../HoloLoom/warp/math/analysis/)
4. **Integrate with your pipeline:** Replace `WeavingOrchestrator` with `AnalyticalWeavingOrchestrator`

---

## Troubleshooting

### Import Error

```python
# Error: Cannot import analytical_orchestrator
# Solution: Ensure math modules are available
from HoloLoom.warp.math.analysis import MetricSpace  # Test import
```

### Slow Performance

```python
# Solution: Disable expensive analyses
weaver = create_analytical_orchestrator(
    enable_hilbert_orthogonalization=False,  # Most expensive
    enable_continuity_verification=False     # Second most expensive
)
```

### No Analytical Metrics

```python
# Check if warp space was tensioned
if not weaver.warp.is_tensioned:
    logger.warning("Warp space not tensioned - no context to analyze")
```

---

## Mathematical Guarantees Summary

| **Check** | **Guarantee** | **Overhead** |
|-----------|---------------|--------------|
| Metric axioms | Valid distances | Low (~10ms) |
| Orthogonalization | Maximal diversity | Medium (~50ms) |
| Gradient optimization | Optimal relevance | Medium (~30ms) |
| Spectral stability | Non-explosive | Low (~20ms) |
| Lipschitz continuity | Bounded sensitivity | Medium (~40ms) |
| Convergence | Learning verification | Negligible |

**Total overhead:** ~150ms for full suite (~67% over base orchestrator)

**Trade-off:** +150ms for mathematical proofs vs. empirical guesswork

---

**From heuristics to proofs in 5 minutes.** 🎯

Ready to make your semantic processing mathematically rigorous?

```python
weaver = create_analytical_orchestrator(enable_all_analysis=True)
spacetime = await weaver.weave("Transform my AI with mathematical rigor!")
```
