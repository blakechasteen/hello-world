# Promptly Integration: Quick Start Guide

**Get started with recursive reasoning in HoloLoom in 5 minutes**

---

## 🚀 Quick Start

### 1. Basic Usage (Auto-Refinement)

```python
from HoloLoom.config import Config
from HoloLoom.weaving_orchestrator_recursive import RecursiveWeavingOrchestrator
from HoloLoom.documentation.types import Query, MemoryShard

# Create test shards
shards = [
    MemoryShard(
        content="Thompson Sampling balances exploration and exploitation",
        entities=["Thompson Sampling"],
        motifs=["exploration", "exploitation"]
    )
]

# Create orchestrator with auto-refinement
orchestrator = RecursiveWeavingOrchestrator(
    cfg=Config.fast(),
    shards=shards,
    enable_recursive=True,      # Enable recursive reasoning
    quality_threshold=0.85,     # Refine if confidence < 0.85
    max_iterations=3            # Maximum refinement passes
)

# Simple weave (auto-refines if needed)
query = Query(text="Explain Thompson Sampling")
spacetime = await orchestrator.weave(query)

print(f"Response: {spacetime.response}")
print(f"Confidence: {spacetime.confidence:.2f}")
print(f"Iterations: {spacetime.iterations}")
```

### 2. Explicit Strategy

```python
from HoloLoom.protocols.recursive_reasoning import ReasoningStrategy

# Use HOFSTADTER for philosophical questions
query = Query(text="What is consciousness?")
spacetime = await orchestrator.weave_with_strategy(
    query=query,
    strategy=ReasoningStrategy.HOFSTADTER,
    max_iterations=5
)

print(f"Strategy: {spacetime.strategy_used.value}")
print(f"Iterations: {spacetime.iterations}")
```

### 3. View Reasoning Provenance

```python
# Get complete reasoning history
if spacetime.reasoning_journal:
    history = spacetime.reasoning_journal.get_history()
    print(history)

    # View confidence trajectory
    trajectory = spacetime.reasoning_journal.get_confidence_trajectory()
    print(f"Quality: {trajectory[0]:.2f} → {trajectory[-1]:.2f}")
```

---

## 📋 Available Strategies

| Strategy | Use Case | Example Query |
|----------|----------|---------------|
| **DIRECT** | Simple factual queries | "What is Python?" |
| **REFINE** | Iterative improvement | "Explain machine learning" |
| **CRITIQUE** | Self-improvement | "Review this code" |
| **DECOMPOSE** | Complex multi-part | "How does X work and why?" |
| **EXPLORE** | Creative solutions | "What are alternatives to X?" |
| **HOFSTADTER** | Meta-reasoning | "What is consciousness?" |
| **VERIFY** | Fact-checking | "Is this claim accurate?" |
| **ADAPTIVE** | Auto-select | Any query (default) |

---

## 🎯 Common Patterns

### Pattern 1: Disable Recursion for Specific Query

```python
# Auto-refinement enabled by default
orchestrator = RecursiveWeavingOrchestrator(enable_recursive=True, ...)

# But disable for this specific query
spacetime = await orchestrator.weave(
    query,
    enable_refinement=False  # Skip refinement
)
```

### Pattern 2: Compare Strategies

```python
strategies = [
    ReasoningStrategy.REFINE,
    ReasoningStrategy.CRITIQUE,
    ReasoningStrategy.DECOMPOSE
]

for strategy in strategies:
    spacetime = await orchestrator.weave_with_strategy(
        query=query,
        strategy=strategy,
        max_iterations=3
    )

    print(f"{strategy.value}: "
          f"{spacetime.iterations} iterations, "
          f"confidence={spacetime.confidence:.2f}")
```

### Pattern 3: Custom Quality Threshold

```python
# Lower threshold = more refinement
orchestrator = RecursiveWeavingOrchestrator(
    quality_threshold=0.7,  # Refine if < 0.7 (more aggressive)
    max_iterations=5
)

# Higher threshold = less refinement
orchestrator = RecursiveWeavingOrchestrator(
    quality_threshold=0.95,  # Only refine if < 0.95 (conservative)
    max_iterations=2
)
```

---

## 🧪 Run the Demo

```bash
# Run comprehensive demo (5 examples)
PYTHONPATH=. python demos/demo_promptly_integration.py

# Outputs:
# 1. Basic weaving (no refinement)
# 2. Automatic refinement
# 3. Strategy comparison
# 4. Reasoning provenance
# 5. Hofstadter meta-reasoning
```

---

## 📖 Learn More

- **Full Documentation:** `PROMPTLY_HOLOLOOM_INTEGRATION.md`
- **Integration Summary:** `PROMPTLY_INTEGRATION_SUMMARY.md`
- **Promptly Review:** See comprehensive review in this chat

---

## 🎓 Key Concepts

### ReasoningJournal (Scratchpad)

Tracks complete thought process:

```python
journal = spacetime.reasoning_journal

# View all iterations
for trace in journal.traces:
    print(f"Iteration {trace.iteration}:")
    print(f"  Thought: {trace.thought}")
    print(f"  Action: {trace.action}")
    print(f"  Confidence: {trace.confidence:.2f}")

# Check if converged
if journal.converged():
    print("Reasoning stabilized (no improvement)")
```

### Stop Conditions

Refinement stops when:
1. **Quality threshold met:** `confidence >= quality_threshold`
2. **Max iterations reached:** `iterations >= max_iterations`
3. **No improvement:** `Δconfidence < min_improvement`
4. **Convergence:** Output stabilizes

### Adaptive Selection

When `strategy=ADAPTIVE`, system auto-selects:
- **Meta-questions** → HOFSTADTER
- **"Why/How" questions** → DECOMPOSE
- **Creative queries** → EXPLORE
- **Review requests** → CRITIQUE
- **Default** → REFINE

---

## ⚡ Performance Tips

1. **Use DIRECT for simple queries** (skip refinement)
2. **Set realistic max_iterations** (3 is usually enough)
3. **Tune quality_threshold** (0.85 is a good default)
4. **Monitor token usage** (refinement uses ~3x tokens)
5. **Cache results** (repeated queries are fast)

---

## 🐛 Troubleshooting

### Issue: Refinement never triggers

**Cause:** Initial quality always > threshold
**Fix:** Lower `quality_threshold` or check confidence scoring

### Issue: Too many iterations

**Cause:** Quality threshold too high or min_improvement too low
**Fix:** Adjust `quality_threshold` or `max_iterations`

### Issue: Wrong strategy selected

**Cause:** Adaptive heuristics don't match query
**Fix:** Use explicit `weave_with_strategy()` instead

---

**That's it! You're ready to use recursive reasoning in HoloLoom.**

**Next:** Explore the full documentation for advanced usage and integration details.
