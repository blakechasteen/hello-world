# Mathematical Operation Selection Architecture

**The Meta-Layer That Chooses Which Math To Expose**

## Overview

HoloLoom has **32 mathematical modules** with hundreds of operations spanning:
- Analysis (real, complex, functional, stochastic)
- Algebra (abstract, module theory, homological)
- Geometry (differential, Riemannian, hyperbolic)
- Probability (measure theory, information theory)
- Decision Theory (game theory, operations research)
- Logic (mathematical logic, computability)
- Extensions (advanced combinatorics, curvature, multivariable calculus)

**The Problem**: Not all queries need Ricci flow or Galois theory.

**The Solution**: `MathOperationSelector` - an intelligent meta-layer that analyzes query intent and selects the **minimal necessary mathematical machinery**.

## Architecture

```
User Query
    ↓
[Intent Classification]
    ↓
[Operation Selection]
    ↓
[Dependency Resolution]
    ↓
[Cost Estimation]
    ↓
Operation Plan → Execution
```

## Complete Flow: Query → Math → Meaning

### Layer 0: Intent Classification

**MathOperationSelector.classify_intent()**
- Analyzes query text and context
- Detects primary intent via keyword matching
- Returns ranked list of intents

**Intent Types**:
- `SIMILARITY` - Find similar items
- `OPTIMIZATION` - Improve/optimize something
- `ANALYSIS` - Analyze/understand structure
- `GENERATION` - Generate new content
- `DECISION` - Make a choice
- `VERIFICATION` - Verify properties
- `TRANSFORMATION` - Transform representation

**Example**:
```python
query = "Find documents similar to this query"
intents = selector.classify_intent(query)
# Result: [QueryIntent.SIMILARITY]
```

### Layer 1: Operation Selection

**MathOperationSelector.plan_operations()**
- Selects operations applicable to detected intents
- Respects prerequisites (topological sort)
- Applies cost budget constraints
- Generates justifications for each selection

**Operation Catalog** (19 core operations):

**BASIC** (O(n) - cheap):
- `inner_product` - Similarity via dot products
- `norm` - Vector magnitudes
- `metric_distance` - Valid distance functions
- `entropy` - Information content

**MODERATE** (O(n²) - reasonable):
- `gradient` - Optimization direction
- `svd` - Dimensionality reduction
- `gram_schmidt` - Orthogonalization
- `hyperbolic_distance` - Hierarchical similarities
- `thompson_sampling` - Exploration/exploitation
- `kl_divergence` - Distribution comparison
- `metric_verification` - Verify metric axioms
- `continuity_check` - Lipschitz continuity
- `convergence_analysis` - Sequence limits

**ADVANCED** (O(n³) - expensive):
- `eigenvalues` - Spectral properties
- `laplacian` - Graph topology
- `fourier_transform` - Frequency analysis
- `geodesic` - Shortest paths on manifolds

**EXPENSIVE** (O(n⁴+) - very expensive):
- `ricci_flow` - Manifold smoothing
- `spectral_clustering` - Graph partitioning

### Layer 2: Dependency Resolution

Operations have **prerequisites** that must execute first:
- `kl_divergence` requires `entropy`
- `gram_schmidt` requires `inner_product` and `norm`
- `spectral_clustering` requires `laplacian` and `eigenvalues`

**Topological sort** ensures correct execution order.

### Layer 3: Cost Estimation

Each operation has an `estimated_cost` (1-100):
- `inner_product`: 1 (cheapest)
- `gradient`: 5
- `eigenvalues`: 15
- `ricci_flow`: 50 (most expensive)

**Budget constraint**: Only execute operations with `total_cost <= budget`

### Layer 4: Execution Plan

**OperationPlan** contains:
- Ordered list of operations
- Justification for each operation
- Total computational cost
- Mathematical domains used
- Execution metadata

**Example Plan**:
```
Query: "Find documents similar to this query"

Mathematical Operation Plan:
  Operations: 4
  Domains: analysis, geometry, probability, algebra
  Est. Cost: 10

Execution Order:
  1. inner_product (algebra)
     -> Applies to similarity intent
  2. metric_distance (analysis)
     -> Applies to similarity intent
  3. hyperbolic_distance (geometry)
     -> Applies to similarity intent
  4. kl_divergence (probability)
     -> Applies to similarity intent
```

## Integration with Weaving Architecture

The MathOperationSelector integrates at **Stage 4: Warp Space** of the weaving cycle:

```
1. LoomCommand → Pattern selection (BARE/FAST/FUSED)
2. ChronoTrigger → Temporal window
3. ResonanceShed → Feature extraction (text → embeddings)
4. WarpSpace → MATH SELECTION HAPPENS HERE
   4.1: MathOperationSelector.plan_operations()
   4.2: Execute selected operations in order
   4.3: Collect results for convergence
5. ConvergenceEngine → Decision collapse
6. Spacetime → Response with trace
```

### AnalyticalWeavingOrchestrator Integration

The `AnalyticalWeavingOrchestrator` uses **hardcoded** operation sequences:
1. Metric verification
2. Hilbert orthogonalization
3. Gradient optimization
4. Spectral stability analysis
5. Continuity verification

**With MathOperationSelector**, this becomes **dynamic**:

```python
from HoloLoom.warp.math.operation_selector import MathOperationSelector

class SmartAnalyticalOrchestrator(AnalyticalWeavingOrchestrator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.math_selector = MathOperationSelector()

    async def weave(self, query, **kwargs):
        # Plan mathematical operations based on query
        plan = self.math_selector.plan_operations(
            query_text=query,
            context={"has_embeddings": True},
            budget=50,  # Cost limit
            enable_expensive=False
        )

        # Execute planned operations in order
        for op in plan.operations:
            result = await self._execute_operation(op, query)
            # Store result in trace

        # Continue with standard weaving
        return await super().weave(query, **kwargs)
```

## Example Use Cases

### Use Case 1: Similarity Search
**Query**: "Find documents similar to this query"

**Selected Operations**:
1. `inner_product` - Compute dot products
2. `metric_distance` - Verify distances are valid
3. `hyperbolic_distance` - Handle hierarchical structure
4. `kl_divergence` - Compare distributions

**Justification**: Minimal operations for valid similarity ranking.

### Use Case 2: Optimization
**Query**: "Optimize the retrieval quality"

**Selected Operations**:
1. `gradient` - Compute optimization direction
2. `gram_schmidt` - Orthogonalize for diversity
3. `geodesic` - Find optimal paths
4. `thompson_sampling` - Balance exploration/exploitation

**Justification**: Gradient-based optimization with diversity and exploration.

### Use Case 3: Analysis
**Query**: "Analyze the convergence of the learning process"

**Selected Operations**:
1. `norm` - Measure magnitudes
2. `gradient` - Analyze direction of change
3. `fourier_transform` - Frequency analysis for oscillations
4. `convergence_analysis` - Sequence limits
5. `metric_verification` - Ensure valid metric space

**Justification**: Comprehensive analysis with verification.

### Use Case 4: Verification
**Query**: "Verify that the metric space is valid"

**Selected Operations**:
1. `metric_distance` - Compute distances
2. `metric_verification` - Check triangle inequality, symmetry
3. `continuity_check` - Verify Lipschitz continuity
4. `convergence_analysis` - Check Cauchy completeness

**Justification**: Full metric space axiom verification.

## Configuration

### Budget Control

Limit computational cost:
```python
plan = selector.plan_operations(
    query_text=query,
    budget=20,  # Only operations with total cost <= 20
)
```

### Enable Expensive Operations

Allow Ricci flow, spectral clustering, etc.:
```python
plan = selector.plan_operations(
    query_text=query,
    enable_expensive=True  # Allow O(n⁴+) operations
)
```

### Context Hints

Provide explicit hints:
```python
plan = selector.plan_operations(
    query_text=query,
    context={
        "has_embeddings": True,
        "requires_optimization": True,
        "needs_verification": False
    }
)
```

## Statistics and Analysis

Track selector usage:
```python
stats = selector.get_statistics()

print(f"Total plans: {stats['total_plans']}")
print(f"Avg operations/plan: {stats['avg_operations_per_plan']}")
print(f"Avg cost/plan: {stats['avg_cost_per_plan']}")
print(f"Most common operations: {stats['most_common_operations']}")
print(f"Intent distribution: {stats['intent_distribution']}")
```

**Example Output**:
```
Total plans: 100
Avg operations/plan: 5.8
Avg cost/plan: 36.8

Most common operations:
  inner_product: 42
  metric_distance: 38
  gradient: 35
  norm: 30
  thompson_sampling: 28

Intent distribution:
  similarity: 35
  optimization: 25
  analysis: 20
  decision: 12
  verification: 8
```

## Extending the Catalog

Add new operations to the catalog:

```python
# In _build_operation_catalog()

ops["your_operation"] = MathOperation(
    name="your_operation",
    domain=MathDomain.GEOMETRY,
    level=OperationLevel.MODERATE,
    description="What your operation does",
    use_cases=[QueryIntent.ANALYSIS, QueryIntent.TRANSFORMATION],
    prerequisites=["metric_distance", "eigenvalues"],
    module_path="HoloLoom.warp.math.your_module",
    function_name="YourClass.your_method",
    estimated_cost=15
)
```

## Mathematical Guarantees

The selector ensures:

1. **Correctness**: Prerequisites always execute before dependents
2. **Efficiency**: Only necessary operations are selected
3. **Budget Compliance**: Total cost never exceeds budget
4. **Justification**: Every operation has explicit rationale
5. **Traceability**: Complete execution plan recorded

## Philosophy

**"Not all queries need Ricci flow"**

The MathOperationSelector embodies the principle of **minimal necessary machinery**:
- Simple similarity search → Basic inner products
- Optimization → Gradients + Thompson Sampling
- Deep analysis → Full spectral + verification suite

This creates:
- **Faster execution** (only run what's needed)
- **Lower cost** (budget-aware planning)
- **Clear reasoning** (justified selections)
- **Extensibility** (easy to add new operations)

## Integration Points

### 1. WeavingOrchestrator (Stage 4)
Select operations for Warp Space computation.

### 2. AnalyticalWeavingOrchestrator
Replace hardcoded operation sequences with dynamic planning.

### 3. Synthesis Layer
Choose math for pattern extraction (entropy, KL divergence, etc.).

### 4. Convergence Engine
Select decision-making math (Thompson Sampling, game theory).

### 5. SpinningWheel
Pick preprocessing math (Fourier transform, SVD).

## Summary

**The Complete Math → Meaning Pipeline**:

```
Query Text
    ↓
[Intent Classification] ← MathOperationSelector
    ↓
[Operation Selection] ← MathOperationSelector
    ↓
[32 Math Modules] ← Execution
    ↓
[Numerical Results]
    ↓
[Convergence Engine] ← Thompson Sampling
    ↓
[Tool Selection]
    ↓
[Response Generation] ← Synthesis Layer
    ↓
Natural Language Response
```

**The selector is the meta-layer that chooses which math to expose**, ensuring that every query uses the **minimal necessary mathematical machinery** to produce **provably correct** results with **full computational provenance**.

---

**Files**:
- Implementation: `HoloLoom/warp/math/operation_selector.py`
- Integration: `HoloLoom/analytical_orchestrator.py`
- Foundation: `HoloLoom/warp/math/` (32 modules)
- Usage: See demo in `operation_selector.py`
