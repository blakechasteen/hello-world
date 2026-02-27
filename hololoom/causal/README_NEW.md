# HoloLoom Causal Reasoning Engine

**Status**: ✅ Production Ready (December 2025)
**Location**: `hololoom/causal/`
**Total Lines**: 3,150 lines of production code across 8 modules
**Date Implemented**: November 2025

## Overview

The Causal Reasoning Engine implements **Pearl's causal hierarchy** for principled causal inference. Unlike traditional machine learning which answers "What is the correlation?" or "What did we observe?", this system answers the deeper causal questions:

- **Level 1 (Association)**: "What is P(Y|X)?" - Observational correlation
- **Level 2 (Intervention)**: "What is P(Y|do(X=x))?" - Causal effect of intervening
- **Level 3 (Counterfactual)**: "What would Y be if X had been x?" - Twin networks for retrospective reasoning

**Key Innovation**: HoloLoom's causal engine integrates symbolic causal structure (human-interpretable DAGs) with neural mechanisms (data-driven learning), enabling both principled reasoning and adaptive learning from data.

## Core Architecture

The system operates across **Pearl's Three-Level Causal Hierarchy**:

```
Level 1: Association (Observational)
├─ P(Y|X) - Conditional probability
├─ Correlation - Covariance-based measures
└─ Association queries - General observational queries

Level 2: Intervention (Do-Calculus)
├─ P(Y|do(X=x)) - Causal effect of intervening
├─ Graph surgery - Remove incoming edges
├─ Backdoor adjustment - Condition on confounders
├─ Frontdoor adjustment - Use mediators
└─ ATE/CATE - Average treatment effects

Level 3: Counterfactual (Twin Networks)
├─ P(Y_x|X',Y') - Twin network inference
├─ 3-step process: Abduction → Action → Prediction
├─ Probability of Necessity (PN) - Was X necessary?
├─ Probability of Sufficiency (PS) - Was X sufficient?
└─ Counterfactual explanation - "What if?" reasoning
```

## Quick Start

```python
from hololoom.causal import CausalDAG, CausalNode, CausalEdge, NodeType
from hololoom.causal import InterventionEngine, CounterfactualEngine, CausalQuery, QueryType

# Build causal model
dag = CausalDAG()
dag.add_node(CausalNode("age", NodeType.OBSERVABLE))
dag.add_node(CausalNode("treatment", NodeType.OBSERVABLE))
dag.add_node(CausalNode("recovery", NodeType.OBSERVABLE))

# Define relationships
dag.add_edge(CausalEdge("age", "recovery", strength=0.4))
dag.add_edge(CausalEdge("treatment", "recovery", strength=0.6))

# Query causal effect
engine = InterventionEngine(dag)
query = CausalQuery(
    query_type=QueryType.INTERVENTION,
    outcome="recovery",
    treatment="treatment"
)
answer = engine.query(query)
print(f"Causal effect: {answer.result:.3f}")

# Counterfactual reasoning
cf_engine = CounterfactualEngine(dag)
result = cf_engine.counterfactual(
    intervention={"treatment": 0},
    evidence={"treatment": 1, "recovery": 1},
    query="recovery"
)
print(f"Would recovery occur without treatment? {result.counterfactual_outcome}")

# Probability of necessity
necessity = cf_engine.probability_of_necessity(
    treatment="treatment",
    outcome="recovery",
    evidence={"treatment": 1, "recovery": 1}
)
print(f"Was treatment necessary? {necessity:.3f}")
```

## Key Components

| Component | Lines | Purpose |
|-----------|-------|---------|
| **dag.py** | 454 | DAG structures, d-separation, adjustment criteria |
| **query.py** | 265 | Causal query definitions, natural language conversion |
| **intervention.py** | 471 | do-operator, effect identification, adjustment |
| **counterfactual.py** | 532 | Twin networks, necessity/sufficiency |
| **neural_scm.py** | 415 | Symbolic + neural causal models |
| **discovery.py** | 526 | PC algorithm, active learning |
| **temporal.py** | 447 | Time-lagged causality, trajectories |
| **__init__.py** | 40 | Public API exports |
| **Total** | **3,150** | Complete Pearl-style inference system |

## Main Classes

### CausalDAG (dag.py)

Directed acyclic graph with graphical causal criteria:

```python
dag = CausalDAG()
dag.add_node(CausalNode("X"))
dag.add_edge(CausalEdge("X", "Y", strength=0.8))

# Query relationships
parents = dag.parents("Y")
ancestors = dag.ancestors("Y")
confounders = dag.find_confounders("X", "Y")
mediators = dag.find_mediators("X", "Y")

# Graphical criteria
backdoor_ok = dag.satisfies_backdoor_criterion("X", "Y", {"Z"})
frontdoor_ok = dag.satisfies_frontdoor_criterion("X", "Y", {"M"})

# d-separation
is_separated = dag.is_d_separated({"X"}, {"Y"}, {"Z"})
```

**Features**: Cycle detection, d-separation, Markov blankets, path finding, collider/confounder/mediator detection.

### CausalQuery & CausalAnswer (query.py)

Structured queries across Pearl's three levels:

```python
# Level 1: Observational
q1 = CausalQuery(QueryType.CONDITIONAL, outcome="Y", treatment="X")

# Level 2: Interventional
q2 = CausalQuery(QueryType.ATE, outcome="Y", treatment="X",
                 treatment_value=1, control_value=0)

# Level 3: Counterfactual
q3 = CausalQuery(QueryType.NECESSITY, outcome="Y", treatment="X",
                 evidence={"X": 1, "Y": 1})

# Natural language
print(q2.to_natural_language())
answer = engine.query(q2)
print(answer.to_natural_language())
```

**Query Types**: CONDITIONAL, CORRELATION, ASSOCIATION, INTERVENTION, ATE, CATE, DIRECT_EFFECT, TOTAL_EFFECT, COUNTERFACTUAL, ETT, NECESSITY, SUFFICIENCY

### InterventionEngine (intervention.py)

Implements do() operator and causal identification:

```python
engine = InterventionEngine(dag)

# Apply intervention (graph surgery)
result = engine.do({"X": 1})

# Identify causal effect
identification = engine.identify_causal_effect("X", "Y")

# Query with data
answer = engine.query(query, data=observational_data)

# Compute ATE
ate = engine.compute_ate("X", "Y", 1, 0, data)

# Path analysis
paths = engine.find_all_paths("X", "Y")
```

**Strategies**: Backdoor adjustment, Frontdoor adjustment, Do-calculus (future).

### CounterfactualEngine (counterfactual.py)

Pearl's twin network method for "what if" reasoning:

```python
engine = CounterfactualEngine(dag)

# Counterfactual inference
result = engine.counterfactual(
    intervention={"X": 0},
    evidence={"X": 1, "Y": 1},
    query="Y"
)

# Necessity: Was X necessary for Y?
necessity = engine.probability_of_necessity("X", "Y",
    evidence={"X": 1, "Y": 1})

# Sufficiency: Is X sufficient for Y?
sufficiency = engine.probability_of_sufficiency("X", "Y",
    evidence={"X": 0, "Y": 0})

# Both necessary and sufficient
nands = engine.probability_of_necessity_and_sufficiency("X", "Y")
```

**Process**:
1. Abduction: Infer latent U from evidence
2. Action: Apply intervention in counterfactual world
3. Prediction: Compute outcome with U and intervention

### NeuralStructuralCausalModel (neural_scm.py)

Hybrid symbolic + neural causal models:

```python
nscm = NeuralStructuralCausalModel(dag)

# Learn mechanisms from data
nscm.fit(data, variable_names, hidden_dim=32, epochs=100)

# Sample from model
samples = nscm.sample(n_samples=1000)

# Interventions
outcomes = nscm.intervene({"X": 1}, n_samples=1000)

# Treatment effects
ate = nscm.estimate_ate("X", "Y", 1, 0, n_samples=1000)

# Counterfactual
cf = nscm.counterfactual({"X": 0}, {"X": 1, "Y": 1}, "Y")
```

**Benefits**: Domain knowledge + data-driven learning, no hand-coded equations, learns complex relationships.

### CausalDiscovery (discovery.py)

Learn causal structure from data:

```python
discoverer = CausalDiscovery(variables=["X", "Y", "Z"])

# PC Algorithm
discoverer.fit_observational(data, variable_names)
learned_dag = discoverer.get_dag()

# Active learning
while not discoverer.is_converged():
    intervention = discoverer.select_intervention()
    result = run_experiment(intervention)
    discoverer.update(intervention, result)
```

**Algorithms**: PC algorithm (constraint-based), Active learning (information gain).

### TemporalCausalDAG (temporal.py)

Causality over time with lag-specific edges:

```python
tcdag = TemporalCausalDAG(variables=["X", "Y"], max_lag=30)
tcdag.add_temporal_edge(TemporalEdge("X", "Y", lag=5, strength=0.6))

# Predict trajectory
trajectory = tcdag.predict_trajectory(
    initial_state={"X": 1, "Y": 0},
    steps=10
)

# Compare factual vs counterfactual
factual, cf = tcdag.intervene_trajectory(
    intervention_time=0,
    intervention={"X": 1},
    initial_state={"X": 0, "Y": 0},
    total_steps=10
)

# Find optimal intervention timing
best_time, effect = tcdag.find_optimal_intervention_timing(
    treatment="X", outcome="Y", treatment_value=1,
    initial_state={"X": 0, "Y": 0}, max_time=30
)

# Granger causality
causes, strength = tcdag.granger_causality("X", "Y")

# Convert to static DAG
static_dag = tcdag.to_static_dag()
```

**Features**: Time-lagged edges, trajectory prediction, optimal timing, Granger causality, temporal ATE.

## Performance Characteristics

| Operation | Time | Complexity |
|-----------|------|-----------|
| DAG operations | <1ms | O(n) or O(n+e) |
| d-separation | <5ms | O(n+e) |
| Cycle check | <1ms | O(n) DFS |
| Backdoor criterion | <10ms | O(paths) |
| PC algorithm | 100-500ms | O(n³) |
| Neural training | 100-1000ms | O(epochs×n×d) |
| Counterfactual | <5ms | O(1) |
| Trajectory (10 steps) | <50ms | O(n×e) |

## Integration Examples

### With HoloLoom Memory

```python
from hololoom import hololoom
from hololoom.causal import CausalDAG, CausalNode, CausalEdge

async with HoloLoom() as loom:
    memories = await loom.recall("causal relationships")

    dag = CausalDAG()
    for mem in memories:
        for entity in mem.entities:
            dag.add_node(CausalNode(entity.name))
        for rel in mem.relations:
            if rel.type == "CAUSES":
                dag.add_edge(CausalEdge(rel.source, rel.target))
```

### With Agentic Reasoning

```python
from hololoom.agentic import create_agentic_orchestrator
from hololoom.causal import InterventionEngine

async with create_agentic_orchestrator(config, shards) as orch:
    dag = await orch.get_system_causal_model()
    engine = InterventionEngine(dag)

    paths = engine.find_all_paths("action", "goal")
    for path in paths:
        print(f"Path: {' → '.join(path)}")
```

## When to Use

### ✅ Use When You Need To:

- Infer causation (not just correlation)
- Predict intervention effects ("what if?")
- Handle confounding
- Reason counterfactually
- Find causal pathways
- Optimize decisions
- Explain predictions
- Discover structure from data
- Model temporal dynamics

### 🟡 Use With Caution:

- Small sample sizes (<50 obs/edge)
- Many hidden confounders
- Strong cycles
- Non-linear relationships
- Real-time requirements

### ❌ Don't Use When:

- Prediction only (correlation sufficient)
- Explanations unnecessary
- No domain knowledge
- No causal patterns
- Interventions infeasible

## Example: Medical Treatment

```python
# Question: Does treatment cause recovery? Account for confounding age.

dag = CausalDAG()
dag.add_node(CausalNode("age"))
dag.add_node(CausalNode("treatment"))
dag.add_node(CausalNode("recovery"))

dag.add_edge(CausalEdge("age", "treatment"))      # Age determines treatment
dag.add_edge(CausalEdge("age", "recovery"))       # Age affects recovery
dag.add_edge(CausalEdge("treatment", "recovery")) # Treatment affects recovery

# Identify causal effect
engine = InterventionEngine(dag)
identification = engine.identify_causal_effect("treatment", "recovery")

print(f"Method: {identification.identification_method}")
# Output: "backdoor adjustment"
print(f"Adjust for: {identification.adjustment_set}")
# Output: "{age}"

# Estimate effect from data
query = CausalQuery(QueryType.ATE, outcome="recovery", treatment="treatment")
answer = engine.query(query, data=observational_data)

print(f"Treatment effect: {answer.result:.3f} ± {answer.confidence:.2f}")

# Counterfactual reasoning
cf_engine = CounterfactualEngine(dag)
result = cf_engine.counterfactual(
    intervention={"treatment": 0},
    evidence={"treatment": 1, "recovery": 1, "age": 65},
    query="recovery"
)

necessity = cf_engine.probability_of_necessity(
    "treatment", "recovery",
    evidence={"treatment": 1, "recovery": 1, "age": 65}
)

print(f"Was treatment necessary? {necessity:.1%}")
```

## References

**Foundational**:
- Pearl, J. (2000). *Causality: Models, Reasoning, and Inference*
- Pearl, J. (2009). "Causal inference in statistics: An overview"
- Spirtes, P., Glymour, C., & Scheines, R. (2000). *Causation, Prediction, and Search*

**Specific Topics**:
- d-separation: Pearl (2000), Ch. 1-2
- Backdoor/Frontdoor: Pearl (2000), Ch. 3
- Counterfactuals: Pearl (2000), Ch. 7
- Granger Causality: Granger (1969)
- PC Algorithm: Meek (1995), Colombo & Maathuis (2014)

## Future Enhancements

- [ ] Full do-calculus (3 rules)
- [ ] Causal forests (heterogeneous effects)
- [ ] Instrumental variables
- [ ] MCMC counterfactuals
- [ ] Graph neural networks
- [ ] Sensitivity analysis
- [ ] Time-varying treatments
- [ ] Dynamic regimes
- [ ] Feedback loops

---

**Created**: December 2025
**Status**: ✅ Production Ready
**Maintainer**: HoloLoom Development Team
