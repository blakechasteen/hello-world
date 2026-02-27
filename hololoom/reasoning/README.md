# HoloLoom Reasoning Engines

**Status**: ✅ Production Ready (December 2025)
**Location**: `hololoom/reasoning/`
**Total Code**: ~84,671 lines across 5 core modules + 1 department integration
**Updated**: 2025-12-11

## Overview

The Reasoning module is Layer 3 of HoloLoom's Cognitive Architecture - implementing **three independent reasoning engines** (Deductive, Abductive, Analogical) with Layer 2-3 integration for reasoning-enhanced planning.

These engines transform raw knowledge into actionable conclusions through logical inference, best-explanation discovery, and knowledge transfer across domains. The system enables HoloLoom to not just retrieve information, but to **think about it** - deriving new facts, explaining observations, and solving novel problems by analogy.

**Core Philosophy**: "Reason beyond the obvious. Connect the dots that others miss."

Unlike simple retrieval systems, HoloLoom's reasoning is **symbolic yet grounded**, **logical yet probabilistic**, and **domain-independent yet context-aware**. Each engine can operate independently or be composed for complex multi-step reasoning.

## Quick Start

### Deductive Reasoning (Logical Inference)

```python
from hololoom.reasoning import (
    DeductiveReasoner, KnowledgeBase,
    create_fact, create_rule
)

# Create knowledge base
kb = KnowledgeBase()

# Add facts (base knowledge)
kb.add_fact(create_fact("human", "Socrates"))
kb.add_fact(create_fact("human", "Plato"))

# Add rules (inference rules)
kb.add_rule(create_rule(
    premises=[create_fact("human", "?x")],
    conclusion=create_fact("mortal", "?x"),
    name="all_humans_mortal"
))

# Create reasoner
reasoner = DeductiveReasoner(kb)

# Forward chaining: derive all consequences
all_facts = reasoner.forward_chain()
# Result: Adds mortal(Socrates), mortal(Plato), ...

# Backward chaining: prove goal
goal = create_fact("mortal", "Socrates")
proof = reasoner.backward_chain(goal)
if proof:
    print(proof.to_string())  # Print proof explanation
```

### Abductive Reasoning (Best Explanation)

```python
from hololoom.reasoning import (
    AbductiveReasoner,
    create_causal_rule, create_observation
)

# Define causal rules (cause → effect)
causal_rules = [
    create_causal_rule("flu", True, "fever", True, strength=0.9),
    create_causal_rule("malaria", True, "fever", True, strength=0.95),
    create_causal_rule("allergy", True, "fever", False, strength=0.1),
]

# Create reasoner
reasoner = AbductiveReasoner(causal_rules)

# Observe symptoms
observations = [
    create_observation("fever", True, confidence=0.95),
    create_observation("chills", True, confidence=0.8),
]

# Find best explanation
explanations = reasoner.explain(observations, max_hypotheses=3)
best = explanations[0]

print(f"Best explanation: {best.explanation}")
print(f"Likelihood: {best.likelihood:.2f}")
print(f"Score: {best.score():.3f}")
```

### Analogical Reasoning (Knowledge Transfer)

```python
from hololoom.reasoning import (
    AnalogicalReasoner,
    create_domain, create_entity, create_relation
)

# Create source domain (familiar: Solar System)
solar_system = create_domain("solar_system")
sun = create_entity("sun", mass="large", temp="hot")
earth = create_entity("earth", mass="small", temp="moderate")

solar_system.add_entity(sun)
solar_system.add_entity(earth)
solar_system.add_relation(
    create_relation("orbits", earth, sun)
)

# Create target domain (new: Atom)
atom = create_domain("atom")
nucleus = create_entity("nucleus", mass="large", charge="positive")
electron = create_entity("electron", mass="small", charge="negative")

atom.add_entity(nucleus)
atom.add_entity(electron)
atom.add_relation(
    create_relation("orbits", electron, nucleus)
)

# Create analogical reasoner
reasoner = AnalogicalReasoner()

# Find mapping: solar_system → atom
mapping = reasoner.find_analogy(solar_system, atom)
if mapping:
    print(f"Analogy score: {mapping.score:.3f}")
    print(f"Entity mappings: {mapping.entity_mappings}")

# Transfer knowledge via mapping
transferred_domain = reasoner.transfer_knowledge(solar_system, mapping)
```

## Key Components

| Component | Lines | Purpose |
|-----------|-------|---------|
| `deductive.py` | ~624 | Logical inference (forward/backward chaining) |
| `abductive.py` | ~673 | Best explanation (hypothesis generation + scoring) |
| `analogical.py` | ~761 | Knowledge transfer (structure mapping + case-based reasoning) |
| `integration.py` | ~495 | Layer 2-3 integration (reasoning-enhanced planning) |
| `departments/reasoning.py` | ~1174 | Department API (multi-hop queries, causal inference, counterfactuals) |
| `__init__.py` | ~83 | Package exports and API |

**Total**: ~3,810 lines of production reasoning code

## Main Classes & Functions

### Deductive Reasoning Module

**Fact** (frozen dataclass)
- Atomic facts in knowledge base
- Example: `Fact("human", ("Socrates",))`
- Methods: `variables()`, `ground()`, `substitute(bindings)`

**Rule** (dataclass)
- Logical rules: premises → conclusion
- Example: `Rule(premises=[Fact("human", ("?x",))], conclusion=Fact("mortal", ("?x",)))`
- Supports confidence values (0-1) for uncertain rules

**Proof** (dataclass)
- Complete proof chain showing derivation path
- Methods: `to_string()` for human-readable explanation
- Tracks rules applied and premises used

**Unifier** (static class)
- Pattern matching algorithm for facts
- Finds variable bindings that unify two facts
- Method: `unify(fact1, fact2, bindings) → Dict[str, Any]`

**KnowledgeBase**
- Repository of facts and rules
- Efficient indexing: predicate → facts, conclusion → rules
- Methods:
  - `add_fact(fact)`, `add_rule(rule)`
  - `query(fact)` - exact match for ground facts
  - `query_with_unification(fact)` - pattern matching with variables
  - `get_rules_for(predicate)` - rules concluding predicate

**DeductiveReasoner**
- Main reasoning engine
- Methods:
  - `forward_chain(max_iterations) → Set[Fact]` - data-driven inference
  - `backward_chain(goal, max_depth) → Optional[Proof]` - goal-driven proof
  - `explain(fact) → Optional[Proof]` - generate proof explanation
  - `explain_to_string(fact) → str` - human-readable proof

### Abductive Reasoning Module

**Hypothesis** (dataclass)
- Candidate explanation for observations
- Attributes: `explanation` (variable assignments), `likelihood`, `prior`, `complexity`
- Method: `score() → float` = (likelihood × prior) / (1 + complexity)

**Observation** (dataclass)
- Single observed evidence
- Attributes: `variable`, `value`, `confidence` (0-1), `timestamp`

**CausalRule** (dataclass)
- Causal knowledge: cause → effect
- Example: `CausalRule("flu", True, "fever", True, strength=0.9)`
- Represents P(effect | cause)

**HypothesisGenerator**
- Generates candidate explanations from observations
- Works backward from observations to possible causes
- Methods:
  - `generate(observations, max_hypotheses, allow_multi_cause) → List[Dict]`
  - Supports single-cause and multi-cause hypotheses

**HypothesisScorer**
- Scores hypotheses using Bayesian inference
- Computes: likelihood × prior / complexity (Occam's razor)
- Methods:
  - `score(hypothesis_dict, observations) → Hypothesis`
  - Weighted by observation confidence

**AbductiveReasoner**
- Main abductive reasoning engine
- Methods:
  - `explain(observations, max_hypotheses) → List[Hypothesis]` - ranked explanations
  - `explain_single(observation) → List[Hypothesis]` - single observation
  - `best_explanation(observations) → Optional[Hypothesis]` - single best
  - `compare_hypotheses(h1, h2, observations) → Tuple[Hypothesis, Hypothesis]`
  - `explain_with_confidence(observations, threshold) → Optional[Hypothesis]` - confidence gating

### Analogical Reasoning Module

**Entity** (dataclass)
- Domain entity (object, concept, component)
- Attributes: `name`, `properties` (dict), `entity_type`
- Hashable for set operations

**Relation** (dataclass)
- Relationship between entities
- Attributes: `relation_type`, `entities` (tuple), `properties`
- Example: `Relation("orbits", (electron, nucleus))`

**Domain** (dataclass)
- Structured domain representation
- Attributes: `name`, `entities` (set), `relations` (set), `facts` (dict)
- Methods: `add_entity()`, `add_relation()`, `get_entity()`, `get_relations_for()`

**AnalogicalMapping** (dataclass)
- Correspondence between source and target domains
- Attributes: `source_domain`, `target_domain`, `entity_mappings`, `relation_mappings`, `score`, `justification`
- Methods: `map_entity()`, `map_relation()`, `reverse()`

**StructureMapper**
- Finds structural correspondences between domains (Gentner's Structure-Mapping Engine)
- Methods:
  - `find_mapping(source, target, max_mappings) → List[AnalogicalMapping]` - ranked by quality
  - Supports custom similarity functions
  - Returns mappings sorted by structural consistency and semantic similarity

**KnowledgeTransferer**
- Transfers knowledge from source to target via mapping
- Methods:
  - `transfer_fact(source_fact, mapping) → Optional[Dict]` - single fact
  - `transfer_relation(source_relation, mapping) → Optional[Relation]` - single relation
  - `transfer_all(source_domain, mapping) → Domain` - entire domain

**Case** (dataclass)
- Past problem-solution pair for case-based reasoning
- Attributes: `problem` (Domain), `solution`, `outcome`, `context`

**CaseLibrary**
- Library of past cases for retrieval and adaptation
- Methods:
  - `add_case(case)` - store past cases
  - `find_similar(problem, mapper, max_cases) → List[Tuple[Case, AnalogicalMapping]]` - retrieve similar cases

**AnalogicalReasoner**
- Main analogical reasoning engine combining structure mapping + transfer + CBR
- Methods:
  - `find_analogy(source, target) → Optional[AnalogicalMapping]` - best mapping
  - `transfer_knowledge(source, mapping) → Domain` - knowledge transfer
  - `solve_by_analogy(problem) → Optional[Dict]` - case-based problem solving
  - `add_case(case)` - add past case to library

### Integration Module

**PlanExplanation** (dataclass)
- Natural language explanation of plan
- Attributes: `plan`, `causal_chain`, `key_actions`, `success_conditions`, `reasoning_trace`
- Method: `to_string() → str` - human-readable explanation

**FailureDiagnosis** (dataclass)
- Diagnosis of action failure
- Attributes: `failed_action`, `expected_state`, `actual_state`, `likely_causes`, `recommendations`
- Method: `to_string() → str` - structured diagnosis

**ReasoningEnhancedPlanner**
- Planning system with integrated reasoning (Layer 2-3 integration)
- Combines HTN planning with deductive, abductive, analogical reasoning
- Methods:
  - `find_preconditions(action_name) → List[Fact]` - deductive reasoning
  - `check_preconditions(action_name, state) → bool` - verify preconditions
  - `explain_plan(plan, goal) → PlanExplanation` - abductive reasoning
  - `transfer_plan(source_plan, source_domain, target_domain) → Optional[Any]` - analogical reasoning
  - `diagnose_failure(failed_action, expected, actual) → FailureDiagnosis` - failure analysis
  - `plan_with_reasoning(goal, initial_state) → Tuple[Plan, PlanExplanation]` - integrated planning

### Department Integration Module

**Reasoning Department** (`hololoom/departments/reasoning/reasoning.py`)
- Advanced multi-hop reasoning as a department service
- Provides 4 main task types:

  1. **MultiHopReasoner** (lines 99-339)
     - Multi-step graph traversal to answer complex queries
     - BFS with path caching for efficiency
     - Methods:
       - `find_paths(kg, start, end, max_hops) → List[ReasoningChain]` - find paths
       - `find_common_connections(kg, entities, max_hops) → List[ReasoningChain]` - common connections
     - Example: "What connects varroa mites to hive temperature through N hops?"

  2. **CausalInferenceEngine** (lines 345-555)
     - Identify cause-effect relationships from knowledge graphs
     - Supports direct and indirect causality inference
     - Methods:
       - `infer_causality(kg, entity_a, entity_b) → Optional[CausalRelation]` - pairwise causality
       - `find_all_causal_relations(kg, entity, max_depth) → Dict[str, List[CausalRelation]]` - all relations
     - Example: "Is there a causal link between A and B? How strong?"

  3. **CounterfactualAnalyzer** (lines 561-719)
     - "What if" scenario analysis using counterfactual reasoning
     - Traces downstream effects of hypothetical modifications
     - Methods:
       - `analyze_counterfactual(kg, modification, entity, causal_engine) → CounterfactualScenario`
       - `compare_scenarios(scenarios) → Dict[str, Any]` - scenario comparison
     - Example: "What if we eliminated varroa mites? What would improve/degrade?"

  4. **ReasoningDepartment** (lines 725-1174)
     - Main department service orchestrating all reasoning engines
     - Async/await interface for production integration
     - Methods:
       - `execute(request) → DepartmentResponse` - main entry point
       - `verify(response) → VerificationResult` - quality verification
       - `refine(request, prior, verification) → DepartmentResponse` - improvement loop

## Performance Characteristics

| Operation | Latency | Notes |
|-----------|---------|-------|
| **Forward chaining** | 5-50ms | 10-50 facts/rules, scales with KB size |
| **Backward chaining** | 10-100ms | Depth-limited (max_depth=10), cycle prevention |
| **Hypothesis generation** | 2-20ms | Per observation, 20-100 hypotheses |
| **Hypothesis scoring** | 5-50ms | Likelihood + prior + complexity calculation |
| **Structure mapping** | 10-100ms | Entity/relation similarity computation |
| **Knowledge transfer** | <5ms | Map-based substitution |
| **Multi-hop path finding** | 20-200ms | BFS with caching, 5-50 entities |
| **Causal inference** | 5-50ms | Direct + indirect causality checking |
| **Counterfactual analysis** | 10-100ms | Effect tracing via causal engine |

**Typical Pipeline Latencies**:
- Deductive-only: ~20-100ms
- Abductive-only: ~10-70ms
- Analogical-only: ~20-150ms
- Multi-engine composition: <300ms total

## Integration with HoloLoom

### With Memory Systems

Reasoning integrates with HoloLoom's 11 specialized memory systems:

```python
from hololoom import hololoom
from hololoom.reasoning import DeductiveReasoner, KnowledgeBase

async with HoloLoom() as loom:
    # Retrieve facts from memory
    memories = await loom.recall("What happened?", k=20)

    # Convert to facts for reasoning
    facts = [
        create_fact(m.id, m.text) for m in memories
    ]

    # Perform deductive reasoning
    kb = KnowledgeBase()
    for fact in facts:
        kb.add_fact(fact)

    reasoner = DeductiveReasoner(kb)
    derived = reasoner.forward_chain()

    # Store derived facts back to memory
    for fact in derived:
        await loom.experience(str(fact))
```

### With Weaving Orchestrator

Reasoning is automatically invoked in complex weaving cycles:

```python
from hololoom.weaving_orchestrator import WeavingOrchestrator
from hololoom.config import Config

# FULL/RESEARCH modes enable reasoning
config = Config.fused()

async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
    # Complex queries automatically trigger deductive + abductive reasoning
    spacetime = await orchestrator.weave(query)

    # Reasoning trace in metadata
    if 'reasoning_trace' in spacetime.metadata:
        print(spacetime.metadata['reasoning_trace'])
```

### With Agentic System

Reasoning powers multi-query exploration:

```python
from hololoom.agentic import create_agentic_orchestrator, ReasoningMode

agent = await create_agentic_orchestrator(config, shards)

# RESEARCH mode uses all three reasoning engines
result = await agent.reason(
    query="Comprehensive analysis of Thompson Sampling",
    mode=ReasoningMode.RESEARCH,
    max_steps=5
)

# View reasoning steps
for step in result.steps_taken:
    print(f"{step['type']}: {step['reasoning']}")
```

### With Alignment Framework

Reasoning informs safety decisions:

```python
from hololoom.alignment import SafetyGuardrails
from hololoom.reasoning import DeductiveReasoner, KnowledgeBase

guardrails = SafetyGuardrails()

# Deductive reasoning verifies preconditions
kb = KnowledgeBase()
kb.add_rule(create_rule(
    premises=[create_fact("sandboxed", "True")],
    conclusion=create_fact("can_execute_code", "True")
))

reasoner = DeductiveReasoner(kb)
proof = reasoner.backward_chain(create_fact("can_execute_code", "True"))

if proof:
    # Safe to proceed based on logical proof
    decision = guardrails.gate_action("execute_code", context)
```

## When to Use

### ✅ Use Deductive Reasoning When You Need:
- **Logical inference**: Derive conclusions from known facts
- **Proof generation**: Explain why something is true
- **Expert systems**: Encode domain rules and apply them
- **Verification**: Check if claims follow logically from axioms
- **Deterministic reasoning**: When outcomes are certain given premises

**Example Use Cases**:
- Verify medical diagnoses from symptoms
- Check contract compliance from rules
- Trace security violations from policies
- Explain system behavior from logs

### ✅ Use Abductive Reasoning When You Need:
- **Best explanation**: Find most likely cause(s) for observations
- **Diagnosis**: Medical, technical, or debugging
- **Hypothesis generation**: Multiple competing explanations
- **Probabilistic inference**: Combining likelihood + priors
- **Uncertainty quantification**: Scoring explanations by quality

**Example Use Cases**:
- Diagnose equipment failures
- Identify diseases from symptoms
- Root cause analysis of bugs
- Anomaly detection and explanation

### ✅ Use Analogical Reasoning When You Need:
- **Knowledge transfer**: Apply solutions from similar domains
- **Case-based reasoning**: Adapt past solutions to new problems
- **Transfer learning**: Leverage familiar concepts for new domains
- **Creative problem solving**: Find unexpected connections
- **Domain mapping**: Understand structure correspondence

**Example Use Cases**:
- Apply medical treatments from similar conditions
- Reuse architectural patterns across domains
- Transfer learning in machine learning
- Creative analogy-based ideation

### ✅ Use Integration (Planning + Reasoning) When You Need:
- **Intelligent planning**: Plans that explain themselves
- **Failure diagnosis**: Understand why plans failed
- **Plan adaptation**: Transfer plans across domains
- **Precondition reasoning**: Verify action prerequisites
- **Goal decomposition**: Break complex goals into logical steps

**Example Use Cases**:
- Plan execution with explanation and diagnosis
- Adapt plans from case library to new situations
- Verify action prerequisites before execution
- Explain plan failure for recovery

### ✅ Use Department (Multi-Hop + Causal + Counterfactual) When You Need:
- **Complex query answering**: Multi-step reasoning across graphs
- **Causal analysis**: Identify cause-effect relationships
- **Scenario analysis**: Predict outcomes of hypothetical changes
- **Impact assessment**: Understand ripple effects
- **Why/How questions**: Deep explanations

**Example Use Cases**:
- "What connects A to B through intermediate steps?"
- "Does X cause Y? How strong is the link?"
- "What if we eliminated X? What would happen?"
- Multi-domain impact analysis and decision support

### ❌ Don't Use When:
- Data is unstructured (use embeddings instead)
- You need fast similarity (use vectors)
- Real-time response required (<100ms)
- Rules/structure unknown (use learning)
- Graph is extremely large (>1M nodes, use approximate algorithms)

## Research Foundations

**Deductive Reasoning**:
- Russell & Norvig (2020): "AI: A Modern Approach" (Ch. 7-9)
- Kowalski (1974): "Predicate Logic as Programming Language"
- Forgy (1982): RETE algorithm for forward chaining

**Abductive Reasoning**:
- Peirce (1878): "Deduction, Induction, and Hypothesis" (origin of abduction)
- Josephson & Josephson (1996): "Abductive Inference"
- Pearl (2000): Causality (causal explanation)
- Hobbs et al. (1993): "Interpretation as Abduction"

**Analogical Reasoning**:
- Gentner (1983): "Structure-Mapping Theory"
- Hofstadter & Mitchell (1994): "Copycat" program
- Holyoak & Thagard (1989): "Analogical mapping by constraint satisfaction"
- Forbus et al. (2011): "Structure-Mapping Engine (SME)"

## Testing & Validation

Run the test suite:

```bash
# Test deductive reasoning
pytest hololoom/tests/unit/test_deductive.py -v

# Test abductive reasoning
pytest hololoom/tests/unit/test_abductive.py -v

# Test analogical reasoning
pytest hololoom/tests/unit/test_analogical.py -v

# Test integration
pytest hololoom/tests/integration/test_reasoning_integration.py -v

# Test department
pytest hololoom/tests/integration/test_reasoning_department.py -v
```

Expected Results: **100+ tests passing**, covering:
- Knowledge base operations
- Forward/backward chaining
- Hypothesis generation and scoring
- Structure mapping and knowledge transfer
- Case-based reasoning
- Integration with planning
- Department API compliance

## Examples

### Example 1: Diagnosis System

```python
# Medical diagnostic system using abductive reasoning
causal_rules = [
    create_causal_rule("pneumonia", True, "cough", True, 0.95),
    create_causal_rule("pneumonia", True, "fever", True, 0.90),
    create_causal_rule("flu", True, "cough", True, 0.8),
    create_causal_rule("flu", True, "fever", True, 0.85),
    create_causal_rule("allergy", True, "cough", True, 0.7),
]

reasoner = AbductiveReasoner(causal_rules)

# Patient symptoms
observations = [
    create_observation("cough", True, confidence=0.95),
    create_observation("fever", True, confidence=0.90),
    create_observation("fatigue", True, confidence=0.8),
]

# Find most likely diagnosis
diagnoses = reasoner.explain(observations, max_hypotheses=5)

for i, diagnosis in enumerate(diagnoses, 1):
    print(f"{i}. {diagnosis.explanation}")
    print(f"   Score: {diagnosis.score():.3f}")
    print(f"   Confidence: {diagnosis.prior:.2f}")
```

### Example 2: Cross-Domain Knowledge Transfer

```python
# Transfer knowledge from familiar (Solar System) to new (Atom) domain
reasoner = AnalogicalReasoner()

# Create mapping: sun ↔ nucleus, earth ↔ electron
mapping = reasoner.find_analogy(solar_system, atom)

# Transfer gravitational properties
transfer = KnowledgeTransferer()
atom_with_transferred = transfer.transfer_all(solar_system, mapping)

# Apply transferred knowledge
print(f"In atom domain:")
print(f"Nucleus plays role of: {mapping.map_entity(sun).name}")
print(f"Electron orbits via: {mapping.map_relation('orbits')}")
```

### Example 3: Multi-Hop Query

```python
# Query: "What connects varroa mites to colony collapse?"
dept = ReasoningDepartment()

request = DepartmentRequest(
    task_id="multihop_001",
    task_type="multi_hop_query",
    parameters={
        "start_entity": "varroa_mites",
        "end_entity": "colony_collapse",
        "max_hops": 5
    }
)

response = await dept.execute(request)

# Print reasoning chains
for chain in response.result['reasoning_chains']:
    print(f"Path ({chain['total_hops']} hops):")
    for step in chain['steps']:
        print(f"  {step['entity']} -{step['relation']}-> {step['target']}")
```

## Future Enhancements (Phase 6+)

1. **Probabilistic Logic Programming**: Hybrid logic + probability
2. **Non-Monotonic Reasoning**: Default logic, circumscription, autoepistemic
3. **Constraint-Based Reasoning**: CSP solving integrated with logic
4. **Incremental Reasoning**: Update conclusions without full recomputation
5. **Distributed Reasoning**: Parallel reasoning across multiple agents
6. **Explainable AI**: Natural language generation for reasoning chains
7. **Neuro-Symbolic Integration**: Combine neural embeddings with symbolic reasoning

## References

- **Module**: `hololoom/reasoning/`
- **Tests**: `hololoom/tests/unit/test_*.py` and `hololoom/tests/integration/`
- **Demos**: `demos/demo_reasoning_*.py`
- **Integration**: `hololoom/departments/reasoning/reasoning.py`

