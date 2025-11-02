# Layer 3: Reasoning Architecture

**Status:** 🚀 IN PROGRESS
**Scope:** Logical inference, explanation generation, knowledge transfer
**Foundation:** Built on Layer 1 (Causal) and Layer 2 (Planning)

---

## Overview

Layer 3 implements human-like reasoning capabilities:

1. **Deductive Reasoning** - Logical inference from known facts
2. **Abductive Reasoning** - Best explanation for observations
3. **Analogical Reasoning** - Transfer knowledge across domains
4. **Integration** - Use reasoning to enhance planning

These transform the system from a planner into a **thinking agent**.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        LAYER 3: REASONING                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  Deductive Reasoning Engine                             │     │
│  │  - Forward chaining (facts → conclusions)              │     │
│  │  - Backward chaining (goal → supporting facts)         │     │
│  │  - Rule-based inference                                 │     │
│  │  - First-order logic                                    │     │
│  └────────────────────────────────────────────────────────┘     │
│                         ▼                                         │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  Abductive Reasoning Engine                             │     │
│  │  - Hypothesis generation                                │     │
│  │  - Explanation scoring (likelihood × parsimony)        │     │
│  │  - Best explanation selection                           │     │
│  │  - Bayesian inference                                   │     │
│  └────────────────────────────────────────────────────────┘     │
│                         ▼                                         │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  Analogical Reasoning Engine                            │     │
│  │  - Structure mapping (source → target)                 │     │
│  │  - Relation preservation                                │     │
│  │  - Transfer learning                                    │     │
│  │  - Case-based reasoning                                 │     │
│  └────────────────────────────────────────────────────────┘     │
│                         ▼                                         │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  Reasoning-Planning Integration                         │     │
│  │  - Reason about preconditions                           │     │
│  │  - Generate plan explanations                           │     │
│  │  - Transfer plans across domains                        │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
                            ▲
                            │ uses
┌──────────────────────────────────────────────────────────────────┐
│                      LAYER 2: PLANNING                            │
│  - HTN Planner + Causal Reasoning                                │
└──────────────────────────────────────────────────────────────────┘
                            ▲
                            │ uses
┌──────────────────────────────────────────────────────────────────┐
│                      LAYER 1: CAUSAL                              │
│  - Causal DAG + Mechanisms                                        │
└──────────────────────────────────────────────────────────────────┘
```

---

## Feature 1: Deductive Reasoning

### Motivation

Humans reason deductively:
- **Forward:** "It's raining → ground is wet"
- **Backward:** "Want dry ground → need to stop rain"

AI systems need the same capability for:
- Deriving consequences of actions
- Finding preconditions for goals
- Logical problem solving

### Core Concepts

**1. Knowledge Base:**
- Facts: "Socrates is human"
- Rules: "All humans are mortal"
- Logical operators: AND, OR, NOT, IMPLIES

**2. Forward Chaining:**
- Start with facts
- Apply rules
- Infer new facts
- Repeat until no new facts

**3. Backward Chaining:**
- Start with goal
- Find rules that conclude goal
- Recursively prove premises
- Search for supporting facts

**4. Inference Rules:**
- Modus Ponens: (P → Q) ∧ P ⊢ Q
- Modus Tollens: (P → Q) ∧ ¬Q ⊢ ¬P
- Chain rule: (P → Q) ∧ (Q → R) ⊢ (P → R)

### Implementation Design

```python
from dataclasses import dataclass
from typing import List, Dict, Set, Optional

@dataclass
class Fact:
    """Atomic fact in knowledge base."""
    predicate: str           # e.g., "human"
    arguments: tuple         # e.g., ("Socrates",)

    def matches(self, pattern: 'Fact') -> bool:
        """Check if fact matches pattern (with variables)."""
        pass

@dataclass
class Rule:
    """Logical rule: premises → conclusion."""
    premises: List[Fact]     # Conditions
    conclusion: Fact         # Derived fact
    confidence: float = 1.0  # Rule certainty

class KnowledgeBase:
    """Repository of facts and rules."""

    def __init__(self):
        self.facts: Set[Fact] = set()
        self.rules: List[Rule] = []

    def add_fact(self, fact: Fact):
        """Add fact to KB."""
        self.facts.add(fact)

    def add_rule(self, rule: Rule):
        """Add inference rule."""
        self.rules.append(rule)

    def query(self, query_fact: Fact) -> bool:
        """Check if fact is in KB or derivable."""
        pass

class DeductiveReasoner:
    """Deductive reasoning engine."""

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def forward_chain(self, max_iterations: int = 100) -> Set[Fact]:
        """
        Forward chaining inference.

        Algorithm:
        1. Start with known facts
        2. For each rule:
           - Check if premises are satisfied
           - If yes, add conclusion to facts
        3. Repeat until no new facts

        Returns:
            All derivable facts
        """
        pass

    def backward_chain(self, goal: Fact, depth: int = 0) -> bool:
        """
        Backward chaining to prove goal.

        Algorithm:
        1. Check if goal in facts → True
        2. Find rules concluding goal
        3. Recursively prove each premise
        4. If all premises proven → True

        Returns:
            True if goal provable
        """
        pass

    def explain(self, fact: Fact) -> List[Rule]:
        """Generate proof chain for fact."""
        pass
```

### Research Alignment

- **Forgy (1982):** RETE algorithm for efficient forward chaining
- **Kowalski (1974):** SLD resolution for backward chaining
- **Russell & Norvig (2020):** "AI: A Modern Approach" (Ch. 7-9)
- **Nilsson (1980):** "Principles of Artificial Intelligence"

---

## Feature 2: Abductive Reasoning

### Motivation

Abduction = inference to best explanation.

**Given:** Observations
**Find:** Most likely explanation

Examples:
- **Medical:** Symptoms → Disease
- **Debugging:** Bug behavior → Root cause
- **Science:** Experimental data → Theory

### Core Concepts

**1. Hypothesis Generation:**
- Generate candidate explanations
- Use domain knowledge
- Causal structure constrains hypotheses

**2. Explanation Scoring:**
- **Likelihood:** How well does hypothesis explain observations?
- **Parsimony:** Simpler explanations preferred (Occam's razor)
- **Prior:** How likely is hypothesis a priori?

**3. Scoring Function:**
```
score(H | O) = P(O | H) × P(H) / complexity(H)
             = likelihood × prior / parsimony
```

**4. Best Explanation:**
- Rank hypotheses by score
- Return top-k explanations
- Can maintain multiple hypotheses

### Implementation Design

```python
from dataclasses import dataclass
from typing import List, Callable
import numpy as np

@dataclass
class Hypothesis:
    """Candidate explanation."""
    explanation: Dict[str, Any]  # Variable assignments
    likelihood: float             # P(obs | hypothesis)
    prior: float                  # P(hypothesis)
    complexity: float             # Complexity penalty

    def score(self) -> float:
        """Combined score."""
        return (self.likelihood * self.prior) / (1 + self.complexity)

class AbductiveReasoner:
    """Abductive reasoning engine."""

    def __init__(self,
                 causal_model: CausalDAG,
                 hypothesis_generator: Callable):
        self.causal_model = causal_model
        self.hypothesis_generator = hypothesis_generator

    def explain(self,
               observations: Dict[str, Any],
               max_hypotheses: int = 10) -> List[Hypothesis]:
        """
        Generate best explanations for observations.

        Algorithm:
        1. Generate candidate hypotheses
        2. For each hypothesis:
           a. Calculate likelihood P(obs | H)
           b. Get prior P(H)
           c. Calculate complexity
        3. Score and rank hypotheses
        4. Return top-k

        Args:
            observations: Observed variables
            max_hypotheses: Max explanations to return

        Returns:
            Ranked list of explanations
        """
        # 1. Generate hypotheses
        candidates = self.hypothesis_generator(observations)

        # 2. Score each
        hypotheses = []
        for candidate in candidates:
            likelihood = self._calculate_likelihood(candidate, observations)
            prior = self._calculate_prior(candidate)
            complexity = self._calculate_complexity(candidate)

            hypothesis = Hypothesis(
                explanation=candidate,
                likelihood=likelihood,
                prior=prior,
                complexity=complexity
            )
            hypotheses.append(hypothesis)

        # 3. Rank
        hypotheses.sort(key=lambda h: h.score(), reverse=True)

        return hypotheses[:max_hypotheses]

    def _calculate_likelihood(self, hypothesis: Dict, observations: Dict) -> float:
        """P(observations | hypothesis)."""
        # Use causal model to predict observations given hypothesis
        pass

    def _calculate_prior(self, hypothesis: Dict) -> float:
        """P(hypothesis) - base rate."""
        pass

    def _calculate_complexity(self, hypothesis: Dict) -> float:
        """Complexity penalty (number of assumptions)."""
        return len(hypothesis)
```

### Research Alignment

- **Peirce (1878):** "Deduction, Induction, and Hypothesis" (origin of abduction)
- **Josephson & Josephson (1996):** "Abductive Inference"
- **Pearl (2000):** Causality (causal explanation)
- **Hobbs et al. (1993):** "Interpretation as Abduction"

---

## Feature 3: Analogical Reasoning

### Motivation

Humans learn by analogy:
- **Atom like solar system** (Rutherford model)
- **Heart like pump** (circulatory system)
- **Brain like computer** (cognitive science)

Transfer knowledge from familiar → unfamiliar domains.

### Core Concepts

**1. Structure Mapping:**
- Identify structural similarity
- Map relations (not just objects)
- Preserve relational structure

**2. Analogy Components:**
- **Source:** Familiar domain (solar system)
- **Target:** New domain (atom)
- **Mapping:** Correspondence (sun ↔ nucleus, planet ↔ electron)

**3. Transfer:**
- Infer unknown properties in target
- Based on known properties in source
- Validate transferred knowledge

**4. Similarity Metrics:**
- **Structural:** Relational alignment
- **Semantic:** Conceptual similarity
- **Pragmatic:** Goal relevance

### Implementation Design

```python
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

@dataclass
class AnalogicalMapping:
    """Mapping between source and target domains."""
    entity_mappings: Dict[str, str]      # source_entity -> target_entity
    relation_mappings: Dict[str, str]    # source_relation -> target_relation
    score: float                          # Mapping quality

class AnalogicalReasoner:
    """Analogical reasoning engine."""

    def __init__(self):
        self.case_library: List[CausalDAG] = []  # Known domains

    def find_analogy(self,
                    source: CausalDAG,
                    target: CausalDAG) -> Optional[AnalogicalMapping]:
        """
        Find structural mapping between domains.

        Algorithm:
        1. Identify candidate entity mappings
        2. Check relation compatibility
        3. Score mapping (structural + semantic)
        4. Optimize alignment

        Returns:
            Best mapping or None
        """
        pass

    def transfer_knowledge(self,
                          mapping: AnalogicalMapping,
                          source_knowledge: Dict) -> Dict:
        """
        Transfer knowledge from source to target via mapping.

        Args:
            mapping: Domain correspondence
            source_knowledge: Facts/rules in source domain

        Returns:
            Inferred knowledge in target domain
        """
        pass

    def solve_by_analogy(self,
                        problem: Problem,
                        case_library: List[Tuple[Problem, Solution]]) -> Solution:
        """
        Case-based reasoning: find similar past problem.

        Algorithm:
        1. Retrieve similar cases
        2. Find best analogy
        3. Adapt solution to current problem

        Returns:
            Adapted solution
        """
        pass
```

### Research Alignment

- **Gentner (1983):** "Structure-Mapping Theory"
- **Hofstadter & Mitchell (1994):** "Copycat" program
- **Holyoak & Thagard (1989):** "Analogical mapping by constraint satisfaction"
- **Forbus et al. (2011):** "Structure-Mapping Engine (SME)"

---

## Integration with Layer 2

### How Reasoning Enhances Planning

**1. Precondition Reasoning:**
```python
# Use backward chaining to find preconditions
goal = "Door is open"
preconditions = deductive_reasoner.backward_chain(goal)
# → ["Key in lock", "Door unlocked"]
```

**2. Plan Explanation:**
```python
# Generate natural language explanation
plan = planner.plan(goal, state)
explanation = abductive_reasoner.explain_plan(plan)
# → "Open door because key enables unlocking"
```

**3. Plan Transfer:**
```python
# Transfer plan from similar domain
source_plan = case_library.get("open_car_door")
mapping = analogical_reasoner.find_analogy(car, house)
adapted_plan = analogical_reasoner.transfer_plan(source_plan, mapping)
```

**4. Failure Explanation:**
```python
# Explain why plan failed
failure_observations = {"door": "locked", "tried": "push"}
explanations = abductive_reasoner.explain(failure_observations)
# → "Door locked because no key"
```

---

## Deliverables

**Code (est. 2,500 lines):**
- `HoloLoom/reasoning/deductive.py` (~800 lines)
- `HoloLoom/reasoning/abductive.py` (~700 lines)
- `HoloLoom/reasoning/analogical.py` (~700 lines)
- `HoloLoom/reasoning/integration.py` (~300 lines)

**Demos (est. 1,000 lines):**
- `demos/demo_deductive_reasoning.py` - Logic puzzle solving
- `demos/demo_abductive_diagnosis.py` - Medical diagnosis
- `demos/demo_analogical_transfer.py` - Knowledge transfer
- `demos/demo_reasoning_planning.py` - Integrated system

**Tests:**
- Unit tests for each reasoning type
- Integration tests with Layer 2
- End-to-end reasoning scenarios

---

## Success Criteria

**Option B Complete When:**
1. ✅ Deductive reasoning working (forward + backward chaining)
2. ✅ Abductive reasoning generating explanations
3. ✅ Analogical reasoning transferring knowledge
4. ✅ Integration with Layer 2 planning
5. ✅ All demos running
6. ✅ Tests passing
7. ✅ Documentation complete

---

**Let's build the thinking layer!** 🧠
