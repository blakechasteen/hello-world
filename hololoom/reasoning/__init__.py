"""
HoloLoom Reasoning Engines

Layer 3 of Cognitive Architecture - Logical inference, explanation, transfer.

Provides three types of reasoning:
- Deductive: Logical inference from known facts
- Abductive: Best explanation for observations
- Analogical: Transfer knowledge across domains

Public API:
    DeductiveReasoner: Forward/backward chaining
    AbductiveReasoner: Hypothesis generation and scoring
    AnalogicalReasoner: Structure mapping and transfer
"""

from .abductive import (
    AbductiveReasoner,
    CausalRule,
    Hypothesis,
    HypothesisGenerator,
    HypothesisScorer,
    Observation,
    create_causal_rule,
    create_observation,
)
from .analogical import (
    AnalogicalMapping,
    AnalogicalReasoner,
    Case,
    CaseLibrary,
    Domain,
    Entity,
    KnowledgeTransferer,
    Relation,
    StructureMapper,
    create_domain,
    create_entity,
    create_relation,
)
from .deductive import (
    DeductiveReasoner,
    Fact,
    KnowledgeBase,
    Proof,
    Rule,
    Unifier,
    create_fact,
    create_rule,
)
from .integration import (
    FailureDiagnosis,
    PlanExplanation,
    ReasoningEnhancedPlanner,
    create_planning_knowledge_base,
)

__all__ = [
    # Deductive reasoning
    'Fact',
    'Rule',
    'Proof',
    'Unifier',
    'KnowledgeBase',
    'DeductiveReasoner',
    'create_fact',
    'create_rule',

    # Abductive reasoning
    'Hypothesis',
    'Observation',
    'CausalRule',
    'HypothesisGenerator',
    'HypothesisScorer',
    'AbductiveReasoner',
    'create_causal_rule',
    'create_observation',

    # Analogical reasoning
    'Entity',
    'Relation',
    'Domain',
    'AnalogicalMapping',
    'StructureMapper',
    'KnowledgeTransferer',
    'Case',
    'CaseLibrary',
    'AnalogicalReasoner',
    'create_entity',
    'create_relation',
    'create_domain',

    # Layer 2-3 Integration
    'ReasoningEnhancedPlanner',
    'PlanExplanation',
    'FailureDiagnosis',
    'create_planning_knowledge_base',
]
