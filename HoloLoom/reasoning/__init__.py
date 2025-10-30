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

from .deductive import (
    Fact, Rule, Proof, Unifier, KnowledgeBase, DeductiveReasoner,
    create_fact, create_rule
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
]
