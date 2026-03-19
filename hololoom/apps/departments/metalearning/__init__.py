#!/usr/bin/env python3
"""
Meta-Learning Department - Few-shot learning, transfer learning, and meta-adaptation.

Exports:
- MetaLearningDepartment: Main department class
- FewShotLearner: Learn from few examples
- TransferLearner: Transfer knowledge across tasks
- MetaAdaptationEngine: Select optimal learning strategies
- KnowledgeConsolidator: Consolidate cross-department knowledge
"""

from .metalearning import (
    AdaptationStrategy,
    ConsolidatedKnowledge,
    Example,
    FewShotLearner,
    KnowledgeConsolidator,
    LearningStrategy,
    MetaAdaptationEngine,
    MetaLearningDepartment,
    Prototype,
    TaskContext,
    TaskType,
    TransferLearner,
    TransferredKnowledge,
    TransferStrategy,
)

__all__ = [
    "MetaLearningDepartment",
    "FewShotLearner",
    "TransferLearner",
    "MetaAdaptationEngine",
    "KnowledgeConsolidator",
    "TaskType",
    "TransferStrategy",
    "AdaptationStrategy",
    "Example",
    "TaskContext",
    "Prototype",
    "TransferredKnowledge",
    "ConsolidatedKnowledge",
    "LearningStrategy"
]
