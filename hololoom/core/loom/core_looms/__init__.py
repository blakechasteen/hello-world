"""
Core Looms - The Five Perspectives
===================================

The five core looms, each a coherent way of seeing:

- RECALL (The Librarian): Dense retrieval, memory-grounded
- REASON (The Logician): Causal inference, logic-grounded
- REACH (The Artist): Lateral association, creativity-grounded
- REFLECT (The Auditor): Consistency checking, verification-grounded
- REFUSE (The Skeptic): Adversarial probing, breaking-grounded

Philosophy:
"Each loom sees different truths. Together, they weave a fabric
richer than any single perspective."

Date: December 2025
Author: HoloLoom Architecture Team
"""

from .recall_loom import RecallLoom
from .reason_loom import ReasonLoom
from .reach_loom import ReachLoom
from .reflect_loom import ReflectLoom
from .refuse_loom import RefuseLoom

__all__ = [
    'RecallLoom',
    'ReasonLoom',
    'ReachLoom',
    'ReflectLoom',
    'RefuseLoom',
]


def create_all_looms(**kwargs):
    """
    Factory function to create all 5 core looms.

    Args:
        **kwargs: Arguments passed to each loom constructor

    Returns:
        List of all 5 core looms
    """
    return [
        RecallLoom(**kwargs),
        ReasonLoom(**kwargs),
        ReachLoom(**kwargs),
        ReflectLoom(**kwargs),
        RefuseLoom(**kwargs),
    ]
