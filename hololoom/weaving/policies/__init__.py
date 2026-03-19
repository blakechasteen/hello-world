"""
Weaving Policies
================

RunPolicy implementations that shape pipeline runtime behavior.
Replaces the BaseStrategy wrapper pattern from v1.

- WeavingPolicyBase: shared lifecycle hooks (logging, events, timing)
- FastPolicy: skip memory if empty, abort only on critical stages
- ResearchPolicy: run everything, dynamic adaptation via recipe_override
"""

from hololoom.protocols import ComplexityLevel

from .bandit import BanditConfig, BanditPolicy
from .base import WeavingPolicyBase
from .fast import FastPolicy
from .research import ResearchPolicy

__all__ = [
    "WeavingPolicyBase",
    "FastPolicy",
    "ResearchPolicy",
    "BanditPolicy",
    "BanditConfig",
    "POLICIES",
]

# Factory mapping: complexity level -> policy constructor
POLICIES = {
    ComplexityLevel.LITE: lambda **kw: WeavingPolicyBase(**kw),
    ComplexityLevel.FAST: lambda **kw: FastPolicy(**kw),
    ComplexityLevel.FULL: lambda **kw: WeavingPolicyBase(**kw),
    ComplexityLevel.RESEARCH: lambda **kw: ResearchPolicy(**kw),
}
