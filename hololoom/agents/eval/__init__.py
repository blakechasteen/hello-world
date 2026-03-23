"""
Agent identity evaluation — measurement infrastructure for the 7-file identity system.

Three axes:
    Coherence:       Does the agent behave consistently with itself?
    Differentiation: Does agent A behave differently from agent B?
    Groundedness:    When the agent says something about itself, is it true?

Continuous measurement. Every interaction is an eval. Sliding window, no epochs.
"""

from hololoom.agents.eval.axes import (
    coherence_score,
    differentiation_score,
    groundedness_score,
    composite_score,
)
from hololoom.agents.eval.probes import Probe, ProbeResult, ProbeSet
from hololoom.agents.eval.runner import EvalRunner
from hololoom.agents.eval.identity_loader import IdentityLoader

__all__ = [
    "coherence_score",
    "differentiation_score",
    "groundedness_score",
    "composite_score",
    "Probe",
    "ProbeResult",
    "ProbeSet",
    "EvalRunner",
    "IdentityLoader",
]
