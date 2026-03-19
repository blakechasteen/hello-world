"""
HoloLoom Reflection Module
===========================
Learning loop for continuous improvement.

Exports:
- ReflectionBuffer: Main learning buffer
- ReflectionMetrics: Performance metrics
- LearningSignal: Actionable learning insights
- FeedbackHub: Central multi-temporal feedback orchestrator
- TemporalFusion: Cross-timescale signal fusion
- CausationTracker: Credit assignment via causation chains
- Feedback adapters and self-feedback diagnostics
"""

from hololoom.reflection.buffer import LearningSignal, ReflectionBuffer, ReflectionMetrics

# FeedbackHub system (March 2026)
try:
    from hololoom.core.reflection.feedback_hub import FeedbackHub
    from hololoom.core.reflection.temporal_fusion import TemporalFusion
    from hololoom.core.reflection.causation import CausationTracker
    _HAS_FEEDBACK_HUB = True
except ImportError:
    _HAS_FEEDBACK_HUB = False

__all__ = [
    "ReflectionBuffer",
    "ReflectionMetrics",
    "LearningSignal",
]

if _HAS_FEEDBACK_HUB:
    __all__.extend([
        "FeedbackHub",
        "TemporalFusion",
        "CausationTracker",
    ])
