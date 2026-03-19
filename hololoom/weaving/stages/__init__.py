"""
Weaving Stages
==============
Individual processing stages for the weaving pipeline.

Each stage is a self-contained unit that:
- Implements WeavingStageProtocol
- Has clear input/output contracts
- Can be tested independently
- Is reusable across different complexity strategies
"""

from .decision_collapse import DecisionCollapseStage
from .feature_extraction import FeatureExtractionStage
from .memory_retrieval import MemoryRetrievalStage
from .pattern_selection import PatternSelectionStage
from .temporal_control import TemporalControlStage

__all__ = [
    "PatternSelectionStage",
    "TemporalControlStage",
    "ThreadSelectionStage",
    "FeatureExtractionStage",
    "WarpTensioningStage",
    "MemoryRetrievalStage",
    "DecisionCollapseStage",
    "ToolExecutionStage",
    "FabricWeavingStage",
    "ReflectionStage",
]
