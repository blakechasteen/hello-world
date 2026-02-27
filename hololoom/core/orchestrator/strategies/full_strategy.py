"""
FULL Strategy
=============
7-step execution strategy for complex queries (<300ms target).

Steps:
1. Pattern Selection
2. Temporal Control
3. Thread Selection
4. Feature Extraction
5. Memory Retrieval
6. Decision Engine
7. Synthesis Bridge
"""

from typing import List

from hololoom.core.protocols import ComplexityLevel
from hololoom.core.orchestrator.strategies.base import BaseStrategy


class FullStrategy(BaseStrategy):
    """
    FULL complexity strategy (7 steps).

    Complete processing for complex queries requiring deep reasoning.
    Target: <300ms latency.
    """

    def get_complexity_level(self) -> ComplexityLevel:
        """Get the complexity level this strategy handles."""
        return ComplexityLevel.FULL

    def get_stage_names(self) -> List[str]:
        """
        Get the ordered list of stages for FULL execution.

        FULL uses 7 comprehensive stages:
        1. Pattern selection
        2. Temporal control
        3. Thread selection
        4. Feature extraction
        5. Memory retrieval
        6. Decision collapse
        7. Tool execution with synthesis

        Returns:
            List of stage names to execute
        """
        return [
            "pattern_selection",
            "temporal_control",
            "thread_selection",
            "feature_extraction",
            "memory_retrieval",
            "decision_collapse",
            "tool_execution",
        ]