"""
RESEARCH Strategy
=================
9-step execution strategy for research mode (no time limit).

Steps:
1. Pattern Selection
2. Temporal Control
3. Thread Selection
4. Feature Extraction (parallel)
5. Warp Space Tensioning (parallel)
6. Memory Retrieval (parallel)
7. Decision Collapse
8. Tool Execution
9. Reflection & Learning
"""

from typing import List

from HoloLoom.core.protocols import ComplexityLevel
from HoloLoom.core.orchestrator.strategies.base import BaseStrategy


class ResearchStrategy(BaseStrategy):
    """
    RESEARCH complexity strategy (9 steps).

    Maximum quality processing for research, debugging, and exploration.
    No time limits - focuses on quality over speed.
    """

    def get_complexity_level(self) -> ComplexityLevel:
        """Get the complexity level this strategy handles."""
        return ComplexityLevel.RESEARCH

    def get_stage_names(self) -> List[str]:
        """
        Get the ordered list of stages for RESEARCH execution.

        RESEARCH uses all 9 stages with full tracing:
        1. Pattern selection
        2. Temporal control
        3. Thread selection
        4. Feature extraction
        5. Warp tensioning
        6. Memory retrieval
        7. Decision collapse
        8. Tool execution
        9. Reflection

        Returns:
            List of stage names to execute
        """
        return [
            "pattern_selection",
            "temporal_control",
            "thread_selection",
            "feature_extraction",
            "warp_tensioning",
            "memory_retrieval",
            "decision_collapse",
            "tool_execution",
            "reflection",
        ]