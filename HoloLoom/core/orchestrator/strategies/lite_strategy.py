"""
LITE Strategy
=============
3-step execution strategy for simple queries (<50ms target).

Steps:
1. Extract (features)
2. Route (decision)
3. Execute (response)
"""

from typing import List

from HoloLoom.protocols import ComplexityLevel
from HoloLoom.orchestrator.strategies.base import BaseStrategy


class LiteStrategy(BaseStrategy):
    """
    LITE complexity strategy (3 steps).

    Minimal processing for simple lookups and cached queries.
    Target: <50ms latency.
    """

    def get_complexity_level(self) -> ComplexityLevel:
        """Get the complexity level this strategy handles."""
        return ComplexityLevel.LITE

    def get_stage_names(self) -> List[str]:
        """
        Get the ordered list of stages for LITE execution.

        LITE uses only 3 essential stages:
        1. Feature extraction (minimal)
        2. Quick decision (routing)
        3. Fast execution (response)

        Returns:
            List of stage names to execute
        """
        return [
            "feature_extraction",  # Minimal feature extraction
            "quick_decision",      # Simple routing decision
            "fast_execution",      # Direct response generation
        ]