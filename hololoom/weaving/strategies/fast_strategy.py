"""
FAST Strategy
=============
5-step execution strategy for standard queries (<150ms target).

Steps:
1. Pattern Selection
2. Temporal Control
3. Feature Extraction
4. Memory Retrieval
5. Quick Decision + Response
"""

from typing import List, Dict, Any

from hololoom.protocols import ComplexityLevel
from hololoom.weaving.strategies.base import BaseStrategy


class FastStrategy(BaseStrategy):
    """
    FAST complexity strategy (5 steps).

    Balanced processing for standard queries with <150ms target latency.
    Includes pattern selection, temporal windows, but simplified decision making.
    """

    def get_complexity_level(self) -> ComplexityLevel:
        """Get the complexity level this strategy handles."""
        return ComplexityLevel.FAST

    def get_stage_names(self) -> List[str]:
        """
        Get the ordered list of stages for FAST execution.

        FAST uses 5 key stages:
        1. Pattern selection (determine processing mode)
        2. Temporal control (create time window)
        3. Feature extraction (extract features)
        4. Memory retrieval (get context)
        5. Decision + Execution (simplified, combined)

        Returns:
            List of stage names to execute
        """
        return [
            "pattern_selection",
            "temporal_control",
            "feature_extraction",
            "memory_retrieval",
            "quick_decision",  # Simplified decision + execution
        ]

    def should_skip_stage(
        self,
        stage_name: str,
        context: Dict[str, Any],
    ) -> bool:
        """
        Check if a stage should be skipped for FAST execution.

        FAST strategy may skip:
        - Thread selection (uses default threads)
        - Warp tensioning (not needed for simple decisions)
        - Complex reflection (basic only)

        Args:
            stage_name: Name of the stage
            context: Current execution context

        Returns:
            True if stage should be skipped
        """
        # Skip these stages in FAST mode
        skip_stages = [
            "thread_selection",  # Use default threads
            "warp_tensioning",   # Not needed for quick decisions
            "reflection",        # Skip detailed reflection
        ]

        if stage_name in skip_stages:
            return True

        # Skip memory retrieval if no memory source
        if stage_name == "memory_retrieval":
            has_memory = context.get("has_memory_source", False)
            if not has_memory:
                self.logger.info("Skipping memory retrieval - no memory source")
                return True

        return False

    def should_terminate_early(
        self,
        stage_name: str,
        result: Any,
        context: Dict[str, Any],
    ) -> bool:
        """
        Check if execution should terminate early.

        FAST strategy may terminate early if:
        - Cache hit found (can skip remaining stages)
        - High confidence quick answer available

        Args:
            stage_name: Name of the stage just executed
            result: Result from the stage
            context: Current execution context

        Returns:
            True if execution should terminate
        """
        # Check for cache hit
        if stage_name == "memory_retrieval":
            if context.get("cache_hit", False):
                self.logger.info("Cache hit - terminating early")
                return True

        # Check for high-confidence quick answer
        if stage_name == "quick_decision":
            confidence = context.get("confidence", 0.0)
            if confidence > 0.95:
                self.logger.info(f"High confidence ({confidence:.2f}) - terminating")
                return True

        return False

    def should_abort_on_failure(self, stage_name: str) -> bool:
        """
        Check if execution should abort on stage failure.

        FAST strategy is more resilient - only aborts on critical failures.

        Args:
            stage_name: Name of the failed stage

        Returns:
            True if execution should abort
        """
        # Only abort on absolutely critical stages
        critical_stages = [
            "pattern_selection",  # Need pattern to proceed
            "feature_extraction",  # Need features for decision
        ]
        return stage_name in critical_stages