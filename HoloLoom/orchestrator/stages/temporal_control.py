"""
Temporal Control Stage
======================
Step 2 of the weaving cycle: Chrono Trigger fires and creates TemporalWindow.

This stage manages temporal aspects of the weaving cycle including:
- Time window creation
- Episode tracking
- Execution limits
"""

import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime

from HoloLoom.protocols.types import Query
from HoloLoom.chrono.trigger import ChronoTrigger, TemporalWindow, ExecutionLimits
from HoloLoom.loom.command import PatternSpec
from HoloLoom.orchestrator.protocols import StageResult


class TemporalControlStage:
    """
    Temporal Control Stage (Step 2: Chrono Trigger).

    Creates and manages temporal windows for memory filtering and
    execution control.
    """

    def __init__(
        self,
        chrono_trigger: ChronoTrigger,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the temporal control stage.

        Args:
            chrono_trigger: The chrono trigger instance
            logger: Logger instance
        """
        self.chrono_trigger = chrono_trigger
        self.logger = logger or logging.getLogger(__name__)
        self.current_episode = 0

    async def execute(
        self,
        query: Query,
        context: Dict[str, Any],
        pattern_spec: PatternSpec,
    ) -> StageResult:
        """
        Execute temporal control setup.

        Args:
            query: The user query
            context: Shared context (should contain pattern_spec)
            pattern_spec: Pattern specification for timing

        Returns:
            StageResult with temporal window and episode info
        """
        start_time = time.time()

        try:
            # Get pattern spec from context if not provided
            if not pattern_spec and "pattern_spec" in context:
                pattern_spec = context["pattern_spec"]

            # Fire the chrono trigger
            temporal_window = self.chrono_trigger.fire(
                query=query,
                execution_limits=ExecutionLimits(
                    max_time=pattern_spec.max_time if pattern_spec else 30.0,
                    max_passes=pattern_spec.max_passes if pattern_spec else 3,
                    enable_heartbeat=pattern_spec.enable_heartbeat if pattern_spec else False
                )
            )

            # Track episode
            self.current_episode = self.chrono_trigger.advance_episode()

            # Create episode ID for this weaving cycle
            episode_id = f"episode_{self.current_episode}_{datetime.now().isoformat()}"

            duration_ms = (time.time() - start_time) * 1000

            self.logger.info(
                f"  [2] Chrono Trigger fired: {temporal_window.pattern}, "
                f"episode={self.current_episode}"
            )

            return StageResult(
                stage_name="temporal_control",
                duration_ms=duration_ms,
                data={
                    "temporal_window": temporal_window,
                    "episode_id": episode_id,
                    "episode_number": self.current_episode,
                    "execution_limits": temporal_window.execution_limits,
                },
                metadata={
                    "window_pattern": temporal_window.pattern,
                    "max_time": temporal_window.execution_limits.max_time,
                }
            )

        except Exception as e:
            self.logger.error(f"Temporal control failed: {e}")
            duration_ms = (time.time() - start_time) * 1000
            return StageResult(
                stage_name="temporal_control",
                duration_ms=duration_ms,
                success=False,
                error=str(e),
                data={}
            )

    def get_stage_name(self) -> str:
        """Get the name of this stage."""
        return "temporal_control"

    def get_required_context_keys(self) -> list[str]:
        """Get required context keys."""
        return []  # Uses pattern_spec if available but not required

    def get_output_context_keys(self) -> list[str]:
        """Get output context keys."""
        return ["temporal_window", "episode_id", "episode_number"]