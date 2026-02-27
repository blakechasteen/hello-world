"""
Base Strategy
=============
Abstract base class for complexity-based execution strategies.
"""

import logging
import time
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod

from HoloLoom.core.protocols.types import Query
from HoloLoom.core.protocols import ComplexityLevel, ProvenanceTrace
from HoloLoom.core.fabric.spacetime import Spacetime, WeavingTrace
from HoloLoom.core.orchestrator.protocols import WeavingStageProtocol, StageResult


class BaseStrategy(ABC):
    """
    Abstract base class for weaving execution strategies.

    Each strategy defines:
    - Which stages to execute
    - In what order
    - With what conditions
    """

    def __init__(
        self,
        stages: Dict[str, WeavingStageProtocol],
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the base strategy.

        Args:
            stages: Map of stage name to stage instance
            logger: Logger instance
        """
        self.stages = stages
        self.logger = logger or logging.getLogger(__name__)
        self.stage_results: List[StageResult] = []

    @abstractmethod
    def get_complexity_level(self) -> ComplexityLevel:
        """Get the complexity level this strategy handles."""
        ...

    @abstractmethod
    def get_stage_names(self) -> List[str]:
        """Get the ordered list of stage names to execute."""
        ...

    async def execute(
        self,
        query: Query,
        initial_context: Dict[str, Any],
    ) -> Spacetime:
        """
        Execute the strategy's stages in order.

        Args:
            query: The user query
            initial_context: Initial execution context

        Returns:
            Final Spacetime result
        """
        start_time = time.time()
        context = initial_context.copy()
        self.stage_results = []

        self.logger.info(
            f"Executing {self.get_complexity_level().value} strategy "
            f"with {len(self.get_stage_names())} stages"
        )

        # Execute stages in order
        for stage_name in self.get_stage_names():
            if stage_name not in self.stages:
                self.logger.warning(f"Stage '{stage_name}' not found, skipping")
                continue

            # Check if stage should be skipped
            if self.should_skip_stage(stage_name, context):
                self.logger.info(f"Skipping stage '{stage_name}'")
                continue

            # Execute the stage
            stage = self.stages[stage_name]
            try:
                result = await self._execute_stage(
                    stage, query, context, stage_name
                )
                self.stage_results.append(result)

                # Update context with stage outputs
                if result.success and result.data:
                    context.update(result.data)

                # Check for early termination
                if self.should_terminate_early(stage_name, result, context):
                    self.logger.info(f"Early termination after stage '{stage_name}'")
                    break

            except Exception as e:
                self.logger.error(f"Stage '{stage_name}' failed: {e}")
                # Decide whether to continue or abort
                if self.should_abort_on_failure(stage_name):
                    raise

        # Create final Spacetime result
        total_duration = (time.time() - start_time) * 1000
        return self._create_spacetime(query, context, total_duration)

    async def _execute_stage(
        self,
        stage: WeavingStageProtocol,
        query: Query,
        context: Dict[str, Any],
        stage_name: str,
    ) -> StageResult:
        """
        Execute a single stage.

        Args:
            stage: The stage to execute
            query: The user query
            context: Current execution context
            stage_name: Name of the stage

        Returns:
            StageResult from the stage
        """
        self.logger.info(f"Executing stage: {stage_name}")

        # Get pattern spec from context
        pattern_spec = context.get("pattern_spec")

        # Execute the stage
        result = await stage.execute(query, context, pattern_spec)

        self.logger.info(
            f"Stage '{stage_name}' completed in {result.duration_ms:.2f}ms"
        )

        return result

    def should_skip_stage(
        self,
        stage_name: str,
        context: Dict[str, Any],
    ) -> bool:
        """
        Check if a stage should be skipped.

        Can be overridden by subclasses for custom logic.

        Args:
            stage_name: Name of the stage
            context: Current execution context

        Returns:
            True if stage should be skipped
        """
        return False

    def should_terminate_early(
        self,
        stage_name: str,
        result: StageResult,
        context: Dict[str, Any],
    ) -> bool:
        """
        Check if execution should terminate early.

        Can be overridden by subclasses for custom logic.

        Args:
            stage_name: Name of the stage just executed
            result: Result from the stage
            context: Current execution context

        Returns:
            True if execution should terminate
        """
        return False

    def should_abort_on_failure(self, stage_name: str) -> bool:
        """
        Check if execution should abort on stage failure.

        Can be overridden by subclasses for custom logic.

        Args:
            stage_name: Name of the failed stage

        Returns:
            True if execution should abort
        """
        # By default, abort on critical stages
        critical_stages = [
            "pattern_selection",
            "feature_extraction",
            "decision_collapse"
        ]
        return stage_name in critical_stages

    def _create_spacetime(
        self,
        query: Query,
        context: Dict[str, Any],
        total_duration_ms: float,
    ) -> Spacetime:
        """
        Create the final Spacetime result.

        Args:
            query: The original query
            context: Final execution context
            total_duration_ms: Total execution time in milliseconds

        Returns:
            Spacetime object with results and provenance
        """
        # Extract key results from context
        selected_tool = context.get("selected_tool", "unknown")
        response = context.get("response", "")
        confidence = context.get("confidence", 0.0)

        # Create weaving trace
        trace = WeavingTrace(
            query=query.text,
            stages_executed=[r.stage_name for r in self.stage_results],
            stage_durations={
                r.stage_name: r.duration_ms
                for r in self.stage_results
            },
            total_duration_ms=total_duration_ms,
            complexity=self.get_complexity_level().value,
        )

        # Create Spacetime
        return Spacetime(
            response=response,
            confidence=confidence,
            tool_used=selected_tool,
            trace=trace,
            metadata={
                "strategy": self.__class__.__name__,
                "complexity": self.get_complexity_level().value,
                "stages_completed": len(self.stage_results),
                "total_stages": len(self.get_stage_names()),
            }
        )

    def get_stage_timings(self) -> Dict[str, float]:
        """
        Get timing information for executed stages.

        Returns:
            Map of stage name to duration in milliseconds
        """
        return {
            result.stage_name: result.duration_ms
            for result in self.stage_results
        }