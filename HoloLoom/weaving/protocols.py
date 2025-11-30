"""
Weaving Protocols
=================
Protocol definitions for the weaving architecture.

These protocols define the contracts for stages, strategies, and results.
"""

from typing import Protocol, Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from HoloLoom.protocols.types import Query, MemoryShard, Features
from HoloLoom.protocols import ComplexityLevel, ProvenanceTrace
from HoloLoom.fabric.spacetime import Spacetime
from HoloLoom.chrono.trigger import TemporalWindow
from HoloLoom.loom.command import PatternSpec


@dataclass
class StageResult:
    """Result from a weaving stage execution."""
    stage_name: str
    duration_ms: float
    data: Dict[str, Any]
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class WeavingStageProtocol(Protocol):
    """Protocol for individual weaving stages."""

    async def execute(
        self,
        query: Query,
        context: Dict[str, Any],
        pattern_spec: PatternSpec,
    ) -> StageResult:
        """
        Execute the stage processing.

        Args:
            query: The user query
            context: Shared context from previous stages
            pattern_spec: Pattern specification for this execution

        Returns:
            StageResult with stage output data
        """
        ...

    def get_stage_name(self) -> str:
        """Get the name of this stage."""
        ...

    def get_required_context_keys(self) -> List[str]:
        """Get the context keys this stage requires."""
        ...

    def get_output_context_keys(self) -> List[str]:
        """Get the context keys this stage produces."""
        ...


class ComplexityStrategyProtocol(Protocol):
    """Protocol for complexity-based execution strategies."""

    def get_complexity_level(self) -> ComplexityLevel:
        """Get the complexity level this strategy handles."""
        ...

    def get_stages(self) -> List[str]:
        """Get the ordered list of stage names to execute."""
        ...

    async def execute(
        self,
        query: Query,
        stages: Dict[str, WeavingStageProtocol],
        initial_context: Dict[str, Any],
    ) -> Spacetime:
        """
        Execute the strategy's stages in order.

        Args:
            query: The user query
            stages: Map of stage name to stage instance
            initial_context: Initial context for execution

        Returns:
            Final Spacetime result
        """
        ...

    def should_skip_stage(
        self,
        stage_name: str,
        context: Dict[str, Any],
    ) -> bool:
        """
        Check if a stage should be skipped based on context.

        Args:
            stage_name: Name of the stage
            context: Current execution context

        Returns:
            True if stage should be skipped
        """
        ...