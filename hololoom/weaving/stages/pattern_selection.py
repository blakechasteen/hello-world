"""
Pattern Selection Stage
=======================
Step 1 of the weaving cycle: Loom Command selects Pattern Card.

This stage determines the processing pattern (BARE/FAST/FUSED) based on
query complexity and configuration.
"""

import logging
import time
from typing import Any

from hololoom.loom.command import LoomCommand, PatternCard, PatternSpec
from hololoom.protocols import ComplexityLevel
from hololoom.protocols.types import Query
from hololoom.weaving.protocols import StageResult


class PatternSelectionStage:
    """
    Pattern Selection Stage (Step 1: Loom Command).

    Selects the appropriate pattern card based on:
    - Query complexity assessment
    - Configuration preferences
    - Override parameters
    """

    def __init__(
        self,
        loom_command: LoomCommand,
        default_pattern: PatternCard | None = None,
        enable_auto_detect: bool = True,
        logger: logging.Logger | None = None,
    ):
        """
        Initialize the pattern selection stage.

        Args:
            loom_command: The loom command instance for pattern selection
            default_pattern: Default pattern if not overridden
            enable_auto_detect: Whether to auto-detect pattern from query
            logger: Logger instance
        """
        self.loom_command = loom_command
        self.default_pattern = default_pattern
        self.enable_auto_detect = enable_auto_detect
        self.logger = logger or logging.getLogger(__name__)

    async def execute(
        self,
        query: Query,
        context: dict[str, Any],
        pattern_spec: PatternSpec | None = None,
    ) -> StageResult:
        """
        Execute pattern selection.

        Args:
            query: The user query
            context: Shared context (may contain pattern_override)
            pattern_spec: Optional pre-determined pattern spec

        Returns:
            StageResult with selected pattern card and spec
        """
        start_time = time.time()

        try:
            # Check for pattern override in context
            pattern_override = context.get("pattern_override")
            complexity = context.get("complexity", ComplexityLevel.FAST)

            # If pattern spec provided, use it directly
            if pattern_spec:
                pattern_card = pattern_spec.card
                self.logger.info(f"  [1] Using provided pattern spec: {pattern_card.value}")
            # Otherwise, select pattern card
            elif pattern_override:
                pattern_card = pattern_override
                pattern_spec = self.loom_command.get_pattern_spec(pattern_card)
                self.logger.info(f"  [1] Pattern override: {pattern_card.value}")
            elif self.enable_auto_detect:
                # Auto-detect based on complexity
                pattern_card = self._detect_pattern_from_complexity(complexity)
                pattern_spec = self.loom_command.get_pattern_spec(pattern_card)
                self.logger.info(f"  [1] Auto-detected pattern: {pattern_card.value} for complexity {complexity.value}")
            else:
                # Use default pattern
                pattern_card = self.default_pattern or self.loom_command.default_pattern
                pattern_spec = self.loom_command.get_pattern_spec(pattern_card)
                self.logger.info(f"  [1] Using default pattern: {pattern_card.value}")

            duration_ms = (time.time() - start_time) * 1000

            return StageResult(
                stage_name="pattern_selection",
                duration_ms=duration_ms,
                data={
                    "pattern_card": pattern_card,
                    "pattern_spec": pattern_spec,
                    "complexity": complexity,
                },
                metadata={
                    "auto_detected": pattern_override is None and self.enable_auto_detect,
                    "pattern_value": pattern_card.value,
                }
            )

        except Exception as e:
            self.logger.error(f"Pattern selection failed: {e}")
            duration_ms = (time.time() - start_time) * 1000
            return StageResult(
                stage_name="pattern_selection",
                duration_ms=duration_ms,
                success=False,
                error=str(e),
                data={}
            )

    def _detect_pattern_from_complexity(self, complexity: ComplexityLevel) -> PatternCard:
        """
        Map complexity level to pattern card.

        Args:
            complexity: The assessed complexity level

        Returns:
            Appropriate pattern card
        """
        mapping = {
            ComplexityLevel.LITE: PatternCard.BARE,
            ComplexityLevel.FAST: PatternCard.FAST,
            ComplexityLevel.FULL: PatternCard.FUSED,
            ComplexityLevel.RESEARCH: PatternCard.FUSED,  # Research uses FUSED with extended limits
        }
        return mapping.get(complexity, PatternCard.FAST)

    def get_stage_name(self) -> str:
        """Get the name of this stage."""
        return "pattern_selection"

    def get_required_context_keys(self) -> list[str]:
        """Get required context keys."""
        return []  # No required keys, but uses optional ones

    def get_output_context_keys(self) -> list[str]:
        """Get output context keys."""
        return ["pattern_card", "pattern_spec"]
