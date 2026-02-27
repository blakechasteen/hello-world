"""
Strategy Factory
================
Factory for creating complexity-based strategies.
"""

import logging
from typing import Dict, Optional

from HoloLoom.protocols import ComplexityLevel
from HoloLoom.orchestrator.protocols import WeavingStageProtocol, ComplexityStrategyProtocol
from HoloLoom.orchestrator.strategies.lite_strategy import LiteStrategy
from HoloLoom.orchestrator.strategies.fast_strategy import FastStrategy
from HoloLoom.orchestrator.strategies.full_strategy import FullStrategy
from HoloLoom.orchestrator.strategies.research_strategy import ResearchStrategy


def create_strategy(
    complexity: ComplexityLevel,
    stages: Dict[str, WeavingStageProtocol],
    logger: Optional[logging.Logger] = None,
) -> ComplexityStrategyProtocol:
    """
    Create a strategy for the given complexity level.

    Args:
        complexity: The complexity level
        stages: Map of stage name to stage instance
        logger: Optional logger

    Returns:
        Strategy instance for the complexity level

    Raises:
        ValueError: If complexity level is not supported
    """
    strategy_map = {
        ComplexityLevel.LITE: LiteStrategy,
        ComplexityLevel.FAST: FastStrategy,
        ComplexityLevel.FULL: FullStrategy,
        ComplexityLevel.RESEARCH: ResearchStrategy,
    }

    strategy_class = strategy_map.get(complexity)
    if not strategy_class:
        raise ValueError(f"No strategy available for complexity: {complexity}")

    return strategy_class(stages=stages, logger=logger)