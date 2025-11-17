#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complexity Strategy - Progressive Processing Modes
====================================================

Defines strategy classes for progressive complexity modes (LITE/FAST/FULL/RESEARCH)
in HoloLoom's mythRL architecture.

**Extracted: November 2025** - Refactored from weaving_orchestrator.py
**Lines: 312** (reduced from 3,476-line monolith)

Each strategy encapsulates:
- Multipass crawl configuration (passes, thresholds, limits)
- Processing steps and performance targets
- Graph expansion behavior

Author: Claude Code (refactoring pass)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
from abc import ABC, abstractmethod

from HoloLoom.protocols import ComplexityLevel


@dataclass
class CrawlConfig:
    """
    Configuration for multipass memory crawling.

    Defines how memory retrieval should be performed at each complexity level:
    - passes: Number of retrieval passes
    - thresholds: Similarity threshold for each pass
    - limits: Maximum items to retrieve per pass
    - graph_expansion: Whether to traverse graph relationships
    """
    passes: int
    thresholds: List[float]
    limits: List[int]
    graph_expansion: bool


class ComplexityStrategy(ABC):
    """
    Abstract base class for complexity strategies.

    Each strategy defines the processing characteristics for a specific
    complexity level in the 3-5-7-9 progressive system.

    **Design Pattern:** Strategy pattern for processing complexity
    **Abstraction:** Encapsulates complexity-specific behavior
    """

    def __init__(self):
        """Initialize the strategy with crawl configuration."""
        self._crawl_config = self._create_crawl_config()

    @property
    @abstractmethod
    def complexity_level(self) -> ComplexityLevel:
        """Return the complexity level for this strategy."""
        pass

    @property
    @abstractmethod
    def steps(self) -> int:
        """Return the number of processing steps."""
        pass

    @property
    @abstractmethod
    def target_latency_ms(self) -> int:
        """Return the target latency in milliseconds."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a description of this complexity level."""
        pass

    @abstractmethod
    def _create_crawl_config(self) -> CrawlConfig:
        """Create the crawl configuration for this complexity level."""
        pass

    @property
    def crawl_config(self) -> CrawlConfig:
        """Get the multipass memory crawl configuration."""
        return self._crawl_config

    def get_config_dict(self) -> Dict[str, Any]:
        """
        Get configuration as dictionary (for backward compatibility).

        Returns:
            Dict with crawl configuration keys
        """
        return {
            'passes': self._crawl_config.passes,
            'thresholds': self._crawl_config.thresholds,
            'limits': self._crawl_config.limits,
            'graph_expansion': self._crawl_config.graph_expansion
        }


class LiteStrategy(ComplexityStrategy):
    """
    LITE complexity strategy (3 steps).

    **Performance Target:** <50ms
    **Use Cases:** Simple lookups, cached queries, low-latency apps
    **Behavior:**
    - Single retrieval pass (no multipass)
    - High similarity threshold (0.7)
    - Small result set (5 items max)
    - No graph expansion

    **Processing Steps:**
    1. Extract features
    2. Route decision
    3. Execute tool
    """

    @property
    def complexity_level(self) -> ComplexityLevel:
        return ComplexityLevel.LITE

    @property
    def steps(self) -> int:
        return 3

    @property
    def target_latency_ms(self) -> int:
        return 50

    @property
    def description(self) -> str:
        return "Simple lookups, cached queries (3 steps, <50ms)"

    def _create_crawl_config(self) -> CrawlConfig:
        return CrawlConfig(
            passes=1,
            thresholds=[0.7],  # Single pass, high threshold
            limits=[5],  # 5 items max
            graph_expansion=False  # No graph traversal
        )


class FastStrategy(ComplexityStrategy):
    """
    FAST complexity strategy (5 steps).

    **Performance Target:** <150ms
    **Use Cases:** Standard queries, real-time apps, production systems
    **Behavior:**
    - Two retrieval passes (broad → focused)
    - Progressive thresholds (0.6 → 0.75)
    - Moderate result sets (8 → 12 items)
    - Light graph traversal

    **Processing Steps:**
    1. Pattern selection
    2. Extract features
    3. Temporal windows
    4. Route decision
    5. Execute tool
    """

    @property
    def complexity_level(self) -> ComplexityLevel:
        return ComplexityLevel.FAST

    @property
    def steps(self) -> int:
        return 5

    @property
    def target_latency_ms(self) -> int:
        return 150

    @property
    def description(self) -> str:
        return "Standard queries, real-time apps (5 steps, <150ms)"

    def _create_crawl_config(self) -> CrawlConfig:
        return CrawlConfig(
            passes=2,
            thresholds=[0.6, 0.75],  # Broad → focused
            limits=[8, 12],  # 8 initial, 12 expansion
            graph_expansion=True  # Light graph traversal
        )


class FullStrategy(ComplexityStrategy):
    """
    FULL complexity strategy (7 steps).

    **Performance Target:** <300ms
    **Use Cases:** Complex queries, production systems, quality-critical
    **Behavior:**
    - Three retrieval passes (progressive gating)
    - Progressive thresholds (0.6 → 0.75 → 0.85)
    - Large result sets (12 → 20 → 15 items)
    - Full graph traversal

    **Processing Steps:**
    1. Pattern selection
    2. Extract features
    3. Temporal windows
    4. Decision engine
    5. Route decision
    6. Synthesis bridge
    7. Execute tool
    """

    @property
    def complexity_level(self) -> ComplexityLevel:
        return ComplexityLevel.FULL

    @property
    def steps(self) -> int:
        return 7

    @property
    def target_latency_ms(self) -> int:
        return 300

    @property
    def description(self) -> str:
        return "Complex queries, production systems (7 steps, <300ms)"

    def _create_crawl_config(self) -> CrawlConfig:
        return CrawlConfig(
            passes=3,
            thresholds=[0.6, 0.75, 0.85],  # Progressive gating
            limits=[12, 20, 15],  # Expanding search space
            graph_expansion=True  # Full graph traversal
        )


class ResearchStrategy(ComplexityStrategy):
    """
    RESEARCH complexity strategy (9 steps).

    **Performance Target:** No limit (quality over speed)
    **Use Cases:** Research, debugging, quality maximization, deep analysis
    **Behavior:**
    - Four retrieval passes (maximum depth)
    - Progressive thresholds (0.5 → 0.65 → 0.8 → 0.9)
    - Very large result sets (20 → 30 → 25 → 15 items)
    - Aggressive graph traversal

    **Processing Steps:**
    1. Pattern selection
    2. Extract features
    3. Temporal windows
    4. Decision engine
    5. Advanced WarpSpace
    6. Route decision
    7. Synthesis bridge
    8. Full tracing
    9. Execute tool
    """

    @property
    def complexity_level(self) -> ComplexityLevel:
        return ComplexityLevel.RESEARCH

    @property
    def steps(self) -> int:
        return 9

    @property
    def target_latency_ms(self) -> int:
        return 0  # No limit

    @property
    def description(self) -> str:
        return "Research, debugging, quality maximization (9 steps, no limit)"

    def _create_crawl_config(self) -> CrawlConfig:
        return CrawlConfig(
            passes=4,
            thresholds=[0.5, 0.65, 0.8, 0.9],  # Maximum depth
            limits=[20, 30, 25, 15],  # Deep exploration
            graph_expansion=True  # Aggressive graph traversal
        )


def create_complexity_strategy(level: ComplexityLevel) -> ComplexityStrategy:
    """
    Factory function to create a complexity strategy.

    Args:
        level: Complexity level (LITE/FAST/FULL/RESEARCH)

    Returns:
        ComplexityStrategy instance for the specified level

    Example:
        >>> strategy = create_complexity_strategy(ComplexityLevel.FAST)
        >>> print(strategy.steps)  # 5
        >>> print(strategy.target_latency_ms)  # 150
        >>> config = strategy.crawl_config
        >>> print(config.passes)  # 2

    Raises:
        ValueError: If complexity level is unknown
    """
    strategies = {
        ComplexityLevel.LITE: LiteStrategy,
        ComplexityLevel.FAST: FastStrategy,
        ComplexityLevel.FULL: FullStrategy,
        ComplexityLevel.RESEARCH: ResearchStrategy,
    }

    strategy_class = strategies.get(level)
    if strategy_class is None:
        raise ValueError(f"Unknown complexity level: {level}")

    return strategy_class()


def get_all_strategies() -> Dict[ComplexityLevel, ComplexityStrategy]:
    """
    Get all complexity strategies.

    Returns:
        Dict mapping ComplexityLevel to strategy instances

    Example:
        >>> strategies = get_all_strategies()
        >>> for level, strategy in strategies.items():
        ...     print(f"{level.value}: {strategy.steps} steps")
    """
    return {
        level: create_complexity_strategy(level)
        for level in ComplexityLevel
    }
