"""
WeaveHouse System Initialization
================================

Factory functions for creating complete WeaveHouse systems with
all 5 core looms, consensus engine, and optional dreaming.

This module provides the main integration point for Phase 7,
connecting the Multi-Loom Architecture to the HoloLoom orchestration layer.

Date: December 2025
Author: HoloLoom Architecture Team
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from hololoom.loom.weave_house import WeaveHouse, create_weave_house
from hololoom.loom.dreaming import DreamOrchestrator, create_dream_orchestrator
from hololoom.loom.consensus import LoomConsensus, create_loom_consensus
from hololoom.loom.core_looms import (
    RecallLoom,
    ReasonLoom,
    ReachLoom,
    ReflectLoom,
    RefuseLoom,
    create_all_looms,
)
from hololoom.loom.protocol import Loom

logger = logging.getLogger(__name__)


async def create_weave_house_system(
    config: Optional[Any] = None,
    memory_backend: Optional[Any] = None,
    enable_dreaming: bool = True,
    dreaming_interval: float = 3600.0,
    math_bleed_rate: float = 0.3,
    pattern_bleed_rate: float = 0.2,
    exploration_depth: int = 2,
    tension_threshold: float = 0.3,
    loom_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[WeaveHouse, Optional[DreamOrchestrator]]:
    """
    Factory function to create a complete WeaveHouse system.

    Creates all 5 core looms with shared configuration, a LoomConsensus
    engine with exploration enabled, a WeaveHouse composite, and optionally
    a DreamOrchestrator for background consolidation.

    Args:
        config: Optional HoloLoom Config object for system-wide settings
        memory_backend: Optional shared memory backend for all looms
        enable_dreaming: Whether to create DreamOrchestrator (default: True)
        dreaming_interval: Seconds between dream cycles (default: 3600)
        math_bleed_rate: Math insight bleed rate for dreaming (default: 0.3)
        pattern_bleed_rate: Pattern insight bleed rate (default: 0.2)
        exploration_depth: How deep to explore disagreements (default: 2)
        tension_threshold: Minimum tension to trigger exploration (default: 0.3)
        loom_kwargs: Additional kwargs passed to loom constructors

    Returns:
        Tuple of (WeaveHouse, DreamOrchestrator or None)

    Example:
        ```python
        from hololoom.config import Config
        from hololoom.loom.initialization import create_weave_house_system

        config = Config.fused()

        weave_house, dreamer = await create_weave_house_system(
            config=config,
            enable_dreaming=True,
            exploration_depth=3
        )

        # Use weave house for queries
        result = await weave_house.weave("What is Thompson Sampling?")

        # Start background dreaming
        if dreamer:
            await dreamer.start_dreaming()
        ```
    """
    loom_kwargs = loom_kwargs or {}

    # Extract relevant config settings if provided
    if config is not None:
        # Apply config settings to loom kwargs
        if hasattr(config, 'weave_house_exploration_depth'):
            exploration_depth = config.weave_house_exploration_depth
        if hasattr(config, 'weave_house_tension_threshold'):
            tension_threshold = config.weave_house_tension_threshold
        if hasattr(config, 'dream_consolidation_interval'):
            dreaming_interval = config.dream_consolidation_interval
        if hasattr(config, 'enable_dreaming'):
            enable_dreaming = config.enable_dreaming

    # Pass memory backend to looms if provided
    if memory_backend is not None:
        loom_kwargs['memory_backend'] = memory_backend

    # Create all 5 core looms
    looms = create_all_looms(**loom_kwargs)

    logger.info(f"Created {len(looms)} core looms: {[l.perspective for l in looms]}")

    # Create consensus engine with exploration enabled
    consensus = create_loom_consensus(
        exploration_enabled=True,
        agreement_threshold=0.7,
    )

    # Create WeaveHouse
    weave_house = WeaveHouse(
        looms=looms,
        consensus=consensus,
        exploration_depth=exploration_depth,
        tension_threshold=tension_threshold,
        name="WeaveHouse",
    )

    logger.info(
        f"Created WeaveHouse with exploration_depth={exploration_depth}, "
        f"tension_threshold={tension_threshold}"
    )

    # Optionally create DreamOrchestrator
    dream_orchestrator = None
    if enable_dreaming:
        dream_orchestrator = create_dream_orchestrator(
            looms=looms,
            consolidation_interval=int(dreaming_interval),
            math_bleed_rate=math_bleed_rate,
            pattern_bleed_rate=pattern_bleed_rate,
        )
        logger.info(
            f"Created DreamOrchestrator with interval={dreaming_interval}s, "
            f"math_bleed={math_bleed_rate}, pattern_bleed={pattern_bleed_rate}"
        )

    return weave_house, dream_orchestrator


def create_research_weave_house(
    config: Optional[Any] = None,
    memory_backend: Optional[Any] = None,
    **kwargs
) -> WeaveHouse:
    """
    Create a WeaveHouse optimized for research tasks.

    Uses deeper exploration and lower tension thresholds to
    thoroughly investigate disagreements.

    Args:
        config: Optional HoloLoom Config
        memory_backend: Optional shared memory backend
        **kwargs: Additional arguments

    Returns:
        Research-optimized WeaveHouse
    """
    from hololoom.loom.domain_houses.research_house import ResearchHouse, ResearchConfig

    research_config = ResearchConfig(
        exploration_depth=3,
        tension_threshold=0.2,
        include_creative=True,
        require_evidence=True,
    )

    return ResearchHouse(config=research_config, **kwargs)


def create_code_review_weave_house(
    config: Optional[Any] = None,
    memory_backend: Optional[Any] = None,
    **kwargs
) -> WeaveHouse:
    """
    Create a WeaveHouse optimized for code review tasks.

    Emphasizes verification and adversarial testing over creativity.

    Args:
        config: Optional HoloLoom Config
        memory_backend: Optional shared memory backend
        **kwargs: Additional arguments

    Returns:
        Code-review-optimized WeaveHouse
    """
    from hololoom.loom.domain_houses.code_review_house import (
        CodeReviewHouse,
        CodeReviewConfig
    )

    review_config = CodeReviewConfig(
        exploration_depth=2,
        tension_threshold=0.25,
        include_creative=False,
        security_focus=True,
        style_checking=True,
    )

    return CodeReviewHouse(config=review_config, **kwargs)


__all__ = [
    'create_weave_house_system',
    'create_research_weave_house',
    'create_code_review_weave_house',
]
