#!/usr/bin/env python3
"""
Recursive Learning Initialization
==================================

Initialize recursive learning components (lazy initialization).

Extracted from weaving_orchestrator.py (November 2025 - Elegance Pass)
Original location: lines 561-645 (~85 lines)

This module handles:
- Scratchpad for provenance tracking (Phase 1)
- Pattern learner for successful query patterns (Phase 2)
- Hot pattern tracker for adaptive retrieval (Phase 3)
- Advanced refiner for low-confidence queries (Phase 4)
- Background learner for Thompson Sampling and policy updates (Phase 5)

This is called on first weave() when enable_recursive_learning=True.

Author: Claude Code (Elegance Pass Refactoring - Phase 2)
Date: 2025-11-22
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hololoom.core.orchestrator.weaving_orchestrator import WeavingOrchestrator


logger = logging.getLogger(__name__)


def initialize_recursive_learning(orchestrator: WeavingOrchestrator) -> None:
    """
    Initialize recursive learning components (lazy initialization).

    This is called on first weave() when enable_recursive_learning=True.
    Initializes all 5 phases of the recursive learning system:

    - Phase 1: Scratchpad for provenance tracking
    - Phase 2: Pattern learner for successful query patterns
    - Phase 3: Hot pattern tracker for adaptive retrieval
    - Phase 4: Advanced refiner for low-confidence queries
    - Phase 5: Background learner for Thompson Sampling and policy updates

    Args:
        orchestrator: The WeavingOrchestrator instance to initialize

    Example:
        >>> initialize_recursive_learning(orchestrator)

    Side Effects:
        Sets orchestrator._recursive_components to a dict containing:
        - scratchpad: Scratchpad instance (if enabled)
        - pattern_learner: PatternLearner instance
        - hot_tracker: HotPatternTracker instance (if enabled)
        - refiner: AdvancedRefiner instance
        - thompson_priors: ThompsonPriors instance
        - policy_weights: PolicyWeights instance
        - metrics: LearningMetrics instance
        - background_learner: BackgroundLearner instance (if enabled)

        Or sets to None if import fails (with warning).

    Note:
        This is lazy-initialized (not called in __init__) because:
        1. Recursive learning imports are heavy
        2. Not all queries need recursive learning
        3. Allows graceful degradation if dependencies unavailable
    """
    if orchestrator._recursive_components is not None:
        return  # Already initialized

    try:
        from hololoom.recursive.advanced_refinement import AdvancedRefiner
        from hololoom.recursive.full_learning_loop import (
            BackgroundLearner,
            LearningMetrics,
            PolicyWeights,
            ThompsonPriors,
        )
        from hololoom.recursive.hot_patterns import HotPatternTracker
        from hololoom.recursive.loop_integration import PatternLearner
        from hololoom.recursive.scratchpad import Scratchpad

        logger.info("Initializing recursive learning components...")

        # Components dictionary
        components: dict[str, Any] = {}

        # 1. Scratchpad (Phase 1)
        if orchestrator.cfg.recursive_learning_enable_scratchpad:
            components['scratchpad'] = Scratchpad(capacity=1000)
            logger.info("  [Phase 1] Scratchpad enabled (provenance tracking)")

        # 2. Pattern Learner (Phase 2)
        components['pattern_learner'] = PatternLearner()
        logger.info("  [Phase 2] Pattern learner enabled")

        # 3. Hot Pattern Tracker (Phase 3)
        if orchestrator.cfg.recursive_learning_enable_hot_patterns:
            components['hot_tracker'] = HotPatternTracker()
            logger.info("  [Phase 3] Hot pattern tracker enabled")

        # 4. Advanced Refiner (Phase 4)
        components['refiner'] = AdvancedRefiner(
            orchestrator=orchestrator,
            scratchpad=components.get('scratchpad'),
            enable_learning=True
        )
        logger.info("  [Phase 4] Advanced refiner enabled")

        # 5. Background Learner (Phase 5)
        components['thompson_priors'] = ThompsonPriors()
        components['policy_weights'] = PolicyWeights()
        components['metrics'] = LearningMetrics()

        if orchestrator.cfg.recursive_learning_enable_background:
            components['background_learner'] = BackgroundLearner(
                orchestrator=orchestrator,
                thompson_priors=components['thompson_priors'],
                policy_weights=components['policy_weights'],
                update_interval=orchestrator.cfg.recursive_learning_update_interval
            )
            logger.info("  [Phase 5] Background learner enabled")
        else:
            components['background_learner'] = None

        orchestrator._recursive_components = components

        # Note: Background learner.start() should be called asynchronously in weave()
        # Cannot use await here since initialize_recursive_learning is synchronous
        logger.info("Recursive learning initialization complete")

    except ImportError as e:
        logger.warning(
            f"Failed to initialize recursive learning components: {e}. "
            f"Recursive learning will be disabled. "
            f"Install dependencies or disable via config.enable_recursive_learning=False"
        )
        orchestrator.enable_recursive_learning = False
        orchestrator._recursive_components = None
