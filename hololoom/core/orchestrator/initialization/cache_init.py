#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Cache Initialization
==============================

Initialize three-tier semantic cache for 244D projections.

Extracted from weaving_orchestrator.py (November 2025 - Elegance Pass)
Original location: lines 912-947 (~36 lines)

This module handles:
- Semantic spectrum creation (244D semantic dimensions)
- Axis learning using embedder
- Adaptive semantic cache setup (hot + warm tiers)

This provides 3-10× speedup for semantic projections by caching at
the semantic dimension level instead of just the embedding level.

Author: Claude Code (Elegance Pass Refactoring - Phase 2)
Date: 2025-11-22
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hololoom.weaving_orchestrator import WeavingOrchestrator


logger = logging.getLogger(__name__)


def initialize_semantic_cache(orchestrator: 'WeavingOrchestrator') -> None:
    """
    Initialize three-tier semantic cache for 244D projections.

    This provides 3-10× speedup for semantic projections by caching at
    the semantic dimension level instead of just the embedding level.

    The three tiers are:
    1. Hot tier: Pre-loaded narrative patterns (~1000 entries)
    2. Warm tier: LRU cache for recently accessed queries (~5000 entries)
    3. Cold tier: On-demand computation (fallback)

    Args:
        orchestrator: The WeavingOrchestrator instance to initialize

    Example:
        >>> initialize_semantic_cache(orchestrator)

    Side Effects:
        Sets the following attributes on orchestrator:
        - semantic_spectrum: SemanticSpectrum instance (or None on failure)
        - semantic_cache: AdaptiveSemanticCache instance (or None on failure)

    Note:
        Falls back gracefully to direct semantic computation if initialization fails.
        This is expected behavior when semantic_calculus module is unavailable.
    """
    try:
        from hololoom.semantic_calculus.dimensions import EXTENDED_244_DIMENSIONS, SemanticSpectrum
        from hololoom.performance.semantic_cache import AdaptiveSemanticCache

        logger.info("Initializing semantic cache...")

        # Create global semantic spectrum
        orchestrator.semantic_spectrum = SemanticSpectrum(dimensions=EXTENDED_244_DIMENSIONS)

        # Learn axes using embedder
        logger.info("  Learning semantic axes...")
        orchestrator.semantic_spectrum.learn_axes(lambda word: orchestrator.embedder.encode([word])[0])

        # Initialize adaptive semantic cache
        orchestrator.semantic_cache = AdaptiveSemanticCache(
            semantic_spectrum=orchestrator.semantic_spectrum,
            embedder=orchestrator.embedder,
            hot_size=1000,  # Pre-loaded narrative patterns
            warm_size=5000   # LRU cache for recent queries
        )

        logger.info("  Semantic cache enabled (hot=1000, warm=5000)")

    except Exception as e:
        logger.warning(f"Failed to initialize semantic cache: {e}")
        logger.warning("Falling back to direct semantic computation")
        orchestrator.semantic_cache = None
        orchestrator.semantic_spectrum = None
