#!/usr/bin/env python3
"""
Reflection and Caching Initialization
======================================

Initialize reflection loop, query cache, and dashboard constructor.

Extracted from weaving_orchestrator.py (November 2025 - Elegance Pass)
Original location: lines 514-560 (~47 lines)

This module handles:
- Reflection buffer setup for learning systems
- Query cache for performance optimization
- Smart query routing configuration
- Dashboard constructor for visualizations

Author: Claude Code (Elegance Pass Refactoring - Phase 2)
Date: 2025-11-22
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hololoom.core.orchestrator.weaving_orchestrator import WeavingOrchestrator

from hololoom.performance.cache import QueryCache
from hololoom.reflection.buffer import ReflectionBuffer
from hololoom.routing.classifier_factory import create_classifier, create_fast_path_router

logger = logging.getLogger(__name__)


def initialize_reflection_and_caching(
    orchestrator: WeavingOrchestrator,
    enable_reflection: bool,
    reflection_capacity: int
) -> None:
    """
    Initialize reflection loop, query cache, and dashboard constructor.

    Sets up learning systems and performance optimizations.

    Args:
        orchestrator: The WeavingOrchestrator instance to initialize
        enable_reflection: Whether to enable reflection loop
        reflection_capacity: Maximum reflection buffer capacity

    Example:
        >>> initialize_reflection_and_caching(
        ...     orchestrator=shuttle,
        ...     enable_reflection=True,
        ...     reflection_capacity=1000
        ... )

    Side Effects:
        Sets the following attributes on orchestrator:
        - enable_reflection: Reflection flag
        - reflection_buffer: ReflectionBuffer instance (or None)
        - query_cache: QueryCache instance
        - query_classifier: Query classifier for routing
        - fast_path_router: Fast path router (or None)
        - dashboard_constructor: DashboardConstructor (or None)
    """
    # Initialize reflection loop
    orchestrator.enable_reflection = enable_reflection
    if enable_reflection:
        orchestrator.reflection_buffer = ReflectionBuffer(
            capacity=reflection_capacity,
            persist_path="./reflections",
            learning_window=100
        )
        logger.info("Reflection loop enabled")
    else:
        orchestrator.reflection_buffer = None
        logger.info("Reflection loop disabled")

    # Initialize performance cache
    orchestrator.query_cache = QueryCache(max_size=50, ttl_seconds=300)
    logger.info("Query cache enabled (max_size=50, ttl=300s)")

    # Initialize smart query routing (November 2025 - Performance Optimization)
    orchestrator.query_classifier = create_classifier(orchestrator.cfg)
    orchestrator.fast_path_router = create_fast_path_router(orchestrator, orchestrator.cfg) if orchestrator.query_classifier else None

    if orchestrator.query_classifier and orchestrator.fast_path_router:
        logger.info(f"✅ Smart query routing enabled "
                        f"(classifier={orchestrator.cfg.routing_classifier}, "
                        f"semantic_tier={orchestrator.cfg.enable_semantic_tier})")
    else:
        logger.info("Smart query routing disabled")

    # Initialize dashboard constructor (Edward Tufte Machine)
    if orchestrator.enable_dashboards:
        from hololoom.visualization.constructor import DashboardConstructor
        orchestrator.dashboard_constructor = DashboardConstructor()
        logger.info("Dashboard generation enabled (Edward Tufte Machine)")
    else:
        orchestrator.dashboard_constructor = None
        logger.info("Dashboard generation disabled")
