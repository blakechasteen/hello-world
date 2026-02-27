#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Component Initialization
========================

Initialize all weaving architecture components.

Extracted from weaving_orchestrator.py (November 2025 - Elegance Pass)
Original location: lines 646-823 (~178 lines)

This module handles initialization of:
1. Loom Command - Pattern selection
2. Yarn Graph / Memory Backend - Thread storage
3. Shuttle Stage - MCTS-powered thread selection (optional)
4. Embedder - Matryoshka embeddings
5. Semantic Cache - 244D projection cache (if enabled)
6. Linguistic Gate - Phase 5 linguistic filtering (if enabled)
7. Tool Executor - Tool execution
8. Gradient Flow Router - Physics-based tool selection (if enabled)
9. Unified Physics Engine - Complete physics stack (if enabled)
10. Statistical Mechanics Engine - Memory consolidation (if enabled)
11. Retriever - Legacy shard retrieval (if using shards)

Author: Claude Code (Elegance Pass Refactoring - Phase 2)
Date: 2025-11-22
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from HoloLoom.weaving_orchestrator import WeavingOrchestrator

from HoloLoom.core.loom.command import LoomCommand
from HoloLoom.core.memory.graph import KG
from HoloLoom.core.embedding.spectral import MatryoshkaEmbeddings
from HoloLoom.tools import ToolExecutor
from HoloLoom.routing.flow_router import ToolRouter, ToolConfig


logger = logging.getLogger(__name__)


# Module availability flags (set by try/except imports in orchestrator)
SHUTTLE_AVAILABLE = False
UNIFIED_PHYSICS_AVAILABLE = False
STATISTICAL_MECHANICS_AVAILABLE = False

try:
    from HoloLoom.shuttle import create_shuttle_stage
    SHUTTLE_AVAILABLE = True
except ImportError:
    pass

try:
    from HoloLoom.physics.unified_physics import UnifiedPhysicsEngine
    UNIFIED_PHYSICS_AVAILABLE = True
except ImportError:
    pass

try:
    from HoloLoom.physics.statistical_mechanics import StatisticalMechanicsEngine
    STATISTICAL_MECHANICS_AVAILABLE = True
except ImportError:
    pass


def initialize_components(orchestrator: 'WeavingOrchestrator') -> None:
    """
    Initialize all weaving architecture components.

    This method sets up the complete weaving pipeline, including:
    - Pattern selection (Loom Command)
    - Memory backend (Yarn Graph / KG)
    - Thread selection (Shuttle Stage - optional)
    - Embeddings (Matryoshka)
    - Caching (Semantic Cache - optional)
    - Filtering (Linguistic Gate - optional)
    - Tool execution
    - Physics-based routing (optional)
    - Unified physics engine (optional)
    - Statistical mechanics (optional)
    - Legacy retriever (if using shards)

    Args:
        orchestrator: The WeavingOrchestrator instance to initialize

    Example:
        >>> initialize_components(orchestrator)

    Side Effects:
        Sets many attributes on orchestrator:
        - loom_command: LoomCommand instance
        - yarn_graph: YarnGraph/KG/WeavingMemoryAdapter
        - shuttle_stage: ShuttleStage (or None)
        - enable_shuttle: Shuttle flag
        - embedder: MatryoshkaEmbeddings instance
        - semantic_cache: AdaptiveSemanticCache (or None)
        - semantic_spectrum: SemanticSpectrum (or None)
        - linguistic_gate: LinguisticMatryoshkaGate (or None)
        - tool_executor: ToolExecutor instance
        - gradient_router: ToolRouter (or None)
        - unified_physics: UnifiedPhysicsEngine (or None)
        - stat_mech_engine: StatisticalMechanicsEngine (or None)
        - retriever: Retriever (or None)

    Note:
        Many components are optional and will be None if disabled or unavailable.
        This method calls _initialize_semantic_cache() and _initialize_linguistic_gate()
        if those features are enabled in the config.
    """
    # 1. Loom Command - Pattern selection
    orchestrator.loom_command = LoomCommand(
        default_pattern=orchestrator.default_pattern,
        auto_select=True,
        guardrails=orchestrator.guardrails,
    )

    # 2. Yarn Graph / Memory Backend - Thread storage
    # Priority: memory > yarn_graph > shards (for backward compatibility)
    if orchestrator.memory:
        # Use persistent memory backend (UnifiedMemory, etc.)
        from HoloLoom.core.memory.weaving_adapter import WeavingMemoryAdapter
        if isinstance(orchestrator.memory, WeavingMemoryAdapter):
            orchestrator.yarn_graph = orchestrator.memory
        else:
            # Wrap raw memory in adapter (use "factory" type for protocol backends)
            orchestrator.yarn_graph = WeavingMemoryAdapter(backend=orchestrator.memory, backend_type="factory")
        logger.info("Using persistent memory backend")
    elif orchestrator.yarn_graph_param:
        # Use KG instance directly (preferred modern API)
        orchestrator.yarn_graph = orchestrator.yarn_graph_param
        logger.info(f"Using Yarn Graph (KG) with {orchestrator.yarn_graph.G.number_of_nodes()} nodes")
    else:
        # Use LegacyShardsAdapter with shards (backward compatibility)
        # This is deprecated - use yarn_graph (KG) or memory parameter instead
        from HoloLoom.core.memory.graph import LegacyShardsAdapter
        orchestrator.yarn_graph = LegacyShardsAdapter(orchestrator.shards)
        logger.info(f"Using LegacyShardsAdapter (deprecated) with {len(orchestrator.shards)} shards")

    # 2a. Shuttle Stage (Step 3: MCTS-powered thread selection) - January 2025
    # OPTIONAL: Enable Shuttle for intelligent Warp↔Yarn intersection
    # If disabled, falls back to simple yarn_graph.select_threads()
    orchestrator.enable_shuttle = getattr(orchestrator.cfg, 'enable_shuttle', True)  # Default: enabled

    if SHUTTLE_AVAILABLE and orchestrator.enable_shuttle:
        try:
            # Create retriever for Warp adapter
            from HoloLoom.core.memory.base import create_retriever
            retriever = create_retriever(orchestrator.cfg)

            # Create ShuttleStage with auto-derived config
            if isinstance(orchestrator.yarn_graph, KG):
                orchestrator.shuttle_stage = create_shuttle_stage(
                    config=orchestrator.cfg,
                    kg=orchestrator.yarn_graph,
                    retriever=retriever,
                    shuttle_config=None  # Auto-derive from cfg
                )
                logger.info(f"[SHUTTLE] Enabled (mode={orchestrator.shuttle_stage.shuttle_config.mode.value})")
            else:
                logger.warning("[SHUTTLE] Requires KG instance, falling back to simple retrieval")
                orchestrator.shuttle_stage = None
                orchestrator.enable_shuttle = False
        except Exception as e:
            logger.warning(f"[SHUTTLE] Failed to initialize, falling back to simple retrieval: {e}")
            orchestrator.shuttle_stage = None
            orchestrator.enable_shuttle = False
    else:
        orchestrator.shuttle_stage = None
        if not SHUTTLE_AVAILABLE:
            logger.info("[SHUTTLE] Not available (module not installed)")
        else:
            logger.info("[SHUTTLE] Disabled, using simple yarn_graph.select_threads()")

    # 3. Component factories (will be instantiated per-query with pattern spec)
    orchestrator.embedder = MatryoshkaEmbeddings(
        sizes=orchestrator.cfg.scales,
        base_model_name=orchestrator.cfg.base_model_name
    )

    # 3a. Semantic Cache - Three-tier caching for 244D semantic projections
    if orchestrator.enable_semantic_cache:
        from HoloLoom.core.orchestrator.initialization.cache_init import initialize_semantic_cache
        initialize_semantic_cache(orchestrator)
    else:
        orchestrator.semantic_cache = None
        orchestrator.semantic_spectrum = None

    # 3b. Phase 5: Linguistic Matryoshka Gate (optional)
    if orchestrator.cfg.enable_linguistic_gate:
        from HoloLoom.core.orchestrator.initialization.linguistic_init import initialize_linguistic_gate
        initialize_linguistic_gate(orchestrator)
    else:
        orchestrator.linguistic_gate = None

    # 4. Tool Executor
    orchestrator.tool_executor = ToolExecutor()

    # 4a. Gradient Flow Router (Phase 1: Physics-based tool selection)
    # Uses gradient descent to route queries to optimal tools based on cost/quality/latency
    try:
        tool_configs = [
            ToolConfig("answer", cost=0.3, quality=0.8, latency=100),
            ToolConfig("search", cost=0.5, quality=0.7, latency=150),
            ToolConfig("notion_write", cost=0.6, quality=0.75, latency=120),
            ToolConfig("calc", cost=0.1, quality=0.9, latency=50)
        ]
        orchestrator.gradient_router = ToolRouter(
            tools=tool_configs,
            cost_weight=0.3,
            quality_weight=0.5,  # Favor quality
            latency_weight=0.2
        )
        logger.info("Gradient flow router initialized (Phase 1 integration)")
    except Exception as e:
        logger.warning(f"Failed to initialize gradient flow router: {e}")
        orchestrator.gradient_router = None

    # 4b. Unified Physics Engine (Phases 1-4: Complete physics stack)
    # Coordinates gradient flow, fluid dynamics, thermodynamics, and wave mechanics
    # for self-optimizing behavior across routing, packing, exploration, and pattern detection
    orchestrator.unified_physics = None
    if UNIFIED_PHYSICS_AVAILABLE and orchestrator.cfg.enable_unified_physics:
        try:
            orchestrator.unified_physics = UnifiedPhysicsEngine(
                enable_routing=True,        # Phase 1: Gradient flow routing
                enable_packing=True,        # Phase 2: Fluid dynamics context packing
                enable_thermodynamics=True, # Phase 3: Exploration/exploitation balance
                enable_wave_mechanics=True, # Phase 4: Pattern detection
                mode="adaptive"            # Adaptive strategy selection
            )
            logger.info("✓ Unified Physics Engine initialized (Phases 1-4 complete)")
            logger.info("  → Routing: Gradient flow (optimal tool selection)")
            logger.info("  → Packing: Fluid dynamics (context optimization)")
            logger.info("  → Exploration: Thermodynamics (F = E - T*S)")
            logger.info("  → Patterns: Wave mechanics (interference detection)")
        except Exception as e:
            logger.warning(f"Failed to initialize unified physics: {e}")
            orchestrator.unified_physics = None
    elif not UNIFIED_PHYSICS_AVAILABLE:
        logger.debug("Unified physics not available (import failed)")
    else:
        logger.debug("Unified physics disabled in config")

    # 4c. Statistical Mechanics Engine (Phase 5: Memory consolidation)
    # Uses Gibbs distribution for emergent clustering and phase transition detection
    if orchestrator.enable_statistical_mechanics:
        try:
            orchestrator.stat_mech_engine = StatisticalMechanicsEngine(
                temperature=orchestrator.consolidation_temperature,
                cooling_rate=orchestrator.consolidation_cooling_rate
            )
            logger.info("✓ Statistical Mechanics Engine initialized (Phase 5 complete)")
            logger.info(f"  → Consolidation interval: {orchestrator.consolidation_interval}s")
            logger.info(f"  → Initial temperature: {orchestrator.consolidation_temperature}")
            logger.info(f"  → Cooling rate: {orchestrator.consolidation_cooling_rate}")
        except Exception as e:
            logger.warning(f"Failed to initialize statistical mechanics: {e}")
            orchestrator.stat_mech_engine = None
    elif not STATISTICAL_MECHANICS_AVAILABLE:
        logger.debug("Statistical mechanics not available (import failed)")
    else:
        logger.debug("Statistical mechanics disabled")

    # 5. Retriever (for context)
    # Only create traditional retriever if using shards (legacy path)
    if orchestrator.shards:
        from HoloLoom.core.memory.base import create_retriever
        # Legacy YarnGraph has .shards dict
        if hasattr(orchestrator.yarn_graph, 'shards'):
            orchestrator.retriever = create_retriever(
                shards=list(orchestrator.yarn_graph.shards.values()),
                emb=orchestrator.embedder,
                fusion_weights=orchestrator.cfg.fusion_weights
            )
        else:
            # Direct shards list (shouldn't happen with new API)
            orchestrator.retriever = create_retriever(
                shards=orchestrator.shards,
                emb=orchestrator.embedder,
                fusion_weights=orchestrator.cfg.fusion_weights
            )
    else:
        # Using KG or dynamic memory backend - retriever not needed
        # Thread selection happens via yarn_graph.select_threads()
        orchestrator.retriever = None

    logger.debug("All weaving components initialized")
