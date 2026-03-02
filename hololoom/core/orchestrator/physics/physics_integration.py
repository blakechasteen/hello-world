#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physics Integration
===================

Unified physics integration for weaving orchestration.

Extracted from weaving_orchestrator.py (November 2025 - Elegance Pass)
Original location: lines 1690-1793 (~103 lines total, 1 method)

This module handles:
- Physics-enhanced weaving with complete provenance
- Integration with UnifiedPhysics system (4 phases)
- Provenance tracking for all physics-based decisions

Author: Claude Code (Elegance Pass Refactoring - Phase 7)
Date: 2025-11-22
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from hololoom.weaving_orchestrator import WeavingOrchestrator
    from hololoom.physics.unified_physics import UnifiedPhysicsResult

from hololoom.protocols.types import Query
from hololoom.fabric.spacetime import Spacetime


logger = logging.getLogger(__name__)


async def weave_with_physics(
    orchestrator: 'WeavingOrchestrator',
    query: Query,
    track_provenance: bool = True
) -> Tuple[Spacetime, Optional['UnifiedPhysicsResult']]:
    """
    Physics-enhanced weaving with complete unified physics integration.

    Processes query through ALL physics layers (Phases 1-4) and provides
    complete provenance of every physics-based decision.

    Args:
        orchestrator: The WeavingOrchestrator instance
        query: Input query
        track_provenance: If True, return full physics result with provenance

    Returns:
        (spacetime, physics_result) tuple where:
            - spacetime: Complete weaving result with physics metadata
            - physics_result: Full physics provenance (if track_provenance=True)

    Example:
        >>> spacetime, physics = await weave_with_physics(
        ...     orchestrator,
        ...     Query(text="What is thermodynamics?"),
        ...     track_provenance=True
        ... )
        >>> print(f"Physics routing: {physics.routing_decision.target}")
        >>> print(f"Temperature: {physics.exploration_temperature:.2f}")
        >>> print(f"Patterns: {len(physics.constructive_patterns)}")

    Note:
        - Falls back to standard weaving if unified_physics not available
        - Enhances spacetime with physics metadata when track_provenance=True
        - Requires cfg.physics_track_provenance to be enabled for full tracking
    """
    if not orchestrator.unified_physics:
        logger.warning("Unified physics not available, falling back to standard weaving")
        spacetime = await orchestrator.weave(query)
        return spacetime, None

    # Standard weaving to get base result
    spacetime = await orchestrator.weave(query)

    # Prepare physics inputs
    actions = orchestrator.tool_executor.tools
    action_metrics = {
        "answer": {"cost": 0.3, "quality": 0.8, "latency": 0.1, "error": 0.1},
        "search": {"cost": 0.6, "quality": 0.7, "latency": 0.2, "error": 0.15},
        "notion_write": {"cost": 0.6, "quality": 0.75, "latency": 0.15, "error": 0.1},
        "calc": {"cost": 0.1, "quality": 0.9, "latency": 0.05, "error": 0.05}
    }

    # Build graph structure for wave mechanics (if yarn_graph available)
    graph_structure = None
    if orchestrator.yarn_graph and hasattr(orchestrator.yarn_graph, 'G'):
        # Extract edges from knowledge graph
        graph_structure = [
            (str(source), str(target))
            for source, target in orchestrator.yarn_graph.G.edges()
        ]

    # Process through unified physics
    physics_result = await orchestrator.unified_physics.process(
        query=query.text,
        actions=actions,
        action_metrics=action_metrics,
        graph_structure=graph_structure
    )

    # Enhance spacetime with physics provenance
    if track_provenance and orchestrator.cfg.physics_track_provenance:
        spacetime.metadata['unified_physics'] = {
            # Phase 1: Routing
            'routing_target': physics_result.routing_decision.target if physics_result.routing_decision else None,
            'routing_loss': physics_result.routing_loss,
            'routing_ms': physics_result.routing_ms,

            # Phase 2: Packing (if available)
            'context_efficiency': physics_result.context_efficiency,
            'packing_ms': physics_result.packing_ms,

            # Phase 3: Thermodynamics
            'selected_action': physics_result.selected_action,
            'exploration_temperature': physics_result.exploration_temperature,
            'free_energy': physics_result.free_energy,
            'thermodynamics_ms': physics_result.thermodynamics_ms,

            # Phase 4: Wave Mechanics
            'constructive_patterns': len(physics_result.constructive_patterns),
            'destructive_patterns': len(physics_result.destructive_patterns),
            'resonances': len(physics_result.resonances),
            'wave_mechanics_ms': physics_result.wave_mechanics_ms,

            # Unified metrics
            'total_energy': physics_result.total_energy,
            'total_entropy': physics_result.total_entropy,
            'total_free_energy': physics_result.total_free_energy,
            'physics_duration_ms': physics_result.duration_ms,

            # System statistics
            'physics_stats': orchestrator.unified_physics.get_statistics()
        }

        logger.info(
            f"✓ Physics enhancement complete: "
            f"routing={physics_result.routing_decision.target if physics_result.routing_decision else 'N/A'}, "
            f"T={physics_result.exploration_temperature:.2f}, "
            f"patterns={len(physics_result.constructive_patterns)}, "
            f"F={physics_result.total_free_energy:.3f}"
        )

    return spacetime, physics_result if track_provenance else None
