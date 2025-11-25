#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recursive Learning Integration
===============================

Recursive learning loop integration for weaving orchestration.

Extracted from weaving_orchestrator.py (November 2025 - Elegance Pass)
Original location: lines 1843-1934 (~92 lines total, 1 method)

This module handles:
- 5-phase recursive learning integration
- Scratchpad provenance tracking
- Pattern learning from successful queries
- Hot pattern tracking for adaptive retrieval
- Refinement for low-confidence results
- Thompson Sampling and policy weight updates

Author: Claude Code (Elegance Pass Refactoring - Phase 8)
Date: 2025-11-22
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from HoloLoom.weaving_orchestrator import WeavingOrchestrator

from HoloLoom.protocols.types import Query
from HoloLoom.fabric.spacetime import Spacetime


logger = logging.getLogger(__name__)


async def apply_recursive_learning(
    orchestrator: 'WeavingOrchestrator',
    spacetime: Spacetime,
    query: Query
) -> Spacetime:
    """
    Apply recursive learning loop to spacetime result.

    This integrates all 5 phases of recursive learning:
    - Phase 1: Scratchpad provenance tracking
    - Phase 2: Pattern learning from successful queries
    - Phase 3: Hot pattern tracking for adaptive retrieval
    - Phase 4: Refinement for low-confidence results
    - Phase 5: Thompson Sampling and policy weight updates

    Args:
        orchestrator: The WeavingOrchestrator instance
        spacetime: Spacetime result from weaving
        query: Original query

    Returns:
        Potentially refined spacetime (or original if no refinement needed)

    Example:
        >>> spacetime = await orchestrator.weave(query)
        >>> spacetime = await apply_recursive_learning(
        ...     orchestrator,
        ...     spacetime,
        ...     query
        ... )
        >>> # Spacetime may have been refined if confidence was low

    Note:
        - Requires orchestrator._recursive_components to be initialized
        - Refinement triggered when confidence < threshold (default 0.75)
        - Thompson Sampling priors updated on every call
        - Policy weights adapt based on success/failure
    """
    if not orchestrator._recursive_components:
        return spacetime

    components = orchestrator._recursive_components
    confidence = spacetime.trace.tool_confidence

    # Phase 1: Scratchpad - Track provenance
    if components.get('scratchpad'):
        components['scratchpad'].add_entry(
            thought=f"Query: {query.text[:100]}",
            action=f"Tool: {spacetime.trace.tool_selected}, Adapter: {spacetime.trace.policy_adapter}",
            observation=f"Confidence: {confidence:.2f}",
            score=confidence
        )

    # Phase 2: Pattern Learning - Learn from high-confidence queries
    if confidence >= orchestrator.cfg.recursive_learning_refinement_threshold:
        # Learn pattern
        components['pattern_learner'].learn_from_spacetime(spacetime)

    # Phase 3: Hot Pattern Tracking - Track access frequency
    if components.get('hot_tracker'):
        components['hot_tracker'].record_access(spacetime)

    # Phase 4: Refinement - Refine low-confidence results
    if confidence < orchestrator.cfg.recursive_learning_refinement_threshold:
        orchestrator.logger.info(
            f"[LEARNING] Low confidence ({confidence:.2f}), triggering refinement"
        )

        refinement_result = await components['refiner'].refine(
            query=query,
            initial_spacetime=spacetime,
            strategy=None,  # Auto-select
            max_iterations=orchestrator.cfg.recursive_learning_max_iterations,
            quality_threshold=0.9
        )

        spacetime = refinement_result.final_spacetime

        # Log refinement to scratchpad
        if components.get('scratchpad'):
            components['scratchpad'].add_entry(
                thought=f"Refinement: {refinement_result.strategy_used.value}",
                action=f"Iterations: {refinement_result.iterations}",
                observation=refinement_result.summary(),
                score=refinement_result.trajectory[-1].score()
            )

    # Phase 5: Thompson Sampling and Policy Updates
    tool = spacetime.trace.tool_selected
    adapter = spacetime.trace.policy_adapter
    final_confidence = spacetime.trace.tool_confidence

    # Update Thompson priors
    if final_confidence >= 0.75:
        components['thompson_priors'].update_success(tool, final_confidence)
    else:
        components['thompson_priors'].update_failure(tool, final_confidence)

    # Update policy weights
    success = final_confidence >= 0.75
    components['policy_weights'].update(adapter, success)

    # Update metrics
    components['metrics'].update(final_confidence)
    components['metrics'].thompson_updates += 1
    components['metrics'].policy_updates += 1

    # Record for background learner
    if components.get('background_learner'):
        components['background_learner'].record_spacetime(spacetime)

    return spacetime
