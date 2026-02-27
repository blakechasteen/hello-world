#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistical Mechanics Integration
==================================

Statistical mechanics engine integration for memory consolidation.

Extracted from weaving_orchestrator.py (November 2025 - Elegance Pass)
Original location: lines 2157-2344 (~188 lines total, 3 methods)

This module handles:
- Conversion of memory shards to microstates
- Conversion of macrostates back to memory shards
- Background consolidation loop via Gibbs distribution

Requires statistical_mechanics module to be available.

Author: Claude Code (Elegance Pass Refactoring - Phase 5)
Date: 2025-11-22
"""

from __future__ import annotations

import asyncio
import logging
import time
import numpy as np
from datetime import datetime
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from hololoom.weaving_orchestrator import WeavingOrchestrator

from hololoom.core.protocols.types import MemoryShard

# Check if statistical mechanics is available
try:
    from hololoom.physics.statistical_mechanics import Microstate, Macrostate
    STATISTICAL_MECHANICS_AVAILABLE = True
except ImportError:
    STATISTICAL_MECHANICS_AVAILABLE = False
    Microstate = None
    Macrostate = None


logger = logging.getLogger(__name__)


def shards_to_microstates(
    orchestrator: 'WeavingOrchestrator',
    shards: List[MemoryShard]
) -> List['Microstate']:
    """
    Convert memory shards to microstates for statistical mechanics.

    Extracts embeddings from memory shards and wraps them in Microstate objects
    for statistical mechanics processing.

    Args:
        orchestrator: The WeavingOrchestrator instance
        shards: Memory shards to convert

    Returns:
        List of Microstate objects (empty list if stat_mech unavailable)

    Example:
        >>> shards = [
        ...     MemoryShard(id="1", text="...", embedding=[0.1, 0.2, ...]),
        ...     MemoryShard(id="2", text="...", embedding=[0.3, 0.4, ...])
        ... ]
        >>> microstates = shards_to_microstates(orchestrator, shards)
        >>> len(microstates)
        2

    Note:
        - Skips shards without embeddings
        - Logs warnings for shards without embeddings
        - Returns empty list if statistical_mechanics module unavailable
    """
    if not STATISTICAL_MECHANICS_AVAILABLE:
        return []

    microstates = []
    for shard in shards:
        # Get embedding (use first scale if available)
        state_vector = None
        if hasattr(shard, 'embedding') and shard.embedding is not None:
            state_vector = shard.embedding
        elif hasattr(shard, 'embeddings') and shard.embeddings:
            # Use first available embedding
            state_vector = shard.embeddings[0] if isinstance(shard.embeddings, list) else shard.embeddings
        elif hasattr(shard, 'metadata') and 'embedding' in shard.metadata:
            # Check metadata for embedding
            state_vector = shard.metadata['embedding']

        if state_vector is None:
            # Skip shards without embeddings
            logger.warning(f"Shard {shard.id} has no embedding, skipping consolidation")
            continue

        microstate = Microstate(
            id=shard.id,
            state_vector=np.array(state_vector),
            energy=0.0,  # Will be computed by engine
            metadata={
                'source': shard.metadata.get('source', 'unknown') if hasattr(shard, 'metadata') else 'unknown',
                'content_preview': shard.text[:100] if hasattr(shard, 'text') else ''
            }
        )
        microstates.append(microstate)

    return microstates


def macrostates_to_shards(
    orchestrator: 'WeavingOrchestrator',
    macrostates: List['Macrostate']
) -> List[MemoryShard]:
    """
    Convert consolidated macrostates back to memory shards.

    Each macrostate becomes a consolidated shard representing the cluster.
    Merges content from multiple microstates into a single representative shard.

    Args:
        orchestrator: The WeavingOrchestrator instance
        macrostates: Consolidated macrostates

    Returns:
        List of consolidated memory shards

    Example:
        >>> macrostates = [Macrostate(label="cluster_0", microstates=[...])]
        >>> consolidated = macrostates_to_shards(orchestrator, macrostates)
        >>> len(consolidated)
        1
        >>> consolidated[0].id
        'cluster_0_consolidated'

    Note:
        - Uses first microstate as representative (could compute centroid)
        - Merges content from up to 5 microstates per cluster
        - Includes statistical mechanics metadata (energy, entropy, etc.)
    """
    consolidated_shards = []

    for i, macro in enumerate(macrostates):
        # Get representative microstate (centroid)
        if not macro.microstates:
            continue

        # Use first microstate as representative (could compute centroid instead)
        representative = macro.microstates[0]

        # Merge content from all microstates
        merged_content = f"[Consolidated Cluster {i}]\n\n"
        for j, micro in enumerate(macro.microstates[:5]):  # Limit to first 5
            preview = micro.metadata.get('content_preview', '')
            merged_content += f"{j+1}. {preview}...\n"

        if len(macro.microstates) > 5:
            merged_content += f"... and {len(macro.microstates) - 5} more memories\n"

        # Create consolidated shard
        consolidated_shard = MemoryShard(
            id=f"{macro.label}_consolidated",
            text=merged_content,
            metadata={
                'embedding': representative.state_vector.tolist(),
                'source': 'statistical_mechanics_consolidation',
                'cluster_label': macro.label,
                'cluster_size': len(macro.microstates),
                'average_energy': macro.average_energy,
                'entropy': macro.entropy,
                'free_energy': macro.free_energy,
                'order_parameter': macro.order_parameter,
                'member_ids': [m.id for m in macro.microstates],
                'consolidation_timestamp': datetime.now().isoformat()
            }
        )
        consolidated_shards.append(consolidated_shard)

    return consolidated_shards


async def background_consolidation_loop(orchestrator: 'WeavingOrchestrator') -> None:
    """
    Background loop for periodic memory consolidation.

    Runs every consolidation_interval seconds, consolidates memory shards
    via statistical mechanics Gibbs distribution, and detects phase transitions.

    Args:
        orchestrator: The WeavingOrchestrator instance

    Example:
        >>> orchestrator = WeavingOrchestrator(...)
        >>> task = asyncio.create_task(background_consolidation_loop(orchestrator))
        >>> # Loop runs in background until orchestrator._closed = True

    Side Effects:
        - Periodically consolidates memory shards
        - Logs consolidation statistics
        - Detects and logs phase transitions
        - Anneals temperature over time

    Note:
        - Runs until orchestrator._closed is True
        - Handles CancelledError gracefully
        - Continues running despite non-fatal errors
        - Requires at least 3 shards to consolidate
    """
    logger.info(f"Starting background consolidation loop (interval={orchestrator.consolidation_interval}s)")

    while not orchestrator._closed:
        try:
            await asyncio.sleep(orchestrator.consolidation_interval)

            if orchestrator._closed:
                break

            logger.info("[CONSOLIDATION] Starting periodic memory consolidation...")

            # Get current shards
            if hasattr(orchestrator.yarn_graph, 'shards') and isinstance(orchestrator.yarn_graph.shards, dict):
                current_shards = list(orchestrator.yarn_graph.shards.values())
            elif hasattr(orchestrator.yarn_graph, 'shards') and isinstance(orchestrator.yarn_graph.shards, list):
                current_shards = orchestrator.yarn_graph.shards
            else:
                current_shards = orchestrator.shards

            if len(current_shards) < 3:
                logger.debug(f"[CONSOLIDATION] Skipping (only {len(current_shards)} shards, need at least 3)")
                continue

            # Convert to microstates
            microstates = shards_to_microstates(orchestrator, current_shards)

            if len(microstates) < 3:
                logger.debug(f"[CONSOLIDATION] Skipping (only {len(microstates)} valid microstates)")
                continue

            # Consolidate via Gibbs distribution
            consolidation_start = time.time()
            macrostates = orchestrator.stat_mech_engine.consolidate_memories(
                microstates,
                num_clusters=None  # Auto-detect optimal clusters
            )
            consolidation_time = (time.time() - consolidation_start) * 1000

            logger.info(f"[CONSOLIDATION] Consolidated {len(microstates)} → {len(macrostates)} clusters ({consolidation_time:.1f}ms)")

            # Detect phase transitions
            transition = orchestrator.stat_mech_engine.detect_phase_transition()

            if transition:
                logger.info(
                    f"[PHASE TRANSITION] Detected at T={transition.critical_temperature:.2f}"
                )
                logger.info(
                    f"  Type: {transition.phase_type.value}, "
                    f"Order: {transition.order_before:.3f} → {transition.order_after:.3f}"
                )
                # Could trigger actions here (save checkpoint, notify, etc.)

            # Get statistics
            stats = orchestrator.stat_mech_engine.get_statistics()
            logger.info(
                f"[STATISTICS] T={stats['temperature']:.2f}, "
                f"S={stats['entropy']:.3f}, "
                f"F={stats['free_energy']:.3f}"
            )

            # Replace shards with consolidated macrostates (optional - could keep both)
            # For now, we keep original shards and just log consolidation results
            # In production, you might want to:
            # 1. Archive original shards
            # 2. Replace with consolidated shards
            # 3. Update yarn_graph

            # Example: Replace shards (commented out for safety)
            # consolidated_shards = macrostates_to_shards(orchestrator, macrostates)
            # if hasattr(orchestrator.yarn_graph, 'shards'):
            #     orchestrator.yarn_graph.shards = {s.id: s for s in consolidated_shards}
            # orchestrator.shards = consolidated_shards

            # Cool temperature (simulated annealing)
            orchestrator.stat_mech_engine.anneal()

        except asyncio.CancelledError:
            logger.info("[CONSOLIDATION] Background consolidation cancelled")
            break
        except Exception as e:
            logger.error(f"[CONSOLIDATION] Error during consolidation: {e}", exc_info=True)
            # Continue running despite errors

    logger.info("[CONSOLIDATION] Background consolidation loop stopped")
