#!/usr/bin/env python3
"""
Multipass Memory Retrieval
===========================

Multi-pass memory retrieval with Matryoshka importance gating.

Extracted from weaving_orchestrator.py (November 2025 - Elegance Pass)
Original location: lines 503-650, 2149-2199 (~197 lines total, 2 methods)

This module handles:
- Recursive gated multipass memory crawling
- Progressive importance thresholding (0.6 → 0.75 → 0.85 → 0.9)
- Graph expansion via entity relationships
- Backend memory querying with result conversion

Author: Claude Code (Elegance Pass Refactoring - Phase 6)
Date: 2025-11-22
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hololoom.core.orchestrator.weaving_orchestrator import WeavingOrchestrator

from hololoom.protocols.types import ComplexityLevel, MemoryShard, ProvenanceTrace, Query

logger = logging.getLogger(__name__)


async def multipass_memory_crawl(
    orchestrator: WeavingOrchestrator,
    query: Query,
    complexity: ComplexityLevel,
    trace: ProvenanceTrace | None = None
) -> list[Any]:
    """
    Recursive gated multipass memory crawling with Matryoshka importance gating.

    **Crawling Strategy:**
    1. **Gated Retrieval**: Start broad (low threshold), progressively focus (high threshold)
    2. **Graph Traversal**: Follow entity relationships for deeper exploration
    3. **Matryoshka Gating**: Increase importance thresholds by depth (0.6 → 0.75 → 0.85 → 0.9)
    4. **Multipass Fusion**: Intelligent deduplication and composite scoring

    **Complexity Scaling:**
    - LITE (1 pass): threshold=0.7, limit=5, no graph traversal
    - FAST (2 passes): thresholds=[0.6, 0.75], limits=[8, 12], light graph
    - FULL (3 passes): thresholds=[0.6, 0.75, 0.85], limits=[12, 20, 15], full graph
    - RESEARCH (4 passes): thresholds=[0.5, 0.65, 0.8, 0.9], limits=[20, 30, 25, 15], aggressive

    Args:
        orchestrator: The WeavingOrchestrator instance
        query: User query
        complexity: Assessed complexity level
        trace: Optional provenance trace

    Returns:
        List of retrieved memory items with composite scores

    Example:
        >>> memories = await multipass_memory_crawl(
        ...     orchestrator,
        ...     Query(text="What is Thompson Sampling?"),
        ...     ComplexityLevel.FAST,
        ...     trace=trace
        ... )
        >>> len(memories)
        12

    Note:
        - Uses orchestrator._crawl_config for complexity-specific settings
        - Supports graph expansion via memory.get_related() if available
        - Returns items sorted by composite score (descending)
    """
    crawl_start = time.perf_counter()
    config = orchestrator._crawl_config[complexity]

    all_results = {}  # item_id -> {item, score, depth, sources}
    seen_ids = set()

    if trace:
        trace.add_shuttle_event(
            "crawl_start",
            f"Starting {config['passes']}-pass crawl with thresholds {config['thresholds']}",
            {'complexity': complexity.name, 'config': config}
        )

    # Multi-pass retrieval with progressive gating
    for pass_idx in range(config['passes']):
        threshold = config['thresholds'][pass_idx]
        limit = config['limits'][pass_idx]

        pass_start = time.perf_counter()

        # Initial retrieval from memory backend
        if orchestrator.memory:
            try:
                # Use memory backend's recall method
                # Note: Threshold is handled by the backend based on relevance scoring
                from hololoom.memory.protocol import MemoryQuery
                mem_query = MemoryQuery(
                    text=query.text,
                    limit=limit
                )
                result = await orchestrator.memory.recall(mem_query, limit=limit)

                # Process results
                for idx, (memory, score) in enumerate(zip(result.memories, result.scores)):
                    if memory.id not in seen_ids:
                        all_results[memory.id] = {
                            'item': memory,
                            'score': score * (1.0 / (pass_idx + 1)),  # Decay by depth
                            'depth': pass_idx,
                            'sources': [f'pass_{pass_idx}']
                        }
                        seen_ids.add(memory.id)
                    else:
                        # Boost score for items found in multiple passes
                        all_results[memory.id]['score'] += score * 0.3
                        all_results[memory.id]['sources'].append(f'pass_{pass_idx}')

                if trace:
                    trace.add_shuttle_event(
                        f"crawl_pass_{pass_idx}",
                        f"Retrieved {len(result.memories)} items (threshold={threshold})",
                        {
                            'pass': pass_idx,
                            'threshold': threshold,
                            'limit': limit,
                            'retrieved': len(result.memories),
                            'time_ms': (time.perf_counter() - pass_start) * 1000
                        }
                    )

            except Exception as e:
                logger.warning(f"Pass {pass_idx} failed: {e}")
                if trace:
                    trace.add_shuttle_event(
                        f"crawl_pass_{pass_idx}_error",
                        f"Retrieval failed: {str(e)}",
                        {'error': str(e)}
                    )

        # Graph expansion (if enabled for this complexity)
        if config['graph_expansion'] and pass_idx < config['passes'] - 1:
            # Expand from top results of this pass
            expand_count = min(3, len(all_results))  # Expand from top 3
            expanded_ids = list(all_results.keys())[:expand_count]

            for item_id in expanded_ids:
                # Try to get related items (graph traversal)
                # Note: This requires the memory backend to support get_related()
                if hasattr(orchestrator.memory, 'get_related'):
                    try:
                        related = await orchestrator.memory.get_related(item_id, limit=5)
                        for rel_item in related:
                            if rel_item.id not in seen_ids:
                                # Related items get slightly lower score
                                all_results[rel_item.id] = {
                                    'item': rel_item,
                                    'score': 0.7 * (1.0 / (pass_idx + 2)),
                                    'depth': pass_idx + 1,
                                    'sources': [f'graph_expansion_from_{item_id}']
                                }
                                seen_ids.add(rel_item.id)
                    except (AttributeError, Exception):
                        # Backend doesn't support graph traversal or error occurred
                        pass

    # Sort by composite score
    ranked_results = sorted(
        all_results.values(),
        key=lambda x: x['score'],
        reverse=True
    )

    crawl_time_ms = (time.perf_counter() - crawl_start) * 1000

    if trace:
        trace.add_shuttle_event(
            "crawl_complete",
            f"Crawl complete: {len(ranked_results)} unique items",
            {
                'total_items': len(ranked_results),
                'passes_completed': config['passes'],
                'time_ms': crawl_time_ms,
                'avg_time_per_pass_ms': crawl_time_ms / config['passes']
            }
        )

    # Return just the items (without metadata for now)
    return [r['item'] for r in ranked_results]


async def query_memory_backend(
    orchestrator: WeavingOrchestrator,
    query_text: str,
    limit: int = 5
) -> list[MemoryShard]:
    """
    Query the unified memory backend and convert results to MemoryShards.

    Provides a simple interface to query the memory backend and get back
    MemoryShard objects for use in the orchestrator.

    Args:
        orchestrator: The WeavingOrchestrator instance
        query_text: Query string
        limit: Maximum number of results

    Returns:
        List of MemoryShard objects

    Example:
        >>> shards = await query_memory_backend(
        ...     orchestrator,
        ...     "Thompson Sampling",
        ...     limit=10
        ... )
        >>> len(shards)
        10
        >>> shards[0].text
        'Thompson Sampling is a Bayesian approach...'

    Note:
        - Returns empty list if no memory backend configured
        - Converts backend Memory objects to MemoryShards
        - Handles errors gracefully with logging
    """
    if not orchestrator.memory:
        return []

    try:
        # Import protocol types
        from hololoom.memory.protocol import MemoryQuery

        # Create query
        mem_query = MemoryQuery(
            text=query_text,
            user_id=getattr(orchestrator.cfg, 'user_id', 'default'),
            limit=limit
        )

        # Query backend
        result = await orchestrator.memory.recall(mem_query)

        # Convert backend Memory objects to MemoryShards
        shards = []
        for mem in result.memories:
            shard = MemoryShard(
                id=mem.id,
                text=mem.text,
                episode=mem.context.get('episode', 'default'),
                entities=mem.context.get('entities', []),
                motifs=mem.metadata.get('motifs', []),
                metadata=mem.metadata
            )
            shards.append(shard)

        logger.debug(f"Retrieved {len(shards)} shards from memory backend")
        return shards

    except Exception as e:
        logger.error(f"Failed to query memory backend: {e}")
        return []
