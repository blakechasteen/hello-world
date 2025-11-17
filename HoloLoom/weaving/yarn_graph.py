#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Yarn Graph - Memory Thread Selection
======================================

Simple in-memory implementation of Yarn Graph for thread storage and selection.

The Yarn Graph represents discrete symbolic memory that is "tensioned" into
continuous Warp Space for tensor operations.

**Extracted: November 2025** - Refactored from weaving_orchestrator.py
**Lines: 69** (reduced from 3,476-line monolith)

In production, this would be backed by Neo4j or NetworkX MultiDiGraph.
This simple implementation uses dict-based storage.

Author: Claude Code (refactoring pass)
"""

from __future__ import annotations

import logging
from typing import List, Dict

from HoloLoom.Documentation.types import Query, MemoryShard
from HoloLoom.chrono.trigger import TemporalWindow


class YarnGraph:
    """
    Simple in-memory Yarn Graph for thread storage.

    The Yarn Graph is the persistent symbolic memory layer in HoloLoom's
    weaving architecture. It stores discrete "threads" (memory shards) that
    are selected based on temporal windows and query relevance.

    **Lifecycle in Weaving Cycle:**
    1. Chrono Trigger creates TemporalWindow
    2. YarnGraph selects threads matching temporal bounds
    3. Threads are "tensioned" into WarpSpace for continuous operations
    4. After computation, results "detension" back to discrete form

    **Production Implementation:**
    In production, this should be backed by:
    - Neo4j: For persistent graph storage with CYPHER queries
    - NetworkX MultiDiGraph: For in-memory graph operations
    - Temporal indexing: For efficient time-based filtering

    **Current Implementation:**
    Simple dict-based storage with basic temporal filtering.

    Example:
        >>> shards = create_memory_shards()
        >>> yarn = YarnGraph(shards)
        >>> window = TemporalWindow(start_time=0, end_time=100)
        >>> threads = yarn.select_threads(window, query)

    Args:
        shards: List of memory shards to store in the graph
    """

    def __init__(self, shards: List[MemoryShard]):
        """
        Initialize Yarn Graph with memory shards.

        Args:
            shards: List of MemoryShard objects to store
        """
        self.shards: Dict[str, MemoryShard] = {shard.id: shard for shard in shards}
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"YarnGraph initialized with {len(shards)} threads")

    def select_threads(
        self,
        temporal_window: TemporalWindow,
        query: Query
    ) -> List[MemoryShard]:
        """
        Select threads based on temporal window and query relevance.

        **Current Implementation:**
        Returns all shards (simple fallback). In production, would filter by:
        - Temporal window bounds (start_time, end_time)
        - Recency weighting (decay over time)
        - Episode filter (specific memory episodes)
        - Query relevance (semantic similarity)

        **Production Enhancement:**
        ```python
        # Filter by temporal bounds
        threads = [
            s for s in self.shards.values()
            if temporal_window.start_time <= s.timestamp <= temporal_window.end_time
        ]

        # Apply recency weighting
        threads = sorted(
            threads,
            key=lambda s: s.timestamp,
            reverse=True
        )

        # Filter by semantic relevance
        threads = [
            s for s in threads
            if cosine_similarity(s.embedding, query.embedding) > threshold
        ]
        ```

        Args:
            temporal_window: Time bounds for thread selection
            query: Query for relevance filtering

        Returns:
            List of relevant memory shards (threads)
        """
        # Simple implementation: return all threads
        threads = list(self.shards.values())
        self.logger.debug(f"Selected {len(threads)} threads from YarnGraph")
        return threads

    def add_thread(self, shard: MemoryShard) -> None:
        """
        Add a new thread (memory shard) to the graph.

        Args:
            shard: Memory shard to add
        """
        self.shards[shard.id] = shard
        self.logger.debug(f"Added thread {shard.id} to YarnGraph")

    def remove_thread(self, shard_id: str) -> None:
        """
        Remove a thread from the graph.

        Args:
            shard_id: ID of shard to remove
        """
        if shard_id in self.shards:
            del self.shards[shard_id]
            self.logger.debug(f"Removed thread {shard_id} from YarnGraph")
        else:
            self.logger.warning(f"Thread {shard_id} not found in YarnGraph")

    def get_thread_count(self) -> int:
        """
        Get the total number of threads in the graph.

        Returns:
            Number of threads (shards) in the graph
        """
        return len(self.shards)
