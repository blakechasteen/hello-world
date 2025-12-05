"""
DHT Routing - Kademlia for capability-based discovery.

O(log n) routing at million-node scale.
No central registry. Just math.
"""

from __future__ import annotations

import asyncio
import heapq
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .identity import node_id_distance, node_id_prefix_length
from .protocols import RoutingProtocol
from .types import Capability, FederationNode, Query, Response

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  K-BUCKET - Kademlia routing table entry
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class KBucket:
    """
    A k-bucket holds nodes at a specific distance range.

    In Kademlia, the routing table is divided into buckets.
    Bucket i contains nodes with distance 2^i to 2^(i+1)-1.
    """

    k: int = 20                               # Max nodes per bucket
    nodes: List[FederationNode] = field(default_factory=list)
    last_update: float = 0.0

    def add(self, node: FederationNode) -> Optional[FederationNode]:
        """
        Add node to bucket.

        Returns:
            Evicted node if bucket was full, else None
        """
        # Check if already present
        for i, existing in enumerate(self.nodes):
            if existing.node_id == node.node_id:
                # Move to end (most recently seen)
                self.nodes.pop(i)
                self.nodes.append(node)
                return None

        # Add new node
        if len(self.nodes) < self.k:
            self.nodes.append(node)
            return None

        # Bucket full - return least recently seen for ping check
        return self.nodes[0]

    def remove(self, node_id: str) -> bool:
        """Remove node from bucket."""
        for i, node in enumerate(self.nodes):
            if node.node_id == node_id:
                self.nodes.pop(i)
                return True
        return False

    def get(self, node_id: str) -> Optional[FederationNode]:
        """Get node by ID."""
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def closest(self, target: str, n: int) -> List[FederationNode]:
        """Get n closest nodes to target."""
        sorted_nodes = sorted(
            self.nodes,
            key=lambda x: node_id_distance(x.node_id, target),
        )
        return sorted_nodes[:n]

    def __len__(self) -> int:
        return len(self.nodes)


# ═══════════════════════════════════════════════════════════════════════════
#  ROUTING TABLE - Collection of k-buckets
# ═══════════════════════════════════════════════════════════════════════════


class RoutingTable:
    """
    Kademlia routing table.

    160 buckets (one per bit of node ID).
    Each bucket holds up to k nodes at that distance.
    """

    def __init__(
        self,
        local_id: str,
        k: int = 20,
    ):
        self._local_id = local_id
        self._k = k
        self._buckets = [KBucket(k=k) for _ in range(160)]

        # Capability index for fast lookup
        self._by_capability: Dict[Capability, Set[str]] = defaultdict(set)

    def _bucket_index(self, node_id: str) -> int:
        """Get bucket index for a node ID."""
        distance = node_id_distance(self._local_id, node_id)
        if distance == 0:
            return 0
        return min(distance.bit_length() - 1, 159)

    def add(self, node: FederationNode) -> Optional[FederationNode]:
        """Add node to routing table."""
        if node.node_id == self._local_id:
            return None  # Don't add ourselves

        bucket_idx = self._bucket_index(node.node_id)
        evicted = self._buckets[bucket_idx].add(node)

        # Update capability index
        for cap in node.capabilities:
            self._by_capability[cap].add(node.node_id)

        return evicted

    def remove(self, node_id: str) -> bool:
        """Remove node from routing table."""
        bucket_idx = self._bucket_index(node_id)
        success = self._buckets[bucket_idx].remove(node_id)

        if success:
            # Clean up capability index
            for cap_set in self._by_capability.values():
                cap_set.discard(node_id)

        return success

    def get(self, node_id: str) -> Optional[FederationNode]:
        """Get node by ID."""
        bucket_idx = self._bucket_index(node_id)
        return self._buckets[bucket_idx].get(node_id)

    def closest(self, target: str, n: int = 20) -> List[FederationNode]:
        """
        Find n closest nodes to target.

        This is the core Kademlia lookup operation.
        """
        # Collect candidates from all buckets
        candidates: List[Tuple[int, FederationNode]] = []

        for bucket in self._buckets:
            for node in bucket.nodes:
                dist = node_id_distance(node.node_id, target)
                heapq.heappush(candidates, (dist, node))

        # Return n closest
        result = []
        seen = set()
        while candidates and len(result) < n:
            _, node = heapq.heappop(candidates)
            if node.node_id not in seen:
                seen.add(node.node_id)
                result.append(node)

        return result

    def with_capabilities(
        self,
        capabilities: Set[Capability],
        min_trust: float = 0.0,
    ) -> List[FederationNode]:
        """Find nodes with all required capabilities."""
        # Start with nodes that have any required capability
        candidate_ids: Optional[Set[str]] = None

        for cap in capabilities:
            cap_ids = self._by_capability.get(cap, set())
            if candidate_ids is None:
                candidate_ids = cap_ids.copy()
            else:
                candidate_ids &= cap_ids

        if not candidate_ids:
            return []

        # Get nodes and filter by trust
        result = []
        for node_id in candidate_ids:
            node = self.get(node_id)
            if node and node.trust_score >= min_trust:
                result.append(node)

        return sorted(result, key=lambda x: -x.trust_score)

    @property
    def size(self) -> int:
        """Total nodes in routing table."""
        return sum(len(b) for b in self._buckets)


# ═══════════════════════════════════════════════════════════════════════════
#  DHT ROUTER - Query routing
# ═══════════════════════════════════════════════════════════════════════════


class KademliaRouter:
    """
    Kademlia-style DHT for capability-based routing.

    Usage:
        router = KademliaRouter(local_node)

        # Find capable nodes
        nodes = await router.find_nodes(
            capabilities={Capability.MEDICAL},
            min_trust=0.7,
            limit=5
        )

        # Route query
        responses = await router.route(query, nodes)
    """

    def __init__(
        self,
        local_node: FederationNode,
        *,
        k: int = 20,                          # Bucket size
        alpha: int = 3,                       # Parallel lookups
        lookup_timeout_ms: int = 5000,
    ):
        self._local = local_node
        self._table = RoutingTable(local_node.node_id, k=k)
        self._alpha = alpha
        self._lookup_timeout = lookup_timeout_ms / 1000

        # Cached lookups
        self._lookup_cache: Dict[str, List[FederationNode]] = {}

    # ───────────────────────────────────────────────────────────────────────
    #  NODE DISCOVERY
    # ───────────────────────────────────────────────────────────────────────

    async def find_nodes(
        self,
        capabilities: Set[Capability],
        *,
        min_trust: float = 0.7,
        guild: Optional[str] = None,
        limit: int = 10,
    ) -> List[FederationNode]:
        """
        Find nodes with required capabilities.

        Args:
            capabilities: Required capabilities
            min_trust: Minimum trust score (0.0-1.0)
            guild: Preferred guild (optional)
            limit: Maximum nodes to return

        Returns:
            List of capable nodes, sorted by trust score
        """
        # First check local routing table
        candidates = self._table.with_capabilities(capabilities, min_trust)

        # Filter by guild if specified
        if guild:
            candidates = [n for n in candidates if guild in n.guilds]

        # If we have enough, return
        if len(candidates) >= limit:
            return candidates[:limit]

        # Otherwise, do iterative lookup
        # TODO: Implement full iterative Kademlia lookup
        # For now, return what we have

        return candidates[:limit]

    async def find_node(self, node_id: str) -> Optional[FederationNode]:
        """Find a specific node by ID."""
        # Check local table first
        node = self._table.get(node_id)
        if node:
            return node

        # Do iterative lookup
        return await self._iterative_find_node(node_id)

    async def _iterative_find_node(
        self,
        target: str,
    ) -> Optional[FederationNode]:
        """
        Iterative Kademlia node lookup.

        1. Start with alpha closest nodes from local table
        2. Query each for their closest nodes
        3. Repeat until convergence
        """
        # Get initial set of closest nodes
        closest = self._table.closest(target, self._alpha)

        if not closest:
            return None

        # Track queried and pending nodes
        queried: Set[str] = set()
        found: Optional[FederationNode] = None

        # Priority queue: (distance, node)
        to_query: List[Tuple[int, FederationNode]] = [
            (node_id_distance(n.node_id, target), n) for n in closest
        ]
        heapq.heapify(to_query)

        while to_query:
            # Get alpha closest unqueried nodes
            batch = []
            while to_query and len(batch) < self._alpha:
                dist, node = heapq.heappop(to_query)
                if node.node_id not in queried:
                    batch.append(node)

            if not batch:
                break

            # Query in parallel
            tasks = [self._query_find_node(n, target) for n in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for node, result in zip(batch, results):
                queried.add(node.node_id)

                if isinstance(result, Exception):
                    continue

                # Check if we found the target
                for found_node in result:
                    if found_node.node_id == target:
                        return found_node

                    # Add to table and queue
                    self._table.add(found_node)
                    dist = node_id_distance(found_node.node_id, target)
                    if found_node.node_id not in queried:
                        heapq.heappush(to_query, (dist, found_node))

        return found

    async def _query_find_node(
        self,
        node: FederationNode,
        target: str,
    ) -> List[FederationNode]:
        """Query a node for nodes close to target."""
        # TODO: Implement actual RPC
        return []

    # ───────────────────────────────────────────────────────────────────────
    #  QUERY ROUTING
    # ───────────────────────────────────────────────────────────────────────

    async def route(
        self,
        query: Query,
        nodes: List[FederationNode],
    ) -> List[Response]:
        """
        Route query to nodes and collect responses.

        Sends query to all nodes in parallel, returns all responses.
        """
        if not nodes:
            return []

        # Query all nodes in parallel
        tasks = [self._query_node(query, node) for node in nodes]

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=query.timeout_ms / 1000,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Query {query.request_id[:8]} timed out")
            results = []

        # Filter successful responses
        responses = []
        for result in results:
            if isinstance(result, Response):
                responses.append(result)
            elif isinstance(result, Exception):
                logger.debug(f"Node query failed: {result}")

        return responses

    async def _query_node(
        self,
        query: Query,
        node: FederationNode,
    ) -> Response:
        """Send query to a single node."""
        # TODO: Implement actual RPC
        # For now, return placeholder
        import time

        return Response(
            text="[Placeholder response]",
            request_id=query.request_id,
            responder=node.node_id,
            confidence=0.0,
            latency_ms=0.0,
        )

    # ───────────────────────────────────────────────────────────────────────
    #  ANNOUNCEMENTS
    # ───────────────────────────────────────────────────────────────────────

    async def announce(self, node: FederationNode) -> None:
        """
        Announce node capabilities to DHT.

        Call this when joining or when capabilities change.
        """
        # Add to local table
        self._table.add(node)

        # TODO: Propagate to k closest nodes
        logger.info(f"Announced {node.node_id[:8]}... with {len(node.capabilities)} capabilities")

    async def refresh_buckets(self) -> None:
        """
        Refresh stale buckets.

        For each bucket that hasn't been touched recently,
        do a lookup for a random ID in that range.
        """
        # TODO: Implement bucket refresh
        pass

    # ───────────────────────────────────────────────────────────────────────
    #  LOAD BALANCING
    # ───────────────────────────────────────────────────────────────────────

    def select_nodes(
        self,
        candidates: List[FederationNode],
        n: int,
        *,
        strategy: str = "weighted",
    ) -> List[FederationNode]:
        """
        Select n nodes from candidates for load balancing.

        Strategies:
        - "round_robin": Simple rotation
        - "weighted": Weight by trust score
        - "random": Random selection
        """
        if len(candidates) <= n:
            return candidates

        if strategy == "random":
            return random.sample(candidates, n)

        elif strategy == "weighted":
            # Weighted selection by trust score
            weights = [node.trust_score for node in candidates]
            total = sum(weights)
            if total == 0:
                return random.sample(candidates, n)

            selected = []
            remaining = candidates.copy()
            remaining_weights = weights.copy()

            for _ in range(n):
                if not remaining:
                    break

                # Weighted random selection
                r = random.uniform(0, sum(remaining_weights))
                cumsum = 0
                for i, (node, w) in enumerate(zip(remaining, remaining_weights)):
                    cumsum += w
                    if r <= cumsum:
                        selected.append(node)
                        remaining.pop(i)
                        remaining_weights.pop(i)
                        break

            return selected

        else:  # round_robin
            return candidates[:n]

    # ───────────────────────────────────────────────────────────────────────
    #  PROPERTIES
    # ───────────────────────────────────────────────────────────────────────

    @property
    def table_size(self) -> int:
        """Number of nodes in routing table."""
        return self._table.size

    def add_node(self, node: FederationNode) -> None:
        """Add node to routing table."""
        self._table.add(node)
