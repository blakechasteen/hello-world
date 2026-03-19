#!/usr/bin/env python3
"""
Federated RAG - Agentic Knowledge Queries
==========================================

Part of Phase 4: Open & Safe Agent Federation (December 2025)
Day 18: Agentic RAG

Federated knowledge queries across multiple nodes with MI-weighted merging.
Implements a local-first pattern: query local RAG first, then federate
only if confidence is below threshold.

Flow:
    1. Local RAG query first
    2. If confidence < threshold (0.7), query federation
    3. Parallel node queries via RPC
    4. Merge results with MI deduplication
    5. Return combined RAGResult

Author: Claude Code (Phase 4 - December 2025)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from hololoom.federation.result_merger import (
    TRUST_WEIGHTS,
    MergedRAGResult,
    NodeRAGResult,
    RAGResultMerger,
    create_result_merger,
)
from hololoom.federation.types import (
    Capability,
    FederationNode,
    GuildTrustLevel,
)

if TYPE_CHECKING:
    from hololoom.federation.rate_limiter import FederatedRateLimiter
    from hololoom.federation.safety import FederationSafetyGate
    from hololoom.federation.wire_protocol import JSONRPCBuilder, RPCRequest, RPCResponse
    from hololoom.rag.simple_rag import RAGResult, SimpleRAG

logger = logging.getLogger(__name__)


# ============================================================================
#  Constants
# ============================================================================

# Default threshold for federation fallback
DEFAULT_FEDERATION_THRESHOLD = 0.7

# Maximum nodes to query in parallel
DEFAULT_MAX_PARALLEL_NODES = 5

# Default timeout for federation queries (seconds)
DEFAULT_FEDERATION_TIMEOUT = 10.0

# RPC method for RAG recall
RAG_RECALL_METHOD = "hololoom.rag.recall"


# ============================================================================
#  Helper Functions
# ============================================================================

def trust_score_to_level(trust_score: float) -> GuildTrustLevel:
    """
    Convert a trust_score (float 0.0-1.0) to GuildTrustLevel enum.

    Mapping:
        - < 0.4: STARTER
        - 0.4 - 0.7: ESTABLISHED
        - >= 0.7: VETERAN

    Args:
        trust_score: Float trust score (0.0-1.0)

    Returns:
        Corresponding GuildTrustLevel
    """
    if trust_score >= 0.7:
        return GuildTrustLevel.VETERAN
    elif trust_score >= 0.4:
        return GuildTrustLevel.ESTABLISHED
    else:
        return GuildTrustLevel.STARTER


def trust_level_to_score(trust_level: GuildTrustLevel) -> float:
    """
    Convert a GuildTrustLevel enum to trust_score (float 0.0-1.0).

    Mapping:
        - STARTER: 0.3
        - ESTABLISHED: 0.6
        - VETERAN: 1.0

    Args:
        trust_level: GuildTrustLevel enum

    Returns:
        Corresponding float trust score
    """
    return TRUST_WEIGHTS.get(trust_level, 0.3)


# ============================================================================
#  Data Classes
# ============================================================================

@dataclass
class FederatedRAGConfig:
    """Configuration for federated RAG queries."""

    federation_threshold: float = DEFAULT_FEDERATION_THRESHOLD
    max_parallel_nodes: int = DEFAULT_MAX_PARALLEL_NODES
    federation_timeout: float = DEFAULT_FEDERATION_TIMEOUT
    enable_safety_checks: bool = True
    enable_rate_limiting: bool = True
    min_node_trust: GuildTrustLevel = GuildTrustLevel.STARTER
    include_local_in_merge: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.federation_threshold <= 1.0:
            raise ValueError(f"federation_threshold must be 0.0-1.0, got {self.federation_threshold}")
        if self.max_parallel_nodes < 1:
            raise ValueError(f"max_parallel_nodes must be >= 1, got {self.max_parallel_nodes}")
        if self.federation_timeout <= 0:
            raise ValueError(f"federation_timeout must be > 0, got {self.federation_timeout}")


@dataclass
class FederatedRAGResult:
    """Result from a federated RAG query."""

    response: str
    sources: list[str]
    confidence: float
    epistemic_confidence: float | None = None
    reasoning_mode: str = "federated"
    local_confidence: float | None = None
    federation_used: bool = False
    nodes_queried: list[str] = field(default_factory=list)
    nodes_responded: list[str] = field(default_factory=list)
    merge_stats: dict[str, Any] | None = None
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_rag_result(self) -> RAGResult:
        """Convert to standard RAGResult for compatibility."""
        from hololoom.rag.simple_rag import RAGResult

        return RAGResult(
            response=self.response,
            sources=self.sources,
            confidence=self.confidence,
            epistemic_confidence=self.epistemic_confidence,
            reasoning_mode=self.reasoning_mode,
            metadata={
                **self.metadata,
                "local_confidence": self.local_confidence,
                "federation_used": self.federation_used,
                "nodes_queried": self.nodes_queried,
                "nodes_responded": self.nodes_responded,
                "merge_stats": self.merge_stats,
                "latency_ms": self.latency_ms,
            }
        )


# ============================================================================
#  Confidence Aggregator
# ============================================================================

class ConfidenceAggregator:
    """
    Aggregates confidence scores from multiple federation nodes.

    Uses trust-weighted averaging:
        confidence = Σ(conf × trust × source_count) / Σ(trust × source_count)

    Also calculates epistemic confidence based on agreement across nodes.
    """

    def __init__(self, trust_weights: dict[GuildTrustLevel, float] | None = None):
        """
        Initialize aggregator.

        Args:
            trust_weights: Custom trust level weights (uses TRUST_WEIGHTS if None)
        """
        self._trust_weights = trust_weights or TRUST_WEIGHTS

    def aggregate(self, results: list[NodeRAGResult]) -> float:
        """
        Aggregate confidence scores from multiple nodes.

        Formula: Σ(conf × trust × source_count) / Σ(trust × source_count)

        Args:
            results: List of node results

        Returns:
            Aggregated confidence score (0.0-1.0)
        """
        if not results:
            return 0.0

        numerator = 0.0
        denominator = 0.0

        for result in results:
            trust_weight = self._trust_weights.get(result.trust_level, 0.3)
            source_count = max(1, result.source_count())

            numerator += result.confidence * trust_weight * source_count
            denominator += trust_weight * source_count

        if denominator == 0:
            return 0.0

        return min(1.0, max(0.0, numerator / denominator))

    def calculate_epistemic(self, results: list[NodeRAGResult]) -> float:
        """
        Calculate epistemic confidence based on agreement across nodes.

        Higher agreement → higher epistemic confidence.

        Args:
            results: List of node results

        Returns:
            Epistemic confidence (0.0-1.0)
        """
        if not results:
            return 0.0

        if len(results) == 1:
            # Single node - use its confidence as epistemic
            return results[0].confidence * 0.7  # Discount for single source

        # Calculate variance in confidence scores
        confidences = [r.confidence for r in results]
        mean_conf = sum(confidences) / len(confidences)

        variance = sum((c - mean_conf) ** 2 for c in confidences) / len(confidences)
        std_dev = variance ** 0.5

        # Low variance = high agreement = high epistemic confidence
        # Map std_dev to epistemic: 0 std = 1.0, 0.5 std = 0.5, 1.0 std = 0.0
        agreement_score = max(0.0, 1.0 - std_dev * 2)

        # Combine agreement with mean confidence
        epistemic = 0.7 * mean_conf + 0.3 * agreement_score

        return min(1.0, max(0.0, epistemic))


# ============================================================================
#  Federated RAG
# ============================================================================

class FederatedRAG:
    """
    Federated RAG implementation with local-first query pattern.

    Coordinates multi-node knowledge queries with safety gating,
    rate limiting, and MI-weighted result merging.

    Example:
        >>> from hololoom.rag import SimpleRAG
        >>> from hololoom.federation import FederationSafetyGate
        >>>
        >>> async with SimpleRAG() as local_rag:
        ...     fed_rag = FederatedRAG(
        ...         local_rag=local_rag,
        ...         get_capable_nodes=lambda: capable_nodes,
        ...     )
        ...     result = await fed_rag.query("What is Thompson Sampling?")
        ...     print(f"Federation used: {result.federation_used}")
    """

    def __init__(
        self,
        local_rag: SimpleRAG | None = None,
        safety_gate: FederationSafetyGate | None = None,
        rate_limiter: FederatedRateLimiter | None = None,
        rpc_builder: JSONRPCBuilder | None = None,
        result_merger: RAGResultMerger | None = None,
        config: FederatedRAGConfig | None = None,
        get_capable_nodes: Callable[[], list[FederationNode]] | None = None,
        send_rpc: Callable[[RPCRequest, str], Coroutine[Any, Any, RPCResponse]] | None = None,
    ):
        """
        Initialize federated RAG.

        Args:
            local_rag: Local RAG instance for initial queries
            safety_gate: Federation safety gate for request validation
            rate_limiter: Rate limiter for federation requests
            rpc_builder: JSON-RPC builder for request construction
            result_merger: RAG result merger for MI-based deduplication
            config: Federated RAG configuration
            get_capable_nodes: Callback to get nodes with RAG capability
            send_rpc: Callback to send RPC request to a node
        """
        self._local_rag = local_rag
        self._safety_gate = safety_gate
        self._rate_limiter = rate_limiter
        self._rpc_builder = rpc_builder
        self._result_merger = result_merger or create_result_merger()
        self._config = config or FederatedRAGConfig()
        self._get_capable_nodes = get_capable_nodes
        self._send_rpc = send_rpc

        # Aggregator for confidence calculations
        self._aggregator = ConfidenceAggregator()

    async def query(
        self,
        text: str,
        k: int = 10,
        mode: str = "direct",
        min_confidence: float = 0.3,
        include_sources: bool = True,
        force_federation: bool = False,
    ) -> FederatedRAGResult:
        """
        Execute federated RAG query.

        Steps:
            1. Query local RAG first
            2. If confidence < threshold, query federation
            3. Parallel node queries via RPC
            4. Merge results with MI deduplication
            5. Return combined result

        Args:
            text: Query text
            k: Number of sources to retrieve
            mode: Reasoning mode (direct, verify, research, plan_execute)
            min_confidence: Minimum source confidence
            include_sources: Include source texts in result
            force_federation: Force federation even if local confidence is high

        Returns:
            FederatedRAGResult with merged knowledge
        """
        start_time = time.time()

        # Step 1: Local RAG query first
        local_result = await self._query_local(text, k, mode)
        local_confidence = local_result.confidence if local_result else 0.0

        # Check if federation is needed
        should_federate = (
            force_federation or
            local_confidence < self._config.federation_threshold
        )

        if not should_federate:
            # Local result is sufficient
            latency_ms = (time.time() - start_time) * 1000

            return FederatedRAGResult(
                response=local_result.response if local_result else "",
                sources=local_result.sources if local_result else [],
                confidence=local_confidence,
                epistemic_confidence=local_result.epistemic_confidence if local_result else None,
                reasoning_mode=mode,
                local_confidence=local_confidence,
                federation_used=False,
                latency_ms=latency_ms,
                metadata={"local_only": True},
            )

        # Step 2-3: Query federation nodes
        node_results = await self._query_federation(text, k, mode, min_confidence)

        # Include local result in merge if configured
        if self._config.include_local_in_merge and local_result:
            local_node_result = NodeRAGResult(
                node_id="local",
                response=local_result.response,
                sources=local_result.sources,
                confidence=local_result.confidence,
                trust_level=GuildTrustLevel.VETERAN,  # Trust local fully
                metadata={"is_local": True},
            )
            node_results.insert(0, local_node_result)

        # Step 4: Merge results with MI deduplication
        merged = self._result_merger.merge(
            results=node_results,
            query=text,
            max_sources=k,
        )

        # Step 5: Calculate aggregated confidence
        aggregated_confidence = self._aggregator.aggregate(node_results)
        epistemic_confidence = self._aggregator.calculate_epistemic(node_results)

        # Build response from top sources
        response = self._build_response(merged, local_result)

        latency_ms = (time.time() - start_time) * 1000

        return FederatedRAGResult(
            response=response,
            sources=merged.unique_sources,
            confidence=aggregated_confidence,
            epistemic_confidence=epistemic_confidence,
            reasoning_mode=mode,
            local_confidence=local_confidence,
            federation_used=True,
            nodes_queried=[r.node_id for r in node_results],
            nodes_responded=[r.node_id for r in node_results if r.confidence > 0],
            merge_stats={
                "original_count": merged.original_count,
                "merged_count": merged.merged_count,
                "redundancy_removed": merged.redundancy_removed,
                "avg_mi": merged.avg_mi,
            },
            latency_ms=latency_ms,
            metadata={
                "federation_threshold": self._config.federation_threshold,
                "force_federation": force_federation,
            },
        )

    async def _query_local(
        self,
        text: str,
        k: int,
        mode: str,
    ) -> RAGResult | None:
        """
        Query local RAG instance.

        Args:
            text: Query text
            k: Number of sources
            mode: Reasoning mode

        Returns:
            RAGResult from local RAG, or None if unavailable
        """
        if self._local_rag is None:
            return None

        try:
            result = await self._local_rag.query(text, k=k, mode=mode)
            return result
        except Exception as e:
            logger.warning(f"Local RAG query failed: {e}")
            return None

    async def _query_federation(
        self,
        text: str,
        k: int,
        mode: str,
        min_confidence: float,
    ) -> list[NodeRAGResult]:
        """
        Query federation nodes in parallel.

        Args:
            text: Query text
            k: Number of sources
            mode: Reasoning mode
            min_confidence: Minimum confidence threshold

        Returns:
            List of NodeRAGResult from federation nodes
        """
        # Get capable nodes
        nodes = self._select_nodes()

        if not nodes:
            logger.info("No federation nodes available")
            return []

        # Limit parallel queries
        nodes = nodes[:self._config.max_parallel_nodes]

        # Build RPC requests
        tasks = []
        for node in nodes:
            task = self._query_node(node, text, k, mode, min_confidence)
            tasks.append(task)

        # Execute in parallel with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self._config.federation_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Federation query timeout after {self._config.federation_timeout}s")
            results = []

        # Filter successful results
        node_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Node {nodes[i].node_id} query failed: {result}")
                continue
            if isinstance(result, NodeRAGResult):
                node_results.append(result)

        return node_results

    async def _query_node(
        self,
        node: FederationNode,
        text: str,
        k: int,
        mode: str,
        min_confidence: float,
    ) -> NodeRAGResult:
        """
        Query a single federation node.

        Args:
            node: Federation node to query
            text: Query text
            k: Number of sources
            mode: Reasoning mode
            min_confidence: Minimum confidence threshold

        Returns:
            NodeRAGResult from the node
        """
        # Check rate limit
        if self._config.enable_rate_limiting and self._rate_limiter:
            rate_result = await self._rate_limiter.check(node.node_id)
            if not rate_result.allowed:
                raise Exception(f"Rate limited: {rate_result.reason}")

        # Build RPC request
        params = {
            "query": text,
            "k": k,
            "mode": mode,
            "min_confidence": min_confidence,
            "include_sources": True,
        }

        if self._rpc_builder:
            request = self._rpc_builder.build_request(
                method=RAG_RECALL_METHOD,
                params=params,
            )
        else:
            # Fallback: create minimal request
            from hololoom.federation.wire_protocol import RPCRequest
            request = RPCRequest(method=RAG_RECALL_METHOD, params=params)

        # Safety check
        if self._config.enable_safety_checks and self._safety_gate:
            safety_result = await self._safety_gate.check_request(request, node)
            if not safety_result.allowed:
                raise Exception(f"Safety blocked: {safety_result.reason}")

        # Send RPC
        if self._send_rpc:
            response = await self._send_rpc(request, node.node_id)

            if response.error:
                raise Exception(f"RPC error: {response.error.message}")

            # Parse response
            result_data = response.result or {}
            return NodeRAGResult(
                node_id=node.node_id,
                response=result_data.get("response", ""),
                sources=result_data.get("sources", []),
                confidence=result_data.get("confidence", 0.0),
                trust_level=trust_score_to_level(node.trust_score),
                metadata=result_data.get("metadata", {}),
            )
        else:
            # No RPC callback - return empty result
            return NodeRAGResult(
                node_id=node.node_id,
                response="",
                sources=[],
                confidence=0.0,
                trust_level=trust_score_to_level(node.trust_score),
            )

    def _select_nodes(self) -> list[FederationNode]:
        """
        Select federation nodes with RAG capability.

        Returns:
            List of capable nodes sorted by trust level
        """
        if not self._get_capable_nodes:
            return []

        nodes = self._get_capable_nodes()

        # Filter by minimum trust level
        min_trust = self._config.min_node_trust
        min_trust_score = trust_level_to_score(min_trust)
        filtered = []

        for node in nodes:
            # Check if node has RAG capability
            if Capability.RAG not in node.capabilities:
                continue

            # Check trust score (FederationNode uses trust_score, not trust_level)
            if node.trust_score < min_trust_score:
                continue

            filtered.append(node)

        # Sort by trust score (highest first)
        filtered.sort(
            key=lambda n: n.trust_score,
            reverse=True,
        )

        return filtered

    def _build_response(
        self,
        merged: MergedRAGResult,
        local_result: RAGResult | None,
    ) -> str:
        """
        Build final response from merged sources.

        For now, uses local response if available, otherwise
        indicates federation sources were merged.

        Args:
            merged: Merged RAG result
            local_result: Original local result

        Returns:
            Response string
        """
        if local_result and local_result.response:
            return local_result.response

        if merged.sources:
            # Construct response from top sources
            top_sources = merged.top_sources[:3]
            if top_sources:
                return f"Based on {len(merged.sources)} federated sources: {' '.join(top_sources)}"

        return "No relevant information found across federation."

    def get_capable_nodes(self) -> list[FederationNode]:
        """
        Get list of nodes with RAG capability.

        Returns:
            List of FederationNode with RAG capability
        """
        return self._select_nodes()


# ============================================================================
#  Factory Functions
# ============================================================================

def create_federated_rag(
    local_rag: SimpleRAG | None = None,
    safety_gate: FederationSafetyGate | None = None,
    rate_limiter: FederatedRateLimiter | None = None,
    config: FederatedRAGConfig | None = None,
    get_capable_nodes: Callable[[], list[FederationNode]] | None = None,
    send_rpc: Callable[[RPCRequest, str], Coroutine[Any, Any, RPCResponse]] | None = None,
) -> FederatedRAG:
    """
    Create a FederatedRAG instance.

    Args:
        local_rag: Local RAG for initial queries
        safety_gate: Federation safety gate
        rate_limiter: Rate limiter for federation requests
        config: Configuration options
        get_capable_nodes: Callback to get RAG-capable nodes
        send_rpc: Callback to send RPC requests

    Returns:
        Configured FederatedRAG instance
    """
    # Lazy-load RPC builder
    try:
        from hololoom.federation.wire_protocol import JSONRPCBuilder
        rpc_builder = JSONRPCBuilder(sender_id="local")
    except ImportError:
        rpc_builder = None

    return FederatedRAG(
        local_rag=local_rag,
        safety_gate=safety_gate,
        rate_limiter=rate_limiter,
        rpc_builder=rpc_builder,
        config=config,
        get_capable_nodes=get_capable_nodes,
        send_rpc=send_rpc,
    )


# ============================================================================
#  Exports
# ============================================================================

__all__ = [
    # Configuration
    "FederatedRAGConfig",
    # Results
    "FederatedRAGResult",
    # Aggregator
    "ConfidenceAggregator",
    # Main class
    "FederatedRAG",
    # Factory
    "create_federated_rag",
    # Constants
    "DEFAULT_FEDERATION_THRESHOLD",
    "DEFAULT_MAX_PARALLEL_NODES",
    "DEFAULT_FEDERATION_TIMEOUT",
    "RAG_RECALL_METHOD",
]
