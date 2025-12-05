"""
Shared fixtures for federation tests.

Provides reusable test infrastructure without boilerplate.
"""

from __future__ import annotations

import asyncio
import pytest
from typing import List

from ..identity import create_node, Identity
from ..types import (
    Capability,
    FederationNode,
    Query,
    Response,
    VerificationLevel,
)
from ..core import Federation, FederationConfig
from ..routing import KademliaRouter, RoutingTable
from ..gossip import SwimMembership
from ..consensus import ConsensusVerifier
from ..guild import GuildManager


# ═══════════════════════════════════════════════════════════════════════════
#  EVENT LOOP
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ═══════════════════════════════════════════════════════════════════════════
#  NODE FIXTURES
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def identity() -> Identity:
    """Create a test identity."""
    return Identity.generate()


@pytest.fixture
def node(identity: Identity) -> FederationNode:
    """Create a test federation node."""
    return create_node(
        identity=identity,
        endpoint="localhost:9000",
        capabilities={Capability.WEAVING, Capability.RAG},
    )


@pytest.fixture
def nodes() -> List[FederationNode]:
    """Create multiple test nodes with different capabilities."""
    node_configs = [
        ("localhost:9000", {Capability.WEAVING, Capability.RAG}),
        ("localhost:9001", {Capability.AGENTIC, Capability.CODE}),
        ("localhost:9002", {Capability.MEDICAL}),
        ("localhost:9003", {Capability.WEAVING, Capability.AGENTIC}),
        ("localhost:9004", {Capability.RAG, Capability.CODE}),
    ]

    result = []
    for endpoint, caps in node_configs:
        identity = Identity.generate()
        node = create_node(
            identity=identity,
            endpoint=endpoint,
            capabilities=caps,
        )
        result.append(node)

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  QUERY FIXTURES
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def simple_query(node: FederationNode) -> Query:
    """Create a simple test query."""
    return Query(
        text="What is Thompson Sampling?",
        request_id="test-simple-query",
        requester=node.node_id,
    )


@pytest.fixture
def complex_query(node: FederationNode) -> Query:
    """Create a complex test query requiring verification."""
    return Query(
        text="Is this medication safe for the patient?",
        request_id="test-complex-query",
        requester=node.node_id,
        level=VerificationLevel.CRITICAL,
        timeout_ms=30000,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  RESPONSE FIXTURES
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_responses(nodes: List[FederationNode]) -> List[Response]:
    """Create sample responses for consensus testing."""
    # Three agreeing responses
    agreeing_text = "Thompson Sampling is a Bayesian algorithm for exploration."

    responses = [
        Response(
            text=agreeing_text,
            request_id="test-123",
            responder=nodes[0].node_id,
            confidence=0.92,
            latency_ms=45.0,
        ),
        Response(
            text=agreeing_text,
            request_id="test-123",
            responder=nodes[1].node_id,
            confidence=0.88,
            latency_ms=52.0,
        ),
        Response(
            text="Thompson Sampling is a Bayesian approach for exploration.",
            request_id="test-123",
            responder=nodes[2].node_id,
            confidence=0.85,
            latency_ms=48.0,
        ),
    ]

    return responses


@pytest.fixture
def disagreeing_responses(nodes: List[FederationNode]) -> List[Response]:
    """Create disagreeing responses for testing consensus failure."""
    return [
        Response(
            text="Thompson Sampling is a Bayesian algorithm.",
            request_id="test-456",
            responder=nodes[0].node_id,
            confidence=0.90,
            latency_ms=45.0,
        ),
        Response(
            text="UCB is better than Thompson Sampling.",
            request_id="test-456",
            responder=nodes[1].node_id,
            confidence=0.85,
            latency_ms=50.0,
        ),
        Response(
            text="Epsilon-greedy is simpler than both.",
            request_id="test-456",
            responder=nodes[2].node_id,
            confidence=0.80,
            latency_ms=55.0,
        ),
    ]


# ═══════════════════════════════════════════════════════════════════════════
#  COMPONENT FIXTURES
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def router(node: FederationNode) -> KademliaRouter:
    """Create a test router."""
    return KademliaRouter(node, k=20, alpha=3)


@pytest.fixture
def routing_table(node: FederationNode) -> RoutingTable:
    """Create a test routing table."""
    return RoutingTable(node.node_id, k=20)


@pytest.fixture
def membership(node: FederationNode) -> SwimMembership:
    """Create a test membership manager."""
    return SwimMembership(
        node,
        ping_interval_ms=1000,
        ping_timeout_ms=500,
    )


@pytest.fixture
def verifier() -> ConsensusVerifier:
    """Create a test consensus verifier."""
    return ConsensusVerifier(
        similarity_threshold=0.8,
        min_confidence=0.3,
    )


@pytest.fixture
def guild_manager() -> GuildManager:
    """Create a test guild manager."""
    return GuildManager()


# ═══════════════════════════════════════════════════════════════════════════
#  FEDERATION FIXTURES
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def config() -> FederationConfig:
    """Create development config for testing."""
    return FederationConfig.development()


@pytest.fixture
async def federation(config: FederationConfig):
    """Create a test federation instance."""
    fed = Federation(config)
    yield fed
    # Cleanup if needed
    if hasattr(fed, '_membership') and fed._membership:
        await fed._membership.leave()
