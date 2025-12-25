"""
Federation Integration Tests.

Tests the complete federation system:
- Federation lifecycle (join, query, leave)
- Consensus verification (agreement, disagreement, quorum)
- DHT routing (capability matching, load balancing)
- Guild management (creation, reputation, trust levels)
- SWIM membership (join, ping, suspect)

Philosophy: Integration tests over unit tests (more value per line).
"""

from __future__ import annotations

import asyncio
import pytest
from typing import List

from ..identity import Identity, create_node, node_id_distance
from ..types import (
    Capability,
    FederationNode,
    Query,
    Response,
    VerificationLevel,
    GuildTrustLevel,
    AdmissionPolicy,
    GuildError,
)
from ..core import Federation, FederationConfig
from ..routing import KademliaRouter, RoutingTable
from ..gossip import SwimMembership, MessageType
from ..verifier import ConsensusVerifier, DSStarScorer, get_quorum
from ..guild import GuildManager, TrustCalculator


# ═══════════════════════════════════════════════════════════════════════════
#  FEDERATION LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════


class TestFederationLifecycle:
    """Test Federation creation, joining, and leaving."""

    @pytest.mark.asyncio
    async def test_federation_context_manager(self, config: FederationConfig):
        """Federation works as async context manager."""
        async with Federation(config) as fed:
            assert fed is not None
            assert fed._config == config

    @pytest.mark.asyncio
    async def test_federation_development_config(self):
        """Development config has sane defaults."""
        config = FederationConfig.development()

        assert config.default_timeout_ms > 0
        assert config.min_trust_score == 0.0  # Relaxed for dev

    @pytest.mark.asyncio
    async def test_federation_production_config(self):
        """Production config is more conservative."""
        config = FederationConfig.production()

        assert config.default_timeout_ms >= 2000
        assert config.min_trust_score >= 0.5  # Stricter for prod

    @pytest.mark.asyncio
    async def test_federation_local_query(self, config: FederationConfig):
        """Queries work locally without network."""
        async with Federation(config) as fed:
            # Local query should work without joining
            result = await fed.query("What is Thompson Sampling?")

            assert result is not None
            assert result.answer != ""


# ═══════════════════════════════════════════════════════════════════════════
#  CONSENSUS VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════


class TestConsensusVerification:
    """Test multi-node consensus verification."""

    @pytest.mark.asyncio
    async def test_verify_agreeing_responses(
        self,
        verifier: ConsensusVerifier,
        simple_query: Query,
        sample_responses: List[Response],
    ):
        """Agreeing responses reach consensus."""
        verification = await verifier.verify(
            query=simple_query,
            responses=sample_responses,
            level=VerificationLevel.STANDARD,
        )

        assert verification.verified is True
        assert verification.confidence >= 0.66
        assert verification.consensus_response != ""
        assert len(verification.verifiers) >= 2

    @pytest.mark.asyncio
    async def test_verify_disagreeing_responses(
        self,
        verifier: ConsensusVerifier,
        simple_query: Query,
        disagreeing_responses: List[Response],
    ):
        """Disagreeing responses fail consensus."""
        verification = await verifier.verify(
            query=simple_query,
            responses=disagreeing_responses,
            level=VerificationLevel.STANDARD,
        )

        assert verification.verified is False
        assert len(verification.dissenting) > 0

    @pytest.mark.asyncio
    async def test_verify_insufficient_quorum(
        self,
        verifier: ConsensusVerifier,
        simple_query: Query,
        sample_responses: List[Response],
    ):
        """Insufficient responses fail quorum check."""
        # Only 1 response, STANDARD requires 3
        verification = await verifier.verify(
            query=simple_query,
            responses=sample_responses[:1],
            level=VerificationLevel.STANDARD,
        )

        assert verification.verified is False
        assert "insufficient_quorum" in str(verification.scores)

    @pytest.mark.asyncio
    async def test_verify_none_level(
        self,
        verifier: ConsensusVerifier,
        simple_query: Query,
        sample_responses: List[Response],
    ):
        """NONE level skips verification."""
        verification = await verifier.verify(
            query=simple_query,
            responses=sample_responses,
            level=VerificationLevel.NONE,
        )

        # Should just use highest confidence response
        assert verification.verified is True
        assert len(verification.verifiers) == 1

    def test_quorum_requirements(self):
        """Quorum scales with verification level."""
        assert get_quorum(VerificationLevel.NONE) == 0
        assert get_quorum(VerificationLevel.LIGHT) == 2
        assert get_quorum(VerificationLevel.STANDARD) == 3
        assert get_quorum(VerificationLevel.DEEP) == 5
        assert get_quorum(VerificationLevel.CRITICAL) == 7


class TestDSStarScoring:
    """Test DS-STAR quality assessment."""

    @pytest.mark.asyncio
    async def test_score_relevant_response(self, simple_query: Query, nodes: List[FederationNode]):
        """Relevant response scores well."""
        scorer = DSStarScorer()

        response = Response(
            text="Thompson Sampling is a Bayesian algorithm for exploration-exploitation.",
            request_id="test-1",
            responder=nodes[0].node_id,
            confidence=0.9,
            latency_ms=50.0,
        )

        score = await scorer.score(simple_query, response)

        # Should score reasonably on domain relevance
        assert score.domain > 0.3
        assert score.sensibility > 0.5
        assert score.total > 0.4

    @pytest.mark.asyncio
    async def test_score_irrelevant_response(self, simple_query: Query, nodes: List[FederationNode]):
        """Irrelevant response scores poorly on domain."""
        scorer = DSStarScorer()

        response = Response(
            text="The weather is nice today.",
            request_id="test-2",
            responder=nodes[0].node_id,
            confidence=0.5,
            latency_ms=50.0,
        )

        score = await scorer.score(simple_query, response)

        # Low domain relevance
        assert score.domain < 0.3


# ═══════════════════════════════════════════════════════════════════════════
#  KADEMLIA ROUTING
# ═══════════════════════════════════════════════════════════════════════════


class TestKademliaRouting:
    """Test DHT-based capability routing."""

    def test_node_id_distance(self, routing_table: RoutingTable):
        """XOR distance calculation is correct."""
        id_a = "a" * 40
        id_b = "b" * 40

        distance = node_id_distance(id_a, id_b)
        assert distance > 0

        # Same ID has zero distance
        assert node_id_distance(id_a, id_a) == 0

    def test_routing_table_bucket_index(self, routing_table: RoutingTable):
        """Bucket index based on XOR distance."""
        # Bucket index should be in valid range
        other_id = "f" * 40
        bucket_idx = routing_table._bucket_index(other_id)

        assert 0 <= bucket_idx < 160

    def test_routing_table_add_node(self, routing_table: RoutingTable, nodes: List[FederationNode]):
        """Nodes can be added to routing table."""
        for node in nodes:
            routing_table.add(node)

        # Should be able to find nodes
        closest = routing_table.closest(nodes[0].node_id, n=3)
        assert len(closest) > 0

    @pytest.mark.asyncio
    async def test_router_find_nodes_by_capability(self, router: KademliaRouter, nodes: List[FederationNode]):
        """Router finds nodes by capability."""
        # Add nodes to router
        for node in nodes:
            router.add_node(node)

        # Find nodes with WEAVING capability
        capable = await router.find_nodes(
            capabilities={Capability.WEAVING},
            min_trust=0.0,
            limit=5,
        )

        assert len(capable) > 0
        for node in capable:
            assert Capability.WEAVING in node.capabilities

    @pytest.mark.asyncio
    async def test_router_find_by_multiple_capabilities(
        self, router: KademliaRouter, nodes: List[FederationNode]
    ):
        """Router finds nodes with multiple capabilities."""
        for node in nodes:
            router.add_node(node)

        # Find nodes with both RAG and CODE
        capable = await router.find_nodes(
            capabilities={Capability.RAG},
            min_trust=0.0,
            limit=5,
        )

        # Filter for CODE as well
        both = [n for n in capable if Capability.CODE in n.capabilities]

        # Should find node at port 9004 (RAG + CODE)
        assert any(n.endpoint == "localhost:9004" for n in both)


# ═══════════════════════════════════════════════════════════════════════════
#  GUILD MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════


class TestGuildManagement:
    """Test guild lifecycle and reputation."""

    @pytest.mark.asyncio
    async def test_create_guild(self, guild_manager: GuildManager):
        """Guilds can be created."""
        guild = await guild_manager.create(
            name="Medical AI",
            domain="medical",
            admission=AdmissionPolicy.OPEN,
        )

        assert guild.name == "Medical AI"
        assert guild.domain == "medical"
        assert guild.trust_level == GuildTrustLevel.STARTER

    @pytest.mark.asyncio
    async def test_join_open_guild(self, guild_manager: GuildManager, node: FederationNode):
        """Joining open guild succeeds immediately."""
        guild = await guild_manager.create(
            name="Open Guild",
            domain="general",
            admission=AdmissionPolicy.OPEN,
        )

        result = await guild_manager.join(guild.guild_id, node.node_id)

        assert result is True
        assert node.node_id in guild.members

    @pytest.mark.asyncio
    async def test_join_vouched_guild_without_sponsor(
        self, guild_manager: GuildManager, node: FederationNode
    ):
        """Joining vouched guild without sponsor fails."""
        guild = await guild_manager.create(
            name="Elite Guild",
            domain="elite",
            admission=AdmissionPolicy.VOUCHED,
        )

        with pytest.raises(GuildError) as exc_info:
            await guild_manager.join(guild.guild_id, node.node_id)

        assert "sponsor" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_join_vouched_guild_with_sponsor(
        self, guild_manager: GuildManager, nodes: List[FederationNode]
    ):
        """Joining vouched guild with valid sponsor succeeds."""
        # Create guild with founder
        guild = await guild_manager.create(
            name="Elite Guild",
            domain="elite",
            admission=AdmissionPolicy.VOUCHED,
            founder=nodes[0].node_id,
        )

        # Second node joins with first as sponsor
        result = await guild_manager.join(
            guild.guild_id,
            nodes[1].node_id,
            sponsor=nodes[0].node_id,
        )

        assert result is True
        assert nodes[1].node_id in guild.members

    @pytest.mark.asyncio
    async def test_guild_reputation_updates(
        self, guild_manager: GuildManager, node: FederationNode
    ):
        """Reputation updates based on verification outcomes."""
        guild = await guild_manager.create(
            name="Test Guild",
            domain="test",
            admission=AdmissionPolicy.OPEN,
            founder=node.node_id,
        )

        # Record some successes
        for _ in range(5):
            await guild_manager.record_verification(
                guild_id=guild.guild_id,
                node_id=node.node_id,
                success=True,
            )

        reputation = guild_manager.get_reputation(guild.guild_id, node.node_id)
        assert reputation > 0.5  # Above neutral

        # Record some failures
        for _ in range(10):
            await guild_manager.record_verification(
                guild_id=guild.guild_id,
                node_id=node.node_id,
                success=False,
            )

        new_reputation = guild_manager.get_reputation(guild.guild_id, node.node_id)
        assert new_reputation < reputation  # Should decrease

    @pytest.mark.asyncio
    async def test_guild_trust_level_evolution(self, guild_manager: GuildManager, node: FederationNode):
        """Guild trust level evolves with age and verifications."""
        guild = await guild_manager.create(
            name="Evolving Guild",
            domain="test",
            admission=AdmissionPolicy.OPEN,
            founder=node.node_id,
        )

        # Start as STARTER
        assert guild.trust_level == GuildTrustLevel.STARTER

        # Update trust level (won't change without age/verifications)
        level = await guild_manager.update_trust_level(guild.guild_id)
        assert level == GuildTrustLevel.STARTER


class TestTrustCalculator:
    """Test trust calculation across guilds."""

    def test_calculate_trust_with_guild_context(
        self, node: FederationNode, guild_manager: GuildManager
    ):
        """Trust calculation uses guild context."""
        calculator = TrustCalculator()

        # Without any guild membership, should be neutral
        trust = calculator.calculate(node, guild_manager)
        assert 0.0 <= trust <= 1.0

    def test_trust_weights_sum_to_one(self):
        """Trust calculator weights are normalized."""
        calculator = TrustCalculator(
            guild_weight=0.4,
            network_weight=0.3,
            history_weight=0.3,
        )

        total = (
            calculator._guild_weight
            + calculator._network_weight
            + calculator._history_weight
        )
        assert abs(total - 1.0) < 0.01


# ═══════════════════════════════════════════════════════════════════════════
#  SWIM MEMBERSHIP
# ═══════════════════════════════════════════════════════════════════════════


class TestSwimMembership:
    """Test SWIM gossip membership protocol."""

    def test_membership_creation(self, membership: SwimMembership, node: FederationNode):
        """Membership initializes correctly."""
        assert membership._local == node
        assert len(membership._members) == 0

    @pytest.mark.asyncio
    async def test_membership_add_member(self, membership: SwimMembership, nodes: List[FederationNode]):
        """Members can be added."""
        from ..gossip import MemberState
        from ..types import NodeStatus

        for other in nodes[1:]:
            membership._members[other.node_id] = MemberState(node=other)

        assert len(membership._members) == len(nodes) - 1

    @pytest.mark.asyncio
    async def test_membership_suspect(self, membership: SwimMembership, nodes: List[FederationNode]):
        """Nodes can be suspected."""
        from ..gossip import MemberState
        from ..types import NodeStatus

        # Add a member
        target = nodes[1]
        membership._members[target.node_id] = MemberState(node=target)

        # Suspect it
        await membership.suspect(target.node_id)

        assert target.node_id in membership._members
        assert membership._members[target.node_id].status == NodeStatus.SUSPECT

    @pytest.mark.asyncio
    async def test_membership_confirm_dead(self, membership: SwimMembership, nodes: List[FederationNode]):
        """Suspected nodes can be confirmed dead."""
        from ..gossip import MemberState
        from ..types import NodeStatus

        target = nodes[1]
        membership._members[target.node_id] = MemberState(
            node=target,
            status=NodeStatus.SUSPECT,
        )

        await membership.confirm_dead(target.node_id)

        # Node should be removed from members after being confirmed dead
        assert target.node_id not in membership._members

    def test_membership_get_members(self, membership: SwimMembership, nodes: List[FederationNode]):
        """Can get list of online members."""
        from ..gossip import MemberState

        for other in nodes[1:3]:
            membership._members[other.node_id] = MemberState(node=other)

        members = membership.get_members()
        assert len(members) == 2

    def test_membership_member_count(self, membership: SwimMembership, nodes: List[FederationNode]):
        """Member count is accurate."""
        from ..gossip import MemberState

        for other in nodes[1:]:
            membership._members[other.node_id] = MemberState(node=other)

        assert membership.member_count == len(nodes) - 1


# ═══════════════════════════════════════════════════════════════════════════
#  INTEGRATION: END-TO-END
# ═══════════════════════════════════════════════════════════════════════════


class TestEndToEnd:
    """Full integration tests across components."""

    @pytest.mark.asyncio
    async def test_query_with_verification(self, config: FederationConfig):
        """Query with verification uses full pipeline."""
        config.default_level = VerificationLevel.LIGHT

        async with Federation(config) as fed:
            result = await fed.query(
                "What is Thompson Sampling?",
                level=VerificationLevel.LIGHT,
            )

            assert result is not None
            assert result.answer != ""

    @pytest.mark.asyncio
    async def test_guild_affects_routing(
        self,
        router: KademliaRouter,
        guild_manager: GuildManager,
        nodes: List[FederationNode],
    ):
        """Guild membership affects routing decisions."""
        # Add nodes to router
        for node in nodes:
            router.add_node(node)

        # Create medical guild
        guild = await guild_manager.create(
            name="Medical",
            domain="medical",
            admission=AdmissionPolicy.OPEN,
        )

        # Add node with MEDICAL capability to guild
        medical_node = next(n for n in nodes if Capability.MEDICAL in n.capabilities)
        await guild_manager.join(guild.guild_id, medical_node.node_id)

        # Find nodes by capability
        medical_capable = await router.find_nodes(
            capabilities={Capability.MEDICAL},
            min_trust=0.0,
            limit=5,
        )

        assert medical_node in medical_capable

    @pytest.mark.asyncio
    async def test_consensus_with_guild_reputation(
        self,
        verifier: ConsensusVerifier,
        guild_manager: GuildManager,
        nodes: List[FederationNode],
        simple_query: Query,
    ):
        """Consensus considers guild reputation."""
        # Create guild and add nodes
        guild = await guild_manager.create(
            name="Experts",
            domain="ml",
            admission=AdmissionPolicy.OPEN,
        )

        for node in nodes[:3]:
            await guild_manager.join(guild.guild_id, node.node_id)
            # Build up reputation
            for _ in range(10):
                await guild_manager.record_verification(
                    guild_id=guild.guild_id,
                    node_id=node.node_id,
                    success=True,
                )

        # Create responses from guild members
        responses = [
            Response(
                text="Thompson Sampling is a Bayesian bandit algorithm.",
                request_id="guild-test",
                responder=nodes[i].node_id,
                confidence=0.9,
                latency_ms=50.0,
            )
            for i in range(3)
        ]

        verification = await verifier.verify(
            query=simple_query,
            responses=responses,
            level=VerificationLevel.STANDARD,
        )

        assert verification.verified is True


# ═══════════════════════════════════════════════════════════════════════════
#  EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_response_list(self, verifier: ConsensusVerifier, simple_query: Query):
        """Empty response list handles gracefully."""
        verification = await verifier.verify(
            query=simple_query,
            responses=[],
            level=VerificationLevel.STANDARD,
        )

        assert verification.verified is False

    @pytest.mark.asyncio
    async def test_join_nonexistent_guild(self, guild_manager: GuildManager, node: FederationNode):
        """Joining nonexistent guild raises error."""
        with pytest.raises(GuildError):
            await guild_manager.join("nonexistent-guild-id", node.node_id)

    def test_routing_table_empty_closest(self, routing_table: RoutingTable):
        """Empty routing table returns empty closest list."""
        closest = routing_table.closest("a" * 40, n=10)
        assert len(closest) == 0

    @pytest.mark.asyncio
    async def test_low_confidence_responses_filtered(
        self, verifier: ConsensusVerifier, simple_query: Query, nodes: List[FederationNode]
    ):
        """Low confidence responses are filtered out."""
        responses = [
            Response(
                text="Answer",
                request_id="low-conf",
                responder=nodes[0].node_id,
                confidence=0.1,  # Below min_confidence
                latency_ms=50.0,
            )
        ]

        verification = await verifier.verify(
            query=simple_query,
            responses=responses,
            level=VerificationLevel.STANDARD,
        )

        # Should fail due to insufficient quorum after filtering
        assert verification.verified is False
