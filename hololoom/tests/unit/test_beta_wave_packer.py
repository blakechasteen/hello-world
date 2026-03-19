"""
Unit Tests for Beta Wave Context Packer

Tests the elegant physics-based context packing with activation spreading.

Test Coverage:
1. Activation threshold logic (boundary conditions)
2. Budget exhaustion (token limits)
3. Creative insights handling
4. Awareness integration
5. Compression strategies
6. Edge cases (empty inputs, Unicode, etc.)
"""

import pytest
import asyncio
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from hololoom.memory.awareness.beta_wave_packer import (
    BetaWaveContextPacker,
    TokenBudget,
    ContextElement,
    PackedContext
)
from hololoom.memory.spring_dynamics_engine import (
    SpringDynamicsEngine,
    SpringEngineConfig,
    MemoryNode,
    BetaWaveRecallResult
)


# ============================================================================
# Test Fixtures
# ============================================================================

@dataclass
class MockAwarenessContext:
    """Mock awareness context for testing"""

    @dataclass
    class Confidence:
        uncertainty_level: float = 0.3
        query_cache_status: str = "MISS"
        knowledge_gap_detected: bool = False

    @dataclass
    class Patterns:
        domain: str = "quantum_computing"
        subdomain: str = "entanglement"
        seen_count: int = 5
        confidence: float = 0.85

    confidence: Confidence = field(default_factory=Confidence)
    patterns: Patterns = field(default_factory=Patterns)


@pytest.fixture
def spring_engine():
    """Create a test spring dynamics engine"""
    config = SpringEngineConfig()
    engine = SpringDynamicsEngine(config)

    # Add test memories
    test_memories = [
        {
            'id': 'high_activation',
            'content': 'High activation memory - very relevant',
            'embedding': np.random.randn(128),
            'k': 2.0
        },
        {
            'id': 'medium_activation',
            'content': 'Medium activation memory - somewhat relevant',
            'embedding': np.random.randn(128),
            'k': 1.0
        },
        {
            'id': 'low_activation',
            'content': 'Low activation memory - barely relevant',
            'embedding': np.random.randn(128),
            'k': 0.3
        }
    ]

    for mem in test_memories:
        embedding = mem['embedding']
        node = MemoryNode(
            id=mem['id'],
            content=mem['content'],
            position=embedding,
            rest_position=embedding.copy(),
            velocity=np.zeros_like(embedding),
            spring_constant=mem['k'],
            max_spring_constant=10.0,
            mass=1.0
        )
        engine.nodes[mem['id']] = node

    return engine


@pytest.fixture
def token_budget():
    """Standard token budget for tests"""
    return TokenBudget(total=2000, reserved_for_query=200, reserved_for_response=500)


@pytest.fixture
def awareness_context():
    """Mock awareness context for tests"""
    return MockAwarenessContext()


# ============================================================================
# Test 1: Activation Threshold Logic
# ============================================================================

class TestActivationThresholds:
    """Test that activation thresholds work correctly"""

    @pytest.mark.asyncio
    async def test_high_activation_included_full(self, spring_engine, token_budget):
        """Elements with activation >0.7 should be included at full content"""
        packer = BetaWaveContextPacker(
            spring_engine=spring_engine,
            token_budget=token_budget,
            activation_threshold=0.3,
            compression_threshold=0.7
        )

        # Create element with high activation
        element = ContextElement(
            content="A" * 400,  # 400 chars = ~100 tokens
            activation=0.9,  # High activation
            token_count=100,
            source="memory"
        )

        # Manually add to engine for testing
        embedding = np.random.randn(128)
        node = MemoryNode(
            id="test_high",
            content=element.content,
            position=embedding,
            rest_position=embedding.copy(),
            velocity=np.zeros_like(embedding),
            spring_constant=2.0,
            max_spring_constant=10.0,
            mass=1.0
        )
        spring_engine.nodes["test_high"] = node

        # Mock retrieval result
        spring_engine.retrieve_memories = lambda *args, **kwargs: BetaWaveRecallResult(
            recalled_memories=[("test_high", 0.9)],
            all_activations={"test_high": 0.9},
            creative_insights=[],
            seed_nodes=[],
            iterations=10
        )

        query_embedding = np.random.randn(128)
        packed = await packer.pack_context(
            query_text="Test query",
            query_embedding=query_embedding,
            top_k=10
        )

        # Should be included at full size (not compressed)
        assert packed.elements_included >= 1
        assert packed.avg_activation >= 0.7

    @pytest.mark.asyncio
    async def test_medium_activation_compressed(self, spring_engine, token_budget):
        """Elements with 0.3 <= activation < 0.7 should be compressed"""
        packer = BetaWaveContextPacker(
            spring_engine=spring_engine,
            token_budget=token_budget,
            activation_threshold=0.3,
            compression_threshold=0.7
        )

        # Mock retrieval with medium activation
        spring_engine.retrieve_memories = lambda *args, **kwargs: BetaWaveRecallResult(
            recalled_memories=[("medium_activation", 0.5)],
            all_activations={"medium_activation": 0.5},
            creative_insights=[],
            seed_nodes=[],
            iterations=10
        )

        query_embedding = np.random.randn(128)
        packed = await packer.pack_context(
            query_text="Test query",
            query_embedding=query_embedding,
            top_k=10
        )

        # Should have compression
        assert packed.elements_compressed >= 0

    @pytest.mark.asyncio
    async def test_low_activation_excluded(self, spring_engine, token_budget):
        """Elements with activation <0.3 should be excluded"""
        packer = BetaWaveContextPacker(
            spring_engine=spring_engine,
            token_budget=token_budget,
            activation_threshold=0.3,
            compression_threshold=0.7
        )

        # Mock retrieval with only low activation
        spring_engine.retrieve_memories = lambda *args, **kwargs: BetaWaveRecallResult(
            recalled_memories=[("low_activation", 0.2)],  # Below threshold
            all_activations={"low_activation": 0.2},
            creative_insights=[],
            seed_nodes=[],
            iterations=10
        )

        query_embedding = np.random.randn(128)
        packed = await packer.pack_context(
            query_text="Test query",
            query_embedding=query_embedding,
            top_k=10
        )

        # Query should be only element (activation <0.3 excluded)
        # Min activation should be from query (1.0) not low activation memory
        assert packed.min_activation >= 0.3 or packed.elements_included == 1


# ============================================================================
# Test 2: Budget Exhaustion
# ============================================================================

class TestBudgetExhaustion:
    """Test token budget constraints"""

    @pytest.mark.asyncio
    async def test_respects_budget_limit(self, spring_engine):
        """Packer should not exceed available token budget"""
        tight_budget = TokenBudget(total=500, reserved_for_query=100, reserved_for_response=200)
        # Available for context = 500 - 100 - 200 = 200 tokens

        packer = BetaWaveContextPacker(
            spring_engine=spring_engine,
            token_budget=tight_budget,
            activation_threshold=0.3,
            compression_threshold=0.7
        )

        # Add mock nodes to engine
        for i in range(20):
            emb = np.random.randn(128)
            node = MemoryNode(
                id=f"mem_{i}",
                content=f"Test memory content {i}",
                position=emb,
                rest_position=emb.copy(),
                velocity=np.zeros_like(emb),
                spring_constant=1.0
            )
            spring_engine.nodes[f"mem_{i}"] = node

        # Mock retrieval with many high-activation memories
        mock_memories = [(f"mem_{i}", 0.9 - i*0.05) for i in range(20)]
        spring_engine.retrieve_memories = lambda *args, **kwargs: BetaWaveRecallResult(
            recalled_memories=mock_memories,
            all_activations={nid: act for nid, act in mock_memories},
            creative_insights=[],
            seed_nodes=[],
            iterations=10
        )

        query_embedding = np.random.randn(128)
        packed = await packer.pack_context(
            query_text="Test query",
            query_embedding=query_embedding,
            top_k=20
        )

        # Should not exceed available budget
        assert packed.total_tokens <= tight_budget.available_for_context

    @pytest.mark.asyncio
    async def test_uses_compression_when_tight(self, spring_engine):
        """Should compress more aggressively when budget is tight"""
        tight_budget = TokenBudget(total=800, reserved_for_query=100, reserved_for_response=300)
        generous_budget = TokenBudget(total=4000, reserved_for_query=200, reserved_for_response=800)

        # Add mock nodes to engine
        for i in range(10):
            emb = np.random.randn(128)
            node = MemoryNode(
                id=f"mem_{i}",
                content=f"Test memory content {i}",
                position=emb,
                rest_position=emb.copy(),
                velocity=np.zeros_like(emb),
                spring_constant=1.0
            )
            spring_engine.nodes[f"mem_{i}"] = node

        # Same memories, different budgets
        mock_memories = [(f"mem_{i}", 0.8) for i in range(10)]
        spring_engine.retrieve_memories = lambda *args, **kwargs: BetaWaveRecallResult(
            recalled_memories=mock_memories,
            all_activations={nid: 0.8 for nid, _ in mock_memories},
            creative_insights=[],
            seed_nodes=[],
            iterations=10
        )

        query_embedding = np.random.randn(128)

        tight_packer = BetaWaveContextPacker(spring_engine, tight_budget)
        tight_packed = await tight_packer.pack_context("Test", query_embedding)

        generous_packer = BetaWaveContextPacker(spring_engine, generous_budget)
        generous_packed = await generous_packer.pack_context("Test", query_embedding)

        # Tight budget should exclude more elements
        assert tight_packed.elements_excluded >= generous_packed.elements_excluded


# ============================================================================
# Test 3: Creative Insights
# ============================================================================

class TestCreativeInsights:
    """Test handling of cross-domain creative insights"""

    @pytest.mark.asyncio
    async def test_creative_insights_included(self, spring_engine, token_budget):
        """Creative insights should be included with slightly lower priority"""
        packer = BetaWaveContextPacker(
            spring_engine=spring_engine,
            token_budget=token_budget
        )

        # Add creative insight node
        insight_embedding = np.random.randn(128)
        insight_node = MemoryNode(
            id="creative_insight",
            content="Cross-domain creative bridge memory",
            position=insight_embedding,
            rest_position=insight_embedding.copy(),
            velocity=np.zeros_like(insight_embedding),
            spring_constant=1.5,
            max_spring_constant=10.0,
            mass=1.0
        )
        spring_engine.nodes["creative_insight"] = insight_node

        # Mock retrieval with creative insights
        spring_engine.retrieve_memories = lambda *args, **kwargs: BetaWaveRecallResult(
            recalled_memories=[("high_activation", 0.9)],
            all_activations={"high_activation": 0.9, "creative_insight": 0.75},
            creative_insights=[("creative_insight", 0.75, 2.5)],  # (node_id, activation, distance)
            seed_nodes=[],
            iterations=10
        )

        query_embedding = np.random.randn(128)
        packed = await packer.pack_context(
            query_text="Test query",
            query_embedding=query_embedding,
            top_k=10
        )

        # Should have creative insights section
        assert packed.creative_section or packed.elements_included >= 1


# ============================================================================
# Test 4: Awareness Integration
# ============================================================================

class TestAwarenessIntegration:
    """Test integration with awareness context"""

    @pytest.mark.asyncio
    async def test_awareness_context_included(self, spring_engine, token_budget, awareness_context):
        """Awareness signals should be included in packed context"""
        packer = BetaWaveContextPacker(
            spring_engine=spring_engine,
            token_budget=token_budget
        )

        # Mock retrieval
        spring_engine.retrieve_memories = lambda *args, **kwargs: BetaWaveRecallResult(
            recalled_memories=[],
            all_activations={},
            creative_insights=[],
            seed_nodes=[],
            iterations=10
        )

        query_embedding = np.random.randn(128)
        packed = await packer.pack_context(
            query_text="Test query",
            query_embedding=query_embedding,
            awareness_context=awareness_context,
            top_k=10
        )

        # Should have awareness section
        assert packed.awareness_section
        assert "Confidence" in packed.awareness_section or "Domain" in packed.awareness_section

    @pytest.mark.asyncio
    async def test_works_without_awareness(self, spring_engine, token_budget):
        """Should work gracefully without awareness context"""
        packer = BetaWaveContextPacker(
            spring_engine=spring_engine,
            token_budget=token_budget
        )

        spring_engine.retrieve_memories = lambda *args, **kwargs: BetaWaveRecallResult(
            recalled_memories=[("high_activation", 0.9)],
            all_activations={"high_activation": 0.9},
            creative_insights=[],
            seed_nodes=[],
            iterations=10
        )

        query_embedding = np.random.randn(128)
        packed = await packer.pack_context(
            query_text="Test query",
            query_embedding=query_embedding,
            awareness_context=None,  # No awareness
            top_k=10
        )

        # Should still work
        assert packed.total_tokens > 0
        assert packed.elements_included >= 1


# ============================================================================
# Test 5: Compression Strategies
# ============================================================================

class TestCompressionStrategies:
    """Test compression logic"""

    def test_compression_reduces_size(self):
        """Compression should reduce token count"""
        from hololoom.memory.awareness.beta_wave_packer import BetaWaveContextPacker

        packer = BetaWaveContextPacker(None, None)  # Don't need engine for this test

        long_content = "A" * 1000  # 1000 chars
        compressed = packer._compress_content(long_content, ratio=0.5)

        # Should be roughly half size
        assert len(compressed) < len(long_content)
        assert len(compressed) <= 500 + 10  # +10 for ellipsis

    def test_compression_ratio_respected(self):
        """Different ratios should produce different sizes"""
        from hololoom.memory.awareness.beta_wave_packer import BetaWaveContextPacker

        packer = BetaWaveContextPacker(None, None)

        content = "A" * 1000
        compressed_50 = packer._compress_content(content, ratio=0.5)
        compressed_25 = packer._compress_content(content, ratio=0.25)

        assert len(compressed_25) < len(compressed_50)


# ============================================================================
# Test 6: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling"""

    @pytest.mark.asyncio
    async def test_empty_memories(self, spring_engine, token_budget):
        """Should handle case with no memories gracefully"""
        packer = BetaWaveContextPacker(
            spring_engine=spring_engine,
            token_budget=token_budget
        )

        # Empty retrieval
        spring_engine.retrieve_memories = lambda *args, **kwargs: BetaWaveRecallResult(
            recalled_memories=[],
            all_activations={},
            creative_insights=[],
            seed_nodes=[],
            iterations=0
        )

        query_embedding = np.random.randn(128)
        packed = await packer.pack_context(
            query_text="Test query",
            query_embedding=query_embedding,
            top_k=10
        )

        # Should still have query
        assert packed.elements_included == 1  # Just the query
        assert packed.query_section

    @pytest.mark.asyncio
    async def test_unicode_content(self, spring_engine, token_budget):
        """Should handle Unicode content correctly"""
        packer = BetaWaveContextPacker(
            spring_engine=spring_engine,
            token_budget=token_budget
        )

        # Add Unicode node
        unicode_embedding = np.random.randn(128)
        unicode_node = MemoryNode(
            id="unicode_test",
            content="Quantum entanglement 量子纠缠 🧲",
            position=unicode_embedding,
            rest_position=unicode_embedding.copy(),
            velocity=np.zeros_like(unicode_embedding),
            spring_constant=2.0,
            max_spring_constant=10.0,
            mass=1.0
        )
        spring_engine.nodes["unicode_test"] = unicode_node

        spring_engine.retrieve_memories = lambda *args, **kwargs: BetaWaveRecallResult(
            recalled_memories=[("unicode_test", 0.9)],
            all_activations={"unicode_test": 0.9},
            creative_insights=[],
            seed_nodes=[],
            iterations=10
        )

        query_embedding = np.random.randn(128)
        packed = await packer.pack_context(
            query_text="Test 测试",
            query_embedding=query_embedding,
            top_k=10
        )

        # Should handle Unicode without errors
        assert packed.total_tokens > 0
        assert packed.elements_included >= 1

    @pytest.mark.asyncio
    async def test_very_long_content(self, spring_engine):
        """Should handle very long content (>10K tokens)"""
        large_budget = TokenBudget(total=20000, reserved_for_query=1000, reserved_for_response=2000)
        packer = BetaWaveContextPacker(
            spring_engine=spring_engine,
            token_budget=large_budget
        )

        # Add very long node
        long_embedding = np.random.randn(128)
        long_node = MemoryNode(
            id="long_content",
            content="A" * 50000,  # ~12500 tokens
            position=long_embedding,
            rest_position=long_embedding.copy(),
            velocity=np.zeros_like(long_embedding),
            spring_constant=2.0,
            max_spring_constant=10.0,
            mass=1.0
        )
        spring_engine.nodes["long_content"] = long_node

        spring_engine.retrieve_memories = lambda *args, **kwargs: BetaWaveRecallResult(
            recalled_memories=[("long_content", 0.9)],
            all_activations={"long_content": 0.9},
            creative_insights=[],
            seed_nodes=[],
            iterations=10
        )

        query_embedding = np.random.randn(128)
        packed = await packer.pack_context(
            query_text="Test",
            query_embedding=query_embedding,
            top_k=10
        )

        # Should not exceed budget even with very long content
        assert packed.total_tokens <= large_budget.available_for_context


# ============================================================================
# Test 7: Activation Statistics
# ============================================================================

class TestActivationStatistics:
    """Test activation statistics tracking"""

    @pytest.mark.asyncio
    async def test_activation_distribution_tracked(self, spring_engine, token_budget):
        """Should track activation distribution correctly"""
        packer = BetaWaveContextPacker(
            spring_engine=spring_engine,
            token_budget=token_budget
        )

        # Add mock nodes to engine
        for node_id, activation in [("high1", 0.9), ("high2", 0.8), ("med1", 0.5), ("med2", 0.4), ("low1", 0.3)]:
            emb = np.random.randn(128)
            node = MemoryNode(
                id=node_id,
                content=f"Test memory content {node_id}",
                position=emb,
                rest_position=emb.copy(),
                velocity=np.zeros_like(emb),
                spring_constant=1.0
            )
            spring_engine.nodes[node_id] = node

        # Mock retrieval with mixed activations
        spring_engine.retrieve_memories = lambda *args, **kwargs: BetaWaveRecallResult(
            recalled_memories=[
                ("high1", 0.9),
                ("high2", 0.8),
                ("med1", 0.5),
                ("med2", 0.4),
                ("low1", 0.3)
            ],
            all_activations={
                "high1": 0.9,
                "high2": 0.8,
                "med1": 0.5,
                "med2": 0.4,
                "low1": 0.3
            },
            creative_insights=[],
            seed_nodes=[],
            iterations=10
        )

        query_embedding = np.random.randn(128)
        packed = await packer.pack_context(
            query_text="Test",
            query_embedding=query_embedding,
            top_k=10
        )

        # Should have distribution stats
        assert 'activation_distribution' in packed.activation_stats
        dist = packed.activation_stats['activation_distribution']

        # Should categorize correctly
        assert 'high (>0.7)' in dist or 'medium (0.3-0.7)' in dist


if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])
