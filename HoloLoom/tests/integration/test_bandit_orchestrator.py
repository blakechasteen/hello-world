"""
Integration tests for Bandit-Enhanced Weaving Orchestrator.

Tests:
- Bandit orchestrator initialization
- A/B testing (traffic split)
- Tool selection with Thompson Sampling
- Safety guardrails integration
- Metrics tracking (ECE, regret, rewards)
- Checkpoint save/load
"""

import pytest
import asyncio
import numpy as np
from HoloLoom.weaving_orchestrator_bandit import (
    BanditOrchestrator,
    BanditConfig,
    create_bandit_orchestrator,
)
from HoloLoom.config import Config
from HoloLoom.protocols.types import Query, MemoryShard


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def test_shards():
    """Create test memory shards."""
    return [
        MemoryShard(
            id=f"shard_{i}",
            text=f"Test content {i}",
            motifs=[f"motif_{i}"],
            entities=[f"entity_{i}"],
        )
        for i in range(10)
    ]


@pytest.fixture
def bandit_orchestrator(test_shards):
    """Create bandit orchestrator for testing."""
    config = Config.fast()  # Use fast mode for testing
    bandit_config = BanditConfig(
        sampler_type="discrete",  # Simpler for testing
        ab_test_ratio=0.5,  # 50% for easier testing
        enable_safety=False,  # Disable for speed
        enable_monitoring=True,
        replay_warmup=10,  # Low warmup for testing
    )

    return BanditOrchestrator(
        cfg=config,
        shards=test_shards,
        bandit_config=bandit_config,
        enable_bandit=True,
    )


# ============================================================================
# Tests
# ============================================================================


def test_bandit_orchestrator_init(bandit_orchestrator):
    """Test orchestrator initializes correctly."""
    assert bandit_orchestrator.enable_bandit is True
    assert bandit_orchestrator.bandit_config.ab_test_ratio == 0.5
    assert len(bandit_orchestrator.tools) == 4
    assert bandit_orchestrator._total_decisions == 0


def test_ab_testing_split(bandit_orchestrator):
    """Test A/B testing traffic split."""
    # Run many decisions to check ratio
    n_trials = 1000
    bandit_count = 0

    for _ in range(n_trials):
        if bandit_orchestrator._should_use_bandit():
            bandit_count += 1

    # Should be close to 50% (allow ±5% variance)
    ratio = bandit_count / n_trials
    assert 0.45 <= ratio <= 0.55, f"A/B ratio {ratio} outside expected range"


@pytest.mark.asyncio
async def test_weave_with_bandit(bandit_orchestrator):
    """Test weaving with bandit enabled."""
    query = Query(text="What is Thompson Sampling?")

    spacetime = await bandit_orchestrator.weave(query)

    assert spacetime is not None
    assert hasattr(spacetime, 'metadata')
    assert 'bandit_used' in spacetime.metadata


@pytest.mark.asyncio
async def test_metrics_tracking(bandit_orchestrator):
    """Test that metrics are tracked."""
    queries = [
        Query(text=f"Test query {i}")
        for i in range(20)
    ]

    # Process queries
    for query in queries:
        await bandit_orchestrator.weave(query)

    # Get metrics
    metrics = bandit_orchestrator.get_bandit_metrics()

    assert metrics["total_decisions"] == 20
    assert metrics["bandit_decisions"] + metrics["baseline_decisions"] == 20
    assert 0.0 <= metrics["ab_ratio_actual"] <= 1.0


@pytest.mark.asyncio
async def test_decision_log(bandit_orchestrator):
    """Test decision logging."""
    queries = [Query(text=f"Query {i}") for i in range(10)]

    for query in queries:
        await bandit_orchestrator.weave(query)

    # Get log
    log = bandit_orchestrator.get_decision_log(limit=10)

    assert len(log) <= 10
    assert all("query" in d for d in log)
    assert all("tool" in d for d in log)


def test_metrics_reset(bandit_orchestrator):
    """Test metrics reset."""
    # Manually set some stats
    bandit_orchestrator._total_decisions = 100
    bandit_orchestrator._bandit_decisions = 50

    # Reset
    bandit_orchestrator.reset_metrics()

    assert bandit_orchestrator._total_decisions == 0
    assert bandit_orchestrator._bandit_decisions == 0
    assert len(bandit_orchestrator.decision_log) == 0


def test_create_bandit_orchestrator_factory(test_shards):
    """Test factory function."""
    config = Config.fast()

    orchestrator = create_bandit_orchestrator(
        cfg=config,
        shards=test_shards,
        enable_bandit=True,
        ab_test_ratio=0.2,
        sampler_type="discrete",
    )

    assert isinstance(orchestrator, BanditOrchestrator)
    assert orchestrator.enable_bandit is True
    assert orchestrator.bandit_config.ab_test_ratio == 0.2


@pytest.mark.asyncio
async def test_safety_integration():
    """Test safety guardrails integration."""
    config = Config.fast()
    shards = [
        MemoryShard(
            id="shard_1",
            text="Test",
            motifs=["test"],
            entities=["test_entity"],
        )
    ]

    # Enable safety
    bandit_config = BanditConfig(
        sampler_type="discrete",
        enable_safety=True,  # Enable guardrails
        ab_test_ratio=1.0,  # Always use bandit
    )

    orchestrator = BanditOrchestrator(
        cfg=config,
        shards=shards,
        bandit_config=bandit_config,
        enable_bandit=True,
    )

    # Safety should be initialized
    assert orchestrator.safety is not None

    # Process query
    query = Query(text="Test query")
    spacetime = await orchestrator.weave(query)

    # Should complete (safety allows low-risk actions)
    assert spacetime is not None


@pytest.mark.asyncio
async def test_neural_bandit_mode(test_shards):
    """Test with neural bandit mode."""
    config = Config.fast()

    bandit_config = BanditConfig(
        sampler_type="neural",
        context_dim=384,
        hidden_dims=[64],  # Small for testing
        n_ensemble=2,  # Small ensemble
        replay_warmup=5,
        train_every=10,
        ab_test_ratio=1.0,  # Always use bandit
    )

    orchestrator = BanditOrchestrator(
        cfg=config,
        shards=test_shards,
        bandit_config=bandit_config,
        enable_bandit=True,
    )

    # Process query
    query = Query(text="Test neural bandit")
    spacetime = await orchestrator.weave(query)

    assert spacetime is not None
    assert spacetime.metadata.get("bandit_used") is True


def test_reward_computation():
    """Test reward computation."""
    config = Config.fast()
    orchestrator = BanditOrchestrator(
        cfg=config,
        shards=[],
        enable_bandit=False,
    )

    # Mock spacetime
    class MockSpacetime:
        confidence = 0.9

    spacetime = MockSpacetime()

    # Test reward
    reward = orchestrator._compute_reward(
        spacetime,
        latency_ms=100.0,
        cost=0.01,
    )

    # Should be close to confidence (with small penalties)
    assert 0.0 <= reward <= 1.0
    assert reward < 0.9  # Penalties applied


@pytest.mark.asyncio
async def test_bandit_learning_over_time(test_shards):
    """Test that bandit learns over multiple queries."""
    config = Config.fast()

    bandit_config = BanditConfig(
        sampler_type="discrete",
        ab_test_ratio=1.0,  # Always use bandit
        replay_warmup=5,
        enable_monitoring=True,
    )

    orchestrator = BanditOrchestrator(
        cfg=config,
        shards=test_shards,
        bandit_config=bandit_config,
        enable_bandit=True,
    )

    # Process many queries
    for i in range(50):
        query = Query(text=f"Query {i}")
        await orchestrator.weave(query)

    # Check metrics
    metrics = orchestrator.get_bandit_metrics()

    assert metrics["total_decisions"] == 50
    assert metrics["count"] > 0  # Evaluator has data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
