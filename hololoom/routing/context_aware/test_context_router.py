"""
Tests for Context-Aware Routing

Tests context enrichment, routing strategies, personalization, and A/B testing.

Author: HoloLoom B2B Framework
Date: November 2025
"""

import pytest
from hololoom.routing.context_aware import (
    ContextAwareRouter,
    UserContext,
    RoutingStrategy,
    PersonalizationEngine,
    ABTestRouter,
    ABTestConfig,
    RoutingVariant
)


@pytest.mark.asyncio
async def test_context_router_rule_based():
    """Test rule-based routing strategy"""
    router = ContextAwareRouter(strategy=RoutingStrategy.RULE_BASED)

    decision = await router.route(
        query="What is Thompson Sampling?",
        user_context=UserContext(user_id="test_user", session_id="session1")
    )

    assert decision.department_id in ["rag", "planning", "orchestration"]
    assert 0.0 <= decision.confidence <= 1.0
    assert decision.strategy_used == RoutingStrategy.RULE_BASED
    assert 0.0 <= decision.context_quality <= 1.0


@pytest.mark.asyncio
async def test_context_router_personalized():
    """Test personalized routing with user preferences"""
    router = ContextAwareRouter(
        strategy=RoutingStrategy.PERSONALIZED,
        enable_personalization=True
    )

    # First query - no preferences yet
    decision1 = await router.route(
        query="Explain machine learning",
        user_context=UserContext(user_id="user123", session_id="s1")
    )

    # Provide positive feedback
    await router.learn_from_feedback(
        user_id="user123",
        department=decision1.department_id,
        outcome="success",
        confidence=0.95
    )

    # Second query - should prefer same department
    decision2 = await router.route(
        query="What is deep learning?",
        user_context=UserContext(user_id="user123", session_id="s2")
    )

    # User preference should influence routing
    assert decision2.strategy_used == RoutingStrategy.PERSONALIZED


@pytest.mark.asyncio
async def test_context_enrichment():
    """Test context enrichment via ContextDepartment"""
    router = ContextAwareRouter(strategy=RoutingStrategy.RULE_BASED)

    user_context = UserContext(
        user_id="user456",
        session_id="session2",
        role="data_scientist",
        history=[
            {"query": "Previous query", "department": "rag"}
        ]
    )

    decision = await router.route(
        query="Analyze dataset",
        user_context=user_context,
        enrich_context=True  # Enable enrichment
    )

    # Context should be enriched
    assert decision.context_quality > 0.5


def test_personalization_engine():
    """Test personalization engine preference learning"""
    engine = PersonalizationEngine(enable_collaborative_filtering=False)

    # Update preference - success
    engine.update_preference(
        user_id="alice",
        department="rag",
        outcome="success",
        confidence=0.92
    )

    profile = engine.get_profile("alice")
    assert profile.total_queries == 1
    assert profile.successful_queries == 1
    assert profile.preferred_departments["rag"] > 0.0

    # Update preference - failure
    engine.update_preference(
        user_id="alice",
        department="planning",
        outcome="failure",
        confidence=0.45
    )

    profile = engine.get_profile("alice")
    assert profile.total_queries == 2
    assert profile.successful_queries == 1
    assert profile.preferred_departments["planning"] < 0.0


def test_collaborative_filtering():
    """Test collaborative filtering for cold start"""
    engine = PersonalizationEngine(enable_collaborative_filtering=True)

    # Create similar users
    for i in range(5):
        engine.update_preference(f"user{i}", "rag", "success", 0.9)
        engine.update_preference(f"user{i}", "planning", "success", 0.8)

    # New user (cold start)
    recommendations = engine.get_recommendation("new_user", top_k=2)

    # Should recommend based on similar users
    assert len(recommendations) >= 1
    assert all(dept in ["rag", "planning"] for dept, _ in recommendations)


def test_ab_test_assignment():
    """Test A/B test variant assignment"""
    config = ABTestConfig(
        test_name="rule_vs_ml",
        variants={
            RoutingVariant.CONTROL: 0.5,
            RoutingVariant.VARIANT_A: 0.5
        }
    )

    ab_router = ABTestRouter(config)

    # Same user should get same variant (sticky)
    variant1 = ab_router.assign_variant("user123")
    variant2 = ab_router.assign_variant("user123")
    assert variant1 == variant2

    # Different users should get mix
    variants = [
        ab_router.assign_variant(f"user{i}")
        for i in range(100)
    ]

    # Should have both variants
    assert RoutingVariant.CONTROL in variants
    assert RoutingVariant.VARIANT_A in variants


def test_ab_test_results():
    """Test A/B test statistical analysis"""
    config = ABTestConfig(
        test_name="test",
        variants={
            RoutingVariant.CONTROL: 0.5,
            RoutingVariant.VARIANT_A: 0.5
        },
        min_sample_size=10
    )

    ab_router = ABTestRouter(config)

    # Record outcomes for control (worse)
    for _ in range(20):
        ab_router.record_outcome(
            RoutingVariant.CONTROL,
            confidence=0.70,
            latency_ms=200,
            success=True
        )

    # Record outcomes for variant A (better)
    for _ in range(20):
        ab_router.record_outcome(
            RoutingVariant.VARIANT_A,
            confidence=0.95,
            latency_ms=150,
            success=True
        )

    results = ab_router.get_results()

    assert results["ready_for_decision"] is True
    assert results["total_samples"] == 40
    assert results["winner"] == RoutingVariant.VARIANT_A
    assert results["confidence"] >= 0.9


def test_routing_metrics():
    """Test routing metrics tracking"""
    router = ContextAwareRouter(strategy=RoutingStrategy.RULE_BASED)

    metrics = router.get_metrics()

    assert "total_routes" in metrics
    assert "context_enrichments" in metrics
    assert "avg_confidence" in metrics
    assert "strategy_distribution" in metrics


@pytest.mark.asyncio
async def test_hybrid_routing():
    """Test hybrid routing (rule + ML)"""
    router = ContextAwareRouter(strategy=RoutingStrategy.HYBRID)

    decision = await router.route(
        query="Complex analysis task",
        user_context=UserContext(user_id="user789", session_id="s3")
    )

    assert decision.strategy_used == RoutingStrategy.HYBRID
    assert decision.department_id is not None
    assert len(decision.alternatives) > 0  # Should have alternatives


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
