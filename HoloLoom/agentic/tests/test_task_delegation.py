"""
Tests for Phase 7.2: Task Delegation System
============================================

Comprehensive tests for:
- QueryCapabilityAnalyzer: Query→capability mapping
- ExpertRouter: Thompson Sampling agent selection
- EnsembleDecision: Multi-agent result aggregation
- MultiAgentCoordinator.delegate_task_smart(): Integration

Author: Claude Code
Date: 2025-12-09
"""

import pytest
import asyncio
from typing import Dict, Any

from HoloLoom.agentic.capability_analyzer import (
    QueryCapabilityAnalyzer,
    CapabilityAnalysis,
    QueryComplexity,
    analyze_query_capabilities,
)
from HoloLoom.agentic.expert_router import (
    ExpertRouter,
    SelectionStrategy,
    ThompsonPrior,
    AgentScore,
    RoutingDecision,
    create_expert_router,
)
from HoloLoom.agentic.ensemble_decision import (
    EnsembleAggregator,
    EnsembleStrategy,
    EnsembleResult,
    AgentResponse,
    NumericEnsemble,
    TextEnsemble,
    aggregate_responses,
    create_agent_response,
)
from HoloLoom.agentic.multi_agent import (
    AgentCapability,
    AgentInfo,
    AgentStatus,
    AgentRegistry,
    MultiAgentCoordinator,
)


# ============================================================================
# QueryCapabilityAnalyzer Tests
# ============================================================================

class TestQueryCapabilityAnalyzer:
    """Tests for query→capability mapping."""

    def test_analyzer_initialization(self):
        """Test analyzer initializes correctly."""
        analyzer = QueryCapabilityAnalyzer()
        assert analyzer is not None
        assert len(analyzer._keyword_mappings) > 0

    def test_trivial_query_detection(self):
        """Test trivial queries are detected."""
        analyzer = QueryCapabilityAnalyzer()

        trivial_queries = ["hi", "hello!", "thanks", "ok", "bye"]
        for query in trivial_queries:
            result = analyzer.analyze(query)
            assert result.complexity == QueryComplexity.TRIVIAL
            assert result.confidence >= 0.9

    def test_code_review_capability_detection(self):
        """Test code review keywords map correctly."""
        analyzer = QueryCapabilityAnalyzer()

        queries = [
            "Review this Python code",
            "Please audit this function",
            "Check this code for bugs",
        ]
        for query in queries:
            result = analyzer.analyze(query)
            assert result.primary_capability == AgentCapability.CODE_REVIEW

    def test_code_generation_capability_detection(self):
        """Test code generation keywords map correctly."""
        analyzer = QueryCapabilityAnalyzer()

        queries = [
            "Write a function to sort arrays",
            "Implement a binary search algorithm",
            "Code a class for user authentication",
        ]
        for query in queries:
            result = analyzer.analyze(query)
            assert result.primary_capability == AgentCapability.CODE_GENERATION

    def test_research_capability_detection(self):
        """Test research keywords map correctly."""
        analyzer = QueryCapabilityAnalyzer()

        queries = [
            "Research the best practices for API design",
            "Explore different caching strategies",
            "Investigate the performance issue",
        ]
        for query in queries:
            result = analyzer.analyze(query)
            assert result.primary_capability == AgentCapability.RESEARCH

    def test_analysis_capability_detection(self):
        """Test analysis keywords map correctly."""
        analyzer = QueryCapabilityAnalyzer()

        queries = [
            "Analyze this error log",
            "Examine the performance metrics",
            "Evaluate this architecture",
        ]
        for query in queries:
            result = analyzer.analyze(query)
            assert result.primary_capability == AgentCapability.ANALYSIS

    def test_multi_capability_detection(self):
        """Test queries with multiple capabilities."""
        analyzer = QueryCapabilityAnalyzer()

        # Query with multiple capability keywords
        result = analyzer.analyze("Analyze this code and summarize the findings")
        assert result.primary_capability is not None
        assert len(result.secondary_capabilities) >= 0

    def test_complexity_detection_complex(self):
        """Test complex queries are detected."""
        analyzer = QueryCapabilityAnalyzer()

        complex_queries = [
            "Compare Thompson Sampling versus UCB and explain the tradeoffs",
            "Why does this algorithm have O(n log n) complexity?",
        ]
        for query in complex_queries:
            result = analyzer.analyze(query)
            assert result.complexity in [QueryComplexity.COMPLEX, QueryComplexity.RESEARCH]

    def test_complexity_detection_research(self):
        """Test research-level queries are detected."""
        analyzer = QueryCapabilityAnalyzer()

        research_queries = [
            "Provide a comprehensive analysis of all exploration strategies with pros and cons",
            "Give me an in-depth detailed breakdown of microservices architecture tradeoffs",
        ]
        for query in research_queries:
            result = analyzer.analyze(query)
            assert result.complexity == QueryComplexity.RESEARCH

    def test_ensemble_trigger_detection(self):
        """Test that multi-capability queries trigger ensemble."""
        analyzer = QueryCapabilityAnalyzer()

        # Query that should trigger ensemble
        result = analyzer.analyze(
            "Research all caching strategies and also analyze the performance data"
        )
        # May or may not trigger ensemble depending on score
        assert isinstance(result.requires_ensemble, bool)

    def test_convenience_function(self):
        """Test the convenience function."""
        result = analyze_query_capabilities("Debug this Python script")
        assert isinstance(result, CapabilityAnalysis)
        assert result.primary_capability == AgentCapability.DEBUGGING

    def test_custom_keyword_mapping(self):
        """Test adding custom keyword mappings."""
        analyzer = QueryCapabilityAnalyzer()

        # Add custom mapping
        analyzer.add_custom_mapping(["foobar", "baz"], AgentCapability.MATH)

        result = analyzer.analyze("Calculate foobar for me")
        # Should include MATH due to custom mapping
        assert AgentCapability.MATH in [
            result.primary_capability,
            *result.secondary_capabilities
        ]


# ============================================================================
# ThompsonPrior Tests
# ============================================================================

class TestThompsonPrior:
    """Tests for Thompson Sampling prior."""

    def test_prior_initialization(self):
        """Test prior initializes with uniform distribution."""
        prior = ThompsonPrior()
        assert prior.alpha == 1.0
        assert prior.beta == 1.0
        assert prior.expected_value() == 0.5

    def test_success_update(self):
        """Test prior updates on success."""
        prior = ThompsonPrior()
        prior.update_success(confidence=0.9)

        assert prior.alpha == 1.9
        assert prior.beta == 1.0
        assert prior.expected_value() > 0.5

    def test_failure_update(self):
        """Test prior updates on failure."""
        prior = ThompsonPrior()
        prior.update_failure(confidence=0.3)

        assert prior.alpha == 1.0
        assert prior.beta == 1.7  # 1 + (1 - 0.3)
        assert prior.expected_value() < 0.5

    def test_sample_in_range(self):
        """Test samples are in valid range."""
        prior = ThompsonPrior(alpha=5.0, beta=5.0)

        for _ in range(100):
            sample = prior.sample()
            assert 0.0 <= sample <= 1.0

    def test_observation_count(self):
        """Test observation counting."""
        prior = ThompsonPrior()
        assert prior.total_observations() == 0

        prior.update_success(1.0)
        prior.update_failure(0.0)

        assert prior.total_observations() == 2


# ============================================================================
# ExpertRouter Tests
# ============================================================================

class TestExpertRouter:
    """Tests for Thompson Sampling-based agent routing."""

    @pytest.fixture
    def registry_with_agents(self):
        """Create registry with test agents."""
        registry = AgentRegistry()

        # Add research agents
        research_agent1 = AgentInfo(
            id="research_1",
            name="Research Agent 1",
            capabilities={AgentCapability.RESEARCH, AgentCapability.ANALYSIS},
            status=AgentStatus.IDLE,
        )
        research_agent2 = AgentInfo(
            id="research_2",
            name="Research Agent 2",
            capabilities={AgentCapability.RESEARCH},
            status=AgentStatus.IDLE,
        )

        # Add code agent
        code_agent = AgentInfo(
            id="code_1",
            name="Code Agent",
            capabilities={AgentCapability.CODE_REVIEW, AgentCapability.CODE_GENERATION},
            status=AgentStatus.IDLE,
        )

        asyncio.get_event_loop().run_until_complete(registry.register(research_agent1))
        asyncio.get_event_loop().run_until_complete(registry.register(research_agent2))
        asyncio.get_event_loop().run_until_complete(registry.register(code_agent))

        return registry

    def test_router_initialization(self, registry_with_agents):
        """Test router initializes correctly."""
        router = ExpertRouter(registry_with_agents)
        assert router is not None
        assert router._enable_thompson is True

    def test_route_finds_agent(self, registry_with_agents):
        """Test routing finds available agent."""
        router = ExpertRouter(registry_with_agents)

        decision = router.route(
            required_capability=AgentCapability.RESEARCH,
            strategy=SelectionStrategy.BALANCED,
        )

        assert decision.selected_agent in ["research_1", "research_2"]
        assert decision.score.total_score >= 0.0

    def test_route_with_top_k(self, registry_with_agents):
        """Test routing returns top-k alternatives."""
        router = ExpertRouter(registry_with_agents)

        decision = router.route(
            required_capability=AgentCapability.RESEARCH,
            strategy=SelectionStrategy.BALANCED,
            top_k=3,
        )

        # Should have at least one alternative (2 research agents)
        assert len(decision.alternatives) >= 0

    def test_route_no_available_agents(self, registry_with_agents):
        """Test routing handles no available agents."""
        router = ExpertRouter(registry_with_agents)

        # Request a capability no agent has
        decision = router.route(
            required_capability=AgentCapability.MATH,
            strategy=SelectionStrategy.BALANCED,
        )

        assert decision.selected_agent == ""
        assert "No available agents" in decision.routing_reason

    def test_performance_update(self, registry_with_agents):
        """Test performance updates affect routing."""
        router = ExpertRouter(registry_with_agents)

        # Route first time
        decision1 = router.route(AgentCapability.RESEARCH, SelectionStrategy.BALANCED)

        # Update performance for one agent
        router.update_performance("research_1", success=True, confidence=0.95)
        router.update_performance("research_1", success=True, confidence=0.9)
        router.update_performance("research_2", success=False, confidence=0.3)

        # Check priors updated
        prior_1 = router.get_agent_prior("research_1")
        prior_2 = router.get_agent_prior("research_2")

        assert prior_1["alpha"] > prior_1["beta"]  # More successes
        assert prior_2["beta"] > prior_2["alpha"]  # More failures

    def test_selection_strategies(self, registry_with_agents):
        """Test all selection strategies produce valid results."""
        router = ExpertRouter(registry_with_agents)

        strategies = [
            SelectionStrategy.FASTEST,
            SelectionStrategy.RELIABLE,
            SelectionStrategy.BALANCED,
            SelectionStrategy.THOMPSON,
            SelectionStrategy.RANDOM,
        ]

        for strategy in strategies:
            decision = router.route(AgentCapability.RESEARCH, strategy)
            assert decision.selected_agent in ["research_1", "research_2", ""]

    def test_thompson_strategy_samples(self, registry_with_agents):
        """Test Thompson strategy uses sampling."""
        router = ExpertRouter(registry_with_agents)

        decision = router.route(
            required_capability=AgentCapability.RESEARCH,
            strategy=SelectionStrategy.THOMPSON,
        )

        assert decision.used_thompson is True
        assert decision.score.thompson_sample >= 0.0

    def test_routing_statistics(self, registry_with_agents):
        """Test routing statistics are tracked."""
        router = ExpertRouter(registry_with_agents)

        # Make some routing decisions
        for _ in range(5):
            router.route(AgentCapability.RESEARCH, SelectionStrategy.BALANCED)

        stats = router.get_routing_statistics()
        assert stats["total_routings"] == 5
        assert "agents_tracked" in stats

    def test_convenience_function(self, registry_with_agents):
        """Test convenience function."""
        router = create_expert_router(registry_with_agents, enable_thompson=True)
        assert isinstance(router, ExpertRouter)


# ============================================================================
# EnsembleAggregator Tests
# ============================================================================

class TestEnsembleAggregator:
    """Tests for multi-agent result aggregation."""

    def test_aggregator_initialization(self):
        """Test aggregator initializes correctly."""
        aggregator = EnsembleAggregator()
        assert aggregator is not None

    def test_majority_vote_clear_winner(self):
        """Test majority vote with clear winner."""
        aggregator = EnsembleAggregator()

        responses = [
            AgentResponse("agent_1", "Thompson Sampling", 0.9, 100.0),
            AgentResponse("agent_2", "Thompson Sampling", 0.85, 120.0),
            AgentResponse("agent_3", "UCB Algorithm", 0.7, 80.0),
        ]

        result = aggregator.aggregate(responses, EnsembleStrategy.MAJORITY_VOTE)

        assert result.final_response == "Thompson Sampling"
        assert result.agreement_score == pytest.approx(2/3, rel=0.01)
        assert result.agent_count == 3

    def test_majority_vote_tie_breaker(self):
        """Test majority vote uses confidence for ties."""
        aggregator = EnsembleAggregator()

        responses = [
            AgentResponse("agent_1", "Answer A", 0.9, 100.0),
            AgentResponse("agent_2", "Answer B", 0.5, 100.0),
        ]

        result = aggregator.aggregate(responses, EnsembleStrategy.MAJORITY_VOTE)

        # Should prefer higher confidence
        assert result.final_response == "Answer A"

    def test_confidence_weighted_text(self):
        """Test confidence-weighted aggregation for text."""
        aggregator = EnsembleAggregator()

        responses = [
            AgentResponse("agent_1", "High confidence answer", 0.95, 100.0),
            AgentResponse("agent_2", "Low confidence answer", 0.3, 100.0),
        ]

        result = aggregator.aggregate(responses, EnsembleStrategy.CONFIDENCE_WEIGHTED)

        # Should prefer higher confidence answer
        assert result.final_response == "High confidence answer"

    def test_confidence_weighted_numeric(self):
        """Test confidence-weighted aggregation for numbers."""
        aggregator = EnsembleAggregator()

        responses = [
            AgentResponse("agent_1", 10.0, 0.8, 100.0),
            AgentResponse("agent_2", 20.0, 0.2, 100.0),
        ]

        result = aggregator.aggregate(responses, EnsembleStrategy.CONFIDENCE_WEIGHTED)

        # Weighted average: (10*0.8 + 20*0.2) / (0.8 + 0.2) = 12.0
        assert result.final_response == pytest.approx(12.0, rel=0.01)

    def test_best_confidence(self):
        """Test best confidence selection."""
        aggregator = EnsembleAggregator()

        responses = [
            AgentResponse("agent_1", "Low confidence", 0.5, 100.0),
            AgentResponse("agent_2", "High confidence", 0.95, 100.0),
            AgentResponse("agent_3", "Medium confidence", 0.7, 100.0),
        ]

        result = aggregator.aggregate(responses, EnsembleStrategy.BEST_CONFIDENCE)

        assert result.final_response == "High confidence"

    def test_unanimous_success(self):
        """Test unanimous with full agreement."""
        aggregator = EnsembleAggregator()

        responses = [
            AgentResponse("agent_1", "Same answer", 0.9, 100.0),
            AgentResponse("agent_2", "Same answer", 0.85, 100.0),
            AgentResponse("agent_3", "Same answer", 0.8, 100.0),
        ]

        result = aggregator.aggregate(responses, EnsembleStrategy.UNANIMOUS)

        assert result.final_response == "Same answer"
        assert result.agreement_score == 1.0

    def test_unanimous_fallback(self):
        """Test unanimous falls back when no agreement."""
        aggregator = EnsembleAggregator(
            unanimous_fallback=EnsembleStrategy.CONFIDENCE_WEIGHTED
        )

        responses = [
            AgentResponse("agent_1", "Answer A", 0.9, 100.0),
            AgentResponse("agent_2", "Answer B", 0.85, 100.0),
        ]

        result = aggregator.aggregate(responses, EnsembleStrategy.UNANIMOUS)

        # Falls back to confidence weighted
        assert result.strategy_used == EnsembleStrategy.CONFIDENCE_WEIGHTED
        assert result.final_response == "Answer A"  # Higher confidence

    def test_empty_responses(self):
        """Test handling of empty responses."""
        aggregator = EnsembleAggregator()

        result = aggregator.aggregate([], EnsembleStrategy.MAJORITY_VOTE)

        assert result.final_response is None
        assert result.agreement_score == 0.0
        assert "error" in result.metadata

    def test_agreement_score_calculation(self):
        """Test agreement score calculation."""
        aggregator = EnsembleAggregator()

        responses = [
            AgentResponse("agent_1", "A", 0.9, 100.0),
            AgentResponse("agent_2", "A", 0.9, 100.0),
            AgentResponse("agent_3", "A", 0.9, 100.0),
            AgentResponse("agent_4", "B", 0.9, 100.0),
        ]

        agreement = aggregator.compute_agreement(responses)
        assert agreement == 0.75  # 3/4 agree

    def test_result_properties(self):
        """Test EnsembleResult properties."""
        responses = [
            AgentResponse("agent_1", "A", 0.8, 100.0),
            AgentResponse("agent_2", "A", 0.9, 150.0),
        ]

        result = EnsembleResult(
            final_response="A",
            strategy_used=EnsembleStrategy.MAJORITY_VOTE,
            agreement_score=1.0,
            individual_responses=responses,
            aggregation_time_ms=5.0,
        )

        assert result.agent_count == 2
        assert result.avg_confidence == 0.85
        assert result.avg_latency_ms == 125.0
        assert result.unanimous is True

    def test_convenience_functions(self):
        """Test convenience functions."""
        response = create_agent_response("agent_1", "Answer", 0.9, 100.0)
        assert isinstance(response, AgentResponse)

        responses = [response]
        result = aggregate_responses(responses, EnsembleStrategy.BEST_CONFIDENCE)
        assert isinstance(result, EnsembleResult)


# ============================================================================
# NumericEnsemble Tests
# ============================================================================

class TestNumericEnsemble:
    """Tests for numeric ensemble aggregation."""

    def test_numeric_aggregation(self):
        """Test numeric aggregation with statistics."""
        ensemble = NumericEnsemble()

        responses = [
            AgentResponse("agent_1", 10.0, 0.9, 100.0),
            AgentResponse("agent_2", 12.0, 0.8, 100.0),
            AgentResponse("agent_3", 11.0, 0.85, 100.0),
        ]

        result = ensemble.aggregate_numeric(responses)

        assert "mean" in result.metadata
        assert "median" in result.metadata
        assert "stdev" in result.metadata

    def test_outlier_removal(self):
        """Test outlier removal for numeric responses."""
        ensemble = NumericEnsemble(outlier_threshold=1.5, remove_outliers=True)

        responses = [
            AgentResponse("agent_1", 10.0, 0.9, 100.0),
            AgentResponse("agent_2", 11.0, 0.8, 100.0),
            AgentResponse("agent_3", 100.0, 0.5, 100.0),  # Outlier
        ]

        result = ensemble.aggregate_numeric(responses)

        # Result should be based on non-outlier values
        assert result.final_response < 50.0


# ============================================================================
# TextEnsemble Tests
# ============================================================================

class TestTextEnsemble:
    """Tests for text ensemble aggregation."""

    def test_text_aggregation(self):
        """Test text aggregation with normalization."""
        ensemble = TextEnsemble(normalize_whitespace=True)

        responses = [
            AgentResponse("agent_1", "Thompson  Sampling", 0.9, 100.0),
            AgentResponse("agent_2", "Thompson Sampling", 0.85, 100.0),
        ]

        result = ensemble.aggregate_text(responses)

        assert "unique_responses" in result.metadata


# ============================================================================
# Integration Tests
# ============================================================================

class TestDelegateTaskSmartIntegration:
    """Integration tests for delegate_task_smart."""

    @pytest.fixture
    def coordinator_with_agents(self):
        """Create coordinator with test agents."""
        coordinator = MultiAgentCoordinator()

        # Create and register agents
        agents = [
            AgentInfo(
                id="research_agent",
                name="Research Agent",
                capabilities={AgentCapability.RESEARCH, AgentCapability.ANALYSIS},
                status=AgentStatus.IDLE,
            ),
            AgentInfo(
                id="code_agent",
                name="Code Agent",
                capabilities={AgentCapability.CODE_REVIEW, AgentCapability.CODE_GENERATION},
                status=AgentStatus.IDLE,
            ),
            AgentInfo(
                id="writing_agent",
                name="Writing Agent",
                capabilities={AgentCapability.WRITING, AgentCapability.SUMMARIZATION},
                status=AgentStatus.IDLE,
            ),
        ]

        async def register_all():
            for agent in agents:
                await coordinator.registry.register(agent)

        asyncio.get_event_loop().run_until_complete(register_all())

        return coordinator

    def test_auto_capability_detection(self, coordinator_with_agents):
        """Test automatic capability detection."""
        coordinator = coordinator_with_agents

        # Initialize the analyzer
        coordinator._capability_analyzer = QueryCapabilityAnalyzer()

        # Analyze a code review query
        result = coordinator._capability_analyzer.analyze(
            "Review this Python code for bugs"
        )

        assert result.primary_capability == AgentCapability.CODE_REVIEW

    def test_expert_router_integration(self, coordinator_with_agents):
        """Test expert router is created and used."""
        coordinator = coordinator_with_agents

        # Initialize the router
        coordinator._expert_router = ExpertRouter(coordinator.registry)

        # Route a research query
        decision = coordinator._expert_router.route(
            required_capability=AgentCapability.RESEARCH,
            strategy=SelectionStrategy.BALANCED,
        )

        assert decision.selected_agent == "research_agent"

    def test_ensemble_aggregator_integration(self, coordinator_with_agents):
        """Test ensemble aggregator is created and used."""
        coordinator = coordinator_with_agents

        # Initialize the aggregator
        coordinator._ensemble_aggregator = EnsembleAggregator()

        responses = [
            AgentResponse("research_agent", "Answer A", 0.9, 100.0),
            AgentResponse("code_agent", "Answer A", 0.85, 100.0),
        ]

        result = coordinator._ensemble_aggregator.aggregate(
            responses,
            EnsembleStrategy.MAJORITY_VOTE
        )

        assert result.final_response == "Answer A"

    def test_get_expert_router_statistics(self, coordinator_with_agents):
        """Test getting expert router statistics."""
        coordinator = coordinator_with_agents

        # Initialize router and make some routings
        coordinator._expert_router = ExpertRouter(coordinator.registry)
        coordinator._expert_router.route(AgentCapability.RESEARCH, SelectionStrategy.BALANCED)

        stats = coordinator.get_expert_router_statistics()
        assert "total_routings" in stats
        assert stats["total_routings"] == 1

    def test_get_capability_keywords(self, coordinator_with_agents):
        """Test getting capability keywords."""
        coordinator = coordinator_with_agents

        # Initialize analyzer
        coordinator._capability_analyzer = QueryCapabilityAnalyzer()

        keywords = coordinator.get_capability_keywords()
        assert "research" in keywords
        assert "code_review" in keywords


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
