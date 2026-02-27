"""
Unit tests for Moonshot Swarm validation framework.

Tests cover:
- Pipeline validators (Quality, Latency, UserSatisfaction, Statistical)
- Pipeline composition and parallel execution
- Swarm agent filtering by dimension
- Swarm orchestration and results aggregation

Created: December 2025
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any

from hololoom.prompting.validation.pipeline import (
    ValidationPipeline,
    ValidationPhase,
    ValidationContext,
    QualityValidator,
    LatencyValidator,
    UserSatisfactionValidator,
    StatisticalValidator,
    PipelineResults,
)
from hololoom.prompting.validation.swarm import (
    MoonshotSwarm,
    SwarmAgent,
    SwarmResults,
    SwarmDimension,
)


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def baseline_queries() -> List[Dict[str, Any]]:
    """Create baseline query data with various metrics."""
    return [
        {
            "quality_score": 0.72 + (i % 10) * 0.01,
            "latency_ms": 140 + (i % 5) * 2,
            "model_provider": ["claude", "gemini", "gpt"][i % 3],
            "system": ["recursive", "skills", "rag", "memory"][i % 4],
            "complexity": ["simple", "moderate", "complex"][i % 3],
            "timestamp": datetime.now() - timedelta(days=i % 28),
        }
        for i in range(100)
    ]


@pytest.fixture
def mrf_queries() -> List[Dict[str, Any]]:
    """Create MRF query data with improved metrics."""
    return [
        {
            "quality_score": 0.91 + (i % 10) * 0.01,
            "latency_ms": 155 + (i % 5) * 2,
            "user_rating": 4 + (i % 2),
            "thumbs_up": (i % 3) > 0,
            "model_provider": ["claude", "gemini", "gpt"][i % 3],
            "system": ["recursive", "skills", "rag", "memory"][i % 4],
            "complexity": ["simple", "moderate", "complex"][i % 3],
            "timestamp": datetime.now() - timedelta(days=i % 28),
        }
        for i in range(100)
    ]


@pytest.fixture
def low_improvement_mrf() -> List[Dict[str, Any]]:
    """MRF queries with low improvement for testing failures."""
    return [
        {
            "quality_score": 0.75 + (i % 5) * 0.01,  # Only ~5% improvement
            "latency_ms": 180 + (i % 5) * 2,  # Higher latency (regression)
            "user_rating": 3,  # Low rating
            "thumbs_up": False,
        }
        for i in range(50)
    ]


# =============================================================================
# Pipeline Validator Tests
# =============================================================================

class TestQualityValidator:
    """Tests for QualityValidator."""

    @pytest.mark.asyncio
    async def test_quality_validator_pass(self, baseline_queries, mrf_queries):
        """Quality improvement >= threshold should pass."""
        validator = QualityValidator(min_improvement_pct=20.0)
        context = ValidationContext(
            phase=ValidationPhase.ANALYSIS,
            baseline_queries=baseline_queries,
            mrf_queries=mrf_queries,
        )

        result = await validator.validate(context)

        assert result["status"] == "pass"
        assert result["improvement_pct"] > 20.0
        assert result["meets_target"] is True
        assert result["baseline_avg"] < result["mrf_avg"]

    @pytest.mark.asyncio
    async def test_quality_validator_fail(self, baseline_queries, low_improvement_mrf):
        """Quality improvement < threshold should fail."""
        validator = QualityValidator(min_improvement_pct=20.0)
        context = ValidationContext(
            phase=ValidationPhase.ANALYSIS,
            baseline_queries=baseline_queries,
            mrf_queries=low_improvement_mrf,
        )

        result = await validator.validate(context)

        assert result["status"] == "fail"
        assert result["improvement_pct"] < 20.0
        assert result["meets_target"] is False

    @pytest.mark.asyncio
    async def test_quality_validator_insufficient_data(self):
        """Empty queries should return insufficient_data status."""
        validator = QualityValidator(min_improvement_pct=10.0)
        context = ValidationContext(
            phase=ValidationPhase.ANALYSIS,
            baseline_queries=[],
            mrf_queries=[],
        )

        result = await validator.validate(context)

        assert result["status"] == "insufficient_data"


class TestLatencyValidator:
    """Tests for LatencyValidator."""

    @pytest.mark.asyncio
    async def test_latency_validator_pass(self, baseline_queries, mrf_queries):
        """Latency regression within limit should pass."""
        validator = LatencyValidator(max_regression_pct=20.0)
        context = ValidationContext(
            phase=ValidationPhase.ANALYSIS,
            baseline_queries=baseline_queries,
            mrf_queries=mrf_queries,
        )

        result = await validator.validate(context)

        assert result["status"] == "pass"
        assert result["change_pct"] <= 20.0
        assert result["acceptable"] is True

    @pytest.mark.asyncio
    async def test_latency_validator_fail(self, baseline_queries, low_improvement_mrf):
        """Latency regression exceeding limit should fail."""
        validator = LatencyValidator(max_regression_pct=10.0)
        context = ValidationContext(
            phase=ValidationPhase.ANALYSIS,
            baseline_queries=baseline_queries,
            mrf_queries=low_improvement_mrf,
        )

        result = await validator.validate(context)

        assert result["status"] == "fail"
        assert result["change_pct"] > 10.0
        assert result["acceptable"] is False


class TestUserSatisfactionValidator:
    """Tests for UserSatisfactionValidator."""

    @pytest.mark.asyncio
    async def test_user_satisfaction_pass(self, mrf_queries):
        """Average rating >= threshold should pass."""
        validator = UserSatisfactionValidator(min_rating=4.0)
        context = ValidationContext(
            phase=ValidationPhase.ANALYSIS,
            baseline_queries=[],
            mrf_queries=mrf_queries,
        )

        result = await validator.validate(context)

        assert result["status"] == "pass"
        assert result["avg_rating"] >= 4.0
        assert result["meets_target"] is True
        assert "thumbs_up_rate" in result

    @pytest.mark.asyncio
    async def test_user_satisfaction_fail(self, low_improvement_mrf):
        """Average rating < threshold should fail."""
        validator = UserSatisfactionValidator(min_rating=4.0)
        context = ValidationContext(
            phase=ValidationPhase.ANALYSIS,
            baseline_queries=[],
            mrf_queries=low_improvement_mrf,
        )

        result = await validator.validate(context)

        assert result["status"] == "fail"
        assert result["avg_rating"] < 4.0


class TestStatisticalValidator:
    """Tests for StatisticalValidator."""

    @pytest.mark.asyncio
    async def test_statistical_validator_sufficient_samples(self, baseline_queries, mrf_queries):
        """StatisticalValidator with >= 30 samples should run t-test."""
        validator = StatisticalValidator(significance_level=0.05)
        context = ValidationContext(
            phase=ValidationPhase.ANALYSIS,
            baseline_queries=baseline_queries,  # 100 samples
            mrf_queries=mrf_queries,  # 100 samples
        )

        result = await validator.validate(context)

        # Should have run full statistical analysis
        assert result["status"] in ["pass", "fail"]
        assert "p_value" in result
        assert "is_significant" in result
        assert "effect_size" in result
        assert "confidence_interval" in result
        assert result["p_value"] >= 0.0
        assert result["p_value"] <= 1.0

    @pytest.mark.asyncio
    async def test_statistical_validator_insufficient_samples(self):
        """StatisticalValidator with < 30 samples should return insufficient_data."""
        validator = StatisticalValidator(significance_level=0.05)

        # Create small sample sets (< 30)
        small_baseline = [{"quality_score": 0.7 + i * 0.01} for i in range(20)]
        small_mrf = [{"quality_score": 0.9 + i * 0.01} for i in range(20)]

        context = ValidationContext(
            phase=ValidationPhase.ANALYSIS,
            baseline_queries=small_baseline,
            mrf_queries=small_mrf,
        )

        result = await validator.validate(context)

        assert result["status"] == "insufficient_data"
        assert "message" in result
        assert "30" in result["message"]

    @pytest.mark.asyncio
    async def test_statistical_validator_edge_case_exactly_30(self):
        """StatisticalValidator with exactly 30 samples should run analysis."""
        validator = StatisticalValidator(significance_level=0.05)

        # Exactly 30 samples
        baseline_30 = [{"quality_score": 0.7 + (i % 10) * 0.01} for i in range(30)]
        mrf_30 = [{"quality_score": 0.9 + (i % 10) * 0.01} for i in range(30)]

        context = ValidationContext(
            phase=ValidationPhase.ANALYSIS,
            baseline_queries=baseline_30,
            mrf_queries=mrf_30,
        )

        result = await validator.validate(context)

        # Should run full analysis (exactly 30 is the threshold)
        assert result["status"] in ["pass", "fail"]
        assert "p_value" in result

    @pytest.mark.asyncio
    async def test_statistical_validator_confidence_interval_structure(self, baseline_queries, mrf_queries):
        """Confidence interval should have lower and upper bounds."""
        validator = StatisticalValidator(significance_level=0.05)
        context = ValidationContext(
            phase=ValidationPhase.ANALYSIS,
            baseline_queries=baseline_queries,
            mrf_queries=mrf_queries,
        )

        result = await validator.validate(context)

        # Check confidence interval structure
        assert "confidence_interval" in result
        ci = result["confidence_interval"]
        assert "lower" in ci
        assert "upper" in ci
        assert ci["lower"] <= ci["upper"]


# =============================================================================
# Pipeline Composition Tests
# =============================================================================

class TestValidationPipeline:
    """Tests for ValidationPipeline composition and execution."""

    def test_pipeline_chaining(self):
        """Method chaining should work correctly."""
        pipeline = (
            ValidationPipeline()
            .add_validator(QualityValidator(min_improvement_pct=20.0))
            .add_validator(LatencyValidator(max_regression_pct=20.0))
            .add_validator(UserSatisfactionValidator(min_rating=4.0))
        )

        assert len(pipeline.validators) == 3
        assert pipeline.validators[0].name == "quality"
        assert pipeline.validators[1].name == "latency"
        assert pipeline.validators[2].name == "user_satisfaction"

    def test_pipeline_remove_validator(self):
        """Removing a validator by name should work."""
        pipeline = (
            ValidationPipeline()
            .add_validator(QualityValidator())
            .add_validator(LatencyValidator())
            .remove_validator("quality")
        )

        assert len(pipeline.validators) == 1
        assert pipeline.validators[0].name == "latency"

    @pytest.mark.asyncio
    async def test_pipeline_parallel_execution(self, baseline_queries, mrf_queries):
        """Validators should run in parallel."""
        pipeline = (
            ValidationPipeline()
            .add_validator(QualityValidator(min_improvement_pct=20.0))
            .add_validator(LatencyValidator(max_regression_pct=20.0))
            .add_validator(UserSatisfactionValidator(min_rating=4.0))
        )

        results = await pipeline.run(
            baseline_queries=baseline_queries,
            mrf_queries=mrf_queries,
            phase=ValidationPhase.ANALYSIS,
        )

        # All validators should have results
        assert "quality" in results.validator_results
        assert "latency" in results.validator_results
        assert "user_satisfaction" in results.validator_results

    @pytest.mark.asyncio
    async def test_pipeline_overall_pass(self, baseline_queries, mrf_queries):
        """Pipeline passes when all validators pass."""
        pipeline = (
            ValidationPipeline()
            .add_validator(QualityValidator(min_improvement_pct=20.0))
            .add_validator(LatencyValidator(max_regression_pct=20.0))
        )

        results = await pipeline.run(
            baseline_queries=baseline_queries,
            mrf_queries=mrf_queries,
        )

        assert results.passed is True
        assert results.validator_results["quality"]["status"] == "pass"
        assert results.validator_results["latency"]["status"] == "pass"


# =============================================================================
# Swarm Agent Tests
# =============================================================================

class TestSwarmAgentFiltering:
    """Tests for SwarmAgent query filtering by dimension."""

    def test_filter_by_model_provider(self, baseline_queries):
        """Agent should filter queries by model_provider."""
        pipeline = ValidationPipeline().add_validator(QualityValidator())
        agent = SwarmAgent(
            agent_id="model_claude",
            dimension=SwarmDimension.MODEL,
            config={"model_provider": "claude"},
            pipeline=pipeline,
        )

        filtered = agent._filter_queries(baseline_queries)

        assert len(filtered) > 0
        assert all(q["model_provider"] == "claude" for q in filtered)
        assert len(filtered) < len(baseline_queries)

    def test_filter_by_system(self, baseline_queries):
        """Agent should filter queries by system."""
        pipeline = ValidationPipeline().add_validator(QualityValidator())
        agent = SwarmAgent(
            agent_id="system_rag",
            dimension=SwarmDimension.SYSTEM,
            config={"system": "rag"},
            pipeline=pipeline,
        )

        filtered = agent._filter_queries(baseline_queries)

        assert len(filtered) > 0
        assert all(q["system"] == "rag" for q in filtered)

    def test_filter_by_complexity(self, baseline_queries):
        """Agent should filter queries by complexity."""
        pipeline = ValidationPipeline().add_validator(QualityValidator())
        agent = SwarmAgent(
            agent_id="complexity_simple",
            dimension=SwarmDimension.COMPLEXITY,
            config={"complexity": "simple"},
            pipeline=pipeline,
        )

        filtered = agent._filter_queries(baseline_queries)

        assert len(filtered) > 0
        assert all(q["complexity"] == "simple" for q in filtered)

    def test_filter_by_timeline(self, baseline_queries):
        """Agent should filter queries by time window."""
        pipeline = ValidationPipeline().add_validator(QualityValidator())

        # Create a time window for "week 1" (7-14 days ago)
        now = datetime.now()
        start_time = now - timedelta(days=14)
        end_time = now - timedelta(days=7)

        agent = SwarmAgent(
            agent_id="week_1",
            dimension=SwarmDimension.TIMELINE,
            config={"start_time": start_time, "end_time": end_time},
            pipeline=pipeline,
        )

        filtered = agent._filter_queries(baseline_queries)

        # Verify all filtered queries fall within the time window
        for q in filtered:
            timestamp = q.get("timestamp")
            assert timestamp is not None
            assert start_time <= timestamp <= end_time

    def test_filter_by_timeline_empty_result(self):
        """Timeline filter with no matching queries should return empty list."""
        pipeline = ValidationPipeline().add_validator(QualityValidator())

        # Time window in the distant past (no queries match)
        now = datetime.now()
        start_time = now - timedelta(days=365)  # 1 year ago
        end_time = now - timedelta(days=364)

        agent = SwarmAgent(
            agent_id="distant_past",
            dimension=SwarmDimension.TIMELINE,
            config={"start_time": start_time, "end_time": end_time},
            pipeline=pipeline,
        )

        # Queries from last 28 days only
        queries = [
            {"timestamp": datetime.now() - timedelta(days=i % 28), "quality_score": 0.8}
            for i in range(50)
        ]

        filtered = agent._filter_queries(queries)

        # No queries should match the distant past window
        assert len(filtered) == 0

    def test_filter_no_config_returns_all(self, baseline_queries):
        """Agent without filter config should return all queries."""
        pipeline = ValidationPipeline().add_validator(QualityValidator())
        agent = SwarmAgent(
            agent_id="no_filter",
            dimension=SwarmDimension.TIMELINE,
            config={},  # Empty config
            pipeline=pipeline,
        )

        filtered = agent._filter_queries(baseline_queries)

        # Should return all queries (no filtering applied)
        assert len(filtered) == len(baseline_queries)


# =============================================================================
# Swarm Edge Case Tests
# =============================================================================

class TestSwarmEdgeCases:
    """Tests for edge cases and error handling in swarm."""

    @pytest.mark.asyncio
    async def test_swarm_pass_rate_threshold_75_percent(self, baseline_queries, mrf_queries):
        """SwarmResults.overall_passed requires 75% pass rate."""
        swarm = MoonshotSwarm()
        swarm.add_model_agents(["claude", "gemini", "gpt"])

        results = await swarm.run(
            baseline_queries=baseline_queries,
            mrf_queries=mrf_queries,
        )

        results.calculate_statistics()

        # Verify 75% threshold logic
        if results.pass_rate >= 0.75:
            assert results.overall_passed is True
        else:
            assert results.overall_passed is False

    @pytest.mark.asyncio
    async def test_swarm_empty_agents_handling(self):
        """Swarm with no agents should handle gracefully."""
        swarm = MoonshotSwarm()
        # No agents added

        results = await swarm.run(
            baseline_queries=[],
            mrf_queries=[],
        )

        results.calculate_statistics()

        assert results.pass_rate == 0.0
        assert results.overall_passed is False
        assert len(results.agents) == 0

    @pytest.mark.asyncio
    async def test_swarm_single_agent(self, baseline_queries, mrf_queries):
        """Swarm with single agent should work correctly."""
        swarm = MoonshotSwarm()
        swarm.add_model_agents(["claude"])  # Only one agent

        results = await swarm.run(
            baseline_queries=baseline_queries,
            mrf_queries=mrf_queries,
        )

        results.calculate_statistics()

        assert len(results.agents) == 1
        # Pass rate is 0% or 100% with single agent
        assert results.pass_rate in [0.0, 1.0]

    @pytest.mark.asyncio
    async def test_swarm_recommendation_below_75_percent(self, baseline_queries, low_improvement_mrf):
        """Swarm below 75% pass rate should recommend INVESTIGATE."""
        swarm = MoonshotSwarm()
        # Add many agents to increase chance of failures
        swarm.add_model_agents(["claude", "gemini", "gpt"])
        swarm.add_complexity_agents(["simple", "moderate", "complex"])

        results = await swarm.run(
            baseline_queries=baseline_queries,
            mrf_queries=low_improvement_mrf,  # Low improvement data
        )

        results.calculate_statistics()

        # With low improvement data, most should fail
        if results.pass_rate < 0.75:
            recommendation = results.get_recommendation()
            assert "INVESTIGATE" in recommendation

    @pytest.mark.asyncio
    async def test_swarm_all_validators_error_handling(self):
        """Swarm should handle validator errors gracefully."""
        # Create queries that will cause issues (missing required fields)
        bad_queries = [{"invalid": "data"} for _ in range(50)]

        swarm = MoonshotSwarm()
        swarm.add_model_agents(["claude"])

        # Should not raise, even with bad data
        results = await swarm.run(
            baseline_queries=bad_queries,
            mrf_queries=bad_queries,
        )

        results.calculate_statistics()

        # Should complete without exception
        assert len(results.agents) == 1

    @pytest.mark.asyncio
    async def test_swarm_concurrent_limit_respected(self, baseline_queries, mrf_queries):
        """Swarm should respect max_concurrent limit."""
        swarm = MoonshotSwarm()
        # Add many agents
        swarm.add_model_agents(["claude", "gemini", "gpt"])
        swarm.add_system_agents(["recursive", "skills", "rag", "memory"])

        results = await swarm.run(
            baseline_queries=baseline_queries,
            mrf_queries=mrf_queries,
            max_concurrent=2,  # Low limit
        )

        # All agents should complete regardless of concurrency limit
        assert len(results.agents) == 7
        assert all(a.results is not None for a in results.agents)


# =============================================================================
# Swarm Orchestration Tests
# =============================================================================

class TestMoonshotSwarm:
    """Tests for MoonshotSwarm orchestration."""

    def test_add_model_agents(self):
        """Adding model agents should create correct agents."""
        swarm = MoonshotSwarm()
        swarm.add_model_agents(["claude", "gemini", "gpt"])

        assert len(swarm.agents) == 3
        assert all(a.dimension == SwarmDimension.MODEL for a in swarm.agents)
        agent_ids = [a.agent_id for a in swarm.agents]
        assert "model_claude" in agent_ids
        assert "model_gemini" in agent_ids
        assert "model_gpt" in agent_ids

    def test_add_system_agents(self):
        """Adding system agents should create correct agents."""
        swarm = MoonshotSwarm()
        swarm.add_system_agents(["recursive", "skills", "rag", "memory"])

        assert len(swarm.agents) == 4
        assert all(a.dimension == SwarmDimension.SYSTEM for a in swarm.agents)

    def test_add_complexity_agents_with_graduated_thresholds(self):
        """Complexity agents should have graduated thresholds."""
        swarm = MoonshotSwarm()
        swarm.add_complexity_agents(["simple", "moderate", "complex"])

        assert len(swarm.agents) == 3

        # Check that each agent has the correct quality threshold
        for agent in swarm.agents:
            quality_validator = agent.pipeline.validators[0]
            if "simple" in agent.agent_id:
                assert quality_validator.min_improvement_pct == 15.0
            elif "moderate" in agent.agent_id:
                assert quality_validator.min_improvement_pct == 18.0
            elif "complex" in agent.agent_id:
                assert quality_validator.min_improvement_pct == 20.0

    def test_method_chaining(self):
        """Swarm methods should support chaining."""
        swarm = (
            MoonshotSwarm()
            .add_model_agents(["claude", "gemini"])
            .add_system_agents(["rag", "memory"])
            .add_complexity_agents(["simple", "complex"])
        )

        assert len(swarm.agents) == 6

    @pytest.mark.asyncio
    async def test_swarm_parallel_execution(self, baseline_queries, mrf_queries):
        """Swarm should run agents in parallel with semaphore."""
        swarm = MoonshotSwarm()
        swarm.add_model_agents(["claude", "gemini", "gpt"])

        results = await swarm.run(
            baseline_queries=baseline_queries,
            mrf_queries=mrf_queries,
            max_concurrent=2,  # Limit concurrency
        )

        assert len(results.agents) == 3
        assert all(a.results is not None for a in results.agents)

    @pytest.mark.asyncio
    async def test_swarm_results_aggregation(self, baseline_queries, mrf_queries):
        """SwarmResults should correctly aggregate pass rate."""
        swarm = MoonshotSwarm()
        swarm.add_model_agents(["claude", "gemini", "gpt"])

        results = await swarm.run(
            baseline_queries=baseline_queries,
            mrf_queries=mrf_queries,
        )

        # Calculate statistics
        results.calculate_statistics()

        assert results.pass_rate >= 0.0
        assert results.pass_rate <= 1.0
        assert len(results.by_dimension) > 0
        assert SwarmDimension.MODEL in results.by_dimension

    @pytest.mark.asyncio
    async def test_swarm_recommendation_deploy(self, baseline_queries, mrf_queries):
        """High pass rate should recommend DEPLOY."""
        swarm = MoonshotSwarm()
        swarm.add_model_agents(["claude", "gemini", "gpt"])

        results = await swarm.run(
            baseline_queries=baseline_queries,
            mrf_queries=mrf_queries,
        )

        recommendation = results.get_recommendation()

        # With good synthetic data, should recommend DEPLOY
        assert "DEPLOY" in recommendation or "GRADUAL" in recommendation

    @pytest.mark.asyncio
    async def test_full_swarm_all_dimensions(self, baseline_queries, mrf_queries):
        """Full swarm with all dimensions should work."""
        swarm = (
            MoonshotSwarm()
            .add_model_agents(["claude", "gemini", "gpt"])
            .add_system_agents(["recursive", "skills", "rag", "memory"])
            .add_complexity_agents(["simple", "moderate", "complex"])
            .add_timeline_agents(weeks=4)
        )

        results = await swarm.run(
            baseline_queries=baseline_queries,
            mrf_queries=mrf_queries,
            max_concurrent=5,
        )

        # Should have 3 + 4 + 3 + 4 = 14 agents
        assert len(results.agents) == 14
        assert results.pass_rate > 0.5  # At least half should pass

        # Check all dimensions are represented
        results.calculate_statistics()
        assert SwarmDimension.MODEL in results.by_dimension
        assert SwarmDimension.SYSTEM in results.by_dimension
        assert SwarmDimension.COMPLEXITY in results.by_dimension
        assert SwarmDimension.TIMELINE in results.by_dimension


# =============================================================================
# Run tests if executed directly
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
