"""
Phase 3+: Moonshot Swarm - Parallel Validation Orchestrator

Elegant swarm orchestration for running multiple validation pipelines in parallel
across different dimensions, timelines, and configurations.

"Moonshot" = ambitious parallel validation
"Swarm" = multiple independent agents working concurrently

Created: November 2025
Status: Production ready
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

from .pipeline import (
    ValidationPipeline,
    ValidationPhase,
    Validator,
    QualityValidator,
    LatencyValidator,
    UserSatisfactionValidator,
    StatisticalValidator,
    PipelineResults
)


class SwarmDimension(Enum):
    """Dimensions for parallel validation."""
    QUALITY = "quality"  # Different quality metrics
    MODEL = "model"  # Different model providers (Claude, Gemini, GPT)
    SYSTEM = "system"  # Different HoloLoom systems
    COMPLEXITY = "complexity"  # Different query complexities
    TIMELINE = "timeline"  # Different time windows


@dataclass
class SwarmAgent:
    """
    Independent validation agent in the swarm.

    Each agent runs a validation pipeline configured for a specific
    dimension/configuration.
    """

    agent_id: str
    dimension: SwarmDimension
    config: Dict[str, Any]
    pipeline: ValidationPipeline

    # Results
    results: Optional[PipelineResults] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0

    async def run(
        self,
        baseline_queries: List[Dict[str, Any]],
        mrf_queries: List[Dict[str, Any]]
    ) -> PipelineResults:
        """Run validation pipeline for this agent."""
        self.start_time = datetime.now()

        # Filter queries based on agent config
        filtered_baseline = self._filter_queries(baseline_queries)
        filtered_mrf = self._filter_queries(mrf_queries)

        # Run pipeline
        self.results = await self.pipeline.run(
            baseline_queries=filtered_baseline,
            mrf_queries=filtered_mrf,
            phase=ValidationPhase.ANALYSIS
        )

        self.end_time = datetime.now()
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()

        return self.results

    def _filter_queries(self, queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter queries based on agent config."""
        # Apply dimension-specific filtering
        if self.dimension == SwarmDimension.MODEL:
            # Filter by model provider
            model = self.config.get('model_provider')
            if model:
                return [q for q in queries if q.get('model_provider') == model]

        elif self.dimension == SwarmDimension.SYSTEM:
            # Filter by HoloLoom system
            system = self.config.get('system')
            if system:
                return [q for q in queries if q.get('system') == system]

        elif self.dimension == SwarmDimension.COMPLEXITY:
            # Filter by query complexity
            complexity = self.config.get('complexity')
            if complexity:
                return [q for q in queries if q.get('complexity') == complexity]

        elif self.dimension == SwarmDimension.TIMELINE:
            # Filter by time window
            start_time = self.config.get('start_time')
            end_time = self.config.get('end_time')
            if start_time and end_time:
                return [
                    q for q in queries
                    if start_time <= q.get('timestamp', datetime.now()) <= end_time
                ]

        # No filtering
        return queries


@dataclass
class SwarmResults:
    """Aggregated results from all swarm agents."""

    agents: List[SwarmAgent]
    total_duration_seconds: float = 0.0

    # Aggregated metrics
    overall_passed: bool = False
    pass_rate: float = 0.0

    # Dimension-specific results
    by_dimension: Dict[SwarmDimension, List[SwarmAgent]] = field(default_factory=dict)

    # Best/worst performers
    best_agent: Optional[SwarmAgent] = None
    worst_agent: Optional[SwarmAgent] = None

    def calculate_statistics(self):
        """Calculate aggregated statistics."""
        passed = sum(1 for a in self.agents if a.results and a.results.passed)
        self.pass_rate = passed / len(self.agents) if self.agents else 0.0
        self.overall_passed = self.pass_rate >= 0.75  # 75% pass rate threshold

        # Group by dimension
        for agent in self.agents:
            if agent.dimension not in self.by_dimension:
                self.by_dimension[agent.dimension] = []
            self.by_dimension[agent.dimension].append(agent)

        # Find best/worst (by quality improvement)
        agents_with_quality = [
            a for a in self.agents
            if a.results and a.results.validator_results.get('quality')
        ]

        if agents_with_quality:
            self.best_agent = max(
                agents_with_quality,
                key=lambda a: a.results.validator_results['quality'].get('improvement_pct', 0)
            )

            self.worst_agent = min(
                agents_with_quality,
                key=lambda a: a.results.validator_results['quality'].get('improvement_pct', 0)
            )

    def get_recommendation(self) -> str:
        """Get overall deployment recommendation."""
        if not self.overall_passed:
            return "INVESTIGATE: <75% of validations passed"

        if self.pass_rate >= 0.90:
            return "DEPLOY: Strong evidence across all dimensions"
        elif self.pass_rate >= 0.75:
            return "GRADUAL_ROLLOUT: Good evidence, monitor edge cases"
        else:
            return "INVESTIGATE: Mixed results, review failures"


class MoonshotSwarm:
    """
    Parallel validation orchestrator.

    Runs multiple validation pipelines concurrently across different dimensions
    for comprehensive, high-confidence validation.

    Usage:
        swarm = MoonshotSwarm()

        # Add agents for different models
        swarm.add_model_agents(["claude", "gemini", "gpt"])

        # Add agents for different systems
        swarm.add_system_agents(["recursive", "skills", "rag", "memory"])

        # Run swarm
        results = await swarm.run(baseline_queries, mrf_queries)

        print(f"Pass rate: {results.pass_rate:.1%}")
        print(f"Recommendation: {results.get_recommendation()}")
    """

    def __init__(self):
        self.agents: List[SwarmAgent] = []

    def add_agent(self, agent: SwarmAgent):
        """Add an agent to the swarm."""
        self.agents.append(agent)
        return self

    def add_model_agents(self, model_providers: List[str]):
        """
        Add agents for different model providers.

        Args:
            model_providers: List of model names (e.g., ["claude", "gemini", "gpt"])
        """
        for model in model_providers:
            pipeline = ValidationPipeline()
            pipeline.add_validator(QualityValidator(min_improvement_pct=20.0))
            pipeline.add_validator(StatisticalValidator(significance_level=0.05))

            agent = SwarmAgent(
                agent_id=f"model_{model}",
                dimension=SwarmDimension.MODEL,
                config={"model_provider": model},
                pipeline=pipeline
            )
            self.add_agent(agent)

        return self

    def add_system_agents(self, systems: List[str]):
        """
        Add agents for different HoloLoom systems.

        Args:
            systems: List of system names (e.g., ["recursive", "skills", "rag", "memory"])
        """
        for system in systems:
            pipeline = ValidationPipeline()
            pipeline.add_validator(QualityValidator(min_improvement_pct=20.0))
            pipeline.add_validator(LatencyValidator(max_regression_pct=20.0))
            pipeline.add_validator(StatisticalValidator(significance_level=0.05))

            agent = SwarmAgent(
                agent_id=f"system_{system}",
                dimension=SwarmDimension.SYSTEM,
                config={"system": system},
                pipeline=pipeline
            )
            self.add_agent(agent)

        return self

    def add_complexity_agents(self, complexities: List[str]):
        """
        Add agents for different query complexities.

        Args:
            complexities: List of complexity levels (e.g., ["simple", "moderate", "complex"])

        Note: Thresholds are graduated by complexity:
        - simple: 15% quality improvement, 15% latency regression allowed
        - moderate: 18% quality improvement, 20% latency regression allowed
        - complex: 20% quality improvement, 30% latency regression allowed
        """
        # Graduated thresholds by complexity level
        # Note: simple queries can tolerate some latency regression (15%) while
        # complex queries may have higher regression due to more processing
        thresholds = {
            "simple": {"quality": 15.0, "latency": 15.0},
            "moderate": {"quality": 18.0, "latency": 20.0},
            "complex": {"quality": 20.0, "latency": 30.0},
        }

        for complexity in complexities:
            pipeline = ValidationPipeline()
            # Use graduated thresholds, with sensible defaults for unknown levels
            quality_threshold = thresholds.get(complexity, {"quality": 20.0})["quality"]
            latency_threshold = thresholds.get(complexity, {"latency": 25.0})["latency"]
            pipeline.add_validator(QualityValidator(min_improvement_pct=quality_threshold))
            pipeline.add_validator(LatencyValidator(max_regression_pct=latency_threshold))

            agent = SwarmAgent(
                agent_id=f"complexity_{complexity}",
                dimension=SwarmDimension.COMPLEXITY,
                config={"complexity": complexity},
                pipeline=pipeline
            )
            self.add_agent(agent)

        return self

    def add_timeline_agents(self, weeks: int = 4):
        """
        Add agents for different time windows (Week 1, Week 2, etc.).

        Args:
            weeks: Number of weeks to create agents for
        """
        now = datetime.now()

        for week in range(1, weeks + 1):
            start_time = now - timedelta(weeks=weeks-week+1)
            end_time = now - timedelta(weeks=weeks-week)

            pipeline = ValidationPipeline()
            pipeline.add_validator(QualityValidator(min_improvement_pct=20.0))

            agent = SwarmAgent(
                agent_id=f"week_{week}",
                dimension=SwarmDimension.TIMELINE,
                config={"start_time": start_time, "end_time": end_time},
                pipeline=pipeline
            )
            self.add_agent(agent)

        return self

    async def run(
        self,
        baseline_queries: List[Dict[str, Any]],
        mrf_queries: List[Dict[str, Any]],
        max_concurrent: int = 10
    ) -> SwarmResults:
        """
        Run all agents in swarm (parallel execution).

        Args:
            baseline_queries: Baseline query data
            mrf_queries: UnifiedMRF query data
            max_concurrent: Maximum concurrent agents (default: 10)

        Returns:
            SwarmResults with aggregated results
        """
        start_time = datetime.now()

        print(f"\n[OK] Starting moonshot swarm with {len(self.agents)} agents")
        print(f"    Max concurrent: {max_concurrent}")

        # Run agents in parallel (with concurrency limit)
        semaphore = asyncio.Semaphore(max_concurrent)

        async def run_agent_with_semaphore(agent: SwarmAgent):
            async with semaphore:
                print(f"  [{agent.agent_id}] Starting validation...")
                await agent.run(baseline_queries, mrf_queries)
                status = "PASS" if agent.results and agent.results.passed else "FAIL"
                print(f"  [{agent.agent_id}] {status} ({agent.duration_seconds:.1f}s)")

        await asyncio.gather(*[run_agent_with_semaphore(a) for a in self.agents])

        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        # Aggregate results
        results = SwarmResults(
            agents=self.agents,
            total_duration_seconds=total_duration
        )
        results.calculate_statistics()

        print(f"\n[OK] Swarm complete ({total_duration:.1f}s)")

        return results

    def print_summary(self, results: SwarmResults):
        """Print summary of swarm results."""
        print("\n" + "="*60)
        print("MOONSHOT SWARM RESULTS")
        print("="*60)

        print(f"\nOverall:")
        print(f"  Total agents: {len(results.agents)}")
        print(f"  Pass rate: {results.pass_rate:.1%}")
        print(f"  Duration: {results.total_duration_seconds:.1f}s")

        print(f"\nBy Dimension:")
        for dimension, agents in results.by_dimension.items():
            passed = sum(1 for a in agents if a.results and a.results.passed)
            print(f"  {dimension.value}: {passed}/{len(agents)} passed")

        if results.best_agent and results.worst_agent:
            best_improvement = results.best_agent.results.validator_results['quality'].get('improvement_pct', 0)
            worst_improvement = results.worst_agent.results.validator_results['quality'].get('improvement_pct', 0)

            print(f"\nPerformance:")
            print(f"  Best: {results.best_agent.agent_id} (+{best_improvement:.1f}%)")
            print(f"  Worst: {results.worst_agent.agent_id} (+{worst_improvement:.1f}%)")

        print(f"\nRecommendation:")
        print(f"  {results.get_recommendation()}")

        print("\n" + "="*60)


# Example usage
if __name__ == "__main__":
    async def main():
        # Simulate data with model/system metadata
        baseline_queries = [
            {
                "quality_score": 0.72 + (i % 10) * 0.01,
                "latency_ms": 140 + (i % 5) * 2,
                "model_provider": ["claude", "gemini", "gpt"][i % 3],
                "system": ["recursive", "skills", "rag", "memory"][i % 4],
                "complexity": ["simple", "moderate", "complex"][i % 3],
                "timestamp": datetime.now() - timedelta(days=i % 28)
            }
            for i in range(100)
        ]

        mrf_queries = [
            {
                "quality_score": 0.91 + (i % 10) * 0.01,
                "latency_ms": 155 + (i % 5) * 2,
                "user_rating": 4 + (i % 2),
                "thumbs_up": (i % 3) > 0,
                "model_provider": ["claude", "gemini", "gpt"][i % 3],
                "system": ["recursive", "skills", "rag", "memory"][i % 4],
                "complexity": ["simple", "moderate", "complex"][i % 3],
                "timestamp": datetime.now() - timedelta(days=i % 28)
            }
            for i in range(100)
        ]

        # Create swarm
        swarm = MoonshotSwarm()

        # Add agents for different dimensions (elegant chaining)
        swarm \
            .add_model_agents(["claude", "gemini", "gpt"]) \
            .add_system_agents(["recursive", "skills", "rag", "memory"]) \
            .add_complexity_agents(["simple", "moderate", "complex"]) \
            .add_timeline_agents(weeks=4)

        # Run swarm (parallel validation)
        results = await swarm.run(
            baseline_queries=baseline_queries,
            mrf_queries=mrf_queries,
            max_concurrent=5  # Run 5 agents at a time
        )

        # Print summary
        swarm.print_summary(results)

    asyncio.run(main())
