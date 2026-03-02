"""
Phase 3: Production Validation - A/B Testing Infrastructure

This module provides infrastructure for comparing traditional prompts vs UnifiedMRF-enhanced
prompts in production settings.

Created: November 2025
Status: Production validation framework
"""

import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hololoom.prompting.unified_mrf import UnifiedMRF, ModelProvider, MetapromptConfig


class PromptVariant(Enum):
    """A/B test variant."""
    TRADITIONAL = "traditional"  # Baseline (pre-MRF)
    UNIFIED_MRF = "unified_mrf"  # Enhanced (post-MRF)


@dataclass
class ABTestConfig:
    """Configuration for A/B testing."""

    # Traffic split (0.0-1.0, what fraction gets UnifiedMRF)
    mrf_traffic_ratio: float = 0.5  # 50/50 split by default

    # Minimum samples per variant for statistical significance
    min_samples_per_variant: int = 30

    # Confidence level for statistical tests (0.0-1.0)
    confidence_level: float = 0.95  # 95% confidence

    # Maximum duration for test (seconds)
    max_duration_seconds: Optional[float] = None

    # Output directory for results
    output_dir: Path = Path("./ab_test_results")

    # Enable human evaluation collection
    enable_human_eval: bool = True

    # Model provider for UnifiedMRF variant
    model_provider: Optional[str] = None


@dataclass
class PromptExecution:
    """Result of a single prompt execution."""

    variant: PromptVariant
    query: str
    response: str
    latency_ms: float
    timestamp: datetime = field(default_factory=datetime.now)

    # Quality metrics (can be auto-computed or human-rated)
    quality_score: Optional[float] = None  # 0.0-1.0
    correctness: Optional[bool] = None
    completeness: Optional[float] = None  # 0.0-1.0
    clarity: Optional[float] = None  # 0.0-1.0

    # Human evaluation (if enabled)
    human_rating: Optional[int] = None  # 1-5 stars
    human_feedback: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "variant": self.variant.value,
            "query": self.query,
            "response": self.response,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
            "quality_score": self.quality_score,
            "correctness": self.correctness,
            "completeness": self.completeness,
            "clarity": self.clarity,
            "human_rating": self.human_rating,
            "human_feedback": self.human_feedback,
            "metadata": self.metadata
        }


@dataclass
class ABTestResults:
    """Aggregated A/B test results with statistical analysis."""

    traditional_executions: List[PromptExecution] = field(default_factory=list)
    mrf_executions: List[PromptExecution] = field(default_factory=list)

    # Summary statistics
    traditional_avg_quality: float = 0.0
    mrf_avg_quality: float = 0.0
    quality_improvement_pct: float = 0.0

    traditional_avg_latency: float = 0.0
    mrf_avg_latency: float = 0.0
    latency_change_pct: float = 0.0

    # Statistical significance
    is_statistically_significant: bool = False
    p_value: Optional[float] = None
    confidence_level: float = 0.95

    # Human evaluation (if enabled)
    traditional_avg_human_rating: Optional[float] = None
    mrf_avg_human_rating: Optional[float] = None
    human_preference_pct: Optional[float] = None  # % preferring MRF

    # Metadata
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_duration_seconds: float = 0.0

    def calculate_statistics(self):
        """Calculate summary statistics from executions."""
        # Quality scores
        trad_qualities = [e.quality_score for e in self.traditional_executions
                         if e.quality_score is not None]
        mrf_qualities = [e.quality_score for e in self.mrf_executions
                        if e.quality_score is not None]

        if trad_qualities:
            self.traditional_avg_quality = sum(trad_qualities) / len(trad_qualities)
        if mrf_qualities:
            self.mrf_avg_quality = sum(mrf_qualities) / len(mrf_qualities)

        if self.traditional_avg_quality > 0:
            self.quality_improvement_pct = (
                (self.mrf_avg_quality - self.traditional_avg_quality)
                / self.traditional_avg_quality
            ) * 100

        # Latency
        trad_latencies = [e.latency_ms for e in self.traditional_executions]
        mrf_latencies = [e.latency_ms for e in self.mrf_executions]

        if trad_latencies:
            self.traditional_avg_latency = sum(trad_latencies) / len(trad_latencies)
        if mrf_latencies:
            self.mrf_avg_latency = sum(mrf_latencies) / len(mrf_latencies)

        if self.traditional_avg_latency > 0:
            self.latency_change_pct = (
                (self.mrf_avg_latency - self.traditional_avg_latency)
                / self.traditional_avg_latency
            ) * 100

        # Human ratings (if available)
        trad_ratings = [e.human_rating for e in self.traditional_executions
                       if e.human_rating is not None]
        mrf_ratings = [e.human_rating for e in self.mrf_executions
                      if e.human_rating is not None]

        if trad_ratings:
            self.traditional_avg_human_rating = sum(trad_ratings) / len(trad_ratings)
        if mrf_ratings:
            self.mrf_avg_human_rating = sum(mrf_ratings) / len(mrf_ratings)

        if trad_ratings and mrf_ratings:
            # Calculate preference: % of queries where MRF rated higher
            preferences = []
            for trad_ex, mrf_ex in zip(self.traditional_executions, self.mrf_executions):
                if trad_ex.human_rating and mrf_ex.human_rating:
                    if mrf_ex.human_rating > trad_ex.human_rating:
                        preferences.append(1)
                    elif mrf_ex.human_rating < trad_ex.human_rating:
                        preferences.append(0)

            if preferences:
                self.human_preference_pct = (sum(preferences) / len(preferences)) * 100

        # Statistical significance (simplified t-test approximation)
        if len(trad_qualities) >= 30 and len(mrf_qualities) >= 30:
            # For production, use scipy.stats.ttest_ind
            # Here we use a simplified heuristic: difference > 2 standard errors
            import statistics

            trad_std = statistics.stdev(trad_qualities) if len(trad_qualities) > 1 else 0
            mrf_std = statistics.stdev(mrf_qualities) if len(mrf_qualities) > 1 else 0

            pooled_std = ((trad_std ** 2 + mrf_std ** 2) / 2) ** 0.5
            std_error = pooled_std * ((1/len(trad_qualities) + 1/len(mrf_qualities)) ** 0.5)

            diff = abs(self.mrf_avg_quality - self.traditional_avg_quality)

            # Simplified: difference > 2 SE ~= 95% confidence
            self.is_statistically_significant = diff > (2 * std_error)

            # Approximate p-value (for actual production, use scipy)
            if std_error > 0:
                z_score = diff / std_error
                # Rough approximation: p-value from z-score
                if z_score > 2.58:  # 99% confidence
                    self.p_value = 0.01
                elif z_score > 1.96:  # 95% confidence
                    self.p_value = 0.05
                elif z_score > 1.64:  # 90% confidence
                    self.p_value = 0.10
                else:
                    self.p_value = 0.20  # Not significant

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "summary": {
                "total_queries": len(self.traditional_executions) + len(self.mrf_executions),
                "traditional_count": len(self.traditional_executions),
                "mrf_count": len(self.mrf_executions),
                "duration_seconds": self.total_duration_seconds
            },
            "quality": {
                "traditional_avg": self.traditional_avg_quality,
                "mrf_avg": self.mrf_avg_quality,
                "improvement_pct": self.quality_improvement_pct,
                "is_significant": self.is_statistically_significant,
                "p_value": self.p_value,
                "confidence_level": self.confidence_level
            },
            "latency": {
                "traditional_avg_ms": self.traditional_avg_latency,
                "mrf_avg_ms": self.mrf_avg_latency,
                "change_pct": self.latency_change_pct
            },
            "human_evaluation": {
                "traditional_avg_rating": self.traditional_avg_human_rating,
                "mrf_avg_rating": self.mrf_avg_human_rating,
                "mrf_preference_pct": self.human_preference_pct
            },
            "timestamps": {
                "start": self.start_time.isoformat(),
                "end": self.end_time.isoformat() if self.end_time else None
            }
        }


class ABTestRunner:
    """
    A/B test runner for comparing traditional vs UnifiedMRF prompts.

    Usage:
        runner = ABTestRunner(config=ABTestConfig(mrf_traffic_ratio=0.5))

        # Run test
        results = await runner.run_test(
            queries=production_queries,
            traditional_prompt_fn=legacy_prompt_builder,
            mrf_prompt_fn=unified_mrf_prompt_builder
        )

        # Analyze results
        print(f"Quality improvement: {results.quality_improvement_pct:.1f}%")
        print(f"Statistically significant: {results.is_statistically_significant}")
    """

    def __init__(self, config: ABTestConfig):
        self.config = config
        self.mrf = UnifiedMRF()

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Results
        self.results = ABTestResults()

    def assign_variant(self) -> PromptVariant:
        """
        Assign a variant using traffic ratio.

        Returns:
            PromptVariant.UNIFIED_MRF with probability = mrf_traffic_ratio
            PromptVariant.TRADITIONAL otherwise
        """
        if random.random() < self.config.mrf_traffic_ratio:
            return PromptVariant.UNIFIED_MRF
        return PromptVariant.TRADITIONAL

    async def execute_traditional(
        self,
        query: str,
        traditional_prompt: str
    ) -> PromptExecution:
        """
        Execute traditional (pre-MRF) prompt.

        Args:
            query: User query
            traditional_prompt: Pre-built traditional prompt string

        Returns:
            PromptExecution with results
        """
        start = time.time()

        # In production, this would call your actual LLM
        # For now, return placeholder
        response = f"[TRADITIONAL] Response to: {query}"

        latency_ms = (time.time() - start) * 1000

        return PromptExecution(
            variant=PromptVariant.TRADITIONAL,
            query=query,
            response=response,
            latency_ms=latency_ms
        )

    async def execute_mrf(
        self,
        query: str,
        metaprompt: MetapromptConfig
    ) -> PromptExecution:
        """
        Execute UnifiedMRF-enhanced prompt.

        Args:
            query: User query
            metaprompt: MetapromptConfig for this query

        Returns:
            PromptExecution with results
        """
        start = time.time()

        # Map model provider
        provider = None
        if self.config.model_provider:
            provider_map = {
                "claude": ModelProvider.CLAUDE,
                "gemini": ModelProvider.GEMINI,
                "gpt": ModelProvider.GPT,
                "ollama": ModelProvider.OLLAMA
            }
            provider = provider_map.get(self.config.model_provider.lower())

        # Generate enhanced prompt
        enhanced_prompt = self.mrf.enhance_prompt(
            metaprompt=metaprompt,
            query=query,
            model=provider
        )

        # In production, this would call your actual LLM
        # For now, return placeholder
        response = f"[UNIFIED_MRF] Enhanced response to: {query}"

        latency_ms = (time.time() - start) * 1000

        return PromptExecution(
            variant=PromptVariant.UNIFIED_MRF,
            query=query,
            response=response,
            latency_ms=latency_ms
        )

    async def run_test(
        self,
        queries: List[str],
        traditional_prompts: List[str],
        metaprompts: List[MetapromptConfig]
    ) -> ABTestResults:
        """
        Run A/B test on a list of queries.

        Args:
            queries: List of user queries to test
            traditional_prompts: Pre-built traditional prompts (same length as queries)
            metaprompts: MetapromptConfigs for UnifiedMRF (same length as queries)

        Returns:
            ABTestResults with statistical analysis
        """
        assert len(queries) == len(traditional_prompts) == len(metaprompts), \
            "queries, traditional_prompts, and metaprompts must have same length"

        self.results.start_time = datetime.now()

        print(f"\n[OK] Starting A/B test with {len(queries)} queries")
        print(f"    MRF traffic ratio: {self.config.mrf_traffic_ratio * 100:.0f}%")
        print(f"    Min samples per variant: {self.config.min_samples_per_variant}")
        print(f"    Confidence level: {self.config.confidence_level * 100:.0f}%\n")

        for i, (query, trad_prompt, metaprompt) in enumerate(zip(queries, traditional_prompts, metaprompts)):
            # Assign variant
            variant = self.assign_variant()

            # Execute
            if variant == PromptVariant.TRADITIONAL:
                execution = await self.execute_traditional(query, trad_prompt)
                self.results.traditional_executions.append(execution)
                variant_str = "TRAD"
            else:
                execution = await self.execute_mrf(query, metaprompt)
                self.results.mrf_executions.append(execution)
                variant_str = "MRF "

            print(f"  [{i+1}/{len(queries)}] {variant_str} | {query[:60]}... | {execution.latency_ms:.1f}ms")

            # Check if we have minimum samples
            if (len(self.results.traditional_executions) >= self.config.min_samples_per_variant and
                len(self.results.mrf_executions) >= self.config.min_samples_per_variant):

                # Check duration limit
                if self.config.max_duration_seconds:
                    elapsed = (datetime.now() - self.results.start_time).total_seconds()
                    if elapsed >= self.config.max_duration_seconds:
                        print(f"\n[WARN] Reached max duration ({self.config.max_duration_seconds}s), stopping early")
                        break

        self.results.end_time = datetime.now()
        self.results.total_duration_seconds = (
            self.results.end_time - self.results.start_time
        ).total_seconds()

        # Calculate statistics
        self.results.calculate_statistics()

        # Save results
        self._save_results()

        return self.results

    def _save_results(self):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.config.output_dir / f"ab_test_results_{timestamp}.json"

        # Save detailed results
        detailed_results = {
            "config": {
                "mrf_traffic_ratio": self.config.mrf_traffic_ratio,
                "min_samples_per_variant": self.config.min_samples_per_variant,
                "confidence_level": self.config.confidence_level,
                "model_provider": self.config.model_provider
            },
            "summary": self.results.to_dict(),
            "executions": {
                "traditional": [e.to_dict() for e in self.results.traditional_executions],
                "unified_mrf": [e.to_dict() for e in self.results.mrf_executions]
            }
        }

        with open(output_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)

        print(f"\n[OK] Results saved to: {output_file}")


# Example usage
if __name__ == "__main__":
    async def main():
        # Configuration
        config = ABTestConfig(
            mrf_traffic_ratio=0.5,  # 50/50 split
            min_samples_per_variant=10,
            confidence_level=0.95,
            model_provider="claude",
            output_dir=Path("./ab_test_results")
        )

        # Create runner
        runner = ABTestRunner(config)

        # Example queries
        queries = [
            "What is Thompson Sampling?",
            "Explain Bayesian optimization",
            "How does exploration-exploitation tradeoff work?",
            "What are the benefits of multi-armed bandits?",
            "Compare UCB and Thompson Sampling",
        ] * 4  # 20 queries total (10 per variant expected)

        # Traditional prompts (simplified)
        traditional_prompts = [
            f"Answer this question: {q}" for q in queries
        ]

        # UnifiedMRF metaprompts (simplified)
        metaprompts = [
            MetapromptConfig(
                role="Expert in machine learning and reinforcement learning",
                objective={
                    "primary": f"Answer: {q}",
                    "secondary": ["Be accurate", "Be concise"]
                },
                process=["Understand question", "Formulate answer", "Validate"],
                format_spec="1-2 paragraphs",
                constraints=["MUST be factually correct", "SHOULD use examples"],
                uncertainty="Note if uncertain",
                validation=["Check accuracy", "Check completeness"]
            )
            for q in queries
        ]

        # Run test
        results = await runner.run_test(queries, traditional_prompts, metaprompts)

        # Print summary
        print("\n" + "="*60)
        print("A/B TEST RESULTS")
        print("="*60)
        print(f"\nTotal queries: {len(results.traditional_executions) + len(results.mrf_executions)}")
        print(f"  Traditional: {len(results.traditional_executions)}")
        print(f"  UnifiedMRF: {len(results.mrf_executions)}")

        print(f"\nQuality:")
        print(f"  Traditional avg: {results.traditional_avg_quality:.3f}")
        print(f"  UnifiedMRF avg: {results.mrf_avg_quality:.3f}")
        print(f"  Improvement: {results.quality_improvement_pct:+.1f}%")

        print(f"\nStatistical Significance:")
        print(f"  Significant: {results.is_statistically_significant}")
        print(f"  P-value: {results.p_value}")
        print(f"  Confidence: {results.confidence_level * 100:.0f}%")

        print(f"\nLatency:")
        print(f"  Traditional avg: {results.traditional_avg_latency:.1f}ms")
        print(f"  UnifiedMRF avg: {results.mrf_avg_latency:.1f}ms")
        print(f"  Change: {results.latency_change_pct:+.1f}%")

        print("\n" + "="*60)

    asyncio.run(main())
