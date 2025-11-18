"""
Phase 6.2: CONSENSUS Refinement

Parallel strategy execution with ensemble voting for robust refinement.

Key Features:
- Execute multiple strategies in parallel (asyncio concurrency)
- Ensemble voting methods (quality-weighted, diversity, unanimous)
- Disagreement detection and analysis
- Confidence aggregation across strategies
- Best-of-N selection with justification

Architecture:
    ConsensusRefiner
        ├── Parallel execution (async)
        ├── Ensemble voting
        ├── Disagreement analysis
        └── Consensus result

Usage:
    refiner = ConsensusRefiner(
        packer=packer,
        strategies=[
            RefinementStrategy.DEPTH_FIRST,
            RefinementStrategy.BREADTH_FIRST,
            RefinementStrategy.FOCUSED
        ],
        voting_method=VotingMethod.QUALITY_WEIGHTED
    )

    result = await refiner.refine(query, ctx, memories)
    print(f"Consensus: {result.consensus_confidence:.2f}")
    print(f"Agreement: {result.agreement_level:.1%}")
    if result.disagreement_points:
        print(f"Disagreements: {len(result.disagreement_points)}")

Created: November 2025
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
import time

try:
    from .refinement_engine import RefinementEngine, RefinementResult, RefinementStrategy
    REFINEMENT_AVAILABLE = True
except ImportError:
    REFINEMENT_AVAILABLE = False
    RefinementEngine = object
    RefinementResult = object

    class RefinementStrategy(str, Enum):
        ADAPTIVE = "adaptive"
        REFINE = "refine"
        CRITIQUE = "critique"
        VERIFY = "verify"
        ELEGANCE = "elegance"
        HOFSTADTER = "hofstadter"
        DEPTH_FIRST = "depth_first"
        BREADTH_FIRST = "breadth_first"
        FOCUSED = "focused"


class VotingMethod(str, Enum):
    """Ensemble voting methods for consensus building"""
    QUALITY_WEIGHTED = "quality_weighted"  # Weight by confidence scores
    DIVERSITY = "diversity"                # Prefer diverse perspectives
    UNANIMOUS = "unanimous"                # Require agreement
    BEST_OF_N = "best_of_n"               # Simply pick highest quality


@dataclass
class StrategyResult:
    """Result from a single strategy execution"""
    strategy: RefinementStrategy
    result: Any  # RefinementResult
    quality: float
    latency_ms: float
    error: Optional[str] = None

    def is_success(self) -> bool:
        """Whether strategy completed successfully"""
        return self.error is None


@dataclass
class DisagreementPoint:
    """Point of disagreement between strategies"""
    dimension: str  # What they disagree on (quality, content, approach)
    strategies: List[RefinementStrategy]  # Which strategies disagree
    values: List[Any]  # Their different values
    severity: float  # 0-1 scale (0=minor, 1=major)
    description: str  # Human-readable explanation


@dataclass
class ConsensusResult:
    """Result from consensus refinement with ensemble voting"""
    # Selected result
    selected_result: Any  # RefinementResult
    selected_strategy: RefinementStrategy

    # Consensus metadata
    consensus_confidence: float  # 0-1 scale
    agreement_level: float  # 0-1 scale (what % strategies agree)
    voting_method: VotingMethod

    # All strategy results
    strategy_results: List[StrategyResult]
    successful_strategies: int
    failed_strategies: int

    # Disagreement analysis
    disagreement_points: List[DisagreementPoint]
    has_major_disagreement: bool

    # Timing
    total_latency_ms: float
    parallel_speedup: float  # vs sequential execution

    # Voting details
    vote_distribution: Dict[RefinementStrategy, float]  # Vote weights per strategy

    def get_all_results(self) -> List[Any]:
        """Get all successful strategy results"""
        return [sr.result for sr in self.strategy_results if sr.is_success()]

    def get_quality_range(self) -> Tuple[float, float]:
        """Get min and max quality across strategies"""
        qualities = [sr.quality for sr in self.strategy_results if sr.is_success()]
        if not qualities:
            return (0.0, 0.0)
        return (min(qualities), max(qualities))

    def get_summary(self) -> str:
        """Human-readable summary"""
        min_q, max_q = self.get_quality_range()

        summary = [
            f"Consensus Refinement Summary",
            f"  Selected: {self.selected_strategy.value}",
            f"  Consensus Confidence: {self.consensus_confidence:.2f}",
            f"  Agreement Level: {self.agreement_level:.1%}",
            f"  Successful Strategies: {self.successful_strategies}/{len(self.strategy_results)}",
            f"  Quality Range: {min_q:.2f} - {max_q:.2f}",
            f"  Parallel Speedup: {self.parallel_speedup:.1f}x",
        ]

        if self.disagreement_points:
            summary.append(f"  Disagreements: {len(self.disagreement_points)}")
            for dp in self.disagreement_points[:3]:  # Top 3
                summary.append(f"    • {dp.description} (severity: {dp.severity:.2f})")

        return "\n".join(summary)


class ConsensusRefiner:
    """
    Refinement engine that executes multiple strategies in parallel
    and uses ensemble voting to select the best result.

    This extends the single-strategy refinement paradigm to:
    - Execute multiple strategies concurrently (async)
    - Vote on best result using ensemble methods
    - Detect disagreements between strategies
    - Aggregate confidence across strategies
    """

    def __init__(
        self,
        packer,
        strategies: Optional[List[RefinementStrategy]] = None,
        voting_method: VotingMethod = VotingMethod.QUALITY_WEIGHTED,
        quality_threshold: float = 0.85,
        max_passes: int = 3,
        require_unanimity: bool = False,
        unanimity_threshold: float = 0.8,
        enable_disagreement_detection: bool = True,
        max_parallel: int = 5,
        timeout_per_strategy: float = 30.0,
    ):
        """
        Initialize consensus refiner.

        Args:
            packer: Context packer instance
            strategies: List of strategies to run in parallel
                       (default: [DEPTH_FIRST, BREADTH_FIRST, FOCUSED])
            voting_method: How to aggregate results
            quality_threshold: Quality threshold for refinement
            max_passes: Max passes per strategy
            require_unanimity: Whether to require unanimous agreement
            unanimity_threshold: Agreement level required (0-1)
            enable_disagreement_detection: Whether to detect disagreements
            max_parallel: Max concurrent strategy executions
            timeout_per_strategy: Timeout in seconds per strategy
        """
        self.packer = packer
        self.voting_method = voting_method
        self.quality_threshold = quality_threshold
        self.max_passes = max_passes
        self.require_unanimity = require_unanimity
        self.unanimity_threshold = unanimity_threshold
        self.enable_disagreement_detection = enable_disagreement_detection
        self.max_parallel = max_parallel
        self.timeout_per_strategy = timeout_per_strategy

        # Default strategies if none provided
        if strategies is None:
            self.strategies = [
                RefinementStrategy.DEPTH_FIRST,
                RefinementStrategy.BREADTH_FIRST,
                RefinementStrategy.FOCUSED,
            ]
        else:
            self.strategies = strategies

        # Statistics
        self.total_refinements = 0
        self.total_parallel_time_ms = 0.0
        self.total_sequential_time_ms = 0.0

    async def refine(
        self,
        query: str,
        awareness_ctx,
        memory_results: Optional[List[Any]] = None,
        **kwargs
    ) -> ConsensusResult:
        """
        Refine query using parallel strategies and ensemble voting.

        Args:
            query: Query text
            awareness_ctx: Awareness context
            memory_results: Memory retrieval results
            **kwargs: Additional arguments

        Returns:
            ConsensusResult with selected result and consensus metadata
        """
        start_time = time.perf_counter()

        # Execute strategies in parallel
        strategy_results = await self._execute_strategies_parallel(
            query=query,
            awareness_ctx=awareness_ctx,
            memory_results=memory_results,
            **kwargs
        )

        parallel_time = (time.perf_counter() - start_time) * 1000

        # Filter successful results
        successful_results = [sr for sr in strategy_results if sr.is_success()]

        if not successful_results:
            raise ValueError("All strategies failed during consensus refinement")

        # Ensemble voting to select best result
        selected_result, vote_distribution = self._ensemble_vote(successful_results)

        # Calculate agreement level
        agreement_level = self._calculate_agreement(successful_results)

        # Detect disagreements if enabled
        disagreement_points = []
        if self.enable_disagreement_detection:
            disagreement_points = self._detect_disagreements(successful_results)

        # Check unanimity requirement
        has_major_disagreement = any(dp.severity >= 0.7 for dp in disagreement_points)
        if self.require_unanimity and agreement_level < self.unanimity_threshold:
            # Could raise exception or trigger re-refinement
            pass

        # Calculate consensus confidence
        consensus_confidence = self._calculate_consensus_confidence(
            selected_result=selected_result,
            agreement_level=agreement_level,
            vote_distribution=vote_distribution
        )

        # Calculate parallel speedup
        sequential_time = sum(sr.latency_ms for sr in strategy_results)
        parallel_speedup = sequential_time / parallel_time if parallel_time > 0 else 1.0

        # Update statistics
        self.total_refinements += 1
        self.total_parallel_time_ms += parallel_time
        self.total_sequential_time_ms += sequential_time

        return ConsensusResult(
            selected_result=selected_result.result,
            selected_strategy=selected_result.strategy,
            consensus_confidence=consensus_confidence,
            agreement_level=agreement_level,
            voting_method=self.voting_method,
            strategy_results=strategy_results,
            successful_strategies=len(successful_results),
            failed_strategies=len(strategy_results) - len(successful_results),
            disagreement_points=disagreement_points,
            has_major_disagreement=has_major_disagreement,
            total_latency_ms=parallel_time,
            parallel_speedup=parallel_speedup,
            vote_distribution=vote_distribution
        )

    async def _execute_strategies_parallel(
        self,
        query: str,
        awareness_ctx,
        memory_results: Optional[List[Any]],
        **kwargs
    ) -> List[StrategyResult]:
        """Execute all strategies in parallel with asyncio"""

        # Create tasks for each strategy
        tasks = []
        for strategy in self.strategies:
            task = self._execute_single_strategy(
                strategy=strategy,
                query=query,
                awareness_ctx=awareness_ctx,
                memory_results=memory_results,
                **kwargs
            )
            tasks.append(task)

        # Execute with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.max_parallel)

        async def bounded_execute(task):
            async with semaphore:
                return await task

        bounded_tasks = [bounded_execute(task) for task in tasks]

        # Gather results
        results = await asyncio.gather(*bounded_tasks, return_exceptions=True)

        # Convert exceptions to StrategyResult with errors
        strategy_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                strategy_results.append(StrategyResult(
                    strategy=self.strategies[i],
                    result=None,
                    quality=0.0,
                    latency_ms=0.0,
                    error=str(result)
                ))
            else:
                strategy_results.append(result)

        return strategy_results

    async def _execute_single_strategy(
        self,
        strategy: RefinementStrategy,
        query: str,
        awareness_ctx,
        memory_results: Optional[List[Any]],
        **kwargs
    ) -> StrategyResult:
        """Execute a single strategy with timeout"""

        start_time = time.perf_counter()

        try:
            # Create refiner for this strategy
            if not REFINEMENT_AVAILABLE:
                raise ImportError("RefinementEngine not available")

            refiner = RefinementEngine(
                packer=self.packer,
                strategy=strategy,
                quality_threshold=self.quality_threshold,
                max_passes=self.max_passes
            )

            # Execute with timeout
            result = await asyncio.wait_for(
                refiner.refine(query, awareness_ctx, memory_results, **kwargs),
                timeout=self.timeout_per_strategy
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            return StrategyResult(
                strategy=strategy,
                result=result,
                quality=result.final_quality,
                latency_ms=latency_ms,
                error=None
            )

        except asyncio.TimeoutError:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return StrategyResult(
                strategy=strategy,
                result=None,
                quality=0.0,
                latency_ms=latency_ms,
                error=f"Timeout after {self.timeout_per_strategy}s"
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return StrategyResult(
                strategy=strategy,
                result=None,
                quality=0.0,
                latency_ms=latency_ms,
                error=str(e)
            )

    def _ensemble_vote(
        self,
        strategy_results: List[StrategyResult]
    ) -> Tuple[StrategyResult, Dict[RefinementStrategy, float]]:
        """
        Use ensemble voting to select best result.

        Returns:
            (selected_result, vote_distribution)
        """

        if self.voting_method == VotingMethod.BEST_OF_N:
            return self._vote_best_of_n(strategy_results)
        elif self.voting_method == VotingMethod.QUALITY_WEIGHTED:
            return self._vote_quality_weighted(strategy_results)
        elif self.voting_method == VotingMethod.DIVERSITY:
            return self._vote_diversity(strategy_results)
        elif self.voting_method == VotingMethod.UNANIMOUS:
            return self._vote_unanimous(strategy_results)
        else:
            return self._vote_best_of_n(strategy_results)

    def _vote_best_of_n(
        self,
        strategy_results: List[StrategyResult]
    ) -> Tuple[StrategyResult, Dict[RefinementStrategy, float]]:
        """Simply pick highest quality result"""

        best_result = max(strategy_results, key=lambda sr: sr.quality)

        # Vote distribution: winner gets 1.0, others get 0.0
        vote_distribution = {
            sr.strategy: 1.0 if sr == best_result else 0.0
            for sr in strategy_results
        }

        return best_result, vote_distribution

    def _vote_quality_weighted(
        self,
        strategy_results: List[StrategyResult]
    ) -> Tuple[StrategyResult, Dict[RefinementStrategy, float]]:
        """Weight votes by quality scores"""

        # Normalize quality scores to weights
        total_quality = sum(sr.quality for sr in strategy_results)

        if total_quality == 0:
            return self._vote_best_of_n(strategy_results)

        vote_distribution = {
            sr.strategy: sr.quality / total_quality
            for sr in strategy_results
        }

        # Select result with highest weight
        best_result = max(strategy_results, key=lambda sr: vote_distribution[sr.strategy])

        return best_result, vote_distribution

    def _vote_diversity(
        self,
        strategy_results: List[StrategyResult]
    ) -> Tuple[StrategyResult, Dict[RefinementStrategy, float]]:
        """
        Prefer diverse perspectives.

        Gives bonus to strategies that provide different insights
        rather than all agreeing on the same thing.
        """

        # Simple heuristic: check if results are similar
        # If all very similar (high agreement), apply diversity penalty
        # If results differ significantly, reward diversity

        qualities = [sr.quality for sr in strategy_results]
        quality_variance = sum((q - sum(qualities)/len(qualities))**2 for q in qualities) / len(qualities)

        # High variance → reward diversity
        # Low variance → standard quality weighting
        diversity_factor = min(1.0, quality_variance * 10)  # Scale variance

        # Combine quality with diversity bonus
        total_score = 0.0
        scores = {}

        for sr in strategy_results:
            # Base score from quality
            base_score = sr.quality

            # Diversity bonus: strategies with different quality get bonus
            avg_quality = sum(qualities) / len(qualities)
            diversity_bonus = abs(sr.quality - avg_quality) * diversity_factor

            score = base_score + diversity_bonus * 0.2  # 20% weight to diversity
            scores[sr.strategy] = score
            total_score += score

        # Normalize to distribution
        vote_distribution = {
            strategy: score / total_score if total_score > 0 else 1.0 / len(scores)
            for strategy, score in scores.items()
        }

        # Select highest scoring
        best_result = max(strategy_results, key=lambda sr: scores[sr.strategy])

        return best_result, vote_distribution

    def _vote_unanimous(
        self,
        strategy_results: List[StrategyResult]
    ) -> Tuple[StrategyResult, Dict[RefinementStrategy, float]]:
        """
        Require unanimous agreement (or near-unanimous).

        Only select a result if most strategies agree on similar quality.
        """

        qualities = [sr.quality for sr in strategy_results]
        avg_quality = sum(qualities) / len(qualities)

        # Check agreement: strategies within 10% of average
        agreement_threshold = 0.1
        agreeing_results = [
            sr for sr in strategy_results
            if abs(sr.quality - avg_quality) / (avg_quality + 1e-6) <= agreement_threshold
        ]

        if len(agreeing_results) / len(strategy_results) >= self.unanimity_threshold:
            # Sufficient agreement, pick best among agreeing
            best_result = max(agreeing_results, key=lambda sr: sr.quality)
        else:
            # Insufficient agreement, fall back to quality-weighted
            return self._vote_quality_weighted(strategy_results)

        # Vote distribution: agreeing strategies get equal weight
        vote_distribution = {
            sr.strategy: 1.0 / len(agreeing_results) if sr in agreeing_results else 0.0
            for sr in strategy_results
        }

        return best_result, vote_distribution

    def _calculate_agreement(self, strategy_results: List[StrategyResult]) -> float:
        """
        Calculate agreement level across strategies (0-1 scale).

        Agreement is high when all strategies produce similar quality.
        """

        if len(strategy_results) <= 1:
            return 1.0

        qualities = [sr.quality for sr in strategy_results]
        avg_quality = sum(qualities) / len(qualities)

        # Calculate coefficient of variation (std dev / mean)
        variance = sum((q - avg_quality)**2 for q in qualities) / len(qualities)
        std_dev = variance ** 0.5

        # Convert to agreement (inverse of variation)
        # High variation → low agreement
        # Low variation → high agreement
        cv = std_dev / (avg_quality + 1e-6)
        agreement = 1.0 / (1.0 + cv)  # Normalize to 0-1

        return agreement

    def _detect_disagreements(
        self,
        strategy_results: List[StrategyResult]
    ) -> List[DisagreementPoint]:
        """
        Detect points of disagreement between strategies.

        Disagreements occur when:
        - Quality scores differ significantly
        - Different passes executed
        - Different stopping criteria
        """

        disagreements = []

        # Quality disagreement
        qualities = [sr.quality for sr in strategy_results]
        quality_range = max(qualities) - min(qualities)

        if quality_range > 0.15:  # >15% quality difference
            disagreements.append(DisagreementPoint(
                dimension="quality",
                strategies=[sr.strategy for sr in strategy_results],
                values=qualities,
                severity=min(1.0, quality_range / 0.3),  # Normalize to 0-1
                description=f"Quality ranges from {min(qualities):.2f} to {max(qualities):.2f}"
            ))

        # Passes disagreement
        passes_executed = [sr.result.passes_executed for sr in strategy_results if hasattr(sr.result, 'passes_executed')]
        if len(set(passes_executed)) > 1:  # Different pass counts
            disagreements.append(DisagreementPoint(
                dimension="passes",
                strategies=[sr.strategy for sr in strategy_results],
                values=passes_executed,
                severity=0.5,  # Medium severity
                description=f"Strategies executed different numbers of passes: {set(passes_executed)}"
            ))

        return disagreements

    def _calculate_consensus_confidence(
        self,
        selected_result: StrategyResult,
        agreement_level: float,
        vote_distribution: Dict[RefinementStrategy, float]
    ) -> float:
        """
        Calculate overall consensus confidence.

        Confidence is high when:
        - Selected result has high quality
        - High agreement among strategies
        - Selected result got strong vote share
        """

        # Component 1: Selected result quality (50% weight)
        quality_component = selected_result.quality * 0.5

        # Component 2: Agreement level (30% weight)
        agreement_component = agreement_level * 0.3

        # Component 3: Vote share (20% weight)
        vote_share = vote_distribution.get(selected_result.strategy, 0.0)
        vote_component = vote_share * 0.2

        consensus_confidence = quality_component + agreement_component + vote_component

        return min(1.0, consensus_confidence)

    def get_statistics(self) -> Dict[str, Any]:
        """Get consensus refinement statistics"""

        avg_parallel_time = (
            self.total_parallel_time_ms / self.total_refinements
            if self.total_refinements > 0 else 0.0
        )

        avg_sequential_time = (
            self.total_sequential_time_ms / self.total_refinements
            if self.total_refinements > 0 else 0.0
        )

        avg_speedup = (
            avg_sequential_time / avg_parallel_time
            if avg_parallel_time > 0 else 1.0
        )

        return {
            'total_refinements': self.total_refinements,
            'avg_parallel_time_ms': avg_parallel_time,
            'avg_sequential_time_ms': avg_sequential_time,
            'avg_parallel_speedup': avg_speedup,
            'strategies_count': len(self.strategies),
            'voting_method': self.voting_method.value
        }
