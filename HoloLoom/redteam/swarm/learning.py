"""Hierarchical learning system for CARTS multi-agent swarm.

CARTS Phase 4: Learning & Adaptation System
Status: Foundation Implementation (December 2025)
Location: HoloLoom/redteam/swarm/learning.py

FIRST PRINCIPLE: "Hierarchical learning at different timescales"

This module implements 4 learning timescales:
1. Per-Attack (Immediate): Payload heat tracking
2. Per-Task (~seconds): Thompson Sampling strategy updates
3. Per-Cycle (~minutes): Cross-strategy learning
4. Background (~hours): Prior updates, pattern mining

Principle 5.2 (from first principles document):
"Share discoveries, not strategies" - Share found vulnerabilities,
blocked patterns, surface maps. Don't share Thompson posteriors
(let each agent learn independently).

Principle 5.3: "Hot patterns propagate, cold patterns decay"
heat_score = access_count × success_rate × recency_decay

Principle 5.6: "Agents compete on execution, cooperate on discovery"
Competition (healthy): Performance metrics
Cooperation (essential): Share discoveries, blocked patterns
"""

import asyncio
import time
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from collections import defaultdict


# =============================================================================
# Learning Timescales
# =============================================================================


class LearningTimescale(Enum):
    """The 4 hierarchical learning timescales."""

    PER_ATTACK = "per_attack"      # Immediate (milliseconds)
    PER_TASK = "per_task"          # Seconds
    PER_CYCLE = "per_cycle"        # Minutes
    BACKGROUND = "background"       # Hours


# =============================================================================
# Learning Events
# =============================================================================


@dataclass
class LearningEvent:
    """A learning event to be processed at the appropriate timescale."""

    timescale: LearningTimescale
    event_type: str
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    agent_id: Optional[str] = None
    priority: float = 1.0  # Higher = more important


@dataclass
class PatternUpdate:
    """An update to a learned pattern."""

    pattern_id: str
    heat_delta: float
    success: bool
    confidence: float
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Timescale 1: Per-Attack Learning (Immediate)
# =============================================================================


@dataclass
class PayloadHeat:
    """Heat tracking for individual payloads.

    Heat = access_count × success_rate × confidence × recency_decay

    Hot payloads get priority for future use.
    Cold payloads decay and are eventually pruned.
    """

    payload_id: str
    payload_hash: str  # For deduplication
    access_count: int = 0
    success_count: int = 0
    total_confidence: float = 0.0
    last_access: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    blocked_count: int = 0  # Times this payload was blocked
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.access_count == 0:
            return 0.0
        return self.success_count / self.access_count

    @property
    def avg_confidence(self) -> float:
        """Calculate average confidence."""
        if self.access_count == 0:
            return 0.0
        return self.total_confidence / self.access_count

    def recency_decay(self, decay_rate: float = 0.05, hours_since: Optional[float] = None) -> float:
        """Calculate recency decay.

        Exponential decay: decay_rate^hours_since_last_access
        Default: 5% decay per hour
        """
        if hours_since is None:
            hours_since = (time.time() - self.last_access) / 3600.0
        return math.pow(1 - decay_rate, hours_since)

    def heat_score(self, decay_rate: float = 0.05) -> float:
        """Calculate heat score.

        heat = access_count × success_rate × avg_confidence × recency_decay
        """
        return (
            self.access_count *
            self.success_rate *
            self.avg_confidence *
            self.recency_decay(decay_rate)
        )


class PerAttackLearner:
    """Per-attack learning: Immediate payload heat tracking.

    FIRST PRINCIPLE: Immediate feedback for fastest adaptation.

    This learner tracks:
    - Individual payload effectiveness
    - Blocked patterns (to avoid repetition)
    - Heat scores for prioritization
    """

    # Heat thresholds
    HOT_THRESHOLD = 10.0  # High enough to propagate
    COLD_THRESHOLD = 0.1  # Low enough to prune

    def __init__(
        self,
        max_payloads: int = 10000,
        decay_rate: float = 0.05,
        prune_interval: float = 3600.0,  # 1 hour
    ):
        """Initialize per-attack learner.

        Args:
            max_payloads: Maximum payloads to track
            decay_rate: Hourly decay rate (0.05 = 5% per hour)
            prune_interval: How often to prune cold payloads (seconds)
        """
        self._payloads: Dict[str, PayloadHeat] = {}
        self._hash_to_id: Dict[str, str] = {}  # Deduplication
        self._blocked_patterns: Set[str] = set()
        self._max_payloads = max_payloads
        self._decay_rate = decay_rate
        self._prune_interval = prune_interval
        self._last_prune = time.time()
        self._lock = asyncio.Lock()

    async def record_attack(
        self,
        payload_id: str,
        payload_hash: str,
        success: bool,
        confidence: float,
        blocked: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PayloadHeat:
        """Record an attack result immediately.

        This is called after every attack attempt.

        Args:
            payload_id: Unique payload identifier
            payload_hash: Hash for deduplication
            success: Whether the attack succeeded
            confidence: Confidence score (0.0-1.0)
            blocked: Whether the payload was blocked
            metadata: Optional additional data

        Returns:
            Updated PayloadHeat
        """
        async with self._lock:
            # Check for duplicate via hash
            if payload_hash in self._hash_to_id:
                existing_id = self._hash_to_id[payload_hash]
                heat = self._payloads[existing_id]
            elif payload_id in self._payloads:
                heat = self._payloads[payload_id]
            else:
                # New payload
                heat = PayloadHeat(
                    payload_id=payload_id,
                    payload_hash=payload_hash,
                    metadata=metadata or {},
                )
                self._payloads[payload_id] = heat
                self._hash_to_id[payload_hash] = payload_id

            # Update heat
            heat.access_count += 1
            heat.last_access = time.time()
            heat.total_confidence += confidence

            if success:
                heat.success_count += 1

            if blocked:
                heat.blocked_count += 1
                self._blocked_patterns.add(payload_hash)

            if metadata:
                heat.metadata.update(metadata)

            # Prune if needed
            await self._maybe_prune()

            return heat

    async def get_hot_payloads(self, limit: int = 100) -> List[PayloadHeat]:
        """Get hottest payloads for priority use.

        Args:
            limit: Maximum payloads to return

        Returns:
            List of PayloadHeat sorted by heat score (descending)
        """
        async with self._lock:
            payloads = list(self._payloads.values())
            payloads.sort(key=lambda p: p.heat_score(self._decay_rate), reverse=True)
            return payloads[:limit]

    async def get_cold_payloads(self, limit: int = 100) -> List[PayloadHeat]:
        """Get coldest payloads for potential pruning.

        Args:
            limit: Maximum payloads to return

        Returns:
            List of PayloadHeat sorted by heat score (ascending)
        """
        async with self._lock:
            payloads = list(self._payloads.values())
            payloads.sort(key=lambda p: p.heat_score(self._decay_rate))
            return payloads[:limit]

    def is_blocked(self, payload_hash: str) -> bool:
        """Check if a payload pattern is blocked.

        Args:
            payload_hash: Hash of payload to check

        Returns:
            True if blocked
        """
        return payload_hash in self._blocked_patterns

    async def _maybe_prune(self) -> int:
        """Prune cold payloads if interval elapsed.

        Returns:
            Number of payloads pruned
        """
        now = time.time()
        if now - self._last_prune < self._prune_interval:
            return 0

        self._last_prune = now

        # Find cold payloads
        to_prune = []
        for payload_id, heat in self._payloads.items():
            if heat.heat_score(self._decay_rate) < self.COLD_THRESHOLD:
                to_prune.append(payload_id)

        # Prune
        for payload_id in to_prune:
            heat = self._payloads.pop(payload_id)
            self._hash_to_id.pop(heat.payload_hash, None)

        return len(to_prune)

    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        payloads = list(self._payloads.values())
        hot = [p for p in payloads if p.heat_score(self._decay_rate) >= self.HOT_THRESHOLD]
        cold = [p for p in payloads if p.heat_score(self._decay_rate) < self.COLD_THRESHOLD]

        return {
            "total_payloads": len(payloads),
            "hot_payloads": len(hot),
            "cold_payloads": len(cold),
            "blocked_patterns": len(self._blocked_patterns),
            "avg_success_rate": (
                sum(p.success_rate for p in payloads) / len(payloads)
                if payloads else 0.0
            ),
        }


# =============================================================================
# Timescale 2: Per-Task Learning (~seconds)
# =============================================================================


@dataclass
class ThompsonSamplingPrior:
    """Thompson Sampling prior for strategy selection.

    Uses Beta(α, β) distribution:
    - α = successes + 1
    - β = failures + 1

    Expected reward: E[X] = α / (α + β)
    """

    strategy_id: str
    alpha: float = 1.0  # Prior successes + 1
    beta: float = 1.0   # Prior failures + 1
    total_attempts: int = 0
    last_updated: float = field(default_factory=time.time)
    context_stats: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    @property
    def expected_reward(self) -> float:
        """Calculate expected reward."""
        return self.alpha / (self.alpha + self.beta)

    def sample(self) -> float:
        """Sample from Beta distribution for Thompson Sampling."""
        import random
        # Use random.betavariate for sampling
        return random.betavariate(self.alpha, self.beta)

    def update(self, success: bool, confidence: float = 1.0) -> None:
        """Update prior based on outcome.

        Args:
            success: Whether the attempt succeeded
            confidence: Confidence weight (0.0-1.0)
        """
        self.total_attempts += 1
        self.last_updated = time.time()

        if success:
            self.alpha += confidence
        else:
            self.beta += (1.0 - confidence)


class PerTaskLearner:
    """Per-task learning: Thompson Sampling strategy updates.

    FIRST PRINCIPLE: "Thompson Sampling is the right paradigm"
    - Explores proportionally to uncertainty
    - Naturally balances exploration/exploitation
    - Updates in seconds (per-task)
    """

    def __init__(
        self,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        context_weight: float = 0.3,
    ):
        """Initialize per-task learner.

        Args:
            prior_alpha: Initial alpha for new strategies
            prior_beta: Initial beta for new strategies
            context_weight: Weight for context-aware learning
        """
        self._priors: Dict[str, ThompsonSamplingPrior] = {}
        self._prior_alpha = prior_alpha
        self._prior_beta = prior_beta
        self._context_weight = context_weight
        self._lock = asyncio.Lock()

    async def select_strategy(
        self,
        available_strategies: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Select strategy using Thompson Sampling.

        Args:
            available_strategies: List of strategy IDs to choose from
            context: Optional context for context-aware selection

        Returns:
            Selected strategy ID
        """
        async with self._lock:
            samples = {}

            for strategy_id in available_strategies:
                prior = self._get_or_create_prior(strategy_id)
                samples[strategy_id] = prior.sample()

            # Select highest sample
            selected = max(samples, key=samples.get)
            return selected

    async def update_strategy(
        self,
        strategy_id: str,
        success: bool,
        confidence: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> ThompsonSamplingPrior:
        """Update strategy prior based on outcome.

        Args:
            strategy_id: Strategy that was used
            success: Whether it succeeded
            confidence: Confidence score (0.0-1.0)
            context: Optional context for context-aware learning

        Returns:
            Updated prior
        """
        async with self._lock:
            prior = self._get_or_create_prior(strategy_id)
            prior.update(success, confidence)

            # Context-aware update
            if context and self._context_weight > 0:
                context_key = self._context_hash(context)
                if context_key not in prior.context_stats:
                    prior.context_stats[context_key] = (1.0, 1.0)

                ctx_alpha, ctx_beta = prior.context_stats[context_key]
                if success:
                    ctx_alpha += confidence * self._context_weight
                else:
                    ctx_beta += (1.0 - confidence) * self._context_weight
                prior.context_stats[context_key] = (ctx_alpha, ctx_beta)

            return prior

    async def get_strategy_stats(self, strategy_id: str) -> Dict[str, Any]:
        """Get statistics for a strategy.

        Args:
            strategy_id: Strategy to get stats for

        Returns:
            Strategy statistics
        """
        async with self._lock:
            if strategy_id not in self._priors:
                return {"exists": False}

            prior = self._priors[strategy_id]
            return {
                "exists": True,
                "alpha": prior.alpha,
                "beta": prior.beta,
                "expected_reward": prior.expected_reward,
                "total_attempts": prior.total_attempts,
                "last_updated": prior.last_updated,
                "context_count": len(prior.context_stats),
            }

    def _get_or_create_prior(self, strategy_id: str) -> ThompsonSamplingPrior:
        """Get or create prior for strategy."""
        if strategy_id not in self._priors:
            self._priors[strategy_id] = ThompsonSamplingPrior(
                strategy_id=strategy_id,
                alpha=self._prior_alpha,
                beta=self._prior_beta,
            )
        return self._priors[strategy_id]

    def _context_hash(self, context: Dict[str, Any]) -> str:
        """Create hash for context."""
        # Simple hash based on sorted keys
        items = sorted((k, str(v)) for k, v in context.items())
        return ":".join(f"{k}={v}" for k, v in items)

    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        priors = list(self._priors.values())

        return {
            "total_strategies": len(priors),
            "total_attempts": sum(p.total_attempts for p in priors),
            "avg_expected_reward": (
                sum(p.expected_reward for p in priors) / len(priors)
                if priors else 0.0
            ),
            "strategies": {
                p.strategy_id: {
                    "alpha": p.alpha,
                    "beta": p.beta,
                    "expected": p.expected_reward,
                }
                for p in priors
            },
        }


# =============================================================================
# Timescale 3: Per-Cycle Learning (~minutes)
# =============================================================================


@dataclass
class CrossStrategyInsight:
    """Insight learned from cross-strategy analysis."""

    insight_id: str
    insight_type: str  # e.g., "correlation", "sequence", "conflict"
    strategies_involved: List[str]
    description: str
    confidence: float
    evidence_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerCycleLearner:
    """Per-cycle learning: Cross-strategy pattern mining.

    FIRST PRINCIPLE: "Share discoveries, not strategies"

    This learner:
    - Analyzes patterns across strategies
    - Detects correlations and sequences
    - Identifies conflicts between strategies
    - Operates on ~minute timescale
    """

    def __init__(
        self,
        cycle_duration: float = 60.0,  # 1 minute
        min_evidence: int = 3,
        correlation_threshold: float = 0.6,
    ):
        """Initialize per-cycle learner.

        Args:
            cycle_duration: Duration of analysis cycle (seconds)
            min_evidence: Minimum evidence count for insights
            correlation_threshold: Threshold for correlation detection
        """
        self._cycle_duration = cycle_duration
        self._min_evidence = min_evidence
        self._correlation_threshold = correlation_threshold

        self._insights: Dict[str, CrossStrategyInsight] = {}
        self._event_buffer: List[Dict[str, Any]] = []
        self._strategy_sequences: List[Tuple[str, ...]] = []
        self._last_cycle = time.time()
        self._lock = asyncio.Lock()

    async def record_event(
        self,
        strategy_id: str,
        event_type: str,
        success: bool,
        confidence: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a strategy event for cross-strategy analysis.

        Args:
            strategy_id: Strategy that generated the event
            event_type: Type of event
            success: Whether successful
            confidence: Confidence score
            metadata: Additional metadata
        """
        async with self._lock:
            self._event_buffer.append({
                "strategy_id": strategy_id,
                "event_type": event_type,
                "success": success,
                "confidence": confidence,
                "timestamp": time.time(),
                "metadata": metadata or {},
            })

    async def analyze_cycle(self) -> List[CrossStrategyInsight]:
        """Analyze current cycle for cross-strategy patterns.

        Called periodically (every cycle_duration seconds).

        Returns:
            List of new or updated insights
        """
        async with self._lock:
            now = time.time()
            if now - self._last_cycle < self._cycle_duration:
                return []

            self._last_cycle = now
            new_insights = []

            # Analyze correlations
            correlations = await self._analyze_correlations()
            for corr in correlations:
                insight = self._update_or_create_insight(
                    insight_type="correlation",
                    strategies=corr["strategies"],
                    description=corr["description"],
                    confidence=corr["confidence"],
                )
                new_insights.append(insight)

            # Analyze sequences
            sequences = await self._analyze_sequences()
            for seq in sequences:
                insight = self._update_or_create_insight(
                    insight_type="sequence",
                    strategies=seq["strategies"],
                    description=seq["description"],
                    confidence=seq["confidence"],
                )
                new_insights.append(insight)

            # Clear old events (keep last cycle's worth)
            cutoff = now - self._cycle_duration * 2
            self._event_buffer = [
                e for e in self._event_buffer
                if e["timestamp"] > cutoff
            ]

            return new_insights

    async def get_insights(
        self,
        insight_type: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> List[CrossStrategyInsight]:
        """Get learned insights.

        Args:
            insight_type: Optional filter by type
            min_confidence: Minimum confidence filter

        Returns:
            List of insights
        """
        async with self._lock:
            insights = list(self._insights.values())

            if insight_type:
                insights = [i for i in insights if i.insight_type == insight_type]

            insights = [i for i in insights if i.confidence >= min_confidence]

            return sorted(insights, key=lambda i: i.confidence, reverse=True)

    async def _analyze_correlations(self) -> List[Dict[str, Any]]:
        """Analyze correlations between strategies."""
        correlations = []

        # Group events by strategy
        by_strategy: Dict[str, List[Dict]] = defaultdict(list)
        for event in self._event_buffer:
            by_strategy[event["strategy_id"]].append(event)

        strategies = list(by_strategy.keys())

        # Check pairwise correlations
        for i, strat1 in enumerate(strategies):
            for strat2 in strategies[i + 1:]:
                events1 = by_strategy[strat1]
                events2 = by_strategy[strat2]

                if len(events1) < self._min_evidence or len(events2) < self._min_evidence:
                    continue

                # Simple correlation: success rate comparison
                success1 = sum(1 for e in events1 if e["success"]) / len(events1)
                success2 = sum(1 for e in events2 if e["success"]) / len(events2)

                correlation = 1 - abs(success1 - success2)

                if correlation >= self._correlation_threshold:
                    correlations.append({
                        "strategies": [strat1, strat2],
                        "description": f"Correlated success rates: {strat1}={success1:.2f}, {strat2}={success2:.2f}",
                        "confidence": correlation,
                    })

        return correlations

    async def _analyze_sequences(self) -> List[Dict[str, Any]]:
        """Analyze strategy sequences."""
        sequences = []

        # Extract strategy sequence from events
        sorted_events = sorted(self._event_buffer, key=lambda e: e["timestamp"])

        if len(sorted_events) < 2:
            return sequences

        # Track successful sequences
        success_sequences: Dict[Tuple[str, str], int] = defaultdict(int)
        total_sequences: Dict[Tuple[str, str], int] = defaultdict(int)

        for i in range(len(sorted_events) - 1):
            prev = sorted_events[i]
            curr = sorted_events[i + 1]

            seq = (prev["strategy_id"], curr["strategy_id"])
            total_sequences[seq] += 1

            if curr["success"]:
                success_sequences[seq] += 1

        # Find high-success sequences
        for seq, total in total_sequences.items():
            if total < self._min_evidence:
                continue

            success_rate = success_sequences[seq] / total

            if success_rate >= 0.7:  # Good sequence
                sequences.append({
                    "strategies": list(seq),
                    "description": f"Sequence {seq[0]} → {seq[1]} has {success_rate:.0%} success rate",
                    "confidence": success_rate,
                })

        return sequences

    def _update_or_create_insight(
        self,
        insight_type: str,
        strategies: List[str],
        description: str,
        confidence: float,
    ) -> CrossStrategyInsight:
        """Update existing or create new insight."""
        insight_id = f"{insight_type}:{':'.join(sorted(strategies))}"

        if insight_id in self._insights:
            insight = self._insights[insight_id]
            insight.confidence = (insight.confidence + confidence) / 2  # Running average
            insight.evidence_count += 1
            insight.last_seen = time.time()
        else:
            insight = CrossStrategyInsight(
                insight_id=insight_id,
                insight_type=insight_type,
                strategies_involved=strategies,
                description=description,
                confidence=confidence,
                evidence_count=1,
            )
            self._insights[insight_id] = insight

        return insight

    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        insights = list(self._insights.values())

        return {
            "total_insights": len(insights),
            "by_type": {
                insight_type: len([i for i in insights if i.insight_type == insight_type])
                for insight_type in set(i.insight_type for i in insights)
            },
            "avg_confidence": (
                sum(i.confidence for i in insights) / len(insights)
                if insights else 0.0
            ),
            "event_buffer_size": len(self._event_buffer),
        }


# =============================================================================
# Timescale 4: Background Learning (~hours)
# =============================================================================


@dataclass
class SystemPrior:
    """System-wide prior learned from long-term observation."""

    prior_id: str
    prior_type: str  # e.g., "target_type", "defense_pattern", "time_of_day"
    parameters: Dict[str, float]
    evidence_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearnedPattern:
    """A pattern learned from background analysis."""

    pattern_id: str
    pattern_type: str
    template: str  # Pattern template/signature
    frequency: int = 0
    success_rate: float = 0.0
    confidence: float = 0.0
    contexts: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)


class BackgroundLearner:
    """Background learning: Prior updates and pattern mining.

    FIRST PRINCIPLE: "Background updates are for system-wide optimization"

    This learner:
    - Updates system-wide priors (target types, defense patterns)
    - Mines patterns from historical data
    - Runs on ~hour timescale to avoid frequent updates
    - Provides stable, well-validated insights
    """

    def __init__(
        self,
        update_interval: float = 3600.0,  # 1 hour
        min_pattern_frequency: int = 10,
        pattern_confidence_threshold: float = 0.7,
    ):
        """Initialize background learner.

        Args:
            update_interval: Interval between updates (seconds)
            min_pattern_frequency: Minimum frequency for patterns
            pattern_confidence_threshold: Minimum confidence for patterns
        """
        self._update_interval = update_interval
        self._min_pattern_frequency = min_pattern_frequency
        self._pattern_confidence_threshold = pattern_confidence_threshold

        self._priors: Dict[str, SystemPrior] = {}
        self._patterns: Dict[str, LearnedPattern] = {}
        self._history_buffer: List[Dict[str, Any]] = []
        self._last_update = time.time()
        self._lock = asyncio.Lock()

        # Background task handle
        self._background_task: Optional[asyncio.Task] = None
        self._running = False

    async def start_background_learning(self) -> None:
        """Start background learning loop."""
        if self._running:
            return

        self._running = True
        self._background_task = asyncio.create_task(self._background_loop())

    async def stop_background_learning(self) -> None:
        """Stop background learning loop."""
        self._running = False
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
            self._background_task = None

    async def record_history(
        self,
        event_type: str,
        data: Dict[str, Any],
        success: bool,
        confidence: float,
    ) -> None:
        """Record historical event for background analysis.

        Args:
            event_type: Type of event
            data: Event data
            success: Whether successful
            confidence: Confidence score
        """
        async with self._lock:
            self._history_buffer.append({
                "event_type": event_type,
                "data": data,
                "success": success,
                "confidence": confidence,
                "timestamp": time.time(),
            })

    async def get_prior(self, prior_type: str, context: Dict[str, Any]) -> Optional[SystemPrior]:
        """Get system prior for a context.

        Args:
            prior_type: Type of prior to get
            context: Context to match

        Returns:
            Matching prior if found
        """
        async with self._lock:
            for prior in self._priors.values():
                if prior.prior_type == prior_type:
                    return prior
            return None

    async def get_patterns(
        self,
        pattern_type: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> List[LearnedPattern]:
        """Get learned patterns.

        Args:
            pattern_type: Optional filter by type
            min_confidence: Minimum confidence filter

        Returns:
            List of patterns
        """
        async with self._lock:
            patterns = list(self._patterns.values())

            if pattern_type:
                patterns = [p for p in patterns if p.pattern_type == pattern_type]

            patterns = [p for p in patterns if p.confidence >= min_confidence]

            return sorted(patterns, key=lambda p: p.confidence, reverse=True)

    async def _background_loop(self) -> None:
        """Background learning loop."""
        while self._running:
            try:
                await asyncio.sleep(self._update_interval)
                await self._perform_update()
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but continue
                pass

    async def _perform_update(self) -> None:
        """Perform background update."""
        async with self._lock:
            now = time.time()

            # Update priors
            await self._update_priors()

            # Mine patterns
            await self._mine_patterns()

            # Clean old history (keep last 24 hours)
            cutoff = now - 86400
            self._history_buffer = [
                h for h in self._history_buffer
                if h["timestamp"] > cutoff
            ]

            self._last_update = now

    async def _update_priors(self) -> None:
        """Update system-wide priors from history."""
        if not self._history_buffer:
            return

        # Aggregate by event type
        by_type: Dict[str, List[Dict]] = defaultdict(list)
        for event in self._history_buffer:
            by_type[event["event_type"]].append(event)

        for event_type, events in by_type.items():
            if len(events) < self._min_pattern_frequency:
                continue

            success_rate = sum(1 for e in events if e["success"]) / len(events)
            avg_confidence = sum(e["confidence"] for e in events) / len(events)

            prior_id = f"prior:{event_type}"
            if prior_id in self._priors:
                prior = self._priors[prior_id]
                # Exponential moving average
                alpha = 0.3
                prior.parameters["success_rate"] = (
                    alpha * success_rate +
                    (1 - alpha) * prior.parameters.get("success_rate", success_rate)
                )
                prior.parameters["avg_confidence"] = (
                    alpha * avg_confidence +
                    (1 - alpha) * prior.parameters.get("avg_confidence", avg_confidence)
                )
                prior.evidence_count += len(events)
                prior.last_updated = time.time()
            else:
                self._priors[prior_id] = SystemPrior(
                    prior_id=prior_id,
                    prior_type=event_type,
                    parameters={
                        "success_rate": success_rate,
                        "avg_confidence": avg_confidence,
                    },
                    evidence_count=len(events),
                )

    async def _mine_patterns(self) -> None:
        """Mine patterns from historical data."""
        if not self._history_buffer:
            return

        # Simple pattern: success patterns by data signature
        success_signatures: Dict[str, List[Dict]] = defaultdict(list)

        for event in self._history_buffer:
            if event["success"]:
                # Create simple signature from data
                sig = self._create_signature(event["data"])
                success_signatures[sig].append(event)

        for sig, events in success_signatures.items():
            if len(events) < self._min_pattern_frequency:
                continue

            avg_confidence = sum(e["confidence"] for e in events) / len(events)

            if avg_confidence < self._pattern_confidence_threshold:
                continue

            pattern_id = f"pattern:{sig}"
            if pattern_id in self._patterns:
                pattern = self._patterns[pattern_id]
                pattern.frequency += len(events)
                pattern.success_rate = 1.0  # All success patterns
                pattern.confidence = (pattern.confidence + avg_confidence) / 2
                pattern.last_seen = time.time()
            else:
                self._patterns[pattern_id] = LearnedPattern(
                    pattern_id=pattern_id,
                    pattern_type="success",
                    template=sig,
                    frequency=len(events),
                    success_rate=1.0,
                    confidence=avg_confidence,
                )

    def _create_signature(self, data: Dict[str, Any]) -> str:
        """Create simple signature from data."""
        # Extract key fields for signature
        key_fields = ["type", "category", "target_type", "strategy"]
        parts = []
        for field in key_fields:
            if field in data:
                parts.append(f"{field}:{data[field]}")
        return "|".join(sorted(parts)) if parts else "default"

    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return {
            "total_priors": len(self._priors),
            "total_patterns": len(self._patterns),
            "history_buffer_size": len(self._history_buffer),
            "last_update": self._last_update,
            "priors": {
                p.prior_id: p.parameters
                for p in self._priors.values()
            },
            "top_patterns": [
                {
                    "pattern_id": p.pattern_id,
                    "frequency": p.frequency,
                    "confidence": p.confidence,
                }
                for p in sorted(
                    self._patterns.values(),
                    key=lambda x: x.confidence,
                    reverse=True,
                )[:10]
            ],
        }


# =============================================================================
# Unified Learning Coordinator
# =============================================================================


class HierarchicalLearningCoordinator:
    """Coordinates all 4 timescales of learning.

    FIRST PRINCIPLE: "Different decisions need different learning speeds"

    This coordinator:
    - Routes learning events to appropriate timescale
    - Propagates insights across timescales
    - Provides unified interface for agents
    - Manages learning lifecycle
    """

    def __init__(
        self,
        per_attack_config: Optional[Dict[str, Any]] = None,
        per_task_config: Optional[Dict[str, Any]] = None,
        per_cycle_config: Optional[Dict[str, Any]] = None,
        background_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize learning coordinator.

        Args:
            per_attack_config: Config for per-attack learner
            per_task_config: Config for per-task learner
            per_cycle_config: Config for per-cycle learner
            background_config: Config for background learner
        """
        self._per_attack = PerAttackLearner(**(per_attack_config or {}))
        self._per_task = PerTaskLearner(**(per_task_config or {}))
        self._per_cycle = PerCycleLearner(**(per_cycle_config or {}))
        self._background = BackgroundLearner(**(background_config or {}))

        self._running = False
        self._cycle_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start all learning systems."""
        if self._running:
            return

        self._running = True

        # Start background learning
        await self._background.start_background_learning()

        # Start cycle analysis loop
        self._cycle_task = asyncio.create_task(self._cycle_loop())

    async def stop(self) -> None:
        """Stop all learning systems."""
        self._running = False

        # Stop background learning
        await self._background.stop_background_learning()

        # Stop cycle analysis
        if self._cycle_task:
            self._cycle_task.cancel()
            try:
                await self._cycle_task
            except asyncio.CancelledError:
                pass
            self._cycle_task = None

    async def learn_from_attack(
        self,
        payload_id: str,
        payload_hash: str,
        strategy_id: str,
        success: bool,
        confidence: float,
        blocked: bool = False,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Learn from a single attack.

        Routes to all appropriate timescales.

        Args:
            payload_id: Payload identifier
            payload_hash: Payload hash for deduplication
            strategy_id: Strategy used
            success: Whether successful
            confidence: Confidence score
            blocked: Whether blocked
            context: Optional context
            metadata: Optional metadata

        Returns:
            Learning results from all timescales
        """
        results = {}

        # Timescale 1: Per-attack (immediate)
        heat = await self._per_attack.record_attack(
            payload_id=payload_id,
            payload_hash=payload_hash,
            success=success,
            confidence=confidence,
            blocked=blocked,
            metadata=metadata,
        )
        results["per_attack"] = {
            "heat_score": heat.heat_score(),
            "success_rate": heat.success_rate,
        }

        # Timescale 2: Per-task (seconds)
        prior = await self._per_task.update_strategy(
            strategy_id=strategy_id,
            success=success,
            confidence=confidence,
            context=context,
        )
        results["per_task"] = {
            "expected_reward": prior.expected_reward,
            "alpha": prior.alpha,
            "beta": prior.beta,
        }

        # Timescale 3: Per-cycle (record for later)
        await self._per_cycle.record_event(
            strategy_id=strategy_id,
            event_type="attack",
            success=success,
            confidence=confidence,
            metadata={"payload_id": payload_id, **(metadata or {})},
        )
        results["per_cycle"] = {"recorded": True}

        # Timescale 4: Background (record for later)
        await self._background.record_history(
            event_type="attack",
            data={
                "payload_id": payload_id,
                "strategy_id": strategy_id,
                **(context or {}),
            },
            success=success,
            confidence=confidence,
        )
        results["background"] = {"recorded": True}

        return results

    async def select_strategy(
        self,
        available_strategies: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Select strategy using learned priors.

        Uses Thompson Sampling from per-task learner.

        Args:
            available_strategies: Strategies to choose from
            context: Optional context

        Returns:
            Selected strategy ID
        """
        return await self._per_task.select_strategy(
            available_strategies=available_strategies,
            context=context,
        )

    async def get_hot_payloads(self, limit: int = 100) -> List[PayloadHeat]:
        """Get hot payloads for priority use."""
        return await self._per_attack.get_hot_payloads(limit)

    def is_payload_blocked(self, payload_hash: str) -> bool:
        """Check if payload is blocked."""
        return self._per_attack.is_blocked(payload_hash)

    async def get_insights(self, min_confidence: float = 0.6) -> List[CrossStrategyInsight]:
        """Get cross-strategy insights."""
        return await self._per_cycle.get_insights(min_confidence=min_confidence)

    async def get_patterns(self, min_confidence: float = 0.7) -> List[LearnedPattern]:
        """Get learned patterns."""
        return await self._background.get_patterns(min_confidence=min_confidence)

    async def _cycle_loop(self) -> None:
        """Periodic cycle analysis loop."""
        while self._running:
            try:
                await asyncio.sleep(60.0)  # Check every minute
                insights = await self._per_cycle.analyze_cycle()

                # Propagate hot insights to other timescales
                for insight in insights:
                    if insight.confidence >= 0.8:
                        await self._propagate_insight(insight)

            except asyncio.CancelledError:
                break
            except Exception:
                pass

    async def _propagate_insight(self, insight: CrossStrategyInsight) -> None:
        """Propagate high-confidence insight."""
        # Record in background for long-term learning
        await self._background.record_history(
            event_type=f"insight:{insight.insight_type}",
            data={
                "strategies": insight.strategies_involved,
                "description": insight.description,
            },
            success=True,
            confidence=insight.confidence,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics."""
        return {
            "per_attack": self._per_attack.get_stats(),
            "per_task": self._per_task.get_stats(),
            "per_cycle": self._per_cycle.get_stats(),
            "background": self._background.get_stats(),
        }


# =============================================================================
# Factory Functions
# =============================================================================


def create_learning_coordinator(
    config: Optional[Dict[str, Any]] = None,
) -> HierarchicalLearningCoordinator:
    """Create a learning coordinator with optional configuration.

    Args:
        config: Optional configuration dict with keys:
            - per_attack: Config for per-attack learner
            - per_task: Config for per-task learner
            - per_cycle: Config for per-cycle learner
            - background: Config for background learner

    Returns:
        Configured HierarchicalLearningCoordinator
    """
    config = config or {}

    return HierarchicalLearningCoordinator(
        per_attack_config=config.get("per_attack"),
        per_task_config=config.get("per_task"),
        per_cycle_config=config.get("per_cycle"),
        background_config=config.get("background"),
    )
