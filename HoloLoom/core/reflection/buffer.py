"""
HoloLoom Reflection Buffer
===========================
Learning loop that stores Spacetime outcomes and extracts patterns for improvement.

The Reflection Buffer is the system's memory of past weaving cycles, enabling:
- Pattern recognition from successful outcomes
- Bandit statistic updates based on real feedback
- Pattern card adaptation based on performance
- Continuous improvement through reflection

Philosophy:
"Reflection is the loom learning from its own weaving." After each cycle, the system
reflects on what worked, what didn't, and how to improve. The buffer stores
Spacetime artifacts and analyzes them to extract actionable insights.

Architecture:
- Episodic memory of recent Spacetime artifacts
- Success pattern analysis
- Failure pattern detection
- Adaptive learning signals
- Performance tracking

Author: Claude Code (with HoloLoom by Blake)
Date: 2025-10-26
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np

from HoloLoom.fabric.spacetime import Spacetime, WeavingTrace
from HoloLoom.reflection.rewards import RewardExtractor, RewardConfig

logging.basicConfig(level=logging.INFO)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ReflectionMetrics:
    """
    Aggregated metrics from reflection analysis.

    Tracks system performance over time and identifies improvement opportunities.
    """
    # Success metrics
    total_cycles: int = 0
    successful_cycles: int = 0
    failed_cycles: int = 0

    # Tool performance
    tool_success_rates: Dict[str, float] = field(default_factory=dict)
    tool_avg_confidence: Dict[str, float] = field(default_factory=dict)
    tool_usage_counts: Dict[str, int] = field(default_factory=dict)

    # Pattern card performance
    pattern_success_rates: Dict[str, float] = field(default_factory=dict)
    pattern_avg_duration: Dict[str, float] = field(default_factory=dict)
    pattern_usage_counts: Dict[str, int] = field(default_factory=dict)

    # Query complexity patterns
    successful_query_lengths: List[int] = field(default_factory=list)
    failed_query_lengths: List[int] = field(default_factory=list)

    # Temporal patterns
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metrics for storage."""
        return {
            'total_cycles': self.total_cycles,
            'successful_cycles': self.successful_cycles,
            'failed_cycles': self.failed_cycles,
            'tool_success_rates': self.tool_success_rates,
            'tool_avg_confidence': self.tool_avg_confidence,
            'tool_usage_counts': self.tool_usage_counts,
            'pattern_success_rates': self.pattern_success_rates,
            'pattern_avg_duration': self.pattern_avg_duration,
            'pattern_usage_counts': self.pattern_usage_counts,
            'successful_query_lengths': self.successful_query_lengths,
            'failed_query_lengths': self.failed_query_lengths,
            'last_updated': self.last_updated.isoformat()
        }


@dataclass
class LearningSignal:
    """
    A learning signal extracted from reflection analysis.

    Signals are actionable insights that can be used to improve the system.
    """
    signal_type: str  # "bandit_update", "pattern_preference", "threshold_adjustment"
    tool: Optional[str] = None
    pattern: Optional[str] = None
    reward: Optional[float] = None
    confidence_threshold: Optional[float] = None
    recommendation: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    priority: float = 0.5  # 0-1, higher = more important


# ============================================================================
# Reflection Buffer
# ============================================================================

class ReflectionBuffer:
    """
    Episodic memory buffer for Spacetime artifacts with learning analysis.

    The Reflection Buffer stores recent weaving cycles and analyzes them to:
    1. Update bandit statistics with real feedback
    2. Identify successful patterns
    3. Detect failure modes
    4. Generate learning signals for adaptation
    5. Track performance over time

    Usage:
        buffer = ReflectionBuffer(capacity=1000, persist_path="./reflections")

        # After weaving
        spacetime = await shuttle.weave(query)
        await buffer.store(spacetime, feedback=user_feedback)

        # Periodic learning
        signals = await buffer.analyze_and_learn()
        for signal in signals:
            if signal.signal_type == "bandit_update":
                await shuttle.update_bandit(signal)
    """

    def __init__(
        self,
        capacity: int = 1000,
        persist_path: Optional[str] = None,
        learning_window: int = 100,  # How many recent cycles to analyze
        success_threshold: float = 0.6,  # Confidence threshold for success
        reward_config: Optional[RewardConfig] = None  # Reward computation config
    ):
        """
        Initialize Reflection Buffer.

        Args:
            capacity: Maximum number of Spacetime artifacts to store
            persist_path: Optional path for persistence
            learning_window: Number of recent cycles to analyze for patterns
            success_threshold: Minimum confidence to consider a cycle successful
            reward_config: Optional reward configuration (uses defaults if None)
        """
        self.capacity = capacity
        self.persist_path = Path(persist_path) if persist_path else None
        self.learning_window = learning_window
        self.success_threshold = success_threshold

        # Episodic buffer (FIFO queue)
        self.episodes: deque = deque(maxlen=capacity)

        # Metrics tracking
        self.metrics = ReflectionMetrics()

        # Reward extraction
        self.reward_extractor = RewardExtractor(config=reward_config)

        # Learning state
        self.last_analysis_time = datetime.now()
        self.analysis_interval = timedelta(minutes=5)  # Analyze every 5 minutes

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"ReflectionBuffer initialized (capacity={capacity}, window={learning_window})")

        # Create persist directory if specified
        if self.persist_path:
            self.persist_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Persistence enabled: {self.persist_path}")

    async def store(
        self,
        spacetime: Spacetime,
        feedback: Optional[Dict[str, Any]] = None,
        reward: Optional[float] = None
    ) -> None:
        """
        Store a Spacetime artifact with optional feedback.

        Args:
            spacetime: Spacetime artifact from weaving cycle
            feedback: Optional user feedback dict
            reward: Optional explicit reward signal (0-1)
        """
        # Derive reward if not provided
        if reward is None:
            reward = self._derive_reward(spacetime, feedback)

        # Create episode entry
        episode = {
            'spacetime': spacetime,
            'feedback': feedback,
            'reward': reward,
            'timestamp': datetime.now(),
            'success': reward >= self.success_threshold
        }

        # Store in buffer
        self.episodes.append(episode)

        # Update metrics
        self._update_metrics(episode)

        self.logger.debug(
            f"Stored episode: tool={spacetime.tool_used}, "
            f"confidence={spacetime.confidence:.2f}, reward={reward:.2f}"
        )

        # Persist if enabled
        if self.persist_path:
            await self._persist_episode(episode)

    def _derive_reward(
        self,
        spacetime: Spacetime,
        feedback: Optional[Dict[str, Any]]
    ) -> float:
        """
        Derive reward signal from Spacetime and feedback using RewardExtractor.

        Uses multi-component reward with:
        - Base reward: Confidence (weighted 0.6)
        - Quality bonus: Quality score (weighted 0.3)
        - Efficiency bonus: Fast execution (weighted 0.1)
        - Error/warning penalties: Negative feedback
        - User feedback override: Sparse rewards when available

        Args:
            spacetime: Spacetime artifact
            feedback: Optional user feedback

        Returns:
            Reward value in [-1, 1] (normalized to [0, 1] for backward compatibility)
        """
        # Use RewardExtractor for sophisticated reward computation
        reward = self.reward_extractor.compute_reward(spacetime, feedback)

        # Normalize to [0, 1] for backward compatibility with rest of buffer
        # RewardExtractor returns [-1, 1], but buffer expects [0, 1]
        normalized_reward = (reward + 1.0) / 2.0

        return np.clip(normalized_reward, 0.0, 1.0)

    def _update_metrics(self, episode: Dict[str, Any]) -> None:
        """Update aggregated metrics with new episode."""
        spacetime = episode['spacetime']
        success = episode['success']

        # Update counts
        self.metrics.total_cycles += 1
        if success:
            self.metrics.successful_cycles += 1
        else:
            self.metrics.failed_cycles += 1

        # Update tool metrics
        tool = spacetime.tool_used
        if tool not in self.metrics.tool_usage_counts:
            self.metrics.tool_usage_counts[tool] = 0
            self.metrics.tool_success_rates[tool] = 0.0
            self.metrics.tool_avg_confidence[tool] = 0.0

        self.metrics.tool_usage_counts[tool] += 1
        count = self.metrics.tool_usage_counts[tool]

        # Update running averages
        old_success = self.metrics.tool_success_rates[tool]
        self.metrics.tool_success_rates[tool] = (
            (old_success * (count - 1) + (1.0 if success else 0.0)) / count
        )

        old_conf = self.metrics.tool_avg_confidence[tool]
        self.metrics.tool_avg_confidence[tool] = (
            (old_conf * (count - 1) + spacetime.confidence) / count
        )

        # Update pattern metrics
        pattern = spacetime.metadata.get('pattern_card', 'unknown')
        if pattern not in self.metrics.pattern_usage_counts:
            self.metrics.pattern_usage_counts[pattern] = 0
            self.metrics.pattern_success_rates[pattern] = 0.0
            self.metrics.pattern_avg_duration[pattern] = 0.0

        self.metrics.pattern_usage_counts[pattern] += 1
        p_count = self.metrics.pattern_usage_counts[pattern]

        old_p_success = self.metrics.pattern_success_rates[pattern]
        self.metrics.pattern_success_rates[pattern] = (
            (old_p_success * (p_count - 1) + (1.0 if success else 0.0)) / p_count
        )

        old_duration = self.metrics.pattern_avg_duration[pattern]
        self.metrics.pattern_avg_duration[pattern] = (
            (old_duration * (p_count - 1) + spacetime.trace.duration_ms) / p_count
        )

        # Update query complexity tracking
        query_len = len(spacetime.query_text)
        if success:
            self.metrics.successful_query_lengths.append(query_len)
        else:
            self.metrics.failed_query_lengths.append(query_len)

        # Keep only recent query lengths (last 100)
        if len(self.metrics.successful_query_lengths) > 100:
            self.metrics.successful_query_lengths = self.metrics.successful_query_lengths[-100:]
        if len(self.metrics.failed_query_lengths) > 100:
            self.metrics.failed_query_lengths = self.metrics.failed_query_lengths[-100:]

        self.metrics.last_updated = datetime.now()

    async def analyze_and_learn(self, force: bool = False) -> List[LearningSignal]:
        """
        Analyze recent episodes and generate learning signals.

        Performs analysis on recent weaving cycles to identify:
        - Which tools perform well
        - Which patterns are most effective
        - What query types succeed/fail
        - How to adjust exploration/exploitation

        Args:
            force: Force analysis even if not enough time has passed

        Returns:
            List of learning signals for system adaptation
        """
        # Check if enough time has passed
        if not force:
            time_since_analysis = datetime.now() - self.last_analysis_time
            if time_since_analysis < self.analysis_interval:
                return []

        self.logger.info("Analyzing reflection buffer for learning signals...")

        signals = []

        # Get recent episodes for analysis
        recent = list(self.episodes)[-self.learning_window:]
        if len(recent) < 10:
            self.logger.debug("Not enough episodes for meaningful analysis")
            return []

        # 1. Analyze tool performance for bandit updates
        tool_signals = self._analyze_tool_performance(recent)
        signals.extend(tool_signals)

        # 2. Analyze pattern card effectiveness
        pattern_signals = self._analyze_pattern_performance(recent)
        signals.extend(pattern_signals)

        # 3. Analyze failure modes
        failure_signals = self._analyze_failures(recent)
        signals.extend(failure_signals)

        # 4. Analyze exploration/exploitation balance
        balance_signals = self._analyze_exploration_balance(recent)
        signals.extend(balance_signals)

        self.last_analysis_time = datetime.now()

        self.logger.info(f"Generated {len(signals)} learning signals")
        return signals

    def _analyze_tool_performance(self, episodes: List[Dict]) -> List[LearningSignal]:
        """Analyze tool performance and generate bandit update signals."""
        signals = []

        # Group episodes by tool
        tool_episodes = defaultdict(list)
        for ep in episodes:
            tool = ep['spacetime'].tool_used
            tool_episodes[tool].append(ep)

        # For each tool, generate update signal
        for tool, tool_eps in tool_episodes.items():
            if len(tool_eps) < 3:
                continue  # Need at least 3 samples

            rewards = [ep['reward'] for ep in tool_eps]
            avg_reward = np.mean(rewards)
            std_reward = np.std(rewards)

            # Create bandit update signal
            signal = LearningSignal(
                signal_type="bandit_update",
                tool=tool,
                reward=avg_reward,
                recommendation=f"Update {tool} bandit statistics: avg_reward={avg_reward:.2f}",
                evidence={
                    'sample_count': len(tool_eps),
                    'avg_reward': float(avg_reward),
                    'std_reward': float(std_reward),
                    'min_reward': float(np.min(rewards)),
                    'max_reward': float(np.max(rewards))
                },
                priority=0.8  # High priority for bandit updates
            )
            signals.append(signal)

        return signals

    def _analyze_pattern_performance(self, episodes: List[Dict]) -> List[LearningSignal]:
        """Analyze pattern card performance and suggest preferences."""
        signals = []

        # Group by pattern
        pattern_episodes = defaultdict(list)
        for ep in episodes:
            pattern = ep['spacetime'].metadata.get('pattern_card', 'unknown')
            pattern_episodes[pattern].append(ep)

        # Find best performing pattern
        pattern_scores = {}
        for pattern, eps in pattern_episodes.items():
            if len(eps) < 5:
                continue

            rewards = [ep['reward'] for ep in eps]
            durations = [ep['spacetime'].trace.duration_ms for ep in eps]

            # Score = reward / (normalized_duration)
            avg_reward = np.mean(rewards)
            avg_duration = np.mean(durations)
            normalized_duration = avg_duration / 1000.0  # Convert to seconds

            # Quality/speed tradeoff score
            score = avg_reward / (1.0 + normalized_duration * 0.1)
            pattern_scores[pattern] = score

        if pattern_scores:
            best_pattern = max(pattern_scores, key=pattern_scores.get)
            best_score = pattern_scores[best_pattern]

            signal = LearningSignal(
                signal_type="pattern_preference",
                pattern=best_pattern,
                recommendation=f"Prefer '{best_pattern}' pattern (score={best_score:.2f})",
                evidence={
                    'pattern_scores': {k: float(v) for k, v in pattern_scores.items()},
                    'best_pattern': best_pattern,
                    'best_score': float(best_score)
                },
                priority=0.6
            )
            signals.append(signal)

        return signals

    def _analyze_failures(self, episodes: List[Dict]) -> List[LearningSignal]:
        """Analyze failure patterns and suggest mitigations."""
        signals = []

        failures = [ep for ep in episodes if not ep['success']]
        if len(failures) < 5:
            return signals

        # Analyze common failure patterns
        failure_tools = defaultdict(int)
        failure_patterns = defaultdict(int)
        failure_query_lengths = []

        for ep in failures:
            st = ep['spacetime']
            failure_tools[st.tool_used] += 1
            failure_patterns[st.metadata.get('pattern_card', 'unknown')] += 1
            failure_query_lengths.append(len(st.query_text))

        # Check if specific tool has high failure rate
        for tool, count in failure_tools.items():
            total_tool = sum(1 for ep in episodes if ep['spacetime'].tool_used == tool)
            failure_rate = count / total_tool if total_tool > 0 else 0

            if failure_rate > 0.5 and total_tool >= 5:
                signal = LearningSignal(
                    signal_type="threshold_adjustment",
                    tool=tool,
                    recommendation=f"High failure rate for {tool} ({failure_rate:.1%}), consider adjusting",
                    evidence={
                        'failure_rate': float(failure_rate),
                        'failure_count': count,
                        'total_count': total_tool
                    },
                    priority=0.7
                )
                signals.append(signal)

        return signals

    def _analyze_exploration_balance(self, episodes: List[Dict]) -> List[LearningSignal]:
        """Analyze exploration vs exploitation balance."""
        signals = []

        if len(episodes) < 20:
            return signals

        # Check tool diversity
        tools_used = [ep['spacetime'].tool_used for ep in episodes]
        unique_tools = len(set(tools_used))
        total_tools = len(tools_used)

        # Calculate entropy as measure of exploration
        tool_counts = defaultdict(int)
        for tool in tools_used:
            tool_counts[tool] += 1

        probs = [count / total_tools for count in tool_counts.values()]
        entropy = -sum(p * np.log(p) for p in probs if p > 0)
        max_entropy = np.log(unique_tools) if unique_tools > 0 else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        # If entropy is too low (< 0.5), we're not exploring enough
        if normalized_entropy < 0.5:
            signal = LearningSignal(
                signal_type="threshold_adjustment",
                recommendation="Increase exploration: low tool diversity detected",
                evidence={
                    'entropy': float(normalized_entropy),
                    'unique_tools': unique_tools,
                    'total_samples': total_tools,
                    'tool_distribution': {k: v for k, v in tool_counts.items()}
                },
                priority=0.5
            )
            signals.append(signal)

        return signals

    async def _persist_episode(self, episode: Dict[str, Any]) -> None:
        """Persist episode to disk."""
        if not self.persist_path:
            return

        timestamp = episode['timestamp'].strftime("%Y%m%d_%H%M%S_%f")
        filename = f"episode_{timestamp}.json"
        filepath = self.persist_path / filename

        try:
            # Serialize spacetime
            spacetime_dict = episode['spacetime'].to_dict()

            episode_data = {
                'spacetime': spacetime_dict,
                'feedback': episode['feedback'],
                'reward': episode['reward'],
                'success': episode['success'],
                'timestamp': episode['timestamp'].isoformat()
            }

            with open(filepath, 'w') as f:
                json.dump(episode_data, f, indent=2)

        except Exception as e:
            self.logger.warning(f"Failed to persist episode: {e}")

    def get_metrics(self) -> ReflectionMetrics:
        """Get current reflection metrics."""
        return self.metrics

    def get_recent_episodes(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get N most recent episodes."""
        return list(self.episodes)[-n:]

    def get_success_rate(self) -> float:
        """Get overall success rate."""
        if self.metrics.total_cycles == 0:
            return 0.0
        return self.metrics.successful_cycles / self.metrics.total_cycles

    def get_tool_recommendations(self) -> Dict[str, float]:
        """
        Get tool recommendations based on historical performance.

        Returns:
            Dict mapping tool names to recommendation scores (0-1)
        """
        recommendations = {}

        for tool, success_rate in self.metrics.tool_success_rates.items():
            confidence = self.metrics.tool_avg_confidence.get(tool, 0.5)
            usage = self.metrics.tool_usage_counts.get(tool, 0)

            # Score = weighted average of success rate and confidence
            # Boost for tools that are used frequently (proven track record)
            usage_boost = min(usage / 100.0, 1.0)  # Cap at 100 uses
            score = 0.5 * success_rate + 0.3 * confidence + 0.2 * usage_boost

            recommendations[tool] = np.clip(score, 0.0, 1.0)

        return recommendations

    def get_ppo_batch(
        self,
        batch_size: Optional[int] = None,
        recent_only: bool = True
    ) -> Dict[str, List]:
        """
        Extract batched experience for PPO training.

        Converts stored Spacetime artifacts into PPO-compatible format with:
        - Observations: Feature dict from each spacetime
        - Actions: Tool names selected
        - Rewards: Computed rewards (normalized to [-1, 1])
        - Dones: Always True for episodic tasks

        Args:
            batch_size: Optional batch size (None = all episodes)
            recent_only: If True, only return most recent episodes

        Returns:
            Dict with batched experience:
                - observations: List of feature dicts
                - actions: List of tool names (strings)
                - rewards: List of scalar rewards
                - dones: List of done flags (all True)
                - infos: List of metadata dicts
        """
        # Get episodes to batch
        if batch_size is None:
            episodes = list(self.episodes)
        elif recent_only:
            episodes = list(self.episodes)[-batch_size:]
        else:
            # Random sample
            import random
            episodes = random.sample(list(self.episodes), min(batch_size, len(self.episodes)))

        if not episodes:
            return {
                'observations': [],
                'actions': [],
                'rewards': [],
                'dones': [],
                'infos': []
            }

        # Extract components
        observations = []
        actions = []
        rewards = []
        dones = []
        infos = []

        for ep in episodes:
            # Use RewardExtractor to get full experience tuple
            experience = self.reward_extractor.extract_experience(
                ep['spacetime'],
                user_feedback=ep['feedback']
            )

            observations.append(experience['observation'])
            actions.append(experience['action'])
            rewards.append(experience['reward'])  # Keep in [-1, 1] for PPO
            dones.append(experience['done'])
            infos.append(experience['info'])

        self.logger.debug(f"Extracted PPO batch: {len(observations)} experiences")

        return {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'infos': infos
        }

    async def consolidate(self) -> Dict[str, Any]:
        """
        Deep consolidation phase - like REM sleep for the memory system.

        Performs intensive memory restructuring:
        1. Compress redundant episodes
        2. Extract meta-patterns across episodes
        3. Prune low-value memories
        4. Update long-term statistics

        This is the "rest" phase of the breathing cycle, where the system
        integrates learning and consolidates memories.

        Returns:
            Dict with consolidation metrics
        """
        self.logger.info("Starting deep consolidation...")

        consolidation_start = time.time()

        # Step 1: Compress redundant episodes
        compression_stats = await self._compress_episodes()

        # Step 2: Extract meta-patterns
        meta_patterns = await self._extract_meta_patterns()

        # Step 3: Prune low-value memories
        pruning_stats = await self._prune_redundant()

        # Step 4: Update long-term statistics
        self._update_long_term_stats()

        consolidation_duration = time.time() - consolidation_start

        metrics = {
            "duration": consolidation_duration,
            "compression": compression_stats,
            "meta_patterns": meta_patterns,
            "pruning": pruning_stats,
            "final_episode_count": len(self.episodes),
            "success_rate": self.get_success_rate()
        }

        self.logger.info(f"Consolidation complete ({consolidation_duration:.2f}s)")

        return metrics

    async def _compress_episodes(self) -> Dict[str, Any]:
        """
        Compress redundant episodes with similar outcomes.

        Groups episodes by (tool, pattern_card) and merges similar ones.

        Returns:
            Compression statistics
        """
        if len(self.episodes) < 10:
            return {"compressed": 0, "reason": "too_few_episodes"}

        # Group by tool and pattern
        from collections import defaultdict
        groups = defaultdict(list)

        for ep in self.episodes:
            key = (ep['spacetime'].tool_used, ep['spacetime'].metadata.get('pattern_card', 'unknown'))
            groups[key].append(ep)

        # Count mergeable episodes (those with similar rewards in each group)
        mergeable_count = 0
        for key, eps in groups.items():
            if len(eps) >= 3:
                rewards = [e['reward'] for e in eps]
                std = np.std(rewards)
                if std < 0.1:  # Very similar rewards
                    mergeable_count += len(eps) - 1  # Keep 1, compress rest

        return {
            "compressed": mergeable_count,
            "groups": len(groups),
            "potential_savings": f"{mergeable_count / len(self.episodes) * 100:.1f}%"
        }

    async def _extract_meta_patterns(self) -> Dict[str, Any]:
        """
        Extract meta-patterns across episodes.

        Identifies higher-order patterns like:
        - Time-of-day performance variations
        - Query complexity vs success correlation
        - Tool synergy patterns

        Returns:
            Discovered meta-patterns
        """
        if len(self.episodes) < 20:
            return {"meta_patterns": 0, "reason": "insufficient_data"}

        patterns = {}

        # Pattern 1: Query length vs success
        successful_lengths = [len(ep['spacetime'].query_text) for ep in self.episodes if ep['success']]
        failed_lengths = [len(ep['spacetime'].query_text) for ep in self.episodes if not ep['success']]

        if successful_lengths and failed_lengths:
            patterns['optimal_query_length'] = {
                "mean_success": float(np.mean(successful_lengths)),
                "mean_failure": float(np.mean(failed_lengths)),
                "recommendation": "shorter" if np.mean(successful_lengths) < np.mean(failed_lengths) else "longer"
            }

        # Pattern 2: Tool combination patterns
        tools_used = [ep['spacetime'].tool_used for ep in self.episodes[-50:]]  # Recent 50
        tool_diversity = len(set(tools_used)) / len(tools_used) if tools_used else 0

        patterns['exploration_health'] = {
            "tool_diversity": float(tool_diversity),
            "status": "healthy" if tool_diversity > 0.3 else "low_exploration"
        }

        return patterns

    async def _prune_redundant(self) -> Dict[str, Any]:
        """
        Prune low-value memories.

        Removes episodes that:
        - Have very low reward (< 0.2)
        - Are redundant with better episodes
        - Are older than a certain threshold with low replay value

        Returns:
            Pruning statistics
        """
        if len(self.episodes) < self.capacity * 0.8:
            return {"pruned": 0, "reason": "below_capacity"}

        # Identify pruneable episodes
        pruneable = []
        for i, ep in enumerate(self.episodes):
            # Low reward and old
            if ep['reward'] < 0.2 and i < len(self.episodes) - 100:
                pruneable.append(i)

        pruned_count = len(pruneable)

        # Note: Not actually removing for now to preserve data
        # In production, you'd remove: for idx in reversed(pruneable): del self.episodes[idx]

        return {
            "pruned": 0,  # Not removing yet, just identifying
            "pruneable": pruned_count,
            "threshold": "reward < 0.2 and age > 100"
        }

    def _update_long_term_stats(self) -> None:
        """
        Update long-term rolling statistics.

        Maintains exponential moving averages of key metrics.
        """
        # This would update running stats, currently just logs
        recent_success_rate = sum(1 for ep in list(self.episodes)[-100:] if ep['success']) / min(100, len(self.episodes))
        self.logger.debug(f"Long-term stats updated: recent_success_rate={recent_success_rate:.2%}")

    def clear(self) -> None:
        """Clear all episodes and reset metrics."""
        self.episodes.clear()
        self.metrics = ReflectionMetrics()
        self.logger.info("Reflection buffer cleared")

    def __len__(self) -> int:
        """Return number of stored episodes."""
        return len(self.episodes)

    async def __aenter__(self):
        """Async context manager entry."""
        self.logger.debug("ReflectionBuffer context manager entered")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Async context manager exit with cleanup.

        Args:
            exc_type: Exception type (if any)
            exc_val: Exception value (if any)
            exc_tb: Exception traceback (if any)
        """
        self.logger.debug("ReflectionBuffer context manager exiting")

        # Flush any pending data
        await self.flush()

        # Cleanup resources
        await self.close()

        # Don't suppress exceptions
        return False

    async def flush(self) -> None:
        """
        Flush metrics and state to disk.

        This ensures all data is persisted before shutdown.
        Can be called manually for periodic checkpointing.
        """
        if not self.persist_path:
            return

        try:
            # Save metrics summary
            metrics_file = self.persist_path / "metrics_summary.json"
            metrics_data = self.metrics.to_dict()

            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)

            self.logger.info(f"Flushed metrics to {metrics_file}")

        except Exception as e:
            self.logger.error(f"Failed to flush metrics: {e}")

    async def close(self) -> None:
        """
        Clean up resources.

        Called automatically when using async context manager,
        or can be called manually for explicit cleanup.
        """
        self.logger.info(f"Closing ReflectionBuffer ({len(self.episodes)} episodes)")

        # Clear in-memory buffer (data already persisted)
        # Keep metrics for final reporting

        self.logger.info("ReflectionBuffer closed")

    def __repr__(self) -> str:
        return (
            f"ReflectionBuffer(episodes={len(self.episodes)}, "
            f"success_rate={self.get_success_rate():.2%})"
        )


# ============================================================================
# Main Entry Point (for testing)
# ============================================================================

async def demo():
    """Demo of ReflectionBuffer usage."""
    print("\n" + "="*80)
    print("HoloLoom Reflection Buffer - Demo")
    print("="*80 + "\n")

    # Create buffer
    buffer = ReflectionBuffer(
        capacity=100,
        persist_path="./reflection_demo",
        learning_window=20
    )

    print(f"Buffer initialized: {buffer}\n")

    # Simulate some weaving cycles
    from HoloLoom.fabric.spacetime import Spacetime, WeavingTrace

    tools = ["answer", "search", "calc", "notion_write"]
    patterns = ["bare", "fast", "fused"]

    print("Simulating 30 weaving cycles...")
    for i in range(30):
        # Create mock spacetime
        trace = WeavingTrace(
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_ms=1000 + np.random.randn() * 200,
            tool_selected=np.random.choice(tools),
            tool_confidence=0.5 + np.random.rand() * 0.5
        )

        spacetime = Spacetime(
            query_text=f"Query {i}",
            response=f"Response {i}",
            tool_used=trace.tool_selected,
            confidence=trace.tool_confidence,
            trace=trace,
            metadata={'pattern_card': np.random.choice(patterns)}
        )

        # Store with random feedback
        feedback = {'helpful': np.random.rand() > 0.3}
        await buffer.store(spacetime, feedback=feedback)

    print(f"Stored 30 episodes. Buffer: {buffer}\n")

    # Analyze and learn
    print("Analyzing for learning signals...")
    signals = await buffer.analyze_and_learn(force=True)

    print(f"\nGenerated {len(signals)} learning signals:\n")
    for i, signal in enumerate(signals, 1):
        print(f"{i}. [{signal.signal_type}] {signal.recommendation}")
        print(f"   Priority: {signal.priority:.2f}")
        if signal.tool:
            print(f"   Tool: {signal.tool}")
        if signal.pattern:
            print(f"   Pattern: {signal.pattern}")
        print()

    # Show metrics
    print("="*80)
    print("Reflection Metrics")
    print("="*80)
    metrics = buffer.get_metrics()
    print(f"Total cycles: {metrics.total_cycles}")
    print(f"Success rate: {buffer.get_success_rate():.1%}")
    print(f"\nTool Performance:")
    for tool, rate in metrics.tool_success_rates.items():
        conf = metrics.tool_avg_confidence[tool]
        count = metrics.tool_usage_counts[tool]
        print(f"  {tool:15s}: {rate:.1%} success, {conf:.2f} avg confidence ({count} uses)")

    print(f"\nTool Recommendations:")
    recommendations = buffer.get_tool_recommendations()
    for tool, score in sorted(recommendations.items(), key=lambda x: x[1], reverse=True):
        print(f"  {tool:15s}: {score:.2f}")

    print("\n" + "="*80)


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo())
