"""
MCTS Breakthrough Detection & Feed-Forward

Detects when MCTS finds breakthrough paths and feeds them forward to:
1. Parallel simulations (bias toward breakthrough regions)
2. Selection phase (boost UCT scores near breakthroughs)
3. Pattern learner (prioritize breakthrough patterns)
4. Future searches (long-term breakthrough memory)

Key Innovation: Don't wait for backpropagation - accelerate discovery
in real-time when breakthroughs occur.

Breakthrough Criteria:
- Reward improvement > 2σ above baseline
- Confidence jump > 0.2 in single step
- Discovers previously unexplored high-value region
- Pattern generalizes across multiple queries

Feed-Forward Mechanisms:
- UCT bias injection (+0.5 to breakthrough paths)
- Parallel simulation broadcasting
- Selection phase priority boost
- Pattern learning acceleration
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ============================================================================
# Breakthrough Detection
# ============================================================================

@dataclass
class Breakthrough:
    """A discovered breakthrough in MCTS search"""
    # Identity
    id: str  # Unique ID
    timestamp: float
    search_id: str  # Which MCTS search found it

    # Path information
    action_sequence: list[Any]  # Actions leading to breakthrough
    state_signature: str  # Hash of state that led to breakthrough

    # Metrics
    reward: float  # Reward value
    reward_improvement: float  # Improvement over baseline
    confidence_jump: float  # Confidence increase in single step
    visits: int  # How many times this path was visited

    # Impact
    impact_score: float  # How impactful (0-1)
    generalization_score: float  # How well it generalizes (0-1)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def decay_impact(self, decay_rate: float = 0.95):
        """Decay impact over time"""
        self.impact_score *= decay_rate


@dataclass
class BreakthroughStats:
    """Statistics about breakthrough detection"""
    total_detected: int = 0
    by_search: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    avg_reward_improvement: float = 0.0
    avg_impact_score: float = 0.0
    false_positives: int = 0  # Breakthroughs that didn't generalize


class BreakthroughDetector:
    """
    Detects breakthroughs in MCTS searches.

    Monitors:
    - Reward improvements
    - Confidence jumps
    - Novel high-value regions
    - Pattern generalization

    Triggers feed-forward when breakthrough detected.
    """

    def __init__(
        self,
        reward_improvement_threshold: float = 2.0,  # 2σ above baseline
        confidence_jump_threshold: float = 0.2,     # 0.2 confidence increase
        min_visits_threshold: int = 3,              # Must be visited 3+ times
        impact_decay_rate: float = 0.95,           # Decay per search
        max_breakthrough_memory: int = 100          # Max breakthroughs to remember
    ):
        self.reward_improvement_threshold = reward_improvement_threshold
        self.confidence_jump_threshold = confidence_jump_threshold
        self.min_visits_threshold = min_visits_threshold
        self.impact_decay_rate = impact_decay_rate
        self.max_breakthrough_memory = max_breakthrough_memory

        # Short-term memory (current search)
        self.current_breakthroughs: list[Breakthrough] = []

        # Long-term memory (across searches)
        self.breakthrough_memory: deque = deque(maxlen=max_breakthrough_memory)

        # Statistics
        self.stats = BreakthroughStats()

        # Baseline tracking
        self.reward_baseline: float = 0.0
        self.reward_std: float = 0.1
        self.reward_history: list[float] = []

    def update_baseline(self, reward: float):
        """Update reward baseline statistics"""
        self.reward_history.append(reward)

        # Keep last 100 rewards
        if len(self.reward_history) > 100:
            self.reward_history.pop(0)

        # Update baseline
        self.reward_baseline = np.mean(self.reward_history)
        self.reward_std = np.std(self.reward_history) if len(self.reward_history) > 1 else 0.1

    def detect_breakthrough(
        self,
        reward: float,
        previous_reward: float,
        confidence: float,
        previous_confidence: float,
        action_sequence: list[Any],
        state_signature: str,
        visits: int,
        search_id: str
    ) -> Breakthrough | None:
        """
        Detect if current path is a breakthrough.

        Returns Breakthrough if detected, None otherwise.
        """
        # Check minimum visits
        if visits < self.min_visits_threshold:
            return None

        # Calculate improvements
        reward_improvement = reward - previous_reward
        confidence_jump = confidence - previous_confidence

        # Calculate z-score for reward improvement
        if self.reward_std > 0:
            reward_z_score = (reward - self.reward_baseline) / self.reward_std
        else:
            reward_z_score = 0.0

        # Breakthrough criteria
        is_breakthrough = False
        breakthrough_reasons = []

        # Criterion 1: Significant reward improvement
        if reward_z_score > self.reward_improvement_threshold:
            is_breakthrough = True
            breakthrough_reasons.append(f"reward_z={reward_z_score:.2f}")

        # Criterion 2: Large confidence jump
        if confidence_jump > self.confidence_jump_threshold:
            is_breakthrough = True
            breakthrough_reasons.append(f"confidence_jump={confidence_jump:.2f}")

        # Criterion 3: High absolute reward
        if reward > 0.9 and previous_reward < 0.7:
            is_breakthrough = True
            breakthrough_reasons.append(f"high_reward={reward:.2f}")

        if not is_breakthrough:
            return None

        # Calculate impact score
        impact_score = min(1.0, (
            0.4 * (reward_z_score / 3.0) +
            0.4 * (confidence_jump / 0.3) +
            0.2 * (visits / 10.0)
        ))

        # Create breakthrough
        breakthrough = Breakthrough(
            id=f"bt_{search_id}_{len(self.current_breakthroughs)}",
            timestamp=time.time(),
            search_id=search_id,
            action_sequence=list(action_sequence),
            state_signature=state_signature,
            reward=reward,
            reward_improvement=reward_improvement,
            confidence_jump=confidence_jump,
            visits=visits,
            impact_score=impact_score,
            generalization_score=0.0,  # Computed later
            metadata={
                'reasons': breakthrough_reasons,
                'reward_z_score': reward_z_score,
                'baseline': self.reward_baseline
            }
        )

        # Add to current breakthroughs
        self.current_breakthroughs.append(breakthrough)

        # Update statistics
        self.stats.total_detected += 1
        self.stats.by_search[search_id] += 1

        return breakthrough

    def commit_breakthroughs(self, search_id: str):
        """
        Commit current breakthroughs to long-term memory.
        Called at end of search.
        """
        # Calculate generalization scores
        for bt in self.current_breakthroughs:
            # Check if similar breakthroughs exist
            similar_count = sum(
                1 for past_bt in self.breakthrough_memory
                if self._compute_similarity(bt, past_bt) > 0.7
            )

            bt.generalization_score = min(1.0, similar_count / 5.0)

        # Add to long-term memory
        self.breakthrough_memory.extend(self.current_breakthroughs)

        # Update statistics
        if self.current_breakthroughs:
            avg_reward = np.mean([bt.reward_improvement for bt in self.current_breakthroughs])
            avg_impact = np.mean([bt.impact_score for bt in self.current_breakthroughs])

            # Moving average
            alpha = 0.3
            self.stats.avg_reward_improvement = (
                alpha * avg_reward +
                (1 - alpha) * self.stats.avg_reward_improvement
            )
            self.stats.avg_impact_score = (
                alpha * avg_impact +
                (1 - alpha) * self.stats.avg_impact_score
            )

        # Clear current breakthroughs
        self.current_breakthroughs = []

    def get_breakthrough_bias(self, state_signature: str) -> float:
        """
        Get UCT bias for a state based on breakthrough memory.

        Returns boost value (0.0 - 1.0) to add to UCT score.
        """
        total_bias = 0.0

        # Check current breakthroughs
        for bt in self.current_breakthroughs:
            if bt.state_signature == state_signature:
                total_bias += bt.impact_score * 0.5  # Strong bias

        # Check long-term memory (with decay)
        age_weight = 1.0
        for bt in reversed(list(self.breakthrough_memory)):
            if bt.state_signature == state_signature:
                total_bias += bt.impact_score * age_weight * 0.2  # Weaker bias
            age_weight *= 0.99  # Decay

        return min(1.0, total_bias)

    def get_top_breakthroughs(self, n: int = 10) -> list[Breakthrough]:
        """Get top N breakthroughs by impact score"""
        all_breakthroughs = list(self.breakthrough_memory) + self.current_breakthroughs
        sorted_breakthroughs = sorted(
            all_breakthroughs,
            key=lambda bt: bt.impact_score * bt.generalization_score,
            reverse=True
        )
        return sorted_breakthroughs[:n]

    def decay_all_impacts(self):
        """Decay all breakthrough impacts (called periodically)"""
        for bt in self.breakthrough_memory:
            bt.decay_impact(self.impact_decay_rate)

    def _compute_similarity(self, bt1: Breakthrough, bt2: Breakthrough) -> float:
        """Compute similarity between two breakthroughs (0-1)"""
        # Simple signature matching for now
        if bt1.state_signature == bt2.state_signature:
            return 1.0

        # Could add more sophisticated similarity metrics
        return 0.0

    def get_stats(self) -> dict[str, Any]:
        """Get breakthrough detection statistics"""
        return {
            'total_detected': self.stats.total_detected,
            'by_search': dict(self.stats.by_search),
            'avg_reward_improvement': self.stats.avg_reward_improvement,
            'avg_impact_score': self.stats.avg_impact_score,
            'false_positives': self.stats.false_positives,
            'memory_size': len(self.breakthrough_memory),
            'current_breakthroughs': len(self.current_breakthroughs)
        }


# ============================================================================
# Feed-Forward Propagation
# ============================================================================

class FeedForwardBroadcaster:
    """
    Broadcasts breakthroughs to parallel MCTS simulations.

    When a breakthrough is detected:
    1. Immediately notify all parallel simulations
    2. Bias their selection toward breakthrough regions
    3. Prioritize breakthrough patterns for learning
    """

    def __init__(self):
        self.listeners: list[Any] = []  # MCTS engines listening
        self.broadcast_count = 0
        self.impact_stats: dict[str, float] = defaultdict(float)

    def register_listener(self, mcts_engine: Any):
        """Register MCTS engine to receive breakthrough broadcasts"""
        self.listeners.append(mcts_engine)

    def unregister_listener(self, mcts_engine: Any):
        """Unregister MCTS engine"""
        if mcts_engine in self.listeners:
            self.listeners.remove(mcts_engine)

    def broadcast_breakthrough(self, breakthrough: Breakthrough):
        """
        Broadcast breakthrough to all listening MCTS engines.

        Engines will:
        - Bias their UCT scores toward breakthrough region
        - Prioritize exploration near breakthrough
        - Update their selection strategy
        """
        self.broadcast_count += 1

        for listener in self.listeners:
            if hasattr(listener, 'receive_breakthrough'):
                listener.receive_breakthrough(breakthrough)

        # Track impact
        self.impact_stats[breakthrough.search_id] += breakthrough.impact_score

    def get_stats(self) -> dict[str, Any]:
        """Get broadcast statistics"""
        return {
            'listeners': len(self.listeners),
            'broadcasts': self.broadcast_count,
            'impact_by_search': dict(self.impact_stats),
            'avg_impact_per_broadcast': (
                sum(self.impact_stats.values()) / self.broadcast_count
                if self.broadcast_count > 0 else 0.0
            )
        }


# ============================================================================
# Breakthrough-Enhanced MCTS
# ============================================================================

class BreakthroughMCTS:
    """
    MCTS engine enhanced with breakthrough detection and feed-forward.

    Key features:
    - Real-time breakthrough detection during search
    - Immediate UCT bias injection when breakthroughs found
    - Broadcasting to parallel searches
    - Long-term breakthrough memory
    """

    def __init__(
        self,
        base_mcts_engine: Any,
        detector: BreakthroughDetector,
        broadcaster: FeedForwardBroadcaster,
        enable_feed_forward: bool = True
    ):
        self.mcts = base_mcts_engine
        self.detector = detector
        self.broadcaster = broadcaster
        self.enable_feed_forward = enable_feed_forward

        # Register for broadcasts
        if enable_feed_forward:
            self.broadcaster.register_listener(self)

        # Tracking
        self.search_id = f"search_{id(self)}"
        self.received_breakthroughs: list[Breakthrough] = []

    def search_with_breakthroughs(
        self,
        initial_state: Any,
        num_simulations: int
    ) -> Any:
        """
        Run MCTS search with breakthrough detection.

        Monitors every iteration for breakthroughs and feeds forward.
        """
        # Start search
        root = self.mcts.create_root(initial_state)

        for i in range(num_simulations):
            # Run one MCTS iteration
            node = self.mcts.select(root)
            reward = self.mcts.simulate(node.state)
            self.mcts.backpropagate(node, reward)

            # Check for breakthrough
            if i > 0:  # Need previous iteration for comparison
                breakthrough = self.detector.detect_breakthrough(
                    reward=reward,
                    previous_reward=self.mcts.previous_reward,
                    confidence=node.value(),
                    previous_confidence=self.mcts.previous_confidence,
                    action_sequence=self.mcts.get_action_sequence(node),
                    state_signature=self.mcts.get_state_signature(node.state),
                    visits=node.visits,
                    search_id=self.search_id
                )

                if breakthrough:
                    # Feed forward immediately
                    if self.enable_feed_forward:
                        self.broadcaster.broadcast_breakthrough(breakthrough)

                    # Update baseline
                    self.detector.update_baseline(reward)

            # Store for next iteration
            self.mcts.previous_reward = reward
            self.mcts.previous_confidence = node.value()

        # Commit breakthroughs to memory
        self.detector.commit_breakthroughs(self.search_id)

        # Get best action with breakthrough bias
        return self.select_best_action_with_bias(root)

    def select_best_action_with_bias(self, root: Any) -> Any:
        """
        Select best action, incorporating breakthrough bias.

        Actions near breakthroughs get boosted scores.
        """
        best_action = None
        best_score = -float('inf')

        for action, child in root.children.items():
            # Base score (visits and value)
            base_score = child.value() + child.visits / 10.0

            # Breakthrough bias
            state_signature = self.mcts.get_state_signature(child.state)
            bias = self.detector.get_breakthrough_bias(state_signature)

            # Combined score
            total_score = base_score + bias

            if total_score > best_score:
                best_score = total_score
                best_action = action

        return best_action

    def receive_breakthrough(self, breakthrough: Breakthrough):
        """
        Receive breakthrough from another parallel search.

        Immediately bias this search toward breakthrough region.
        """
        self.received_breakthroughs.append(breakthrough)

        # Update detector's baseline with breakthrough info
        self.detector.update_baseline(breakthrough.reward)

    def get_stats(self) -> dict[str, Any]:
        """Get breakthrough statistics for this search"""
        return {
            'search_id': self.search_id,
            'detector': self.detector.get_stats(),
            'received_breakthroughs': len(self.received_breakthroughs),
            'feed_forward_enabled': self.enable_feed_forward
        }


# ============================================================================
# Factory Functions
# ============================================================================

def create_breakthrough_mcts(
    base_mcts_engine: Any,
    reward_improvement_threshold: float = 2.0,
    confidence_jump_threshold: float = 0.2,
    enable_feed_forward: bool = True
) -> BreakthroughMCTS:
    """Create breakthrough-enhanced MCTS engine"""
    detector = BreakthroughDetector(
        reward_improvement_threshold=reward_improvement_threshold,
        confidence_jump_threshold=confidence_jump_threshold
    )

    broadcaster = FeedForwardBroadcaster()

    return BreakthroughMCTS(
        base_mcts_engine=base_mcts_engine,
        detector=detector,
        broadcaster=broadcaster,
        enable_feed_forward=enable_feed_forward
    )
