"""
Experience Replay Buffer
========================
Store and sample decision trajectories for continual learning.

Capabilities:
- Prioritized experience replay
- Contrastive learning (positive vs negative examples)
- Diversity-based sampling
- Memory consolidation

Philosophy:
Not all experiences are equally valuable for learning. The replay buffer
intelligently stores and samples trajectories to maximize learning efficiency.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from collections import deque
import heapq
import random

logger = logging.getLogger(__name__)

from . import DecisionTrajectory, ReflectionInsights


@dataclass
class ExperienceEntry:
    """Single experience entry with metadata."""

    trajectory: DecisionTrajectory
    priority: float
    timestamp: float
    sample_count: int = 0

    # Metadata for sampling
    is_success: bool = True
    is_novel: bool = False
    teaching_value: float = 0.0  # How much can we learn from this?

    def __lt__(self, other):
        """For heap operations - higher priority = higher value."""
        return self.priority > other.priority


@dataclass
class ReplayStats:
    """Statistics about replay buffer."""

    total_entries: int
    success_rate: float
    avg_priority: float
    novel_count: int

    # Diversity metrics
    unique_tools_used: int
    query_diversity: float


class ExperienceReplayBuffer:
    """
    Prioritized experience replay buffer for continual learning.

    Implements:
    - Prioritized sampling based on TD error or novelty
    - Contrastive pairs (success vs failure)
    - Diversity-based retention
    - Memory consolidation
    """

    def __init__(
        self,
        max_size: int = 10000,
        alpha: float = 0.6,  # Prioritization exponent
        beta: float = 0.4,  # Importance sampling exponent
        novelty_threshold: float = 0.7,
        consolidation_interval: int = 1000
    ):
        """
        Args:
            max_size: Maximum buffer size
            alpha: How much prioritization to use (0=uniform, 1=full priority)
            beta: Importance sampling correction (0=none, 1=full correction)
            novelty_threshold: Threshold for marking experiences as novel
            consolidation_interval: How often to consolidate memory
        """
        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta
        self.novelty_threshold = novelty_threshold
        self.consolidation_interval = consolidation_interval

        # Storage
        self.buffer: List[ExperienceEntry] = []
        self.priority_heap: List[ExperienceEntry] = []

        # Indexing for efficient retrieval
        self.success_indices: List[int] = []
        self.failure_indices: List[int] = []
        self.novel_indices: List[int] = []

        # Statistics
        self.total_added = 0
        self.total_sampled = 0

        # For computing novelty
        self.feature_history: deque = deque(maxlen=100)

    def add(
        self,
        trajectory: DecisionTrajectory,
        priority: Optional[float] = None,
        is_novel: Optional[bool] = None
    ):
        """
        Add a trajectory to the replay buffer.

        Args:
            trajectory: Decision trajectory to store
            priority: Optional explicit priority (otherwise computed)
            is_novel: Optional novelty flag (otherwise computed)
        """

        # Compute priority if not provided
        if priority is None:
            priority = self._compute_priority(trajectory)

        # Compute novelty if not provided
        if is_novel is None:
            is_novel = self._is_novel(trajectory)

        # Compute teaching value
        teaching_value = self._compute_teaching_value(trajectory, is_novel)

        # Create entry
        entry = ExperienceEntry(
            trajectory=trajectory,
            priority=priority,
            timestamp=self.total_added,
            is_success=trajectory.outcome.success,
            is_novel=is_novel,
            teaching_value=teaching_value
        )

        # Add to buffer
        if len(self.buffer) < self.max_size:
            self.buffer.append(entry)
            heapq.heappush(self.priority_heap, entry)
        else:
            # Replace lowest priority entry
            if entry.priority > self.priority_heap[0].priority:
                # Remove lowest priority
                old_entry = heapq.heappushpop(self.priority_heap, entry)
                old_idx = self.buffer.index(old_entry)
                self.buffer[old_idx] = entry

        # Update indices
        self._update_indices()

        # Update feature history for novelty detection
        if hasattr(trajectory, 'features'):
            self.feature_history.append(trajectory.features)

        self.total_added += 1

        # Periodic consolidation
        if self.total_added % self.consolidation_interval == 0:
            self._consolidate_memory()

        logger.debug(
            f"Added trajectory {trajectory.trajectory_id} "
            f"(priority={priority:.3f}, novel={is_novel})"
        )

    def sample(
        self,
        batch_size: int,
        strategy: str = "prioritized"
    ) -> Tuple[List[DecisionTrajectory], np.ndarray, np.ndarray]:
        """
        Sample a batch of trajectories.

        Args:
            batch_size: Number of trajectories to sample
            strategy: Sampling strategy:
                - "prioritized": Sample based on priority
                - "uniform": Random uniform sampling
                - "recent": Sample recent experiences
                - "diverse": Maximize diversity in batch

        Returns:
            Tuple of (trajectories, priorities, importance_weights)
        """

        if len(self.buffer) == 0:
            return [], np.array([]), np.array([])

        batch_size = min(batch_size, len(self.buffer))

        if strategy == "prioritized":
            entries, weights = self._sample_prioritized(batch_size)
        elif strategy == "uniform":
            entries = random.sample(self.buffer, batch_size)
            weights = np.ones(batch_size)
        elif strategy == "recent":
            entries = sorted(self.buffer, key=lambda e: e.timestamp, reverse=True)[:batch_size]
            weights = np.ones(batch_size)
        elif strategy == "diverse":
            entries = self._sample_diverse(batch_size)
            weights = np.ones(batch_size)
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

        # Extract trajectories and priorities
        trajectories = [e.trajectory for e in entries]
        priorities = np.array([e.priority for e in entries])

        # Update sample counts
        for entry in entries:
            entry.sample_count += 1

        self.total_sampled += batch_size

        return trajectories, priorities, weights

    def sample_contrastive_pairs(
        self,
        n_pairs: int
    ) -> List[Tuple[DecisionTrajectory, DecisionTrajectory]]:
        """
        Sample contrastive pairs for learning.

        Returns pairs of (success, failure) trajectories with similar
        queries/contexts to enable contrastive learning.

        Args:
            n_pairs: Number of pairs to sample

        Returns:
            List of (positive_example, negative_example) tuples
        """

        if len(self.success_indices) == 0 or len(self.failure_indices) == 0:
            logger.warning("Not enough success/failure examples for contrastive sampling")
            return []

        pairs = []

        for _ in range(n_pairs):
            # Sample a success
            success_idx = random.choice(self.success_indices)
            success_entry = self.buffer[success_idx]

            # Find most similar failure
            failure_idx = self._find_similar_failure(success_entry)
            if failure_idx is not None:
                failure_entry = self.buffer[failure_idx]
                pairs.append((success_entry.trajectory, failure_entry.trajectory))

        logger.info(f"Sampled {len(pairs)} contrastive pairs")

        return pairs

    def get_high_value_experiences(
        self,
        top_k: int = 100
    ) -> List[DecisionTrajectory]:
        """
        Get highest teaching-value experiences for targeted learning.

        Args:
            top_k: Number of experiences to retrieve

        Returns:
            List of high-value trajectories
        """

        sorted_entries = sorted(
            self.buffer,
            key=lambda e: e.teaching_value,
            reverse=True
        )

        return [e.trajectory for e in sorted_entries[:top_k]]

    def update_priorities(
        self,
        trajectory_ids: List[str],
        new_priorities: List[float]
    ):
        """
        Update priorities for specific trajectories.

        Used after learning to increase priority of experiences
        with high TD error.
        """

        id_to_priority = dict(zip(trajectory_ids, new_priorities))

        updated_count = 0
        for entry in self.buffer:
            if entry.trajectory.trajectory_id in id_to_priority:
                old_priority = entry.priority
                entry.priority = id_to_priority[entry.trajectory.trajectory_id]
                updated_count += 1

        # Rebuild heap with new priorities
        heapq.heapify(self.priority_heap)

        logger.info(f"Updated priorities for {updated_count} trajectories")

    def get_stats(self) -> ReplayStats:
        """Get buffer statistics."""

        if len(self.buffer) == 0:
            return ReplayStats(
                total_entries=0,
                success_rate=0.0,
                avg_priority=0.0,
                novel_count=0,
                unique_tools_used=0,
                query_diversity=0.0
            )

        # Compute statistics
        success_count = len(self.success_indices)
        success_rate = success_count / len(self.buffer)

        priorities = [e.priority for e in self.buffer]
        avg_priority = np.mean(priorities)

        novel_count = len(self.novel_indices)

        # Tool diversity
        tools_used = set()
        for entry in self.buffer:
            if hasattr(entry.trajectory.action_plan, 'selected_tool'):
                tools_used.add(entry.trajectory.action_plan.selected_tool)

        # Query diversity (simplified - based on unique query texts)
        unique_queries = set()
        for entry in self.buffer:
            query_text = str(entry.trajectory.query.get('text', ''))
            unique_queries.add(query_text[:100])  # First 100 chars

        query_diversity = len(unique_queries) / len(self.buffer)

        return ReplayStats(
            total_entries=len(self.buffer),
            success_rate=success_rate,
            avg_priority=avg_priority,
            novel_count=novel_count,
            unique_tools_used=len(tools_used),
            query_diversity=query_diversity
        )

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _compute_priority(self, trajectory: DecisionTrajectory) -> float:
        """
        Compute priority for a trajectory.

        Higher priority for:
        - Novel experiences
        - Failures (to learn from mistakes)
        - High-quality successes (to reinforce)
        - Experiences with strong feedback
        """

        priority = 1.0

        # Quality-based priority
        if hasattr(trajectory, 'quality_score'):
            quality = trajectory.quality_score
            # High quality successes and low quality failures are both valuable
            if trajectory.outcome.success:
                priority += quality
            else:
                priority += (1.0 - quality)

        # Feedback boost
        if trajectory.feedback and trajectory.feedback.rating is not None:
            feedback_signal = abs(trajectory.feedback.rating - 3.0) / 2.0
            priority += feedback_signal

        # Novelty boost
        if self._is_novel(trajectory):
            priority *= 1.5

        # Temporal decay - recent experiences are more relevant
        age = self.total_added - len(self.buffer)
        decay = 1.0 / (1.0 + age / 1000.0)
        priority *= decay

        return priority

    def _is_novel(self, trajectory: DecisionTrajectory) -> bool:
        """
        Determine if a trajectory is novel.

        Novel if its features are significantly different from
        recent experiences.
        """

        if len(self.feature_history) == 0:
            return True

        if not hasattr(trajectory, 'features'):
            return False

        # Simplified novelty: check if query is very different
        query_text = str(trajectory.query.get('text', ''))

        # Check against recent queries
        recent_queries = [
            str(f.get('text', '')) for f in list(self.feature_history)[-20:]
            if isinstance(f, dict)
        ]

        # Simple similarity check
        similar_count = sum(1 for q in recent_queries if query_text[:50] in q or q[:50] in query_text)

        is_novel = (similar_count == 0)

        return is_novel

    def _compute_teaching_value(
        self,
        trajectory: DecisionTrajectory,
        is_novel: bool
    ) -> float:
        """
        Compute how much we can learn from this trajectory.

        High teaching value:
        - Novel situations
        - Clear successes or failures
        - Strong user feedback
        - Interesting edge cases
        """

        value = 0.0

        # Novelty
        if is_novel:
            value += 0.3

        # Clear outcomes (not ambiguous)
        if hasattr(trajectory, 'quality_score'):
            clarity = abs(trajectory.quality_score - 0.5) * 2.0
            value += clarity * 0.2

        # User feedback
        if trajectory.feedback and trajectory.feedback.rating is not None:
            # Extreme ratings are more informative
            extremity = abs(trajectory.feedback.rating - 3.0) / 2.0
            value += extremity * 0.3

        # Efficiency (fast execution is good for success, slow for failure is bad)
        if trajectory.outcome.success:
            if trajectory.outcome.execution_time < 1.0:
                value += 0.1
        else:
            if trajectory.outcome.execution_time > 3.0:
                value += 0.1  # Slow failures teach us what not to do

        # Normalize to [0, 1]
        value = min(1.0, value)

        return value

    def _update_indices(self):
        """Update success/failure/novel indices."""

        self.success_indices = []
        self.failure_indices = []
        self.novel_indices = []

        for i, entry in enumerate(self.buffer):
            if entry.is_success:
                self.success_indices.append(i)
            else:
                self.failure_indices.append(i)

            if entry.is_novel:
                self.novel_indices.append(i)

    def _sample_prioritized(
        self,
        batch_size: int
    ) -> Tuple[List[ExperienceEntry], np.ndarray]:
        """Sample based on priorities."""

        # Compute sampling probabilities
        priorities = np.array([e.priority for e in self.buffer])
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        # Sample indices
        indices = np.random.choice(
            len(self.buffer),
            size=batch_size,
            replace=False,
            p=probabilities
        )

        # Compute importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize

        entries = [self.buffer[i] for i in indices]

        return entries, weights

    def _sample_diverse(self, batch_size: int) -> List[ExperienceEntry]:
        """
        Sample a diverse batch.

        Uses greedy diversity selection - iteratively add the most
        different experience from those already selected.
        """

        if len(self.buffer) == 0:
            return []

        selected = []
        remaining = list(self.buffer)

        # Start with a random experience
        first = random.choice(remaining)
        selected.append(first)
        remaining.remove(first)

        # Greedily add most diverse
        for _ in range(batch_size - 1):
            if not remaining:
                break

            # Find most diverse from selected
            best_entry = None
            best_diversity = -1.0

            for entry in remaining:
                diversity = self._compute_diversity(entry, selected)
                if diversity > best_diversity:
                    best_diversity = diversity
                    best_entry = entry

            if best_entry:
                selected.append(best_entry)
                remaining.remove(best_entry)

        return selected

    def _compute_diversity(
        self,
        entry: ExperienceEntry,
        selected: List[ExperienceEntry]
    ) -> float:
        """
        Compute diversity score for an entry relative to selected.

        Higher score = more diverse.
        """

        diversity = 0.0

        # Tool diversity
        if hasattr(entry.trajectory.action_plan, 'selected_tool'):
            tool = entry.trajectory.action_plan.selected_tool
            selected_tools = [
                e.trajectory.action_plan.selected_tool
                for e in selected
                if hasattr(e.trajectory.action_plan, 'selected_tool')
            ]
            if tool not in selected_tools:
                diversity += 0.5

        # Outcome diversity
        success_count = sum(1 for e in selected if e.is_success)
        if entry.is_success:
            # Prefer minority class
            diversity += (len(selected) - success_count) / max(1, len(selected))
        else:
            diversity += success_count / max(1, len(selected))

        # Query diversity (simplified)
        query_text = str(entry.trajectory.query.get('text', ''))[:100]
        selected_queries = [
            str(e.trajectory.query.get('text', ''))[:100]
            for e in selected
        ]

        similar = any(query_text in q or q in query_text for q in selected_queries)
        if not similar:
            diversity += 0.3

        return diversity

    def _find_similar_failure(self, success_entry: ExperienceEntry) -> Optional[int]:
        """
        Find a failure with similar query/context to a success.

        For contrastive learning.
        """

        if len(self.failure_indices) == 0:
            return None

        success_query = str(success_entry.trajectory.query.get('text', ''))[:100]

        best_idx = None
        best_similarity = 0.0

        for idx in self.failure_indices:
            failure_entry = self.buffer[idx]
            failure_query = str(failure_entry.trajectory.query.get('text', ''))[:100]

            # Simple similarity
            similarity = len(set(success_query.split()) & set(failure_query.split()))

            if similarity > best_similarity:
                best_similarity = similarity
                best_idx = idx

        return best_idx if best_similarity > 0 else random.choice(self.failure_indices)

    def _consolidate_memory(self):
        """
        Consolidate memory by removing low-value experiences.

        This is like 'sleep' - strengthen important memories,
        weaken unimportant ones.
        """

        if len(self.buffer) < self.max_size * 0.9:
            return  # Not full enough to consolidate

        logger.info("Consolidating memory...")

        # Sort by teaching value
        sorted_entries = sorted(
            self.buffer,
            key=lambda e: e.teaching_value + e.priority * 0.5,
            reverse=True
        )

        # Keep top entries
        keep_count = int(self.max_size * 0.8)
        self.buffer = sorted_entries[:keep_count]

        # Rebuild heap
        self.priority_heap = list(self.buffer)
        heapq.heapify(self.priority_heap)

        # Update indices
        self._update_indices()

        logger.info(f"Consolidated to {len(self.buffer)} high-value experiences")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import asyncio

    async def demo():
        print("="*80)
        print("Experience Replay Buffer Demo")
        print("="*80 + "\n")

        # Create buffer
        buffer = ExperienceReplayBuffer(
            max_size=100,
            alpha=0.6,
            beta=0.4
        )

        # Simulate adding experiences
        print("📥 Adding experiences...\n")

        from . import DecisionTrajectory, Outcome, Feedback

        for i in range(50):
            # Mix of successes and failures
            is_success = random.random() > 0.3

            trajectory = DecisionTrajectory(
                trajectory_id=f"traj_{i}",
                query={"text": f"Query {i // 10}"},  # Some similar queries
                features={"motifs": []},
                context={"shards": []},
                action_plan={"selected_tool": random.choice(["tool_a", "tool_b", "tool_c"])},
                outcome=Outcome(
                    success=is_success,
                    execution_time=random.uniform(0.5, 3.0),
                    user_satisfaction=random.random() if is_success else random.random() * 0.5
                ),
                feedback=Feedback(
                    rating=random.randint(4, 5) if is_success else random.randint(1, 3),
                    helpful=is_success
                )
            )

            buffer.add(trajectory)

        # Get statistics
        stats = buffer.get_stats()
        print("📊 Buffer Statistics:")
        print(f"  Total entries: {stats.total_entries}")
        print(f"  Success rate: {stats.success_rate:.2%}")
        print(f"  Avg priority: {stats.avg_priority:.3f}")
        print(f"  Novel count: {stats.novel_count}")
        print(f"  Unique tools: {stats.unique_tools_used}")
        print(f"  Query diversity: {stats.query_diversity:.2%}")

        # Sample prioritized batch
        print("\n\n🎲 Prioritized Sampling...\n")

        trajectories, priorities, weights = buffer.sample(
            batch_size=10,
            strategy="prioritized"
        )

        print(f"Sampled {len(trajectories)} trajectories:")
        for i, (traj, p, w) in enumerate(zip(trajectories, priorities, weights)):
            success_mark = "✅" if traj.outcome.success else "❌"
            print(f"  {i+1}. {traj.trajectory_id} {success_mark} (priority={p:.3f}, weight={w:.3f})")

        # Sample contrastive pairs
        print("\n\n🔄 Contrastive Pairs...\n")

        pairs = buffer.sample_contrastive_pairs(n_pairs=5)

        print(f"Sampled {len(pairs)} contrastive pairs:")
        for i, (success, failure) in enumerate(pairs):
            print(f"  {i+1}. ✅ {success.trajectory_id} vs ❌ {failure.trajectory_id}")

        # Get high-value experiences
        print("\n\n⭐ High-Value Experiences...\n")

        high_value = buffer.get_high_value_experiences(top_k=5)

        print(f"Top {len(high_value)} high-value experiences:")
        for i, traj in enumerate(high_value):
            success_mark = "✅" if traj.outcome.success else "❌"
            print(f"  {i+1}. {traj.trajectory_id} {success_mark}")

        # Diverse sampling
        print("\n\n🎨 Diverse Sampling...\n")

        diverse_trajs, _, _ = buffer.sample(batch_size=8, strategy="diverse")

        tools_used = [
            traj.action_plan['selected_tool']
            for traj in diverse_trajs
        ]
        outcomes = [traj.outcome.success for traj in diverse_trajs]

        print(f"Diverse batch of {len(diverse_trajs)} trajectories:")
        print(f"  Tools: {set(tools_used)}")
        print(f"  Success rate: {sum(outcomes)/len(outcomes):.2%}")

        print("\n" + "="*80)
        print("✅ Experience Replay Demo Complete!")
        print("="*80)

    asyncio.run(demo())
