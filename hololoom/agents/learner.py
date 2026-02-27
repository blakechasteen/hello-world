"""
Working Memory Learner - Pattern Learning & Persistence

Learns from successful working memory states:
- Records snapshots of semantic focus + graph activation + tensioned threads
- Identifies patterns that lead to high-confidence outcomes
- Suggests improvements based on learned patterns
- Persists learning across sessions

Key Insight: "Save what works" - especially the computational warp space state.
"""

from typing import Dict, List, Optional
from pathlib import Path
import json
import time
import uuid
import numpy as np

from HoloLoom.agents.types import (
    WorkingMemorySnapshot,
    LearnedPattern,
    WorkingMemoryState,
    AgentProfile
)
from HoloLoom.protocols.types import Query
from HoloLoom.semantic_calculus.dimensions import EXTENDED_244_DIMENSIONS


class WorkingMemoryLearner:
    """
    Learn from successful working memory states.

    Tracks:
    - What semantic focus worked (244D position)
    - Which nodes were activated (graph state)
    - What threads were tensioned (computational state)
    - What outcomes resulted (confidence, tool, success)
    """

    def __init__(
        self,
        profile: AgentProfile,
        persist_dir: Path = Path('./agents_memory')
    ):
        self.profile = profile
        self.persist_dir = persist_dir / profile.agent_id
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # History (rolling window)
        self.snapshots: List[WorkingMemorySnapshot] = []
        self.max_snapshots = 1000

        # Learned patterns
        self.patterns: Dict[str, LearnedPattern] = {}

        # Load previous session
        self._load_state()

    def record_snapshot(
        self,
        query: Query,
        state: WorkingMemoryState,
        outcome: Dict
    ):
        """
        Record working memory state + outcome for learning.

        Args:
            query: The query that was processed
            state: Working memory state (semantic + graph + computational)
            outcome: Result dict with 'confidence', 'tool_used', 'warp_operations'
        """
        # Extract top semantic dimensions
        top_dims_idx = np.argsort(np.abs(state.focus_vector))[-5:]
        # Handle embeddings larger than named dimensions
        top_dims = []
        for i in top_dims_idx:
            if i < len(EXTENDED_244_DIMENSIONS):
                top_dims.append(EXTENDED_244_DIMENSIONS[i])
            else:
                top_dims.append(f"dim_{i}")  # Unnamed dimension

        # Get highly activated nodes
        highly_activated = [
            node_id for node_id, activation
            in state.activation_map.items()
            if activation > 0.7
        ]

        # Create snapshot
        snapshot = WorkingMemorySnapshot(
            timestamp=time.time(),
            query_text=query.text,
            focus_vector=state.focus_vector.copy(),
            attention_radius=state.attention_radius,
            top_semantic_dimensions=top_dims,
            activation_map=state.activation_map.copy(),
            highly_activated=highly_activated,
            tensioned_threads=list(state.tensioned_threads),
            tension_profile=state.tension_profile.copy(),
            warp_operations=outcome.get('warp_operations', []),
            confidence=outcome.get('confidence', 0.0),
            success=outcome.get('confidence', 0.0) >= self.profile.success_threshold,
            tool_used=outcome.get('tool_used')
        )

        # Add to history
        self.snapshots.append(snapshot)
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots.pop(0)  # Keep rolling window

        # Learn from successful snapshots
        if snapshot.success:
            self._update_patterns(snapshot)

    def _update_patterns(self, snapshot: WorkingMemorySnapshot):
        """Extract and update learned patterns"""

        # Find similar existing pattern
        pattern = self._find_similar_pattern(snapshot)

        if pattern:
            # Update existing pattern
            pattern.total_count += 1
            pattern.success_count += 1

            # Update semantic region (moving average)
            alpha = 0.1  # Learning rate
            pattern.semantic_region = (
                (1 - alpha) * pattern.semantic_region +
                alpha * snapshot.focus_vector
            )

            # Update typical activation pattern
            for node_id, activation in snapshot.activation_map.items():
                if node_id in pattern.typical_activation_pattern:
                    pattern.typical_activation_pattern[node_id] = (
                        (1 - alpha) * pattern.typical_activation_pattern[node_id] +
                        alpha * activation
                    )
                else:
                    pattern.typical_activation_pattern[node_id] = activation

            # Update critical threads (appear in >80% of successes)
            pattern.critical_threads = self._identify_critical_threads(pattern)

            # Update stats
            pattern.avg_confidence = (
                (pattern.avg_confidence * (pattern.total_count - 1) + snapshot.confidence) /
                pattern.total_count
            )
            pattern.last_used = snapshot.timestamp

        else:
            # Create new pattern
            pattern_id = str(uuid.uuid4())[:8]

            self.patterns[pattern_id] = LearnedPattern(
                pattern_id=pattern_id,
                semantic_region=snapshot.focus_vector.copy(),
                typical_activation_pattern=snapshot.activation_map.copy(),
                critical_threads=snapshot.tensioned_threads.copy(),
                success_count=1,
                total_count=1,
                avg_confidence=snapshot.confidence,
                first_seen=snapshot.timestamp,
                last_used=snapshot.timestamp
            )

    def _find_similar_pattern(
        self,
        snapshot: WorkingMemorySnapshot,
        similarity_threshold: float = 0.85
    ) -> Optional[LearnedPattern]:
        """Find pattern similar to current snapshot (semantic similarity)"""

        for pattern in self.patterns.values():
            # Semantic similarity
            norm_product = (
                np.linalg.norm(snapshot.focus_vector) *
                np.linalg.norm(pattern.semantic_region)
            )

            if norm_product > 0:
                semantic_sim = np.dot(
                    snapshot.focus_vector,
                    pattern.semantic_region
                ) / norm_product

                if semantic_sim > similarity_threshold:
                    return pattern

        return None

    def _identify_critical_threads(self, pattern: LearnedPattern) -> List[str]:
        """Find threads that appear in >80% of successful uses of this pattern"""

        # Get all snapshots that matched this pattern
        matching_snapshots = [
            snap for snap in self.snapshots
            if snap.success and self._snapshot_matches_pattern(snap, pattern)
        ]

        if not matching_snapshots:
            return pattern.critical_threads

        # Count thread frequency
        thread_counts = {}
        for snap in matching_snapshots:
            for thread in snap.tensioned_threads:
                thread_counts[thread] = thread_counts.get(thread, 0) + 1

        # Return threads appearing in >80% of successes
        threshold = 0.8 * len(matching_snapshots)
        critical = [
            thread for thread, count in thread_counts.items()
            if count >= threshold
        ]

        return critical

    def _snapshot_matches_pattern(
        self,
        snapshot: WorkingMemorySnapshot,
        pattern: LearnedPattern
    ) -> bool:
        """Check if snapshot matches pattern (semantic similarity > 0.85)"""
        norm_product = (
            np.linalg.norm(snapshot.focus_vector) *
            np.linalg.norm(pattern.semantic_region)
        )

        if norm_product > 0:
            semantic_sim = np.dot(
                snapshot.focus_vector,
                pattern.semantic_region
            ) / norm_product
            return semantic_sim > 0.85

        return False

    def suggest_improvements(
        self,
        current_state: WorkingMemoryState,
        query: Query
    ) -> Dict:
        """
        Suggest improvements based on learned patterns.

        Returns:
            Dict with 'pattern_id', 'similarity', 'suggestions' list
        """

        # Find most similar successful pattern
        query_vector = current_state.focus_vector

        best_pattern = None
        best_similarity = 0.0

        for pattern in self.patterns.values():
            if pattern.success_rate() < 0.7:
                continue  # Only consider successful patterns

            norm_product = (
                np.linalg.norm(query_vector) *
                np.linalg.norm(pattern.semantic_region)
            )

            if norm_product > 0:
                similarity = np.dot(query_vector, pattern.semantic_region) / norm_product

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_pattern = pattern

        if not best_pattern:
            return {'suggestions': []}

        suggestions = []

        # Suggest critical threads to tension
        missing_threads = set(best_pattern.critical_threads) - current_state.tensioned_threads
        if missing_threads:
            suggestions.append({
                'type': 'add_threads',
                'threads': list(missing_threads),
                'reason': f'Present in {best_pattern.success_rate():.0%} of successful queries'
            })

        # Suggest activation boosts
        for node_id, target_activation in best_pattern.typical_activation_pattern.items():
            current_activation = current_state.activation_map.get(node_id, 0.0)
            if target_activation > current_activation + 0.2:
                suggestions.append({
                    'type': 'boost_activation',
                    'node': node_id,
                    'from': current_activation,
                    'to': target_activation,
                    'reason': 'Typical in successful patterns'
                })

        # Suggest semantic adjustment
        semantic_diff = best_pattern.semantic_region - query_vector
        if np.linalg.norm(semantic_diff) > 0.1:
            # Find dimensions to adjust
            top_diffs_idx = np.argsort(np.abs(semantic_diff))[-3:]
            top_diffs = [EXTENDED_244_DIMENSIONS[i] for i in top_diffs_idx]

            suggestions.append({
                'type': 'adjust_focus',
                'dimensions': top_diffs,
                'direction': 'increase' if semantic_diff[top_diffs_idx[0]] > 0 else 'decrease',
                'reason': f'Match pattern with {best_pattern.avg_confidence:.2f} avg confidence'
            })

        return {
            'pattern_id': best_pattern.pattern_id,
            'pattern_success_rate': best_pattern.success_rate(),
            'pattern_avg_confidence': best_pattern.avg_confidence,
            'similarity': best_similarity,
            'suggestions': suggestions
        }

    def get_stats(self) -> Dict:
        """Get learning statistics"""
        if not self.snapshots:
            return {
                'total_snapshots': 0,
                'successful_snapshots': 0,
                'patterns_learned': 0
            }

        successful = [s for s in self.snapshots if s.success]

        return {
            'total_snapshots': len(self.snapshots),
            'successful_snapshots': len(successful),
            'success_rate': len(successful) / len(self.snapshots),
            'patterns_learned': len(self.patterns),
            'avg_confidence': np.mean([s.confidence for s in self.snapshots]),
            'avg_confidence_successful': np.mean([s.confidence for s in successful]) if successful else 0.0
        }

    def save_state(self):
        """Persist snapshots and patterns to disk"""

        # Save snapshots (JSONL format)
        snapshots_file = self.persist_dir / 'snapshots.jsonl'
        with open(snapshots_file, 'w') as f:
            for snap in self.snapshots:
                f.write(json.dumps(snap.to_dict()) + '\n')

        # Save patterns (JSON format)
        patterns_file = self.persist_dir / 'patterns.json'
        patterns_data = {
            pattern_id: pattern.to_dict()
            for pattern_id, pattern in self.patterns.items()
        }

        with open(patterns_file, 'w') as f:
            json.dump(patterns_data, f, indent=2)

    def _load_state(self):
        """Load previous snapshots and patterns from disk"""

        # Load snapshots
        snapshots_file = self.persist_dir / 'snapshots.jsonl'
        if snapshots_file.exists():
            with open(snapshots_file) as f:
                for line in f:
                    snap_dict = json.loads(line)
                    self.snapshots.append(WorkingMemorySnapshot.from_dict(snap_dict))

        # Load patterns
        patterns_file = self.persist_dir / 'patterns.json'
        if patterns_file.exists():
            with open(patterns_file) as f:
                patterns_data = json.load(f)

            for pattern_id, data in patterns_data.items():
                self.patterns[pattern_id] = LearnedPattern.from_dict(data)
