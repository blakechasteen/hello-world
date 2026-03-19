"""
Pattern Miner for Workflow Builder AI Suggestions.

Extracts patterns from workflow execution history to suggest next nodes
based on frequency and success rates. Uses Thompson Sampling for
intelligent recommendations.

Created: December 2025 (Wave 3.2)
"""

import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class NodePattern:
    """A pattern of node sequences with success metrics."""
    sequence: tuple[str, ...]
    frequency: int = 0
    success_count: int = 0
    failure_count: int = 0
    avg_latency_ms: float = 0.0
    last_seen: str | None = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.5

    @property
    def thompson_sample(self) -> float:
        """Thompson Sampling score (Beta distribution)."""
        # Beta(alpha, beta) where alpha = successes + 1, beta = failures + 1
        alpha = self.success_count + 1
        beta = self.failure_count + 1
        return random.betavariate(alpha, beta)


@dataclass
class WorkflowPatternMiner:
    """
    Mines workflow execution patterns for AI suggestions.

    Features:
    - N-gram pattern extraction (pairs, triples)
    - Success rate tracking
    - Thompson Sampling recommendations
    - Pattern persistence
    """

    # Pattern storage
    pair_patterns: dict[tuple[str, str], NodePattern] = field(default_factory=dict)
    triple_patterns: dict[tuple[str, str, str], NodePattern] = field(default_factory=dict)

    # Node popularity (how often each node type is used)
    node_frequency: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    node_success: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    node_failure: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Pattern cache for fast lookups
    _suggestions_cache: dict[str, list[str]] = field(default_factory=dict)
    _cache_timestamp: float | None = None
    _cache_ttl: float = 60.0  # Cache for 60 seconds

    def extract_patterns(self, workflow_history: list[dict]) -> None:
        """
        Extract patterns from workflow execution history.

        Args:
            workflow_history: List of workflow execution records with
                             'node_sequence', 'success', 'total_latency_ms'
        """
        for record in workflow_history:
            sequence = record.get('node_sequence', [])
            success = record.get('success', True)
            latency = record.get('total_latency_ms', 0)

            if len(sequence) < 2:
                continue

            # Update node frequency
            for node_type in sequence:
                self.node_frequency[node_type] += 1
                if success:
                    self.node_success[node_type] += 1
                else:
                    self.node_failure[node_type] += 1

            # Extract pairs (bigrams)
            for i in range(len(sequence) - 1):
                pair = (sequence[i], sequence[i + 1])
                self._update_pattern(
                    self.pair_patterns, pair, success, latency / len(sequence)
                )

            # Extract triples (trigrams)
            for i in range(len(sequence) - 2):
                triple = (sequence[i], sequence[i + 1], sequence[i + 2])
                self._update_pattern(
                    self.triple_patterns, triple, success, latency / len(sequence)
                )

        # Invalidate cache
        self._suggestions_cache.clear()
        self._cache_timestamp = None

        logger.info(f"Extracted {len(self.pair_patterns)} pair patterns, "
                   f"{len(self.triple_patterns)} triple patterns")

    def _update_pattern(
        self,
        pattern_dict: dict,
        pattern: tuple,
        success: bool,
        latency_ms: float
    ) -> None:
        """Update pattern statistics."""
        if pattern not in pattern_dict:
            pattern_dict[pattern] = NodePattern(sequence=pattern)

        p = pattern_dict[pattern]
        p.frequency += 1
        if success:
            p.success_count += 1
        else:
            p.failure_count += 1

        # Running average for latency
        p.avg_latency_ms = (
            (p.avg_latency_ms * (p.frequency - 1) + latency_ms) / p.frequency
        )
        p.last_seen = datetime.now().isoformat()

    def suggest_next_nodes(
        self,
        current_nodes: list[str],
        k: int = 5,
        strategy: str = 'thompson'
    ) -> list[dict[str, any]]:
        """
        Suggest likely next nodes based on current workflow.

        Args:
            current_nodes: List of node types in current workflow (execution order)
            k: Number of suggestions to return
            strategy: 'thompson' (exploration), 'frequency' (exploitation), or 'hybrid'

        Returns:
            List of suggestions with scores and metadata
        """
        if not current_nodes:
            # No current nodes - suggest popular starting nodes
            return self._suggest_starting_nodes(k)

        candidates = defaultdict(lambda: {
            'score': 0.0,
            'frequency': 0,
            'success_rate': 0.0,
            'sources': []
        })

        # Get last 1-2 nodes for pattern matching
        last_node = current_nodes[-1]
        last_two = tuple(current_nodes[-2:]) if len(current_nodes) >= 2 else None

        # Match against pair patterns (what typically follows last_node)
        for pattern, stats in self.pair_patterns.items():
            if pattern[0] == last_node:
                next_node = pattern[1]
                score = self._calculate_score(stats, strategy)
                candidates[next_node]['score'] = max(
                    candidates[next_node]['score'], score
                )
                candidates[next_node]['frequency'] += stats.frequency
                candidates[next_node]['success_rate'] = stats.success_rate
                candidates[next_node]['sources'].append(f"pair:{pattern}")

        # Match against triple patterns (higher weight for matching context)
        if last_two:
            for pattern, stats in self.triple_patterns.items():
                if pattern[:2] == last_two:
                    next_node = pattern[2]
                    # Triple patterns get 1.5x weight (more context)
                    score = self._calculate_score(stats, strategy) * 1.5
                    candidates[next_node]['score'] = max(
                        candidates[next_node]['score'], score
                    )
                    candidates[next_node]['frequency'] += stats.frequency
                    candidates[next_node]['success_rate'] = max(
                        candidates[next_node]['success_rate'],
                        stats.success_rate
                    )
                    candidates[next_node]['sources'].append(f"triple:{pattern}")

        # Sort by score and return top k
        sorted_candidates = sorted(
            candidates.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )[:k]

        return [
            {
                'node_type': node_type,
                'score': data['score'],
                'frequency': data['frequency'],
                'success_rate': data['success_rate'],
                'sources': data['sources'],
                'confidence': min(1.0, data['frequency'] / 10)  # Confidence grows with samples
            }
            for node_type, data in sorted_candidates
        ]

    def _calculate_score(self, stats: NodePattern, strategy: str) -> float:
        """Calculate suggestion score based on strategy."""
        if strategy == 'thompson':
            return stats.thompson_sample
        elif strategy == 'frequency':
            return stats.frequency * stats.success_rate
        else:  # hybrid
            thompson = stats.thompson_sample
            frequency = stats.frequency / max(1, sum(p.frequency for p in self.pair_patterns.values()))
            return 0.6 * thompson + 0.4 * frequency

    def _suggest_starting_nodes(self, k: int) -> list[dict[str, any]]:
        """Suggest good starting nodes based on overall popularity and success."""
        candidates = []

        for node_type, freq in self.node_frequency.items():
            success = self.node_success.get(node_type, 0)
            failure = self.node_failure.get(node_type, 0)
            total = success + failure

            # Thompson sample for this node type
            alpha = success + 1
            beta = failure + 1
            score = random.betavariate(alpha, beta) * (freq / max(1, sum(self.node_frequency.values())))

            candidates.append({
                'node_type': node_type,
                'score': score,
                'frequency': freq,
                'success_rate': success / total if total > 0 else 0.5,
                'sources': ['popularity'],
                'confidence': min(1.0, freq / 10)
            })

        # Sort and return top k
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:k]

    def get_common_patterns(
        self,
        workflow_type: str | None = None,
        min_frequency: int = 2
    ) -> list[dict[str, any]]:
        """
        Get common workflow patterns (templates).

        Args:
            workflow_type: Filter by workflow name prefix (optional)
            min_frequency: Minimum occurrences to include

        Returns:
            List of common patterns with statistics
        """
        patterns = []

        # Combine pair and triple patterns
        for pattern, stats in self.pair_patterns.items():
            if stats.frequency >= min_frequency:
                patterns.append({
                    'pattern': list(pattern),
                    'type': 'pair',
                    'frequency': stats.frequency,
                    'success_rate': stats.success_rate,
                    'avg_latency_ms': stats.avg_latency_ms
                })

        for pattern, stats in self.triple_patterns.items():
            if stats.frequency >= min_frequency:
                patterns.append({
                    'pattern': list(pattern),
                    'type': 'triple',
                    'frequency': stats.frequency,
                    'success_rate': stats.success_rate,
                    'avg_latency_ms': stats.avg_latency_ms
                })

        # Sort by frequency
        patterns.sort(key=lambda x: x['frequency'], reverse=True)
        return patterns

    def get_node_performance_ranking(self) -> list[dict[str, any]]:
        """
        Get performance ranking of all node types.

        Returns:
            List of node types ranked by Thompson Sampling score
        """
        rankings = []

        for node_type in self.node_frequency.keys():
            success = self.node_success.get(node_type, 0)
            failure = self.node_failure.get(node_type, 0)
            freq = self.node_frequency.get(node_type, 0)
            total = success + failure

            # Thompson sample
            alpha = success + 1
            beta = failure + 1
            thompson_score = random.betavariate(alpha, beta)

            rankings.append({
                'node_type': node_type,
                'thompson_score': thompson_score,
                'frequency': freq,
                'success_rate': success / total if total > 0 else 0.5,
                'success_count': success,
                'failure_count': failure
            })

        rankings.sort(key=lambda x: x['thompson_score'], reverse=True)
        return rankings

    def save_patterns(self, filepath: str = 'workflow_patterns.json') -> None:
        """Save patterns to JSON file."""
        data = {
            'pair_patterns': {
                str(k): {
                    'sequence': list(v.sequence),
                    'frequency': v.frequency,
                    'success_count': v.success_count,
                    'failure_count': v.failure_count,
                    'avg_latency_ms': v.avg_latency_ms,
                    'last_seen': v.last_seen
                }
                for k, v in self.pair_patterns.items()
            },
            'triple_patterns': {
                str(k): {
                    'sequence': list(v.sequence),
                    'frequency': v.frequency,
                    'success_count': v.success_count,
                    'failure_count': v.failure_count,
                    'avg_latency_ms': v.avg_latency_ms,
                    'last_seen': v.last_seen
                }
                for k, v in self.triple_patterns.items()
            },
            'node_frequency': dict(self.node_frequency),
            'node_success': dict(self.node_success),
            'node_failure': dict(self.node_failure),
            'saved_at': datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved patterns to {filepath}")

    def load_patterns(self, filepath: str = 'workflow_patterns.json') -> bool:
        """Load patterns from JSON file."""
        if not Path(filepath).exists():
            logger.warning(f"Pattern file not found: {filepath}")
            return False

        try:
            with open(filepath) as f:
                data = json.load(f)

            # Load pair patterns
            self.pair_patterns.clear()
            for k, v in data.get('pair_patterns', {}).items():
                pattern = NodePattern(
                    sequence=tuple(v['sequence']),
                    frequency=v['frequency'],
                    success_count=v['success_count'],
                    failure_count=v['failure_count'],
                    avg_latency_ms=v['avg_latency_ms'],
                    last_seen=v.get('last_seen')
                )
                self.pair_patterns[tuple(v['sequence'])] = pattern

            # Load triple patterns
            self.triple_patterns.clear()
            for k, v in data.get('triple_patterns', {}).items():
                pattern = NodePattern(
                    sequence=tuple(v['sequence']),
                    frequency=v['frequency'],
                    success_count=v['success_count'],
                    failure_count=v['failure_count'],
                    avg_latency_ms=v['avg_latency_ms'],
                    last_seen=v.get('last_seen')
                )
                self.triple_patterns[tuple(v['sequence'])] = pattern

            # Load node stats
            self.node_frequency = defaultdict(int, data.get('node_frequency', {}))
            self.node_success = defaultdict(int, data.get('node_success', {}))
            self.node_failure = defaultdict(int, data.get('node_failure', {}))

            logger.info(f"Loaded {len(self.pair_patterns)} pair patterns, "
                       f"{len(self.triple_patterns)} triple patterns from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to load patterns: {e}")
            return False


# Global pattern miner instance
_pattern_miner: WorkflowPatternMiner | None = None


def get_pattern_miner() -> WorkflowPatternMiner:
    """Get or create global pattern miner instance."""
    global _pattern_miner
    if _pattern_miner is None:
        _pattern_miner = WorkflowPatternMiner()
        # Try to load existing patterns
        _pattern_miner.load_patterns()
    return _pattern_miner


def update_patterns_from_history(workflow_history: list[dict]) -> None:
    """Update global pattern miner from workflow history."""
    miner = get_pattern_miner()
    miner.extract_patterns(workflow_history)
    miner.save_patterns()


def suggest_next_nodes(current_nodes: list[str], k: int = 5) -> list[dict[str, any]]:
    """Suggest next nodes using global pattern miner."""
    miner = get_pattern_miner()
    return miner.suggest_next_nodes(current_nodes, k=k, strategy='thompson')


def get_common_patterns(min_frequency: int = 2) -> list[dict[str, any]]:
    """Get common patterns using global pattern miner."""
    miner = get_pattern_miner()
    return miner.get_common_patterns(min_frequency=min_frequency)
