# -*- coding: utf-8 -*-
"""
Learning Tracker
================

Tracks user interactions with HoloLoom queries to improve suggestions over time.

Learns from:
- Which queries users run most often
- Which suggestions they accept vs. reject
- Time of day patterns
- Task completion patterns
- Query reformulations

This data feeds back into:
- Query classification (improve intent detection)
- Thompson Sampling (tool selection probabilities)
- Response ranking (what to show first)
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import Counter, defaultdict


@dataclass
class QueryFeedback:
    """
    Record of user interaction with a query.
    """
    query: str                          # Original query string
    query_type: str                     # Classified type
    result_count: int                   # Number of results
    accepted: bool                      # Did user accept suggestion?
    timestamp: datetime                 # When query was made
    time_to_action: Optional[float] = None  # Seconds until user took action
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Serialize to dict."""
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> 'QueryFeedback':
        """Deserialize from dict."""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class LearningTracker:
    """
    Tracks query usage and learns patterns.

    Features:
    - Query frequency tracking
    - Acceptance/rejection tracking
    - Time-of-day patterns
    - Query effectiveness metrics
    - Reformulation detection

    Usage:
        tracker = LearningTracker()

        # Log query
        tracker.log_query(
            query="What should I work on?",
            query_type="next_task",
            result_count=3,
            accepted=True
        )

        # Get insights
        stats = tracker.get_statistics()
        recommendations = tracker.get_query_recommendations()
    """

    def __init__(self, log_file: Optional[str] = None):
        self.log_file = Path(log_file) if log_file else Path.home() / ".hololoom-learning.jsonl"
        self.feedback_log: List[QueryFeedback] = []

        # Load existing data
        if self.log_file.exists():
            self._load()

    def _load(self):
        """Load learning data from file."""
        with self.log_file.open('r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    feedback = QueryFeedback.from_dict(data)
                    self.feedback_log.append(feedback)
                except Exception as e:
                    print(f"Warning: Failed to load learning entry: {e}")

    def _save(self, feedback: QueryFeedback):
        """Append feedback to log file."""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        with self.log_file.open('a') as f:
            f.write(json.dumps(feedback.to_dict()) + "\n")

    def log_query(
        self,
        query: str,
        query_type: str,
        result_count: int,
        accepted: bool = False,
        time_to_action: Optional[float] = None,
        **metadata
    ):
        """
        Log a query interaction.

        Args:
            query: The query string
            query_type: Classified query type
            result_count: Number of results returned
            accepted: Whether user accepted the suggestion
            time_to_action: Seconds until user took action
            **metadata: Additional metadata
        """
        feedback = QueryFeedback(
            query=query,
            query_type=query_type,
            result_count=result_count,
            accepted=accepted,
            timestamp=datetime.now(),
            time_to_action=time_to_action,
            metadata=metadata
        )

        self.feedback_log.append(feedback)
        self._save(feedback)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get learning statistics.

        Returns:
            Dict with various metrics
        """
        if not self.feedback_log:
            return {
                'total_queries': 0,
                'message': 'No data yet'
            }

        total = len(self.feedback_log)
        accepted = sum(1 for f in self.feedback_log if f.accepted)

        # Query type distribution
        type_counts = Counter(f.query_type for f in self.feedback_log)

        # Acceptance rate by type
        acceptance_by_type = defaultdict(lambda: {'total': 0, 'accepted': 0})
        for f in self.feedback_log:
            acceptance_by_type[f.query_type]['total'] += 1
            if f.accepted:
                acceptance_by_type[f.query_type]['accepted'] += 1

        acceptance_rates = {
            qtype: stats['accepted'] / stats['total']
            for qtype, stats in acceptance_by_type.items()
        }

        # Time of day patterns
        hour_counts = Counter(f.timestamp.hour for f in self.feedback_log)
        peak_hour = hour_counts.most_common(1)[0][0] if hour_counts else None

        # Most common queries
        query_counts = Counter(f.query for f in self.feedback_log)
        top_queries = query_counts.most_common(5)

        return {
            'total_queries': total,
            'accepted_queries': accepted,
            'acceptance_rate': accepted / total if total > 0 else 0,
            'query_types': dict(type_counts),
            'acceptance_by_type': acceptance_rates,
            'peak_hour': peak_hour,
            'top_queries': top_queries,
            'avg_results': sum(f.result_count for f in self.feedback_log) / total,
        }

    def get_query_recommendations(self) -> List[str]:
        """
        Get recommended queries based on usage patterns.

        Returns:
            List of query strings to suggest
        """
        if not self.feedback_log:
            return [
                "What should I work on?",
                "What's due this week?",
                "Show me statistics"
            ]

        # Find most accepted queries
        accepted_queries = [f.query for f in self.feedback_log if f.accepted]

        if not accepted_queries:
            # Return most common queries
            query_counts = Counter(f.query for f in self.feedback_log)
            return [q for q, _ in query_counts.most_common(5)]

        # Return most accepted queries
        query_counts = Counter(accepted_queries)
        return [q for q, _ in query_counts.most_common(5)]

    def get_effectiveness_score(self, query_type: str) -> float:
        """
        Get effectiveness score for a query type.

        Effectiveness = acceptance_rate * frequency

        Args:
            query_type: The query type to score

        Returns:
            Score between 0 and 1
        """
        type_queries = [f for f in self.feedback_log if f.query_type == query_type]

        if not type_queries:
            return 0.0

        acceptance_rate = sum(1 for f in type_queries if f.accepted) / len(type_queries)
        frequency = len(type_queries) / len(self.feedback_log)

        # Weighted combination
        return (acceptance_rate * 0.7) + (frequency * 0.3)

    def get_time_pattern_insights(self) -> Dict[str, Any]:
        """
        Analyze time-of-day patterns.

        Returns:
            Insights about when user queries HoloLoom
        """
        if not self.feedback_log:
            return {}

        hour_queries = defaultdict(list)
        for f in self.feedback_log:
            hour_queries[f.timestamp.hour].append(f)

        # Find peak hours
        hour_counts = {h: len(queries) for h, queries in hour_queries.items()}
        sorted_hours = sorted(hour_counts.items(), key=lambda x: -x[1])

        # Find high-acceptance hours
        hour_acceptance = {}
        for hour, queries in hour_queries.items():
            if len(queries) >= 3:  # Minimum sample size
                accepted = sum(1 for q in queries if q.accepted)
                hour_acceptance[hour] = accepted / len(queries)

        best_hours = sorted(hour_acceptance.items(), key=lambda x: -x[1])[:3]

        return {
            'peak_hours': [h for h, _ in sorted_hours[:3]],
            'total_hours_active': len(hour_queries),
            'best_acceptance_hours': [h for h, _ in best_hours],
            'query_distribution': hour_counts,
        }

    def detect_reformulations(self) -> List[tuple]:
        """
        Detect query reformulations (user trying different phrasings).

        Returns:
            List of (original_query, reformulated_query, time_diff_seconds)
        """
        reformulations = []

        sorted_queries = sorted(self.feedback_log, key=lambda f: f.timestamp)

        for i in range(len(sorted_queries) - 1):
            current = sorted_queries[i]
            next_query = sorted_queries[i + 1]

            # Check if queries are similar but different
            if (current.query != next_query.query and
                current.query_type == next_query.query_type):

                time_diff = (next_query.timestamp - current.timestamp).total_seconds()

                # If within 5 minutes, likely a reformulation
                if time_diff < 300:
                    reformulations.append((
                        current.query,
                        next_query.query,
                        time_diff
                    ))

        return reformulations

    def suggest_query_improvements(self) -> List[Dict[str, str]]:
        """
        Suggest improvements to query interface based on learning.

        Returns:
            List of suggestions
        """
        suggestions = []
        stats = self.get_statistics()

        # Low acceptance rate overall
        if stats['acceptance_rate'] < 0.3 and stats['total_queries'] > 10:
            suggestions.append({
                'type': 'low_acceptance',
                'message': 'Acceptance rate is low - consider improving result quality',
                'action': 'Review query handlers and ranking'
            })

        # Specific query type has low acceptance
        for qtype, rate in stats['acceptance_by_type'].items():
            if rate < 0.2 and stats['query_types'].get(qtype, 0) > 5:
                suggestions.append({
                    'type': 'low_type_acceptance',
                    'message': f'Query type "{qtype}" has low acceptance ({rate:.1%})',
                    'action': f'Improve handler for {qtype} queries'
                })

        # Frequent reformulations
        reformulations = self.detect_reformulations()
        if len(reformulations) > 5:
            suggestions.append({
                'type': 'frequent_reformulations',
                'message': f'Detected {len(reformulations)} query reformulations',
                'action': 'Improve query classification or add query suggestions'
            })

        # High use of specific query
        top_queries = stats.get('top_queries', [])
        if top_queries and top_queries[0][1] > stats['total_queries'] * 0.3:
            suggestions.append({
                'type': 'dominant_query',
                'message': f'Query "{top_queries[0][0]}" used {top_queries[0][1]} times',
                'action': 'Consider adding shortcut or improving similar queries'
            })

        return suggestions


def load_learning_data(log_file: Optional[str] = None) -> LearningTracker:
    """
    Load learning data from file.

    Args:
        log_file: Path to learning log (default: ~/.hololoom-learning.jsonl)

    Returns:
        LearningTracker instance with loaded data
    """
    return LearningTracker(log_file)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Create tracker
    tracker = LearningTracker("/tmp/hololoom-test.jsonl")

    # Simulate some queries
    tracker.log_query("What should I work on?", "next_task", 3, accepted=True)
    tracker.log_query("What's due this week?", "deadlines", 5, accepted=True)
    tracker.log_query("Show me statistics", "stats", 1, accepted=False)
    tracker.log_query("What should I work on?", "next_task", 3, accepted=True)
    tracker.log_query("What am I working on?", "current_tasks", 2, accepted=True)

    # Get statistics
    stats = tracker.get_statistics()
    print("Statistics:")
    print(f"  Total queries: {stats['total_queries']}")
    print(f"  Acceptance rate: {stats['acceptance_rate']:.1%}")
    print(f"  Query types: {stats['query_types']}")

    # Get recommendations
    recommendations = tracker.get_query_recommendations()
    print(f"\nRecommended queries:")
    for i, query in enumerate(recommendations, 1):
        print(f"  {i}. {query}")

    # Get insights
    insights = tracker.get_time_pattern_insights()
    print(f"\nTime patterns:")
    print(f"  Peak hours: {insights.get('peak_hours', [])}")

    # Get suggestions
    suggestions = tracker.suggest_query_improvements()
    print(f"\nImprovement suggestions:")
    for s in suggestions:
        print(f"  - {s['message']}")
        print(f"    Action: {s['action']}")
