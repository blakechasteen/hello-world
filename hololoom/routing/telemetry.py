"""
Classification Telemetry System
================================
Tracks classifier performance in production for continuous improvement.

Features:
- Misclassification tracking
- Daily log rotation
- Export for analysis
- Integration with adaptive learning

Author: Claude Code
Date: 2025-11-13
"""

import json
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path


class ClassificationTelemetry:
    """
    Production telemetry for query classification.

    Logs every classification with:
    - Query text
    - Predicted complexity
    - Actual complexity (from feedback)
    - Confidence score
    - Tier used
    - Latency
    - Timestamp

    Enables:
    - Misclassification analysis
    - Pattern mining for adaptive learning
    - Performance monitoring
    """

    def __init__(self, log_path: str = "./classification_logs"):
        """
        Initialize telemetry system.

        Args:
            log_path: Directory for log files (daily rotation)
        """
        self.log_path = Path(log_path)
        self.log_path.mkdir(parents=True, exist_ok=True)

        # In-memory buffer (flush every 100 entries or 60 seconds)
        self.buffer = []
        self.buffer_size_limit = 100
        self.last_flush_time = time.time()
        self.flush_interval = 60.0  # seconds

        # Statistics
        self.misclassifications = []
        self.total_classifications = 0
        self.correct_classifications = 0

    async def log_classification(
        self,
        query: str,
        predicted: str,
        actual: str | None = None,
        confidence: float = 0.0,
        tier_used: str = "unknown",
        latency_ms: float = 0.0,
        user_feedback: str | None = None,
        metadata: dict | None = None
    ):
        """
        Log a classification result.

        Args:
            query: Query text
            predicted: Predicted complexity (TRIVIAL/SIMPLE/COMPLEX/RESEARCH)
            actual: Actual complexity (from user feedback, optional)
            confidence: Classification confidence (0-1)
            tier_used: Which tier made the decision (tier1_pattern/tier2_heuristic/tier3_semantic/tier4_conservative)
            latency_ms: Classification latency in milliseconds
            user_feedback: Optional user feedback string
            metadata: Optional additional metadata dict
        """
        timestamp = time.time()

        entry = {
            'query': query,
            'predicted': predicted,
            'actual': actual,
            'confidence': confidence,
            'tier_used': tier_used,
            'latency_ms': latency_ms,
            'user_feedback': user_feedback,
            'metadata': metadata or {},
            'timestamp': timestamp,
            'date': date.today().isoformat(),
            'datetime': datetime.now().isoformat()
        }

        # Add to buffer
        self.buffer.append(entry)

        # Track statistics
        self.total_classifications += 1

        if actual is not None:
            if predicted == actual:
                self.correct_classifications += 1
            else:
                self.misclassifications.append(entry)

        # Auto-flush if buffer full or interval exceeded
        if (len(self.buffer) >= self.buffer_size_limit or
                time.time() - self.last_flush_time >= self.flush_interval):
            await self.flush()

    async def flush(self):
        """Flush buffer to disk (daily log file)."""
        if not self.buffer:
            return

        # Write to daily log file
        log_file = self.log_path / f"classifications_{date.today()}.jsonl"

        with open(log_file, 'a') as f:
            for entry in self.buffer:
                json.dump(entry, f)
                f.write('\n')

        # Clear buffer
        self.buffer.clear()
        self.last_flush_time = time.time()

    def get_misclassifications(
        self,
        limit: int | None = None,
        since_days: int | None = None
    ) -> list[dict]:
        """
        Get recent misclassifications.

        Args:
            limit: Maximum number to return (most recent first)
            since_days: Only return misclassifications from last N days

        Returns:
            List of misclassification entries
        """
        misses = self.misclassifications

        # Filter by date
        if since_days is not None:
            cutoff = time.time() - (since_days * 86400)
            misses = [m for m in misses if m['timestamp'] >= cutoff]

        # Sort by timestamp (most recent first)
        misses = sorted(misses, key=lambda x: x['timestamp'], reverse=True)

        # Limit
        if limit is not None:
            misses = misses[:limit]

        return misses

    def get_accuracy(self, since_days: int | None = None) -> float:
        """
        Get classification accuracy.

        Args:
            since_days: Calculate accuracy for last N days only

        Returns:
            Accuracy (0-1)
        """
        if self.total_classifications == 0:
            return 0.0

        if since_days is None:
            return self.correct_classifications / self.total_classifications

        # Load recent logs
        cutoff_date = date.today() - timedelta(days=since_days)
        total = 0
        correct = 0

        for day in range(since_days + 1):
            log_date = date.today() - timedelta(days=day)
            if log_date < cutoff_date:
                break

            log_file = self.log_path / f"classifications_{log_date}.jsonl"
            if not log_file.exists():
                continue

            with open(log_file) as f:
                for line in f:
                    entry = json.loads(line)
                    if entry.get('actual') is not None:
                        total += 1
                        if entry['predicted'] == entry['actual']:
                            correct += 1

        return correct / total if total > 0 else 0.0

    def get_tier_distribution(self, since_days: int | None = None) -> dict[str, int]:
        """
        Get distribution of queries by tier.

        Args:
            since_days: Last N days only

        Returns:
            Dict mapping tier name to count
        """
        tier_counts = defaultdict(int)

        if since_days is None:
            # Use in-memory statistics
            for entry in self.misclassifications + self.buffer:
                tier_counts[entry['tier_used']] += 1
            return dict(tier_counts)

        # Load recent logs
        for day in range(since_days + 1):
            log_date = date.today() - timedelta(days=day)
            log_file = self.log_path / f"classifications_{log_date}.jsonl"

            if not log_file.exists():
                continue

            with open(log_file) as f:
                for line in f:
                    entry = json.loads(line)
                    tier_counts[entry['tier_used']] += 1

        return dict(tier_counts)

    def get_latency_by_tier(self, since_days: int | None = None) -> dict[str, dict[str, float]]:
        """
        Get latency statistics by tier.

        Args:
            since_days: Last N days only

        Returns:
            Dict mapping tier to {avg_ms, min_ms, max_ms, p50_ms, p95_ms, p99_ms}
        """
        tier_latencies = defaultdict(list)

        # Load recent logs
        if since_days is not None:
            for day in range(since_days + 1):
                log_date = date.today() - timedelta(days=day)
                log_file = self.log_path / f"classifications_{log_date}.jsonl"

                if not log_file.exists():
                    continue

                with open(log_file) as f:
                    for line in f:
                        entry = json.loads(line)
                        tier_latencies[entry['tier_used']].append(entry['latency_ms'])
        else:
            # Use in-memory buffer
            for entry in self.buffer:
                tier_latencies[entry['tier_used']].append(entry['latency_ms'])

        # Calculate statistics
        stats = {}
        for tier, latencies in tier_latencies.items():
            if not latencies:
                continue

            latencies_sorted = sorted(latencies)
            n = len(latencies_sorted)

            stats[tier] = {
                'avg_ms': sum(latencies) / n,
                'min_ms': min(latencies),
                'max_ms': max(latencies),
                'p50_ms': latencies_sorted[int(n * 0.50)],
                'p95_ms': latencies_sorted[int(n * 0.95)],
                'p99_ms': latencies_sorted[int(n * 0.99)],
                'count': n
            }

        return stats

    def export_for_learning(
        self,
        output_file: str,
        min_confidence_delta: float = 0.2,
        since_days: int | None = 7
    ) -> int:
        """
        Export misclassifications for adaptive learning.

        Args:
            output_file: Path to export JSON file
            min_confidence_delta: Only export if confidence was high but wrong
            since_days: Last N days to export

        Returns:
            Number of entries exported
        """
        misses = self.get_misclassifications(since_days=since_days)

        # Filter by confidence delta (high confidence but wrong = learning opportunity)
        learning_opportunities = [
            m for m in misses
            if m['confidence'] >= min_confidence_delta
        ]

        # Export
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump({
                'exported_at': datetime.now().isoformat(),
                'total_misclassifications': len(misses),
                'high_confidence_misses': len(learning_opportunities),
                'min_confidence_delta': min_confidence_delta,
                'since_days': since_days,
                'entries': learning_opportunities
            }, f, indent=2)

        return len(learning_opportunities)

    def get_stats_summary(self, since_days: int | None = None) -> dict:
        """
        Get comprehensive statistics summary.

        Args:
            since_days: Last N days only

        Returns:
            Dict with all statistics
        """
        return {
            'accuracy': self.get_accuracy(since_days=since_days),
            'total_classifications': self.total_classifications,
            'correct_classifications': self.correct_classifications,
            'misclassifications_count': len(self.misclassifications),
            'tier_distribution': self.get_tier_distribution(since_days=since_days),
            'latency_by_tier': self.get_latency_by_tier(since_days=since_days),
            'top_10_misses': self.get_misclassifications(limit=10, since_days=since_days)
        }


# Singleton instance for global access
_telemetry_instance: ClassificationTelemetry | None = None


def get_telemetry(log_path: str = "./classification_logs") -> ClassificationTelemetry:
    """
    Get global telemetry instance (singleton pattern).

    Args:
        log_path: Directory for log files

    Returns:
        ClassificationTelemetry instance
    """
    global _telemetry_instance
    if _telemetry_instance is None:
        _telemetry_instance = ClassificationTelemetry(log_path=log_path)
    return _telemetry_instance
