"""
Routing Metrics Collection

Tracks query -> backend -> performance mappings for learned routing.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class RoutingMetrics:
    """Single routing decision and its outcome."""

    query: str
    query_type: str  # 'factual', 'analytical', 'creative', 'conversational'
    complexity: str  # 'LITE', 'FAST', 'FULL', 'RESEARCH'
    backend_selected: str  # 'HYBRID', 'NEO4J_QDRANT', 'NETWORKX', 'HYPERSPACE'

    # Performance metrics
    latency_ms: float
    relevance_score: float  # 0.0-1.0
    confidence: float  # 0.0-1.0
    success: bool

    # Context
    timestamp: datetime = field(default_factory=datetime.now)
    memory_size: int = 0  # Number of shards in memory

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'query': self.query,
            'query_type': self.query_type,
            'complexity': self.complexity,
            'backend_selected': self.backend_selected,
            'latency_ms': self.latency_ms,
            'relevance_score': self.relevance_score,
            'confidence': self.confidence,
            'success': self.success,
            'timestamp': self.timestamp.isoformat(),
            'memory_size': self.memory_size
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'RoutingMetrics':
        """Create from dictionary."""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class MetricsCollector:
    """Collects and stores routing metrics for training."""

    def __init__(self, storage_path: Path | None = None):
        """
        Initialize metrics collector.
        
        Args:
            storage_path: Path to store metrics (default: HoloLoom/routing/metrics.jsonl)
        """
        self.storage_path = storage_path or Path(__file__).parent / 'metrics.jsonl'
        self.metrics: list[RoutingMetrics] = []

        # Load existing metrics if available
        self._load_metrics()

    def record(self, metrics: RoutingMetrics):
        """Record a routing decision and its outcome."""
        self.metrics.append(metrics)

        # Append to storage file
        with open(self.storage_path, 'a') as f:
            f.write(json.dumps(metrics.to_dict()) + '\n')

    def _load_metrics(self):
        """Load existing metrics from storage."""
        if not self.storage_path.exists():
            return

        with open(self.storage_path) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    self.metrics.append(RoutingMetrics.from_dict(data))

    def get_metrics(
        self,
        query_type: str | None = None,
        complexity: str | None = None,
        backend: str | None = None,
        min_timestamp: datetime | None = None
    ) -> list[RoutingMetrics]:
        """
        Get filtered metrics.
        
        Args:
            query_type: Filter by query type
            complexity: Filter by complexity level
            backend: Filter by backend
            min_timestamp: Filter by minimum timestamp
        
        Returns:
            Filtered list of metrics
        """
        filtered = self.metrics

        if query_type:
            filtered = [m for m in filtered if m.query_type == query_type]

        if complexity:
            filtered = [m for m in filtered if m.complexity == complexity]

        if backend:
            filtered = [m for m in filtered if m.backend_selected == backend]

        if min_timestamp:
            filtered = [m for m in filtered if m.timestamp >= min_timestamp]

        return filtered

    def get_backend_stats(self, backend: str) -> dict:
        """Get statistics for a specific backend."""
        backend_metrics = self.get_metrics(backend=backend)

        if not backend_metrics:
            return {
                'count': 0,
                'avg_latency_ms': 0.0,
                'avg_relevance': 0.0,
                'avg_confidence': 0.0,
                'success_rate': 0.0
            }

        return {
            'count': len(backend_metrics),
            'avg_latency_ms': sum(m.latency_ms for m in backend_metrics) / len(backend_metrics),
            'avg_relevance': sum(m.relevance_score for m in backend_metrics) / len(backend_metrics),
            'avg_confidence': sum(m.confidence for m in backend_metrics) / len(backend_metrics),
            'success_rate': sum(1 for m in backend_metrics if m.success) / len(backend_metrics)
        }

    def get_all_stats(self) -> dict[str, dict]:
        """Get statistics for all backends."""
        backends = set(m.backend_selected for m in self.metrics)
        return {backend: self.get_backend_stats(backend) for backend in backends}

    def clear(self):
        """Clear all metrics (for testing)."""
        self.metrics.clear()
        if self.storage_path.exists():
            self.storage_path.unlink()
