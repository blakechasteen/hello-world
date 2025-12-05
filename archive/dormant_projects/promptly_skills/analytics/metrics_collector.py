"""
Metrics Collector - Phase 5

Collects performance metrics from strategy execution:
- Query metrics (latency, confidence, cache hits)
- Strategy metrics (usage, success rate, improvement)
- Learning metrics (Thompson Sampling stats, exploration rate)
- System metrics (CPU, memory, error rate)

Features:
- Buffered writes (batch every 5s to reduce DB load)
- Async collection (non-blocking)
- Event hooks (easy integration)
- Type-safe metric events
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics we collect."""
    QUERY = "query"              # Query execution metrics
    STRATEGY = "strategy"         # Strategy-specific metrics
    LEARNING = "learning"         # Learning algorithm metrics
    SYSTEM = "system"            # System health metrics
    CACHE = "cache"              # Cache performance metrics


@dataclass
class MetricEvent:
    """
    A single metric event.

    Examples:
        Query event:
            MetricEvent(
                type=MetricType.QUERY,
                timestamp=time.time(),
                tags={'strategy': 'deep', 'user_id': 'alice'},
                values={'latency_ms': 150, 'confidence': 0.95}
            )

        Cache event:
            MetricEvent(
                type=MetricType.CACHE,
                timestamp=time.time(),
                tags={'cache_type': 'lru'},
                values={'hit': 1, 'miss': 0}
            )
    """
    type: MetricType
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    values: Dict[str, float] = field(default_factory=dict)


class MetricsCollector:
    """
    Collects and buffers metrics before writing to time-series DB.

    Usage:
        collector = MetricsCollector()
        await collector.start()

        # Record query
        await collector.record_query(
            strategy='deep',
            latency_ms=150,
            confidence=0.95,
            cache_hit=True
        )

        # Record strategy performance
        await collector.record_strategy_performance(
            strategy='deep',
            success_rate=0.92,
            avg_confidence=0.91
        )

        await collector.stop()
    """

    def __init__(
        self,
        buffer_size: int = 10000,
        flush_interval: float = 5.0,
        db: Optional[Any] = None
    ):
        """
        Initialize metrics collector.

        Args:
            buffer_size: Maximum events to buffer before forced flush
            flush_interval: Seconds between automatic flushes
            db: Time-series database (will be set later if None)
        """
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.db = db

        # Event buffer (thread-safe deque)
        self.buffer: deque = deque(maxlen=buffer_size)

        # Statistics
        self.total_events = 0
        self.total_flushes = 0
        self.last_flush_time = time.time()

        # Background task
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self):
        """Start background flush task."""
        if self._running:
            logger.warning("MetricsCollector already running")
            return

        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info(f"MetricsCollector started (flush every {self.flush_interval}s)")

    async def stop(self):
        """Stop collector and flush remaining events."""
        if not self._running:
            return

        self._running = False

        # Cancel flush task
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Final flush
        await self._flush()
        logger.info("MetricsCollector stopped")

    async def record_query(
        self,
        query: str,
        strategy: str,
        latency_ms: float,
        confidence: float,
        cache_hit: bool = False,
        user_id: Optional[str] = None,
        **extra_tags
    ):
        """
        Record a query execution.

        Args:
            query: The query text
            strategy: Strategy used
            latency_ms: Execution time in milliseconds
            confidence: Confidence score (0-1)
            cache_hit: Whether result came from cache
            user_id: Optional user identifier
            **extra_tags: Additional tags
        """
        event = MetricEvent(
            type=MetricType.QUERY,
            timestamp=time.time(),
            tags={
                'strategy': strategy,
                'cache_hit': str(cache_hit),
                'user_id': user_id or 'anonymous',
                **extra_tags
            },
            values={
                'latency_ms': latency_ms,
                'confidence': confidence,
                'query_length': len(query),
                'cache_hit_binary': 1.0 if cache_hit else 0.0
            }
        )

        await self._add_event(event)

    async def record_strategy_performance(
        self,
        strategy: str,
        success_rate: float,
        avg_confidence: float,
        avg_latency_ms: float,
        total_uses: int,
        **extra_values
    ):
        """
        Record strategy performance metrics.

        Args:
            strategy: Strategy name
            success_rate: Percentage of successful queries (confidence ≥ 0.75)
            avg_confidence: Average confidence score
            avg_latency_ms: Average latency
            total_uses: Total number of times used
            **extra_values: Additional metric values
        """
        event = MetricEvent(
            type=MetricType.STRATEGY,
            timestamp=time.time(),
            tags={'strategy': strategy},
            values={
                'success_rate': success_rate,
                'avg_confidence': avg_confidence,
                'avg_latency_ms': avg_latency_ms,
                'total_uses': float(total_uses),
                **extra_values
            }
        )

        await self._add_event(event)

    async def record_learning_update(
        self,
        strategy: str,
        alpha: float,
        beta: float,
        expected_reward: float,
        exploration_rate: float
    ):
        """
        Record Thompson Sampling learning update.

        Args:
            strategy: Strategy name
            alpha: Thompson Sampling alpha parameter
            beta: Thompson Sampling beta parameter
            expected_reward: Expected reward (α / (α + β))
            exploration_rate: Current exploration rate
        """
        event = MetricEvent(
            type=MetricType.LEARNING,
            timestamp=time.time(),
            tags={'strategy': strategy, 'algorithm': 'thompson_sampling'},
            values={
                'alpha': alpha,
                'beta': beta,
                'expected_reward': expected_reward,
                'exploration_rate': exploration_rate
            }
        )

        await self._add_event(event)

    async def record_cache_stats(
        self,
        cache_type: str,
        hit_rate: float,
        total_hits: int,
        total_misses: int,
        size: int,
        capacity: int
    ):
        """
        Record cache performance.

        Args:
            cache_type: Type of cache (lru, fifo, etc.)
            hit_rate: Cache hit rate (0-1)
            total_hits: Total cache hits
            total_misses: Total cache misses
            size: Current cache size
            capacity: Maximum cache capacity
        """
        event = MetricEvent(
            type=MetricType.CACHE,
            timestamp=time.time(),
            tags={'cache_type': cache_type},
            values={
                'hit_rate': hit_rate,
                'total_hits': float(total_hits),
                'total_misses': float(total_misses),
                'size': float(size),
                'capacity': float(capacity),
                'utilization': size / capacity if capacity > 0 else 0.0
            }
        )

        await self._add_event(event)

    async def record_system_metrics(
        self,
        cpu_percent: float,
        memory_mb: float,
        memory_percent: float,
        active_connections: int,
        error_rate: float
    ):
        """
        Record system health metrics.

        Args:
            cpu_percent: CPU usage percentage
            memory_mb: Memory usage in MB
            memory_percent: Memory usage percentage
            active_connections: Number of active connections
            error_rate: Error rate (0-1)
        """
        event = MetricEvent(
            type=MetricType.SYSTEM,
            timestamp=time.time(),
            tags={},
            values={
                'cpu_percent': cpu_percent,
                'memory_mb': memory_mb,
                'memory_percent': memory_percent,
                'active_connections': float(active_connections),
                'error_rate': error_rate
            }
        )

        await self._add_event(event)

    async def _add_event(self, event: MetricEvent):
        """Add event to buffer."""
        self.buffer.append(event)
        self.total_events += 1

        # Force flush if buffer full
        if len(self.buffer) >= self.buffer_size:
            logger.warning(f"Buffer full ({self.buffer_size} events), forcing flush")
            await self._flush()

    async def _flush_loop(self):
        """Background task that flushes buffer periodically."""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in flush loop: {e}")

    async def _flush(self):
        """Flush buffered events to database."""
        if not self.buffer:
            return

        if not self.db:
            logger.warning("No database configured, discarding events")
            self.buffer.clear()
            return

        # Get events from buffer
        events = list(self.buffer)
        self.buffer.clear()

        try:
            # Write to database
            await self.db.write_batch(events)

            self.total_flushes += 1
            self.last_flush_time = time.time()

            logger.debug(f"Flushed {len(events)} events to database")

        except Exception as e:
            logger.error(f"Error flushing events: {e}")
            # Re-add events to buffer
            self.buffer.extend(events)

    def get_stats(self) -> Dict[str, Any]:
        """Get collector statistics."""
        return {
            'total_events': self.total_events,
            'total_flushes': self.total_flushes,
            'buffer_size': len(self.buffer),
            'buffer_capacity': self.buffer_size,
            'buffer_utilization': len(self.buffer) / self.buffer_size,
            'last_flush_time': self.last_flush_time,
            'seconds_since_last_flush': time.time() - self.last_flush_time,
            'running': self._running
        }


# Global singleton instance
_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector


def set_metrics_collector(collector: MetricsCollector):
    """Set global metrics collector instance."""
    global _collector
    _collector = collector
