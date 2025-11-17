"""
Performance Monitoring - Track operation timing, resource utilization, and benchmarks

Provides:
- Operation timing (add, get, eval, etc.)
- Query performance metrics
- Storage backend benchmarks
- Evaluator performance tracking
- Resource utilization (CPU, memory)
"""

import time
import psutil
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import threading
import json
from pathlib import Path


@dataclass
class OperationMetrics:
    """Metrics for a single operation"""
    operation: str
    duration_ms: float
    cpu_percent: float
    memory_mb: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class ResourceSnapshot:
    """Snapshot of resource utilization"""
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class OperationTimer:
    """Context manager for timing operations with resource tracking"""

    def __init__(self, operation: str, monitor: 'PerformanceMonitor', metadata: Dict = None):
        self.operation = operation
        self.monitor = monitor
        self.metadata = metadata or {}
        self.start_time = None
        self.start_cpu = None
        self.start_memory = None

    def __enter__(self):
        """Start timing"""
        self.start_time = time.perf_counter()
        process = psutil.Process()
        self.start_cpu = process.cpu_percent()
        self.start_memory = process.memory_info().rss / 1024 / 1024  # MB
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and record metrics"""
        duration_ms = (time.perf_counter() - self.start_time) * 1000
        process = psutil.Process()
        cpu_percent = process.cpu_percent()
        memory_mb = process.memory_info().rss / 1024 / 1024

        metrics = OperationMetrics(
            operation=self.operation,
            duration_ms=duration_ms,
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            metadata=self.metadata
        )

        self.monitor.record_operation(metrics)
        return False  # Don't suppress exceptions


class PerformanceMonitor:
    """Monitors and tracks performance metrics for Promptly operations"""

    def __init__(self, db_path: str, retention_days: int = 30,
                 enable_sampling: bool = True, sample_interval_seconds: int = 60):
        """
        Initialize performance monitor

        Args:
            db_path: Path to metrics database
            retention_days: How long to keep metrics (default 30 days)
            enable_sampling: Whether to sample resource utilization periodically
            sample_interval_seconds: How often to sample resources
        """
        self.db_path = Path(db_path)
        self.retention_days = retention_days
        self.enable_sampling = enable_sampling
        self.sample_interval_seconds = sample_interval_seconds

        # In-memory buffers for recent metrics
        self._operation_buffer = deque(maxlen=1000)
        self._resource_buffer = deque(maxlen=1000)

        # Aggregated statistics
        self._operation_stats = defaultdict(lambda: {
            'count': 0,
            'total_ms': 0.0,
            'min_ms': float('inf'),
            'max_ms': 0.0,
            'avg_ms': 0.0
        })

        self._lock = threading.Lock()
        self._sampling_thread = None
        self._stop_sampling = threading.Event()

        # Initialize database
        self._init_db()

        # Start resource sampling if enabled
        if self.enable_sampling:
            self.start_sampling()

    def _init_db(self):
        """Initialize metrics database"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Operation metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS operation_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation TEXT NOT NULL,
                duration_ms REAL NOT NULL,
                cpu_percent REAL,
                memory_mb REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        """)

        # Create index on operation and timestamp
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_operation_metrics_operation
            ON operation_metrics(operation)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_operation_metrics_timestamp
            ON operation_metrics(timestamp)
        """)

        # Resource snapshots table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS resource_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cpu_percent REAL,
                memory_mb REAL,
                memory_percent REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create index on timestamp
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_resource_snapshots_timestamp
            ON resource_snapshots(timestamp)
        """)

        # Aggregated statistics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS operation_stats (
                operation TEXT PRIMARY KEY,
                count INTEGER,
                total_ms REAL,
                min_ms REAL,
                max_ms REAL,
                avg_ms REAL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    @contextmanager
    def _get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def record_operation(self, metrics: OperationMetrics):
        """Record operation metrics"""
        with self._lock:
            # Add to buffer
            self._operation_buffer.append(metrics)

            # Update statistics
            stats = self._operation_stats[metrics.operation]
            stats['count'] += 1
            stats['total_ms'] += metrics.duration_ms
            stats['min_ms'] = min(stats['min_ms'], metrics.duration_ms)
            stats['max_ms'] = max(stats['max_ms'], metrics.duration_ms)
            stats['avg_ms'] = stats['total_ms'] / stats['count']

        # Persist to database
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO operation_metrics
                (operation, duration_ms, cpu_percent, memory_mb, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                metrics.operation,
                metrics.duration_ms,
                metrics.cpu_percent,
                metrics.memory_mb,
                metrics.timestamp.isoformat(),
                json.dumps(metrics.metadata)
            ))
            conn.commit()

    def record_resource_snapshot(self, snapshot: ResourceSnapshot):
        """Record resource utilization snapshot"""
        with self._lock:
            self._resource_buffer.append(snapshot)

        # Persist to database
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO resource_snapshots
                (cpu_percent, memory_mb, memory_percent, timestamp)
                VALUES (?, ?, ?, ?)
            """, (
                snapshot.cpu_percent,
                snapshot.memory_mb,
                snapshot.memory_percent,
                snapshot.timestamp.isoformat()
            ))
            conn.commit()

    def time_operation(self, operation: str, metadata: Dict = None) -> OperationTimer:
        """Get a context manager for timing an operation"""
        return OperationTimer(operation, self, metadata)

    def start_sampling(self):
        """Start periodic resource sampling"""
        if self._sampling_thread is not None:
            return  # Already running

        self._stop_sampling.clear()
        self._sampling_thread = threading.Thread(
            target=self._sample_resources,
            daemon=True
        )
        self._sampling_thread.start()

    def stop_sampling(self):
        """Stop periodic resource sampling"""
        if self._sampling_thread is None:
            return

        self._stop_sampling.set()
        self._sampling_thread.join(timeout=5)
        self._sampling_thread = None

    def _sample_resources(self):
        """Background thread for sampling resources"""
        process = psutil.Process()

        while not self._stop_sampling.is_set():
            try:
                cpu_percent = process.cpu_percent(interval=0.1)
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                memory_percent = process.memory_percent()

                snapshot = ResourceSnapshot(
                    cpu_percent=cpu_percent,
                    memory_mb=memory_mb,
                    memory_percent=memory_percent
                )

                self.record_resource_snapshot(snapshot)

            except Exception as e:
                # Log error but continue sampling
                print(f"Error sampling resources: {e}")

            # Wait for next sample
            self._stop_sampling.wait(self.sample_interval_seconds)

    def get_operation_stats(self, operation: str = None) -> Dict:
        """Get aggregated statistics for operations"""
        with self._lock:
            if operation:
                return dict(self._operation_stats.get(operation, {}))
            else:
                return {op: dict(stats) for op, stats in self._operation_stats.items()}

    def get_recent_operations(self, limit: int = 100, operation: str = None) -> List[Dict]:
        """Get recent operation metrics"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if operation:
                cursor.execute("""
                    SELECT operation, duration_ms, cpu_percent, memory_mb, timestamp, metadata
                    FROM operation_metrics
                    WHERE operation = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (operation, limit))
            else:
                cursor.execute("""
                    SELECT operation, duration_ms, cpu_percent, memory_mb, timestamp, metadata
                    FROM operation_metrics
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,))

            results = []
            for row in cursor.fetchall():
                results.append({
                    'operation': row[0],
                    'duration_ms': row[1],
                    'cpu_percent': row[2],
                    'memory_mb': row[3],
                    'timestamp': row[4],
                    'metadata': json.loads(row[5]) if row[5] else {}
                })

            return results

    def get_operation_percentiles(self, operation: str, percentiles: List[float] = None) -> Dict:
        """Calculate percentiles for operation duration"""
        if percentiles is None:
            percentiles = [0.5, 0.75, 0.90, 0.95, 0.99]

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT duration_ms FROM operation_metrics
                WHERE operation = ?
                ORDER BY duration_ms
            """, (operation,))

            durations = [row[0] for row in cursor.fetchall()]

            if not durations:
                return {}

            result = {}
            for p in percentiles:
                idx = int(len(durations) * p)
                result[f'p{int(p * 100)}'] = durations[idx]

            return result

    def get_resource_stats(self, hours: int = 24) -> Dict:
        """Get resource utilization statistics"""
        cutoff = datetime.now() - timedelta(hours=hours)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    AVG(cpu_percent) as avg_cpu,
                    MAX(cpu_percent) as max_cpu,
                    AVG(memory_mb) as avg_memory_mb,
                    MAX(memory_mb) as max_memory_mb,
                    AVG(memory_percent) as avg_memory_percent,
                    MAX(memory_percent) as max_memory_percent
                FROM resource_snapshots
                WHERE timestamp > ?
            """, (cutoff.isoformat(),))

            row = cursor.fetchone()

            if not row or row[0] is None:
                return {}

            return {
                'avg_cpu_percent': row[0],
                'max_cpu_percent': row[1],
                'avg_memory_mb': row[2],
                'max_memory_mb': row[3],
                'avg_memory_percent': row[4],
                'max_memory_percent': row[5]
            }

    def get_throughput(self, operation: str = None, hours: int = 1) -> Dict:
        """Calculate operations per second/minute/hour"""
        cutoff = datetime.now() - timedelta(hours=hours)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            if operation:
                cursor.execute("""
                    SELECT COUNT(*) FROM operation_metrics
                    WHERE operation = ? AND timestamp > ?
                """, (operation, cutoff.isoformat()))
            else:
                cursor.execute("""
                    SELECT COUNT(*) FROM operation_metrics
                    WHERE timestamp > ?
                """, (cutoff.isoformat(),))

            count = cursor.fetchone()[0]

            total_seconds = hours * 3600

            return {
                'total_operations': count,
                'operations_per_second': count / total_seconds,
                'operations_per_minute': count / (total_seconds / 60),
                'operations_per_hour': count / hours
            }

    def cleanup_old_metrics(self):
        """Remove metrics older than retention period"""
        cutoff = datetime.now() - timedelta(days=self.retention_days)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Clean up old operation metrics
            cursor.execute("""
                DELETE FROM operation_metrics
                WHERE timestamp < ?
            """, (cutoff.isoformat(),))

            # Clean up old resource snapshots
            cursor.execute("""
                DELETE FROM resource_snapshots
                WHERE timestamp < ?
            """, (cutoff.isoformat(),))

            deleted = cursor.rowcount
            conn.commit()

            return deleted

    def get_slow_operations(self, threshold_ms: float = 1000, limit: int = 20) -> List[Dict]:
        """Get slowest operations above threshold"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT operation, duration_ms, cpu_percent, memory_mb, timestamp, metadata
                FROM operation_metrics
                WHERE duration_ms > ?
                ORDER BY duration_ms DESC
                LIMIT ?
            """, (threshold_ms, limit))

            results = []
            for row in cursor.fetchall():
                results.append({
                    'operation': row[0],
                    'duration_ms': row[1],
                    'cpu_percent': row[2],
                    'memory_mb': row[3],
                    'timestamp': row[4],
                    'metadata': json.loads(row[5]) if row[5] else {}
                })

            return results

    def benchmark_operation(self, operation: Callable, name: str,
                           iterations: int = 10, metadata: Dict = None) -> Dict:
        """Benchmark an operation by running it multiple times"""
        durations = []

        for i in range(iterations):
            with self.time_operation(name, metadata={'iteration': i, **(metadata or {})}):
                operation()

            # Get the last recorded operation
            with self._lock:
                if self._operation_buffer:
                    durations.append(self._operation_buffer[-1].duration_ms)

        if not durations:
            return {}

        return {
            'iterations': iterations,
            'min_ms': min(durations),
            'max_ms': max(durations),
            'avg_ms': sum(durations) / len(durations),
            'total_ms': sum(durations),
            'durations': durations
        }

    def export_metrics(self, output_path: str, format: str = 'json',
                      start_date: datetime = None, end_date: datetime = None):
        """Export metrics to file"""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=7)
        if end_date is None:
            end_date = datetime.now()

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get operation metrics
            cursor.execute("""
                SELECT operation, duration_ms, cpu_percent, memory_mb, timestamp, metadata
                FROM operation_metrics
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            """, (start_date.isoformat(), end_date.isoformat()))

            operations = []
            for row in cursor.fetchall():
                operations.append({
                    'operation': row[0],
                    'duration_ms': row[1],
                    'cpu_percent': row[2],
                    'memory_mb': row[3],
                    'timestamp': row[4],
                    'metadata': json.loads(row[5]) if row[5] else {}
                })

            # Get resource snapshots
            cursor.execute("""
                SELECT cpu_percent, memory_mb, memory_percent, timestamp
                FROM resource_snapshots
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            """, (start_date.isoformat(), end_date.isoformat()))

            resources = []
            for row in cursor.fetchall():
                resources.append({
                    'cpu_percent': row[0],
                    'memory_mb': row[1],
                    'memory_percent': row[2],
                    'timestamp': row[3]
                })

        export_data = {
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'operations': operations,
            'resources': resources,
            'stats': self.get_operation_stats()
        }

        output_path = Path(output_path)

        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
        elif format == 'csv':
            import csv

            # Export operations to CSV
            ops_path = output_path.parent / f"{output_path.stem}_operations.csv"
            with open(ops_path, 'w', newline='') as f:
                if operations:
                    writer = csv.DictWriter(f, fieldnames=operations[0].keys())
                    writer.writeheader()
                    writer.writerows(operations)

            # Export resources to CSV
            res_path = output_path.parent / f"{output_path.stem}_resources.csv"
            with open(res_path, 'w', newline='') as f:
                if resources:
                    writer = csv.DictWriter(f, fieldnames=resources[0].keys())
                    writer.writeheader()
                    writer.writerows(resources)

        return str(output_path)

    def __del__(self):
        """Cleanup on deletion"""
        self.stop_sampling()
