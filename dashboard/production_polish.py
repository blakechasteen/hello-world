#!/usr/bin/env python3
"""
🎨 PRODUCTION POLISH & PERFORMANCE
===================================
Final layer of production-grade features for Next-Level Query API.

Features:
- 🛡️ Security (rate limiting, input validation, authentication)
- ⚡ Performance (compression, pooling, caching strategies)
- 📊 Monitoring (health checks, metrics, structured logging)
- 🎨 Quality of Life (bookmarks, export, comparisons)
- 🚀 Auto-Tuning (AI recommendations, performance optimization)
"""

import time
import hashlib
import gzip
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
import json
import logging
from pathlib import Path


# ============================================================================
# SECURITY & VALIDATION
# ============================================================================

class RateLimiter:
    """
    Token bucket rate limiter.

    Prevents abuse by limiting requests per time window.
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_size: int = 10
    ):
        self.rpm = requests_per_minute
        self.rph = requests_per_hour
        self.burst_size = burst_size

        # Per-client tracking
        self.client_buckets: Dict[str, Dict] = defaultdict(lambda: {
            'tokens': burst_size,
            'last_update': time.time(),
            'minute_count': 0,
            'minute_reset': time.time() + 60,
            'hour_count': 0,
            'hour_reset': time.time() + 3600
        })

    def _get_client_id(self, request: Request) -> str:
        """Get unique client identifier."""
        # Try API key first
        api_key = request.headers.get('X-API-Key')
        if api_key:
            return f"key_{hashlib.md5(api_key.encode()).hexdigest()[:8]}"

        # Fall back to IP
        client_ip = request.client.host
        return f"ip_{client_ip}"

    def check_rate_limit(self, request: Request) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is within rate limits.

        Returns:
            (allowed, info_dict)
        """
        client_id = self._get_client_id(request)
        bucket = self.client_buckets[client_id]
        now = time.time()

        # Reset minute counter
        if now > bucket['minute_reset']:
            bucket['minute_count'] = 0
            bucket['minute_reset'] = now + 60

        # Reset hour counter
        if now > bucket['hour_reset']:
            bucket['hour_count'] = 0
            bucket['hour_reset'] = now + 3600

        # Check limits
        if bucket['minute_count'] >= self.rpm:
            return False, {
                'limit_type': 'per_minute',
                'limit': self.rpm,
                'retry_after': int(bucket['minute_reset'] - now)
            }

        if bucket['hour_count'] >= self.rph:
            return False, {
                'limit_type': 'per_hour',
                'limit': self.rph,
                'retry_after': int(bucket['hour_reset'] - now)
            }

        # Refill tokens (token bucket)
        time_passed = now - bucket['last_update']
        tokens_to_add = (time_passed / 60.0) * (self.rpm / 60.0)
        bucket['tokens'] = min(self.burst_size, bucket['tokens'] + tokens_to_add)
        bucket['last_update'] = now

        # Check burst
        if bucket['tokens'] < 1:
            return False, {
                'limit_type': 'burst',
                'limit': self.burst_size,
                'retry_after': 1
            }

        # Consume token
        bucket['tokens'] -= 1
        bucket['minute_count'] += 1
        bucket['hour_count'] += 1

        return True, {
            'remaining_tokens': bucket['tokens'],
            'minute_remaining': self.rpm - bucket['minute_count'],
            'hour_remaining': self.rph - bucket['hour_count']
        }


class InputValidator:
    """Validate and sanitize user inputs."""

    @staticmethod
    def validate_query_text(text: str) -> Tuple[bool, Optional[str]]:
        """Validate query text."""
        if not text or not text.strip():
            return False, "Query text cannot be empty"

        if len(text) > 10000:
            return False, "Query text too long (max 10,000 characters)"

        # Check for suspicious patterns
        suspicious = ['<script', 'javascript:', 'onerror=', 'onclick=']
        text_lower = text.lower()
        for pattern in suspicious:
            if pattern in text_lower:
                return False, f"Suspicious pattern detected: {pattern}"

        return True, None

    @staticmethod
    def sanitize_text(text: str) -> str:
        """Sanitize text input."""
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')

        # Trim whitespace
        text = text.strip()

        return text


# ============================================================================
# PERFORMANCE OPTIMIZATIONS
# ============================================================================

class ResponseCompressor:
    """Compress responses for faster transmission."""

    @staticmethod
    def should_compress(request: Request) -> bool:
        """Check if client accepts gzip."""
        accept_encoding = request.headers.get('accept-encoding', '')
        return 'gzip' in accept_encoding.lower()

    @staticmethod
    def compress_response(content: str) -> bytes:
        """Compress response content."""
        return gzip.compress(content.encode('utf-8'))

    @staticmethod
    def create_compressed_response(data: Dict, request: Request) -> JSONResponse:
        """Create compressed JSON response if supported."""
        if ResponseCompressor.should_compress(request):
            json_str = json.dumps(data)
            compressed = ResponseCompressor.compress_response(json_str)
            return JSONResponse(
                content=compressed,
                headers={
                    'Content-Encoding': 'gzip',
                    'Content-Type': 'application/json'
                }
            )
        return JSONResponse(content=data)


class ConnectionPool:
    """
    Connection pooling for database/external services.

    Reuses connections instead of creating new ones.
    """

    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.pool: deque = deque(maxlen=max_size)
        self.in_use: set = set()
        self.stats = {
            'created': 0,
            'reused': 0,
            'closed': 0
        }

    def acquire(self):
        """Acquire connection from pool."""
        if self.pool:
            conn = self.pool.pop()
            self.in_use.add(id(conn))
            self.stats['reused'] += 1
            return conn

        # Create new connection
        conn = self._create_connection()
        self.in_use.add(id(conn))
        self.stats['created'] += 1
        return conn

    def release(self, conn):
        """Release connection back to pool."""
        conn_id = id(conn)
        if conn_id in self.in_use:
            self.in_use.remove(conn_id)
            self.pool.append(conn)

    def _create_connection(self):
        """Create new connection (override in subclass)."""
        return object()  # Placeholder


# ============================================================================
# PRODUCTION MONITORING
# ============================================================================

class HealthChecker:
    """Production health and readiness checks."""

    def __init__(self):
        self.checks = {}
        self.last_check_time = {}
        self.check_cache_ttl = 5  # Cache health check results for 5 seconds

    def register_check(self, name: str, check_func: callable):
        """Register a health check."""
        self.checks[name] = check_func

    async def check_health(self, force: bool = False) -> Dict[str, Any]:
        """Run all health checks."""
        results = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'checks': {}
        }

        for name, check_func in self.checks.items():
            # Use cache if available and fresh
            if not force and name in self.last_check_time:
                if time.time() - self.last_check_time[name] < self.check_cache_ttl:
                    continue

            try:
                result = await check_func() if asyncio.iscoroutinefunction(check_func) else check_func()
                results['checks'][name] = {
                    'status': 'pass',
                    'details': result
                }
                self.last_check_time[name] = time.time()
            except Exception as e:
                results['checks'][name] = {
                    'status': 'fail',
                    'error': str(e)
                }
                results['status'] = 'unhealthy'

        return results


class MetricsCollector:
    """
    Prometheus-style metrics collection.

    Exports metrics for monitoring systems.
    """

    def __init__(self):
        self.counters = defaultdict(int)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)
        self.start_time = time.time()

    def increment(self, metric: str, value: int = 1, labels: Dict = None):
        """Increment a counter."""
        key = self._make_key(metric, labels)
        self.counters[key] += value

    def set_gauge(self, metric: str, value: float, labels: Dict = None):
        """Set a gauge value."""
        key = self._make_key(metric, labels)
        self.gauges[key] = value

    def observe(self, metric: str, value: float, labels: Dict = None):
        """Add observation to histogram."""
        key = self._make_key(metric, labels)
        self.histograms[key].append(value)

        # Keep last 1000 observations
        if len(self.histograms[key]) > 1000:
            self.histograms[key] = self.histograms[key][-1000:]

    def _make_key(self, metric: str, labels: Dict = None) -> str:
        """Create metric key with labels."""
        if not labels:
            return metric
        label_str = ','.join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{metric}{{{label_str}}}"

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        # Counters
        for key, value in self.counters.items():
            lines.append(f"{key} {value}")

        # Gauges
        for key, value in self.gauges.items():
            lines.append(f"{key} {value}")

        # Histograms (summary statistics)
        for key, values in self.histograms.items():
            if values:
                import statistics
                lines.append(f"{key}_count {len(values)}")
                lines.append(f"{key}_sum {sum(values)}")
                lines.append(f"{key}_avg {statistics.mean(values)}")
                lines.append(f"{key}_p50 {statistics.median(values)}")
                if len(values) > 1:
                    lines.append(f"{key}_p95 {statistics.quantiles(values, n=20)[18]}")
                    lines.append(f"{key}_p99 {statistics.quantiles(values, n=100)[98]}")

        # Uptime
        uptime = time.time() - self.start_time
        lines.append(f"process_uptime_seconds {uptime}")

        return '\n'.join(lines)


class StructuredLogger:
    """Structured logging with context."""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # JSON formatter
        formatter = logging.Formatter(
            '{"timestamp":"%(asctime)s","level":"%(levelname)s","message":"%(message)s","extra":%(extra)s}'
        )

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def log(self, level: str, message: str, **kwargs):
        """Log with structured context."""
        extra_json = json.dumps(kwargs) if kwargs else '{}'
        getattr(self.logger, level)(message, extra={'extra': extra_json})


# ============================================================================
# QUALITY OF LIFE FEATURES
# ============================================================================

@dataclass
class QueryBookmark:
    """Bookmarked query for later use."""
    id: str
    name: str
    query_text: str
    pattern: str
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    used_count: int = 0
    avg_confidence: float = 0.0


class BookmarkManager:
    """Manage query bookmarks/favorites."""

    def __init__(self):
        self.bookmarks: Dict[str, QueryBookmark] = {}

    def add(self, name: str, query_text: str, pattern: str = "fast", tags: List[str] = None) -> str:
        """Add bookmark."""
        bookmark_id = f"bm_{int(time.time() * 1000)}"
        self.bookmarks[bookmark_id] = QueryBookmark(
            id=bookmark_id,
            name=name,
            query_text=query_text,
            pattern=pattern,
            tags=tags or []
        )
        return bookmark_id

    def get(self, bookmark_id: str) -> Optional[QueryBookmark]:
        """Get bookmark by ID."""
        return self.bookmarks.get(bookmark_id)

    def search(self, query: str = "", tags: List[str] = None) -> List[QueryBookmark]:
        """Search bookmarks."""
        results = []
        for bm in self.bookmarks.values():
            # Text search
            if query and query.lower() not in bm.name.lower() and query.lower() not in bm.query_text.lower():
                continue

            # Tag filter
            if tags and not any(tag in bm.tags for tag in tags):
                continue

            results.append(bm)

        return sorted(results, key=lambda x: x.used_count, reverse=True)

    def use(self, bookmark_id: str, confidence: float):
        """Record bookmark usage."""
        if bookmark_id in self.bookmarks:
            bm = self.bookmarks[bookmark_id]
            bm.used_count += 1
            # Running average
            bm.avg_confidence = (bm.avg_confidence * (bm.used_count - 1) + confidence) / bm.used_count


class ResultExporter:
    """Export query results in various formats."""

    @staticmethod
    def to_json(data: Dict) -> str:
        """Export as pretty JSON."""
        return json.dumps(data, indent=2, default=str)

    @staticmethod
    def to_csv(data: Dict) -> str:
        """Export as CSV (flattened)."""
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        # Flatten dict
        def flatten(d, parent_key=''):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}.{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten(v, new_key).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        flat = flatten(data)
        writer.writerow(flat.keys())
        writer.writerow(flat.values())

        return output.getvalue()

    @staticmethod
    def to_markdown(data: Dict) -> str:
        """Export as Markdown."""
        md = [f"# Query Result\n"]
        md.append(f"**Query**: {data.get('query_text', 'N/A')}\n")
        md.append(f"**Confidence**: {data.get('confidence', 0):.1%}\n")
        md.append(f"**Tool**: {data.get('tool_used', 'N/A')}\n")
        md.append(f"\n## Response\n")
        md.append(f"{data.get('response', 'N/A')}\n")

        if 'trace' in data and data['trace']:
            md.append(f"\n## Performance\n")
            md.append(f"- Duration: {data['trace'].get('duration_ms', 0):.1f}ms\n")
            md.append(f"- Stages: {len(data['trace'].get('stage_durations', {}))}\n")

        return ''.join(md)


class QueryComparator:
    """Compare multiple query results."""

    @staticmethod
    def compare(results: List[Dict]) -> Dict[str, Any]:
        """Compare query results."""
        if not results:
            return {}

        comparison = {
            'count': len(results),
            'metrics': {
                'avg_confidence': sum(r.get('confidence', 0) for r in results) / len(results),
                'avg_duration_ms': sum(r.get('duration_ms', 0) for r in results) / len(results),
                'max_confidence': max(r.get('confidence', 0) for r in results),
                'min_confidence': min(r.get('confidence', 0) for r in results),
            },
            'tools_used': {},
            'patterns_used': {}
        }

        # Count tools and patterns
        for result in results:
            tool = result.get('tool_used', 'unknown')
            pattern = result.get('pattern_used', 'unknown')
            comparison['tools_used'][tool] = comparison['tools_used'].get(tool, 0) + 1
            comparison['patterns_used'][pattern] = comparison['patterns_used'].get(pattern, 0) + 1

        # Best result
        best = max(results, key=lambda r: r.get('confidence', 0))
        comparison['best_result'] = {
            'query_id': best.get('query_id'),
            'confidence': best.get('confidence'),
            'pattern': best.get('pattern_used')
        }

        return comparison


# ============================================================================
# AUTO-TUNING & RECOMMENDATIONS
# ============================================================================

class AutoTuner:
    """AI-powered automatic tuning and recommendations."""

    def __init__(self):
        self.performance_history = []
        self.pattern_success_rates = defaultdict(list)

    def record_result(self, query: str, pattern: str, confidence: float, duration_ms: float):
        """Record query result for learning."""
        self.performance_history.append({
            'query': query,
            'pattern': pattern,
            'confidence': confidence,
            'duration_ms': duration_ms,
            'timestamp': time.time()
        })

        self.pattern_success_rates[pattern].append(confidence)

        # Keep last 1000 results
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]

    def recommend_pattern(self, query: str) -> Dict[str, Any]:
        """Recommend best pattern for query."""
        # Analyze query characteristics
        query_len = len(query.split())
        is_complex = query_len > 15
        has_question = '?' in query

        # Calculate pattern scores
        scores = {}
        for pattern in ['bare', 'fast', 'fused']:
            if pattern in self.pattern_success_rates:
                avg_conf = sum(self.pattern_success_rates[pattern]) / len(self.pattern_success_rates[pattern])
                scores[pattern] = avg_conf
            else:
                scores[pattern] = 0.5  # Default

        # Adjust for complexity
        if is_complex:
            scores['fused'] *= 1.2
            scores['bare'] *= 0.8
        else:
            scores['bare'] *= 1.2
            scores['fused'] *= 0.9

        # Select best
        best_pattern = max(scores, key=scores.get)

        return {
            'recommended_pattern': best_pattern,
            'confidence': scores[best_pattern],
            'reason': 'complex query' if is_complex else 'simple query',
            'all_scores': scores
        }

    def get_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """Get optimization suggestions based on history."""
        suggestions = []

        if not self.performance_history:
            return suggestions

        # Check for slow queries
        recent = self.performance_history[-100:]
        avg_duration = sum(r['duration_ms'] for r in recent) / len(recent)

        if avg_duration > 15:
            suggestions.append({
                'type': 'performance',
                'severity': 'warning',
                'message': f'Average query time is high ({avg_duration:.1f}ms)',
                'recommendation': 'Consider using BARE pattern for simple queries'
            })

        # Check confidence trends
        recent_confidence = [r['confidence'] for r in recent]
        avg_conf = sum(recent_confidence) / len(recent_confidence)

        if avg_conf < 0.7:
            suggestions.append({
                'type': 'quality',
                'severity': 'info',
                'message': f'Average confidence is low ({avg_conf:.1%})',
                'recommendation': 'Consider using FUSED pattern or refining queries'
            })

        # Check cache utilization
        # (Would need cache stats passed in)

        return suggestions


# Initialize global instances
rate_limiter = RateLimiter()
input_validator = InputValidator()
health_checker = HealthChecker()
metrics_collector = MetricsCollector()
bookmark_manager = BookmarkManager()
auto_tuner = AutoTuner()


# ============================================================================
# EXPORT EVERYTHING
# ============================================================================

__all__ = [
    'RateLimiter',
    'InputValidator',
    'ResponseCompressor',
    'ConnectionPool',
    'HealthChecker',
    'MetricsCollector',
    'StructuredLogger',
    'BookmarkManager',
    'ResultExporter',
    'QueryComparator',
    'AutoTuner',
    # Global instances
    'rate_limiter',
    'input_validator',
    'health_checker',
    'metrics_collector',
    'bookmark_manager',
    'auto_tuner'
]
