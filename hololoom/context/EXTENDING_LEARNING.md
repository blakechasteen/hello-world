# Extending the Learning System

A guide to customizing and extending Part 4: Learning Mechanisms.

## 🎯 Extension Points

The learning system has 5 main extension points:

1. **Custom Metrics** - Track additional performance indicators
2. **Learning Callbacks** - React to learning events
3. **Strategy Rules** - Add new adaptation strategies
4. **Calibration Methods** - Customize confidence adjustment
5. **External Integration** - Connect to monitoring systems

---

## 1. Adding Custom Metrics

### Track Additional Performance Indicators

**Example: Track query complexity**

```python
from hololoom.context import LearningTracker, RoutingEvent
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class ExtendedRoutingEvent(RoutingEvent):
    """Add custom fields to routing events"""
    query_complexity: float = 0.0  # Custom metric
    num_joins: int = 0
    result_size_bytes: int = 0

class ExtendedLearningTracker(LearningTracker):
    """Learning tracker with custom metrics"""

    def __init__(self, reflection_buffer=None):
        super().__init__(reflection_buffer)
        self.complexity_history = []

    async def record_routing(
        self,
        session_id: str,
        query: str,
        backend: str,
        predicted_confidence: float,
        actual_confidence: float,
        latency_ms: float,
        cache_hit: bool = False,
        fallback_used: bool = False,
        # Custom parameters
        query_complexity: float = 0.0,
        num_joins: int = 0,
        result_size_bytes: int = 0
    ):
        # Call parent
        await super().record_routing(
            session_id, query, backend,
            predicted_confidence, actual_confidence,
            latency_ms, cache_hit, fallback_used
        )

        # Track custom metrics
        self.complexity_history.append({
            "complexity": query_complexity,
            "num_joins": num_joins,
            "result_size_bytes": result_size_bytes,
            "latency_ms": latency_ms,
            "backend": backend
        })

    def get_complexity_correlation(self) -> Dict[str, float]:
        """Analyze complexity vs latency correlation"""
        if not self.complexity_history:
            return {}

        # Simple correlation calculation
        complexities = [h["complexity"] for h in self.complexity_history]
        latencies = [h["latency_ms"] for h in self.complexity_history]

        # Pearson correlation
        import statistics
        if len(complexities) < 2:
            return {"correlation": 0.0}

        mean_c = statistics.mean(complexities)
        mean_l = statistics.mean(latencies)

        numerator = sum((c - mean_c) * (l - mean_l)
                       for c, l in zip(complexities, latencies))
        denom_c = sum((c - mean_c) ** 2 for c in complexities) ** 0.5
        denom_l = sum((l - mean_l) ** 2 for l in latencies) ** 0.5

        if denom_c == 0 or denom_l == 0:
            correlation = 0.0
        else:
            correlation = numerator / (denom_c * denom_l)

        return {
            "correlation": correlation,
            "avg_complexity": mean_c,
            "avg_latency": mean_l
        }

# Usage
tracker = ExtendedLearningTracker()
await tracker.record_routing(
    session_id="s1",
    query="SELECT * FROM policies p JOIN rules r ON p.id = r.policy_id",
    backend="sql",
    predicted_confidence=0.85,
    actual_confidence=0.90,
    latency_ms=45.0,
    query_complexity=0.7,  # Custom!
    num_joins=1,           # Custom!
    result_size_bytes=2048 # Custom!
)

# Analyze
stats = tracker.get_complexity_correlation()
print(f"Complexity-latency correlation: {stats['correlation']:.2f}")
```

---

## 2. Learning Callbacks

### React to Learning Events

**Example: Alert on performance degradation**

```python
from hololoom.context import StrategyUpdater
from typing import Callable, List

class CallbackStrategyUpdater(StrategyUpdater):
    """Strategy updater with callback hooks"""

    def __init__(self, query_router, **kwargs):
        super().__init__(query_router, **kwargs)
        self.on_weight_change_callbacks: List[Callable] = []
        self.on_degradation_callbacks: List[Callable] = []

    def on_weight_change(self, callback: Callable):
        """Register callback for weight changes"""
        self.on_weight_change_callbacks.append(callback)

    def on_degradation(self, callback: Callable):
        """Register callback for performance degradation"""
        self.on_degradation_callbacks.append(callback)

    async def _update_backend_weights(self, backend_perf):
        """Override to add callbacks"""
        for backend, metrics in backend_perf.items():
            if metrics.count == 0:
                continue

            current_weight = self.router.classifier.backend_weights.get(backend, 1.0)

            # Determine adjustment (same logic as parent)
            is_good = (metrics.avg_latency_ms < 50.0 and
                      metrics.avg_confidence > 0.80)
            is_poor = (metrics.avg_latency_ms > 150.0 or
                      metrics.avg_confidence < 0.70)

            if is_good:
                new_weight = min(current_weight * 1.20, current_weight + 0.20)
            elif is_poor:
                new_weight = max(current_weight * 0.80, current_weight - 0.20)
            else:
                continue

            # Apply weight
            self.router.classifier.backend_weights[backend] = new_weight

            # Trigger callbacks
            for callback in self.on_weight_change_callbacks:
                await callback(backend, current_weight, new_weight, metrics)

            # Check for degradation
            if is_poor:
                for callback in self.on_degradation_callbacks:
                    await callback(backend, metrics)

# Usage
async def alert_weight_change(backend, old_weight, new_weight, metrics):
    """Alert on weight changes"""
    change_pct = ((new_weight - old_weight) / old_weight) * 100
    print(f"⚠️  {backend} weight changed: {old_weight:.2f} → {new_weight:.2f} ({change_pct:+.1f}%)")
    print(f"   Reason: confidence={metrics.avg_confidence:.2f}, latency={metrics.avg_latency_ms:.1f}ms")

async def alert_degradation(backend, metrics):
    """Alert on performance issues"""
    print(f"🚨 DEGRADATION ALERT: {backend}")
    print(f"   Confidence: {metrics.avg_confidence:.2f}")
    print(f"   Latency: {metrics.avg_latency_ms:.1f}ms")
    print(f"   Fallback rate: {metrics.fallback_rate*100:.1f}%")

    # Could send to monitoring system, Slack, PagerDuty, etc.

updater = CallbackStrategyUpdater(router)
updater.on_weight_change(alert_weight_change)
updater.on_degradation(alert_degradation)
```

---

## 3. Custom Strategy Rules

### Add New Adaptation Strategies

**Example: Time-of-day routing**

```python
from hololoom.context import StrategyUpdater
from datetime import datetime, time

class TimeAwareStrategyUpdater(StrategyUpdater):
    """Strategy updater with time-of-day rules"""

    def __init__(self, query_router, **kwargs):
        super().__init__(query_router, **kwargs)

        # Define time-of-day rules
        self.time_rules = {
            "peak_hours": (time(9, 0), time(17, 0)),    # 9am-5pm
            "off_peak": (time(17, 0), time(9, 0)),       # 5pm-9am
        }

        self.peak_backend = "sql"      # Prefer SQL during peak
        self.offpeak_backend = "neo4j" # Prefer Neo4j off-peak

    def is_peak_hours(self) -> bool:
        """Check if current time is peak hours"""
        now = datetime.now().time()
        start, end = self.time_rules["peak_hours"]

        if start < end:
            return start <= now <= end
        else:  # Wraps midnight
            return now >= start or now <= end

    async def force_update(self):
        """Override to add time-based rules"""
        # First, run standard update
        await super().force_update()

        # Then apply time-based adjustments
        weights = self.router.classifier.backend_weights

        if self.is_peak_hours():
            # Peak hours: boost SQL, reduce Neo4j
            weights[self.peak_backend] = min(weights.get(self.peak_backend, 1.0) * 1.1, 2.0)
            weights[self.offpeak_backend] = max(weights.get(self.offpeak_backend, 1.0) * 0.9, 0.5)

            logger.info("Applied peak-hours routing strategy")
        else:
            # Off-peak: boost Neo4j, reduce SQL
            weights[self.offpeak_backend] = min(weights.get(self.offpeak_backend, 1.0) * 1.1, 2.0)
            weights[self.peak_backend] = max(weights.get(self.peak_backend, 1.0) * 0.9, 0.5)

            logger.info("Applied off-peak routing strategy")

# Usage
updater = TimeAwareStrategyUpdater(
    router,
    update_interval=300.0  # Update every 5 minutes
)
```

---

## 4. Custom Calibration Methods

### Implement Alternative Calibration Strategies

**Example: Exponential smoothing calibration**

```python
from hololoom.context import ConfidenceCalibrator, CalibrationCurve
from typing import Dict

class SmoothingCalibrator(ConfidenceCalibrator):
    """Calibrator using exponential smoothing"""

    def __init__(self, alpha: float = 0.1, min_observations: int = 10):
        super().__init__(min_observations)
        self.alpha = alpha  # Smoothing factor
        self.smoothed_errors: Dict[str, float] = {}

    def add_observation(
        self,
        predicted_confidence: float,
        actual_confidence: float,
        backend: str
    ):
        """Add observation with exponential smoothing"""
        # Call parent to store history
        super().add_observation(predicted_confidence, actual_confidence, backend)

        # Update smoothed error
        error = abs(predicted_confidence - actual_confidence)

        if backend not in self.smoothed_errors:
            self.smoothed_errors[backend] = error
        else:
            # Exponential smoothing: S_t = α * x_t + (1-α) * S_{t-1}
            self.smoothed_errors[backend] = (
                self.alpha * error +
                (1 - self.alpha) * self.smoothed_errors[backend]
            )

    def adjust_confidence(self, predicted: float, backend: str) -> float:
        """Adjust using smoothed error"""
        # Get smoothed error for this backend
        smoothed_error = self.smoothed_errors.get(backend, 0.0)

        # Check if we have enough observations
        backend_obs = [obs for obs in self.calibration_history
                      if obs.backend == backend]

        if len(backend_obs) < self.min_observations:
            return predicted

        # Adjust: reduce by smoothed error
        adjusted = predicted - smoothed_error

        # Clamp to [0, 1]
        return max(0.0, min(1.0, adjusted))

# Usage
calibrator = SmoothingCalibrator(alpha=0.2, min_observations=50)

for i in range(100):
    calibrator.add_observation(
        predicted_confidence=0.85,
        actual_confidence=0.75,
        backend="sql"
    )

adjusted = calibrator.adjust_confidence(0.85, "sql")
print(f"Smoothed adjustment: 0.85 → {adjusted:.2f}")
```

---

## 5. External Integration

### Connect to Monitoring Systems

**Example: Prometheus metrics**

```python
from hololoom.context import LearningTracker
from prometheus_client import Counter, Histogram, Gauge
import asyncio

class PrometheusLearningTracker(LearningTracker):
    """Learning tracker with Prometheus metrics"""

    def __init__(self, reflection_buffer=None):
        super().__init__(reflection_buffer)

        # Define Prometheus metrics
        self.query_counter = Counter(
            'hololoom_queries_total',
            'Total queries processed',
            ['backend', 'success']
        )

        self.latency_histogram = Histogram(
            'hololoom_query_latency_seconds',
            'Query latency distribution',
            ['backend'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
        )

        self.confidence_gauge = Gauge(
            'hololoom_confidence_average',
            'Average confidence per backend',
            ['backend']
        )

    async def record_routing(
        self,
        session_id: str,
        query: str,
        backend: str,
        predicted_confidence: float,
        actual_confidence: float,
        latency_ms: float,
        cache_hit: bool = False,
        fallback_used: bool = False
    ):
        # Call parent
        await super().record_routing(
            session_id, query, backend,
            predicted_confidence, actual_confidence,
            latency_ms, cache_hit, fallback_used
        )

        # Update Prometheus metrics
        success = not fallback_used and actual_confidence >= 0.75
        self.query_counter.labels(backend=backend, success=str(success)).inc()

        self.latency_histogram.labels(backend=backend).observe(latency_ms / 1000.0)

        # Update rolling average confidence
        recent = self.get_recent_performance(backend, window=100)
        self.confidence_gauge.labels(backend=backend).set(recent.avg_confidence)

# Start Prometheus HTTP server
from prometheus_client import start_http_server
start_http_server(8000)

# Use tracker
tracker = PrometheusLearningTracker()

# Metrics available at http://localhost:8000/metrics
```

**Example: Send to external logging**

```python
import logging
import json
from hololoom.context import LearningTracker

class StructuredLoggingTracker(LearningTracker):
    """Learning tracker with structured logging"""

    def __init__(self, reflection_buffer=None):
        super().__init__(reflection_buffer)
        self.logger = logging.getLogger("hololoom.learning")

    async def record_routing(
        self,
        session_id: str,
        query: str,
        backend: str,
        predicted_confidence: float,
        actual_confidence: float,
        latency_ms: float,
        cache_hit: bool = False,
        fallback_used: bool = False
    ):
        # Call parent
        await super().record_routing(
            session_id, query, backend,
            predicted_confidence, actual_confidence,
            latency_ms, cache_hit, fallback_used
        )

        # Structured logging (JSON format for ELK, Splunk, etc.)
        log_entry = {
            "event": "routing_decision",
            "session_id": session_id,
            "query_hash": hash(query) % 10000,  # Don't log sensitive data
            "backend": backend,
            "predicted_confidence": predicted_confidence,
            "actual_confidence": actual_confidence,
            "confidence_error": abs(predicted_confidence - actual_confidence),
            "latency_ms": latency_ms,
            "cache_hit": cache_hit,
            "fallback_used": fallback_used,
            "success": not fallback_used and actual_confidence >= 0.75
        }

        self.logger.info(json.dumps(log_entry))

# Configure structured logging
logging.basicConfig(
    format='%(message)s',
    level=logging.INFO
)

tracker = StructuredLoggingTracker()
```

---

## 6. Complete Example: Custom Learning System

**Putting it all together:**

```python
import asyncio
from hololoom.infrastructure.sql import SQLConfig
from hololoom.infrastructure.mcp import create_mcp_server, generate_session_id
from hololoom.context import QueryRouter, QueryClassifier, ThompsonBandit

# Use custom components
from my_extensions import (
    ExtendedLearningTracker,
    CallbackStrategyUpdater,
    SmoothingCalibrator
)

async def main():
    # Setup
    sql_config = SQLConfig(sqlite_path="./data/custom.db")
    mcp_server = await create_mcp_server(sql_config)

    # Create custom learning components
    calibrator = SmoothingCalibrator(alpha=0.2, min_observations=50)
    tracker = ExtendedLearningTracker()

    # Create router with custom components
    classifier = QueryClassifier()
    bandit = ThompsonBandit(["sql", "neo4j", "qdrant"])

    router = QueryRouter(
        classifier=classifier,
        bandit=bandit,
        mcp_server=mcp_server,
        session_id=generate_session_id(),
        enable_learning=True,
        enable_calibration=True,
        enable_strategy_updates=True
    )

    # Replace default components with custom ones
    router.calibrator = calibrator
    router.learning_tracker = tracker

    # Create custom strategy updater with callbacks
    updater = CallbackStrategyUpdater(router)

    async def on_degradation(backend, metrics):
        print(f"🚨 Alert: {backend} performance degraded!")
        # Send to Slack, PagerDuty, etc.

    updater.on_degradation(on_degradation)
    router.strategy_updater = updater

    # Process queries
    result = await router.route("What is the policy?")

    # Access custom metrics
    complexity_stats = tracker.get_complexity_correlation()
    print(f"Complexity correlation: {complexity_stats['correlation']:.2f}")

asyncio.run(main())
```

---

## 7. Testing Custom Extensions

```python
import pytest
from my_extensions import ExtendedLearningTracker

@pytest.mark.asyncio
async def test_custom_metrics():
    """Test custom complexity tracking"""
    tracker = ExtendedLearningTracker()

    # Record events with custom metrics
    await tracker.record_routing(
        session_id="s1",
        query="SELECT * FROM policies",
        backend="sql",
        predicted_confidence=0.85,
        actual_confidence=0.90,
        latency_ms=10.0,
        query_complexity=0.3,
        num_joins=0
    )

    await tracker.record_routing(
        session_id="s2",
        query="SELECT * FROM policies JOIN rules",
        backend="sql",
        predicted_confidence=0.80,
        actual_confidence=0.85,
        latency_ms=50.0,
        query_complexity=0.7,
        num_joins=1
    )

    # Check custom metrics
    stats = tracker.get_complexity_correlation()
    assert "correlation" in stats
    assert stats["correlation"] > 0  # Higher complexity → higher latency

if __name__ == "__main__":
    pytest.main([__file__])
```

---

## 8. Best Practices

### Do's ✅

1. **Extend, don't replace**: Inherit from base classes
2. **Call super()**: Always call parent methods
3. **Add tests**: Test custom extensions thoroughly
4. **Document**: Explain why you extended the system
5. **Monitor**: Track impact of custom learning rules
6. **Version**: Track which extensions are active

### Don'ts ❌

1. **Don't break interfaces**: Keep method signatures compatible
2. **Don't ignore errors**: Handle exceptions gracefully
3. **Don't over-tune**: Too many rules → instability
4. **Don't skip validation**: Validate custom metrics
5. **Don't forget cleanup**: Close resources properly

---

## 9. Common Extensions

### A. Query Type Detection

Track performance by query type:

```python
def classify_query_type(query: str) -> str:
    """Classify query into type"""
    query_lower = query.lower()

    if any(word in query_lower for word in ['select', 'get', 'show', 'list']):
        return 'read'
    elif any(word in query_lower for word in ['insert', 'create', 'add']):
        return 'write'
    elif any(word in query_lower for word in ['update', 'modify', 'change']):
        return 'update'
    elif any(word in query_lower for word in ['delete', 'remove']):
        return 'delete'
    else:
        return 'unknown'

# Track per-type performance
type_performance = {}
query_type = classify_query_type(query)
# Record in tracker...
```

### B. Cost Tracking

Monitor query costs:

```python
# Track cost per backend
cost_per_backend = {
    "sql": 0.001,      # $0.001 per query
    "neo4j": 0.005,    # $0.005 per query
    "qdrant": 0.002    # $0.002 per query
}

total_cost = sum(
    cost_per_backend[event.backend]
    for event in tracker.routing_history
)
```

### C. A/B Testing

Compare strategies:

```python
import random

class ABTestingRouter:
    def __init__(self, router_a, router_b, split_ratio=0.5):
        self.router_a = router_a
        self.router_b = router_b
        self.split_ratio = split_ratio

    async def route(self, query):
        # Route to A or B based on split
        if random.random() < self.split_ratio:
            return await self.router_a.route(query)
        else:
            return await self.router_b.route(query)
```

---

## 📚 See Also

- [QUICK_START.md](QUICK_START.md) - Basic usage
- [PART_4_LEARNING_COMPLETE.md](PART_4_LEARNING_COMPLETE.md) - Implementation details
- [demo_learning.py](demo_learning.py) - Working example
- [test_learning_routing.py](test_learning_routing.py) - Test patterns
