# Part 5: Production Hardening - Implementation Plan

**Status**: Planning Phase
**Timeline**: Days 21-25 (5 days)
**Dependencies**: Parts 2-4 Complete (25/25 tests passing)

## Executive Summary

Part 5 focuses on making the Context Department production-ready through:
- Error handling and graceful degradation
- Performance monitoring and metrics
- Circuit breakers for external dependencies
- Rate limiting and resource management
- Production deployment configurations

## Goals

1. **Reliability**: System handles failures gracefully without crashes
2. **Observability**: Complete visibility into system health and performance
3. **Safety**: Circuit breakers prevent cascading failures
4. **Scalability**: Rate limiting and resource management prevent overload
5. **Operations**: Production-ready deployment configurations

## Components to Implement

### 1. Error Handling (`error_handling.py`) - Day 21

**Purpose**: Graceful degradation and comprehensive error recovery

**Features**:
- Custom exception hierarchy for context operations
- Retry logic with exponential backoff
- Fallback strategies for failed operations
- Error categorization (transient vs permanent)
- Automatic recovery mechanisms

**Exception Types**:
```python
class ContextError(Exception): pass
class RoutingError(ContextError): pass
class BackendError(ContextError): pass
class CalibrationError(ContextError): pass
class LearningError(ContextError): pass
```

**Retry Decorator**:
```python
@retry(max_attempts=3, backoff_factor=2.0, retry_on=(BackendError,))
async def route_with_retry(query: str) -> RoutingResult:
    pass
```

**Fallback Strategy**:
```python
# Primary: Use calibrated confidence
# Fallback 1: Use raw confidence (if calibration fails)
# Fallback 2: Use fixed confidence (if prediction fails)
```

### 2. Monitoring (`monitoring.py`) - Day 22

**Purpose**: Comprehensive system health and performance tracking

**Metrics to Track**:

**Routing Metrics**:
- Queries per second (QPS)
- Latency percentiles (p50, p90, p95, p99)
- Error rate by error type
- Cache hit rate
- Fallback usage rate

**Learning Metrics**:
- Calibration ECE over time
- Strategy update frequency
- Weight adjustment magnitude
- Learning convergence indicators

**Resource Metrics**:
- Memory usage
- CPU usage
- Network I/O (for remote backends)
- Queue depth (for async operations)

**Export Formats**:
- Prometheus metrics endpoint
- JSON metrics dump
- Human-readable summary

**Implementation**:
```python
class PerformanceMonitor:
    def __init__(self):
        self.query_counter = 0
        self.latency_histogram = []
        self.error_counts = defaultdict(int)

    def record_query(self, latency_ms: float, error: Optional[str] = None):
        self.query_counter += 1
        self.latency_histogram.append(latency_ms)
        if error:
            self.error_counts[error] += 1

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "qps": self.query_counter / elapsed_time,
            "p50_latency": np.percentile(self.latency_histogram, 50),
            "p95_latency": np.percentile(self.latency_histogram, 95),
            "error_rate": sum(self.error_counts.values()) / self.query_counter
        }
```

### 3. Circuit Breakers (`circuit_breaker.py`) - Day 22

**Purpose**: Prevent cascading failures from external dependencies

**Dependencies to Protect**:
- MCP backend calls
- Neo4j graph queries
- Qdrant vector searches
- External enrichment services (Ollama)

**States**:
- CLOSED: Normal operation (all requests pass through)
- OPEN: Failure threshold exceeded (fail fast, no requests)
- HALF_OPEN: Testing recovery (limited requests allowed)

**Configuration**:
```python
class CircuitBreakerConfig:
    failure_threshold: int = 5        # Failures before opening
    recovery_timeout: float = 60.0    # Seconds before trying HALF_OPEN
    success_threshold: int = 2        # Successes in HALF_OPEN to close
    timeout: float = 5.0              # Request timeout
```

**Implementation**:
```python
class CircuitBreaker:
    def __init__(self, config: CircuitBreakerConfig):
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None

    async def call(self, func, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitBreakerOpenError("Circuit is open")

        try:
            result = await asyncio.wait_for(func(*args, **kwargs), timeout=self.timeout)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise
```

### 4. Rate Limiting (`rate_limiter.py`) - Day 23

**Purpose**: Prevent resource exhaustion and ensure fair usage

**Limits to Enforce**:
- Queries per second (global and per-session)
- Concurrent routing operations
- Backend query budget (from Part 3)
- Memory usage for caching

**Algorithms**:
- Token Bucket (for bursty traffic)
- Sliding Window (for precise rate limiting)
- Concurrent Semaphore (for parallelism control)

**Implementation**:
```python
class TokenBucketRateLimiter:
    def __init__(self, rate: float, capacity: int):
        self.rate = rate           # Tokens per second
        self.capacity = capacity   # Bucket size
        self.tokens = capacity
        self.last_update = time.time()

    async def acquire(self) -> bool:
        self._refill()
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False

    def _refill(self):
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_update = now
```

**Integration**:
```python
# Per-session rate limiting
session_limiter = TokenBucketRateLimiter(rate=10.0, capacity=20)  # 10 QPS burst to 20

async def route(query: str) -> RoutingResult:
    if not await session_limiter.acquire():
        raise RateLimitExceededError("Too many requests")
    return await self._route_internal(query)
```

### 5. Production Configuration (`production_config.py`) - Day 24

**Purpose**: Production-ready deployment configurations

**Configuration Profiles**:

**Development**:
```python
class DevelopmentConfig:
    enable_learning = True
    enable_calibration = True
    enable_strategy_updates = True

    # Monitoring
    enable_monitoring = True
    log_level = "DEBUG"

    # Error Handling
    max_retries = 3
    retry_backoff = 1.0

    # Circuit Breakers
    circuit_breaker_enabled = False  # Disable in dev

    # Rate Limiting
    rate_limit_enabled = False  # No limits in dev
```

**Staging**:
```python
class StagingConfig:
    enable_learning = True
    enable_calibration = True
    enable_strategy_updates = True

    # Monitoring
    enable_monitoring = True
    metrics_export = "prometheus"
    log_level = "INFO"

    # Error Handling
    max_retries = 3
    retry_backoff = 2.0

    # Circuit Breakers
    circuit_breaker_enabled = True
    failure_threshold = 5
    recovery_timeout = 60.0

    # Rate Limiting
    rate_limit_enabled = True
    global_qps = 100.0
    session_qps = 10.0
```

**Production**:
```python
class ProductionConfig:
    enable_learning = True
    enable_calibration = True
    enable_strategy_updates = True

    # Monitoring
    enable_monitoring = True
    metrics_export = "prometheus"
    metrics_port = 9090
    log_level = "WARNING"

    # Error Handling
    max_retries = 5
    retry_backoff = 2.0
    retry_jitter = 0.1

    # Circuit Breakers
    circuit_breaker_enabled = True
    failure_threshold = 3
    recovery_timeout = 120.0
    timeout = 10.0

    # Rate Limiting
    rate_limit_enabled = True
    global_qps = 1000.0
    session_qps = 50.0
    max_concurrent = 100

    # Resource Management
    max_memory_mb = 2048
    max_cache_size = 10000
```

### 6. Health Checks (`health_check.py`) - Day 24

**Purpose**: Endpoint for load balancers and monitoring systems

**Checks**:
- System health (overall status)
- Backend connectivity (MCP, Neo4j, Qdrant)
- Learning system status (calibration, tracking)
- Resource usage (memory, CPU)
- Circuit breaker states

**Implementation**:
```python
class HealthCheck:
    async def check_health(self) -> HealthStatus:
        checks = {
            "overall": self._check_overall(),
            "backends": await self._check_backends(),
            "learning": self._check_learning(),
            "resources": self._check_resources(),
            "circuit_breakers": self._check_circuit_breakers()
        }

        return HealthStatus(
            healthy=all(c.healthy for c in checks.values()),
            checks=checks,
            timestamp=time.time()
        )
```

**HTTP Endpoint** (for integration):
```python
# GET /health
{
    "healthy": true,
    "checks": {
        "overall": {"healthy": true, "latency_p95": 120.5},
        "backends": {"healthy": true, "mcp": "ok", "neo4j": "ok"},
        "learning": {"healthy": true, "calibration_ece": 0.08},
        "resources": {"healthy": true, "memory_mb": 512, "cpu_percent": 25},
        "circuit_breakers": {"healthy": true, "all_closed": true}
    },
    "timestamp": 1698595200.0
}
```

### 7. Integration Tests (`test_production_hardening.py`) - Day 25

**Purpose**: Validate all production hardening features

**Tests**:

1. **test_error_handling**: Verify retry logic and fallbacks
2. **test_circuit_breaker**: Verify state transitions and recovery
3. **test_rate_limiting**: Verify limits enforce correctly
4. **test_monitoring_metrics**: Verify metrics export
5. **test_health_checks**: Verify health endpoint accuracy
6. **test_production_config**: Verify configuration loading
7. **test_end_to_end_production**: Full production scenario

**Test Scenarios**:
```python
async def test_circuit_breaker_opens_on_failures():
    """Circuit breaker opens after threshold failures"""
    breaker = CircuitBreaker(failure_threshold=3)

    # Simulate 3 failures
    for i in range(3):
        with pytest.raises(Exception):
            await breaker.call(failing_operation)

    # Circuit should be open
    assert breaker.state == CircuitState.OPEN

    # Next call should fail fast
    with pytest.raises(CircuitBreakerOpenError):
        await breaker.call(working_operation)

async def test_rate_limiter_enforces_limits():
    """Rate limiter prevents exceeding QPS"""
    limiter = TokenBucketRateLimiter(rate=10.0, capacity=10)

    # Should allow 10 immediate requests
    for i in range(10):
        assert await limiter.acquire()

    # 11th should be rejected
    assert not await limiter.acquire()

    # After 0.1s, should allow 1 more (10 QPS)
    await asyncio.sleep(0.1)
    assert await limiter.acquire()
```

## Implementation Schedule

### Day 21: Error Handling
- Create exception hierarchy
- Implement retry decorator
- Add fallback strategies
- Test error recovery
- **Deliverable**: error_handling.py (300+ lines), 5 tests

### Day 22: Monitoring & Circuit Breakers
- Implement performance monitor
- Add Prometheus metrics export
- Create circuit breaker
- Test state transitions
- **Deliverable**: monitoring.py (400+ lines), circuit_breaker.py (300+ lines), 8 tests

### Day 23: Rate Limiting
- Implement token bucket
- Add sliding window
- Create concurrent limiter
- Integrate with router
- **Deliverable**: rate_limiter.py (350+ lines), 6 tests

### Day 24: Production Config & Health
- Create configuration profiles
- Add environment detection
- Implement health checks
- Create HTTP endpoint
- **Deliverable**: production_config.py (250+ lines), health_check.py (200+ lines), 4 tests

### Day 25: Integration & Documentation
- Create comprehensive integration tests
- Write production deployment guide
- Create runbook for operations
- Update README with production info
- **Deliverable**: test_production_hardening.py (500+ lines), PART_5_PRODUCTION_COMPLETE.md

## Success Criteria

**Code Quality**:
- All new code follows existing patterns
- Type hints throughout
- Comprehensive docstrings
- No breaking changes to existing API

**Test Coverage**:
- Minimum 23 new tests (target: all passing)
- Integration test covering full production scenario
- Error injection tests
- Load testing scenarios

**Documentation**:
- Production deployment guide
- Operations runbook
- Configuration reference
- Troubleshooting guide

**Performance**:
- Monitoring overhead <1ms per query
- Circuit breaker overhead <0.5ms per call
- Rate limiter overhead <0.1ms per check
- Total production overhead <5ms

**Reliability**:
- Graceful degradation under all error conditions
- No crashes from external failures
- Circuit breakers prevent cascading failures
- Rate limiting prevents resource exhaustion

## Integration Points

### With Existing Components

**Router Integration**:
```python
class QueryRouter:
    def __init__(self, ...,
                 error_handler: ErrorHandler,
                 monitor: PerformanceMonitor,
                 circuit_breaker: CircuitBreaker,
                 rate_limiter: RateLimiter):
        self.error_handler = error_handler
        self.monitor = monitor
        self.circuit_breaker = circuit_breaker
        self.rate_limiter = rate_limiter

    async def route(self, query: str) -> RoutingResult:
        # Rate limiting
        if not await self.rate_limiter.acquire():
            raise RateLimitExceededError()

        start_time = time.time()
        try:
            # Circuit breaker protection
            result = await self.circuit_breaker.call(self._route_internal, query)

            # Monitoring
            latency = (time.time() - start_time) * 1000
            self.monitor.record_query(latency)

            return result

        except Exception as e:
            # Error handling
            return await self.error_handler.handle(e, query)
```

**Configuration Loading**:
```python
# Automatic environment detection
config = ProductionConfig.from_environment()

# Or explicit
config = ProductionConfig.load("production")
```

## Risks and Mitigations

**Risk 1**: Performance overhead from monitoring
- **Mitigation**: Sampling (only monitor 10% of requests in high-volume scenarios)
- **Mitigation**: Async metrics export (non-blocking)

**Risk 2**: Circuit breaker false positives
- **Mitigation**: Configurable thresholds per environment
- **Mitigation**: Manual circuit breaker override for operators

**Risk 3**: Rate limiting blocking legitimate traffic
- **Mitigation**: Token bucket allows bursts
- **Mitigation**: Per-session limits prevent single user monopolizing

**Risk 4**: Configuration complexity
- **Mitigation**: Sensible defaults for all settings
- **Mitigation**: Configuration validation on load

## Backward Compatibility

All Part 5 features are **opt-in** via configuration:
- Default: Production features disabled (development mode)
- Enable via `Config.production()` factory
- Existing code continues to work unchanged

## Dependencies

**External**:
- None (all implementations use stdlib)

**Internal**:
- Parts 2-4 complete (router, learning, infrastructure)
- Existing test infrastructure

## Documentation Deliverables

1. **PART_5_PRODUCTION_COMPLETE.md**: Implementation summary
2. **PRODUCTION_DEPLOYMENT_GUIDE.md**: Step-by-step deployment
3. **OPERATIONS_RUNBOOK.md**: Day-to-day operations
4. **CONFIGURATION_REFERENCE.md**: All settings documented
5. **TROUBLESHOOTING_GUIDE.md**: Common issues and solutions

## Next Steps

After approval, begin implementation with Day 21: Error Handling.
