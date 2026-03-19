"""
Context Department for Hybrid Query Routing

Part 3: Classification and Basic Routing (Days 11-15)
Part 4: Learning Mechanisms (Days 16-20)
Part 5: Production Hardening (Days 21-25)

Components (Part 3):
- QueryClassifier: 7-rule decision tree (100% accuracy)
- ThompsonBandit: Adaptive backend selection (converged @ 164 iterations)
- QueryRouter: Multi-backend coordination (4 routing patterns)

Components (Part 4):
- ConfidenceCalibrator: Confidence prediction calibration (ECE tracking)
- LearningTracker: Routing decision tracking and analytics
- StrategyUpdater: Strategy adaptation based on learning signals

Components (Part 5):
- ErrorHandler: Graceful degradation and error recovery
- RetryConfig: Configurable retry with exponential backoff
- FallbackStrategy: Cascading fallback strategies
- SystemMonitor: Performance, resource, and learning metrics
- CircuitBreaker: Prevent cascading failures from external dependencies

Public API:
- QueryClassifier, BackendSelection, Backend, ConfidenceTier
- ThompsonBandit, BanditArm
- QueryRouter, RoutingPattern, RoutingResult
- ConfidenceCalibrator, CalibrationObservation, CalibrationCurve
- LearningTracker, RoutingEvent, PerformanceMetrics
- StrategyUpdater, StrategyUpdate
- ErrorHandler, RetryConfig, FallbackStrategy (Part 5)
- Factory functions: create_*
"""

from hololoom.context.bandit import BanditArm, ThompsonBandit, create_thompson_bandit

# Part 4: Learning Mechanisms
from hololoom.context.calibration import (
    CalibrationCurve,
    CalibrationObservation,
    ConfidenceCalibrator,
    create_confidence_calibrator,
)
from hololoom.context.circuit_breaker import (
    # Circuit Breaker
    CircuitBreaker,
    # Configuration
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    # States
    CircuitState,
    # Factory
    create_circuit_breaker,
    create_circuit_breaker_registry,
)
from hololoom.context.classifier import (
    Backend,
    BackendSelection,
    ConfidenceTier,
    QueryClassifier,
    create_classifier,
)

# Part 5: Production Hardening
from hololoom.context.error_handling import (
    BackendError,
    CalibrationError,
    CircuitBreakerOpenError,
    # Exceptions
    ContextError,
    # Categorization
    ErrorCategory,
    # Handler
    ErrorHandler,
    # Fallback
    FallbackStrategy,
    LearningError,
    RateLimitExceededError,
    # Retry
    RetryConfig,
    RoutingError,
    categorize_error,
    create_error_handler,
    is_retryable,
    retry,
    should_fallback,
)
from hololoom.context.health_check import (
    # Results
    ComponentCheck,
    # Checker
    HealthChecker,
    HealthCheckResult,
    # Status
    HealthStatus,
    create_health_checker,
)
from hololoom.context.learning_tracker import (
    LearningTracker,
    PerformanceMetrics,
    RoutingEvent,
    create_learning_tracker,
)
from hololoom.context.monitoring import (
    LearningMetricsMonitor,
    # Monitors
    PerformanceMonitor,
    ResourceMonitor,
    SystemMonitor,
    # Factory
    create_system_monitor,
)
from hololoom.context.production_config import CircuitBreakerConfig as ConfigCircuitBreakerConfig
from hololoom.context.production_config import (
    # Environment
    Environment,
    ErrorHandlingConfig,
    LearningConfig,
    MonitoringConfig,
    # Configurations
    ProductionConfig,
    ResourceConfig,
    # Factory
    create_config,
    detect_environment,
)
from hololoom.context.production_config import RateLimitConfig as ConfigRateLimitConfig
from hololoom.context.rate_limiter import (
    ConcurrentLimiter,
    RateLimiter,
    # Configuration
    RateLimiterConfig,
    # Types
    RateLimiterType,
    SlidingWindowRateLimiter,
    # Rate Limiters
    TokenBucketRateLimiter,
    create_concurrent_limiter,
    # Factory
    create_rate_limiter,
    create_sliding_window_limiter,
    create_token_bucket_limiter,
)
from hololoom.context.router import QueryRouter, RoutingPattern, RoutingResult, create_query_router
from hololoom.context.strategy_updater import (
    StrategyUpdate,
    StrategyUpdater,
    create_strategy_updater,
)

__all__ = [
    # Classifier
    "QueryClassifier",
    "BackendSelection",
    "Backend",
    "ConfidenceTier",
    "create_classifier",

    # Bandit
    "ThompsonBandit",
    "BanditArm",
    "create_thompson_bandit",

    # Router
    "QueryRouter",
    "RoutingPattern",
    "RoutingResult",
    "create_query_router",

    # Learning (Part 4)
    "ConfidenceCalibrator",
    "CalibrationObservation",
    "CalibrationCurve",
    "create_confidence_calibrator",

    "LearningTracker",
    "RoutingEvent",
    "PerformanceMetrics",
    "create_learning_tracker",

    "StrategyUpdater",
    "StrategyUpdate",
    "create_strategy_updater",

    # Error Handling (Part 5)
    "ContextError",
    "RoutingError",
    "BackendError",
    "CalibrationError",
    "LearningError",
    "RateLimitExceededError",
    "CircuitBreakerOpenError",
    "ErrorCategory",
    "categorize_error",
    "is_retryable",
    "should_fallback",
    "RetryConfig",
    "retry",
    "FallbackStrategy",
    "ErrorHandler",
    "create_error_handler",

    # Monitoring (Part 5)
    "PerformanceMonitor",
    "ResourceMonitor",
    "LearningMetricsMonitor",
    "SystemMonitor",
    "create_system_monitor",

    # Circuit Breaker (Part 5)
    "CircuitState",
    "CircuitBreakerConfig",
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "create_circuit_breaker",
    "create_circuit_breaker_registry",

    # Rate Limiter (Part 5)
    "RateLimiterType",
    "RateLimiterConfig",
    "TokenBucketRateLimiter",
    "SlidingWindowRateLimiter",
    "ConcurrentLimiter",
    "RateLimiter",
    "create_rate_limiter",
    "create_token_bucket_limiter",
    "create_sliding_window_limiter",
    "create_concurrent_limiter",

    # Production Configuration (Part 5)
    "Environment",
    "ProductionConfig",
    "MonitoringConfig",
    "ErrorHandlingConfig",
    "ConfigCircuitBreakerConfig",
    "ConfigRateLimitConfig",
    "ResourceConfig",
    "LearningConfig",
    "create_config",
    "detect_environment",

    # Health Check (Part 5)
    "HealthStatus",
    "ComponentCheck",
    "HealthCheckResult",
    "HealthChecker",
    "create_health_checker",
]
