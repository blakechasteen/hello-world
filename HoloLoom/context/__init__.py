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

from HoloLoom.context.classifier import (
    QueryClassifier,
    BackendSelection,
    Backend,
    ConfidenceTier,
    create_classifier
)

from HoloLoom.context.bandit import (
    ThompsonBandit,
    BanditArm,
    create_thompson_bandit
)

from HoloLoom.context.router import (
    QueryRouter,
    RoutingPattern,
    RoutingResult,
    create_query_router
)

# Part 4: Learning Mechanisms
from HoloLoom.context.calibration import (
    ConfidenceCalibrator,
    CalibrationObservation,
    CalibrationCurve,
    create_confidence_calibrator
)

from HoloLoom.context.learning_tracker import (
    LearningTracker,
    RoutingEvent,
    PerformanceMetrics,
    create_learning_tracker
)

from HoloLoom.context.strategy_updater import (
    StrategyUpdater,
    StrategyUpdate,
    create_strategy_updater
)

# Part 5: Production Hardening
from HoloLoom.context.error_handling import (
    # Exceptions
    ContextError,
    RoutingError,
    BackendError,
    CalibrationError,
    LearningError,
    RateLimitExceededError,
    CircuitBreakerOpenError,
    # Categorization
    ErrorCategory,
    categorize_error,
    is_retryable,
    should_fallback,
    # Retry
    RetryConfig,
    retry,
    # Fallback
    FallbackStrategy,
    # Handler
    ErrorHandler,
    create_error_handler
)

from HoloLoom.context.monitoring import (
    # Monitors
    PerformanceMonitor,
    ResourceMonitor,
    LearningMetricsMonitor,
    SystemMonitor,
    # Factory
    create_system_monitor
)

from HoloLoom.context.circuit_breaker import (
    # States
    CircuitState,
    # Configuration
    CircuitBreakerConfig,
    # Circuit Breaker
    CircuitBreaker,
    CircuitBreakerRegistry,
    # Factory
    create_circuit_breaker,
    create_circuit_breaker_registry
)

from HoloLoom.context.rate_limiter import (
    # Types
    RateLimiterType,
    # Configuration
    RateLimiterConfig,
    # Rate Limiters
    TokenBucketRateLimiter,
    SlidingWindowRateLimiter,
    ConcurrentLimiter,
    RateLimiter,
    # Factory
    create_rate_limiter,
    create_token_bucket_limiter,
    create_sliding_window_limiter,
    create_concurrent_limiter
)

from HoloLoom.context.production_config import (
    # Environment
    Environment,
    # Configurations
    ProductionConfig,
    MonitoringConfig,
    ErrorHandlingConfig,
    CircuitBreakerConfig as ConfigCircuitBreakerConfig,
    RateLimitConfig as ConfigRateLimitConfig,
    ResourceConfig,
    LearningConfig,
    # Factory
    create_config,
    detect_environment
)

from HoloLoom.context.health_check import (
    # Status
    HealthStatus,
    # Results
    ComponentCheck,
    HealthCheckResult,
    # Checker
    HealthChecker,
    create_health_checker
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
