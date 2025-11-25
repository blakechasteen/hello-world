"""
Ernest Production Module
=========================
Production hardening for Ernest creative writing system.

Main exports:
- ErnestProductionGuard: Circuit breakers + rate limiting
- ErnestHealthMonitor: Health checks and monitoring
- ErnestAlertManager: Alert management
- create_production_guard: Quick production guard creation
- create_health_monitor: Quick health monitor creation
"""

from .circuit_breakers import (
    ErnestProductionGuard,
    ErnestRateLimiter,
    ErnestCircuitBreaker,
    RateLimitConfig,
    CircuitBreakerConfig,
    CircuitState,
    RateLimitExceeded,
    CircuitBreakerOpen,
    create_production_guard
)

from .monitoring import (
    ErnestHealthMonitor,
    ErnestAlertManager,
    HealthStatus,
    HemingwayScoreTrend,
    ModeConvergenceTracker,
    PerformanceMetrics,
    create_health_monitor,
    create_alert_manager
)

__all__ = [
    # Circuit breakers
    "ErnestProductionGuard",
    "ErnestRateLimiter",
    "ErnestCircuitBreaker",
    "RateLimitConfig",
    "CircuitBreakerConfig",
    "CircuitState",
    "RateLimitExceeded",
    "CircuitBreakerOpen",
    "create_production_guard",

    # Monitoring
    "ErnestHealthMonitor",
    "ErnestAlertManager",
    "HealthStatus",
    "HemingwayScoreTrend",
    "ModeConvergenceTracker",
    "PerformanceMetrics",
    "create_health_monitor",
    "create_alert_manager"
]
