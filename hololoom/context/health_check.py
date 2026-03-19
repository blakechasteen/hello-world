"""
Health Check: System Health Monitoring

Part 5: Production Hardening - Day 24

Provides health check endpoint for load balancers and monitoring systems:
- Overall system health
- Backend connectivity (MCP, Neo4j, Qdrant)
- Learning system status
- Resource usage
- Circuit breaker states

Returns:
- HTTP-compatible JSON response
- Health status (healthy/unhealthy)
- Individual component checks
- Timestamps
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ============================================================================
# Health Status
# ============================================================================

class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


# ============================================================================
# Component Check Result
# ============================================================================

@dataclass
class ComponentCheck:
    """Result of a component health check"""
    name: str
    healthy: bool
    status: HealthStatus
    message: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "healthy": self.healthy,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp
        }


# ============================================================================
# Health Check Result
# ============================================================================

@dataclass
class HealthCheckResult:
    """Overall health check result"""
    healthy: bool
    status: HealthStatus
    checks: dict[str, ComponentCheck]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        """Convert to dictionary (HTTP response compatible)"""
        return {
            "healthy": self.healthy,
            "status": self.status.value,
            "checks": {
                name: check.to_dict()
                for name, check in self.checks.items()
            },
            "timestamp": self.timestamp
        }

    def to_json(self) -> str:
        """Convert to JSON string"""
        import json
        return json.dumps(self.to_dict(), indent=2)


# ============================================================================
# Health Checker
# ============================================================================

class HealthChecker:
    """
    System health checker

    Performs health checks on all components:
    - Overall system health
    - Backend connectivity
    - Learning system
    - Resource usage
    - Circuit breakers
    """

    def __init__(
        self,
        performance_monitor=None,
        resource_monitor=None,
        learning_monitor=None,
        circuit_breaker_registry=None
    ):
        """
        Initialize health checker

        Args:
            performance_monitor: PerformanceMonitor instance (optional)
            resource_monitor: ResourceMonitor instance (optional)
            learning_monitor: LearningMetricsMonitor instance (optional)
            circuit_breaker_registry: CircuitBreakerRegistry instance (optional)
        """
        self.performance_monitor = performance_monitor
        self.resource_monitor = resource_monitor
        self.learning_monitor = learning_monitor
        self.circuit_breaker_registry = circuit_breaker_registry

    async def check_health(self) -> HealthCheckResult:
        """
        Perform complete health check

        Returns:
            HealthCheckResult with all component checks
        """
        checks = {}

        # Overall system check
        checks["overall"] = self._check_overall()

        # Backend connectivity (if available)
        if self.circuit_breaker_registry:
            checks["backends"] = self._check_backends()

        # Learning system
        if self.learning_monitor:
            checks["learning"] = self._check_learning()

        # Resource usage
        if self.resource_monitor:
            checks["resources"] = self._check_resources()

        # Circuit breakers
        if self.circuit_breaker_registry:
            checks["circuit_breakers"] = self._check_circuit_breakers()

        # Determine overall health
        healthy = all(check.healthy for check in checks.values())
        status = self._determine_overall_status(checks)

        return HealthCheckResult(
            healthy=healthy,
            status=status,
            checks=checks
        )

    def _check_overall(self) -> ComponentCheck:
        """Check overall system health"""
        if not self.performance_monitor:
            return ComponentCheck(
                name="overall",
                healthy=True,
                status=HealthStatus.HEALTHY,
                message="System running (no performance monitor)",
                details={}
            )

        metrics = self.performance_monitor.get_metrics()

        # Check error rate
        error_rate = metrics.get("error_rate", 0.0)
        if error_rate > 0.10:  # >10% errors
            return ComponentCheck(
                name="overall",
                healthy=False,
                status=HealthStatus.UNHEALTHY,
                message=f"High error rate: {error_rate*100:.1f}%",
                details={"error_rate": error_rate}
            )

        # Check latency
        latency_p95 = metrics.get("latency_p95", 0.0)
        if latency_p95 > 1000:  # >1s p95 latency
            return ComponentCheck(
                name="overall",
                healthy=False,
                status=HealthStatus.DEGRADED,
                message=f"High latency: p95={latency_p95:.1f}ms",
                details={"latency_p95": latency_p95}
            )

        return ComponentCheck(
            name="overall",
            healthy=True,
            status=HealthStatus.HEALTHY,
            message="System healthy",
            details={
                "error_rate": error_rate,
                "latency_p95": latency_p95,
                "qps": metrics.get("qps", 0.0)
            }
        )

    def _check_backends(self) -> ComponentCheck:
        """Check backend connectivity via circuit breakers"""
        if not self.circuit_breaker_registry:
            return ComponentCheck(
                name="backends",
                healthy=True,
                status=HealthStatus.HEALTHY,
                message="No circuit breakers configured"
            )

        health_summary = self.circuit_breaker_registry.get_health_summary()

        if health_summary["healthy"]:
            return ComponentCheck(
                name="backends",
                healthy=True,
                status=HealthStatus.HEALTHY,
                message="All backends healthy",
                details={
                    "total_breakers": health_summary["total_breakers"],
                    "all_closed": health_summary["all_closed"]
                }
            )
        else:
            open_breakers = health_summary.get("open_breakers", [])
            return ComponentCheck(
                name="backends",
                healthy=False,
                status=HealthStatus.DEGRADED,
                message=f"Circuit breakers open: {', '.join(open_breakers)}",
                details={
                    "open_breakers": open_breakers,
                    "half_open_breakers": health_summary.get("half_open_breakers", [])
                }
            )

    def _check_learning(self) -> ComponentCheck:
        """Check learning system health"""
        if not self.learning_monitor:
            return ComponentCheck(
                name="learning",
                healthy=True,
                status=HealthStatus.HEALTHY,
                message="Learning system running"
            )

        metrics = self.learning_monitor.get_metrics()

        # Check calibration ECE
        ece = metrics.get("calibration_ece")
        if ece is not None and ece > 0.15:  # ECE > 0.15 (poorly calibrated)
            return ComponentCheck(
                name="learning",
                healthy=False,
                status=HealthStatus.DEGRADED,
                message=f"Poor calibration: ECE={ece:.3f}",
                details={"ece": ece}
            )

        return ComponentCheck(
            name="learning",
            healthy=True,
            status=HealthStatus.HEALTHY,
            message="Learning system healthy",
            details={
                "ece": ece,
                "strategy_updates": metrics.get("strategy_update_count", 0)
            }
        )

    def _check_resources(self) -> ComponentCheck:
        """Check resource usage"""
        if not self.resource_monitor:
            return ComponentCheck(
                name="resources",
                healthy=True,
                status=HealthStatus.HEALTHY,
                message="Resource monitoring not available"
            )

        metrics = self.resource_monitor.get_metrics()

        memory_mb = metrics.get("memory_mb", 0.0)
        cpu_percent = metrics.get("cpu_percent", 0.0)

        # Check memory (>80% of 2GB = 1.6GB)
        if memory_mb > 1600:
            return ComponentCheck(
                name="resources",
                healthy=False,
                status=HealthStatus.DEGRADED,
                message=f"High memory usage: {memory_mb:.1f} MB",
                details={"memory_mb": memory_mb, "cpu_percent": cpu_percent}
            )

        # Check CPU (>80%)
        if cpu_percent > 80:
            return ComponentCheck(
                name="resources",
                healthy=False,
                status=HealthStatus.DEGRADED,
                message=f"High CPU usage: {cpu_percent:.1f}%",
                details={"memory_mb": memory_mb, "cpu_percent": cpu_percent}
            )

        return ComponentCheck(
            name="resources",
            healthy=True,
            status=HealthStatus.HEALTHY,
            message="Resources healthy",
            details={"memory_mb": memory_mb, "cpu_percent": cpu_percent}
        )

    def _check_circuit_breakers(self) -> ComponentCheck:
        """Check circuit breaker states"""
        if not self.circuit_breaker_registry:
            return ComponentCheck(
                name="circuit_breakers",
                healthy=True,
                status=HealthStatus.HEALTHY,
                message="No circuit breakers"
            )

        health_summary = self.circuit_breaker_registry.get_health_summary()

        if health_summary["all_closed"]:
            return ComponentCheck(
                name="circuit_breakers",
                healthy=True,
                status=HealthStatus.HEALTHY,
                message="All circuit breakers closed",
                details={"total_breakers": health_summary["total_breakers"]}
            )
        else:
            return ComponentCheck(
                name="circuit_breakers",
                healthy=False,
                status=HealthStatus.DEGRADED,
                message="Some circuit breakers open",
                details={
                    "open_breakers": health_summary.get("open_breakers", []),
                    "half_open_breakers": health_summary.get("half_open_breakers", [])
                }
            )

    def _determine_overall_status(
        self,
        checks: dict[str, ComponentCheck]
    ) -> HealthStatus:
        """Determine overall health status from component checks"""
        if all(check.healthy for check in checks.values()):
            return HealthStatus.HEALTHY

        # Count unhealthy components
        unhealthy_count = sum(
            1 for check in checks.values()
            if check.status == HealthStatus.UNHEALTHY
        )

        if unhealthy_count > 0:
            return HealthStatus.UNHEALTHY
        else:
            return HealthStatus.DEGRADED

    def get_quick_status(self) -> dict:
        """
        Get quick health status (non-async, for simple checks)

        Returns:
            Dictionary with basic health info
        """
        checks = {}

        # Overall
        checks["overall"] = self._check_overall().to_dict()

        # Resources (if available)
        if self.resource_monitor:
            checks["resources"] = self._check_resources().to_dict()

        healthy = all(
            check["healthy"]
            for check in checks.values()
        )

        return {
            "healthy": healthy,
            "checks": checks,
            "timestamp": time.time()
        }


# ============================================================================
# Factory Function
# ============================================================================

def create_health_checker(
    performance_monitor=None,
    resource_monitor=None,
    learning_monitor=None,
    circuit_breaker_registry=None
) -> HealthChecker:
    """
    Create health checker

    Args:
        performance_monitor: PerformanceMonitor instance (optional)
        resource_monitor: ResourceMonitor instance (optional)
        learning_monitor: LearningMetricsMonitor instance (optional)
        circuit_breaker_registry: CircuitBreakerRegistry instance (optional)

    Returns:
        HealthChecker instance
    """
    return HealthChecker(
        performance_monitor=performance_monitor,
        resource_monitor=resource_monitor,
        learning_monitor=learning_monitor,
        circuit_breaker_registry=circuit_breaker_registry
    )
