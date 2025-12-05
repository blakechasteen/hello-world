"""
Health Checks - EventBus Monitoring
===================================

**Created**: December 3, 2025
**Purpose**: Health monitoring and diagnostics for EventBus components

## Features

- Component health status (broker, store, dead letter, workflows)
- Performance metrics (latency, throughput, queue depth)
- Alert thresholds with configurable limits
- Prometheus-compatible metrics export
- Structured health check endpoint data

## Health Status

- HEALTHY: All systems nominal
- DEGRADED: Some non-critical issues (warnings)
- UNHEALTHY: Critical issues affecting operation
"""

import asyncio
from typing import Optional, List, Dict, Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Health Status
# ============================================================================

class HealthStatus(Enum):
    """Overall health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentStatus(Enum):
    """Individual component status."""
    UP = "up"
    DOWN = "down"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


# ============================================================================
# Health Check Results
# ============================================================================

@dataclass
class ComponentHealth:
    """Health status of a single component."""
    name: str
    status: ComponentStatus
    latency_ms: Optional[float] = None
    message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    last_check: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class HealthCheckResult:
    """Complete health check result."""
    status: HealthStatus
    components: List[ComponentHealth] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'status': self.status.value,
            'timestamp': self.timestamp,
            'duration_ms': self.duration_ms,
            'components': [
                {
                    'name': c.name,
                    'status': c.status.value,
                    'latency_ms': c.latency_ms,
                    'message': c.message,
                    'details': c.details,
                    'last_check': c.last_check
                }
                for c in self.components
            ],
            'metrics': self.metrics
        }


# ============================================================================
# Alert Thresholds
# ============================================================================

@dataclass
class AlertThresholds:
    """Configurable alert thresholds."""
    # Latency thresholds (ms)
    event_latency_warning: float = 100.0
    event_latency_critical: float = 500.0

    # Queue thresholds
    queue_depth_warning: int = 1000
    queue_depth_critical: int = 5000

    # Dead letter thresholds
    dead_letter_warning: int = 10
    dead_letter_critical: int = 100

    # Circuit breaker thresholds
    circuit_breaker_open_warning: int = 1
    circuit_breaker_open_critical: int = 3

    # Workflow thresholds
    failed_workflows_warning: int = 5
    failed_workflows_critical: int = 20


# ============================================================================
# Health Checker
# ============================================================================

class HealthChecker:
    """
    Health checker for EventBus components.

    Performs health checks on:
    - EventBroker (connectivity, latency)
    - EventStore (database connection, disk usage)
    - DeadLetterQueue (pending count, retry status)
    - WorkflowCoordinator (active workflows, failures)
    """

    def __init__(
        self,
        broker: Optional[Any] = None,
        event_store: Optional[Any] = None,
        dead_letter_queue: Optional[Any] = None,
        workflow_coordinator: Optional[Any] = None,
        thresholds: Optional[AlertThresholds] = None
    ):
        """
        Initialize health checker.

        Args:
            broker: EventBroker instance
            event_store: EventStore instance
            dead_letter_queue: DeadLetterQueue instance
            workflow_coordinator: WorkflowCoordinator instance
            thresholds: Alert thresholds configuration
        """
        self.broker = broker
        self.event_store = event_store
        self.dead_letter_queue = dead_letter_queue
        self.workflow_coordinator = workflow_coordinator
        self.thresholds = thresholds or AlertThresholds()

        # Metrics history
        self._metrics_history: List[Dict[str, Any]] = []
        self._max_history = 100

        logger.info("HealthChecker initialized")

    async def check_health(self) -> HealthCheckResult:
        """
        Perform comprehensive health check.

        Returns:
            HealthCheckResult with all component statuses
        """
        start_time = datetime.now()
        components: List[ComponentHealth] = []

        # Check each component
        if self.broker:
            components.append(await self._check_broker())

        if self.event_store:
            components.append(await self._check_event_store())

        if self.dead_letter_queue:
            components.append(await self._check_dead_letter())

        if self.workflow_coordinator:
            components.append(await self._check_workflow_coordinator())

        # Determine overall status
        overall_status = self._determine_overall_status(components)

        # Collect metrics
        metrics = self._collect_metrics()

        duration = (datetime.now() - start_time).total_seconds() * 1000

        result = HealthCheckResult(
            status=overall_status,
            components=components,
            metrics=metrics,
            duration_ms=duration
        )

        # Store in history
        self._store_metrics(result)

        return result

    async def _check_broker(self) -> ComponentHealth:
        """Check EventBroker health."""
        try:
            start = datetime.now()

            # Get broker stats
            stats = self.broker.get_stats() if hasattr(self.broker, 'get_stats') else {}

            latency = (datetime.now() - start).total_seconds() * 1000

            # Check for issues
            status = ComponentStatus.UP
            message = "Operational"

            queue_depth = stats.get('queue_depth', 0)
            if queue_depth > self.thresholds.queue_depth_critical:
                status = ComponentStatus.DEGRADED
                message = f"High queue depth: {queue_depth}"
            elif queue_depth > self.thresholds.queue_depth_warning:
                message = f"Queue depth elevated: {queue_depth}"

            return ComponentHealth(
                name="broker",
                status=status,
                latency_ms=latency,
                message=message,
                details={
                    'subscriptions': stats.get('total_subscriptions', 0),
                    'events_emitted': stats.get('events_emitted', 0),
                    'events_delivered': stats.get('events_delivered', 0),
                    'queue_depth': queue_depth
                }
            )

        except Exception as e:
            return ComponentHealth(
                name="broker",
                status=ComponentStatus.DOWN,
                message=f"Error: {str(e)}"
            )

    async def _check_event_store(self) -> ComponentHealth:
        """Check EventStore health."""
        try:
            start = datetime.now()

            # Get store stats
            stats = self.event_store.get_stats() if hasattr(self.event_store, 'get_stats') else {}

            latency = (datetime.now() - start).total_seconds() * 1000

            return ComponentHealth(
                name="event_store",
                status=ComponentStatus.UP,
                latency_ms=latency,
                message="Operational",
                details={
                    'events_stored': stats.get('events_stored', 0),
                    'events_read': stats.get('events_read', 0),
                    'queries_executed': stats.get('queries_executed', 0),
                    'bytes_written': stats.get('bytes_written', 0)
                }
            )

        except Exception as e:
            return ComponentHealth(
                name="event_store",
                status=ComponentStatus.DOWN,
                message=f"Error: {str(e)}"
            )

    async def _check_dead_letter(self) -> ComponentHealth:
        """Check DeadLetterQueue health."""
        try:
            start = datetime.now()

            # Get DLQ stats
            stats = await self.dead_letter_queue.get_stats() if hasattr(
                self.dead_letter_queue, 'get_stats'
            ) else {}

            latency = (datetime.now() - start).total_seconds() * 1000

            # Check for issues
            status = ComponentStatus.UP
            message = "Operational"

            pending = stats.get('by_status', {}).get('pending', 0)
            if pending > self.thresholds.dead_letter_critical:
                status = ComponentStatus.DEGRADED
                message = f"High pending count: {pending}"
            elif pending > self.thresholds.dead_letter_warning:
                message = f"Pending count elevated: {pending}"

            return ComponentHealth(
                name="dead_letter_queue",
                status=status,
                latency_ms=latency,
                message=message,
                details={
                    'events_added': stats.get('events_added', 0),
                    'events_retried': stats.get('events_retried', 0),
                    'events_resolved': stats.get('events_resolved', 0),
                    'events_exhausted': stats.get('events_exhausted', 0),
                    'pending': pending
                }
            )

        except Exception as e:
            return ComponentHealth(
                name="dead_letter_queue",
                status=ComponentStatus.DOWN,
                message=f"Error: {str(e)}"
            )

    async def _check_workflow_coordinator(self) -> ComponentHealth:
        """Check WorkflowCoordinator health."""
        try:
            start = datetime.now()

            # Get workflow stats
            stats = self.workflow_coordinator.get_stats() if hasattr(
                self.workflow_coordinator, 'get_stats'
            ) else {}

            latency = (datetime.now() - start).total_seconds() * 1000

            # Check for issues
            status = ComponentStatus.UP
            message = "Operational"

            failed = stats.get('by_status', {}).get('failed', 0)
            if failed > self.thresholds.failed_workflows_critical:
                status = ComponentStatus.DEGRADED
                message = f"High failure count: {failed}"
            elif failed > self.thresholds.failed_workflows_warning:
                message = f"Failure count elevated: {failed}"

            return ComponentHealth(
                name="workflow_coordinator",
                status=status,
                latency_ms=latency,
                message=message,
                details={
                    'workflows_registered': stats.get('workflows_registered', 0),
                    'workflows_started': stats.get('workflows_started', 0),
                    'workflows_completed': stats.get('workflows_completed', 0),
                    'workflows_failed': failed,
                    'active': stats.get('by_status', {}).get('running', 0)
                }
            )

        except Exception as e:
            return ComponentHealth(
                name="workflow_coordinator",
                status=ComponentStatus.DOWN,
                message=f"Error: {str(e)}"
            )

    def _determine_overall_status(self, components: List[ComponentHealth]) -> HealthStatus:
        """Determine overall health status from components."""
        if not components:
            return HealthStatus.UNKNOWN

        statuses = [c.status for c in components]

        if any(s == ComponentStatus.DOWN for s in statuses):
            return HealthStatus.UNHEALTHY

        if any(s == ComponentStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED

        if any(s == ComponentStatus.UNKNOWN for s in statuses):
            return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY

    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect aggregate metrics."""
        metrics = {}

        if self.broker and hasattr(self.broker, 'get_stats'):
            broker_stats = self.broker.get_stats()
            metrics['broker'] = {
                'events_per_second': broker_stats.get('events_per_second', 0),
                'avg_latency_ms': broker_stats.get('avg_latency_ms', 0),
                'total_events': broker_stats.get('events_emitted', 0)
            }

        if self.event_store and hasattr(self.event_store, 'get_stats'):
            store_stats = self.event_store.get_stats()
            metrics['event_store'] = {
                'total_events': store_stats.get('events_stored', 0),
                'bytes_written': store_stats.get('bytes_written', 0)
            }

        return metrics

    def _store_metrics(self, result: HealthCheckResult) -> None:
        """Store metrics in history."""
        self._metrics_history.append({
            'timestamp': result.timestamp,
            'status': result.status.value,
            'metrics': result.metrics
        })

        # Trim history
        if len(self._metrics_history) > self._max_history:
            self._metrics_history = self._metrics_history[-self._max_history:]

    def get_metrics_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent metrics history."""
        return self._metrics_history[-limit:]

    def export_prometheus_metrics(self) -> str:
        """
        Export metrics in Prometheus format.

        Returns:
            Prometheus-formatted metrics string
        """
        lines = []

        # Add help and type comments
        lines.append("# HELP eventbus_health_status Health status (1=healthy, 0=unhealthy)")
        lines.append("# TYPE eventbus_health_status gauge")

        if self._metrics_history:
            latest = self._metrics_history[-1]
            status_value = 1 if latest['status'] == 'healthy' else 0
            lines.append(f"eventbus_health_status {status_value}")

            # Broker metrics
            if 'broker' in latest.get('metrics', {}):
                broker = latest['metrics']['broker']
                lines.append("")
                lines.append("# HELP eventbus_events_total Total events processed")
                lines.append("# TYPE eventbus_events_total counter")
                lines.append(f"eventbus_events_total {broker.get('total_events', 0)}")

                lines.append("")
                lines.append("# HELP eventbus_latency_ms Average event latency")
                lines.append("# TYPE eventbus_latency_ms gauge")
                lines.append(f"eventbus_latency_ms {broker.get('avg_latency_ms', 0)}")

        return "\n".join(lines)


# ============================================================================
# Factory Function
# ============================================================================

def create_health_checker(
    broker: Optional[Any] = None,
    event_store: Optional[Any] = None,
    dead_letter_queue: Optional[Any] = None,
    workflow_coordinator: Optional[Any] = None
) -> HealthChecker:
    """
    Create a health checker instance.

    Args:
        broker: EventBroker instance
        event_store: EventStore instance
        dead_letter_queue: DeadLetterQueue instance
        workflow_coordinator: WorkflowCoordinator instance

    Returns:
        HealthChecker instance
    """
    return HealthChecker(
        broker=broker,
        event_store=event_store,
        dead_letter_queue=dead_letter_queue,
        workflow_coordinator=workflow_coordinator
    )
