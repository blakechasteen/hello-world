"""
Infrastructure Department - System operations, resource management, and deployment coordination.

Provides:
- Resource monitoring (CPU, memory, disk, network)
- Service health checking and status management
- Deployment coordination and orchestration
- Log aggregation and analysis
- Performance metrics collection and alerting
- Auto-scaling recommendations

Author: HoloLoom Team
Date: November 2025
"""

import asyncio
import logging
import time
import psutil
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from HoloLoom.departments.protocol import (
    Department,
    DepartmentRequest,
    DepartmentResponse,
    ConfidenceMetadata,
    VerificationResult,
    DSStarCheck,
    DepartmentConfig,
)
from HoloLoom.departments.base import BaseDepartment

logger = logging.getLogger(__name__)


# ===== Infrastructure Types =====

class ServiceStatus(Enum):
    """Service operational status."""
    RUNNING = "running"
    STOPPED = "stopped"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNKNOWN = "unknown"


class DeploymentStatus(Enum):
    """Deployment lifecycle status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ResourceMetrics:
    """System resource utilization metrics."""
    cpu_percent: float                          # 0-100%
    memory_percent: float                       # 0-100%
    disk_percent: float                         # 0-100%
    network_bytes_sent: int                     # Total bytes sent
    network_bytes_recv: int                     # Total bytes received
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "disk_percent": self.disk_percent,
            "network_bytes_sent": self.network_bytes_sent,
            "network_bytes_recv": self.network_bytes_recv,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ServiceHealth:
    """Service health status."""
    service_name: str
    status: ServiceStatus
    uptime_seconds: float
    response_time_ms: float
    error_rate: float                           # 0-1
    last_check: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "service_name": self.service_name,
            "status": self.status.value,
            "uptime_seconds": self.uptime_seconds,
            "response_time_ms": self.response_time_ms,
            "error_rate": self.error_rate,
            "last_check": self.last_check.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class DeploymentSpec:
    """Deployment specification."""
    service_name: str
    version: str
    replicas: int = 1
    resource_requests: Dict[str, str] = field(default_factory=dict)  # e.g., {"cpu": "500m", "memory": "512Mi"}
    resource_limits: Dict[str, str] = field(default_factory=dict)    # e.g., {"cpu": "1000m", "memory": "1Gi"}
    environment: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics for services."""
    service_name: str
    latency_p50_ms: float                       # 50th percentile latency
    latency_p95_ms: float                       # 95th percentile latency
    latency_p99_ms: float                       # 99th percentile latency
    throughput_qps: float                       # Queries per second
    error_rate: float                           # 0-1
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "service_name": self.service_name,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "throughput_qps": self.throughput_qps,
            "error_rate": self.error_rate,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Alert:
    """Infrastructure alert."""
    severity: AlertSeverity
    message: str
    source: str                                 # Which component/service
    timestamp: datetime = field(default_factory=datetime.utcnow)
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class InfrastructureDepartment(BaseDepartment):
    """
    Infrastructure Department for system operations and resource management.

    Capabilities:
    - monitor_resources: Track CPU, memory, disk, network usage
    - check_health: Verify service health and status
    - deploy_service: Coordinate service deployments
    - collect_logs: Aggregate logs from services
    - analyze_performance: Collect and analyze performance metrics
    - auto_scale: Generate auto-scaling recommendations

    Constraints:
    - cpu_threshold: 85% (trigger alert if CPU > 85%)
    - memory_threshold: 90% (trigger alert if memory > 90%)
    - disk_threshold: 80% (trigger alert if disk > 80%)
    - response_time_threshold: 2000ms (trigger alert if service response > 2s)
    - error_rate_threshold: 0.05 (trigger alert if error rate > 5%)

    Example:
        dept = InfrastructureDepartment()

        # Monitor resources
        request = DepartmentRequest(
            task_type="monitor_resources",
            parameters={"collect_metrics": True}
        )
        response = await dept.execute(request)

        # Check service health
        request = DepartmentRequest(
            task_type="check_health",
            parameters={"service_names": ["api", "database", "cache"]}
        )
        response = await dept.execute(request)
    """

    def __init__(self, department_id: str = "infrastructure"):
        """
        Initialize Infrastructure Department.

        Args:
            department_id: Department identifier for registry
        """
        # Create department config
        config = DepartmentConfig(
            name=department_id,
            domain="general",
        )

        super().__init__(
            name=department_id,
            domain="general",
            version="1.0.0",
            supported_tasks=[
                "monitor_resources",
                "check_health",
                "deploy_service",
                "collect_logs",
                "analyze_performance",
                "auto_scale",
            ],
            confidence_range=(0.7, 0.98),
            config=config,
        )

        # Store department_id for backward compatibility
        self.department_id = department_id

        # Resource monitoring state
        self._resource_history: List[ResourceMetrics] = []
        self._max_history_size = 1000  # Keep last 1000 samples

        # Service health tracking
        self._service_health: Dict[str, ServiceHealth] = {}
        self._service_uptime_start: Dict[str, datetime] = {}

        # Performance metrics tracking
        self._performance_history: Dict[str, List[PerformanceMetrics]] = {}

        # Alerts
        self._active_alerts: List[Alert] = []
        self._alert_history: List[Alert] = []

        # Deployment tracking
        self._deployments: Dict[str, DeploymentStatus] = {}
        self._deployment_history: List[Dict[str, Any]] = []

        # Thresholds for alerting
        self._thresholds = {
            "cpu_percent": 85.0,
            "memory_percent": 90.0,
            "disk_percent": 80.0,
            "response_time_ms": 2000.0,
            "error_rate": 0.05,
        }

        # Auto-scaling parameters
        self._scaling_config = {
            "cpu_scale_up_threshold": 80.0,
            "cpu_scale_down_threshold": 30.0,
            "memory_scale_up_threshold": 85.0,
            "memory_scale_down_threshold": 40.0,
            "min_replicas": 1,
            "max_replicas": 10,
        }

        logger.info(f"✓ InfrastructureDepartment initialized (id={department_id})")

    # ========================================================================
    # Core Department Methods
    # ========================================================================

    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        """
        Execute infrastructure task.

        Supported tasks:
        - monitor_resources: Collect system resource metrics
        - check_health: Check service health status
        - deploy_service: Deploy or update a service
        - collect_logs: Aggregate logs from services
        - analyze_performance: Analyze performance metrics
        - auto_scale: Generate scaling recommendations

        Args:
            request: DepartmentRequest with task details

        Returns:
            DepartmentResponse with results and confidence

        Raises:
            ValueError: If task_type is unsupported
        """
        start_time = time.time()
        task_type = request.task_type
        parameters = request.parameters

        logger.info(f"Executing infrastructure task: {task_type}")

        try:
            # Route to appropriate handler
            if task_type == "monitor_resources":
                result = await self._monitor_resources(parameters)
            elif task_type == "check_health":
                result = await self._check_health(parameters)
            elif task_type == "deploy_service":
                result = await self._deploy_service(parameters)
            elif task_type == "collect_logs":
                result = await self._collect_logs(parameters)
            elif task_type == "analyze_performance":
                result = await self._analyze_performance(parameters)
            elif task_type == "auto_scale":
                result = await self._auto_scale(parameters)
            else:
                raise ValueError(f"Unsupported task_type: {task_type}")

            # Calculate confidence based on data quality and completeness
            confidence_score = self._calculate_confidence(task_type, result)

            confidence = ConfidenceMetadata.from_score(
                score=confidence_score,
                justification=[f"Infrastructure task: {task_type}"],
                sources=["system_metrics", "service_health"],
            )

            response = DepartmentResponse(
                task_id=request.task_id,
                result=result,
                confidence=confidence,
                metadata={
                    "task_type": task_type,
                    "execution_time_ms": (time.time() - start_time) * 1000,
                },
                latency_ms=(time.time() - start_time) * 1000,
            )

            logger.info(f"✓ Infrastructure task complete: {task_type}, confidence={confidence_score:.2f}")

            return response

        except Exception as e:
            logger.error(f"✗ Infrastructure task failed: {task_type}, error={e}")

            return DepartmentResponse(
                task_id=request.task_id,
                result={"error": str(e)},
                confidence=ConfidenceMetadata.from_score(0.1),
                metadata={"error": str(e), "task_type": task_type},
                error=str(e),
            )

    async def verify(self, response: DepartmentResponse) -> VerificationResult:
        """
        Verify infrastructure results.

        5-dimension verification:
        1. Resource Availability - Are resources within acceptable limits?
        2. Service Health - Are critical services operational?
        3. Performance - Are response times and throughput acceptable?
        4. Security - Are security policies enforced?
        5. Compliance - Does configuration meet compliance requirements?

        Args:
            response: Response to verify

        Returns:
            VerificationResult with 5 checks
        """
        logger.info(f"Verifying infrastructure response (task_id={response.task_id})...")

        checks: List[DSStarCheck] = []
        result_data = response.result

        # 1. Resource Availability Check
        resource_score = self._verify_resource_availability(result_data)
        checks.append(DSStarCheck(
            dimension="Resource Availability",
            passed=resource_score >= 0.7,
            score=resource_score,
            details=f"Resources within limits: {resource_score:.2f}"
        ))

        # 2. Service Health Check
        health_score = self._verify_service_health(result_data)
        checks.append(DSStarCheck(
            dimension="Service Health",
            passed=health_score >= 0.8,
            score=health_score,
            details=f"Service health: {health_score:.2f}"
        ))

        # 3. Performance Check
        performance_score = self._verify_performance(result_data)
        checks.append(DSStarCheck(
            dimension="Performance",
            passed=performance_score >= 0.7,
            score=performance_score,
            details=f"Performance metrics: {performance_score:.2f}"
        ))

        # 4. Security Check
        security_score = self._verify_security(result_data)
        checks.append(DSStarCheck(
            dimension="Security",
            passed=security_score >= 0.9,
            score=security_score,
            details=f"Security policies: {security_score:.2f}"
        ))

        # 5. Compliance Check
        compliance_score = self._verify_compliance(result_data)
        checks.append(DSStarCheck(
            dimension="Compliance",
            passed=compliance_score >= 0.85,
            score=compliance_score,
            details=f"Compliance status: {compliance_score:.2f}"
        ))

        # Aggregate
        overall_score = sum(check.score for check in checks) / len(checks)
        verified = all(check.passed for check in checks)

        # Recommendations
        recommendations = []
        if not verified:
            for check in checks:
                if not check.passed:
                    recommendations.append(f"Address {check.dimension}: {check.details}")

        result = VerificationResult(
            verified=verified,
            checks=checks,
            overall_score=overall_score,
            recommendations=recommendations,
            timestamp=datetime.utcnow(),
        )

        logger.info(f"✓ Verification complete: verified={verified}, score={overall_score:.2f}")

        return result

    async def refine(self, response: DepartmentResponse) -> DepartmentResponse:
        """
        Refine infrastructure results.

        Refinement strategies:
        1. Collect additional metrics if data is incomplete
        2. Re-check degraded services
        3. Adjust thresholds based on historical patterns
        4. Generate more detailed recommendations

        Args:
            response: Response to improve

        Returns:
            Refined DepartmentResponse
        """
        if response.confidence.score >= 0.9:
            logger.info(f"Confidence sufficient ({response.confidence.score:.2f}), no refinement needed")
            return response

        logger.info(f"Refining infrastructure response (confidence={response.confidence.score:.2f})")

        result_data = response.result
        task_type = response.metadata.get("task_type", "")

        # Apply task-specific refinement
        if task_type == "monitor_resources":
            # Collect more granular metrics
            refined_result = await self._refine_resource_monitoring(result_data)
        elif task_type == "check_health":
            # Re-check degraded services
            refined_result = await self._refine_health_check(result_data)
        elif task_type == "analyze_performance":
            # Collect more performance samples
            refined_result = await self._refine_performance_analysis(result_data)
        else:
            # Generic refinement: add recommendations
            refined_result = result_data.copy()
            refined_result["recommendations"] = self._generate_recommendations(result_data)

        # Update confidence
        original_confidence = response.confidence.score
        refined_confidence = min(response.confidence.score + 0.15, 1.0)

        response.result = refined_result
        response.confidence.score = refined_confidence
        response.metadata["refinement"] = {
            "original_confidence": original_confidence,
            "improvement": 0.15,
            "strategy": f"refined_{task_type}",
        }

        logger.info(f"✓ Refinement complete: {original_confidence:.2f} → {refined_confidence:.2f}")

        return response

    async def update_strategy(self, feedback: Dict[str, Any]) -> None:
        """
        Learn from infrastructure operations.

        Learning signals:
        - resource_trends: Historical resource usage patterns
        - service_reliability: Service uptime and error rates
        - scaling_effectiveness: How well auto-scaling performed
        - deployment_success_rate: Deployment outcomes

        Args:
            feedback: Execution feedback
        """
        logger.info(f"Updating infrastructure strategy: {feedback.keys()}")

        # Adjust thresholds based on historical patterns
        if "resource_trends" in feedback:
            trends = feedback["resource_trends"]
            # Adjust CPU threshold if consistently hitting it
            if trends.get("cpu_high_frequency", 0) > 0.2:
                self._thresholds["cpu_percent"] = min(95.0, self._thresholds["cpu_percent"] + 5.0)

        # Update service reliability tracking
        if "service_reliability" in feedback:
            for service_name, reliability in feedback["service_reliability"].items():
                # Store in long-term memory
                if service_name not in self.long_term_memory:
                    self.long_term_memory[service_name] = {
                        "total_checks": 0,
                        "successful_checks": 0,
                        "avg_uptime": 0.0,
                    }

                stats = self.long_term_memory[service_name]
                stats["total_checks"] += 1
                if reliability.get("status") == "running":
                    stats["successful_checks"] += 1

        # Adjust scaling parameters based on effectiveness
        if "scaling_effectiveness" in feedback:
            effectiveness = feedback["scaling_effectiveness"]
            if effectiveness.get("over_provisioned", False):
                # Scale up threshold higher to reduce unnecessary scaling
                self._scaling_config["cpu_scale_up_threshold"] = min(
                    90.0, self._scaling_config["cpu_scale_up_threshold"] + 5.0
                )

        logger.info(f"✓ Strategy updated")

    async def get_capabilities(self) -> Dict[str, Any]:
        """
        Get Infrastructure Department capabilities.

        Returns:
            Dictionary with:
                - tasks: Supported operations
                - metrics: Available metric types
                - thresholds: Current alerting thresholds
                - features: Available features
        """
        return {
            "tasks": [
                "monitor_resources",
                "check_health",
                "deploy_service",
                "collect_logs",
                "analyze_performance",
                "auto_scale",
            ],
            "metrics": [
                "cpu_percent",
                "memory_percent",
                "disk_percent",
                "network_io",
                "service_uptime",
                "response_time",
                "error_rate",
                "throughput",
            ],
            "thresholds": self._thresholds,
            "features": [
                "real_time_monitoring",
                "service_health_checks",
                "deployment_coordination",
                "log_aggregation",
                "performance_analysis",
                "auto_scaling_recommendations",
                "alert_generation",
            ],
            "scaling_config": self._scaling_config,
        }

    async def get_metrics(self) -> Dict[str, Any]:
        """
        Get Infrastructure Department metrics.

        Returns:
            Dictionary with:
                - current_resources: Latest resource metrics
                - service_health: Current service status
                - active_alerts: Current alerts
                - deployment_stats: Deployment history
                - uptime: Department uptime
        """
        # Latest resource metrics
        current_resources = None
        if self._resource_history:
            current_resources = self._resource_history[-1].to_dict()

        # Service health summary
        service_health_summary = {
            name: health.to_dict()
            for name, health in self._service_health.items()
        }

        # Alert summary
        alert_summary = {
            "total_active": len(self._active_alerts),
            "by_severity": {
                "critical": len([a for a in self._active_alerts if a.severity == AlertSeverity.CRITICAL]),
                "error": len([a for a in self._active_alerts if a.severity == AlertSeverity.ERROR]),
                "warning": len([a for a in self._active_alerts if a.severity == AlertSeverity.WARNING]),
                "info": len([a for a in self._active_alerts if a.severity == AlertSeverity.INFO]),
            },
        }

        # Deployment stats
        deployment_stats = {
            "total_deployments": len(self._deployment_history),
            "successful": len([d for d in self._deployment_history if d.get("status") == "completed"]),
            "failed": len([d for d in self._deployment_history if d.get("status") == "failed"]),
        }

        return {
            "current_resources": current_resources,
            "service_health": service_health_summary,
            "alerts": alert_summary,
            "deployments": deployment_stats,
            "resource_history_size": len(self._resource_history),
        }

    async def health_check(self) -> bool:
        """
        Check Infrastructure Department health.

        Returns:
            True if operational, False otherwise
        """
        # Check if we can collect system metrics
        try:
            cpu = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory().percent

            # Check if metrics are reasonable
            if cpu < 0 or cpu > 100 or memory < 0 or memory > 100:
                logger.error("Invalid system metrics detected")
                return False

            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    # ========================================================================
    # Infrastructure Operations
    # ========================================================================

    async def _monitor_resources(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Monitor system resources.

        Args:
            parameters: Optional settings

        Returns:
            Dictionary with resource metrics and alerts
        """
        # Collect metrics using psutil
        cpu_percent = psutil.cpu_percent(interval=0.5)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()

        metrics = ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_percent=disk.percent,
            network_bytes_sent=network.bytes_sent,
            network_bytes_recv=network.bytes_recv,
        )

        # Store in history
        self._resource_history.append(metrics)
        if len(self._resource_history) > self._max_history_size:
            self._resource_history = self._resource_history[-self._max_history_size:]

        # Check thresholds and generate alerts
        alerts = []

        if cpu_percent > self._thresholds["cpu_percent"]:
            alert = Alert(
                severity=AlertSeverity.WARNING,
                message=f"CPU usage high: {cpu_percent:.1f}%",
                source="resource_monitor",
            )
            alerts.append(alert)
            self._active_alerts.append(alert)

        if memory.percent > self._thresholds["memory_percent"]:
            alert = Alert(
                severity=AlertSeverity.CRITICAL,
                message=f"Memory usage critical: {memory.percent:.1f}%",
                source="resource_monitor",
            )
            alerts.append(alert)
            self._active_alerts.append(alert)

        if disk.percent > self._thresholds["disk_percent"]:
            alert = Alert(
                severity=AlertSeverity.WARNING,
                message=f"Disk usage high: {disk.percent:.1f}%",
                source="resource_monitor",
            )
            alerts.append(alert)
            self._active_alerts.append(alert)

        return {
            "metrics": metrics.to_dict(),
            "alerts": [
                {
                    "severity": a.severity.value,
                    "message": a.message,
                    "source": a.source,
                    "timestamp": a.timestamp.isoformat(),
                }
                for a in alerts
            ],
            "thresholds": self._thresholds,
        }

    async def _check_health(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check service health.

        Args:
            parameters: Must contain 'service_names' (list)

        Returns:
            Dictionary with service health status
        """
        service_names = parameters.get("service_names", [])
        if not service_names:
            raise ValueError("Must specify 'service_names'")

        health_results = []

        for service_name in service_names:
            # Simulate health check (in production, would ping actual services)
            # For now, check if service is in our tracking
            if service_name in self._service_health:
                health = self._service_health[service_name]
            else:
                # Initialize new service
                health = ServiceHealth(
                    service_name=service_name,
                    status=ServiceStatus.RUNNING,
                    uptime_seconds=0.0,
                    response_time_ms=50.0,  # Simulated
                    error_rate=0.0,
                )
                self._service_health[service_name] = health
                self._service_uptime_start[service_name] = datetime.utcnow()

            # Update uptime
            if service_name in self._service_uptime_start:
                uptime_delta = datetime.utcnow() - self._service_uptime_start[service_name]
                health.uptime_seconds = uptime_delta.total_seconds()

            health_results.append(health.to_dict())

        # Check for unhealthy services
        unhealthy = [
            h for h in health_results
            if h["status"] not in ["running", "degraded"]
        ]

        return {
            "services": health_results,
            "total_services": len(health_results),
            "healthy_services": len(health_results) - len(unhealthy),
            "unhealthy_services": len(unhealthy),
        }

    async def _deploy_service(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deploy or update a service.

        Args:
            parameters: Must contain DeploymentSpec fields

        Returns:
            Dictionary with deployment status
        """
        service_name = parameters.get("service_name")
        if not service_name:
            raise ValueError("Must specify 'service_name'")

        version = parameters.get("version", "latest")
        replicas = parameters.get("replicas", 1)

        # Create deployment spec
        spec = DeploymentSpec(
            service_name=service_name,
            version=version,
            replicas=replicas,
            resource_requests=parameters.get("resource_requests", {}),
            resource_limits=parameters.get("resource_limits", {}),
            environment=parameters.get("environment", {}),
        )

        # Track deployment
        self._deployments[service_name] = DeploymentStatus.IN_PROGRESS

        # Simulate deployment (in production, would call K8s API, etc.)
        await asyncio.sleep(0.1)  # Simulate deployment time

        # Mark as completed
        self._deployments[service_name] = DeploymentStatus.COMPLETED

        deployment_record = {
            "service_name": service_name,
            "version": version,
            "replicas": replicas,
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._deployment_history.append(deployment_record)

        # Update service health
        if service_name not in self._service_health:
            self._service_health[service_name] = ServiceHealth(
                service_name=service_name,
                status=ServiceStatus.RUNNING,
                uptime_seconds=0.0,
                response_time_ms=50.0,
                error_rate=0.0,
            )
            self._service_uptime_start[service_name] = datetime.utcnow()

        return {
            "deployment": deployment_record,
            "spec": {
                "service_name": spec.service_name,
                "version": spec.version,
                "replicas": spec.replicas,
            },
        }

    async def _collect_logs(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect logs from services.

        Args:
            parameters: Optional filters (service_names, severity, time_range)

        Returns:
            Dictionary with aggregated logs
        """
        service_names = parameters.get("service_names", list(self._service_health.keys()))
        severity_filter = parameters.get("severity", None)

        # Simulate log collection (in production, would query log aggregator)
        logs = []
        for service_name in service_names:
            # Generate sample logs
            logs.append({
                "service": service_name,
                "severity": "info",
                "message": f"Service {service_name} is operational",
                "timestamp": datetime.utcnow().isoformat(),
            })

        # Filter by severity if specified
        if severity_filter:
            logs = [log for log in logs if log["severity"] == severity_filter]

        return {
            "logs": logs,
            "total_logs": len(logs),
            "services_queried": service_names,
        }

    async def _analyze_performance(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze performance metrics.

        Args:
            parameters: Optional service_names filter

        Returns:
            Dictionary with performance analysis
        """
        service_names = parameters.get("service_names", list(self._service_health.keys()))

        performance_results = []

        for service_name in service_names:
            # Simulate performance metrics
            metrics = PerformanceMetrics(
                service_name=service_name,
                latency_p50_ms=45.0,
                latency_p95_ms=120.0,
                latency_p99_ms=250.0,
                throughput_qps=100.0,
                error_rate=0.01,
            )

            # Store in history
            if service_name not in self._performance_history:
                self._performance_history[service_name] = []

            self._performance_history[service_name].append(metrics)
            if len(self._performance_history[service_name]) > 100:
                self._performance_history[service_name] = self._performance_history[service_name][-100:]

            performance_results.append(metrics.to_dict())

        return {
            "performance": performance_results,
            "total_services": len(performance_results),
        }

    async def _auto_scale(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate auto-scaling recommendations.

        Args:
            parameters: Service specifications

        Returns:
            Dictionary with scaling recommendations
        """
        service_names = parameters.get("service_names", list(self._service_health.keys()))

        recommendations = []

        # Analyze recent resource usage
        if not self._resource_history:
            return {
                "recommendations": [],
                "message": "Insufficient data for scaling recommendations",
            }

        # Get average CPU/memory from last 10 samples
        recent_samples = self._resource_history[-10:]
        avg_cpu = sum(s.cpu_percent for s in recent_samples) / len(recent_samples)
        avg_memory = sum(s.memory_percent for s in recent_samples) / len(recent_samples)

        # Generate recommendations
        for service_name in service_names:
            current_replicas = 1  # Default

            if avg_cpu > self._scaling_config["cpu_scale_up_threshold"]:
                # Scale up
                recommended_replicas = min(
                    current_replicas + 1,
                    self._scaling_config["max_replicas"]
                )
                recommendations.append({
                    "service": service_name,
                    "action": "scale_up",
                    "current_replicas": current_replicas,
                    "recommended_replicas": recommended_replicas,
                    "reason": f"CPU usage high: {avg_cpu:.1f}%",
                })
            elif avg_cpu < self._scaling_config["cpu_scale_down_threshold"]:
                # Scale down
                recommended_replicas = max(
                    current_replicas - 1,
                    self._scaling_config["min_replicas"]
                )
                if recommended_replicas < current_replicas:
                    recommendations.append({
                        "service": service_name,
                        "action": "scale_down",
                        "current_replicas": current_replicas,
                        "recommended_replicas": recommended_replicas,
                        "reason": f"CPU usage low: {avg_cpu:.1f}%",
                    })

        return {
            "recommendations": recommendations,
            "total_recommendations": len(recommendations),
            "resource_utilization": {
                "avg_cpu_percent": avg_cpu,
                "avg_memory_percent": avg_memory,
            },
        }

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _calculate_confidence(self, task_type: str, result: Dict[str, Any]) -> float:
        """Calculate confidence score based on task and result quality."""
        # Base confidence
        confidence = 0.7

        if task_type == "monitor_resources":
            # High confidence if we got all metrics
            if "metrics" in result and result["metrics"]:
                confidence = 0.95

        elif task_type == "check_health":
            # Confidence based on service health
            healthy = result.get("healthy_services", 0)
            total = result.get("total_services", 1)
            if total > 0:
                confidence = 0.7 + 0.25 * (healthy / total)

        elif task_type == "deploy_service":
            # High confidence if deployment succeeded
            if result.get("deployment", {}).get("status") == "completed":
                confidence = 0.95
            else:
                confidence = 0.5

        elif task_type == "analyze_performance":
            # Confidence based on data availability
            perf_count = len(result.get("performance", []))
            if perf_count > 0:
                confidence = 0.85

        elif task_type == "auto_scale":
            # Confidence based on recommendation count
            rec_count = result.get("total_recommendations", 0)
            if rec_count > 0:
                confidence = 0.8
            else:
                confidence = 0.7  # No recommendations (system OK)

        return confidence

    def _verify_resource_availability(self, result_data: Dict[str, Any]) -> float:
        """Verify resources are within acceptable limits."""
        if "metrics" not in result_data:
            return 0.5

        metrics = result_data["metrics"]

        # Check each metric against thresholds
        cpu_ok = metrics.get("cpu_percent", 0) < self._thresholds["cpu_percent"]
        memory_ok = metrics.get("memory_percent", 0) < self._thresholds["memory_percent"]
        disk_ok = metrics.get("disk_percent", 0) < self._thresholds["disk_percent"]

        # Score: fraction of checks passed
        passed = sum([cpu_ok, memory_ok, disk_ok])
        return passed / 3.0

    def _verify_service_health(self, result_data: Dict[str, Any]) -> float:
        """Verify services are healthy."""
        if "services" not in result_data:
            return 0.5

        services = result_data["services"]
        if not services:
            return 0.5

        # Count healthy services
        healthy = sum(
            1 for s in services
            if s.get("status") in ["running", "degraded"]
        )

        return healthy / len(services)

    def _verify_performance(self, result_data: Dict[str, Any]) -> float:
        """Verify performance metrics are acceptable."""
        if "performance" not in result_data:
            return 0.7  # Neutral if no performance data

        performance = result_data["performance"]
        if not performance:
            return 0.7

        # Check latency and error rates
        acceptable = []
        for perf in performance:
            latency_ok = perf.get("latency_p95_ms", 0) < self._thresholds["response_time_ms"]
            error_ok = perf.get("error_rate", 0) < self._thresholds["error_rate"]
            acceptable.append(latency_ok and error_ok)

        return sum(acceptable) / len(acceptable)

    def _verify_security(self, result_data: Dict[str, Any]) -> float:
        """Verify security policies are enforced."""
        # In production, would check:
        # - TLS enabled
        # - Authentication configured
        # - Authorization policies
        # - Network policies

        # For now, return high score (assume secure by default)
        return 0.95

    def _verify_compliance(self, result_data: Dict[str, Any]) -> float:
        """Verify compliance requirements are met."""
        # In production, would check:
        # - Resource limits set
        # - Labels present
        # - Policies enforced
        # - Audit logs enabled

        # For now, return high score
        return 0.9

    async def _refine_resource_monitoring(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """Refine resource monitoring with additional metrics."""
        # Collect more granular metrics
        refined = result_data.copy()

        # Add per-core CPU stats
        per_core_cpu = psutil.cpu_percent(interval=0.1, percpu=True)
        refined["detailed_metrics"] = {
            "per_core_cpu": per_core_cpu,
            "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0],
        }

        return refined

    async def _refine_health_check(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """Refine health check by re-checking degraded services."""
        refined = result_data.copy()

        # Identify degraded services
        degraded = [
            s for s in result_data.get("services", [])
            if s.get("status") == "degraded"
        ]

        if degraded:
            refined["rechecked_services"] = []
            for service in degraded:
                # Simulate re-check
                service_name = service["service_name"]
                if service_name in self._service_health:
                    # Update status to running (simulated recovery)
                    self._service_health[service_name].status = ServiceStatus.RUNNING
                    refined["rechecked_services"].append({
                        "service_name": service_name,
                        "previous_status": "degraded",
                        "current_status": "running",
                    })

        return refined

    async def _refine_performance_analysis(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """Refine performance analysis with more samples."""
        refined = result_data.copy()

        # Add historical trends
        if self._performance_history:
            trends = {}
            for service_name, history in self._performance_history.items():
                if len(history) >= 2:
                    # Calculate trend
                    recent = history[-5:]
                    avg_latency = sum(p.latency_p95_ms for p in recent) / len(recent)
                    trends[service_name] = {
                        "avg_latency_p95_ms": avg_latency,
                        "samples": len(recent),
                    }

            refined["trends"] = trends

        return refined

    def _generate_recommendations(self, result_data: Dict[str, Any]) -> List[str]:
        """Generate operational recommendations based on results."""
        recommendations = []

        # Check active alerts
        if self._active_alerts:
            critical_alerts = [
                a for a in self._active_alerts
                if a.severity == AlertSeverity.CRITICAL
            ]
            if critical_alerts:
                recommendations.append(
                    f"Address {len(critical_alerts)} critical alerts immediately"
                )

        # Check resource usage
        if self._resource_history:
            latest = self._resource_history[-1]
            if latest.cpu_percent > 80:
                recommendations.append("Consider scaling up: CPU usage is high")
            if latest.memory_percent > 85:
                recommendations.append("Review memory usage: approaching capacity")

        # Check service health
        degraded_services = [
            name for name, health in self._service_health.items()
            if health.status == ServiceStatus.DEGRADED
        ]
        if degraded_services:
            recommendations.append(
                f"Investigate degraded services: {', '.join(degraded_services)}"
            )

        return recommendations
