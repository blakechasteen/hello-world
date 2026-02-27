"""
Integration tests for Infrastructure Department with Department Protocol.

Validates:
1. Infrastructure Department implements all 7 protocol methods
2. Resource monitoring collects accurate metrics
3. Service health checks work correctly
4. Deployment coordination functions properly
5. Log collection and aggregation works
6. Performance analysis provides insights
7. Auto-scaling recommendations are generated
8. Verification checks all 5 dimensions
9. Refinement improves low-confidence results
10. Learning signals update department strategy
11. Alerts are generated for threshold violations
12. Metrics tracking is comprehensive

Author: HoloLoom Testing
Date: November 2025
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any

# Import protocol types
from HoloLoom.apps.departments.protocol import (
    Department,
    DepartmentRequest,
    DepartmentResponse,
    ConfidenceMetadata,
    VerificationResult,
    DSStarCheck,
)

# Import Infrastructure Department
from HoloLoom.apps.departments.infrastructure_department import (
    InfrastructureDepartment,
    ResourceMetrics,
    ServiceHealth,
    ServiceStatus,
    DeploymentSpec,
    PerformanceMetrics,
    Alert,
    AlertSeverity,
)


# ===== Fixtures =====

@pytest.fixture
def infrastructure_dept():
    """Create Infrastructure Department instance."""
    return InfrastructureDepartment(department_id="infrastructure")


@pytest.fixture
async def infrastructure_dept_async():
    """Create Infrastructure Department with async context manager."""
    dept = InfrastructureDepartment()
    async with dept:
        yield dept


# ===== Protocol Compliance Tests =====

@pytest.mark.asyncio
async def test_protocol_compliance():
    """Test that Infrastructure Department implements Department protocol."""
    dept = InfrastructureDepartment()

    # Verify all 7 methods exist
    assert hasattr(dept, 'execute')
    assert hasattr(dept, 'verify')
    assert hasattr(dept, 'refine')
    assert hasattr(dept, 'update_strategy')
    assert hasattr(dept, 'get_capabilities')
    assert hasattr(dept, 'get_metrics')
    assert hasattr(dept, 'health_check')

    # Verify all are async
    assert asyncio.iscoroutinefunction(dept.execute)
    assert asyncio.iscoroutinefunction(dept.verify)
    assert asyncio.iscoroutinefunction(dept.refine)
    assert asyncio.iscoroutinefunction(dept.update_strategy)
    assert asyncio.iscoroutinefunction(dept.get_capabilities)
    assert asyncio.iscoroutinefunction(dept.get_metrics)
    assert asyncio.iscoroutinefunction(dept.health_check)


@pytest.mark.asyncio
async def test_department_initialization(infrastructure_dept):
    """Test department initialization and configuration."""
    assert infrastructure_dept.name == "infrastructure"
    assert infrastructure_dept.domain == "general"
    assert infrastructure_dept.version == "1.0.0"
    assert len(infrastructure_dept.supported_tasks) == 6
    assert "monitor_resources" in infrastructure_dept.supported_tasks
    assert "check_health" in infrastructure_dept.supported_tasks
    assert "deploy_service" in infrastructure_dept.supported_tasks


# ===== Resource Monitoring Tests =====

@pytest.mark.asyncio
async def test_monitor_resources(infrastructure_dept):
    """Test resource monitoring functionality."""
    request = DepartmentRequest(
        task_type="monitor_resources",
        parameters={"collect_metrics": True},
    )

    response = await infrastructure_dept.execute(request)

    # Validate response
    assert isinstance(response, DepartmentResponse)
    assert response.task_id == request.task_id
    assert "metrics" in response.result
    assert "alerts" in response.result
    assert "thresholds" in response.result

    # Validate metrics
    metrics = response.result["metrics"]
    assert "cpu_percent" in metrics
    assert "memory_percent" in metrics
    assert "disk_percent" in metrics
    assert "network_bytes_sent" in metrics
    assert "network_bytes_recv" in metrics

    # Validate confidence
    assert 0.0 <= response.confidence.score <= 1.0
    assert response.confidence.score >= 0.7  # Should be high confidence


@pytest.mark.asyncio
async def test_resource_monitoring_alerts(infrastructure_dept):
    """Test that alerts are generated for high resource usage."""
    # Set low thresholds to trigger alerts
    infrastructure_dept._thresholds["cpu_percent"] = 1.0
    infrastructure_dept._thresholds["memory_percent"] = 1.0
    infrastructure_dept._thresholds["disk_percent"] = 1.0

    request = DepartmentRequest(
        task_type="monitor_resources",
        parameters={},
    )

    response = await infrastructure_dept.execute(request)

    # Should have alerts since thresholds are very low
    alerts = response.result.get("alerts", [])
    # At least one resource should exceed 1% threshold
    assert len(alerts) >= 1

    # Check alert structure
    if alerts:
        alert = alerts[0]
        assert "severity" in alert
        assert "message" in alert
        assert "source" in alert
        assert alert["source"] == "resource_monitor"


@pytest.mark.asyncio
async def test_resource_history_tracking(infrastructure_dept):
    """Test that resource metrics are stored in history."""
    # Monitor resources multiple times
    for _ in range(3):
        request = DepartmentRequest(
            task_type="monitor_resources",
            parameters={},
        )
        await infrastructure_dept.execute(request)

    # Check history
    assert len(infrastructure_dept._resource_history) == 3

    # Verify history contains ResourceMetrics
    for metric in infrastructure_dept._resource_history:
        assert isinstance(metric, ResourceMetrics)
        assert hasattr(metric, 'cpu_percent')
        assert hasattr(metric, 'memory_percent')


# ===== Service Health Tests =====

@pytest.mark.asyncio
async def test_check_health(infrastructure_dept):
    """Test service health checking."""
    request = DepartmentRequest(
        task_type="check_health",
        parameters={
            "service_names": ["api", "database", "cache"],
        },
    )

    response = await infrastructure_dept.execute(request)

    # Validate response
    assert isinstance(response, DepartmentResponse)
    assert "services" in response.result
    assert "total_services" in response.result
    assert "healthy_services" in response.result
    assert "unhealthy_services" in response.result

    # Should have 3 services
    services = response.result["services"]
    assert len(services) == 3

    # Check service structure
    for service in services:
        assert "service_name" in service
        assert "status" in service
        assert "uptime_seconds" in service
        assert "response_time_ms" in service
        assert "error_rate" in service


@pytest.mark.asyncio
async def test_service_health_tracking(infrastructure_dept):
    """Test that service health is tracked over time."""
    # Check health twice for same services
    service_names = ["api", "worker"]

    for _ in range(2):
        request = DepartmentRequest(
            task_type="check_health",
            parameters={"service_names": service_names},
        )
        await infrastructure_dept.execute(request)
        await asyncio.sleep(0.1)  # Small delay

    # Verify services are tracked
    assert "api" in infrastructure_dept._service_health
    assert "worker" in infrastructure_dept._service_health

    # Verify uptime is increasing
    api_health = infrastructure_dept._service_health["api"]
    assert api_health.uptime_seconds > 0


@pytest.mark.asyncio
async def test_health_check_missing_services(infrastructure_dept):
    """Test error handling when service_names is missing."""
    request = DepartmentRequest(
        task_type="check_health",
        parameters={},  # No service_names
    )

    response = await infrastructure_dept.execute(request)

    # Should return error response
    assert response.error is not None
    assert "service_names" in response.error


# ===== Deployment Tests =====

@pytest.mark.asyncio
async def test_deploy_service(infrastructure_dept):
    """Test service deployment coordination."""
    request = DepartmentRequest(
        task_type="deploy_service",
        parameters={
            "service_name": "api",
            "version": "v1.2.3",
            "replicas": 3,
            "resource_requests": {"cpu": "500m", "memory": "512Mi"},
            "resource_limits": {"cpu": "1000m", "memory": "1Gi"},
        },
    )

    response = await infrastructure_dept.execute(request)

    # Validate response
    assert isinstance(response, DepartmentResponse)
    assert "deployment" in response.result
    assert "spec" in response.result

    # Check deployment record
    deployment = response.result["deployment"]
    assert deployment["service_name"] == "api"
    assert deployment["version"] == "v1.2.3"
    assert deployment["replicas"] == 3
    assert deployment["status"] == "completed"

    # Verify deployment is tracked
    assert "api" in infrastructure_dept._deployments
    assert len(infrastructure_dept._deployment_history) == 1


@pytest.mark.asyncio
async def test_deployment_updates_service_health(infrastructure_dept):
    """Test that deployment creates/updates service health."""
    request = DepartmentRequest(
        task_type="deploy_service",
        parameters={
            "service_name": "new-service",
            "version": "v1.0.0",
        },
    )

    response = await infrastructure_dept.execute(request)

    # Service should now be in health tracking
    assert "new-service" in infrastructure_dept._service_health
    health = infrastructure_dept._service_health["new-service"]
    assert health.status == ServiceStatus.RUNNING


@pytest.mark.asyncio
async def test_deploy_service_missing_name(infrastructure_dept):
    """Test error handling when service_name is missing."""
    request = DepartmentRequest(
        task_type="deploy_service",
        parameters={},  # No service_name
    )

    response = await infrastructure_dept.execute(request)

    # Should return error response
    assert response.error is not None


# ===== Log Collection Tests =====

@pytest.mark.asyncio
async def test_collect_logs(infrastructure_dept):
    """Test log collection and aggregation."""
    # First create some services
    await infrastructure_dept._check_health({
        "service_names": ["api", "worker", "database"]
    })

    # Now collect logs
    request = DepartmentRequest(
        task_type="collect_logs",
        parameters={
            "service_names": ["api", "worker"],
        },
    )

    response = await infrastructure_dept.execute(request)

    # Validate response
    assert "logs" in response.result
    assert "total_logs" in response.result
    assert "services_queried" in response.result

    logs = response.result["logs"]
    assert len(logs) >= 2  # At least one log per service

    # Check log structure
    for log in logs:
        assert "service" in log
        assert "severity" in log
        assert "message" in log
        assert "timestamp" in log


@pytest.mark.asyncio
async def test_collect_logs_with_severity_filter(infrastructure_dept):
    """Test log collection with severity filtering."""
    # Create services
    await infrastructure_dept._check_health({
        "service_names": ["api"]
    })

    request = DepartmentRequest(
        task_type="collect_logs",
        parameters={
            "service_names": ["api"],
            "severity": "error",
        },
    )

    response = await infrastructure_dept.execute(request)

    # If filtered, should only have logs matching severity
    logs = response.result["logs"]
    # With severity filter "error", might have 0 logs (no errors)
    for log in logs:
        assert log["severity"] == "error"


# ===== Performance Analysis Tests =====

@pytest.mark.asyncio
async def test_analyze_performance(infrastructure_dept):
    """Test performance metrics collection and analysis."""
    # Create some services first
    await infrastructure_dept._check_health({
        "service_names": ["api", "database"]
    })

    request = DepartmentRequest(
        task_type="analyze_performance",
        parameters={
            "service_names": ["api", "database"],
        },
    )

    response = await infrastructure_dept.execute(request)

    # Validate response
    assert "performance" in response.result
    assert "total_services" in response.result

    performance = response.result["performance"]
    assert len(performance) == 2

    # Check performance metrics structure
    for perf in performance:
        assert "service_name" in perf
        assert "latency_p50_ms" in perf
        assert "latency_p95_ms" in perf
        assert "latency_p99_ms" in perf
        assert "throughput_qps" in perf
        assert "error_rate" in perf


@pytest.mark.asyncio
async def test_performance_history_tracking(infrastructure_dept):
    """Test that performance metrics are stored in history."""
    # Create service
    await infrastructure_dept._check_health({"service_names": ["api"]})

    # Analyze performance multiple times
    for _ in range(3):
        request = DepartmentRequest(
            task_type="analyze_performance",
            parameters={"service_names": ["api"]},
        )
        await infrastructure_dept.execute(request)

    # Check history
    assert "api" in infrastructure_dept._performance_history
    assert len(infrastructure_dept._performance_history["api"]) == 3


# ===== Auto-Scaling Tests =====

@pytest.mark.asyncio
async def test_auto_scale_recommendations(infrastructure_dept):
    """Test auto-scaling recommendation generation."""
    # First collect some resource metrics
    monitor_request = DepartmentRequest(
        task_type="monitor_resources",
        parameters={},
    )
    await infrastructure_dept.execute(monitor_request)

    # Create services
    await infrastructure_dept._check_health({
        "service_names": ["api", "worker"]
    })

    # Get scaling recommendations
    request = DepartmentRequest(
        task_type="auto_scale",
        parameters={
            "service_names": ["api", "worker"],
        },
    )

    response = await infrastructure_dept.execute(request)

    # Validate response
    assert "recommendations" in response.result
    assert "total_recommendations" in response.result
    assert "resource_utilization" in response.result

    # Check resource utilization
    utilization = response.result["resource_utilization"]
    assert "avg_cpu_percent" in utilization
    assert "avg_memory_percent" in utilization


@pytest.mark.asyncio
async def test_auto_scale_high_cpu(infrastructure_dept):
    """Test scale-up recommendation when CPU is high."""
    # Simulate high CPU by collecting metrics multiple times
    for _ in range(10):
        monitor_request = DepartmentRequest(
            task_type="monitor_resources",
            parameters={},
        )
        await infrastructure_dept.execute(monitor_request)

    # Manually set high CPU to force scale-up
    if infrastructure_dept._resource_history:
        for metric in infrastructure_dept._resource_history[-5:]:
            metric.cpu_percent = 90.0

    # Create service
    await infrastructure_dept._check_health({"service_names": ["api"]})

    # Get recommendations
    request = DepartmentRequest(
        task_type="auto_scale",
        parameters={"service_names": ["api"]},
    )

    response = await infrastructure_dept.execute(request)

    recommendations = response.result["recommendations"]
    # Should recommend scale-up
    if recommendations:
        rec = recommendations[0]
        assert rec["action"] == "scale_up"
        assert "CPU usage high" in rec["reason"]


@pytest.mark.asyncio
async def test_auto_scale_no_data(infrastructure_dept):
    """Test auto-scaling with no resource history."""
    request = DepartmentRequest(
        task_type="auto_scale",
        parameters={"service_names": ["api"]},
    )

    response = await infrastructure_dept.execute(request)

    # Should return message about insufficient data
    assert "message" in response.result
    assert "Insufficient data" in response.result["message"]


# ===== Verification Tests =====

@pytest.mark.asyncio
async def test_verify_response(infrastructure_dept):
    """Test response verification with 5-dimension checks."""
    # Create a response to verify
    request = DepartmentRequest(
        task_type="monitor_resources",
        parameters={},
    )
    response = await infrastructure_dept.execute(request)

    # Verify the response
    verification = await infrastructure_dept.verify(response)

    # Validate verification result
    assert isinstance(verification, VerificationResult)
    assert len(verification.checks) == 5  # 5 dimensions

    # Check dimension names
    dimension_names = [check.dimension for check in verification.checks]
    assert "Resource Availability" in dimension_names
    assert "Service Health" in dimension_names
    assert "Performance" in dimension_names
    assert "Security" in dimension_names
    assert "Compliance" in dimension_names

    # All checks should be DSStarCheck
    for check in verification.checks:
        assert isinstance(check, DSStarCheck)
        assert hasattr(check, 'dimension')
        assert hasattr(check, 'passed')
        assert hasattr(check, 'score')


@pytest.mark.asyncio
async def test_verification_recommendations(infrastructure_dept):
    """Test that verification generates recommendations for failed checks."""
    # Create a low-confidence response
    low_conf_response = DepartmentResponse(
        task_id="test",
        result={"error": "simulated failure"},
        confidence=ConfidenceMetadata.from_score(0.2),
    )

    verification = await infrastructure_dept.verify(low_conf_response)

    # Should have recommendations if checks failed
    if not verification.verified:
        assert len(verification.recommendations) > 0


# ===== Refinement Tests =====

@pytest.mark.asyncio
async def test_refine_low_confidence(infrastructure_dept):
    """Test refinement improves low-confidence results."""
    # Create low-confidence response
    request = DepartmentRequest(
        task_type="monitor_resources",
        parameters={},
    )
    response = await infrastructure_dept.execute(request)

    # Manually lower confidence
    original_confidence = response.confidence.score
    response.confidence.score = 0.6

    # Refine
    refined_response = await infrastructure_dept.refine(response)

    # Confidence should improve
    assert refined_response.confidence.score > 0.6
    assert "refinement" in refined_response.metadata


@pytest.mark.asyncio
async def test_refine_high_confidence_no_change(infrastructure_dept):
    """Test that high-confidence responses don't need refinement."""
    request = DepartmentRequest(
        task_type="monitor_resources",
        parameters={},
    )
    response = await infrastructure_dept.execute(request)

    # Set high confidence
    response.confidence.score = 0.95

    # Refine
    refined_response = await infrastructure_dept.refine(response)

    # Should not change (already high confidence)
    assert refined_response.confidence.score == 0.95


@pytest.mark.asyncio
async def test_refine_resource_monitoring(infrastructure_dept):
    """Test refinement adds detailed metrics for resource monitoring."""
    request = DepartmentRequest(
        task_type="monitor_resources",
        parameters={},
    )
    response = await infrastructure_dept.execute(request)

    # Lower confidence
    response.confidence.score = 0.65

    # Refine
    refined_response = await infrastructure_dept.refine(response)

    # Should have additional detailed metrics
    assert "detailed_metrics" in refined_response.result or "recommendations" in refined_response.result


# ===== Learning Tests =====

@pytest.mark.asyncio
async def test_update_strategy(infrastructure_dept):
    """Test that update_strategy learns from feedback."""
    feedback = {
        "resource_trends": {
            "cpu_high_frequency": 0.3,  # CPU high 30% of time
        },
        "service_reliability": {
            "api": {"status": "running"},
            "worker": {"status": "degraded"},
        },
    }

    # Get original threshold
    original_threshold = infrastructure_dept._thresholds["cpu_percent"]

    # Update strategy
    await infrastructure_dept.update_strategy(feedback)

    # CPU threshold should increase (since frequently hitting it)
    assert infrastructure_dept._thresholds["cpu_percent"] >= original_threshold


@pytest.mark.asyncio
async def test_update_strategy_scaling_effectiveness(infrastructure_dept):
    """Test learning from scaling effectiveness."""
    feedback = {
        "scaling_effectiveness": {
            "over_provisioned": True,
        },
    }

    original_scale_up = infrastructure_dept._scaling_config["cpu_scale_up_threshold"]

    await infrastructure_dept.update_strategy(feedback)

    # Scale-up threshold should increase (to reduce over-provisioning)
    assert infrastructure_dept._scaling_config["cpu_scale_up_threshold"] >= original_scale_up


# ===== Capabilities and Metrics Tests =====

@pytest.mark.asyncio
async def test_get_capabilities(infrastructure_dept):
    """Test get_capabilities returns complete information."""
    capabilities = await infrastructure_dept.get_capabilities()

    # Validate structure
    assert "tasks" in capabilities
    assert "metrics" in capabilities
    assert "thresholds" in capabilities
    assert "features" in capabilities
    assert "scaling_config" in capabilities

    # Validate tasks
    tasks = capabilities["tasks"]
    assert "monitor_resources" in tasks
    assert "check_health" in tasks
    assert "deploy_service" in tasks
    assert "collect_logs" in tasks
    assert "analyze_performance" in tasks
    assert "auto_scale" in tasks


@pytest.mark.asyncio
async def test_get_metrics(infrastructure_dept):
    """Test get_metrics returns comprehensive metrics."""
    # Execute some operations first
    await infrastructure_dept.execute(DepartmentRequest(
        task_type="monitor_resources",
        parameters={},
    ))

    # Get metrics
    metrics = await infrastructure_dept.get_metrics()

    # Validate structure
    assert "current_resources" in metrics
    assert "service_health" in metrics
    assert "alerts" in metrics
    assert "deployments" in metrics
    assert "resource_history_size" in metrics

    # Should have resource history
    assert metrics["resource_history_size"] > 0


@pytest.mark.asyncio
async def test_health_check(infrastructure_dept):
    """Test department health check."""
    health = await infrastructure_dept.health_check()

    # Should be healthy
    assert health is True


@pytest.mark.asyncio
async def test_unsupported_task_type(infrastructure_dept):
    """Test error handling for unsupported task types."""
    request = DepartmentRequest(
        task_type="unsupported_task",
        parameters={},
    )

    response = await infrastructure_dept.execute(request)

    # Should return error response
    assert response.error is not None
    assert "Unsupported task_type" in response.error


# ===== Integration Tests =====

@pytest.mark.asyncio
async def test_end_to_end_monitoring_workflow(infrastructure_dept):
    """Test complete monitoring workflow."""
    # 1. Monitor resources
    monitor_request = DepartmentRequest(
        task_type="monitor_resources",
        parameters={},
    )
    monitor_response = await infrastructure_dept.execute(monitor_request)
    assert monitor_response.confidence.score >= 0.7

    # 2. Check service health
    health_request = DepartmentRequest(
        task_type="check_health",
        parameters={"service_names": ["api", "database"]},
    )
    health_response = await infrastructure_dept.execute(health_request)
    assert "services" in health_response.result

    # 3. Analyze performance
    perf_request = DepartmentRequest(
        task_type="analyze_performance",
        parameters={"service_names": ["api", "database"]},
    )
    perf_response = await infrastructure_dept.execute(perf_request)
    assert "performance" in perf_response.result

    # 4. Get scaling recommendations
    scale_request = DepartmentRequest(
        task_type="auto_scale",
        parameters={"service_names": ["api", "database"]},
    )
    scale_response = await infrastructure_dept.execute(scale_request)
    assert "recommendations" in scale_response.result


@pytest.mark.asyncio
async def test_concurrent_operations(infrastructure_dept):
    """Test handling concurrent infrastructure operations."""
    # Execute multiple operations concurrently
    tasks = [
        infrastructure_dept.execute(DepartmentRequest(
            task_type="monitor_resources",
            parameters={},
        )),
        infrastructure_dept.execute(DepartmentRequest(
            task_type="check_health",
            parameters={"service_names": ["api"]},
        )),
        infrastructure_dept.execute(DepartmentRequest(
            task_type="collect_logs",
            parameters={"service_names": ["api"]},
        )),
    ]

    responses = await asyncio.gather(*tasks)

    # All should succeed
    assert len(responses) == 3
    for response in responses:
        assert isinstance(response, DepartmentResponse)
        assert response.confidence.score > 0.0
