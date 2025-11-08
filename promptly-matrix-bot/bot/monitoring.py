"""
Phase 6B: Monitoring & Metrics
Prometheus metrics collection, health checks, and performance monitoring.

Features:
- Prometheus metrics endpoint
- Custom metrics (counters, gauges, histograms)
- Health check endpoint
- Performance monitoring (CPU, memory, response times)
- GitHub API usage tracking
- HoloLoom query metrics
- Rate limit metrics

Metrics Collected:
- HTTP requests (total, duration, status codes)
- Matrix bot commands (total, duration, errors)
- GitHub API calls (total, rate limit remaining)
- HoloLoom queries (total, confidence, latency)
- System resources (CPU, memory, disk)
- Authentication events (logins, failures)
"""

import os
import time
import psutil
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        Summary,
        CollectorRegistry,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
except ImportError:
    print("⚠️  prometheus-client not installed. Run: pip install prometheus-client==0.19.0")
    Counter = Gauge = Histogram = Summary = None


class MetricsCollector:
    """Prometheus metrics collector."""

    def __init__(self, registry: Optional[Any] = None):
        """Initialize metrics collector."""
        if Counter is None:
            raise ImportError("prometheus-client not installed")

        self.registry = registry or CollectorRegistry()

        # HTTP Metrics
        self.http_requests_total = Counter(
            "http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status"],
            registry=self.registry,
        )

        self.http_request_duration_seconds = Histogram(
            "http_request_duration_seconds",
            "HTTP request duration",
            ["method", "endpoint"],
            registry=self.registry,
        )

        # Matrix Bot Metrics
        self.bot_commands_total = Counter(
            "bot_commands_total",
            "Total bot commands executed",
            ["command", "status"],  # status: success, error
            registry=self.registry,
        )

        self.bot_command_duration_seconds = Histogram(
            "bot_command_duration_seconds",
            "Bot command execution time",
            ["command"],
            registry=self.registry,
        )

        # GitHub API Metrics
        self.github_api_calls_total = Counter(
            "github_api_calls_total",
            "Total GitHub API calls",
            ["endpoint", "status"],
            registry=self.registry,
        )

        self.github_rate_limit_remaining = Gauge(
            "github_rate_limit_remaining",
            "GitHub API rate limit remaining",
            registry=self.registry,
        )

        # HoloLoom Metrics
        self.hololoom_queries_total = Counter(
            "hololoom_queries_total",
            "Total HoloLoom queries",
            ["complexity"],  # LITE, FAST, FULL, RESEARCH
            registry=self.registry,
        )

        self.hololoom_query_confidence = Histogram(
            "hololoom_query_confidence",
            "HoloLoom query confidence scores",
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry,
        )

        self.hololoom_query_latency_ms = Histogram(
            "hololoom_query_latency_ms",
            "HoloLoom query latency (milliseconds)",
            buckets=[10, 50, 100, 200, 500, 1000, 2000, 5000],
            registry=self.registry,
        )

        # Authentication Metrics
        self.auth_attempts_total = Counter(
            "auth_attempts_total",
            "Total authentication attempts",
            ["method", "status"],  # method: password, api_key; status: success, failure
            registry=self.registry,
        )

        self.active_sessions = Gauge(
            "active_sessions", "Number of active sessions", registry=self.registry
        )

        # Rate Limiting Metrics
        self.rate_limit_hits_total = Counter(
            "rate_limit_hits_total",
            "Total rate limit hits",
            ["user_id"],
            registry=self.registry,
        )

        # System Metrics
        self.system_cpu_percent = Gauge(
            "system_cpu_percent", "System CPU usage (%)", registry=self.registry
        )

        self.system_memory_percent = Gauge(
            "system_memory_percent", "System memory usage (%)", registry=self.registry
        )

        self.system_disk_percent = Gauge(
            "system_disk_percent", "System disk usage (%)", registry=self.registry
        )

        # Application Metrics
        self.app_uptime_seconds = Gauge(
            "app_uptime_seconds", "Application uptime (seconds)", registry=self.registry
        )

        self.app_start_time = time.time()

    def record_http_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record HTTP request metrics."""
        self.http_requests_total.labels(method=method, endpoint=endpoint, status=status).inc()
        self.http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)

    def record_bot_command(self, command: str, status: str, duration: float):
        """Record bot command metrics."""
        self.bot_commands_total.labels(command=command, status=status).inc()
        self.bot_command_duration_seconds.labels(command=command).observe(duration)

    def record_github_api_call(self, endpoint: str, status: int, rate_limit_remaining: int):
        """Record GitHub API call metrics."""
        self.github_api_calls_total.labels(endpoint=endpoint, status=status).inc()
        self.github_rate_limit_remaining.set(rate_limit_remaining)

    def record_hololoom_query(self, complexity: str, confidence: float, latency_ms: float):
        """Record HoloLoom query metrics."""
        self.hololoom_queries_total.labels(complexity=complexity).inc()
        self.hololoom_query_confidence.observe(confidence)
        self.hololoom_query_latency_ms.observe(latency_ms)

    def record_auth_attempt(self, method: str, status: str):
        """Record authentication attempt."""
        self.auth_attempts_total.labels(method=method, status=status).inc()

    def set_active_sessions(self, count: int):
        """Set active sessions count."""
        self.active_sessions.set(count)

    def record_rate_limit_hit(self, user_id: str):
        """Record rate limit hit."""
        self.rate_limit_hits_total.labels(user_id=user_id).inc()

    def update_system_metrics(self):
        """Update system resource metrics."""
        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.system_cpu_percent.set(cpu_percent)

        # Memory
        memory = psutil.virtual_memory()
        self.system_memory_percent.set(memory.percent)

        # Disk
        disk = psutil.disk_usage("/")
        self.system_disk_percent.set(disk.percent)

        # Uptime
        uptime = time.time() - self.app_start_time
        self.app_uptime_seconds.set(uptime)

    def get_metrics(self) -> bytes:
        """Get Prometheus metrics in text format."""
        self.update_system_metrics()
        return generate_latest(self.registry)


class HealthChecker:
    """Health check manager."""

    def __init__(self):
        """Initialize health checker."""
        self.checks: Dict[str, Dict[str, Any]] = {}

    def register_check(self, name: str, check_func, critical: bool = True):
        """
        Register a health check.

        Args:
            name: Check name
            check_func: Async function that returns (bool, str) - (healthy, message)
            critical: Whether failure is critical
        """
        self.checks[name] = {"func": check_func, "critical": critical, "last_check": None}

    async def run_checks(self) -> Dict[str, Any]:
        """
        Run all health checks.

        Returns:
            Health check results
        """
        results = {}
        all_healthy = True

        for name, check in self.checks.items():
            try:
                healthy, message = await check["func"]()

                results[name] = {
                    "healthy": healthy,
                    "message": message,
                    "critical": check["critical"],
                    "timestamp": datetime.utcnow().isoformat(),
                }

                check["last_check"] = results[name]

                if not healthy and check["critical"]:
                    all_healthy = False

            except Exception as e:
                results[name] = {
                    "healthy": False,
                    "message": f"Check failed: {str(e)}",
                    "critical": check["critical"],
                    "timestamp": datetime.utcnow().isoformat(),
                }

                if check["critical"]:
                    all_healthy = False

        return {"healthy": all_healthy, "checks": results, "timestamp": datetime.utcnow().isoformat()}

    def get_status(self) -> str:
        """Get overall health status."""
        if not self.checks:
            return "healthy"

        for check in self.checks.values():
            if check["last_check"] and not check["last_check"]["healthy"] and check["critical"]:
                return "unhealthy"

        return "healthy"


# Example health check functions
async def check_database_connection() -> tuple[bool, str]:
    """Check database connectivity."""
    try:
        # Example: Check if database file exists
        db_path = Path("data/promptly.db")
        if db_path.exists():
            return True, "Database file exists"
        else:
            return False, "Database file not found"
    except Exception as e:
        return False, f"Database check failed: {str(e)}"


async def check_github_api() -> tuple[bool, str]:
    """Check GitHub API connectivity."""
    try:
        import requests

        response = requests.get("https://api.github.com", timeout=5)
        if response.status_code == 200:
            return True, "GitHub API reachable"
        else:
            return False, f"GitHub API returned {response.status_code}"
    except Exception as e:
        return False, f"GitHub API unreachable: {str(e)}"


async def check_system_resources() -> tuple[bool, str]:
    """Check system resource usage."""
    try:
        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        if cpu_percent > 90:
            return False, f"High CPU usage: {cpu_percent}%"

        # Memory
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            return False, f"High memory usage: {memory.percent}%"

        # Disk
        disk = psutil.disk_usage("/")
        if disk.percent > 90:
            return False, f"High disk usage: {disk.percent}%"

        return True, "System resources OK"

    except Exception as e:
        return False, f"Resource check failed: {str(e)}"


# FastAPI integration
def create_metrics_app(metrics: MetricsCollector, health: HealthChecker):
    """Create FastAPI app with metrics and health endpoints."""
    try:
        from fastapi import FastAPI, Response
        from fastapi.responses import JSONResponse
    except ImportError:
        print("⚠️  FastAPI not installed. Run: pip install fastapi uvicorn")
        return None

    app = FastAPI(title="Promptly Bot Metrics", version="1.0.0")

    @app.get("/metrics")
    async def get_metrics():
        """Prometheus metrics endpoint."""
        metrics_data = metrics.get_metrics()
        return Response(content=metrics_data, media_type=CONTENT_TYPE_LATEST)

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        results = await health.run_checks()

        status_code = 200 if results["healthy"] else 503

        return JSONResponse(content=results, status_code=status_code)

    @app.get("/health/live")
    async def liveness_check():
        """Liveness check (always returns 200 if app is running)."""
        return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}

    @app.get("/health/ready")
    async def readiness_check():
        """Readiness check (returns 200 if app is ready to serve traffic)."""
        results = await health.run_checks()

        status_code = 200 if results["healthy"] else 503

        return JSONResponse(
            content={"status": "ready" if results["healthy"] else "not ready", "checks": results},
            status_code=status_code,
        )

    return app


# Example usage
if __name__ == "__main__":
    import asyncio

    # Initialize metrics collector
    metrics = MetricsCollector()

    # Simulate some metrics
    print("📊 Recording sample metrics...")

    # HTTP requests
    metrics.record_http_request("GET", "/api/query", 200, 0.150)
    metrics.record_http_request("POST", "/api/command", 200, 0.320)
    metrics.record_http_request("GET", "/metrics", 200, 0.005)

    # Bot commands
    metrics.record_bot_command("pr_create", "success", 2.5)
    metrics.record_bot_command("review_pr", "success", 5.3)
    metrics.record_bot_command("issue_create", "error", 0.1)

    # GitHub API
    metrics.record_github_api_call("pulls", 200, 4850)
    metrics.record_github_api_call("issues", 200, 4849)

    # HoloLoom
    metrics.record_hololoom_query("FAST", 0.92, 145.5)
    metrics.record_hololoom_query("FULL", 0.88, 287.3)

    # Auth
    metrics.record_auth_attempt("password", "success")
    metrics.record_auth_attempt("api_key", "success")
    metrics.record_auth_attempt("password", "failure")

    # Sessions
    metrics.set_active_sessions(5)

    # System metrics
    metrics.update_system_metrics()

    print("✅ Metrics recorded\n")

    # Get metrics
    metrics_output = metrics.get_metrics().decode("utf-8")
    print("📊 Prometheus Metrics (sample):")
    print("=" * 60)
    for line in metrics_output.split("\n")[:30]:
        if line and not line.startswith("#"):
            print(line)
    print("=" * 60)

    # Health checks
    async def run_health_checks():
        health = HealthChecker()

        # Register checks
        health.register_check("database", check_database_connection, critical=True)
        health.register_check("github_api", check_github_api, critical=False)
        health.register_check("system_resources", check_system_resources, critical=True)

        # Run checks
        results = await health.run_checks()

        print("\n🏥 Health Check Results:")
        print(f"Overall: {'✅ HEALTHY' if results['healthy'] else '❌ UNHEALTHY'}")
        print()

        for name, check in results["checks"].items():
            emoji = "✅" if check["healthy"] else "❌"
            critical = "🚨 CRITICAL" if check["critical"] else "ℹ️  Non-critical"
            print(f"{emoji} {name} ({critical}): {check['message']}")

    asyncio.run(run_health_checks())

    # Show how to run metrics server
    print("\n" + "=" * 60)
    print("To start metrics server:")
    print("  uvicorn bot.monitoring:app --host 0.0.0.0 --port 9090")
    print()
    print("Endpoints:")
    print("  http://localhost:9090/metrics - Prometheus metrics")
    print("  http://localhost:9090/health - Health check")
    print("  http://localhost:9090/health/live - Liveness probe")
    print("  http://localhost:9090/health/ready - Readiness probe")
    print("=" * 60)
