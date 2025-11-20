"""
Performance Baselines and Benchmarks for HoloLoom VoiceAgent + Elle AR

This module defines performance thresholds and validates test results against
established baselines. SLAs are based on production requirements and user
experience targets.

Performance Baselines:
- Voice queries: p95 < 300ms (user expects quick response)
- TTS cached: p95 < 30ms (instant feedback for cached responses)
- TTS uncached: p95 < 800ms (acceptable for speech synthesis)
- Health checks: p95 < 10ms (lightweight monitoring)
- Photo annotation: p95 < 500ms (backgrounded operation)
- Context updates: p95 < 100ms (AR tracking requirement)
- Batch operations: p95 < 1500ms (per query aggregated)

Author: Wave 3 Production Hardening
Date: 2025-11-16
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json


@dataclass
class PerformanceSLA:
    """Performance Service Level Agreement."""

    endpoint: str
    p50_ms: float  # Median expected latency
    p95_ms: float  # 95th percentile (SLA threshold)
    p99_ms: float  # 99th percentile (absolute max acceptable)
    max_acceptable_ms: float  # Hard limit
    error_rate_percent: float  # Max acceptable error rate
    cache_hit_rate_percent: Optional[float] = None  # Expected cache hit rate


# ============================================================================
# Performance Baselines (2025-11-16 Production Targets)
# ============================================================================

PERFORMANCE_BASELINES: Dict[str, PerformanceSLA] = {
    # Primary workload: voice command processing
    "/query [voice_command]": PerformanceSLA(
        endpoint="/query [voice_command]",
        p50_ms=150,
        p95_ms=300,
        p99_ms=500,
        max_acceptable_ms=2000,
        error_rate_percent=0.5
    ),

    # TTS with caching (high hit rate expected)
    "/tts/synthesize [cached]": PerformanceSLA(
        endpoint="/tts/synthesize [cached]",
        p50_ms=15,
        p95_ms=30,
        p99_ms=50,
        max_acceptable_ms=100,
        error_rate_percent=0.1,
        cache_hit_rate_percent=95
    ),

    # TTS without caching (speech synthesis is slow)
    "/tts/synthesize [uncached]": PerformanceSLA(
        endpoint="/tts/synthesize [uncached]",
        p50_ms=500,
        p95_ms=800,
        p99_ms=1000,
        max_acceptable_ms=1500,
        error_rate_percent=1.0
    ),

    # Health endpoint (monitoring priority: fast + reliable)
    "/health": PerformanceSLA(
        endpoint="/health",
        p50_ms=5,
        p95_ms=10,
        p99_ms=20,
        max_acceptable_ms=50,
        error_rate_percent=0.0
    ),

    # Statistics endpoint (dashboard use)
    "/stats": PerformanceSLA(
        endpoint="/stats",
        p50_ms=50,
        p95_ms=100,
        p99_ms=200,
        max_acceptable_ms=500,
        error_rate_percent=0.5
    ),

    # Photo annotation (Elle AR integration)
    "/annotate-photo": PerformanceSLA(
        endpoint="/annotate-photo",
        p50_ms=200,
        p95_ms=500,
        p99_ms=800,
        max_acceptable_ms=1500,
        error_rate_percent=1.0
    ),

    # Context updates (AR tracking real-time)
    "/context/update": PerformanceSLA(
        endpoint="/context/update",
        p50_ms=20,
        p95_ms=100,
        p99_ms=150,
        max_acceptable_ms=300,
        error_rate_percent=0.2
    ),

    # Batch operations (aggregated)
    "/batch/query": PerformanceSLA(
        endpoint="/batch/query",
        p50_ms=500,
        p95_ms=1000,
        p99_ms=1500,
        max_acceptable_ms=3000,
        error_rate_percent=1.0
    ),
}


# ============================================================================
# Load Scenario Configurations
# ============================================================================

@dataclass
class LoadScenario:
    """Configuration for a load test scenario."""

    name: str
    description: str
    target_users: int
    spawn_rate: float  # users per second
    run_time_seconds: int
    ramp_up_seconds: Optional[int] = None
    ramp_down_seconds: Optional[int] = None

    def spawn_rate_str(self) -> str:
        """Get spawn rate as string."""
        if self.spawn_rate == int(self.spawn_rate):
            return f"{int(self.spawn_rate)} users/sec"
        return f"{self.spawn_rate:.1f} users/sec"

    def duration_str(self) -> str:
        """Get duration as readable string."""
        if self.run_time_seconds < 60:
            return f"{self.run_time_seconds}s"
        elif self.run_time_seconds < 3600:
            minutes = self.run_time_seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = self.run_time_seconds / 3600
            return f"{hours:.1f}h"

    def total_time_str(self) -> str:
        """Get total time including ramp-up/down."""
        total = self.run_time_seconds
        if self.ramp_up_seconds:
            total += self.ramp_up_seconds
        if self.ramp_down_seconds:
            total += self.ramp_down_seconds
        return self._seconds_to_str(total)

    @staticmethod
    def _seconds_to_str(seconds: int) -> str:
        """Convert seconds to readable string."""
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}m"
        else:
            return f"{seconds / 3600:.1f}h"


# Load test scenarios
LOAD_SCENARIOS = {
    "baseline": LoadScenario(
        name="Baseline Load Test",
        description="Normal load: 10 concurrent users for 5 minutes",
        target_users=10,
        spawn_rate=2.0,  # 2 users per second
        run_time_seconds=300,
        ramp_up_seconds=30,
        ramp_down_seconds=30
    ),

    "stress": LoadScenario(
        name="Stress Test",
        description="Ramping load: 10→100 users over 10 minutes",
        target_users=100,
        spawn_rate=10.0,  # 10 users per second
        run_time_seconds=600,
        ramp_up_seconds=60,
        ramp_down_seconds=60
    ),

    "spike": LoadScenario(
        name="Spike Test",
        description="Sudden spike: 0→200 users in 2 minutes, maintain 2 minutes",
        target_users=200,
        spawn_rate=50.0,  # 50 users per second
        run_time_seconds=120,
        ramp_up_seconds=20,
        ramp_down_seconds=30
    ),

    "endurance": LoadScenario(
        name="Endurance Test",
        description="Long-running: 50 sustained users for 1 hour",
        target_users=50,
        spawn_rate=5.0,  # 5 users per second
        run_time_seconds=3600,
        ramp_up_seconds=60,
        ramp_down_seconds=60
    ),

    "extreme": LoadScenario(
        name="Extreme Load Test",
        description="Maximum capacity: 500 users for 5 minutes",
        target_users=500,
        spawn_rate=100.0,  # 100 users per second
        run_time_seconds=300,
        ramp_up_seconds=30,
        ramp_down_seconds=60
    ),
}


# ============================================================================
# Validation & Reporting
# ============================================================================

@dataclass
class ValidationResult:
    """Result of baseline validation."""

    endpoint: str
    passed: bool
    violations: List[str]
    actual: Dict
    baseline: PerformanceSLA

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "endpoint": self.endpoint,
            "passed": self.passed,
            "violations": self.violations,
            "actual": self.actual,
            "baseline": {
                "p50_ms": self.baseline.p50_ms,
                "p95_ms": self.baseline.p95_ms,
                "p99_ms": self.baseline.p99_ms,
                "error_rate_percent": self.baseline.error_rate_percent,
            }
        }


def validate_endpoint(endpoint: str, metrics: Dict) -> ValidationResult:
    """
    Validate endpoint metrics against baseline.

    Args:
        endpoint: Endpoint name (e.g., "/query [voice_command]")
        metrics: Metrics dict with keys: p50, p95, p99, max, errors, count

    Returns:
        ValidationResult with pass/fail and violations
    """

    if endpoint not in PERFORMANCE_BASELINES:
        return ValidationResult(
            endpoint=endpoint,
            passed=True,  # Unknown endpoints pass by default
            violations=[],
            actual=metrics,
            baseline=None
        )

    baseline = PERFORMANCE_BASELINES[endpoint]
    violations = []

    # Calculate error rate
    error_rate = (metrics['errors'] / metrics['count'] * 100) if metrics['count'] > 0 else 0

    # Check p95 SLA (primary threshold)
    if metrics['p95'] > baseline.p95_ms:
        violations.append(
            f"p95 {metrics['p95']:.0f}ms exceeds baseline {baseline.p95_ms}ms "
            f"({(metrics['p95'] / baseline.p95_ms - 1) * 100:.1f}% over)"
        )

    # Check error rate
    if error_rate > baseline.error_rate_percent:
        violations.append(
            f"Error rate {error_rate:.2f}% exceeds baseline {baseline.error_rate_percent}%"
        )

    # Check cache hit rate (if applicable)
    if baseline.cache_hit_rate_percent and 'cache_hit_rate' in metrics:
        if metrics['cache_hit_rate'] < baseline.cache_hit_rate_percent:
            violations.append(
                f"Cache hit rate {metrics['cache_hit_rate']:.1f}% below "
                f"baseline {baseline.cache_hit_rate_percent}%"
            )

    return ValidationResult(
        endpoint=endpoint,
        passed=len(violations) == 0,
        violations=violations,
        actual=metrics,
        baseline=baseline
    )


def validate_results(summary: Dict) -> Tuple[List[ValidationResult], bool]:
    """
    Validate all endpoints against baselines.

    Args:
        summary: Metrics summary from MetricsCollector

    Returns:
        Tuple of (validation results, all_passed bool)
    """

    results = []
    for endpoint, metrics in summary.get('endpoints', {}).items():
        result = validate_endpoint(endpoint, metrics)
        results.append(result)

    all_passed = all(r.passed for r in results)
    return results, all_passed


def format_validation_report(results: List[ValidationResult]) -> str:
    """Format validation results as report."""

    report = "\n" + "=" * 100 + "\n"
    report += "LOAD TEST VALIDATION REPORT\n"
    report += "=" * 100 + "\n\n"

    # Summary
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed

    report += f"Summary: {passed}/{total} endpoints passed"
    if failed > 0:
        report += f" ({failed} FAILED)"
    report += "\n\n"

    # Detailed results
    report += "Endpoint Details:\n"
    report += "-" * 100 + "\n"
    report += f"{'Endpoint':<30} {'p50':<8} {'p95':<8} {'Errors%':<10} {'Status':<8}\n"
    report += "-" * 100 + "\n"

    for result in results:
        status = "✅ PASS" if result.passed else "❌ FAIL"
        error_rate = (result.actual['errors'] / result.actual['count'] * 100) \
                    if result.actual['count'] > 0 else 0

        report += f"{result.endpoint:<30} "
        report += f"{result.actual['p50']:<8.1f} "
        report += f"{result.actual['p95']:<8.1f} "
        report += f"{error_rate:<10.2f} "
        report += f"{status:<8}\n"

    # Violations
    report += "\n" + "-" * 100 + "\n"
    report += "Violations:\n"
    report += "-" * 100 + "\n"

    has_violations = False
    for result in results:
        if not result.passed:
            has_violations = True
            report += f"\n{result.endpoint}:\n"
            for violation in result.violations:
                report += f"  • {violation}\n"

    if not has_violations:
        report += "  None - all endpoints within SLA\n"

    report += "\n" + "=" * 100 + "\n"

    return report


def get_scenario(name: str) -> Optional[LoadScenario]:
    """Get load scenario by name."""
    return LOAD_SCENARIOS.get(name)


def list_scenarios() -> List[str]:
    """List all available scenario names."""
    return list(LOAD_SCENARIOS.keys())


def get_baseline(endpoint: str) -> Optional[PerformanceSLA]:
    """Get performance baseline for endpoint."""
    return PERFORMANCE_BASELINES.get(endpoint)


# ============================================================================
# Capacity Planning
# ============================================================================

class CapacityPlanner:
    """Calculate capacity based on load test results."""

    @staticmethod
    def estimate_max_users(rps: float, avg_request_time_ms: float) -> int:
        """
        Estimate maximum concurrent users from RPS and request time.

        Formula: Users = (RPS * Response Time) / 1000

        Args:
            rps: Requests per second
            avg_request_time_ms: Average response time in milliseconds

        Returns:
            Estimated maximum concurrent users
        """
        return int((rps * avg_request_time_ms) / 1000)

    @staticmethod
    def estimate_required_instances(
        target_rps: float,
        rps_per_instance: float,
        headroom: float = 0.3
    ) -> int:
        """
        Calculate required instances with headroom.

        Args:
            target_rps: Target requests per second
            rps_per_instance: RPS capacity per instance
            headroom: Headroom factor (0.3 = 30% extra capacity)

        Returns:
            Number of instances needed
        """
        required = target_rps / rps_per_instance
        with_headroom = required * (1 + headroom)
        return max(1, int(with_headroom + 0.5))  # Round up

    @staticmethod
    def estimate_burst_capacity(
        sustained_rps: float,
        burst_multiplier: float = 2.0,
        burst_duration_seconds: float = 30.0
    ) -> Dict[str, float]:
        """
        Calculate burst capacity planning.

        Args:
            sustained_rps: Sustained RPS capacity
            burst_multiplier: Peak/sustained ratio (e.g., 2x)
            burst_duration_seconds: Duration of burst

        Returns:
            Dict with capacity planning metrics
        """
        peak_rps = sustained_rps * burst_multiplier
        total_requests_in_burst = peak_rps * burst_duration_seconds

        return {
            "sustained_rps": sustained_rps,
            "peak_rps": peak_rps,
            "burst_duration_seconds": burst_duration_seconds,
            "total_requests_in_burst": total_requests_in_burst,
            "queue_size_needed": int(total_requests_in_burst * 0.1),  # 10% buffer
        }
