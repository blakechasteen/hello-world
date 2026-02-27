"""
SLA Definitions and Validation

Service Level Agreement (SLA) definitions and compliance validation.

Features:
- Pre-defined SLA tiers (Bronze, Silver, Gold, Platinum)
- Custom SLA definitions
- Real-time compliance monitoring
- Violation alerting

Author: HoloLoom B2B Framework
Date: November 2025
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class SLAMetric(Enum):
    """SLA metrics"""
    LATENCY_P95 = "latency_p95"
    LATENCY_P99 = "latency_p99"
    SUCCESS_RATE = "success_rate"
    AVAILABILITY = "availability"
    THROUGHPUT_QPS = "throughput_qps"
    ERROR_RATE = "error_rate"


class SLATier(Enum):
    """Pre-defined SLA tiers"""
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"
    PLATINUM = "platinum"


@dataclass
class SLADefinition:
    """
    SLA definition with metric thresholds.

    Example:
        >>> sla = SLADefinition(
        ...     tier=SLATier.GOLD,
        ...     latency_p95_ms=200.0,
        ...     latency_p99_ms=500.0,
        ...     success_rate=0.99,
        ...     availability=0.999,
        ...     min_throughput_qps=50.0
        ... )
    """
    tier: SLATier
    latency_p95_ms: float
    latency_p99_ms: float
    success_rate: float
    availability: float
    min_throughput_qps: float
    max_error_rate: float = 0.01

    @classmethod
    def bronze(cls) -> "SLADefinition":
        """Bronze tier SLA (basic)"""
        return cls(
            tier=SLATier.BRONZE,
            latency_p95_ms=1000.0,   # 1s p95
            latency_p99_ms=2000.0,   # 2s p99
            success_rate=0.90,       # 90% success
            availability=0.95,       # 95% uptime
            min_throughput_qps=10.0, # 10 QPS minimum
            max_error_rate=0.10      # 10% errors
        )

    @classmethod
    def silver(cls) -> "SLADefinition":
        """Silver tier SLA (standard)"""
        return cls(
            tier=SLATier.SILVER,
            latency_p95_ms=500.0,    # 500ms p95
            latency_p99_ms=1000.0,   # 1s p99
            success_rate=0.95,       # 95% success
            availability=0.99,       # 99% uptime
            min_throughput_qps=25.0, # 25 QPS minimum
            max_error_rate=0.05      # 5% errors
        )

    @classmethod
    def gold(cls) -> "SLADefinition":
        """Gold tier SLA (premium)"""
        return cls(
            tier=SLATier.GOLD,
            latency_p95_ms=200.0,    # 200ms p95
            latency_p99_ms=500.0,    # 500ms p99
            success_rate=0.99,       # 99% success
            availability=0.999,      # 99.9% uptime
            min_throughput_qps=50.0, # 50 QPS minimum
            max_error_rate=0.01      # 1% errors
        )

    @classmethod
    def platinum(cls) -> "SLADefinition":
        """Platinum tier SLA (enterprise)"""
        return cls(
            tier=SLATier.PLATINUM,
            latency_p95_ms=100.0,     # 100ms p95
            latency_p99_ms=200.0,     # 200ms p99
            success_rate=0.999,       # 99.9% success
            availability=0.9999,      # 99.99% uptime
            min_throughput_qps=100.0, # 100 QPS minimum
            max_error_rate=0.001      # 0.1% errors
        )


@dataclass
class SLAViolation:
    """SLA violation record"""
    metric: SLAMetric
    threshold: float
    actual: float
    severity: str  # "warning" | "critical"
    message: str


class SLAValidator:
    """
    Validate department performance against SLA.

    Example:
        >>> sla = SLADefinition.gold()
        >>> validator = SLAValidator(sla)
        >>>
        >>> # Validate benchmark results
        >>> violations = validator.validate(benchmark_result)
        >>> if violations:
        ...     for v in violations:
        ...         print(f"{v.severity}: {v.message}")
    """

    def __init__(self, sla: SLADefinition):
        """
        Initialize SLA validator.

        Args:
            sla: SLA definition to validate against
        """
        self.sla = sla

    def validate(self, metrics: Dict[str, float]) -> List[SLAViolation]:
        """
        Validate metrics against SLA.

        Args:
            metrics: Dict of metric_name → value

        Returns:
            List of SLA violations (empty if compliant)
        """
        violations = []

        # Latency p95
        if "latency_p95" in metrics:
            if metrics["latency_p95"] > self.sla.latency_p95_ms:
                violations.append(SLAViolation(
                    metric=SLAMetric.LATENCY_P95,
                    threshold=self.sla.latency_p95_ms,
                    actual=metrics["latency_p95"],
                    severity="critical" if metrics["latency_p95"] > self.sla.latency_p95_ms * 1.5 else "warning",
                    message=f"Latency p95 {metrics['latency_p95']:.1f}ms exceeds SLA {self.sla.latency_p95_ms:.1f}ms"
                ))

        # Latency p99
        if "latency_p99" in metrics:
            if metrics["latency_p99"] > self.sla.latency_p99_ms:
                violations.append(SLAViolation(
                    metric=SLAMetric.LATENCY_P99,
                    threshold=self.sla.latency_p99_ms,
                    actual=metrics["latency_p99"],
                    severity="warning",
                    message=f"Latency p99 {metrics['latency_p99']:.1f}ms exceeds SLA {self.sla.latency_p99_ms:.1f}ms"
                ))

        # Success rate
        if "success_rate" in metrics:
            if metrics["success_rate"] < self.sla.success_rate:
                violations.append(SLAViolation(
                    metric=SLAMetric.SUCCESS_RATE,
                    threshold=self.sla.success_rate,
                    actual=metrics["success_rate"],
                    severity="critical",
                    message=f"Success rate {metrics['success_rate']:.1%} below SLA {self.sla.success_rate:.1%}"
                ))

        # Throughput
        if "queries_per_second" in metrics:
            if metrics["queries_per_second"] < self.sla.min_throughput_qps:
                violations.append(SLAViolation(
                    metric=SLAMetric.THROUGHPUT_QPS,
                    threshold=self.sla.min_throughput_qps,
                    actual=metrics["queries_per_second"],
                    severity="warning",
                    message=f"Throughput {metrics['queries_per_second']:.1f} QPS below SLA {self.sla.min_throughput_qps:.1f} QPS"
                ))

        # Error rate
        if "error_rate" in metrics:
            if metrics["error_rate"] > self.sla.max_error_rate:
                violations.append(SLAViolation(
                    metric=SLAMetric.ERROR_RATE,
                    threshold=self.sla.max_error_rate,
                    actual=metrics["error_rate"],
                    severity="critical",
                    message=f"Error rate {metrics['error_rate']:.1%} exceeds SLA {self.sla.max_error_rate:.1%}"
                ))

        return violations

    def is_compliant(self, metrics: Dict[str, float]) -> bool:
        """
        Check if metrics are SLA compliant.

        Args:
            metrics: Dict of metric_name → value

        Returns:
            True if compliant, False otherwise
        """
        violations = self.validate(metrics)
        return len(violations) == 0

    def get_compliance_report(
        self,
        department_id: str,
        metrics: Dict[str, float]
    ) -> str:
        """
        Generate compliance report.

        Args:
            department_id: Department being validated
            metrics: Performance metrics

        Returns:
            Formatted compliance report
        """
        violations = self.validate(metrics)

        report = []
        report.append(f"=== SLA Compliance Report: {department_id} ===\n")
        report.append(f"SLA Tier: {self.sla.tier.value.upper()}\n\n")

        if not violations:
            report.append("[PASS] All metrics within SLA\n")
        else:
            report.append(f"[FAIL] {len(violations)} SLA violation(s)\n\n")

            for v in violations:
                report.append(
                    f"  [{v.severity.upper()}] {v.message}\n"
                    f"    Threshold: {v.threshold}\n"
                    f"    Actual: {v.actual}\n\n"
                )

        # Metrics summary
        report.append("=== Metrics Summary ===\n")
        for metric, value in metrics.items():
            report.append(f"  {metric}: {value}\n")

        return "".join(report)
