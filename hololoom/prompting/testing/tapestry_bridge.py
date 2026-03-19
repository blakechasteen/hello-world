"""Bridge FabricInspector signals to MetricsCollector.

Maps Tapestry verification signals to MetricType values:
- TestSignal → PASS_RATE
- TroughSignal → QUALITY_SCORE, SECURITY_VIOLATIONS
- AlignmentSignal → SAFETY_RISK_LEVEL
- ArchitectureSignal → ARCHITECTURE_ADHERENCE
- DocumentationSignal → DOCUMENTATION_COVERAGE
- IntegrationSignal → INTEGRATION_HEALTH
- Aggregated → FABRIC_CONFIDENCE, THREAD_BLOCKERS
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .metrics_collector import MetricsCollector

from .metrics_collector import MetricType


class TapestryMetricsBridge:
    """Converts FabricInspector results to metrics.

    Enables unified metrics collection across prompt testing
    and Tapestry verification signals.
    """

    def __init__(self, collector: "MetricsCollector"):
        """Initialize bridge.

        Args:
            collector: MetricsCollector to record metrics to
        """
        self.collector = collector

    def record_fabric_check(
        self,
        check_result: dict[str, Any],
        thread_id: str,
        extra_tags: dict[str, str] | None = None
    ) -> None:
        """Record all metrics from a FabricCheckResult.

        Args:
            check_result: FabricCheckResult dict or dataclass with:
                - confidence: float
                - passed: bool
                - signals: Dict[str, SignalResult]
                - blockers: List[str]
            thread_id: Identifier for the thread being checked
            extra_tags: Additional tags to include
        """
        tags = {"thread_id": thread_id}
        if extra_tags:
            tags.update(extra_tags)

        # Extract confidence (handle both dict and dataclass)
        confidence = self._get_attr(check_result, "confidence", 0.0)
        passed = self._get_attr(check_result, "passed", False)
        blockers = self._get_attr(check_result, "blockers", [])
        signals = self._get_attr(check_result, "signals", {})

        # Overall fabric confidence
        self.collector.record(
            name="fabric_check",
            value=confidence,
            metric_type=MetricType.FABRIC_CONFIDENCE,
            tags=tags
        )

        # Blocker count
        self.collector.record(
            name="blockers",
            value=len(blockers) if isinstance(blockers, list) else 0,
            metric_type=MetricType.THREAD_BLOCKERS,
            tags=tags
        )

        # Individual signals
        if isinstance(signals, dict):
            for signal_name, result in signals.items():
                self._record_signal(signal_name, result, tags)

    def record_signal_result(
        self,
        signal_name: str,
        result: dict[str, Any],
        thread_id: str,
        extra_tags: dict[str, str] | None = None
    ) -> None:
        """Record metrics from a single signal result.

        Args:
            signal_name: Name of the signal (e.g., "tests", "quality")
            result: SignalResult dict or dataclass
            thread_id: Identifier for the thread
            extra_tags: Additional tags to include
        """
        tags = {"thread_id": thread_id}
        if extra_tags:
            tags.update(extra_tags)
        self._record_signal(signal_name, result, tags)

    def _record_signal(
        self,
        signal_name: str,
        result: Any,
        base_tags: dict[str, str]
    ) -> None:
        """Map signal to appropriate metric type.

        Args:
            signal_name: Name of the signal
            result: SignalResult dict or dataclass
            base_tags: Tags to include with metrics
        """
        tags = {**base_tags, "signal": signal_name}

        # Extract score from result
        score = self._get_attr(result, "score", 0.0)
        details = self._get_attr(result, "details", {})

        # Map signal names to metric types
        signal_mapping = {
            "tests": (MetricType.PASS_RATE, score),
            "quality": (MetricType.QUALITY_SCORE, score),
            "alignment": (MetricType.SAFETY_RISK_LEVEL, 1.0 - score),  # Invert: high score = low risk
            "architecture": (MetricType.ARCHITECTURE_ADHERENCE, score),
            "documentation": (MetricType.DOCUMENTATION_COVERAGE, score),
            "integration": (MetricType.INTEGRATION_HEALTH, score),
        }

        if signal_name in signal_mapping:
            metric_type, value = signal_mapping[signal_name]
            self.collector.record(
                name=f"signal_{signal_name}",
                value=value,
                metric_type=metric_type,
                tags=tags
            )

        # Extract security violations from quality signal
        if signal_name == "quality" and isinstance(details, dict):
            security_issues = details.get("security_issues", 0)
            if isinstance(security_issues, list):
                security_count = len(security_issues)
            elif isinstance(security_issues, (int, float)):
                security_count = int(security_issues)
            else:
                security_count = 0

            if security_count > 0:
                self.collector.record(
                    name="security_violations",
                    value=security_count,
                    metric_type=MetricType.SECURITY_VIOLATIONS,
                    tags=tags
                )

    def _get_attr(self, obj: Any, name: str, default: Any) -> Any:
        """Get attribute from dict or dataclass.

        Args:
            obj: Dict or dataclass object
            name: Attribute/key name
            default: Default value if not found

        Returns:
            Attribute value or default
        """
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)


def create_tapestry_bridge(
    collector: "MetricsCollector"
) -> TapestryMetricsBridge:
    """Factory function for TapestryMetricsBridge.

    Args:
        collector: MetricsCollector instance

    Returns:
        New TapestryMetricsBridge instance
    """
    return TapestryMetricsBridge(collector)
