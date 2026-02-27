"""Tests for MetricsCollector extensions.

Tests for:
- Statistical analysis (std dev, correlation, anomaly detection, trends)
- Alerting system (thresholds, cooldown, handlers)
- Tapestry bridge (signal mapping)
- Enhanced MetricsCollector integration
"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from hololoom.prompting.testing.metrics_collector import (
    MetricType,
    Metric,
    MetricsCollector,
    MetricsAggregator,
    create_metrics_collector,
)
from hololoom.prompting.testing.statistical_analysis import (
    StatisticalSummary,
    StatisticalAnalyzer,
    create_statistical_analyzer,
)
from hololoom.prompting.testing.alerting import (
    AlertSeverity,
    ThresholdRule,
    Alert,
    AlertingSystem,
    get_default_rules,
    create_alerting_system,
)
from hololoom.prompting.testing.tapestry_bridge import (
    TapestryMetricsBridge,
    create_tapestry_bridge,
)


class TestStatisticalAnalyzer(unittest.TestCase):
    """Tests for StatisticalAnalyzer."""

    def setUp(self):
        self.analyzer = create_statistical_analyzer()

    def test_empty_values(self):
        """Test analysis with empty values."""
        result = self.analyzer.analyze([])
        self.assertEqual(result.mean, 0.0)
        self.assertEqual(result.std_dev, 0.0)
        self.assertEqual(result.trend, "stable")
        self.assertEqual(result.anomalies, [])

    def test_basic_statistics(self):
        """Test mean, std_dev, variance calculation."""
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = self.analyzer.analyze(values)

        self.assertAlmostEqual(result.mean, 30.0, places=2)
        self.assertGreater(result.std_dev, 0)
        self.assertAlmostEqual(result.variance, result.std_dev ** 2, places=6)

    def test_median_calculation(self):
        """Test median for odd and even counts."""
        odd_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = self.analyzer.analyze(odd_values)
        self.assertEqual(result.median, 3.0)

        even_values = [1.0, 2.0, 3.0, 4.0]
        result = self.analyzer.analyze(even_values)
        self.assertEqual(result.median, 2.5)

    def test_percentiles(self):
        """Test percentile calculation."""
        values = list(range(1, 101))  # 1-100
        result = self.analyzer.analyze(values)

        self.assertIn(25, result.percentiles)
        self.assertIn(50, result.percentiles)
        self.assertIn(75, result.percentiles)
        self.assertIn(95, result.percentiles)

    def test_trend_increasing(self):
        """Test increasing trend detection."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        result = self.analyzer.analyze(values)
        self.assertEqual(result.trend, "increasing")

    def test_trend_decreasing(self):
        """Test decreasing trend detection."""
        values = [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        result = self.analyzer.analyze(values)
        self.assertEqual(result.trend, "decreasing")

    def test_trend_stable(self):
        """Test stable trend detection."""
        values = [5.0, 5.1, 4.9, 5.0, 5.1, 4.9, 5.0, 5.1, 4.9]
        result = self.analyzer.analyze(values)
        self.assertEqual(result.trend, "stable")

    def test_anomaly_detection(self):
        """Test Z-score based anomaly detection."""
        # Normal values with one clear outlier
        values = [10.0, 11.0, 10.5, 10.2, 10.8, 100.0, 10.3]  # 100.0 is anomaly
        result = self.analyzer.analyze(values)

        self.assertIn(5, result.anomalies)  # Index 5 is the outlier

    def test_correlation_positive(self):
        """Test positive correlation."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        correlation = self.analyzer.correlation(x, y)

        self.assertAlmostEqual(correlation, 1.0, places=2)

    def test_correlation_negative(self):
        """Test negative correlation."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 8.0, 6.0, 4.0, 2.0]
        correlation = self.analyzer.correlation(x, y)

        self.assertAlmostEqual(correlation, -1.0, places=2)

    def test_correlation_weak(self):
        """Test weak correlation (not perfectly correlated)."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 3.0, 5.0, 4.0]  # Weak positive relationship
        correlation = self.analyzer.correlation(x, y)

        # Should be weaker than perfect correlation
        self.assertLess(abs(correlation), 1.0)
        self.assertGreater(abs(correlation), 0.0)

    def test_correlation_mismatched_lengths(self):
        """Test correlation with mismatched lengths."""
        x = [1.0, 2.0, 3.0]
        y = [1.0, 2.0]
        correlation = self.analyzer.correlation(x, y)

        self.assertEqual(correlation, 0.0)

    def test_rolling_stats(self):
        """Test rolling statistics calculation."""
        values = list(range(1, 21))  # 1-20
        results = self.analyzer.rolling_stats(values, window_size=5)

        self.assertGreater(len(results), 0)
        for r in results:
            self.assertIn("index", r)
            self.assertIn("rolling_mean", r)
            self.assertIn("rolling_std", r)


class TestAlertingSystem(unittest.TestCase):
    """Tests for AlertingSystem."""

    def setUp(self):
        self.alerting = create_alerting_system(with_defaults=False)

    def test_add_rule(self):
        """Test adding threshold rules."""
        rule = ThresholdRule(
            metric_type=MetricType.ERROR_RATE,
            threshold=0.1,
            comparison="gt",
            severity=AlertSeverity.CRITICAL,
        )
        self.alerting.add_rule(rule)

        self.assertEqual(len(self.alerting.rules), 1)

    def test_check_triggers_alert(self):
        """Test that threshold violation triggers alert."""
        rule = ThresholdRule(
            metric_type=MetricType.ERROR_RATE,
            threshold=0.05,
            comparison="gt",
            severity=AlertSeverity.CRITICAL,
            cooldown_seconds=0,  # No cooldown for test
        )
        self.alerting.add_rule(rule)

        # Should trigger (0.1 > 0.05)
        alert = self.alerting.check(MetricType.ERROR_RATE, 0.1, {})
        self.assertIsNotNone(alert)
        self.assertEqual(alert.actual_value, 0.1)

    def test_check_no_trigger(self):
        """Test that values within threshold don't trigger."""
        rule = ThresholdRule(
            metric_type=MetricType.ERROR_RATE,
            threshold=0.1,
            comparison="gt",
            severity=AlertSeverity.WARNING,
        )
        self.alerting.add_rule(rule)

        # Should not trigger (0.05 is not > 0.1)
        alert = self.alerting.check(MetricType.ERROR_RATE, 0.05, {})
        self.assertIsNone(alert)

    def test_comparison_operators(self):
        """Test all comparison operators."""
        test_cases = [
            ("gt", 10, 5, True),   # 10 > 5
            ("gt", 5, 10, False),  # 5 > 10
            ("lt", 5, 10, True),   # 5 < 10
            ("lt", 10, 5, False),  # 10 < 5
            ("gte", 10, 10, True), # 10 >= 10
            ("lte", 5, 5, True),   # 5 <= 5
            ("eq", 5, 5, True),    # 5 == 5
            ("eq", 5, 10, False),  # 5 != 10
        ]

        for comparison, value, threshold, should_trigger in test_cases:
            alerting = create_alerting_system(with_defaults=False)
            rule = ThresholdRule(
                metric_type=MetricType.LATENCY_MS,
                threshold=threshold,
                comparison=comparison,
                severity=AlertSeverity.WARNING,
                cooldown_seconds=0,
            )
            alerting.add_rule(rule)

            alert = alerting.check(MetricType.LATENCY_MS, value, {})
            if should_trigger:
                self.assertIsNotNone(alert, f"Expected alert for {value} {comparison} {threshold}")
            else:
                self.assertIsNone(alert, f"Unexpected alert for {value} {comparison} {threshold}")

    def test_cooldown(self):
        """Test cooldown prevents repeated alerts."""
        rule = ThresholdRule(
            metric_type=MetricType.ERROR_RATE,
            threshold=0.05,
            comparison="gt",
            severity=AlertSeverity.CRITICAL,
            cooldown_seconds=60,  # 60 second cooldown
        )
        self.alerting.add_rule(rule)

        # First alert should trigger
        alert1 = self.alerting.check(MetricType.ERROR_RATE, 0.1, {})
        self.assertIsNotNone(alert1)

        # Second alert should be blocked (within cooldown)
        alert2 = self.alerting.check(MetricType.ERROR_RATE, 0.1, {})
        self.assertIsNone(alert2)

    def test_alert_handler(self):
        """Test notification handlers are called."""
        handler_called = []

        def mock_handler(alert):
            handler_called.append(alert)

        self.alerting.add_handler(mock_handler)

        rule = ThresholdRule(
            metric_type=MetricType.ERROR_RATE,
            threshold=0.05,
            comparison="gt",
            severity=AlertSeverity.CRITICAL,
            cooldown_seconds=0,
        )
        self.alerting.add_rule(rule)

        self.alerting.check(MetricType.ERROR_RATE, 0.1, {})

        self.assertEqual(len(handler_called), 1)

    def test_get_active_alerts(self):
        """Test getting unacknowledged alerts."""
        rule = ThresholdRule(
            metric_type=MetricType.ERROR_RATE,
            threshold=0.05,
            comparison="gt",
            severity=AlertSeverity.CRITICAL,
            cooldown_seconds=0,
        )
        self.alerting.add_rule(rule)

        # Trigger alert
        self.alerting.check(MetricType.ERROR_RATE, 0.1, {})

        active = self.alerting.get_active_alerts()
        self.assertEqual(len(active), 1)

    def test_acknowledge_alert(self):
        """Test acknowledging alerts."""
        rule = ThresholdRule(
            metric_type=MetricType.ERROR_RATE,
            threshold=0.05,
            comparison="gt",
            severity=AlertSeverity.CRITICAL,
            cooldown_seconds=0,
        )
        self.alerting.add_rule(rule)

        alert = self.alerting.check(MetricType.ERROR_RATE, 0.1, {})
        self.alerting.acknowledge(alert)

        active = self.alerting.get_active_alerts()
        self.assertEqual(len(active), 0)

    def test_default_rules(self):
        """Test default rules are created."""
        alerting = create_alerting_system(with_defaults=True)

        self.assertGreater(len(alerting.rules), 0)

    def test_alert_message(self):
        """Test alert message generation."""
        rule = ThresholdRule(
            metric_type=MetricType.ERROR_RATE,
            threshold=0.05,
            comparison="gt",
            severity=AlertSeverity.CRITICAL,
            message_template="Error rate {actual:.1%} exceeds {threshold:.1%}",
        )
        alert = Alert(rule=rule, actual_value=0.1)

        message = alert.message()
        self.assertIn("10.0%", message)
        self.assertIn("5.0%", message)


class TestTapestryMetricsBridge(unittest.TestCase):
    """Tests for TapestryMetricsBridge."""

    def setUp(self):
        self.collector = create_metrics_collector()
        self.bridge = create_tapestry_bridge(self.collector)

    def test_record_fabric_check(self):
        """Test recording a FabricCheckResult."""
        check_result = {
            "confidence": 0.85,
            "passed": True,
            "blockers": [],
            "signals": {
                "tests": {"score": 0.95, "details": {}},
                "quality": {"score": 0.88, "details": {}},
            },
        }

        self.bridge.record_fabric_check(check_result, thread_id="test-123")

        # Check that metrics were recorded
        metrics = self.collector.get_metrics()
        self.assertGreater(len(metrics), 0)

        # Check FABRIC_CONFIDENCE was recorded
        confidence_metrics = self.collector.get_metrics(MetricType.FABRIC_CONFIDENCE)
        self.assertEqual(len(confidence_metrics), 1)
        self.assertEqual(confidence_metrics[0].value, 0.85)

    def test_record_blockers(self):
        """Test recording blocker count."""
        check_result = {
            "confidence": 0.5,
            "passed": False,
            "blockers": ["blocker1", "blocker2", "blocker3"],
            "signals": {},
        }

        self.bridge.record_fabric_check(check_result, thread_id="test-456")

        blocker_metrics = self.collector.get_metrics(MetricType.THREAD_BLOCKERS)
        self.assertEqual(len(blocker_metrics), 1)
        self.assertEqual(blocker_metrics[0].value, 3)

    def test_signal_mapping(self):
        """Test individual signal to metric mapping."""
        # Test alignment signal (inverted score for safety risk)
        alignment_result = {"score": 0.9, "details": {}}
        self.bridge.record_signal_result("alignment", alignment_result, thread_id="test")

        safety_metrics = self.collector.get_metrics(MetricType.SAFETY_RISK_LEVEL)
        self.assertEqual(len(safety_metrics), 1)
        # Score is inverted: 1.0 - 0.9 = 0.1 (low risk)
        self.assertAlmostEqual(safety_metrics[0].value, 0.1, places=2)

    def test_security_violations_extraction(self):
        """Test extracting security violations from quality signal."""
        quality_result = {
            "score": 0.7,
            "details": {
                "security_issues": ["SQL injection", "XSS"]
            }
        }
        self.bridge.record_signal_result("quality", quality_result, thread_id="test")

        security_metrics = self.collector.get_metrics(MetricType.SECURITY_VIOLATIONS)
        self.assertEqual(len(security_metrics), 1)
        self.assertEqual(security_metrics[0].value, 2)

    def test_thread_id_in_tags(self):
        """Test that thread_id is included in tags."""
        check_result = {
            "confidence": 0.8,
            "passed": True,
            "blockers": [],
            "signals": {},
        }

        self.bridge.record_fabric_check(check_result, thread_id="my-thread-id")

        metrics = self.collector.get_metrics()
        for m in metrics:
            self.assertEqual(m.tags.get("thread_id"), "my-thread-id")

    def test_extra_tags(self):
        """Test extra tags are included."""
        check_result = {
            "confidence": 0.8,
            "passed": True,
            "blockers": [],
            "signals": {},
        }

        self.bridge.record_fabric_check(
            check_result,
            thread_id="test",
            extra_tags={"environment": "production"}
        )

        metrics = self.collector.get_metrics()
        for m in metrics:
            self.assertEqual(m.tags.get("environment"), "production")


class TestEnhancedMetricsCollector(unittest.TestCase):
    """Tests for enhanced MetricsCollector with statistics and alerting."""

    def test_statistics_disabled_by_default(self):
        """Test that statistics is disabled by default."""
        collector = create_metrics_collector()

        collector.record("test", 1.0, MetricType.LATENCY_MS)
        stats = collector.get_statistics(MetricType.LATENCY_MS)

        self.assertIsNone(stats)

    def test_statistics_enabled(self):
        """Test statistics when enabled."""
        collector = create_metrics_collector(enable_statistics=True)

        for i in range(10):
            collector.record("test", float(i), MetricType.LATENCY_MS)

        stats = collector.get_statistics(MetricType.LATENCY_MS)

        self.assertIsNotNone(stats)
        self.assertGreater(stats.std_dev, 0)

    def test_alerting_disabled_by_default(self):
        """Test that alerting is disabled by default."""
        collector = create_metrics_collector()

        alert = collector.record("test", 0.99, MetricType.ERROR_RATE)

        self.assertIsNone(alert)

    def test_alerting_enabled(self):
        """Test alerting when enabled."""
        collector = create_metrics_collector(enable_alerting=True)

        # ERROR_RATE > 0.05 triggers default alert
        alert = collector.record("test", 0.1, MetricType.ERROR_RATE)

        self.assertIsNotNone(alert)
        self.assertEqual(alert.rule.severity, AlertSeverity.CRITICAL)

    def test_custom_alert_rule(self):
        """Test adding custom alert rules."""
        collector = create_metrics_collector(enable_alerting=True)

        collector.add_alert_rule(
            metric_type=MetricType.LATENCY_MS,
            threshold=100,
            comparison="gt",
            severity="warning",
        )

        alert = collector.record("test", 150, MetricType.LATENCY_MS)

        self.assertIsNotNone(alert)

    def test_get_correlation(self):
        """Test correlation calculation via collector."""
        collector = create_metrics_collector(enable_statistics=True)

        # Record correlated metrics
        for i in range(10):
            collector.record("latency", float(i * 10), MetricType.LATENCY_MS)
            collector.record("tokens", float(i * 100), MetricType.TOKEN_COUNT)

        correlation = collector.get_correlation(
            MetricType.LATENCY_MS,
            MetricType.TOKEN_COUNT
        )

        # Should be strongly correlated
        self.assertGreater(correlation, 0.9)

    def test_get_rolling_stats(self):
        """Test rolling statistics via collector."""
        collector = create_metrics_collector(enable_statistics=True)

        for i in range(20):
            collector.record("latency", float(i), MetricType.LATENCY_MS)

        rolling = collector.get_rolling_stats(MetricType.LATENCY_MS, window_size=5)

        self.assertGreater(len(rolling), 0)

    def test_get_enhanced_summary(self):
        """Test enhanced summary with statistics and alerts."""
        collector = create_metrics_collector(
            enable_statistics=True,
            enable_alerting=True
        )

        # Record some metrics
        for i in range(10):
            collector.record("latency", float(i * 10), MetricType.LATENCY_MS)

        # Trigger an alert
        collector.record("error", 0.1, MetricType.ERROR_RATE)

        summary = collector.get_enhanced_summary()

        # Check statistics are included
        if MetricType.LATENCY_MS.value in summary:
            self.assertIn("std_dev", summary[MetricType.LATENCY_MS.value])
            self.assertIn("trend", summary[MetricType.LATENCY_MS.value])

        # Check alerts are included
        self.assertIn("active_alerts", summary)
        self.assertGreater(summary["active_alerts"], 0)

    def test_acknowledge_all_alerts(self):
        """Test acknowledging all alerts."""
        collector = create_metrics_collector(enable_alerting=True)

        # Trigger multiple alerts
        collector.record("error1", 0.1, MetricType.ERROR_RATE)

        count = collector.acknowledge_all_alerts()

        self.assertGreater(count, 0)
        self.assertEqual(len(collector.get_active_alerts()), 0)

    def test_add_alert_handler(self):
        """Test adding alert handlers."""
        collector = create_metrics_collector(enable_alerting=True)

        handler_called = []

        def handler(alert):
            handler_called.append(alert)

        collector.add_alert_handler(handler)
        collector.record("error", 0.1, MetricType.ERROR_RATE)

        self.assertEqual(len(handler_called), 1)


class TestNewMetricTypes(unittest.TestCase):
    """Tests for new Tapestry-related MetricTypes."""

    def test_all_metric_types_exist(self):
        """Test all 16 metric types exist."""
        expected_types = [
            # Original 9
            "latency_ms",
            "token_count",
            "quality_score",
            "pass_rate",
            "mutation_robustness",
            "regression_count",
            "cache_hit_rate",
            "error_rate",
            "cost_estimate",
            # New 7
            "security_violations",
            "safety_risk_level",
            "doc_coverage",
            "arch_adherence",
            "integration_health",
            "fabric_confidence",
            "thread_blockers",
        ]

        for type_value in expected_types:
            metric_type = MetricType(type_value)
            self.assertEqual(metric_type.value, type_value)

    def test_record_new_metric_types(self):
        """Test recording all new metric types."""
        collector = create_metrics_collector()

        new_types = [
            (MetricType.SECURITY_VIOLATIONS, 2),
            (MetricType.SAFETY_RISK_LEVEL, 0.3),
            (MetricType.DOCUMENTATION_COVERAGE, 0.85),
            (MetricType.ARCHITECTURE_ADHERENCE, 0.92),
            (MetricType.INTEGRATION_HEALTH, 0.88),
            (MetricType.FABRIC_CONFIDENCE, 0.95),
            (MetricType.THREAD_BLOCKERS, 0),
        ]

        for metric_type, value in new_types:
            collector.record("test", value, metric_type)

        # Verify all were recorded
        for metric_type, expected_value in new_types:
            metrics = collector.get_metrics(metric_type)
            self.assertEqual(len(metrics), 1)
            self.assertEqual(metrics[0].value, expected_value)


if __name__ == "__main__":
    unittest.main()
