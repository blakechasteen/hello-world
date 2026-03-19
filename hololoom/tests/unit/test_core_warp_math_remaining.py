"""
Unit tests for warp math modules with 0% or low coverage:
- monitoring_dashboard.py (218 stmts, 0%)
- explainability.py (159 stmts, 0%)
- operation_selector.py (179 stmts — uncovered lines: plan_operations, topological_sort, get_statistics, etc.)

Target: 60%+ on each file.
"""

import time
import json
import pytest
import numpy as np
from pathlib import Path

from hololoom.core.warp.math.monitoring_dashboard import (
    RequestMetrics,
    AggregateMetrics,
    MetricsCollector,
    ABVariant,
    ABTester,
    AlertRule,
    LatencyAlert,
    SuccessRateAlert,
    CostAlert,
    AlertManager,
    MonitoringDashboard,
)
from hololoom.core.warp.math.explainability import (
    WhyExplanation,
    WhyNotExplanation,
    CounterfactualExplanation,
    FeatureImportanceAnalyzer,
    ExplanationGenerator,
    ExplainableSelector,
)
from hololoom.core.warp.math.operation_selector import (
    MathDomain,
    OperationLevel,
    QueryIntent,
    MathOperation,
    OperationPlan,
    MathOperationSelector,
    create_selector,
)


# ============================================================================
# Helpers
# ============================================================================

def _make_request_metrics(
    success=True,
    latency_ms=25.0,
    cost=15.0,
    confidence=0.8,
    ops=None,
    intent="similarity",
    ts=None,
) -> RequestMetrics:
    return RequestMetrics(
        timestamp=ts or time.time(),
        query="test query",
        intent=intent,
        operations_executed=ops or ["inner_product"],
        total_cost=cost,
        latency_ms=latency_ms,
        success=success,
        confidence=confidence,
        num_operations=len(ops) if ops else 1,
    )


# ============================================================================
# monitoring_dashboard.py — MetricsCollector
# ============================================================================


class TestMetricsCollector:
    def test_record_single_request(self):
        collector = MetricsCollector()
        m = _make_request_metrics()
        collector.record_request(m)

        assert collector.total_requests == 1
        assert collector.total_successes == 1
        assert collector.total_failures == 0
        assert len(collector.recent_requests) == 1

    def test_record_failure(self):
        collector = MetricsCollector()
        collector.record_request(_make_request_metrics(success=False))

        assert collector.total_failures == 1
        assert collector.total_successes == 0

    def test_operation_tracking(self):
        collector = MetricsCollector()
        collector.record_request(
            _make_request_metrics(ops=["inner_product", "metric_distance"])
        )

        assert collector.operation_counts["inner_product"] == 1
        assert collector.operation_counts["metric_distance"] == 1
        assert len(collector.operation_latencies["inner_product"]) == 1

    def test_intent_tracking(self):
        collector = MetricsCollector()
        collector.record_request(_make_request_metrics(intent="analysis"))
        collector.record_request(_make_request_metrics(intent="analysis"))
        collector.record_request(_make_request_metrics(intent="similarity"))

        assert collector.intent_counts["analysis"] == 2
        assert collector.intent_counts["similarity"] == 1

    def test_window_size_limit(self):
        collector = MetricsCollector(window_size=5)
        for _ in range(10):
            collector.record_request(_make_request_metrics())

        assert len(collector.recent_requests) == 5
        assert collector.total_requests == 10

    def test_get_current_metrics_empty(self):
        collector = MetricsCollector()
        result = collector.get_current_metrics()
        assert "error" in result

    def test_get_current_metrics_with_data(self):
        collector = MetricsCollector()
        for i in range(3):
            collector.record_request(_make_request_metrics(latency_ms=10.0 + i))

        result = collector.get_current_metrics()

        assert result["total_requests"] == 3
        assert result["total_successes"] == 3
        assert result["success_rate"] == 1.0
        assert "top_operations" in result
        assert "intent_distribution" in result
        assert result["recent_aggregate"] is not None

    def test_aggregation_triggered_by_interval(self):
        collector = MetricsCollector(aggregation_interval_s=0.0)
        collector.record_request(_make_request_metrics())

        # Force the last_aggregation into the past so the interval check passes
        collector.last_aggregation = time.time() - 1.0
        collector.record_request(_make_request_metrics())
        assert len(collector.aggregate_history) >= 1

    def test_aggregate_computes_percentiles(self):
        collector = MetricsCollector()
        latencies = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        for lat in latencies:
            collector.record_request(_make_request_metrics(latency_ms=float(lat)))

        collector._aggregate()
        agg = collector.aggregate_history[-1]

        assert agg.total_requests == 10
        assert agg.p50_latency_ms == pytest.approx(np.percentile(latencies, 50))
        assert agg.p95_latency_ms == pytest.approx(np.percentile(latencies, 95))
        assert agg.success_rate == 1.0


# ============================================================================
# monitoring_dashboard.py — ABTester
# ============================================================================


class TestABTester:
    def _variants(self):
        return [
            ABVariant("control", use_contextual_bandit=False, use_data_understanding=False),
            ABVariant("treatment", use_contextual_bandit=True, use_data_understanding=True),
        ]

    def test_assign_random(self):
        tester = ABTester(self._variants())
        variant = tester.assign_variant()
        assert variant.name in ("control", "treatment")

    def test_assign_deterministic(self):
        tester = ABTester(self._variants())
        v0 = tester.assign_variant(query_hash=0)
        v1 = tester.assign_variant(query_hash=1)

        assert v0.name == "control"
        assert v1.name == "treatment"

    def test_record_and_compare(self):
        tester = ABTester(self._variants())
        m = _make_request_metrics()

        tester.record_result("control", m)
        tester.record_result("treatment", m)

        comparison = tester.get_comparison()
        assert "variants" in comparison
        assert "winner" in comparison
        assert comparison["winner"] in ("control", "treatment")

    def test_record_unknown_variant_ignored(self):
        tester = ABTester(self._variants())
        tester.record_result("nonexistent", _make_request_metrics())
        # Should not raise


# ============================================================================
# monitoring_dashboard.py — Alerts
# ============================================================================


class TestAlerts:
    def test_latency_alert_fires(self):
        alert = LatencyAlert("latency_high", threshold=100.0)
        metrics = {"recent_aggregate": {"p95_latency_ms": 200.0}}
        result = alert.check(metrics)
        assert result is not None
        assert "200" in result

    def test_latency_alert_no_fire(self):
        alert = LatencyAlert("latency_high", threshold=500.0)
        metrics = {"recent_aggregate": {"p95_latency_ms": 50.0}}
        assert alert.check(metrics) is None

    def test_latency_alert_no_aggregate(self):
        alert = LatencyAlert("latency_high", threshold=100.0)
        assert alert.check({}) is None

    def test_success_rate_alert_fires(self):
        alert = SuccessRateAlert("low_success", threshold=0.9)
        result = alert.check({"success_rate": 0.5})
        assert result is not None
        assert "50" in result

    def test_success_rate_alert_no_fire(self):
        alert = SuccessRateAlert("low_success", threshold=0.5)
        assert alert.check({"success_rate": 0.9}) is None

    def test_cost_alert_fires(self):
        alert = CostAlert("high_cost", threshold=10.0)
        metrics = {"recent_aggregate": {"avg_cost": 25.0}}
        result = alert.check(metrics)
        assert result is not None

    def test_cost_alert_no_fire(self):
        alert = CostAlert("high_cost", threshold=50.0)
        metrics = {"recent_aggregate": {"avg_cost": 10.0}}
        assert alert.check(metrics) is None

    def test_cost_alert_no_aggregate(self):
        alert = CostAlert("high_cost", threshold=10.0)
        assert alert.check({}) is None

    def test_alert_manager_fires_and_stores(self):
        rules = [
            SuccessRateAlert("sr", threshold=0.95),
            LatencyAlert("lat", threshold=10.0),
        ]
        manager = AlertManager(rules)
        manager.check_alerts({
            "success_rate": 0.5,
            "recent_aggregate": {"p95_latency_ms": 500.0},
        })

        alerts = manager.get_recent_alerts()
        assert len(alerts) == 2
        assert alerts[0]["rule"] in ("sr", "lat")

    def test_alert_manager_no_rules(self):
        manager = AlertManager()
        manager.check_alerts({"success_rate": 0.1})
        assert manager.get_recent_alerts() == []


# ============================================================================
# monitoring_dashboard.py — MonitoringDashboard
# ============================================================================


class TestMonitoringDashboard:
    def test_basic_creation(self):
        dash = MonitoringDashboard()
        state = dash.get_dashboard_state()
        assert "metrics" in state
        assert "alerts" in state
        assert state["ab_testing"] is None

    def test_record_and_get_state(self):
        dash = MonitoringDashboard()
        dash.record_request(_make_request_metrics())
        state = dash.get_dashboard_state()
        assert state["metrics"]["total_requests"] == 1

    def test_with_ab_testing(self):
        variants = [
            ABVariant("a", use_contextual_bandit=False, use_data_understanding=False),
            ABVariant("b", use_contextual_bandit=True, use_data_understanding=True),
        ]
        dash = MonitoringDashboard(enable_ab_testing=True, ab_variants=variants)
        dash.record_request(_make_request_metrics(), variant_name="a")

        state = dash.get_dashboard_state()
        assert state["ab_testing"] is not None

    def test_with_alert_rules(self):
        rules = [SuccessRateAlert("sr", threshold=0.99)]
        dash = MonitoringDashboard(alert_rules=rules)

        # Record a failure to trigger alert
        dash.record_request(_make_request_metrics(success=False))

        state = dash.get_dashboard_state()
        assert len(state["alerts"]) >= 1

    def test_save_state(self, tmp_path):
        dash = MonitoringDashboard()
        dash.record_request(_make_request_metrics())

        filepath = tmp_path / "dashboard.json"
        dash.save_state(filepath)

        assert filepath.exists()
        data = json.loads(filepath.read_text())
        assert "metrics" in data

    def test_print_summary(self, capsys):
        dash = MonitoringDashboard()
        for i in range(3):
            dash.record_request(
                _make_request_metrics(ops=["inner_product", "gradient"], latency_ms=20.0 + i)
            )

        dash.print_summary()
        captured = capsys.readouterr()
        assert "MONITORING DASHBOARD" in captured.out
        assert "Total Requests: 3" in captured.out

    def test_print_summary_with_ab_and_alerts(self, capsys):
        variants = [
            ABVariant("ctrl", use_contextual_bandit=False, use_data_understanding=False),
            ABVariant("test", use_contextual_bandit=True, use_data_understanding=True),
        ]
        rules = [SuccessRateAlert("sr", threshold=0.99)]
        dash = MonitoringDashboard(
            enable_ab_testing=True, ab_variants=variants, alert_rules=rules
        )
        dash.record_request(_make_request_metrics(success=False), variant_name="ctrl")
        dash.record_request(_make_request_metrics(success=True), variant_name="test")

        dash.print_summary()
        captured = capsys.readouterr()
        assert "A/B Testing" in captured.out
        assert "Recent Alerts" in captured.out


# ============================================================================
# explainability.py — Dataclasses
# ============================================================================


class TestExplainabilityDataclasses:
    def test_why_explanation(self):
        e = WhyExplanation(
            operation="inner_product",
            reason="High score",
            full_explanation="Best option",
        )
        assert e.confidence == 1.0
        assert e.operation == "inner_product"

    def test_why_not_explanation(self):
        e = WhyNotExplanation(
            operation="gradient",
            reason="Low score",
            full_explanation="Scored too low",
            blockers=["Score: 10 vs 150"],
        )
        assert len(e.blockers) == 1

    def test_counterfactual_explanation(self):
        e = CounterfactualExplanation(
            operation="gradient",
            current_score=60.0,
            required_score=150.01,
            changes_needed=["More observations"],
            minimal_change="5 successful runs",
        )
        assert e.required_score > e.current_score


# ============================================================================
# explainability.py — FeatureImportanceAnalyzer
# ============================================================================


class TestFeatureImportanceAnalyzer:
    def test_analyze_basic(self):
        analyzer = FeatureImportanceAnalyzer()
        context = np.random.rand(100)

        # Selector always picks "inner_product"
        def selector_fn(ctx):
            return "inner_product", {}

        result = analyzer.analyze(context, "inner_product", selector_fn, top_k=5)
        assert isinstance(result, dict)
        assert len(result) <= 5

    def test_analyze_detects_important_feature(self):
        analyzer = FeatureImportanceAnalyzer()
        context = np.zeros(60)
        context[0] = 1.0  # Only feature 0 is nonzero

        def selector_fn(ctx):
            if ctx[0] > 0.5:
                return "op_a", {}
            return "op_b", {}

        result = analyzer.analyze(context, "op_a", selector_fn, top_k=3)
        # Feature 0 should be most important (zeroing it changes the selection)
        assert 0 in result
        assert result[0] == 1.0

    def test_analyze_handles_selector_exception(self):
        analyzer = FeatureImportanceAnalyzer()
        context = np.ones(10)

        def bad_selector(ctx):
            raise RuntimeError("boom")

        result = analyzer.analyze(context, "op_a", bad_selector, top_k=3)
        assert isinstance(result, dict)


# ============================================================================
# explainability.py — ExplanationGenerator
# ============================================================================


class TestExplanationGenerator:
    def _metadata(self, scores, uncertainties=None, exploration_bonus=None):
        return {
            "scores": scores,
            "uncertainties": uncertainties or {},
            "exploration_bonus": exploration_bonus or {},
        }

    def test_explain_why_high_reward(self):
        gen = ExplanationGenerator()
        md = self._metadata(
            scores={"inner_product": 200.0},
            uncertainties={"inner_product": 0.1},
            exploration_bonus={"inner_product": 5.0},
        )
        result = gen.explain_why("inner_product", {"intent": "similarity"}, md)
        assert result.reason == "High predicted reward"
        assert "highest predicted reward" in result.full_explanation
        assert result.confidence == 0.9

    def test_explain_why_exploration(self):
        gen = ExplanationGenerator()
        md = self._metadata(
            scores={"gradient": 10.0},
            uncertainties={"gradient": 0.9},
            exploration_bonus={"gradient": 50.0},
        )
        result = gen.explain_why("gradient", {}, md)
        assert result.reason == "High exploration value"

    def test_explain_why_best_available(self):
        gen = ExplanationGenerator()
        md = self._metadata(
            scores={"norm": 50.0},
            uncertainties={"norm": 0.2},
            exploration_bonus={"norm": 1.0},
        )
        result = gen.explain_why("norm", {}, md)
        assert result.reason == "Best available option"

    def test_explain_why_context_factors(self):
        gen = ExplanationGenerator()
        md = self._metadata(scores={"op": 200.0}, exploration_bonus={"op": 0.0})
        ctx = {
            "intent": "analysis",
            "budget_remaining": 10,
            "query_complexity": 0.9,
        }
        result = gen.explain_why("op", ctx, md)
        factors = result.supporting_evidence["context_factors"]
        assert any("analysis" in f.lower() for f in factors)
        assert any("budget" in f.lower() for f in factors)
        assert any("complex" in f.lower() for f in factors)

    def test_explain_why_not_not_applicable(self):
        gen = ExplanationGenerator()
        md = self._metadata(scores={"inner_product": 150.0})
        result = gen.explain_why_not("gradient", "inner_product", md)
        assert result.reason == "Not applicable"
        assert "Not in candidate set" in result.blockers

    def test_explain_why_not_much_lower(self):
        gen = ExplanationGenerator()
        md = self._metadata(scores={"inner_product": 150.0, "gradient": 30.0})
        result = gen.explain_why_not("gradient", "inner_product", md)
        assert result.reason == "Much lower predicted reward"

    def test_explain_why_not_slightly_lower(self):
        gen = ExplanationGenerator()
        md = self._metadata(scores={"inner_product": 100.0, "gradient": 90.0})
        result = gen.explain_why_not("gradient", "inner_product", md)
        assert result.reason == "Slightly lower reward"

    def test_counterfactual_small_gap(self):
        gen = ExplanationGenerator()
        md = self._metadata(scores={"inner_product": 100.0, "gradient": 95.0})
        result = gen.explain_counterfactual("gradient", "inner_product", {}, md)
        assert result.current_score == 95.0
        assert "One successful execution" in result.minimal_change

    def test_counterfactual_medium_gap(self):
        gen = ExplanationGenerator()
        md = self._metadata(scores={"inner_product": 100.0, "gradient": 60.0})
        result = gen.explain_counterfactual("gradient", "inner_product", {}, md)
        assert "3-5 successful" in result.minimal_change

    def test_counterfactual_large_gap(self):
        gen = ExplanationGenerator()
        md = self._metadata(scores={"inner_product": 200.0, "gradient": 10.0})
        result = gen.explain_counterfactual("gradient", "inner_product", {}, md)
        assert "10+ successful" in result.minimal_change

    def test_counterfactual_budget_constraint(self):
        gen = ExplanationGenerator()
        md = self._metadata(scores={"inner_product": 100.0, "gradient": 98.0})
        ctx = {"budget_remaining": 10}
        result = gen.explain_counterfactual("gradient", "inner_product", ctx, md)
        assert any("budget" in c.lower() for c in result.changes_needed)

    def test_feature_names_created(self):
        gen = ExplanationGenerator()
        assert gen.feature_names[64] == "query_length"
        assert gen.feature_names[100] == "intent_similarity"
        assert gen.feature_names[420] == "budget_remaining"

    def test_extract_context_simple_query(self):
        gen = ExplanationGenerator()
        factors = gen._extract_context_factors({"query_complexity": 0.1})
        assert any("simple" in f.lower() for f in factors)

    def test_extract_context_high_budget(self):
        gen = ExplanationGenerator()
        factors = gen._extract_context_factors({"budget_remaining": 50})
        assert any("high budget" in f.lower() for f in factors)


# ============================================================================
# explainability.py — ExplainableSelector
# ============================================================================


class TestExplainableSelector:
    class _MockSelector:
        def select(self, query_text, intent=None, **kwargs):
            scores = {"inner_product": 150.0, "gradient": 60.0}
            return "inner_product", {
                "scores": scores,
                "uncertainties": {"inner_product": 0.2, "gradient": 0.8},
                "exploration_bonus": {"inner_product": 2.0, "gradient": 8.0},
            }

    def test_select_and_explain(self):
        es = ExplainableSelector(self._MockSelector())
        op, md = es.select_with_explanation("find similar docs", intent="similarity")

        assert op == "inner_product"
        assert es.last_selection == "inner_product"

        why = es.explain_why()
        assert why.operation == "inner_product"

    def test_explain_why_not(self):
        es = ExplainableSelector(self._MockSelector())
        es.select_with_explanation("test query")

        why_not = es.explain_why_not("gradient")
        assert why_not.operation == "gradient"
        assert len(why_not.blockers) > 0

    def test_explain_counterfactual(self):
        es = ExplainableSelector(self._MockSelector())
        es.select_with_explanation("test query")

        cf = es.explain_counterfactual("gradient")
        assert cf.operation == "gradient"
        assert cf.required_score > cf.current_score

    def test_explain_before_select_raises(self):
        es = ExplainableSelector(self._MockSelector())
        with pytest.raises(ValueError, match="No selection"):
            es.explain_why()
        with pytest.raises(ValueError, match="No selection"):
            es.explain_why_not("gradient")
        with pytest.raises(ValueError, match="No selection"):
            es.explain_counterfactual("gradient")

    def test_explain_why_specific_operation(self):
        es = ExplainableSelector(self._MockSelector())
        es.select_with_explanation("query")
        why = es.explain_why("gradient")
        assert why.operation == "gradient"


# ============================================================================
# operation_selector.py — MathOperation
# ============================================================================


class TestMathOperation:
    def test_is_applicable(self):
        op = MathOperation(
            name="test",
            domain=MathDomain.ALGEBRA,
            level=OperationLevel.BASIC,
            description="test op",
            use_cases=[QueryIntent.SIMILARITY],
        )
        assert op.is_applicable(QueryIntent.SIMILARITY) is True
        assert op.is_applicable(QueryIntent.GENERATION) is False


# ============================================================================
# operation_selector.py — OperationPlan
# ============================================================================


class TestOperationPlan:
    def _make_plan(self):
        ops = [
            MathOperation("inner_product", MathDomain.ALGEBRA, OperationLevel.BASIC,
                          "ip", [QueryIntent.SIMILARITY], estimated_cost=1),
            MathOperation("gradient", MathDomain.ANALYSIS, OperationLevel.MODERATE,
                          "grad", [QueryIntent.OPTIMIZATION], estimated_cost=5),
        ]
        return OperationPlan(
            operations=ops,
            justifications={"inner_product": "similarity", "gradient": "optimization"},
            total_cost=6,
            domains_used={MathDomain.ALGEBRA, MathDomain.ANALYSIS},
        )

    def test_get_execution_order(self):
        plan = self._make_plan()
        order = plan.get_execution_order()
        assert order == ["inner_product", "gradient"]

    def test_summarize(self):
        plan = self._make_plan()
        summary = plan.summarize()
        assert "Mathematical Operation Plan" in summary
        assert "inner_product" in summary
        assert "gradient" in summary
        assert "Operations: 2" in summary


# ============================================================================
# operation_selector.py — MathOperationSelector
# ============================================================================


class TestMathOperationSelector:
    def test_init_builds_catalog(self):
        selector = MathOperationSelector()
        assert len(selector.operations) > 10
        assert "inner_product" in selector.operations
        assert "ricci_flow" in selector.operations

    def test_classify_intent_similarity(self):
        selector = MathOperationSelector()
        intents = selector.classify_intent("Find similar documents")
        assert QueryIntent.SIMILARITY in intents

    def test_classify_intent_optimization(self):
        selector = MathOperationSelector()
        intents = selector.classify_intent("Optimize the retrieval quality")
        assert QueryIntent.OPTIMIZATION in intents

    def test_classify_intent_verification(self):
        selector = MathOperationSelector()
        intents = selector.classify_intent("Verify metric correctness")
        assert QueryIntent.VERIFICATION in intents

    def test_classify_intent_default_analysis(self):
        selector = MathOperationSelector()
        intents = selector.classify_intent("xyzabc nonsense")
        assert intents == [QueryIntent.ANALYSIS]

    def test_classify_intent_with_context(self):
        selector = MathOperationSelector()
        intents = selector.classify_intent("some query", context={"has_embeddings": True})
        assert QueryIntent.SIMILARITY in intents

    def test_classify_intent_optimization_context(self):
        selector = MathOperationSelector()
        intents = selector.classify_intent("data", context={"requires_optimization": True})
        assert QueryIntent.OPTIMIZATION in intents

    def test_classify_intent_verification_context(self):
        selector = MathOperationSelector()
        intents = selector.classify_intent("data", context={"needs_verification": True})
        assert QueryIntent.VERIFICATION in intents

    def test_plan_operations_basic(self):
        selector = MathOperationSelector()
        plan = selector.plan_operations("Find similar documents")

        assert len(plan.operations) > 0
        assert plan.total_cost > 0
        assert len(plan.domains_used) > 0
        assert plan.metadata["primary_intent"] == "similarity"

    def test_plan_operations_budget_constraint(self):
        selector = MathOperationSelector()
        plan = selector.plan_operations("Analyze everything", budget=5)

        assert plan.total_cost <= 5

    def test_plan_operations_no_expensive_by_default(self):
        selector = MathOperationSelector()
        plan = selector.plan_operations("Transform and optimize all features")
        op_names = plan.get_execution_order()
        assert "ricci_flow" not in op_names

    def test_plan_operations_enable_expensive(self):
        selector = MathOperationSelector()
        plan = selector.plan_operations(
            "Transform features deeply",
            enable_expensive=True,
        )
        # Expensive ops should be candidates now
        assert plan.total_cost > 0

    def test_topological_sort_respects_prerequisites(self):
        selector = MathOperationSelector()
        plan = selector.plan_operations("Analyze convergence patterns")
        order = plan.get_execution_order()

        # If both inner_product and eigenvalues are present,
        # inner_product (prerequisite) should come before eigenvalues
        if "inner_product" in order and "eigenvalues" in order:
            assert order.index("inner_product") < order.index("eigenvalues")

    def test_topological_sort_cycle_detection(self):
        selector = MathOperationSelector()
        # Create ops with a cycle
        op_a = MathOperation("a", MathDomain.ALGEBRA, OperationLevel.BASIC,
                             "a", [QueryIntent.SIMILARITY], prerequisites=["b"])
        op_b = MathOperation("b", MathDomain.ALGEBRA, OperationLevel.BASIC,
                             "b", [QueryIntent.SIMILARITY], prerequisites=["a"])
        result = selector._topological_sort([op_a, op_b])
        # Should return original order when cycle detected
        assert len(result) == 2

    def test_get_statistics_empty(self):
        selector = MathOperationSelector()
        stats = selector.get_statistics()
        assert stats["total_plans"] == 0

    def test_get_statistics_with_history(self):
        selector = MathOperationSelector()
        selector.plan_operations("Find similar items")
        selector.plan_operations("Optimize retrieval")

        stats = selector.get_statistics()
        assert stats["total_plans"] == 2
        assert stats["avg_operations_per_plan"] > 0
        assert stats["avg_cost_per_plan"] > 0
        assert len(stats["most_common_operations"]) > 0
        assert len(stats["intent_distribution"]) > 0
        assert stats["total_operations_available"] == len(selector.operations)

    def test_create_selector_factory(self):
        selector = create_selector()
        assert isinstance(selector, MathOperationSelector)
        assert len(selector.operations) > 0

    def test_plan_records_history(self):
        selector = MathOperationSelector()
        assert len(selector.planning_history) == 0
        selector.plan_operations("test query")
        assert len(selector.planning_history) == 1

    def test_multiple_intents_detected(self):
        selector = MathOperationSelector()
        intents = selector.classify_intent("Find and verify similar patterns")
        # Should detect both SIMILARITY and VERIFICATION
        assert len(intents) >= 2
