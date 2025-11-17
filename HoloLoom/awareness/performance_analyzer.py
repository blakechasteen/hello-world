#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3: Learning from Outcomes - Performance Analyzer

Identifies performance bottlenecks in context packing + LLM generation pipeline.

Analyzes:
1. Latency breakdown (packing vs. LLM vs. other)
2. Budget utilization (allocated vs. used)
3. Context utilization (context elements vs. actually used in response)
4. Anomaly detection (outliers, regressions)

Usage:
    analyzer = PerformanceAnalyzer(outcome_tracker=tracker)

    # Analyze performance
    analysis = analyzer.analyze()

    # Get bottlenecks
    bottlenecks = analyzer.get_bottlenecks()
    for b in bottlenecks:
        print(f"{b.component}: {b.severity} - {b.description}")

    # Get anomalies
    anomalies = analyzer.detect_anomalies()
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import statistics

from .outcome_tracker import OutcomeTracker, Outcome


class BottleneckSeverity(str, Enum):
    """Bottleneck severity levels"""
    LOW = "low"           # <10% impact
    MEDIUM = "medium"     # 10-25% impact
    HIGH = "high"         # 25-50% impact
    CRITICAL = "critical" # >50% impact


class AnomalyType(str, Enum):
    """Types of performance anomalies"""
    LATENCY_SPIKE = "latency_spike"           # Sudden latency increase
    QUALITY_DROP = "quality_drop"             # Quality degradation
    EFFICIENCY_DROP = "efficiency_drop"       # Budget efficiency drop
    UNDERUTILIZATION = "underutilization"     # Context not being used
    OVERALLOCATION = "overallocation"         # Budget >> actual usage


@dataclass
class Bottleneck:
    """
    Performance bottleneck identification.

    Points to a specific component causing slowdown.
    """

    component: str                  # "packing", "llm", "total"
    severity: BottleneckSeverity    # Impact level
    description: str                # Human-readable description
    current_value: float            # Current latency/metric
    expected_value: float           # Expected latency/metric
    impact_pct: float               # Percentage impact
    recommendation: str             # How to fix


@dataclass
class Anomaly:
    """
    Performance anomaly detection result.

    Identifies outliers or regressions.
    """

    type: AnomalyType
    severity: BottleneckSeverity
    description: str
    affected_queries: int           # Number of queries affected
    metric_value: float             # Anomalous value
    baseline_value: float           # Expected/baseline value
    recommendation: str


class PerformanceAnalyzer:
    """
    Analyze performance of context packing + LLM generation.

    Identifies bottlenecks and anomalies.
    """

    def __init__(
        self,
        outcome_tracker: OutcomeTracker,
        latency_threshold_ms: float = 500.0,     # Queries >500ms considered slow
        efficiency_threshold: float = 0.0001,    # Minimum acceptable efficiency
        utilization_threshold: float = 0.3       # Minimum context utilization
    ):
        """
        Initialize performance analyzer.

        Args:
            outcome_tracker: OutcomeTracker instance
            latency_threshold_ms: Latency threshold for slow queries
            efficiency_threshold: Minimum acceptable budget efficiency
            utilization_threshold: Minimum context utilization percentage
        """
        self.tracker = outcome_tracker
        self.latency_threshold = latency_threshold_ms
        self.efficiency_threshold = efficiency_threshold
        self.utilization_threshold = utilization_threshold

    def analyze(self) -> Dict[str, Any]:
        """
        Comprehensive performance analysis.

        Returns:
            Analysis dict with breakdowns, trends, insights
        """
        if not self.tracker.outcomes:
            return {"error": "No outcomes to analyze"}

        outcomes = self.tracker.outcomes

        return {
            "latency_breakdown": self._analyze_latency(outcomes),
            "budget_analysis": self._analyze_budget(outcomes),
            "quality_trends": self._analyze_quality(outcomes),
            "utilization_analysis": self._analyze_utilization(outcomes),
            "sample_size": len(outcomes)
        }

    def _analyze_latency(self, outcomes: List[Outcome]) -> Dict[str, Any]:
        """Analyze latency breakdown"""
        total_latencies = [o.latency_ms for o in outcomes if o.latency_ms > 0]
        packing_latencies = [o.latency_packing_ms for o in outcomes if o.latency_packing_ms > 0]
        llm_latencies = [o.latency_llm_ms for o in outcomes if o.latency_llm_ms > 0]

        if not total_latencies:
            return {}

        # Calculate averages
        avg_total = statistics.mean(total_latencies)
        avg_packing = statistics.mean(packing_latencies) if packing_latencies else 0.0
        avg_llm = statistics.mean(llm_latencies) if llm_latencies else 0.0

        # Calculate percentages
        packing_pct = (avg_packing / avg_total) * 100 if avg_total > 0 else 0.0
        llm_pct = (avg_llm / avg_total) * 100 if avg_total > 0 else 0.0
        other_pct = 100.0 - packing_pct - llm_pct

        return {
            "avg_total_ms": avg_total,
            "avg_packing_ms": avg_packing,
            "avg_llm_ms": avg_llm,
            "packing_percentage": packing_pct,
            "llm_percentage": llm_pct,
            "other_percentage": other_pct,
            "slow_queries": len([l for l in total_latencies if l > self.latency_threshold]),
            "p50_ms": statistics.median(total_latencies),
            "p95_ms": self._percentile(total_latencies, 0.95),
            "p99_ms": self._percentile(total_latencies, 0.99)
        }

    def _analyze_budget(self, outcomes: List[Outcome]) -> Dict[str, Any]:
        """Analyze budget allocation vs. usage"""
        allocated = [o.budget_allocated for o in outcomes]
        used = [o.budget_used for o in outcomes]
        efficiencies = [o.budget_efficiency for o in outcomes]

        utilization_rates = [
            (u / a) * 100 if a > 0 else 0.0
            for u, a in zip(used, allocated)
        ]

        return {
            "avg_allocated": statistics.mean(allocated),
            "avg_used": statistics.mean(used),
            "avg_utilization_pct": statistics.mean(utilization_rates),
            "avg_efficiency": statistics.mean(efficiencies),
            "waste_pct": 100.0 - statistics.mean(utilization_rates),
            "underutilized": len([r for r in utilization_rates if r < 50.0]),
            "overallocated": len([r for r in utilization_rates if r < 30.0])
        }

    def _analyze_quality(self, outcomes: List[Outcome]) -> Dict[str, Any]:
        """Analyze quality trends"""
        qualities = [o.quality_overall for o in outcomes]
        coherence = [o.quality_coherence for o in outcomes if o.quality_coherence > 0]
        completeness = [o.quality_completeness for o in outcomes if o.quality_completeness > 0]
        relevance = [o.quality_relevance for o in outcomes if o.quality_relevance > 0]

        # Detect trend (improving/degrading/stable)
        if len(qualities) >= 10:
            recent = statistics.mean(qualities[-10:])
            older = statistics.mean(qualities[:10])
            trend = "improving" if recent > older else ("degrading" if recent < older else "stable")
        else:
            trend = "insufficient_data"

        return {
            "avg_quality": statistics.mean(qualities),
            "std_quality": statistics.stdev(qualities) if len(qualities) > 1 else 0.0,
            "min_quality": min(qualities),
            "max_quality": max(qualities),
            "avg_coherence": statistics.mean(coherence) if coherence else 0.0,
            "avg_completeness": statistics.mean(completeness) if completeness else 0.0,
            "avg_relevance": statistics.mean(relevance) if relevance else 0.0,
            "trend": trend,
            "low_quality_queries": len([q for q in qualities if q < 0.6])
        }

    def _analyze_utilization(self, outcomes: List[Outcome]) -> Dict[str, Any]:
        """Analyze context utilization"""
        utilizations = [o.context_used_pct for o in outcomes if o.context_used_pct > 0]

        if not utilizations:
            return {"error": "No utilization data"}

        return {
            "avg_utilization_pct": statistics.mean(utilizations),
            "median_utilization_pct": statistics.median(utilizations),
            "low_utilization": len([u for u in utilizations if u < self.utilization_threshold * 100]),
            "waste_estimate_pct": 100.0 - statistics.mean(utilizations)
        }

    def get_bottlenecks(
        self,
        min_severity: BottleneckSeverity = BottleneckSeverity.MEDIUM
    ) -> List[Bottleneck]:
        """
        Identify performance bottlenecks.

        Args:
            min_severity: Minimum severity to include

        Returns:
            List of bottlenecks
        """
        bottlenecks = []

        analysis = self.analyze()
        if "error" in analysis:
            return bottlenecks

        latency = analysis.get("latency_breakdown", {})
        budget = analysis.get("budget_analysis", {})

        # Check packing latency
        packing_pct = latency.get("packing_percentage", 0.0)
        if packing_pct > 50.0:
            bottlenecks.append(Bottleneck(
                component="packing",
                severity=BottleneckSeverity.CRITICAL if packing_pct > 70 else BottleneckSeverity.HIGH,
                description=f"Packing consumes {packing_pct:.1f}% of total latency",
                current_value=latency.get("avg_packing_ms", 0.0),
                expected_value=latency.get("avg_total_ms", 0.0) * 0.3,  # Expect <30%
                impact_pct=packing_pct,
                recommendation="Optimize memory retrieval, reduce compression overhead, or cache packed contexts"
            ))

        # Check LLM latency
        llm_pct = latency.get("llm_percentage", 0.0)
        if llm_pct > 70.0:
            bottlenecks.append(Bottleneck(
                component="llm",
                severity=BottleneckSeverity.HIGH,
                description=f"LLM consumes {llm_pct:.1f}% of total latency",
                current_value=latency.get("avg_llm_ms", 0.0),
                expected_value=latency.get("avg_total_ms", 0.0) * 0.6,  # Expect <60%
                impact_pct=llm_pct,
                recommendation="Reduce context size, use faster model, or implement streaming"
            ))

        # Check budget waste
        waste_pct = budget.get("waste_pct", 0.0)
        if waste_pct > 40.0:
            bottlenecks.append(Bottleneck(
                component="budget",
                severity=BottleneckSeverity.HIGH if waste_pct > 60 else BottleneckSeverity.MEDIUM,
                description=f"Budget waste: {waste_pct:.1f}% allocated but not used",
                current_value=budget.get("avg_allocated", 0.0),
                expected_value=budget.get("avg_used", 0.0) * 1.2,  # Expect ≤20% overhead
                impact_pct=waste_pct,
                recommendation="Reduce budget allocation or improve adaptive budgeting"
            ))

        # Filter by severity
        severity_order = {
            BottleneckSeverity.LOW: 0,
            BottleneckSeverity.MEDIUM: 1,
            BottleneckSeverity.HIGH: 2,
            BottleneckSeverity.CRITICAL: 3
        }

        return [
            b for b in bottlenecks
            if severity_order[b.severity] >= severity_order[min_severity]
        ]

    def detect_anomalies(self) -> List[Anomaly]:
        """
        Detect performance anomalies.

        Returns:
            List of anomalies
        """
        anomalies = []

        outcomes = self.tracker.outcomes
        if len(outcomes) < 10:
            return anomalies  # Not enough data

        # Recent outcomes (last 20%)
        split_point = int(len(outcomes) * 0.8)
        baseline = outcomes[:split_point]
        recent = outcomes[split_point:]

        # Latency spike detection
        baseline_latency = statistics.mean(o.latency_ms for o in baseline if o.latency_ms > 0)
        recent_latency = statistics.mean(o.latency_ms for o in recent if o.latency_ms > 0)

        if recent_latency > baseline_latency * 1.5:  # 50% increase
            anomalies.append(Anomaly(
                type=AnomalyType.LATENCY_SPIKE,
                severity=BottleneckSeverity.HIGH,
                description=f"Latency increased {((recent_latency / baseline_latency - 1) * 100):.1f}%",
                affected_queries=len(recent),
                metric_value=recent_latency,
                baseline_value=baseline_latency,
                recommendation="Investigate recent changes, check for LLM provider issues"
            ))

        # Quality drop detection
        baseline_quality = statistics.mean(o.quality_overall for o in baseline)
        recent_quality = statistics.mean(o.quality_overall for o in recent)

        if recent_quality < baseline_quality - 0.1:  # 10% drop
            anomalies.append(Anomaly(
                type=AnomalyType.QUALITY_DROP,
                severity=BottleneckSeverity.CRITICAL,
                description=f"Quality dropped {(baseline_quality - recent_quality):.2f}",
                affected_queries=len(recent),
                metric_value=recent_quality,
                baseline_value=baseline_quality,
                recommendation="Check adaptive threshold tuner, verify LLM provider, review recent config changes"
            ))

        # Efficiency drop detection
        baseline_efficiency = statistics.mean(o.budget_efficiency for o in baseline)
        recent_efficiency = statistics.mean(o.budget_efficiency for o in recent)

        if recent_efficiency < baseline_efficiency * 0.7:  # 30% drop
            anomalies.append(Anomaly(
                type=AnomalyType.EFFICIENCY_DROP,
                severity=BottleneckSeverity.MEDIUM,
                description=f"Efficiency dropped {((1 - recent_efficiency / baseline_efficiency) * 100):.1f}%",
                affected_queries=len(recent),
                metric_value=recent_efficiency,
                baseline_value=baseline_efficiency,
                recommendation="Review budget allocation, check for quality regressions"
            ))

        # Underutilization detection
        underutilized = [o for o in recent if o.context_used_pct > 0 and o.context_used_pct < 30.0]
        if len(underutilized) > len(recent) * 0.3:  # >30% of queries
            anomalies.append(Anomaly(
                type=AnomalyType.UNDERUTILIZATION,
                severity=BottleneckSeverity.MEDIUM,
                description=f"{len(underutilized)} queries using <30% of context",
                affected_queries=len(underutilized),
                metric_value=statistics.mean(o.context_used_pct for o in underutilized),
                baseline_value=70.0,  # Expected utilization
                recommendation="Reduce importance threshold, improve context relevance"
            ))

        # Overallocation detection
        overallocated = [
            o for o in recent
            if o.budget_allocated > 0 and (o.budget_used / o.budget_allocated) < 0.4
        ]
        if len(overallocated) > len(recent) * 0.3:  # >30% of queries
            anomalies.append(Anomaly(
                type=AnomalyType.OVERALLOCATION,
                severity=BottleneckSeverity.MEDIUM,
                description=f"{len(overallocated)} queries using <40% of allocated budget",
                affected_queries=len(overallocated),
                metric_value=statistics.mean(
                    o.budget_used / o.budget_allocated for o in overallocated
                ) * 100,
                baseline_value=80.0,  # Expected utilization
                recommendation="Reduce budget allocation in adaptive budgeting (Phase 2)"
            ))

        return anomalies

    def _percentile(self, data: List[float], p: float) -> float:
        """Calculate percentile"""
        if not data:
            return 0.0

        sorted_data = sorted(data)
        index = int(len(sorted_data) * p)
        return sorted_data[min(index, len(sorted_data) - 1)]
