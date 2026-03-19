"""
MRF Analytics for CARTS
=======================

Impact assessment and analytics for MRF-enhanced attack strategies.

Tracks:
- Enhancement effectiveness (success rate before/after)
- Strategy-level impact metrics
- Enhancement type performance
- A/B testing for enhancement validation
- Prometheus-style metrics export

Philosophy: "Measure what matters, learn what works."

Author: CARTS Team
Date: 2025-12-03
"""

import hashlib
import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .mrf_payloads import EnhancementType
from .strategies import AttackStrategy

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EnhancementMetric:
    """
    Single metric for an MRF enhancement event.

    Tracks before/after effectiveness to measure MRF impact.
    """
    timestamp: float
    strategy: AttackStrategy
    enhancement_type: EnhancementType
    success_before: bool
    success_after: bool
    severity_before: float  # 0.0-1.0
    severity_after: float  # 0.0-1.0
    enhancement_confidence: float  # MRF enhancement confidence
    payload_id: str  # Hash of payload for deduplication
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def improved(self) -> bool:
        """True if enhancement improved attack effectiveness."""
        if self.success_after and not self.success_before:
            return True
        if self.success_before and self.success_after:
            return self.severity_after > self.severity_before
        return False

    @property
    def effectiveness_delta(self) -> float:
        """
        Calculate effectiveness improvement.

        Formula: (success_after × severity_after) - (success_before × severity_before)
        """
        before = (1.0 if self.success_before else 0.0) * self.severity_before
        after = (1.0 if self.success_after else 0.0) * self.severity_after
        return after - before

    @property
    def timestamp_dt(self) -> datetime:
        """Convert to datetime."""
        return datetime.fromtimestamp(self.timestamp)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'timestamp': self.timestamp,
            'strategy': self.strategy.value,
            'enhancement_type': self.enhancement_type.value,
            'success_before': self.success_before,
            'success_after': self.success_after,
            'severity_before': self.severity_before,
            'severity_after': self.severity_after,
            'enhancement_confidence': self.enhancement_confidence,
            'payload_id': self.payload_id,
            'improved': self.improved,
            'effectiveness_delta': self.effectiveness_delta,
            'metadata': self.metadata,
        }


@dataclass
class StrategyImpact:
    """
    Aggregated impact metrics for a single attack strategy.
    """
    strategy: AttackStrategy
    total_enhancements: int = 0
    improvements: int = 0
    degradations: int = 0
    neutral: int = 0
    avg_effectiveness_delta: float = 0.0
    avg_enhancement_confidence: float = 0.0
    success_rate_before: float = 0.0
    success_rate_after: float = 0.0
    enhancement_type_usage: dict[str, int] = field(default_factory=dict)

    @property
    def improvement_rate(self) -> float:
        """Fraction of enhancements that improved effectiveness."""
        if self.total_enhancements == 0:
            return 0.0
        return self.improvements / self.total_enhancements

    @property
    def success_rate_lift(self) -> float:
        """Absolute lift in success rate."""
        return self.success_rate_after - self.success_rate_before

    @property
    def relative_improvement(self) -> float:
        """Relative improvement percentage."""
        if self.success_rate_before == 0:
            return 0.0 if self.success_rate_after == 0 else 1.0
        return (self.success_rate_after - self.success_rate_before) / self.success_rate_before

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'strategy': self.strategy.value,
            'total_enhancements': self.total_enhancements,
            'improvements': self.improvements,
            'degradations': self.degradations,
            'neutral': self.neutral,
            'improvement_rate': self.improvement_rate,
            'avg_effectiveness_delta': self.avg_effectiveness_delta,
            'avg_enhancement_confidence': self.avg_enhancement_confidence,
            'success_rate_before': self.success_rate_before,
            'success_rate_after': self.success_rate_after,
            'success_rate_lift': self.success_rate_lift,
            'relative_improvement': self.relative_improvement,
            'enhancement_type_usage': self.enhancement_type_usage,
        }


@dataclass
class EnhancementTypeImpact:
    """
    Aggregated impact metrics for an enhancement type.
    """
    enhancement_type: EnhancementType
    total_uses: int = 0
    improvements: int = 0
    degradations: int = 0
    avg_effectiveness_delta: float = 0.0
    strategy_success: dict[str, float] = field(default_factory=dict)

    @property
    def improvement_rate(self) -> float:
        """Fraction of uses that improved effectiveness."""
        if self.total_uses == 0:
            return 0.0
        return self.improvements / self.total_uses

    @property
    def expected_reward(self) -> float:
        """
        Expected reward for Thompson Sampling.

        Combines improvement rate and average effectiveness delta.
        """
        return 0.7 * self.improvement_rate + 0.3 * max(0, self.avg_effectiveness_delta)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'enhancement_type': self.enhancement_type.value,
            'total_uses': self.total_uses,
            'improvements': self.improvements,
            'degradations': self.degradations,
            'improvement_rate': self.improvement_rate,
            'avg_effectiveness_delta': self.avg_effectiveness_delta,
            'expected_reward': self.expected_reward,
            'strategy_success': self.strategy_success,
        }


# =============================================================================
# A/B Test Group
# =============================================================================

class ABGroup(Enum):
    """A/B test group for enhancement validation."""
    CONTROL = "control"  # No MRF enhancement
    TREATMENT = "treatment"  # MRF enhancement applied


@dataclass
class ABTestResult:
    """Result from an A/B test observation."""
    timestamp: float
    group: ABGroup
    strategy: AttackStrategy
    success: bool
    severity: float
    user_id: str  # For consistent assignment

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'timestamp': self.timestamp,
            'group': self.group.value,
            'strategy': self.strategy.value,
            'success': self.success,
            'severity': self.severity,
            'user_id': self.user_id,
        }


# =============================================================================
# Main Analytics Class
# =============================================================================

class MRFImpactAnalytics:
    """
    Analytics for measuring MRF enhancement impact on attack effectiveness.

    Features:
    - Track before/after effectiveness for all enhancements
    - Aggregate metrics by strategy and enhancement type
    - A/B testing for statistically validating improvements
    - Prometheus-style metrics export
    - Dashboard HTML generation

    Example:
        analytics = MRFImpactAnalytics()

        # Log enhancement event
        analytics.log_enhancement(
            strategy=AttackStrategy.PROMPT_INJECTION,
            enhancement_type=EnhancementType.SEMANTIC_OBFUSCATION,
            success_before=False,
            success_after=True,
            severity_before=0.0,
            severity_after=0.72,
            enhancement_confidence=0.85,
            payload_id="payload_hash_123"
        )

        # Get strategy impact
        impact = analytics.get_strategy_impact(AttackStrategy.PROMPT_INJECTION)
        print(f"Improvement rate: {impact.improvement_rate:.1%}")

        # Export metrics
        metrics = analytics.export_prometheus_metrics()
    """

    def __init__(
        self,
        persist_path: Path | None = None,
        enable_ab_testing: bool = True,
        ab_traffic_split: float = 0.2  # 20% treatment
    ):
        """
        Initialize analytics.

        Args:
            persist_path: Path to persist analytics data
            enable_ab_testing: Enable A/B testing framework
            ab_traffic_split: Fraction of traffic to treatment group
        """
        self.persist_path = persist_path or Path("./redteam_analytics")
        self.persist_path.mkdir(parents=True, exist_ok=True)

        self.enable_ab_testing = enable_ab_testing
        self.ab_traffic_split = ab_traffic_split

        # Metrics storage
        self.metrics: list[EnhancementMetric] = []

        # Aggregated impact by strategy
        self.strategy_impact: dict[AttackStrategy, StrategyImpact] = {
            strategy: StrategyImpact(strategy=strategy)
            for strategy in AttackStrategy
        }

        # Aggregated impact by enhancement type
        self.enhancement_impact: dict[EnhancementType, EnhancementTypeImpact] = {
            etype: EnhancementTypeImpact(enhancement_type=etype)
            for etype in EnhancementType
        }

        # A/B testing
        self.ab_results: dict[ABGroup, list[ABTestResult]] = {
            ABGroup.CONTROL: [],
            ABGroup.TREATMENT: []
        }
        self.ab_assignments: dict[str, ABGroup] = {}

        # Load persisted data
        self._load_data()

    # =========================================================================
    # Logging Methods
    # =========================================================================

    def log_enhancement(
        self,
        strategy: AttackStrategy,
        enhancement_type: EnhancementType,
        success_before: bool,
        success_after: bool,
        severity_before: float,
        severity_after: float,
        enhancement_confidence: float,
        payload_id: str,
        metadata: dict[str, Any] | None = None
    ) -> EnhancementMetric:
        """
        Log an MRF enhancement event with before/after effectiveness.

        Args:
            strategy: Attack strategy used
            enhancement_type: Type of MRF enhancement applied
            success_before: Attack succeeded before enhancement?
            success_after: Attack succeeded after enhancement?
            severity_before: Severity before enhancement (0-1)
            severity_after: Severity after enhancement (0-1)
            enhancement_confidence: MRF enhancement confidence
            payload_id: Unique payload identifier
            metadata: Additional context

        Returns:
            EnhancementMetric object
        """
        metric = EnhancementMetric(
            timestamp=time.time(),
            strategy=strategy,
            enhancement_type=enhancement_type,
            success_before=success_before,
            success_after=success_after,
            severity_before=severity_before,
            severity_after=severity_after,
            enhancement_confidence=enhancement_confidence,
            payload_id=payload_id,
            metadata=metadata or {}
        )

        self.metrics.append(metric)
        self._update_aggregates(metric)

        # Auto-persist every 100 entries
        if len(self.metrics) % 100 == 0:
            self._save_data()

        logger.debug(
            f"Logged enhancement: {strategy.value}/{enhancement_type.value} "
            f"improved={metric.improved} delta={metric.effectiveness_delta:.3f}"
        )

        return metric

    def log_ab_result(
        self,
        strategy: AttackStrategy,
        success: bool,
        severity: float,
        user_id: str,
        force_group: ABGroup | None = None
    ) -> ABTestResult:
        """
        Log an A/B test observation.

        Args:
            strategy: Attack strategy used
            success: Attack succeeded?
            severity: Attack severity (0-1)
            user_id: User/session identifier for consistent assignment
            force_group: Force assignment to specific group

        Returns:
            ABTestResult object
        """
        if not self.enable_ab_testing:
            raise ValueError("A/B testing not enabled")

        # Assign group (deterministic by user_id)
        if force_group:
            group = force_group
        else:
            group = self._assign_ab_group(user_id)

        result = ABTestResult(
            timestamp=time.time(),
            group=group,
            strategy=strategy,
            success=success,
            severity=severity,
            user_id=user_id
        )

        self.ab_results[group].append(result)

        logger.debug(
            f"A/B result: {group.value}/{strategy.value} "
            f"success={success} severity={severity:.2f}"
        )

        return result

    # =========================================================================
    # Analysis Methods
    # =========================================================================

    def get_strategy_impact(self, strategy: AttackStrategy) -> StrategyImpact:
        """Get aggregated impact for a specific strategy."""
        return self.strategy_impact[strategy]

    def get_enhancement_impact(self, enhancement_type: EnhancementType) -> EnhancementTypeImpact:
        """Get aggregated impact for a specific enhancement type."""
        return self.enhancement_impact[enhancement_type]

    def get_best_strategies(self, limit: int = 5) -> list[StrategyImpact]:
        """
        Get strategies with highest improvement rates.

        Args:
            limit: Maximum strategies to return

        Returns:
            List of StrategyImpact sorted by improvement rate
        """
        impacts = [
            impact for impact in self.strategy_impact.values()
            if impact.total_enhancements > 0
        ]
        impacts.sort(key=lambda x: x.improvement_rate, reverse=True)
        return impacts[:limit]

    def get_best_enhancement_types(self, limit: int = 5) -> list[EnhancementTypeImpact]:
        """
        Get enhancement types with highest expected reward.

        Args:
            limit: Maximum types to return

        Returns:
            List of EnhancementTypeImpact sorted by expected reward
        """
        impacts = [
            impact for impact in self.enhancement_impact.values()
            if impact.total_uses > 0
        ]
        impacts.sort(key=lambda x: x.expected_reward, reverse=True)
        return impacts[:limit]

    def get_ab_analysis(self, min_samples: int = 30) -> dict[str, Any]:
        """
        Analyze A/B test results.

        Args:
            min_samples: Minimum samples per group for analysis

        Returns:
            Dictionary with analysis results including:
            - is_significant: True if statistically significant
            - p_value: Statistical significance
            - cohens_d: Effect size
            - control_success_rate: Control group success rate
            - treatment_success_rate: Treatment group success rate
            - lift: Relative improvement
            - deployment_decision: "deploy", "monitor", "reject", "inconclusive"
        """
        control = self.ab_results[ABGroup.CONTROL]
        treatment = self.ab_results[ABGroup.TREATMENT]

        result = {
            'control_count': len(control),
            'treatment_count': len(treatment),
            'is_significant': False,
            'p_value': 1.0,
            'cohens_d': 0.0,
            'control_success_rate': 0.0,
            'treatment_success_rate': 0.0,
            'lift': 0.0,
            'deployment_decision': 'inconclusive'
        }

        # Need minimum samples
        if len(control) < min_samples or len(treatment) < min_samples:
            return result

        # Calculate success rates
        control_successes = [1.0 if r.success else 0.0 for r in control]
        treatment_successes = [1.0 if r.success else 0.0 for r in treatment]

        control_rate = statistics.mean(control_successes)
        treatment_rate = statistics.mean(treatment_successes)

        result['control_success_rate'] = control_rate
        result['treatment_success_rate'] = treatment_rate
        result['lift'] = (
            (treatment_rate - control_rate) / control_rate
            if control_rate > 0 else 0.0
        )

        # Statistical significance (Welch's t-test approximation)
        try:
            control_std = statistics.stdev(control_successes) if len(control_successes) > 1 else 0.01
            treatment_std = statistics.stdev(treatment_successes) if len(treatment_successes) > 1 else 0.01

            # Pooled standard error
            se = ((control_std**2 / len(control)) + (treatment_std**2 / len(treatment))) ** 0.5

            if se > 0:
                t_stat = abs(treatment_rate - control_rate) / se
                # Rough p-value approximation (two-tailed)
                # For t > 1.96, p < 0.05; for t > 2.58, p < 0.01
                if t_stat > 2.58:
                    p_value = 0.01
                elif t_stat > 1.96:
                    p_value = 0.05
                elif t_stat > 1.645:
                    p_value = 0.10
                else:
                    p_value = 0.20

                result['p_value'] = p_value
                result['is_significant'] = p_value < 0.05

            # Cohen's d effect size
            pooled_std = ((control_std**2 + treatment_std**2) / 2) ** 0.5
            if pooled_std > 0:
                result['cohens_d'] = (treatment_rate - control_rate) / pooled_std

        except Exception as e:
            logger.warning(f"Statistical analysis error: {e}")

        # Deployment decision
        if not result['is_significant']:
            result['deployment_decision'] = 'inconclusive'
        elif treatment_rate > control_rate and result['cohens_d'] >= 0.2:
            result['deployment_decision'] = 'deploy'
        elif treatment_rate >= control_rate:
            result['deployment_decision'] = 'monitor'
        else:
            result['deployment_decision'] = 'reject'

        return result

    def get_overall_statistics(self) -> dict[str, Any]:
        """
        Get overall analytics statistics.

        Returns:
            Dictionary with:
            - total_enhancements: Total enhancement events
            - overall_improvement_rate: Fraction that improved
            - avg_effectiveness_delta: Average improvement
            - best_strategy: Strategy with highest improvement rate
            - best_enhancement_type: Enhancement type with highest expected reward
            - total_ab_observations: Total A/B test observations
        """
        total = len(self.metrics)
        improvements = sum(1 for m in self.metrics if m.improved)

        stats = {
            'total_enhancements': total,
            'overall_improvement_rate': improvements / total if total > 0 else 0.0,
            'avg_effectiveness_delta': (
                statistics.mean([m.effectiveness_delta for m in self.metrics])
                if self.metrics else 0.0
            ),
            'best_strategy': None,
            'best_enhancement_type': None,
            'total_ab_observations': (
                len(self.ab_results[ABGroup.CONTROL]) +
                len(self.ab_results[ABGroup.TREATMENT])
            ),
            'strategies_with_data': sum(
                1 for s in self.strategy_impact.values()
                if s.total_enhancements > 0
            ),
            'enhancement_types_with_data': sum(
                1 for e in self.enhancement_impact.values()
                if e.total_uses > 0
            )
        }

        # Best strategy
        best_strategies = self.get_best_strategies(1)
        if best_strategies:
            stats['best_strategy'] = {
                'strategy': best_strategies[0].strategy.value,
                'improvement_rate': best_strategies[0].improvement_rate
            }

        # Best enhancement type
        best_types = self.get_best_enhancement_types(1)
        if best_types:
            stats['best_enhancement_type'] = {
                'enhancement_type': best_types[0].enhancement_type.value,
                'expected_reward': best_types[0].expected_reward
            }

        return stats

    # =========================================================================
    # Export Methods
    # =========================================================================

    def export_prometheus_metrics(self) -> str:
        """
        Export metrics in Prometheus format.

        Returns:
            Prometheus-formatted metrics string
        """
        lines = [
            '# HELP carts_mrf_enhancements_total Total MRF enhancement events',
            '# TYPE carts_mrf_enhancements_total counter',
        ]

        # Total enhancements by strategy
        for strategy, impact in self.strategy_impact.items():
            lines.append(
                f'carts_mrf_enhancements_total{{strategy="{strategy.value}"}} '
                f'{impact.total_enhancements}'
            )

        lines.extend([
            '',
            '# HELP carts_mrf_improvement_rate MRF improvement rate by strategy',
            '# TYPE carts_mrf_improvement_rate gauge',
        ])

        for strategy, impact in self.strategy_impact.items():
            if impact.total_enhancements > 0:
                lines.append(
                    f'carts_mrf_improvement_rate{{strategy="{strategy.value}"}} '
                    f'{impact.improvement_rate:.4f}'
                )

        lines.extend([
            '',
            '# HELP carts_mrf_effectiveness_delta Average effectiveness delta by strategy',
            '# TYPE carts_mrf_effectiveness_delta gauge',
        ])

        for strategy, impact in self.strategy_impact.items():
            if impact.total_enhancements > 0:
                lines.append(
                    f'carts_mrf_effectiveness_delta{{strategy="{strategy.value}"}} '
                    f'{impact.avg_effectiveness_delta:.4f}'
                )

        lines.extend([
            '',
            '# HELP carts_mrf_enhancement_type_uses Enhancement type usage count',
            '# TYPE carts_mrf_enhancement_type_uses counter',
        ])

        for etype, impact in self.enhancement_impact.items():
            lines.append(
                f'carts_mrf_enhancement_type_uses{{type="{etype.value}"}} '
                f'{impact.total_uses}'
            )

        # A/B testing metrics
        if self.enable_ab_testing:
            lines.extend([
                '',
                '# HELP carts_ab_observations_total A/B test observations',
                '# TYPE carts_ab_observations_total counter',
                f'carts_ab_observations_total{{group="control"}} {len(self.ab_results[ABGroup.CONTROL])}',
                f'carts_ab_observations_total{{group="treatment"}} {len(self.ab_results[ABGroup.TREATMENT])}',
            ])

            analysis = self.get_ab_analysis()
            lines.extend([
                '',
                '# HELP carts_ab_success_rate Success rate by group',
                '# TYPE carts_ab_success_rate gauge',
                f'carts_ab_success_rate{{group="control"}} {analysis["control_success_rate"]:.4f}',
                f'carts_ab_success_rate{{group="treatment"}} {analysis["treatment_success_rate"]:.4f}',
            ])

        return '\n'.join(lines)

    def export_json_report(self) -> dict[str, Any]:
        """
        Export comprehensive JSON report.

        Returns:
            Dictionary with complete analytics data
        """
        return {
            'generated_at': datetime.now().isoformat(),
            'overall_statistics': self.get_overall_statistics(),
            'strategy_impact': {
                strategy.value: impact.to_dict()
                for strategy, impact in self.strategy_impact.items()
                if impact.total_enhancements > 0
            },
            'enhancement_type_impact': {
                etype.value: impact.to_dict()
                for etype, impact in self.enhancement_impact.items()
                if impact.total_uses > 0
            },
            'ab_analysis': self.get_ab_analysis() if self.enable_ab_testing else None,
            'recent_metrics': [
                m.to_dict() for m in self.metrics[-50:]  # Last 50
            ]
        }

    def generate_dashboard_html(self) -> str:
        """
        Generate Tufte-style HTML dashboard.

        Returns:
            HTML string with analytics dashboard
        """
        stats = self.get_overall_statistics()
        best_strategies = self.get_best_strategies(5)
        best_types = self.get_best_enhancement_types(5)
        ab_analysis = self.get_ab_analysis() if self.enable_ab_testing else None

        html_parts = [
            '<!DOCTYPE html>',
            '<html lang="en">',
            '<head>',
            '<meta charset="UTF-8">',
            '<title>CARTS MRF Impact Analytics</title>',
            '<style>',
            'body { font-family: "Palatino Linotype", serif; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #333; }',
            'h1 { font-size: 1.8em; border-bottom: 1px solid #333; padding-bottom: 10px; }',
            'h2 { font-size: 1.3em; margin-top: 30px; color: #555; }',
            '.metric { display: inline-block; margin-right: 30px; margin-bottom: 15px; }',
            '.metric-value { font-size: 2em; font-weight: bold; color: #1a1a1a; }',
            '.metric-label { font-size: 0.85em; color: #666; }',
            'table { border-collapse: collapse; width: 100%; margin-top: 15px; font-size: 0.9em; }',
            'th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid #ddd; }',
            'th { background: #f5f5f5; font-weight: normal; color: #555; }',
            '.good { color: #2e7d32; }',
            '.bad { color: #c62828; }',
            '.neutral { color: #666; }',
            '.sparkline { width: 60px; height: 20px; display: inline-block; }',
            '</style>',
            '</head>',
            '<body>',
            '<h1>CARTS MRF Impact Analytics</h1>',
            f'<p style="color: #666; font-size: 0.9em;">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>',
            '',
            '<h2>Overall Performance</h2>',
            '<div class="metrics-row">',
            f'<div class="metric"><div class="metric-value">{stats["total_enhancements"]}</div><div class="metric-label">Total Enhancements</div></div>',
            f'<div class="metric"><div class="metric-value">{stats["overall_improvement_rate"]:.1%}</div><div class="metric-label">Improvement Rate</div></div>',
            f'<div class="metric"><div class="metric-value">{stats["avg_effectiveness_delta"]:+.3f}</div><div class="metric-label">Avg Effectiveness Delta</div></div>',
            f'<div class="metric"><div class="metric-value">{stats["strategies_with_data"]}/{len(AttackStrategy)}</div><div class="metric-label">Strategies with Data</div></div>',
            '</div>',
        ]

        # Best strategies table
        if best_strategies:
            html_parts.extend([
                '<h2>Top Strategies by Improvement Rate</h2>',
                '<table>',
                '<tr><th>Strategy</th><th>Enhancements</th><th>Improvement Rate</th><th>Success Lift</th><th>Avg Delta</th></tr>',
            ])

            for impact in best_strategies:
                rate_class = 'good' if impact.improvement_rate > 0.5 else ('neutral' if impact.improvement_rate > 0.3 else 'bad')
                html_parts.append(
                    f'<tr>'
                    f'<td>{impact.strategy.value}</td>'
                    f'<td>{impact.total_enhancements}</td>'
                    f'<td class="{rate_class}">{impact.improvement_rate:.1%}</td>'
                    f'<td>{impact.success_rate_lift:+.1%}</td>'
                    f'<td>{impact.avg_effectiveness_delta:+.3f}</td>'
                    f'</tr>'
                )

            html_parts.append('</table>')

        # Best enhancement types table
        if best_types:
            html_parts.extend([
                '<h2>Top Enhancement Types by Expected Reward</h2>',
                '<table>',
                '<tr><th>Enhancement Type</th><th>Uses</th><th>Improvement Rate</th><th>Expected Reward</th></tr>',
            ])

            for impact in best_types:
                reward_class = 'good' if impact.expected_reward > 0.5 else ('neutral' if impact.expected_reward > 0.3 else 'bad')
                html_parts.append(
                    f'<tr>'
                    f'<td>{impact.enhancement_type.value}</td>'
                    f'<td>{impact.total_uses}</td>'
                    f'<td>{impact.improvement_rate:.1%}</td>'
                    f'<td class="{reward_class}">{impact.expected_reward:.3f}</td>'
                    f'</tr>'
                )

            html_parts.append('</table>')

        # A/B test analysis
        if ab_analysis and ab_analysis['control_count'] > 0:
            html_parts.extend([
                '<h2>A/B Test Analysis</h2>',
                '<table>',
                f'<tr><td>Control Observations</td><td>{ab_analysis["control_count"]}</td></tr>',
                f'<tr><td>Treatment Observations</td><td>{ab_analysis["treatment_count"]}</td></tr>',
                f'<tr><td>Control Success Rate</td><td>{ab_analysis["control_success_rate"]:.1%}</td></tr>',
                f'<tr><td>Treatment Success Rate</td><td>{ab_analysis["treatment_success_rate"]:.1%}</td></tr>',
                f'<tr><td>Lift</td><td class="{"good" if ab_analysis["lift"] > 0 else "bad"}">{ab_analysis["lift"]:+.1%}</td></tr>',
                f'<tr><td>p-value</td><td>{ab_analysis["p_value"]:.3f}</td></tr>',
                f'<tr><td>Cohen\'s d</td><td>{ab_analysis["cohens_d"]:.3f}</td></tr>',
                f'<tr><td>Statistically Significant</td><td>{"Yes ✓" if ab_analysis["is_significant"] else "No"}</td></tr>',
                f'<tr><td><strong>Deployment Decision</strong></td><td><strong>{ab_analysis["deployment_decision"].upper()}</strong></td></tr>',
                '</table>',
            ])

        html_parts.extend([
            '</body>',
            '</html>'
        ])

        return '\n'.join(html_parts)

    def save_dashboard(self, path: Path | None = None) -> Path:
        """
        Save HTML dashboard to file.

        Args:
            path: Output path (default: persist_path/dashboard.html)

        Returns:
            Path to saved file
        """
        path = path or self.persist_path / "dashboard.html"
        html = self.generate_dashboard_html()
        path.write_text(html)
        logger.info(f"Dashboard saved to {path}")
        return path

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _update_aggregates(self, metric: EnhancementMetric):
        """Update aggregated statistics with new metric."""
        # Update strategy impact
        impact = self.strategy_impact[metric.strategy]
        impact.total_enhancements += 1

        if metric.improved:
            impact.improvements += 1
        elif metric.effectiveness_delta < 0:
            impact.degradations += 1
        else:
            impact.neutral += 1

        # Running averages
        n = impact.total_enhancements
        impact.avg_effectiveness_delta = (
            (impact.avg_effectiveness_delta * (n - 1) + metric.effectiveness_delta) / n
        )
        impact.avg_enhancement_confidence = (
            (impact.avg_enhancement_confidence * (n - 1) + metric.enhancement_confidence) / n
        )

        # Success rates
        successes_before = sum(1 for m in self.metrics if m.strategy == metric.strategy and m.success_before)
        successes_after = sum(1 for m in self.metrics if m.strategy == metric.strategy and m.success_after)
        impact.success_rate_before = successes_before / n
        impact.success_rate_after = successes_after / n

        # Enhancement type usage
        etype_key = metric.enhancement_type.value
        impact.enhancement_type_usage[etype_key] = (
            impact.enhancement_type_usage.get(etype_key, 0) + 1
        )

        # Update enhancement type impact
        etype_impact = self.enhancement_impact[metric.enhancement_type]
        etype_impact.total_uses += 1

        if metric.improved:
            etype_impact.improvements += 1
        elif metric.effectiveness_delta < 0:
            etype_impact.degradations += 1

        m = etype_impact.total_uses
        etype_impact.avg_effectiveness_delta = (
            (etype_impact.avg_effectiveness_delta * (m - 1) + metric.effectiveness_delta) / m
        )

        # Strategy success for this enhancement type
        strategy_key = metric.strategy.value
        current_success = etype_impact.strategy_success.get(strategy_key, 0.0)
        count_for_strategy = sum(
            1 for met in self.metrics
            if met.enhancement_type == metric.enhancement_type and met.strategy == metric.strategy
        )
        if count_for_strategy > 0:
            successes = sum(
                1 for met in self.metrics
                if met.enhancement_type == metric.enhancement_type
                and met.strategy == metric.strategy
                and met.improved
            )
            etype_impact.strategy_success[strategy_key] = successes / count_for_strategy

    def _assign_ab_group(self, user_id: str) -> ABGroup:
        """Deterministically assign user to A/B group."""
        if user_id in self.ab_assignments:
            return self.ab_assignments[user_id]

        # Deterministic hash-based assignment
        hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        if (hash_val % 100) / 100 < self.ab_traffic_split:
            group = ABGroup.TREATMENT
        else:
            group = ABGroup.CONTROL

        self.ab_assignments[user_id] = group
        return group

    def _save_data(self):
        """Persist analytics data to disk."""
        data = {
            'metrics': [m.to_dict() for m in self.metrics],
            'ab_results': {
                'control': [r.to_dict() for r in self.ab_results[ABGroup.CONTROL]],
                'treatment': [r.to_dict() for r in self.ab_results[ABGroup.TREATMENT]]
            },
            'ab_assignments': {k: v.value for k, v in self.ab_assignments.items()}
        }

        path = self.persist_path / "analytics_data.json"
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Analytics data saved to {path}")

    def _load_data(self):
        """Load persisted analytics data."""
        path = self.persist_path / "analytics_data.json"
        if not path.exists():
            return

        try:
            with open(path) as f:
                data = json.load(f)

            # Restore metrics
            for m_dict in data.get('metrics', []):
                metric = EnhancementMetric(
                    timestamp=m_dict['timestamp'],
                    strategy=AttackStrategy(m_dict['strategy']),
                    enhancement_type=EnhancementType(m_dict['enhancement_type']),
                    success_before=m_dict['success_before'],
                    success_after=m_dict['success_after'],
                    severity_before=m_dict['severity_before'],
                    severity_after=m_dict['severity_after'],
                    enhancement_confidence=m_dict['enhancement_confidence'],
                    payload_id=m_dict['payload_id'],
                    metadata=m_dict.get('metadata', {})
                )
                self.metrics.append(metric)
                self._update_aggregates(metric)

            # Restore A/B results
            for r_dict in data.get('ab_results', {}).get('control', []):
                self.ab_results[ABGroup.CONTROL].append(ABTestResult(
                    timestamp=r_dict['timestamp'],
                    group=ABGroup.CONTROL,
                    strategy=AttackStrategy(r_dict['strategy']),
                    success=r_dict['success'],
                    severity=r_dict['severity'],
                    user_id=r_dict['user_id']
                ))

            for r_dict in data.get('ab_results', {}).get('treatment', []):
                self.ab_results[ABGroup.TREATMENT].append(ABTestResult(
                    timestamp=r_dict['timestamp'],
                    group=ABGroup.TREATMENT,
                    strategy=AttackStrategy(r_dict['strategy']),
                    success=r_dict['success'],
                    severity=r_dict['severity'],
                    user_id=r_dict['user_id']
                ))

            # Restore assignments
            for user_id, group_value in data.get('ab_assignments', {}).items():
                self.ab_assignments[user_id] = ABGroup(group_value)

            logger.info(f"Loaded {len(self.metrics)} metrics from {path}")

        except Exception as e:
            logger.warning(f"Error loading analytics data: {e}")


# =============================================================================
# Convenience Functions
# =============================================================================

def create_analytics(
    persist_path: Path | None = None,
    enable_ab_testing: bool = True
) -> MRFImpactAnalytics:
    """
    Create MRF impact analytics instance.

    Args:
        persist_path: Path to persist data
        enable_ab_testing: Enable A/B testing framework

    Returns:
        MRFImpactAnalytics instance
    """
    return MRFImpactAnalytics(
        persist_path=persist_path,
        enable_ab_testing=enable_ab_testing
    )


def log_enhancement_event(
    analytics: MRFImpactAnalytics,
    strategy: AttackStrategy,
    enhancement_type: EnhancementType,
    success_before: bool,
    success_after: bool,
    severity_before: float,
    severity_after: float,
    enhancement_confidence: float,
    payload: str
) -> EnhancementMetric:
    """
    Log an enhancement event with auto-generated payload ID.

    Args:
        analytics: Analytics instance
        strategy: Attack strategy
        enhancement_type: Enhancement type applied
        success_before: Success before enhancement
        success_after: Success after enhancement
        severity_before: Severity before
        severity_after: Severity after
        enhancement_confidence: MRF confidence
        payload: Payload string (will be hashed)

    Returns:
        EnhancementMetric
    """
    payload_id = hashlib.md5(payload.encode()).hexdigest()[:12]

    return analytics.log_enhancement(
        strategy=strategy,
        enhancement_type=enhancement_type,
        success_before=success_before,
        success_after=success_after,
        severity_before=severity_before,
        severity_after=severity_after,
        enhancement_confidence=enhancement_confidence,
        payload_id=payload_id
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Data Classes
    'EnhancementMetric',
    'StrategyImpact',
    'EnhancementTypeImpact',
    'ABGroup',
    'ABTestResult',

    # Main Class
    'MRFImpactAnalytics',

    # Convenience Functions
    'create_analytics',
    'log_enhancement_event',
]
