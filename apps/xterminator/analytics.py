"""
Analytics and Reporting
========================

🐷 Phase 5: Advanced Analytics and Reporting 🐷

Provides comprehensive analytics on xTerminator performance, learning,
and quality trends across customers, departments, and strategies.

Analytics Features:
- Performance dashboards
- Learning trend analysis
- Customer-specific reports
- Department quality scores
- Strategy effectiveness comparison
- Time-series analysis

This enables:
- Data-driven decisions
- Customer insights
- System optimization
- Transparency and trust
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import time
from collections import defaultdict


class ReportType(Enum):
    """Type of analytics report"""
    PERFORMANCE = "performance"      # Overall system performance
    LEARNING = "learning"            # Learning trends
    CUSTOMER = "customer"            # Per-customer analytics
    DEPARTMENT = "department"        # Per-department analytics
    STRATEGY = "strategy"            # Strategy comparison
    QUALITY = "quality"              # Quality trends


@dataclass
class PerformanceMetrics:
    """
    Performance metrics for a time period.
    """
    period_start: float
    period_end: float

    # Fix attempt metrics
    total_attempts: int = 0
    successful_fixes: int = 0
    failed_fixes: int = 0
    skipped_fixes: int = 0

    # Performance
    success_rate: float = 0.0
    avg_confidence: float = 0.0
    avg_duration_ms: float = 0.0

    # Quality
    avg_quality_score: float = 0.0
    ece: float = 0.0  # Expected Calibration Error

    # Strategy distribution
    strategy_distribution: Dict[str, int] = field(default_factory=dict)

    # Risk distribution
    risk_distribution: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'period_start': self.period_start,
            'period_end': self.period_end,
            'total_attempts': self.total_attempts,
            'successful_fixes': self.successful_fixes,
            'failed_fixes': self.failed_fixes,
            'skipped_fixes': self.skipped_fixes,
            'success_rate': self.success_rate,
            'avg_confidence': self.avg_confidence,
            'avg_duration_ms': self.avg_duration_ms,
            'avg_quality_score': self.avg_quality_score,
            'ece': self.ece,
            'strategy_distribution': self.strategy_distribution,
            'risk_distribution': self.risk_distribution
        }


@dataclass
class LearningTrend:
    """
    Learning trend analysis.

    Tracks how the system improves over time.
    """
    start_period: float
    end_period: float

    # Success rate progression
    initial_success_rate: float
    final_success_rate: float
    improvement: float  # Percentage point improvement

    # Thompson Sampling convergence
    converged: bool
    convergence_attempt: Optional[int] = None
    best_strategy: Optional[str] = None

    # Confidence calibration improvement
    initial_ece: float = 0.0
    final_ece: float = 0.0
    ece_improvement: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'start_period': self.start_period,
            'end_period': self.end_period,
            'initial_success_rate': self.initial_success_rate,
            'final_success_rate': self.final_success_rate,
            'improvement': self.improvement,
            'converged': self.converged,
            'convergence_attempt': self.convergence_attempt,
            'best_strategy': self.best_strategy,
            'initial_ece': self.initial_ece,
            'final_ece': self.final_ece,
            'ece_improvement': self.ece_improvement
        }


class AnalyticsEngine:
    """
    Analytics engine for xTerminator.

    Aggregates data from feedback tracker, Thompson bandit,
    confidence calibrator, and generates reports.
    """

    def __init__(self):
        """Initialize analytics engine"""
        self.performance_cache: Dict[str, PerformanceMetrics] = {}

    def generate_performance_report(
        self,
        feedback_data: List[Dict],
        period_start: Optional[float] = None,
        period_end: Optional[float] = None
    ) -> PerformanceMetrics:
        """
        Generate performance report from feedback data.

        Args:
            feedback_data: List of fix attempts from FeedbackTracker
            period_start: Start of reporting period (optional)
            period_end: End of reporting period (optional)

        Returns:
            PerformanceMetrics
        """
        # Filter by period if specified
        if period_start or period_end:
            feedback_data = [
                attempt for attempt in feedback_data
                if (not period_start or attempt.get('timestamp', 0) >= period_start) and
                   (not period_end or attempt.get('timestamp', float('inf')) <= period_end)
            ]

        if not feedback_data:
            return PerformanceMetrics(
                period_start=period_start or 0.0,
                period_end=period_end or time.time()
            )

        # Aggregate metrics
        total = len(feedback_data)
        successful = sum(1 for a in feedback_data if a.get('outcome') == 'success')
        failed = sum(1 for a in feedback_data if a.get('outcome') in ['failure', 'rollback'])
        skipped = sum(1 for a in feedback_data if a.get('outcome') == 'skipped')

        success_rate = successful / total if total > 0 else 0.0
        avg_confidence = sum(a.get('confidence', 0) for a in feedback_data) / total if total > 0 else 0.0
        avg_duration = sum(a.get('total_duration_ms', 0) for a in feedback_data) / total if total > 0 else 0.0

        # Strategy distribution
        strategy_dist = defaultdict(int)
        for attempt in feedback_data:
            strategy = attempt.get('fix_strategy', 'unknown')
            strategy_dist[strategy] += 1

        # Risk distribution
        risk_dist = defaultdict(int)
        for attempt in feedback_data:
            risk = attempt.get('risk_level', 'unknown')
            risk_dist[risk] += 1

        return PerformanceMetrics(
            period_start=period_start or min(a.get('timestamp', 0) for a in feedback_data),
            period_end=period_end or max(a.get('timestamp', 0) for a in feedback_data),
            total_attempts=total,
            successful_fixes=successful,
            failed_fixes=failed,
            skipped_fixes=skipped,
            success_rate=success_rate,
            avg_confidence=avg_confidence,
            avg_duration_ms=avg_duration,
            strategy_distribution=dict(strategy_dist),
            risk_distribution=dict(risk_dist)
        )

    def generate_learning_trend_report(
        self,
        learning_curve: List[Dict],
        calibration_summary: Dict
    ) -> LearningTrend:
        """
        Generate learning trend report.

        Args:
            learning_curve: Learning curve from Thompson bandit
            calibration_summary: Calibration summary from ConfidenceCalibrator

        Returns:
            LearningTrend
        """
        if not learning_curve or len(learning_curve) < 2:
            return LearningTrend(
                start_period=0.0,
                end_period=time.time(),
                initial_success_rate=0.0,
                final_success_rate=0.0,
                improvement=0.0,
                converged=False
            )

        # Get initial and final success rates
        initial_rate = learning_curve[0]['rolling_success_rate']
        final_rate = learning_curve[-1]['rolling_success_rate']
        improvement = final_rate - initial_rate

        # Detect convergence (simple heuristic)
        # If last 20% of curve has variance < 0.01, consider converged
        tail_size = max(int(len(learning_curve) * 0.2), 10)
        tail = learning_curve[-tail_size:]
        tail_rates = [p['rolling_success_rate'] for p in tail]
        variance = sum((r - final_rate) ** 2 for r in tail_rates) / len(tail_rates)
        converged = variance < 0.01

        # Find convergence point
        convergence_attempt = None
        if converged:
            convergence_attempt = len(learning_curve) - tail_size

        # Best strategy (most selected in final 20%)
        strategy_counts = defaultdict(int)
        for point in tail:
            strategy_counts[point['strategy']] += 1
        best_strategy = max(strategy_counts.keys(), key=lambda s: strategy_counts[s]) if strategy_counts else None

        return LearningTrend(
            start_period=0.0,
            end_period=time.time(),
            initial_success_rate=initial_rate,
            final_success_rate=final_rate,
            improvement=improvement,
            converged=converged,
            convergence_attempt=convergence_attempt,
            best_strategy=best_strategy,
            initial_ece=calibration_summary.get('ece', 0.0),
            final_ece=calibration_summary.get('ece', 0.0),
            ece_improvement=0.0  # Would need historical ECE data
        )

    def generate_strategy_comparison(
        self,
        thompson_performance: Dict
    ) -> Dict:
        """
        Generate strategy comparison report.

        Args:
            thompson_performance: Performance data from Thompson bandit

        Returns:
            Dictionary with strategy comparison
        """
        per_strategy = thompson_performance.get('per_strategy', {})

        comparison = {
            'strategies': [],
            'best_strategy': None,
            'worst_strategy': None,
            'total_selections': thompson_performance.get('total_selections', 0)
        }

        max_reward = 0.0
        min_reward = 1.0

        for strategy, stats in per_strategy.items():
            strategy_data = {
                'strategy': strategy,
                'total_pulls': stats['total_pulls'],
                'success_rate': stats['success_rate'],
                'expected_reward': stats['expected_reward'],
                'avg_confidence': stats['avg_confidence'],
                'selection_frequency': stats['total_pulls'] / comparison['total_selections'] if comparison['total_selections'] > 0 else 0.0
            }

            comparison['strategies'].append(strategy_data)

            if stats['expected_reward'] > max_reward:
                max_reward = stats['expected_reward']
                comparison['best_strategy'] = strategy

            if stats['expected_reward'] < min_reward:
                min_reward = stats['expected_reward']
                comparison['worst_strategy'] = strategy

        # Sort by expected reward
        comparison['strategies'].sort(key=lambda s: s['expected_reward'], reverse=True)

        return comparison

    def generate_customer_report(
        self,
        customer_id: str,
        feedback_data: List[Dict],
        customer_policy: Optional[Dict] = None
    ) -> Dict:
        """
        Generate customer-specific report.

        Args:
            customer_id: Customer ID
            feedback_data: Feedback data filtered for this customer
            customer_policy: Customer's policy configuration

        Returns:
            Customer report dictionary
        """
        # Generate performance metrics for customer
        metrics = self.generate_performance_report(feedback_data)

        report = {
            'customer_id': customer_id,
            'reporting_period': {
                'start': metrics.period_start,
                'end': metrics.period_end
            },
            'performance': metrics.to_dict(),
            'policy': customer_policy,
            'recommendations': []
        }

        # Add recommendations based on performance
        if metrics.success_rate < 0.75:
            report['recommendations'].append("Success rate below 75% - consider stricter quality gates")

        if metrics.ece > 0.10:
            report['recommendations'].append("ECE above 0.10 - confidence calibration needs improvement")

        return report

    def generate_time_series_data(
        self,
        feedback_data: List[Dict],
        bucket_size_hours: int = 24
    ) -> List[Dict]:
        """
        Generate time-series data for visualization.

        Args:
            feedback_data: Feedback data
            bucket_size_hours: Size of each time bucket in hours

        Returns:
            List of time buckets with metrics
        """
        if not feedback_data:
            return []

        # Sort by timestamp
        sorted_data = sorted(feedback_data, key=lambda a: a.get('timestamp', 0))

        # Create time buckets
        start_time = sorted_data[0].get('timestamp', 0)
        end_time = sorted_data[-1].get('timestamp', time.time())
        bucket_size_seconds = bucket_size_hours * 3600

        buckets = []
        current_bucket_start = start_time

        while current_bucket_start < end_time:
            bucket_end = current_bucket_start + bucket_size_seconds

            # Get attempts in this bucket
            bucket_attempts = [
                a for a in sorted_data
                if current_bucket_start <= a.get('timestamp', 0) < bucket_end
            ]

            if bucket_attempts:
                # Calculate metrics for bucket
                total = len(bucket_attempts)
                successful = sum(1 for a in bucket_attempts if a.get('outcome') == 'success')
                success_rate = successful / total if total > 0 else 0.0
                avg_conf = sum(a.get('confidence', 0) for a in bucket_attempts) / total if total > 0 else 0.0

                buckets.append({
                    'timestamp': current_bucket_start,
                    'bucket_start': current_bucket_start,
                    'bucket_end': bucket_end,
                    'total_attempts': total,
                    'success_rate': success_rate,
                    'avg_confidence': avg_conf
                })

            current_bucket_start = bucket_end

        return buckets


def create_analytics_engine() -> AnalyticsEngine:
    """
    Factory function to create analytics engine.

    Returns:
        AnalyticsEngine instance
    """
    return AnalyticsEngine()
