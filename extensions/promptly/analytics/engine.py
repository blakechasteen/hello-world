"""
Promptly Analytics Engine

Core analytics engine for prompt performance analysis, quality monitoring,
trend detection, and optimization recommendations.

Features:
- Comprehensive prompt analytics with quality trends
- Statistical comparison between prompts (A/B testing support)
- Underperforming prompt identification
- Quality anomaly detection
- Thompson Sampling integration for recommendations
- Export capabilities (JSON, CSV)

Usage:
    from promptly.analytics import PromptAnalyticsEngine
    from promptly.core.database import PromptlyDatabase

    db = PromptlyDatabase()
    engine = PromptAnalyticsEngine(db)

    # Get analytics for a prompt
    analytics = engine.get_prompt_analytics("summarize", days=30)
    print(f"Avg quality: {analytics.avg_quality:.2f}")
    print(f"Trend: {analytics.quality_trend.value}")
    print(analytics.generate_recommendation())

    # Compare two prompts
    comparison = engine.compare_prompts("prompt_a", "prompt_b")
    print(f"Winner: {comparison.winner}")
    print(f"Significant: {comparison.is_significant}")

    # Find underperforming prompts
    underperforming = engine.identify_underperforming(threshold=0.6)
    for prompt in underperforming:
        print(f"{prompt.prompt_name}: {prompt.avg_quality:.2f}")
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json
import math
import random

from promptly.core.analytics import (
    QualityTrend,
    Severity,
    PromptExecutionSummary,
    VersionPerformance,
    TaskTypeStats,
    DailyBreakdown,
    PromptAnalytics,
    ComparisonStatistics,
    ComparisonResult,
    QualityAnomaly,
    UnderperformingPrompt,
    ThompsonRecommendation,
    AnalyticsExport,
    # MRF Analytics
    MRFStrategy,
    MRFRefinementRecord,
    MRFStrategyStats,
    MRFComponentUsage,
    MRFModelProviderStats,
    MRFRefinementAnalytics,
    MRFThompsonState,
    MRFAnalyticsExport,
)


class PromptAnalyticsEngine:
    """
    Core analytics engine for prompt performance analysis.

    Provides comprehensive analytics including:
    - Quality metrics and trends
    - Version performance comparison
    - Task type distribution analysis
    - Thompson Sampling recommendations
    - Anomaly detection
    - Statistical comparisons
    """

    def __init__(self, db, quality_threshold: float = 0.7):
        """
        Initialize analytics engine.

        Args:
            db: PromptlyDatabase instance
            quality_threshold: Threshold for success (quality >= threshold)
        """
        self.db = db
        self.quality_threshold = quality_threshold

    def get_prompt_analytics(
        self,
        prompt_name: str,
        days: int = 30,
        include_thompson: bool = True
    ) -> Optional[PromptAnalytics]:
        """
        Get comprehensive analytics for a prompt.

        Args:
            prompt_name: Name of the prompt to analyze
            days: Number of days to analyze
            include_thompson: Include Thompson Sampling expected quality

        Returns:
            PromptAnalytics object or None if prompt not found
        """
        # Get executions for the prompt
        executions = self.db.get_executions(
            prompt_name=prompt_name,
            days=days,
            limit=10000  # High limit for full analysis
        )

        if not executions:
            return None

        # Basic metrics
        total_executions = len(executions)
        quality_scores = [e.quality_score for e in executions if e.quality_score is not None]
        latencies = [e.latency_ms for e in executions if e.latency_ms is not None]
        tokens = [e.token_count for e in executions if e.token_count is not None]

        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        avg_tokens = int(sum(tokens) / len(tokens)) if tokens else None

        # Success rate
        successes = sum(1 for q in quality_scores if q >= self.quality_threshold)
        success_rate = successes / len(quality_scores) if quality_scores else 0.0

        # Quality range
        quality_range = {
            "min": min(quality_scores) if quality_scores else 0.0,
            "max": max(quality_scores) if quality_scores else 0.0,
            "std": self._std_dev(quality_scores) if len(quality_scores) > 1 else 0.0
        }

        # Quality trend
        quality_trend = self._calculate_quality_trend(quality_scores)

        # Task type distribution
        task_type_distribution = self._calculate_task_distribution(executions)

        # Version performance
        version_performance = self._calculate_version_performance(executions)

        # Daily breakdown
        daily_breakdown = self._calculate_daily_breakdown(executions, days)

        # Thompson Sampling expected quality
        thompson_expected = None
        if include_thompson:
            thompson_expected = self._get_thompson_expected_quality(prompt_name)

        analytics = PromptAnalytics(
            prompt_name=prompt_name,
            total_executions=total_executions,
            avg_quality=avg_quality,
            avg_latency_ms=avg_latency,
            success_rate=success_rate,
            quality_trend=quality_trend,
            quality_range=quality_range,
            avg_tokens=avg_tokens,
            task_type_distribution=task_type_distribution,
            version_performance=version_performance,
            thompson_expected_quality=thompson_expected,
            daily_breakdown=daily_breakdown,
            days_analyzed=days
        )

        # Generate recommendation
        analytics.generate_recommendation()

        return analytics

    def get_quality_trend(
        self,
        prompt_name: str,
        days: int = 30,
        granularity: str = "daily"
    ) -> List[Dict[str, Any]]:
        """
        Get quality trend over time.

        Args:
            prompt_name: Name of the prompt
            days: Number of days to analyze
            granularity: "daily" or "hourly"

        Returns:
            List of trend data points with timestamps
        """
        executions = self.db.get_executions(
            prompt_name=prompt_name,
            days=days,
            limit=10000
        )

        if not executions:
            return []

        # Group by time period
        if granularity == "hourly":
            trend_data = self._group_by_hour(executions)
        else:
            trend_data = self._group_by_day(executions)

        return trend_data

    def compare_prompts(
        self,
        prompt_a: str,
        prompt_b: str,
        task_type: Optional[str] = None,
        days: int = 30,
        significance_threshold: float = 0.05
    ) -> Optional[ComparisonResult]:
        """
        Compare two prompts statistically.

        Args:
            prompt_a: First prompt name
            prompt_b: Second prompt name
            task_type: Optional task type filter
            days: Number of days to analyze
            significance_threshold: P-value threshold for significance

        Returns:
            ComparisonResult or None if insufficient data
        """
        # Get analytics for both prompts
        analytics_a = self.get_prompt_analytics(prompt_a, days=days)
        analytics_b = self.get_prompt_analytics(prompt_b, days=days)

        if not analytics_a or not analytics_b:
            return None

        # Get raw scores for statistical tests
        executions_a = self.db.get_executions(prompt_name=prompt_a, task_type=task_type, days=days)
        executions_b = self.db.get_executions(prompt_name=prompt_b, task_type=task_type, days=days)

        scores_a = [e.quality_score for e in executions_a if e.quality_score is not None]
        scores_b = [e.quality_score for e in executions_b if e.quality_score is not None]

        if len(scores_a) < 5 or len(scores_b) < 5:
            return None  # Insufficient data

        # Calculate statistics
        quality_diff = analytics_a.avg_quality - analytics_b.avg_quality
        latency_diff = analytics_a.avg_latency_ms - analytics_b.avg_latency_ms
        success_rate_diff = analytics_a.success_rate - analytics_b.success_rate

        # Statistical significance (simplified t-test approximation)
        significance, cohens_d = self._calculate_significance(scores_a, scores_b)

        is_significant = significance < significance_threshold

        statistics = ComparisonStatistics(
            quality_difference=quality_diff,
            latency_difference_ms=latency_diff,
            success_rate_difference=success_rate_diff,
            sample_size_a=len(scores_a),
            sample_size_b=len(scores_b),
            statistical_significance=significance,
            cohens_d=cohens_d
        )

        # Determine winner
        winner = None
        if is_significant:
            if quality_diff > 0:
                winner = prompt_a
            elif quality_diff < 0:
                winner = prompt_b

        # Generate recommendation
        recommendation = self._generate_comparison_recommendation(
            analytics_a, analytics_b, statistics, is_significant
        )

        # Deployment decision
        deployment_decision = self._make_deployment_decision(
            winner, is_significant, statistics
        )

        return ComparisonResult(
            prompt_a=prompt_a,
            prompt_b=prompt_b,
            prompt_a_stats=analytics_a,
            prompt_b_stats=analytics_b,
            winner=winner,
            statistics=statistics,
            is_significant=is_significant,
            recommendation=recommendation,
            deployment_decision=deployment_decision
        )

    def identify_underperforming(
        self,
        threshold: float = 0.6,
        min_executions: int = 10,
        days: int = 30
    ) -> List[UnderperformingPrompt]:
        """
        Identify underperforming prompts.

        Args:
            threshold: Quality threshold (prompts below this are flagged)
            min_executions: Minimum executions to consider
            days: Number of days to analyze

        Returns:
            List of underperforming prompts with details
        """
        underperforming = []

        # Get all prompts
        prompts = self.db.list_prompts()

        for prompt in prompts:
            analytics = self.get_prompt_analytics(prompt.name, days=days)
            if not analytics or analytics.total_executions < min_executions:
                continue

            if analytics.avg_quality < threshold or analytics.success_rate < 0.5:
                # Identify specific issues
                issues = []
                suggested_actions = []

                if analytics.avg_quality < threshold:
                    issues.append(f"Low quality score ({analytics.avg_quality:.2f})")
                    suggested_actions.append("Review and revise prompt content")

                if analytics.success_rate < 0.5:
                    issues.append(f"Low success rate ({analytics.success_rate:.1%})")
                    suggested_actions.append("Analyze failure cases")

                if analytics.quality_trend == QualityTrend.DECLINING:
                    issues.append("Quality declining over time")
                    suggested_actions.append("Investigate recent changes")

                if analytics.avg_latency_ms and analytics.avg_latency_ms > 5000:
                    issues.append(f"High latency ({analytics.avg_latency_ms:.0f}ms)")
                    suggested_actions.append("Consider prompt simplification")

                # Determine severity
                severity = self._determine_severity(analytics, threshold)

                underperforming.append(UnderperformingPrompt(
                    prompt_name=prompt.name,
                    avg_quality=analytics.avg_quality,
                    success_rate=analytics.success_rate,
                    total_executions=analytics.total_executions,
                    quality_trend=analytics.quality_trend,
                    severity=severity,
                    issues=issues,
                    suggested_actions=suggested_actions
                ))

        # Sort by severity and quality
        severity_order = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3}
        underperforming.sort(key=lambda x: (severity_order[x.severity], x.avg_quality))

        return underperforming

    def detect_quality_anomalies(
        self,
        prompt_name: str,
        days: int = 30,
        sensitivity: float = 2.0
    ) -> List[QualityAnomaly]:
        """
        Detect quality anomalies for a prompt.

        Args:
            prompt_name: Name of the prompt
            days: Number of days to analyze
            sensitivity: Standard deviations for anomaly detection

        Returns:
            List of detected anomalies
        """
        executions = self.db.get_executions(
            prompt_name=prompt_name,
            days=days,
            limit=10000
        )

        if len(executions) < 10:
            return []

        anomalies = []
        quality_scores = [e.quality_score for e in executions if e.quality_score is not None]

        if not quality_scores:
            return []

        mean_quality = sum(quality_scores) / len(quality_scores)
        std_quality = self._std_dev(quality_scores)

        # Detect sudden drops
        for i in range(1, len(executions)):
            if executions[i].quality_score is None or executions[i-1].quality_score is None:
                continue

            drop = executions[i-1].quality_score - executions[i].quality_score
            if drop > sensitivity * std_quality and drop > 0.2:
                anomalies.append(QualityAnomaly(
                    anomaly_type="sudden_drop",
                    severity=Severity.HIGH if drop > 0.3 else Severity.MEDIUM,
                    description=f"Quality dropped by {drop:.2f} ({executions[i-1].quality_score:.2f} -> {executions[i].quality_score:.2f})",
                    timestamp=executions[i].created_at,
                    affected_executions=[executions[i].id],
                    suggested_action="Investigate input or model changes around this time"
                ))

        # Detect prolonged low quality
        low_quality_streak = []
        for exec in executions:
            if exec.quality_score is not None and exec.quality_score < mean_quality - std_quality:
                low_quality_streak.append(exec)
            else:
                if len(low_quality_streak) >= 3:
                    anomalies.append(QualityAnomaly(
                        anomaly_type="prolonged_low",
                        severity=Severity.MEDIUM,
                        description=f"Quality below average for {len(low_quality_streak)} consecutive executions",
                        timestamp=low_quality_streak[0].created_at,
                        affected_executions=[e.id for e in low_quality_streak],
                        suggested_action="Review prompt effectiveness for this period"
                    ))
                low_quality_streak = []

        # Detect high variance periods
        window_size = min(10, len(quality_scores) // 3)
        if window_size >= 3:
            for i in range(0, len(quality_scores) - window_size + 1, window_size):
                window = quality_scores[i:i + window_size]
                window_std = self._std_dev(window)
                if window_std > sensitivity * std_quality:
                    anomalies.append(QualityAnomaly(
                        anomaly_type="high_variance",
                        severity=Severity.LOW,
                        description=f"High quality variance (std: {window_std:.3f}) in window",
                        timestamp=executions[i].created_at,
                        affected_executions=[e.id for e in executions[i:i + window_size]],
                        suggested_action="Investigate inconsistent inputs or model behavior"
                    ))

        # Detect outliers
        for exec in executions:
            if exec.quality_score is None:
                continue
            z_score = (exec.quality_score - mean_quality) / std_quality if std_quality > 0 else 0
            if abs(z_score) > sensitivity:
                anomalies.append(QualityAnomaly(
                    anomaly_type="outlier",
                    severity=Severity.LOW,
                    description=f"Outlier quality score: {exec.quality_score:.2f} (z-score: {z_score:.2f})",
                    timestamp=exec.created_at,
                    affected_executions=[exec.id],
                    suggested_action="Review this specific execution for unusual inputs"
                ))

        return anomalies

    def get_thompson_recommendation(
        self,
        task_type: str,
        candidates: Optional[List[str]] = None
    ) -> Optional[ThompsonRecommendation]:
        """
        Get Thompson Sampling recommendation for a task type.

        Args:
            task_type: Type of task
            candidates: Optional list of candidate prompt names

        Returns:
            ThompsonRecommendation or None
        """
        result = self.db.get_thompson_recommendation(task_type, candidates)

        if not result:
            return None

        # Get alternatives
        alternatives = []
        priors = self.db.get_thompson_priors(task_type=task_type)

        for prior in priors:
            if prior.prompt_name != result["prompt_name"]:
                expected = prior.alpha / (prior.alpha + prior.beta)
                alternatives.append({
                    "name": prior.prompt_name,
                    "expected_quality": expected,
                    "alpha": prior.alpha,
                    "beta": prior.beta
                })

        # Sort alternatives by expected quality
        alternatives.sort(key=lambda x: x["expected_quality"], reverse=True)

        # Calculate confidence
        total_samples = result["alpha"] + result["beta"] - 2
        confidence = min(1.0, total_samples / 100)  # More samples = higher confidence

        # Generate reasoning
        reasoning = self._generate_thompson_reasoning(result, alternatives, confidence)

        return ThompsonRecommendation(
            recommended_prompt=result["prompt_name"],
            expected_quality=result["expected_quality"],
            confidence=confidence,
            task_type=task_type,
            alternatives=alternatives[:5],  # Top 5 alternatives
            reasoning=reasoning
        )

    def export_analytics(
        self,
        prompts: Optional[List[str]] = None,
        days: int = 30,
        include_anomalies: bool = True,
        include_thompson: bool = True
    ) -> AnalyticsExport:
        """
        Export comprehensive analytics.

        Args:
            prompts: Optional list of prompt names (None = all)
            days: Number of days to analyze
            include_anomalies: Include anomaly detection
            include_thompson: Include Thompson recommendations

        Returns:
            AnalyticsExport object with to_json() and to_csv() methods
        """
        if prompts is None:
            all_prompts = self.db.list_prompts()
            prompts = [p.name for p in all_prompts]

        # Collect analytics for all prompts
        prompt_analytics = []
        total_executions = 0

        for prompt_name in prompts:
            analytics = self.get_prompt_analytics(prompt_name, days=days)
            if analytics:
                prompt_analytics.append(analytics)
                total_executions += analytics.total_executions

        # Find underperforming prompts
        underperforming = self.identify_underperforming(days=days)

        # Detect anomalies for each prompt
        all_anomalies = []
        if include_anomalies:
            for prompt_name in prompts:
                anomalies = self.detect_quality_anomalies(prompt_name, days=days)
                all_anomalies.extend(anomalies)

        # Get Thompson recommendations for all task types
        thompson_recommendations = {}
        if include_thompson:
            priors = self.db.get_thompson_priors()
            task_types = set(p.task_type for p in priors)
            for task_type in task_types:
                rec = self.get_thompson_recommendation(task_type)
                if rec:
                    thompson_recommendations[task_type] = rec

        # Date range
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")

        return AnalyticsExport(
            export_timestamp=datetime.now(),
            total_prompts=len(prompt_analytics),
            total_executions=total_executions,
            date_range={"start": start_date, "end": end_date},
            prompts=prompt_analytics,
            underperforming=underperforming,
            anomalies=all_anomalies,
            thompson_recommendations=thompson_recommendations,
            metadata={
                "quality_threshold": self.quality_threshold,
                "days_analyzed": days,
                "generated_by": "PromptAnalyticsEngine"
            }
        )

    # ==================== MRF Analytics Methods ====================

    def record_mrf_refinement(
        self,
        prompt_name: str,
        strategy: MRFStrategy,
        quality_before: float,
        quality_after: float,
        latency_ms: float,
        model_provider: str,
        components_applied: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Record an MRF refinement execution.

        Args:
            prompt_name: Name of the refined prompt
            strategy: MRF strategy used
            quality_before: Quality score before refinement
            quality_after: Quality score after refinement
            latency_ms: Refinement latency in milliseconds
            model_provider: Model provider used (claude, gemini, gpt, ollama)
            components_applied: Which of the 7 components were applied
            metadata: Additional metadata

        Returns:
            ID of the recorded refinement
        """
        return self.db.add_mrf_refinement(
            prompt_name=prompt_name,
            strategy=strategy.value,
            quality_before=quality_before,
            quality_after=quality_after,
            latency_ms=latency_ms,
            model_provider=model_provider,
            components_applied=components_applied,
            metadata=metadata or {}
        )

    def get_mrf_analytics(
        self,
        days: int = 30,
        prompt_name: Optional[str] = None
    ) -> Optional[MRFRefinementAnalytics]:
        """
        Get comprehensive MRF refinement analytics.

        Args:
            days: Number of days to analyze
            prompt_name: Optional prompt name filter

        Returns:
            MRFRefinementAnalytics or None if no data
        """
        # Get all MRF refinements
        refinements = self.db.get_mrf_refinements(
            prompt_name=prompt_name,
            days=days,
            limit=10000
        )

        if not refinements:
            return None

        # Basic metrics
        total_refinements = len(refinements)
        quality_before_scores = [r.quality_before for r in refinements]
        quality_after_scores = [r.quality_after for r in refinements]
        improvements = [r.quality_after - r.quality_before for r in refinements]
        latencies = [r.latency_ms for r in refinements]

        avg_quality_before = sum(quality_before_scores) / len(quality_before_scores)
        avg_quality_after = sum(quality_after_scores) / len(quality_after_scores)
        overall_improvement = avg_quality_after - avg_quality_before
        overall_improvement_percent = (overall_improvement / avg_quality_before * 100) if avg_quality_before > 0 else 0.0
        avg_latency = sum(latencies) / len(latencies)

        # Success rate (quality_after >= 0.7)
        successes = sum(1 for r in refinements if r.quality_after >= self.quality_threshold)
        success_rate = successes / total_refinements

        # Quality trend
        quality_trend = self._calculate_quality_trend(quality_after_scores)

        # Strategy distribution
        strategy_distribution = {}
        for r in refinements:
            strategy_name = r.strategy.value if hasattr(r.strategy, 'value') else str(r.strategy)
            strategy_distribution[strategy_name] = strategy_distribution.get(strategy_name, 0) + 1

        # Strategy stats
        strategy_stats = self._calculate_mrf_strategy_stats(refinements)

        # Component usage
        component_usage = self._calculate_mrf_component_usage(refinements)

        # Provider stats
        provider_stats = self._calculate_mrf_provider_stats(refinements)

        # Recommended strategy (based on Thompson Sampling)
        recommended_strategy, recommendation_confidence = self._get_best_mrf_strategy(strategy_stats)

        analytics = MRFRefinementAnalytics(
            total_refinements=total_refinements,
            avg_quality_before=avg_quality_before,
            avg_quality_after=avg_quality_after,
            overall_improvement=overall_improvement,
            overall_improvement_percent=overall_improvement_percent,
            success_rate=success_rate,
            avg_latency_ms=avg_latency,
            quality_trend=quality_trend,
            strategy_distribution=strategy_distribution,
            strategy_stats=strategy_stats,
            component_usage=component_usage,
            provider_stats=provider_stats,
            recommended_strategy=recommended_strategy,
            recommendation_confidence=recommendation_confidence,
            days_analyzed=days
        )

        # Generate recommendation
        analytics.generate_recommendation()

        return analytics

    def get_mrf_strategy_recommendation(
        self,
        candidates: Optional[List[MRFStrategy]] = None
    ) -> Optional[MRFThompsonState]:
        """
        Get Thompson Sampling recommendation for best MRF strategy.

        Args:
            candidates: Optional list of candidate strategies (None = all)

        Returns:
            MRFThompsonState with recommended strategy or None
        """
        # Get Thompson priors for MRF strategies
        priors = self.db.get_mrf_thompson_priors()

        if not priors:
            return None

        # Filter by candidates if specified
        if candidates:
            candidate_values = {c.value for c in candidates}
            priors = [p for p in priors if p.strategy in candidate_values]

        if not priors:
            return None

        # Thompson Sampling: sample from Beta(alpha, beta) for each strategy
        best_prior = None
        best_sample = -1.0

        for prior in priors:
            # Sample from Beta distribution
            sample = random.betavariate(prior.alpha, prior.beta)
            if sample > best_sample:
                best_sample = sample
                best_prior = prior

        if best_prior:
            strategy = MRFStrategy(best_prior.strategy) if isinstance(best_prior.strategy, str) else best_prior.strategy
            return MRFThompsonState(
                strategy=strategy,
                alpha=best_prior.alpha,
                beta=best_prior.beta,
                total_samples=int(best_prior.alpha + best_prior.beta - 2)
            )

        return None

    def compare_mrf_strategies(
        self,
        strategy_a: MRFStrategy,
        strategy_b: MRFStrategy,
        days: int = 30
    ) -> Optional[Dict[str, Any]]:
        """
        Compare two MRF strategies statistically.

        Args:
            strategy_a: First strategy
            strategy_b: Second strategy
            days: Number of days to analyze

        Returns:
            Comparison result dict or None if insufficient data
        """
        # Get refinements for each strategy
        refinements_a = self.db.get_mrf_refinements(
            strategy=strategy_a.value,
            days=days,
            limit=10000
        )
        refinements_b = self.db.get_mrf_refinements(
            strategy=strategy_b.value,
            days=days,
            limit=10000
        )

        if len(refinements_a) < 5 or len(refinements_b) < 5:
            return None

        # Calculate improvements for each
        improvements_a = [r.quality_after - r.quality_before for r in refinements_a]
        improvements_b = [r.quality_after - r.quality_before for r in refinements_b]

        avg_improvement_a = sum(improvements_a) / len(improvements_a)
        avg_improvement_b = sum(improvements_b) / len(improvements_b)

        # Statistical significance
        significance, cohens_d = self._calculate_significance(improvements_a, improvements_b)

        is_significant = significance < 0.05

        # Determine winner
        winner = None
        if is_significant:
            if avg_improvement_a > avg_improvement_b:
                winner = strategy_a
            elif avg_improvement_b > avg_improvement_a:
                winner = strategy_b

        # Calculate additional stats
        success_rate_a = sum(1 for r in refinements_a if r.quality_after >= self.quality_threshold) / len(refinements_a)
        success_rate_b = sum(1 for r in refinements_b if r.quality_after >= self.quality_threshold) / len(refinements_b)

        avg_latency_a = sum(r.latency_ms for r in refinements_a) / len(refinements_a)
        avg_latency_b = sum(r.latency_ms for r in refinements_b) / len(refinements_b)

        return {
            "strategy_a": strategy_a.value,
            "strategy_b": strategy_b.value,
            "winner": winner.value if winner else None,
            "is_significant": is_significant,
            "statistics": {
                "improvement_difference": avg_improvement_a - avg_improvement_b,
                "improvement_difference_percent": ((avg_improvement_a - avg_improvement_b) / max(avg_improvement_b, 0.01)) * 100,
                "p_value": significance,
                "cohens_d": cohens_d,
                "sample_size_a": len(refinements_a),
                "sample_size_b": len(refinements_b)
            },
            "strategy_a_stats": {
                "avg_improvement": avg_improvement_a,
                "success_rate": success_rate_a,
                "avg_latency_ms": avg_latency_a,
                "total_refinements": len(refinements_a)
            },
            "strategy_b_stats": {
                "avg_improvement": avg_improvement_b,
                "success_rate": success_rate_b,
                "avg_latency_ms": avg_latency_b,
                "total_refinements": len(refinements_b)
            },
            "recommendation": self._generate_mrf_comparison_recommendation(
                strategy_a, strategy_b, winner, is_significant, cohens_d
            )
        }

    def export_mrf_analytics(
        self,
        days: int = 30,
        include_recent_refinements: int = 100
    ) -> Optional[MRFAnalyticsExport]:
        """
        Export comprehensive MRF analytics.

        Args:
            days: Number of days to analyze
            include_recent_refinements: Number of recent refinements to include

        Returns:
            MRFAnalyticsExport or None if no data
        """
        analytics = self.get_mrf_analytics(days=days)

        if not analytics:
            return None

        # Get Thompson states
        priors = self.db.get_mrf_thompson_priors()
        thompson_states = []
        for prior in priors:
            strategy = MRFStrategy(prior.strategy) if isinstance(prior.strategy, str) else prior.strategy
            thompson_states.append(MRFThompsonState(
                strategy=strategy,
                alpha=prior.alpha,
                beta=prior.beta,
                total_samples=int(prior.alpha + prior.beta - 2)
            ))

        # Get recent refinements
        recent_refinements_raw = self.db.get_mrf_refinements(
            days=days,
            limit=include_recent_refinements
        )

        recent_refinements = []
        for r in recent_refinements_raw[:include_recent_refinements]:
            strategy = MRFStrategy(r.strategy) if isinstance(r.strategy, str) else r.strategy
            recent_refinements.append(MRFRefinementRecord(
                id=r.id,
                prompt_name=r.prompt_name,
                strategy=strategy,
                quality_before=r.quality_before,
                quality_after=r.quality_after,
                latency_ms=r.latency_ms,
                model_provider=r.model_provider,
                components_applied=r.components_applied,
                created_at=r.created_at,
                metadata=r.metadata
            ))

        # Date range
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")

        return MRFAnalyticsExport(
            export_timestamp=datetime.now(),
            total_refinements=analytics.total_refinements,
            date_range={"start": start_date, "end": end_date},
            overall_analytics=analytics,
            thompson_states=thompson_states,
            recent_refinements=recent_refinements,
            metadata={
                "quality_threshold": self.quality_threshold,
                "days_analyzed": days,
                "generated_by": "PromptAnalyticsEngine"
            }
        )

    def update_mrf_thompson_prior(
        self,
        strategy: MRFStrategy,
        success: bool,
        quality_score: float
    ) -> None:
        """
        Update Thompson Sampling prior for an MRF strategy.

        Args:
            strategy: MRF strategy
            success: Whether refinement was successful (quality >= threshold)
            quality_score: Quality score achieved
        """
        self.db.update_mrf_thompson_prior(
            strategy=strategy.value,
            success=success,
            quality_score=quality_score
        )

    # MRF Private helper methods

    def _calculate_mrf_strategy_stats(
        self,
        refinements: List
    ) -> List[MRFStrategyStats]:
        """Calculate statistics for each MRF strategy."""
        strategy_data = {}

        for r in refinements:
            strategy_name = r.strategy.value if hasattr(r.strategy, 'value') else str(r.strategy)
            if strategy_name not in strategy_data:
                strategy_data[strategy_name] = {
                    "quality_before": [],
                    "quality_after": [],
                    "improvements": [],
                    "latencies": [],
                    "successes": 0
                }

            data = strategy_data[strategy_name]
            data["quality_before"].append(r.quality_before)
            data["quality_after"].append(r.quality_after)
            data["improvements"].append(r.quality_after - r.quality_before)
            data["latencies"].append(r.latency_ms)
            if r.quality_after >= self.quality_threshold:
                data["successes"] += 1

        results = []
        for strategy_name, data in strategy_data.items():
            n = len(data["quality_before"])
            if n == 0:
                continue

            avg_quality_before = sum(data["quality_before"]) / n
            avg_quality_after = sum(data["quality_after"]) / n
            avg_improvement = sum(data["improvements"]) / n
            avg_improvement_percent = (avg_improvement / avg_quality_before * 100) if avg_quality_before > 0 else 0.0
            success_rate = data["successes"] / n
            avg_latency = sum(data["latencies"]) / n

            # Get Thompson priors
            priors = self.db.get_mrf_thompson_priors(strategy=strategy_name)
            if priors:
                alpha = priors[0].alpha
                beta = priors[0].beta
            else:
                alpha = 1.0
                beta = 1.0

            expected_quality = alpha / (alpha + beta)

            try:
                strategy = MRFStrategy(strategy_name)
            except ValueError:
                strategy = MRFStrategy.AUTO

            results.append(MRFStrategyStats(
                strategy=strategy,
                total_refinements=n,
                avg_quality_before=avg_quality_before,
                avg_quality_after=avg_quality_after,
                avg_improvement=avg_improvement,
                avg_improvement_percent=avg_improvement_percent,
                success_rate=success_rate,
                avg_latency_ms=avg_latency,
                thompson_alpha=alpha,
                thompson_beta=beta,
                expected_quality=expected_quality
            ))

        # Sort by expected quality
        results.sort(key=lambda x: x.expected_quality, reverse=True)
        return results

    def _calculate_mrf_component_usage(
        self,
        refinements: List
    ) -> List[MRFComponentUsage]:
        """Calculate usage statistics for MRF 7-component structure."""
        components = ["ROLE", "OBJECTIVE", "PROCESS", "FORMAT", "CONSTRAINTS", "UNCERTAINTY", "VALIDATION"]
        component_data = {c: {"used": [], "skipped": []} for c in components}

        for r in refinements:
            applied = set(r.components_applied) if r.components_applied else set()
            for component in components:
                if component in applied:
                    component_data[component]["used"].append(r.quality_after)
                else:
                    component_data[component]["skipped"].append(r.quality_after)

        results = []
        for component in components:
            used = component_data[component]["used"]
            skipped = component_data[component]["skipped"]

            usage_count = len(used)
            avg_quality_when_used = sum(used) / len(used) if used else 0.0
            avg_quality_when_skipped = sum(skipped) / len(skipped) if skipped else 0.0

            # Effectiveness score: how much better is quality when this component is used
            if avg_quality_when_skipped > 0:
                effectiveness = (avg_quality_when_used - avg_quality_when_skipped) / avg_quality_when_skipped
            else:
                effectiveness = 0.0 if avg_quality_when_used == 0 else 1.0

            results.append(MRFComponentUsage(
                component_name=component,
                usage_count=usage_count,
                avg_quality_when_used=avg_quality_when_used,
                avg_quality_when_skipped=avg_quality_when_skipped,
                effectiveness_score=effectiveness
            ))

        # Sort by effectiveness
        results.sort(key=lambda x: x.effectiveness_score, reverse=True)
        return results

    def _calculate_mrf_provider_stats(
        self,
        refinements: List
    ) -> List[MRFModelProviderStats]:
        """Calculate statistics per model provider."""
        provider_data = {}

        for r in refinements:
            provider = r.model_provider
            if provider not in provider_data:
                provider_data[provider] = {
                    "improvements": [],
                    "latencies": [],
                    "strategies": {},
                    "successes": 0
                }

            data = provider_data[provider]
            improvement = r.quality_after - r.quality_before
            data["improvements"].append(improvement)
            data["latencies"].append(r.latency_ms)

            strategy_name = r.strategy.value if hasattr(r.strategy, 'value') else str(r.strategy)
            data["strategies"][strategy_name] = data["strategies"].get(strategy_name, 0) + 1

            if r.quality_after >= self.quality_threshold:
                data["successes"] += 1

        results = []
        for provider, data in provider_data.items():
            n = len(data["improvements"])
            if n == 0:
                continue

            avg_improvement = sum(data["improvements"]) / n
            avg_latency = sum(data["latencies"]) / n
            success_rate = data["successes"] / n

            # Find best strategy for this provider
            best_strategy_name = max(data["strategies"], key=data["strategies"].get)
            try:
                best_strategy = MRFStrategy(best_strategy_name)
            except ValueError:
                best_strategy = MRFStrategy.AUTO

            results.append(MRFModelProviderStats(
                provider=provider,
                total_refinements=n,
                avg_quality_improvement=avg_improvement,
                avg_latency_ms=avg_latency,
                success_rate=success_rate,
                best_strategy=best_strategy
            ))

        # Sort by average improvement
        results.sort(key=lambda x: x.avg_quality_improvement, reverse=True)
        return results

    def _get_best_mrf_strategy(
        self,
        strategy_stats: List[MRFStrategyStats]
    ) -> Tuple[MRFStrategy, float]:
        """Get best MRF strategy based on Thompson Sampling expected quality."""
        if not strategy_stats:
            return MRFStrategy.AUTO, 0.0

        best = max(strategy_stats, key=lambda x: x.expected_quality)

        # Confidence based on total samples
        total_samples = int(best.thompson_alpha + best.thompson_beta - 2)
        confidence = min(1.0, total_samples / 100)

        return best.strategy, confidence

    def _generate_mrf_comparison_recommendation(
        self,
        strategy_a: MRFStrategy,
        strategy_b: MRFStrategy,
        winner: Optional[MRFStrategy],
        is_significant: bool,
        cohens_d: float
    ) -> str:
        """Generate recommendation for MRF strategy comparison."""
        recommendations = []

        if not is_significant:
            recommendations.append(f"No statistically significant difference between {strategy_a.value} and {strategy_b.value}.")
            recommendations.append("Consider running more refinements for conclusive results.")
        else:
            if winner:
                recommendations.append(f"Strategy '{winner.value}' shows significantly better improvement.")

                if abs(cohens_d) > 0.8:
                    recommendations.append("Effect size is large (Cohen's d > 0.8) - strong recommendation to use this strategy.")
                elif abs(cohens_d) > 0.5:
                    recommendations.append("Effect size is medium (Cohen's d > 0.5).")
                else:
                    recommendations.append("Effect size is small - consider other factors like latency and specific use case.")

        return " ".join(recommendations)

    # Private helper methods

    def _std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)

    def _calculate_quality_trend(self, quality_scores: List[float]) -> QualityTrend:
        """Determine quality trend from scores."""
        if len(quality_scores) < 5:
            return QualityTrend.STABLE

        # Split into first and second half
        mid = len(quality_scores) // 2
        first_half = quality_scores[:mid]
        second_half = quality_scores[mid:]

        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)

        diff = second_avg - first_avg

        if diff > 0.05:
            return QualityTrend.IMPROVING
        elif diff < -0.05:
            return QualityTrend.DECLINING
        else:
            return QualityTrend.STABLE

    def _calculate_task_distribution(
        self,
        executions: List
    ) -> Dict[str, int]:
        """Calculate task type distribution."""
        distribution = {}
        for exec in executions:
            task_type = exec.task_type or "unknown"
            distribution[task_type] = distribution.get(task_type, 0) + 1
        return distribution

    def _calculate_version_performance(
        self,
        executions: List
    ) -> List[VersionPerformance]:
        """Calculate performance by version."""
        version_data = {}

        for exec in executions:
            version = exec.version
            if version not in version_data:
                version_data[version] = {
                    "executions": [],
                    "quality_scores": [],
                    "latencies": [],
                    "first_used": exec.created_at,
                    "last_used": exec.created_at
                }

            version_data[version]["executions"].append(exec)
            if exec.quality_score is not None:
                version_data[version]["quality_scores"].append(exec.quality_score)
            if exec.latency_ms is not None:
                version_data[version]["latencies"].append(exec.latency_ms)
            version_data[version]["last_used"] = max(
                version_data[version]["last_used"],
                exec.created_at
            )

        results = []
        for version, data in version_data.items():
            quality_scores = data["quality_scores"]
            latencies = data["latencies"]

            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
            avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
            successes = sum(1 for q in quality_scores if q >= self.quality_threshold)
            success_rate = successes / len(quality_scores) if quality_scores else 0.0

            results.append(VersionPerformance(
                version=version,
                total_executions=len(data["executions"]),
                avg_quality=avg_quality,
                avg_latency_ms=avg_latency,
                success_rate=success_rate,
                first_used=data["first_used"],
                last_used=data["last_used"]
            ))

        # Sort by version
        results.sort(key=lambda x: x.version)
        return results

    def _calculate_daily_breakdown(
        self,
        executions: List,
        days: int
    ) -> List[DailyBreakdown]:
        """Calculate daily performance breakdown."""
        daily_data = {}

        for exec in executions:
            date_str = exec.created_at.strftime("%Y-%m-%d")
            if date_str not in daily_data:
                daily_data[date_str] = {
                    "quality_scores": [],
                    "latencies": []
                }

            if exec.quality_score is not None:
                daily_data[date_str]["quality_scores"].append(exec.quality_score)
            if exec.latency_ms is not None:
                daily_data[date_str]["latencies"].append(exec.latency_ms)

        results = []
        for date_str, data in sorted(daily_data.items()):
            quality_scores = data["quality_scores"]
            latencies = data["latencies"]

            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
            avg_latency = sum(latencies) / len(latencies) if latencies else None
            successes = sum(1 for q in quality_scores if q >= self.quality_threshold)
            success_rate = successes / len(quality_scores) if quality_scores else 0.0

            results.append(DailyBreakdown(
                date=date_str,
                executions=len(quality_scores),
                avg_quality=avg_quality,
                success_rate=success_rate,
                avg_latency_ms=avg_latency
            ))

        return results

    def _get_thompson_expected_quality(self, prompt_name: str) -> Optional[float]:
        """Get Thompson Sampling expected quality for a prompt."""
        priors = self.db.get_thompson_priors(prompt_name=prompt_name)
        if not priors:
            return None

        # Average across task types
        expected_qualities = []
        for prior in priors:
            expected = prior.alpha / (prior.alpha + prior.beta)
            expected_qualities.append(expected)

        return sum(expected_qualities) / len(expected_qualities) if expected_qualities else None

    def _group_by_day(self, executions: List) -> List[Dict[str, Any]]:
        """Group executions by day."""
        daily_data = {}

        for exec in executions:
            date_str = exec.created_at.strftime("%Y-%m-%d")
            if date_str not in daily_data:
                daily_data[date_str] = []
            if exec.quality_score is not None:
                daily_data[date_str].append(exec.quality_score)

        return [
            {
                "date": date_str,
                "avg_quality": sum(scores) / len(scores) if scores else 0.0,
                "count": len(scores),
                "min_quality": min(scores) if scores else 0.0,
                "max_quality": max(scores) if scores else 0.0
            }
            for date_str, scores in sorted(daily_data.items())
        ]

    def _group_by_hour(self, executions: List) -> List[Dict[str, Any]]:
        """Group executions by hour."""
        hourly_data = {}

        for exec in executions:
            hour_str = exec.created_at.strftime("%Y-%m-%d %H:00")
            if hour_str not in hourly_data:
                hourly_data[hour_str] = []
            if exec.quality_score is not None:
                hourly_data[hour_str].append(exec.quality_score)

        return [
            {
                "hour": hour_str,
                "avg_quality": sum(scores) / len(scores) if scores else 0.0,
                "count": len(scores),
                "min_quality": min(scores) if scores else 0.0,
                "max_quality": max(scores) if scores else 0.0
            }
            for hour_str, scores in sorted(hourly_data.items())
        ]

    def _calculate_significance(
        self,
        scores_a: List[float],
        scores_b: List[float]
    ) -> Tuple[float, float]:
        """
        Calculate statistical significance using Welch's t-test approximation.

        Returns:
            (p_value, cohens_d)
        """
        n1, n2 = len(scores_a), len(scores_b)
        mean1 = sum(scores_a) / n1
        mean2 = sum(scores_b) / n2
        var1 = sum((x - mean1) ** 2 for x in scores_a) / (n1 - 1) if n1 > 1 else 0
        var2 = sum((x - mean2) ** 2 for x in scores_b) / (n2 - 1) if n2 > 1 else 0

        # Welch's t-statistic
        se = math.sqrt(var1 / n1 + var2 / n2) if (var1 / n1 + var2 / n2) > 0 else 1e-10
        t_stat = abs(mean1 - mean2) / se

        # Degrees of freedom (Welch-Satterthwaite)
        num = (var1 / n1 + var2 / n2) ** 2
        denom = (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1) if n1 > 1 and n2 > 1 else 1
        df = num / denom if denom > 0 else 1

        # Approximate p-value using normal distribution for large df
        # (simplified - use scipy.stats.t.sf for exact values)
        p_value = 2 * (1 - self._normal_cdf(t_stat)) if df > 30 else 0.5

        # Cohen's d (effect size)
        pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0

        return p_value, cohens_d

    def _normal_cdf(self, x: float) -> float:
        """Approximate normal CDF using error function approximation."""
        # Approximation of the normal CDF
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def _generate_comparison_recommendation(
        self,
        analytics_a: PromptAnalytics,
        analytics_b: PromptAnalytics,
        statistics: ComparisonStatistics,
        is_significant: bool
    ) -> str:
        """Generate recommendation for prompt comparison."""
        recommendations = []

        if not is_significant:
            recommendations.append("No statistically significant difference found.")
            recommendations.append("Consider running more executions for conclusive results.")
        else:
            quality_diff_pct = abs(statistics.quality_difference) * 100
            if statistics.quality_difference > 0:
                better = analytics_a.prompt_name
            else:
                better = analytics_b.prompt_name

            recommendations.append(f"{better} shows {quality_diff_pct:.1f}% better quality.")

            if statistics.cohens_d and abs(statistics.cohens_d) > 0.8:
                recommendations.append("Effect size is large (Cohen's d > 0.8).")
            elif statistics.cohens_d and abs(statistics.cohens_d) > 0.5:
                recommendations.append("Effect size is medium (Cohen's d > 0.5).")
            else:
                recommendations.append("Effect size is small - consider other factors.")

        # Check sample sizes
        if statistics.sample_size_a < 30 or statistics.sample_size_b < 30:
            recommendations.append("Warning: Small sample sizes may affect reliability.")

        return " ".join(recommendations)

    def _make_deployment_decision(
        self,
        winner: Optional[str],
        is_significant: bool,
        statistics: ComparisonStatistics
    ) -> str:
        """Make deployment decision based on comparison."""
        if not is_significant:
            return "HOLD - Insufficient evidence for change"

        if winner and statistics.cohens_d and abs(statistics.cohens_d) > 0.5:
            return f"DEPLOY - Strong evidence to use {winner}"

        if winner:
            return f"CONSIDER - Moderate evidence favors {winner}"

        return "HOLD - No clear winner"

    def _determine_severity(
        self,
        analytics: PromptAnalytics,
        threshold: float
    ) -> Severity:
        """Determine severity of underperformance."""
        # Critical: Very low quality AND declining
        if analytics.avg_quality < threshold - 0.2 and analytics.quality_trend == QualityTrend.DECLINING:
            return Severity.CRITICAL

        # High: Very low quality OR (low quality AND declining)
        if analytics.avg_quality < threshold - 0.2:
            return Severity.HIGH
        if analytics.avg_quality < threshold and analytics.quality_trend == QualityTrend.DECLINING:
            return Severity.HIGH

        # Medium: Below threshold
        if analytics.avg_quality < threshold:
            return Severity.MEDIUM

        # Low: Success rate issues but quality OK
        return Severity.LOW

    def _generate_thompson_reasoning(
        self,
        result: Dict[str, Any],
        alternatives: List[Dict[str, Any]],
        confidence: float
    ) -> str:
        """Generate human-readable reasoning for Thompson recommendation."""
        reasoning_parts = []

        reasoning_parts.append(
            f"Selected '{result['prompt_name']}' with expected quality {result['expected_quality']:.2f}."
        )

        if confidence > 0.8:
            reasoning_parts.append("High confidence based on sufficient execution history.")
        elif confidence > 0.5:
            reasoning_parts.append("Moderate confidence - more executions will improve accuracy.")
        else:
            reasoning_parts.append("Low confidence - recommendation may change with more data.")

        if alternatives:
            best_alt = alternatives[0]
            if best_alt["expected_quality"] > result["expected_quality"] * 0.95:
                reasoning_parts.append(
                    f"Alternative '{best_alt['name']}' is competitive "
                    f"(expected: {best_alt['expected_quality']:.2f})."
                )

        return " ".join(reasoning_parts)


def create_analytics_engine(db, quality_threshold: float = 0.7) -> PromptAnalyticsEngine:
    """
    Factory function to create analytics engine.

    Args:
        db: PromptlyDatabase instance
        quality_threshold: Threshold for success

    Returns:
        PromptAnalyticsEngine instance
    """
    return PromptAnalyticsEngine(db, quality_threshold)
