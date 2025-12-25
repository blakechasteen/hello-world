"""
Promptly Analytics Dataclasses

Data structures for prompt analytics, performance tracking, and optimization.
Includes dataclasses for execution records, analytics summaries, and comparisons.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum


class QualityTrend(Enum):
    """Quality trend direction."""
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"


class Severity(Enum):
    """Issue severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PromptExecutionSummary:
    """Summary of a single prompt execution."""
    id: int
    prompt_name: str
    version: int
    task_type: Optional[str]
    quality_score: float
    latency_ms: float
    token_count: Optional[int]
    created_at: datetime

    @property
    def is_successful(self) -> bool:
        """Whether execution was successful (quality >= 0.7)."""
        return self.quality_score >= 0.7

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "prompt_name": self.prompt_name,
            "version": self.version,
            "task_type": self.task_type,
            "quality_score": self.quality_score,
            "latency_ms": self.latency_ms,
            "token_count": self.token_count,
            "created_at": self.created_at.isoformat(),
            "is_successful": self.is_successful
        }


@dataclass
class VersionPerformance:
    """Performance metrics for a specific prompt version."""
    version: int
    total_executions: int
    avg_quality: float
    avg_latency_ms: float
    success_rate: float
    first_used: datetime
    last_used: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "total_executions": self.total_executions,
            "avg_quality": self.avg_quality,
            "avg_latency_ms": self.avg_latency_ms,
            "success_rate": self.success_rate,
            "first_used": self.first_used.isoformat(),
            "last_used": self.last_used.isoformat()
        }


@dataclass
class TaskTypeStats:
    """Statistics for a specific task type."""
    task_type: str
    total_executions: int
    avg_quality: float
    success_rate: float
    top_prompts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_type": self.task_type,
            "total_executions": self.total_executions,
            "avg_quality": self.avg_quality,
            "success_rate": self.success_rate,
            "top_prompts": self.top_prompts
        }


@dataclass
class DailyBreakdown:
    """Daily performance breakdown."""
    date: str
    executions: int
    avg_quality: float
    success_rate: float
    avg_latency_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "date": self.date,
            "executions": self.executions,
            "avg_quality": self.avg_quality,
            "success_rate": self.success_rate,
            "avg_latency_ms": self.avg_latency_ms
        }


@dataclass
class PromptAnalytics:
    """Comprehensive analytics for a prompt."""
    prompt_name: str
    total_executions: int
    avg_quality: float
    avg_latency_ms: float
    success_rate: float
    quality_trend: QualityTrend
    quality_range: Dict[str, float]
    avg_tokens: Optional[int]
    task_type_distribution: Dict[str, int]
    version_performance: List[VersionPerformance]
    thompson_expected_quality: Optional[float]
    daily_breakdown: List[DailyBreakdown]
    days_analyzed: int
    recommendation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "prompt_name": self.prompt_name,
            "total_executions": self.total_executions,
            "avg_quality": self.avg_quality,
            "avg_latency_ms": self.avg_latency_ms,
            "success_rate": self.success_rate,
            "quality_trend": self.quality_trend.value,
            "quality_range": self.quality_range,
            "avg_tokens": self.avg_tokens,
            "task_type_distribution": self.task_type_distribution,
            "version_performance": [v.to_dict() for v in self.version_performance],
            "thompson_expected_quality": self.thompson_expected_quality,
            "daily_breakdown": [d.to_dict() for d in self.daily_breakdown],
            "days_analyzed": self.days_analyzed,
            "recommendation": self.recommendation
        }

    def generate_recommendation(self) -> str:
        """Generate a recommendation based on analytics."""
        if self.total_executions < 10:
            return "Insufficient data for recommendation. Run more executions."

        recommendations = []

        # Quality trend recommendations
        if self.quality_trend == QualityTrend.DECLINING:
            recommendations.append("Quality declining - consider prompt revision.")
        elif self.quality_trend == QualityTrend.IMPROVING:
            recommendations.append("Quality improving - current approach effective.")

        # Success rate recommendations
        if self.success_rate < 0.7:
            recommendations.append(f"Low success rate ({self.success_rate:.1%}) - needs optimization.")
        elif self.success_rate >= 0.9:
            recommendations.append("Excellent success rate - prompt is well-optimized.")

        # Thompson Sampling comparison
        if self.thompson_expected_quality:
            diff = self.avg_quality - self.thompson_expected_quality
            if diff < -0.1:
                recommendations.append("Actual quality below Thompson prediction - investigate recent changes.")

        # Latency recommendations
        if self.avg_latency_ms and self.avg_latency_ms > 5000:
            recommendations.append("High latency detected - consider prompt simplification.")

        self.recommendation = " ".join(recommendations) if recommendations else "Performance nominal."
        return self.recommendation


@dataclass
class ComparisonStatistics:
    """Statistical comparison between two prompts."""
    quality_difference: float
    latency_difference_ms: float
    success_rate_difference: float
    sample_size_a: int
    sample_size_b: int
    statistical_significance: float
    cohens_d: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "quality_difference": self.quality_difference,
            "quality_improvement_percent": self.quality_difference * 100,
            "latency_difference_ms": self.latency_difference_ms,
            "success_rate_difference": self.success_rate_difference,
            "sample_size_a": self.sample_size_a,
            "sample_size_b": self.sample_size_b,
            "statistical_significance": self.statistical_significance,
            "cohens_d": self.cohens_d
        }


@dataclass
class ComparisonResult:
    """Result of comparing two prompts."""
    prompt_a: str
    prompt_b: str
    prompt_a_stats: PromptAnalytics
    prompt_b_stats: PromptAnalytics
    winner: Optional[str]
    statistics: ComparisonStatistics
    is_significant: bool
    recommendation: str
    deployment_decision: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "prompt_a": self.prompt_a,
            "prompt_b": self.prompt_b,
            "prompt_a_stats": self.prompt_a_stats.to_dict(),
            "prompt_b_stats": self.prompt_b_stats.to_dict(),
            "winner": self.winner,
            "statistics": {
                "difference": self.statistics.to_dict(),
                "is_significant": self.is_significant
            },
            "recommendation": self.recommendation,
            "deployment_decision": self.deployment_decision
        }


@dataclass
class QualityAnomaly:
    """Detected quality anomaly."""
    anomaly_type: str  # 'sudden_drop', 'prolonged_low', 'high_variance', 'outlier'
    severity: Severity
    description: str
    timestamp: datetime
    affected_executions: List[int]
    suggested_action: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "anomaly_type": self.anomaly_type,
            "severity": self.severity.value,
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
            "affected_executions": self.affected_executions,
            "suggested_action": self.suggested_action
        }


@dataclass
class UnderperformingPrompt:
    """Information about an underperforming prompt."""
    prompt_name: str
    avg_quality: float
    success_rate: float
    total_executions: int
    quality_trend: QualityTrend
    severity: Severity
    issues: List[str]
    suggested_actions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt_name": self.prompt_name,
            "avg_quality": self.avg_quality,
            "success_rate": self.success_rate,
            "total_executions": self.total_executions,
            "quality_trend": self.quality_trend.value,
            "severity": self.severity.value,
            "issues": self.issues,
            "suggested_actions": self.suggested_actions
        }


@dataclass
class ThompsonRecommendation:
    """Thompson Sampling prompt recommendation."""
    recommended_prompt: str
    expected_quality: float
    confidence: float
    task_type: str
    alternatives: List[Dict[str, Any]]
    reasoning: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "recommended_prompt": self.recommended_prompt,
            "expected_quality": self.expected_quality,
            "confidence": self.confidence,
            "task_type": self.task_type,
            "alternatives": self.alternatives,
            "reasoning": self.reasoning
        }


@dataclass
class AnalyticsExport:
    """Complete analytics export for a set of prompts."""
    export_timestamp: datetime
    total_prompts: int
    total_executions: int
    date_range: Dict[str, str]
    prompts: List[PromptAnalytics]
    underperforming: List[UnderperformingPrompt]
    anomalies: List[QualityAnomaly]
    thompson_recommendations: Dict[str, ThompsonRecommendation]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "export_timestamp": self.export_timestamp.isoformat(),
            "total_prompts": self.total_prompts,
            "total_executions": self.total_executions,
            "date_range": self.date_range,
            "prompts": [p.to_dict() for p in self.prompts],
            "underperforming": [u.to_dict() for u in self.underperforming],
            "anomalies": [a.to_dict() for a in self.anomalies],
            "thompson_recommendations": {
                k: v.to_dict() for k, v in self.thompson_recommendations.items()
            },
            "metadata": self.metadata
        }

    def to_json(self, indent: int = 2) -> str:
        """Export as JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=indent)

    def to_csv(self) -> str:
        """Export prompts as CSV."""
        lines = [
            "prompt_name,total_executions,avg_quality,avg_latency_ms,success_rate,quality_trend,thompson_expected"
        ]
        for p in self.prompts:
            lines.append(
                f"{p.prompt_name},{p.total_executions},{p.avg_quality:.3f},"
                f"{p.avg_latency_ms:.1f},{p.success_rate:.3f},"
                f"{p.quality_trend.value},{p.thompson_expected_quality or ''}"
            )
        return "\n".join(lines)


# ==================== MRF Analytics Dataclasses ====================


class MRFStrategy(Enum):
    """MRF refinement strategies."""
    REFINE = "refine"
    CRITIQUE = "critique"
    VERIFY = "verify"
    ELEGANCE = "elegance"
    HOFSTADTER = "hofstadter"
    AUTO = "auto"


@dataclass
class MRFRefinementRecord:
    """Record of a single MRF refinement execution."""
    id: int
    prompt_name: str
    strategy: MRFStrategy
    quality_before: float
    quality_after: float
    latency_ms: float
    model_provider: str
    components_applied: List[str]  # Which of 7 components were used
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def quality_improvement(self) -> float:
        """Calculate quality improvement."""
        return self.quality_after - self.quality_before

    @property
    def improvement_percent(self) -> float:
        """Calculate percentage improvement."""
        if self.quality_before == 0:
            return 0.0
        return (self.quality_improvement / self.quality_before) * 100

    @property
    def is_successful(self) -> bool:
        """Whether refinement was successful (quality >= 0.7 after)."""
        return self.quality_after >= 0.7

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "prompt_name": self.prompt_name,
            "strategy": self.strategy.value,
            "quality_before": self.quality_before,
            "quality_after": self.quality_after,
            "quality_improvement": self.quality_improvement,
            "improvement_percent": self.improvement_percent,
            "latency_ms": self.latency_ms,
            "model_provider": self.model_provider,
            "components_applied": self.components_applied,
            "is_successful": self.is_successful,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class MRFStrategyStats:
    """Statistics for a specific MRF strategy."""
    strategy: MRFStrategy
    total_refinements: int
    avg_quality_before: float
    avg_quality_after: float
    avg_improvement: float
    avg_improvement_percent: float
    success_rate: float  # Percentage that reached quality >= 0.7
    avg_latency_ms: float
    thompson_alpha: float  # Thompson Sampling prior alpha
    thompson_beta: float   # Thompson Sampling prior beta
    expected_quality: float  # E[X] = alpha / (alpha + beta)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy": self.strategy.value,
            "total_refinements": self.total_refinements,
            "avg_quality_before": self.avg_quality_before,
            "avg_quality_after": self.avg_quality_after,
            "avg_improvement": self.avg_improvement,
            "avg_improvement_percent": self.avg_improvement_percent,
            "success_rate": self.success_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "thompson_alpha": self.thompson_alpha,
            "thompson_beta": self.thompson_beta,
            "expected_quality": self.expected_quality
        }


@dataclass
class MRFComponentUsage:
    """Usage statistics for MRF 7-component structure."""
    component_name: str  # ROLE, OBJECTIVE, PROCESS, FORMAT, CONSTRAINTS, UNCERTAINTY, VALIDATION
    usage_count: int
    avg_quality_when_used: float
    avg_quality_when_skipped: float
    effectiveness_score: float  # Quality difference when used vs skipped

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component_name": self.component_name,
            "usage_count": self.usage_count,
            "avg_quality_when_used": self.avg_quality_when_used,
            "avg_quality_when_skipped": self.avg_quality_when_skipped,
            "effectiveness_score": self.effectiveness_score
        }


@dataclass
class MRFModelProviderStats:
    """Statistics per model provider for MRF refinements."""
    provider: str  # claude, gemini, gpt, ollama
    total_refinements: int
    avg_quality_improvement: float
    avg_latency_ms: float
    success_rate: float
    best_strategy: MRFStrategy

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "provider": self.provider,
            "total_refinements": self.total_refinements,
            "avg_quality_improvement": self.avg_quality_improvement,
            "avg_latency_ms": self.avg_latency_ms,
            "success_rate": self.success_rate,
            "best_strategy": self.best_strategy.value
        }


@dataclass
class MRFRefinementAnalytics:
    """Comprehensive MRF refinement analytics."""
    total_refinements: int
    avg_quality_before: float
    avg_quality_after: float
    overall_improvement: float
    overall_improvement_percent: float
    success_rate: float
    avg_latency_ms: float
    quality_trend: QualityTrend
    strategy_distribution: Dict[str, int]  # Strategy -> count
    strategy_stats: List[MRFStrategyStats]
    component_usage: List[MRFComponentUsage]
    provider_stats: List[MRFModelProviderStats]
    recommended_strategy: MRFStrategy
    recommendation_confidence: float
    days_analyzed: int
    recommendation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_refinements": self.total_refinements,
            "avg_quality_before": self.avg_quality_before,
            "avg_quality_after": self.avg_quality_after,
            "overall_improvement": self.overall_improvement,
            "overall_improvement_percent": self.overall_improvement_percent,
            "success_rate": self.success_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "quality_trend": self.quality_trend.value,
            "strategy_distribution": self.strategy_distribution,
            "strategy_stats": [s.to_dict() for s in self.strategy_stats],
            "component_usage": [c.to_dict() for c in self.component_usage],
            "provider_stats": [p.to_dict() for p in self.provider_stats],
            "recommended_strategy": self.recommended_strategy.value,
            "recommendation_confidence": self.recommendation_confidence,
            "days_analyzed": self.days_analyzed,
            "recommendation": self.recommendation
        }

    def generate_recommendation(self) -> str:
        """Generate a recommendation based on MRF analytics."""
        if self.total_refinements < 10:
            return "Insufficient data for recommendation. Run more refinements."

        recommendations = []

        # Quality trend recommendations
        if self.quality_trend == QualityTrend.DECLINING:
            recommendations.append("MRF quality declining - review strategy selection.")
        elif self.quality_trend == QualityTrend.IMPROVING:
            recommendations.append("MRF quality improving - current approach effective.")

        # Success rate recommendations
        if self.success_rate < 0.7:
            recommendations.append(f"Low success rate ({self.success_rate:.1%}) - consider different strategies.")
        elif self.success_rate >= 0.9:
            recommendations.append("Excellent success rate - MRF configuration is well-optimized.")

        # Strategy recommendation
        if self.recommendation_confidence >= 0.8:
            recommendations.append(
                f"Recommended strategy: {self.recommended_strategy.value} "
                f"(confidence: {self.recommendation_confidence:.1%})"
            )

        # Improvement recommendations
        if self.overall_improvement_percent < 10:
            recommendations.append("Low improvement - consider using VERIFY or ELEGANCE strategies.")
        elif self.overall_improvement_percent >= 30:
            recommendations.append(f"Excellent improvement ({self.overall_improvement_percent:.1f}%) - MRF is highly effective.")

        self.recommendation = " ".join(recommendations) if recommendations else "Performance nominal."
        return self.recommendation


@dataclass
class MRFThompsonState:
    """Thompson Sampling state for MRF strategy selection."""
    strategy: MRFStrategy
    alpha: float  # Success count + 1
    beta: float   # Failure count + 1
    total_samples: int

    @property
    def expected_quality(self) -> float:
        """Expected quality from Thompson Sampling: E[X] = alpha / (alpha + beta)."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def confidence(self) -> float:
        """Confidence based on sample size."""
        # More samples = higher confidence
        return min(1.0, self.total_samples / 100)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy": self.strategy.value,
            "alpha": self.alpha,
            "beta": self.beta,
            "total_samples": self.total_samples,
            "expected_quality": self.expected_quality,
            "confidence": self.confidence
        }


@dataclass
class MRFAnalyticsExport:
    """Complete MRF analytics export."""
    export_timestamp: datetime
    total_refinements: int
    date_range: Dict[str, str]
    overall_analytics: MRFRefinementAnalytics
    thompson_states: List[MRFThompsonState]
    recent_refinements: List[MRFRefinementRecord]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "export_timestamp": self.export_timestamp.isoformat(),
            "total_refinements": self.total_refinements,
            "date_range": self.date_range,
            "overall_analytics": self.overall_analytics.to_dict(),
            "thompson_states": [t.to_dict() for t in self.thompson_states],
            "recent_refinements": [r.to_dict() for r in self.recent_refinements],
            "metadata": self.metadata
        }

    def to_json(self, indent: int = 2) -> str:
        """Export as JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=indent)


# Convenience type aliases for API responses
AnalyticsResponse = Dict[str, Any]
ComparisonResponse = Dict[str, Any]
RecommendationResponse = Dict[str, Any]
MRFAnalyticsResponse = Dict[str, Any]
