"""
Advanced statistical analytics for Keep.

Provides sophisticated analysis including:
- Statistical modeling
- Predictive analytics
- Correlation analysis
- Anomaly detection
- Optimization recommendations
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict
import math

from apps.keep.apiary import Apiary
from apps.keep.models import Colony, Inspection
from apps.keep.types import HealthStatus, QueenStatus
from apps.keep.transforms import extract_time_series, compute_trend


@dataclass
class StatisticalSummary:
    """Statistical summary of a metric."""
    mean: float
    median: float
    std_dev: float
    min_val: float
    max_val: float
    percentile_25: float
    percentile_75: float
    sample_size: int


@dataclass
class CorrelationResult:
    """Correlation between two metrics."""
    metric_a: str
    metric_b: str
    correlation: float
    p_value: Optional[float]
    sample_size: int
    interpretation: str


@dataclass
class PredictionResult:
    """Prediction for future metric value."""
    metric: str
    current_value: float
    predicted_value: float
    days_ahead: int
    confidence_lower: float
    confidence_upper: float
    confidence_level: float
    method: str


@dataclass
class AnomalyResult:
    """Detected anomaly in data."""
    entity_id: str
    metric: str
    value: float
    expected_range: Tuple[float, float]
    severity: str  # 'mild', 'moderate', 'severe'
    timestamp: datetime


class AdvancedAnalytics:
    """
    Advanced analytics engine for apiary data.

    Provides statistical modeling, predictions, and insights beyond
    basic analytics.
    """

    def __init__(self, apiary: Apiary):
        """
        Initialize advanced analytics.

        Args:
            apiary: Apiary to analyze
        """
        self.apiary = apiary

    # =========================================================================
    # Statistical Summaries
    # =========================================================================

    def compute_statistical_summary(self, values: List[float]) -> StatisticalSummary:
        """
        Compute statistical summary of values.

        Args:
            values: List of numeric values

        Returns:
            Statistical summary
        """
        if not values:
            return StatisticalSummary(
                mean=0.0, median=0.0, std_dev=0.0,
                min_val=0.0, max_val=0.0,
                percentile_25=0.0, percentile_75=0.0,
                sample_size=0
            )

        sorted_vals = sorted(values)
        n = len(sorted_vals)

        # Mean
        mean = sum(sorted_vals) / n

        # Median
        if n % 2 == 0:
            median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
        else:
            median = sorted_vals[n // 2]

        # Standard deviation
        variance = sum((x - mean) ** 2 for x in sorted_vals) / n
        std_dev = math.sqrt(variance)

        # Percentiles
        def percentile(p):
            k = (n - 1) * p
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return sorted_vals[int(k)]
            d0 = sorted_vals[int(f)] * (c - k)
            d1 = sorted_vals[int(c)] * (k - f)
            return d0 + d1

        return StatisticalSummary(
            mean=mean,
            median=median,
            std_dev=std_dev,
            min_val=sorted_vals[0],
            max_val=sorted_vals[-1],
            percentile_25=percentile(0.25),
            percentile_75=percentile(0.75),
            sample_size=n
        )

    def analyze_population_statistics(self) -> Dict[str, StatisticalSummary]:
        """
        Analyze population statistics across apiary.

        Returns:
            Dict with statistics by health status
        """
        results = {}

        # Overall statistics
        all_pops = [c.population_estimate for c in self.apiary.colonies.values()]
        results["all_colonies"] = self.compute_statistical_summary(all_pops)

        # By health status
        by_health = defaultdict(list)
        for colony in self.apiary.colonies.values():
            by_health[colony.health_status].append(colony.population_estimate)

        for status, pops in by_health.items():
            results[f"health_{status.value}"] = self.compute_statistical_summary(pops)

        return results

    # =========================================================================
    # Correlation Analysis
    # =========================================================================

    def compute_correlation(
        self,
        values_a: List[float],
        values_b: List[float]
    ) -> float:
        """
        Compute Pearson correlation coefficient.

        Args:
            values_a: First variable
            values_b: Second variable

        Returns:
            Correlation coefficient (-1 to 1)
        """
        if len(values_a) != len(values_b) or len(values_a) < 2:
            return 0.0

        n = len(values_a)
        mean_a = sum(values_a) / n
        mean_b = sum(values_b) / n

        numerator = sum((a - mean_a) * (b - mean_b) for a, b in zip(values_a, values_b))

        std_a = math.sqrt(sum((a - mean_a) ** 2 for a in values_a))
        std_b = math.sqrt(sum((b - mean_b) ** 2 for b in values_b))

        denominator = std_a * std_b

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def analyze_queen_age_health_correlation(self) -> Optional[CorrelationResult]:
        """
        Analyze correlation between queen age and colony health.

        Returns:
            Correlation result or None if insufficient data
        """
        queen_ages = []
        health_scores = []

        health_value = {
            HealthStatus.EXCELLENT: 5,
            HealthStatus.GOOD: 4,
            HealthStatus.FAIR: 3,
            HealthStatus.POOR: 2,
            HealthStatus.CRITICAL: 1,
        }

        for colony in self.apiary.colonies.values():
            if colony.queen_age_months is not None:
                queen_ages.append(colony.queen_age_months)
                health_scores.append(health_value[colony.health_status])

        if len(queen_ages) < 3:
            return None

        corr = self.compute_correlation(queen_ages, health_scores)

        # Interpretation
        if abs(corr) < 0.3:
            interpretation = "weak correlation"
        elif abs(corr) < 0.7:
            interpretation = "moderate correlation"
        else:
            interpretation = "strong correlation"

        if corr < 0:
            interpretation = "negative " + interpretation
        else:
            interpretation = "positive " + interpretation

        return CorrelationResult(
            metric_a="queen_age_months",
            metric_b="health_status",
            correlation=corr,
            p_value=None,  # Would need more complex stats for p-value
            sample_size=len(queen_ages),
            interpretation=interpretation
        )

    # =========================================================================
    # Predictive Analytics
    # =========================================================================

    def predict_population_growth(
        self,
        colony: Colony,
        days_ahead: int = 30
    ) -> Optional[PredictionResult]:
        """
        Predict colony population growth.

        Args:
            colony: Colony to predict
            days_ahead: Days into future to predict

        Returns:
            Prediction result or None if insufficient data
        """
        # Get inspection history for this colony
        inspections = [
            i for i in self.apiary.inspections
            if i.colony_id == colony.colony_id
        ]

        if len(inspections) < 3:
            return None  # Need at least 3 data points

        # Extract population time series
        def extract_population(insp: Inspection) -> Optional[float]:
            return insp.findings.get("population_estimate")

        series = extract_time_series(inspections, extract_population)

        if len(series) < 3:
            return None

        # Compute trend
        trend = compute_trend(series)

        # Simple linear extrapolation
        current_value = series[-1][1]
        predicted_value = current_value + (trend["slope"] * days_ahead)

        # Bounds check
        predicted_value = max(0, min(predicted_value, 100000))

        # Confidence interval (rough estimate based on std dev)
        values = [v for t, v in series]
        std_dev = math.sqrt(sum((v - current_value) ** 2 for v in values) / len(values))

        confidence_lower = max(0, predicted_value - 1.96 * std_dev)
        confidence_upper = min(100000, predicted_value + 1.96 * std_dev)

        return PredictionResult(
            metric="population",
            current_value=current_value,
            predicted_value=predicted_value,
            days_ahead=days_ahead,
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            confidence_level=0.95,
            method="linear_regression"
        )

    def predict_harvest_potential(
        self,
        hive_id: str,
        based_on_days: int = 365
    ) -> Optional[PredictionResult]:
        """
        Predict harvest potential for a hive.

        Args:
            hive_id: Hive to predict
            based_on_days: Historical days to base prediction on

        Returns:
            Prediction result or None if insufficient data
        """
        cutoff = datetime.now() - timedelta(days=based_on_days)

        harvests = [
            h for h in self.apiary.harvests
            if h.hive_id == hive_id and h.timestamp >= cutoff and h.product_type == "honey"
        ]

        if not harvests:
            return None

        # Compute average harvest
        total = sum(h.quantity for h in harvests)
        avg_per_harvest = total / len(harvests)

        # Predict next year based on average
        predicted = avg_per_harvest * (12 / len(harvests) * (365 / based_on_days))

        # Confidence based on variance
        variance = sum((h.quantity - avg_per_harvest) ** 2 for h in harvests) / len(harvests)
        std_dev = math.sqrt(variance)

        return PredictionResult(
            metric="honey_harvest_lbs",
            current_value=total,
            predicted_value=predicted,
            days_ahead=365,
            confidence_lower=max(0, predicted - std_dev),
            confidence_upper=predicted + std_dev,
            confidence_level=0.68,
            method="historical_average"
        )

    # =========================================================================
    # Anomaly Detection
    # =========================================================================

    def detect_anomalies(self) -> List[AnomalyResult]:
        """
        Detect anomalies in apiary data.

        Returns:
            List of detected anomalies
        """
        anomalies = []

        # Check population anomalies
        populations = [c.population_estimate for c in self.apiary.colonies.values()]
        if populations:
            stats = self.compute_statistical_summary(populations)

            # Use IQR method for outliers
            iqr = stats.percentile_75 - stats.percentile_25
            lower_bound = stats.percentile_25 - 1.5 * iqr
            upper_bound = stats.percentile_75 + 1.5 * iqr

            for colony in self.apiary.colonies.values():
                pop = colony.population_estimate

                if pop < lower_bound:
                    severity = "severe" if pop < stats.percentile_25 - 3 * iqr else "moderate"
                    anomalies.append(AnomalyResult(
                        entity_id=colony.colony_id,
                        metric="population",
                        value=pop,
                        expected_range=(lower_bound, upper_bound),
                        severity=severity,
                        timestamp=datetime.now()
                    ))
                elif pop > upper_bound:
                    severity = "mild"  # High population is usually good
                    anomalies.append(AnomalyResult(
                        entity_id=colony.colony_id,
                        metric="population",
                        value=pop,
                        expected_range=(lower_bound, upper_bound),
                        severity=severity,
                        timestamp=datetime.now()
                    ))

        # Check inspection frequency anomalies
        for hive_id in self.apiary.hives.keys():
            inspections = [
                i for i in self.apiary.inspections
                if i.hive_id == hive_id
            ]

            if len(inspections) >= 2:
                # Sort by time
                inspections.sort(key=lambda i: i.timestamp)

                # Check days since last inspection
                days_since = (datetime.now() - inspections[-1].timestamp).days

                if days_since > 21:  # More than 3 weeks
                    severity = "severe" if days_since > 30 else "moderate"
                    anomalies.append(AnomalyResult(
                        entity_id=hive_id,
                        metric="days_since_inspection",
                        value=days_since,
                        expected_range=(0, 14),
                        severity=severity,
                        timestamp=datetime.now()
                    ))

        return anomalies

    # =========================================================================
    # Optimization Recommendations
    # =========================================================================

    def recommend_interventions(self) -> List[Dict[str, Any]]:
        """
        Recommend interventions based on statistical analysis.

        Returns:
            List of intervention recommendations
        """
        recommendations = []

        # Analyze queen age patterns
        queen_health_corr = self.analyze_queen_age_health_correlation()
        if queen_health_corr and queen_health_corr.correlation < -0.5:
            # Negative correlation: older queens = worse health
            recommendations.append({
                "type": "queen_management",
                "priority": "high",
                "title": "Consider proactive queen replacement program",
                "reasoning": (
                    f"Statistical analysis shows {queen_health_corr.interpretation} "
                    f"between queen age and colony health (r={queen_health_corr.correlation:.2f}). "
                    "Colonies with older queens tend to have poorer health."
                ),
                "actions": [
                    "Identify colonies with queens older than 24 months",
                    "Plan requeening for oldest queens first",
                    "Consider raising own queens for sustainability"
                ]
            })

        # Analyze population distribution
        pop_stats = self.analyze_population_statistics()
        overall = pop_stats.get("all_colonies")

        if overall and overall.std_dev > overall.mean * 0.5:
            # High variance in populations
            recommendations.append({
                "type": "population_balancing",
                "priority": "medium",
                "title": "High population variance detected",
                "reasoning": (
                    f"Colony populations vary widely (σ={overall.std_dev:.0f}, "
                    f"μ={overall.mean:.0f}). Some colonies may need support."
                ),
                "actions": [
                    "Consider balancing strong and weak colonies",
                    "Move frames of brood from strong to weak colonies",
                    "Investigate why some colonies are underperforming"
                ]
            })

        # Check for anomalies
        anomalies = self.detect_anomalies()
        severe_anomalies = [a for a in anomalies if a.severity == "severe"]

        if severe_anomalies:
            recommendations.append({
                "type": "anomaly_response",
                "priority": "urgent",
                "title": f"{len(severe_anomalies)} severe anomalies detected",
                "reasoning": "Statistical outliers detected requiring immediate attention",
                "actions": [
                    f"Investigate {a.metric} anomaly in {a.entity_id}"
                    for a in severe_anomalies[:3]
                ]
            })

        return recommendations


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_statistical_analysis(apiary: Apiary) -> Dict[str, Any]:
    """
    Quick statistical analysis of apiary.

    Args:
        apiary: Apiary to analyze

    Returns:
        Dict with key statistics
    """
    analytics = AdvancedAnalytics(apiary)

    pop_stats = analytics.analyze_population_statistics()
    anomalies = analytics.detect_anomalies()
    recommendations = analytics.recommend_interventions()

    return {
        "population_statistics": pop_stats.get("all_colonies").__dict__ if pop_stats.get("all_colonies") else None,
        "anomalies_detected": len(anomalies),
        "severe_anomalies": len([a for a in anomalies if a.severity == "severe"]),
        "recommendations": len(recommendations),
        "high_priority_actions": len([r for r in recommendations if r["priority"] in ["urgent", "high"]]),
    }
