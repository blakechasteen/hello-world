"""
Anomaly Explainer

Provides human-readable explanations for anomaly detections.

Features:
- Feature-level explanations (which features contributed most)
- Rule-based explanations (e.g., "Login from unusual location")
- Baseline comparison (how does event differ from normal)
- SHAP-like feature importance (without requiring SHAP library)

Author: HoloLoom Security Team
Created: 2025-11-15
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class ExplanationType(Enum):
    """Types of explanations that can be generated."""
    GEOGRAPHIC = "geographic"  # Unusual location
    TEMPORAL = "temporal"  # Unusual time
    ACCESS_PATTERN = "access_pattern"  # Unusual resource access
    BEHAVIORAL = "behavioral"  # Unusual behavior
    REQUEST_RATE = "request_rate"  # Unusual request rate
    AUTHENTICATION = "authentication"  # Unusual auth pattern
    UNKNOWN = "unknown"


@dataclass
class FeatureContribution:
    """
    Contribution of a single feature to anomaly score.
    """
    feature_name: str
    value: float
    typical_value: float
    deviation: float  # How many std devs from normal
    contribution: float  # 0.0-1.0, contribution to anomaly score


@dataclass
class Explanation:
    """
    Complete explanation for an anomaly detection.
    """
    explanation_type: ExplanationType
    summary: str  # One-line human-readable summary
    details: List[str]  # Detailed explanations
    feature_contributions: List[FeatureContribution]
    confidence: float  # 0.0-1.0, confidence in explanation


class AnomalyExplainer:
    """
    Anomaly explainer for generating human-readable explanations.

    Features:
    - Rule-based explanations for common anomaly types
    - Feature-level contributions (which features were anomalous)
    - Baseline comparison
    - Simple, interpretable explanations
    """

    def __init__(self):
        """Initialize anomaly explainer."""
        logger.info("AnomalyExplainer initialized")

    async def explain(
        self,
        event: 'SecurityEvent',
        score: float,
        anomaly_type: 'AnomalyType',
        detector_scores: Dict[str, float],
        baseline: Optional['UserBaseline'] = None
    ) -> str:
        """
        Generate explanation for an anomaly detection.

        Args:
            event: Security event that was flagged
            score: Anomaly score (0.0-1.0)
            anomaly_type: Type of anomaly detected
            detector_scores: Individual detector scores
            baseline: User baseline (if available)

        Returns:
            Human-readable explanation string
        """
        # Build explanation parts
        parts = []

        # Main explanation based on anomaly type
        if anomaly_type.value == "geographic" and event.geographic_location:
            if baseline and baseline.typical_location:
                parts.append(
                    f"Login from unusual location: {event.geographic_location} "
                    f"(typical: {baseline.typical_location})"
                )
            else:
                parts.append(f"Login from unusual location: {event.geographic_location}")

        elif anomaly_type.value == "temporal" and event.hour_of_day is not None:
            if baseline and baseline.typical_hours:
                typical_hours_str = self._format_hour_range(baseline.typical_hours)
                parts.append(
                    f"Activity at unusual hour: {event.hour_of_day}:00 "
                    f"(typical: {typical_hours_str})"
                )
            else:
                parts.append(f"Activity at unusual hour: {event.hour_of_day}:00")

        elif anomaly_type.value == "access_pattern" and event.endpoint:
            if baseline and event.endpoint not in baseline.typical_endpoints:
                parts.append(
                    f"Accessing unusual endpoint: {event.endpoint} "
                    f"(not in typical access pattern)"
                )
            else:
                parts.append(f"Accessing unusual endpoint: {event.endpoint}")

        elif anomaly_type.value == "behavioral":
            if event.query_complexity and event.query_complexity > 0.8:
                parts.append(f"Unusually complex query (complexity: {event.query_complexity:.2f})")
            else:
                parts.append("Unusual behavioral pattern detected")

        elif anomaly_type.value == "authentication":
            parts.append("Unusual authentication pattern detected")

        else:
            parts.append(f"Anomalous behavior detected (type: {anomaly_type.value})")

        # Add detector contributions
        if detector_scores:
            detector_parts = []
            for detector, detector_score in detector_scores.items():
                if detector_score > 0.3:  # Only mention significant detectors
                    detector_parts.append(f"{detector}={detector_score:.2f}")

            if detector_parts:
                parts.append(f"Detector scores: {', '.join(detector_parts)}")

        # Add baseline comparison if available
        if baseline:
            deviation_score = baseline.get_deviation_score(event)
            if deviation_score > 0.5:
                parts.append(f"Baseline deviation: {deviation_score:.2f}")

        # Add overall score
        parts.append(f"Overall anomaly score: {score:.2f}")

        # Combine into single explanation
        explanation = " | ".join(parts)

        return explanation

    async def explain_detailed(
        self,
        event: 'SecurityEvent',
        score: float,
        anomaly_type: 'AnomalyType',
        detector_scores: Dict[str, float],
        baseline: Optional['UserBaseline'] = None
    ) -> Explanation:
        """
        Generate detailed structured explanation.

        Args:
            event: Security event
            score: Anomaly score
            anomaly_type: Type of anomaly
            detector_scores: Individual detector scores
            baseline: User baseline

        Returns:
            Structured Explanation object
        """
        # Convert anomaly type to explanation type
        if anomaly_type.value == "geographic":
            explanation_type = ExplanationType.GEOGRAPHIC
        elif anomaly_type.value == "temporal":
            explanation_type = ExplanationType.TEMPORAL
        elif anomaly_type.value == "access_pattern":
            explanation_type = ExplanationType.ACCESS_PATTERN
        elif anomaly_type.value == "behavioral":
            explanation_type = ExplanationType.BEHAVIORAL
        elif anomaly_type.value == "authentication":
            explanation_type = ExplanationType.AUTHENTICATION
        elif anomaly_type.value == "request_rate":
            explanation_type = ExplanationType.REQUEST_RATE
        else:
            explanation_type = ExplanationType.UNKNOWN

        # Generate summary
        summary = await self.explain(event, score, anomaly_type, detector_scores, baseline)

        # Generate detailed explanations
        details = []

        if baseline:
            # Temporal details
            if event.hour_of_day is not None and baseline.typical_hours:
                if event.hour_of_day not in baseline.typical_hours:
                    details.append(
                        f"User typically active during hours: {self._format_hour_range(baseline.typical_hours)}"
                    )
                    details.append(f"This event occurred at: {event.hour_of_day}:00")

            # Geographic details
            if event.geographic_location and baseline.known_locations:
                if event.geographic_location not in baseline.known_locations:
                    details.append(
                        f"User typically accesses from: {', '.join(sorted(baseline.known_locations))}"
                    )
                    details.append(f"This event from: {event.geographic_location}")

            # Endpoint details
            if event.endpoint and baseline.typical_endpoints:
                if event.endpoint not in baseline.typical_endpoints:
                    top_endpoints = sorted(
                        baseline.endpoint_frequencies.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:3]
                    top_endpoints_str = ", ".join(f"{ep} ({count})" for ep, count in top_endpoints)
                    details.append(f"User typically accesses: {top_endpoints_str}")
                    details.append(f"This event accessed: {event.endpoint}")

        # Feature contributions (simplified - would need actual feature importance from models)
        feature_contributions = self._compute_feature_contributions(event, baseline)

        # Confidence in explanation (based on how many details we could provide)
        confidence = min(1.0, len(details) / 3.0)

        return Explanation(
            explanation_type=explanation_type,
            summary=summary,
            details=details,
            feature_contributions=feature_contributions,
            confidence=confidence
        )

    def _format_hour_range(self, hours: set) -> str:
        """
        Format set of hours into readable ranges.

        Args:
            hours: Set of hours (0-23)

        Returns:
            Formatted string (e.g., "9-17, 20-22")
        """
        if not hours:
            return "N/A"

        sorted_hours = sorted(hours)

        # Find consecutive ranges
        ranges = []
        start = sorted_hours[0]
        prev = start

        for hour in sorted_hours[1:] + [None]:
            if hour is None or hour != prev + 1:
                if start == prev:
                    ranges.append(f"{start}")
                else:
                    ranges.append(f"{start}-{prev}")

                if hour is not None:
                    start = hour

            if hour is not None:
                prev = hour

        return ", ".join(ranges)

    def _compute_feature_contributions(
        self,
        event: 'SecurityEvent',
        baseline: Optional['UserBaseline']
    ) -> List[FeatureContribution]:
        """
        Compute feature-level contributions to anomaly score.

        Simplified version - in production would use SHAP or similar.

        Args:
            event: Security event
            baseline: User baseline

        Returns:
            List of FeatureContribution objects
        """
        contributions = []

        if not baseline:
            return contributions

        # Hour of day contribution
        if event.hour_of_day is not None and baseline.typical_hours:
            is_typical = event.hour_of_day in baseline.typical_hours
            contributions.append(FeatureContribution(
                feature_name="hour_of_day",
                value=float(event.hour_of_day),
                typical_value=float(baseline.peak_hour) if baseline.peak_hour else 12.0,
                deviation=0.0 if is_typical else 1.0,
                contribution=0.0 if is_typical else 0.3
            ))

        # Geographic location contribution
        if event.geographic_location and baseline.typical_location:
            is_typical = event.geographic_location == baseline.typical_location
            contributions.append(FeatureContribution(
                feature_name="geographic_location",
                value=hash(event.geographic_location) % 100 / 100.0,  # Dummy numeric value
                typical_value=hash(baseline.typical_location) % 100 / 100.0,
                deviation=0.0 if is_typical else 1.0,
                contribution=0.0 if is_typical else 0.4
            ))

        # Query complexity contribution
        if event.query_complexity is not None and baseline.std_query_complexity > 0:
            deviation = abs(event.query_complexity - baseline.avg_query_complexity) / baseline.std_query_complexity
            contributions.append(FeatureContribution(
                feature_name="query_complexity",
                value=event.query_complexity,
                typical_value=baseline.avg_query_complexity,
                deviation=deviation,
                contribution=min(1.0, deviation / 3.0)
            ))

        return contributions
