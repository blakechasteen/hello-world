"""Hive Health Assessment from Visual Analysis.

Assesses hive health by analyzing bee population, activity, brood patterns,
and presence of pests/disease indicators.

Classes:
- HealthMetrics: Health assessment results
- HealthAssessor: Main health assessment engine

Created: 2025-11-17
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np
from enum import Enum

from .object_detector import ObjectDetector, Detection, ObjectClass
from .bee_tracker import BeeTracker, BeeTrack


class HealthStatus(Enum):
    """Overall health status categories."""
    EXCELLENT = "excellent"  # >0.8
    GOOD = "good"  # 0.6-0.8
    FAIR = "fair"  # 0.4-0.6
    POOR = "poor"  # 0.2-0.4
    CRITICAL = "critical"  # <0.2


@dataclass
class HealthMetrics:
    """Hive health metrics from visual analysis."""

    # Population metrics
    bee_population: int  # Estimated visible bee count
    queen_present: bool  # Queen detected

    # Activity metrics
    activity_level: float  # 0-1, bee movement/activity

    # Brood metrics
    brood_pattern_score: float  # 0-1, compact = healthy

    # Pest/disease metrics
    varroa_detected: bool  # Varroa mites present

    # Resource metrics
    honey_stores: float  # 0-1, percentage of frame with honey
    pollen_stores: float  # 0-1, percentage with pollen

    # Overall assessment
    overall_health: float  # 0-1, composite score
    health_status: HealthStatus

    # Additional info
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "bee_population": self.bee_population,
            "queen_present": self.queen_present,
            "activity_level": float(self.activity_level),
            "brood_pattern_score": float(self.brood_pattern_score),
            "varroa_detected": self.varroa_detected,
            "honey_stores": float(self.honey_stores),
            "pollen_stores": float(self.pollen_stores),
            "overall_health": float(self.overall_health),
            "health_status": self.health_status.value,
            "metadata": self.metadata,
        }

    def get_summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"Hive Health: {self.health_status.value.upper()} ({self.overall_health:.2f})",
            f"",
            f"Population & Activity:",
            f"  • Bee count: {self.bee_population}",
            f"  • Queen present: {'✓' if self.queen_present else '✗'}",
            f"  • Activity level: {self.activity_level:.2f}",
            f"",
            f"Brood & Resources:",
            f"  • Brood pattern: {self.brood_pattern_score:.2f}",
            f"  • Honey stores: {self.honey_stores:.1%}",
            f"  • Pollen stores: {self.pollen_stores:.1%}",
            f"",
            f"Health Concerns:",
            f"  • Varroa mites: {'⚠ DETECTED' if self.varroa_detected else 'None detected'}",
        ]

        # Add recommendations
        recommendations = self._get_recommendations()
        if recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for rec in recommendations:
                lines.append(f"  • {rec}")

        return "\n".join(lines)

    def _get_recommendations(self) -> List[str]:
        """Generate recommendations based on metrics."""
        recommendations = []

        # Low population
        if self.bee_population < 50:
            recommendations.append("Low bee count - check for queen issues or colony decline")

        # No queen
        if not self.queen_present:
            recommendations.append("Queen not detected - verify queen presence manually")

        # Low activity
        if self.activity_level < 0.3:
            recommendations.append("Low activity - check temperature, weather conditions")

        # Poor brood pattern
        if self.brood_pattern_score < 0.4:
            recommendations.append("Spotty brood pattern - possible queen or disease issue")

        # Varroa detected
        if self.varroa_detected:
            recommendations.append("⚠ Varroa mites detected - treatment recommended")

        # Low resources
        if self.honey_stores < 0.2:
            recommendations.append("Low honey stores - consider supplemental feeding")

        if self.pollen_stores < 0.15:
            recommendations.append("Low pollen stores - check forage availability")

        return recommendations


class HealthAssessor:
    """Assess hive health from visual analysis.

    Combines object detection, bee tracking, and spatial analysis to
    evaluate hive health metrics.

    Example:
        ```python
        detector = ObjectDetector()
        tracker = BeeTracker()
        assessor = HealthAssessor(detector, tracker)

        for frame_number, frame in enumerate(video):
            metrics = await assessor.assess(frame, frame_number)
            print(metrics.get_summary())
        ```
    """

    def __init__(
        self,
        object_detector: ObjectDetector,
        bee_tracker: BeeTracker,
        ideal_population: int = 50000,  # Ideal hive population
    ):
        """Initialize health assessor.

        Args:
            object_detector: Object detector instance
            bee_tracker: Bee tracker instance
            ideal_population: Ideal bee population for normalization
        """
        self.detector = object_detector
        self.tracker = bee_tracker
        self.ideal_population = ideal_population

        # Historical metrics for trend analysis
        self.history: List[HealthMetrics] = []
        self.max_history = 100  # Keep last 100 assessments

    async def assess(
        self,
        frame: np.ndarray,
        frame_number: int
    ) -> HealthMetrics:
        """Assess hive health from single frame.

        Args:
            frame: RGB image
            frame_number: Current frame number

        Returns:
            Health metrics
        """
        # Detect objects
        detections = await self.detector.detect(frame)

        # Update bee tracking
        tracks = await self.tracker.update(detections, frame_number)

        # Compute population metrics
        bee_population = len(tracks)

        queen_present = any(
            d.class_name == ObjectClass.QUEEN
            for d in detections
        )

        # Compute activity metrics
        activity_level = await self.tracker.compute_hive_activity()

        # Analyze brood pattern
        brood_pattern_score = await self._analyze_brood_pattern(
            detections,
            frame
        )

        # Check for varroa mites
        varroa_detected = any(
            d.class_name == ObjectClass.VARROA_MITE
            for d in detections
        )

        # Estimate resource stores
        honey_stores = await self._estimate_stores(
            detections,
            ObjectClass.HONEY,
            frame
        )

        pollen_stores = await self._estimate_stores(
            detections,
            ObjectClass.POLLEN,
            frame
        )

        # Compute overall health
        overall_health = self._compute_overall_health(
            bee_population=bee_population,
            activity_level=activity_level,
            brood_pattern_score=brood_pattern_score,
            varroa_detected=varroa_detected,
            queen_present=queen_present,
            honey_stores=honey_stores,
            pollen_stores=pollen_stores,
        )

        # Determine health status
        health_status = self._classify_health_status(overall_health)

        # Create metrics
        metrics = HealthMetrics(
            bee_population=bee_population,
            queen_present=queen_present,
            activity_level=activity_level,
            brood_pattern_score=brood_pattern_score,
            varroa_detected=varroa_detected,
            honey_stores=honey_stores,
            pollen_stores=pollen_stores,
            overall_health=overall_health,
            health_status=health_status,
            metadata={
                "frame_number": frame_number,
                "total_detections": len(detections),
                "confirmed_tracks": len(tracks),
            }
        )

        # Store in history
        self.history.append(metrics)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        return metrics

    async def _analyze_brood_pattern(
        self,
        detections: List[Detection],
        frame: np.ndarray
    ) -> float:
        """Analyze brood pattern quality.

        Healthy brood: compact, solid pattern with few empty cells
        Unhealthy: spotty, irregular pattern (possible disease)

        Args:
            detections: List of detections
            frame: RGB image

        Returns:
            Brood pattern score (0-1, higher = healthier)
        """
        brood_detections = [
            d for d in detections
            if d.class_name == ObjectClass.BROOD
        ]

        if not brood_detections:
            return 0.0  # No brood visible

        # Analyze spatial distribution (compactness)
        positions = np.array([
            [d.bounding_box.x, d.bounding_box.y]
            for d in brood_detections
        ])

        # Compute clustering metric
        if len(positions) > 1:
            # Use pairwise distances
            try:
                from scipy.spatial.distance import pdist

                distances = pdist(positions)
                avg_distance = np.mean(distances)

                # Lower average distance = more compact = healthier
                # Normalize: assume max distance is 0.5 (half frame)
                compactness_score = 1.0 - min(avg_distance / 0.5, 1.0)

            except ImportError:
                # Fallback: use simple std dev
                std_dev = np.std(positions, axis=0).mean()
                compactness_score = 1.0 - min(std_dev / 0.25, 1.0)

        else:
            compactness_score = 0.5  # Single brood cell (uncertain)

        # Analyze coverage (what percentage of frame has brood)
        total_brood_area = sum(
            d.bounding_box.area()
            for d in brood_detections
        )

        # Ideal brood coverage: 30-60% of frame
        if 0.3 <= total_brood_area <= 0.6:
            coverage_score = 1.0
        elif total_brood_area < 0.3:
            coverage_score = total_brood_area / 0.3
        else:
            coverage_score = max(0, 1.0 - (total_brood_area - 0.6) / 0.4)

        # Weighted combination
        score = 0.7 * compactness_score + 0.3 * coverage_score

        return score

    async def _estimate_stores(
        self,
        detections: List[Detection],
        resource_class: ObjectClass,
        frame: np.ndarray
    ) -> float:
        """Estimate resource stores (honey or pollen).

        Args:
            detections: List of detections
            resource_class: ObjectClass.HONEY or ObjectClass.POLLEN
            frame: RGB image

        Returns:
            Store level (0-1, percentage of frame covered)
        """
        resource_detections = [
            d for d in detections
            if d.class_name == resource_class
        ]

        if not resource_detections:
            return 0.0

        # Total area covered
        total_area = sum(
            d.bounding_box.area()
            for d in resource_detections
        )

        # Normalize to 0-1 range
        # Assume full frame coverage = 1.0
        store_level = min(total_area, 1.0)

        return store_level

    def _compute_overall_health(
        self,
        bee_population: int,
        activity_level: float,
        brood_pattern_score: float,
        varroa_detected: bool,
        queen_present: bool,
        honey_stores: float,
        pollen_stores: float,
    ) -> float:
        """Compute composite health score.

        Args:
            bee_population: Visible bee count
            activity_level: Activity level (0-1)
            brood_pattern_score: Brood pattern quality (0-1)
            varroa_detected: Whether varroa mites detected
            queen_present: Whether queen detected
            honey_stores: Honey store level (0-1)
            pollen_stores: Pollen store level (0-1)

        Returns:
            Overall health score (0-1)
        """
        # Normalize population (visible bees, not full hive)
        # Assume max 100 visible bees is ideal for frame
        pop_score = min(bee_population / 100, 1.0)

        # Queen factor (binary)
        queen_score = 1.0 if queen_present else 0.5  # Penalty if not visible

        # Varroa factor (binary penalty)
        varroa_score = 0.0 if varroa_detected else 1.0

        # Weighted average
        score = (
            pop_score * 0.20 +
            activity_level * 0.15 +
            brood_pattern_score * 0.20 +
            queen_score * 0.15 +
            varroa_score * 0.10 +
            honey_stores * 0.10 +
            pollen_stores * 0.10
        )

        return score

    def _classify_health_status(self, score: float) -> HealthStatus:
        """Classify health status from score.

        Args:
            score: Overall health score (0-1)

        Returns:
            Health status category
        """
        if score >= 0.8:
            return HealthStatus.EXCELLENT
        elif score >= 0.6:
            return HealthStatus.GOOD
        elif score >= 0.4:
            return HealthStatus.FAIR
        elif score >= 0.2:
            return HealthStatus.POOR
        else:
            return HealthStatus.CRITICAL

    def get_trend(self, window: int = 10) -> Optional[str]:
        """Analyze health trend over recent history.

        Args:
            window: Number of recent frames to analyze

        Returns:
            Trend description ("improving", "stable", "declining")
        """
        if len(self.history) < 2:
            return None

        # Get recent scores
        recent = self.history[-window:]
        scores = [m.overall_health for m in recent]

        if len(scores) < 2:
            return "stable"

        # Simple linear regression
        x = np.arange(len(scores))
        y = np.array(scores)

        # Compute slope
        slope = np.polyfit(x, y, 1)[0]

        # Classify trend
        if slope > 0.01:  # Improving
            return "improving"
        elif slope < -0.01:  # Declining
            return "declining"
        else:
            return "stable"

    def get_statistics(self) -> Dict[str, Any]:
        """Get health assessment statistics.

        Returns:
            Dictionary of statistics
        """
        if not self.history:
            return {
                "assessments_count": 0,
                "avg_health": 0.0,
                "trend": None,
            }

        scores = [m.overall_health for m in self.history]

        return {
            "assessments_count": len(self.history),
            "avg_health": float(np.mean(scores)),
            "min_health": float(np.min(scores)),
            "max_health": float(np.max(scores)),
            "std_health": float(np.std(scores)),
            "trend": self.get_trend(),
            "varroa_detection_rate": sum(
                1 for m in self.history if m.varroa_detected
            ) / len(self.history),
            "queen_detection_rate": sum(
                1 for m in self.history if m.queen_present
            ) / len(self.history),
        }

    def reset(self):
        """Reset assessment history."""
        self.history.clear()
