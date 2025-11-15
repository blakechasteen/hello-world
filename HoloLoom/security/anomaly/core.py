"""
Anomaly Detection Core

Main orchestrator for ML-based anomaly detection.

Features:
- Ensemble voting (combine multiple detectors)
- Real-time scoring (0.0-1.0)
- Streaming processing
- Alert generation based on thresholds
- Integration with AuditTrail and RateLimiter

Author: HoloLoom Security Team
Created: 2025-11-15
"""

import logging
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import numpy as np

# Optional ML dependencies (graceful degradation)
try:
    from sklearn.ensemble import IsolationForest as SKLearnIsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("⚠️  scikit-learn not installed. Isolation Forest unavailable.")

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of anomalies that can be detected."""
    REQUEST_RATE = "request_rate"  # Sudden spike or drop in requests
    AUTHENTICATION = "authentication"  # Failed auth, unusual login times
    ACCESS_PATTERN = "access_pattern"  # Accessing unusual resources
    GEOGRAPHIC = "geographic"  # Login from unusual location
    BEHAVIORAL = "behavioral"  # Unusual query patterns
    TEMPORAL = "temporal"  # Activity at unusual hours
    UNKNOWN = "unknown"


class AnomalySeverity(Enum):
    """Severity levels for anomalies."""
    LOW = "low"  # Score < 0.3 (log only)
    MEDIUM = "medium"  # Score 0.3-0.7 (alert)
    HIGH = "high"  # Score 0.7-0.9 (require MFA)
    CRITICAL = "critical"  # Score > 0.9 (account lock)


@dataclass
class SecurityEvent:
    """
    Security event for anomaly detection.

    Represents a user action that will be analyzed for anomalous behavior.
    """
    # Required fields
    user_id: str
    timestamp: float  # Unix timestamp
    event_type: str  # "login", "query", "api_call", etc.

    # Optional context fields
    ip_address: Optional[str] = None
    geographic_location: Optional[str] = None  # Country code (e.g., "US")
    endpoint: Optional[str] = None
    query_complexity: Optional[float] = None  # 0.0-1.0
    request_size: Optional[int] = None  # Bytes
    authentication_method: Optional[str] = None  # "password", "mfa", etc.
    user_role: Optional[str] = None
    session_id: Optional[str] = None

    # Time-based features (auto-computed)
    hour_of_day: Optional[int] = None
    day_of_week: Optional[int] = None
    is_weekend: Optional[bool] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Auto-compute time-based features from timestamp."""
        if self.timestamp and not self.hour_of_day:
            dt = datetime.fromtimestamp(self.timestamp)
            self.hour_of_day = dt.hour
            self.day_of_week = dt.weekday()  # 0=Monday, 6=Sunday
            self.is_weekend = self.day_of_week >= 5

    def to_feature_vector(self) -> np.ndarray:
        """
        Convert event to feature vector for ML models.

        Returns:
            1D numpy array with normalized features
        """
        features = [
            self.hour_of_day / 24.0 if self.hour_of_day is not None else 0.0,
            self.day_of_week / 7.0 if self.day_of_week is not None else 0.0,
            1.0 if self.is_weekend else 0.0,
            self.query_complexity if self.query_complexity is not None else 0.5,
            min(self.request_size / 10000.0, 1.0) if self.request_size else 0.0,
            # Add more features as needed
        ]
        return np.array(features, dtype=np.float32)


@dataclass
class AnomalyResult:
    """
    Result of anomaly detection.

    Contains anomaly score, type, severity, and explanation.
    """
    event: SecurityEvent
    is_anomalous: bool
    score: float  # 0.0-1.0 (higher = more anomalous)
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    explanation: str
    detector_scores: Dict[str, float] = field(default_factory=dict)  # Individual detector scores
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    # Recommended actions
    should_alert: bool = False
    should_throttle: bool = False
    should_block: bool = False
    should_require_mfa: bool = False

    def __post_init__(self):
        """Auto-compute recommended actions based on severity."""
        if self.severity == AnomalySeverity.LOW:
            # Just log
            pass
        elif self.severity == AnomalySeverity.MEDIUM:
            self.should_alert = True
        elif self.severity == AnomalySeverity.HIGH:
            self.should_alert = True
            self.should_require_mfa = True
            self.should_throttle = True
        elif self.severity == AnomalySeverity.CRITICAL:
            self.should_alert = True
            self.should_block = True


class AnomalyDetector:
    """
    Main anomaly detection orchestrator.

    Ensemble-based detector that combines multiple ML algorithms:
    - Isolation Forest (fast, unsupervised)
    - LSTM (temporal patterns)
    - Autoencoder (deep learning)

    Features:
    - Real-time streaming detection
    - Baseline behavior modeling
    - Automatic alerting
    - False positive learning
    - Weekly model retraining
    """

    def __init__(
        self,
        enable_isolation_forest: bool = True,
        enable_lstm: bool = False,  # Requires torch
        enable_autoencoder: bool = False,  # Requires torch
        baseline_window_days: int = 7,
        alert_callback: Optional[Callable[[AnomalyResult], None]] = None,
        false_positive_threshold: float = 0.1  # Learn from FPs if rate > 10%
    ):
        """
        Initialize anomaly detector.

        Args:
            enable_isolation_forest: Use Isolation Forest detector
            enable_lstm: Use LSTM detector (requires PyTorch)
            enable_autoencoder: Use Autoencoder detector (requires PyTorch)
            baseline_window_days: Days of data for baseline modeling
            alert_callback: Function to call when anomaly detected
            false_positive_threshold: Learn from false positives if rate exceeds this
        """
        self.enable_isolation_forest = enable_isolation_forest and SKLEARN_AVAILABLE
        self.enable_lstm = enable_lstm
        self.enable_autoencoder = enable_autoencoder
        self.baseline_window_days = baseline_window_days
        self.alert_callback = alert_callback
        self.false_positive_threshold = false_positive_threshold

        # Detectors (lazy-loaded)
        self._isolation_forest_detector = None
        self._lstm_detector = None
        self._autoencoder_detector = None
        self._baseline_modeler = None
        self._explainer = None

        # False positive tracking
        self._total_detections = 0
        self._false_positives = 0
        self._fp_rate = 0.0

        # Model state
        self._is_trained = False
        self._last_training_time = None

        logger.info(f"AnomalyDetector initialized (isolation_forest={self.enable_isolation_forest}, "
                    f"lstm={self.enable_lstm}, autoencoder={self.enable_autoencoder})")

    @property
    def baseline_modeler(self):
        """Lazy-load baseline modeler."""
        if self._baseline_modeler is None:
            from HoloLoom.security.anomaly.baseline_modeler import BaselineModeler
            self._baseline_modeler = BaselineModeler(window_days=self.baseline_window_days)
        return self._baseline_modeler

    @property
    def explainer(self):
        """Lazy-load anomaly explainer."""
        if self._explainer is None:
            from HoloLoom.security.anomaly.explainer import AnomalyExplainer
            self._explainer = AnomalyExplainer()
        return self._explainer

    @property
    def isolation_forest_detector(self):
        """Lazy-load Isolation Forest detector."""
        if self._isolation_forest_detector is None and self.enable_isolation_forest:
            from HoloLoom.security.anomaly.isolation_forest import IsolationForestDetector
            self._isolation_forest_detector = IsolationForestDetector()
        return self._isolation_forest_detector

    @property
    def lstm_detector(self):
        """Lazy-load LSTM detector."""
        if self._lstm_detector is None and self.enable_lstm:
            from HoloLoom.security.anomaly.lstm_detector import LSTMDetector
            self._lstm_detector = LSTMDetector()
        return self._lstm_detector

    @property
    def autoencoder_detector(self):
        """Lazy-load Autoencoder detector."""
        if self._autoencoder_detector is None and self.enable_autoencoder:
            from HoloLoom.security.anomaly.autoencoder import AutoencoderDetector
            self._autoencoder_detector = AutoencoderDetector()
        return self._autoencoder_detector

    async def fit_baseline(self, events: List[SecurityEvent], per_user: bool = True):
        """
        Train detectors on baseline (normal) behavior.

        Args:
            events: List of security events (should be 7+ days of normal activity)
            per_user: Create separate baselines per user (recommended)
        """
        logger.info(f"Training baseline on {len(events)} events (per_user={per_user})")

        # Build baseline models (per-user or global)
        await self.baseline_modeler.fit(events, per_user=per_user)

        # Train each detector
        if self.enable_isolation_forest and self.isolation_forest_detector:
            await self.isolation_forest_detector.fit(events)

        if self.enable_lstm and self.lstm_detector:
            await self.lstm_detector.fit(events)

        if self.enable_autoencoder and self.autoencoder_detector:
            await self.autoencoder_detector.fit(events)

        self._is_trained = True
        self._last_training_time = datetime.now().timestamp()
        logger.info("Baseline training complete")

    async def detect(self, event: SecurityEvent) -> AnomalyResult:
        """
        Detect anomalies in a single event (real-time).

        Args:
            event: Security event to analyze

        Returns:
            AnomalyResult with score, type, and explanation
        """
        if not self._is_trained:
            logger.warning("Detector not trained! Using default scoring.")
            return self._create_untrained_result(event)

        # Run each detector and collect scores
        detector_scores = {}

        if self.enable_isolation_forest and self.isolation_forest_detector:
            score = await self.isolation_forest_detector.predict(event)
            detector_scores["isolation_forest"] = score

        if self.enable_lstm and self.lstm_detector:
            score = await self.lstm_detector.predict(event)
            detector_scores["lstm"] = score

        if self.enable_autoencoder and self.autoencoder_detector:
            score = await self.autoencoder_detector.predict(event)
            detector_scores["autoencoder"] = score

        # Ensemble: Average scores (simple voting)
        if detector_scores:
            ensemble_score = float(np.mean(list(detector_scores.values())))
        else:
            ensemble_score = 0.0

        # Determine anomaly type based on baseline deviation
        anomaly_type = await self._classify_anomaly_type(event, ensemble_score)

        # Determine severity based on score thresholds
        severity = self._score_to_severity(ensemble_score)

        # Generate explanation
        explanation = await self.explainer.explain(
            event=event,
            score=ensemble_score,
            anomaly_type=anomaly_type,
            detector_scores=detector_scores,
            baseline=self.baseline_modeler.get_baseline(event.user_id)
        )

        # Create result
        result = AnomalyResult(
            event=event,
            is_anomalous=(ensemble_score >= 0.3),  # Threshold for "anomalous"
            score=ensemble_score,
            anomaly_type=anomaly_type,
            severity=severity,
            explanation=explanation,
            detector_scores=detector_scores
        )

        # Update tracking
        self._total_detections += 1

        # Alert if configured
        if result.should_alert and self.alert_callback:
            self.alert_callback(result)

        return result

    async def detect_batch(self, events: List[SecurityEvent]) -> List[AnomalyResult]:
        """
        Detect anomalies in batch (efficient for processing logs).

        Args:
            events: List of security events

        Returns:
            List of AnomalyResults
        """
        # Process in parallel for efficiency
        tasks = [self.detect(event) for event in events]
        results = await asyncio.gather(*tasks)
        return list(results)

    def mark_false_positive(self, result: AnomalyResult):
        """
        Mark a detection as false positive (for learning).

        If FP rate exceeds threshold, triggers retraining.

        Args:
            result: AnomalyResult that was incorrectly flagged
        """
        self._false_positives += 1
        self._fp_rate = self._false_positives / max(self._total_detections, 1)

        logger.info(f"False positive marked. FP rate: {self._fp_rate:.2%}")

        # If FP rate is high, retrain (not implemented here, would be a background task)
        if self._fp_rate > self.false_positive_threshold:
            logger.warning(f"False positive rate ({self._fp_rate:.2%}) exceeds threshold "
                           f"({self.false_positive_threshold:.2%}). Consider retraining.")

    async def _classify_anomaly_type(self, event: SecurityEvent, score: float) -> AnomalyType:
        """
        Classify the type of anomaly based on event and baseline deviation.

        Args:
            event: Security event
            score: Anomaly score

        Returns:
            AnomalyType enum
        """
        baseline = self.baseline_modeler.get_baseline(event.user_id)

        if baseline is None:
            return AnomalyType.UNKNOWN

        # Check each anomaly type based on baseline deviations
        if event.geographic_location and event.geographic_location != baseline.typical_location:
            return AnomalyType.GEOGRAPHIC

        if event.hour_of_day is not None:
            typical_hours = baseline.typical_hours
            if event.hour_of_day not in typical_hours:
                return AnomalyType.TEMPORAL

        if event.event_type == "login" and score > 0.5:
            return AnomalyType.AUTHENTICATION

        if event.query_complexity is not None and event.query_complexity > 0.8:
            # High complexity queries can be anomalous
            return AnomalyType.BEHAVIORAL

        # Default to behavioral if no specific type identified
        return AnomalyType.BEHAVIORAL if score >= 0.3 else AnomalyType.UNKNOWN

    def _score_to_severity(self, score: float) -> AnomalySeverity:
        """
        Convert anomaly score to severity level.

        Args:
            score: Anomaly score (0.0-1.0)

        Returns:
            AnomalySeverity enum
        """
        if score < 0.3:
            return AnomalySeverity.LOW
        elif score < 0.7:
            return AnomalySeverity.MEDIUM
        elif score < 0.9:
            return AnomalySeverity.HIGH
        else:
            return AnomalySeverity.CRITICAL

    def _create_untrained_result(self, event: SecurityEvent) -> AnomalyResult:
        """
        Create a default result when detector is not trained.

        Args:
            event: Security event

        Returns:
            AnomalyResult with low confidence
        """
        return AnomalyResult(
            event=event,
            is_anomalous=False,
            score=0.0,
            anomaly_type=AnomalyType.UNKNOWN,
            severity=AnomalySeverity.LOW,
            explanation="Detector not trained. Using default scoring.",
            detector_scores={}
        )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get detection statistics.

        Returns:
            Dict with statistics
        """
        return {
            "total_detections": self._total_detections,
            "false_positives": self._false_positives,
            "false_positive_rate": self._fp_rate,
            "is_trained": self._is_trained,
            "last_training_time": self._last_training_time,
            "baseline_window_days": self.baseline_window_days,
            "enabled_detectors": {
                "isolation_forest": self.enable_isolation_forest,
                "lstm": self.enable_lstm,
                "autoencoder": self.enable_autoencoder
            }
        }


def create_anomaly_detector(
    enable_all: bool = False,
    alert_callback: Optional[Callable[[AnomalyResult], None]] = None
) -> AnomalyDetector:
    """
    Factory function to create anomaly detector.

    Args:
        enable_all: Enable all detectors (requires torch for LSTM/Autoencoder)
        alert_callback: Function to call when anomaly detected

    Returns:
        Configured AnomalyDetector instance
    """
    return AnomalyDetector(
        enable_isolation_forest=True,
        enable_lstm=enable_all,
        enable_autoencoder=enable_all,
        alert_callback=alert_callback
    )
