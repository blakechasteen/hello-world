"""
HoloLoom Anomaly Detection Module

ML-based anomaly detection for security pipeline (Phase 3).

Features:
- Real-time anomaly detection (streaming)
- Multiple detection algorithms (Isolation Forest, LSTM, Autoencoder)
- Baseline behavior modeling (normal user patterns)
- Anomaly scoring (0.0-1.0, higher = more anomalous)
- Automatic alerting on anomalies
- False positive tracking and learning
- Model retraining (weekly)
- Explainability (why was this flagged?)

Algorithms:
1. Isolation Forest: Fast, unsupervised, works well for high-dimensional data
2. LSTM: Good for sequential patterns, detects temporal anomalies
3. Autoencoder: Reconstruction error as anomaly score, deep learning

Anomaly response thresholds:
- Score < 0.3: Log only (low confidence)
- Score 0.3-0.7: Alert security team
- Score 0.7-0.9: Require MFA re-authentication
- Score > 0.9: Temporary account lock + alert

Integration points:
- Consume events from SIEM
- Send alerts to monitoring system (Grafana)
- Log to AuditTrail with anomaly scores
- Integrate with DistributedRateLimiter (auto-throttle anomalous users)

Usage:
    >>> from HoloLoom.security.anomaly import AnomalyDetector, SecurityEvent
    >>> detector = AnomalyDetector()
    >>>
    >>> # Train on normal behavior (7+ days recommended)
    >>> await detector.fit_baseline(events)
    >>>
    >>> # Detect anomalies in real-time
    >>> event = SecurityEvent(user_id="user_123", ...)
    >>> result = await detector.detect(event)
    >>>
    >>> if result.is_anomalous:
    ...     print(f"Anomaly score: {result.score:.2f}")
    ...     print(f"Explanation: {result.explanation}")

Author: HoloLoom Security Team
Created: 2025-11-15
"""

from HoloLoom.security.anomaly.core import (
    AnomalyDetector,
    SecurityEvent,
    AnomalyResult,
    AnomalyType,
    create_anomaly_detector
)
from HoloLoom.security.anomaly.baseline_modeler import (
    BaselineModeler,
    UserBaseline
)
from HoloLoom.security.anomaly.explainer import (
    AnomalyExplainer,
    Explanation,
    FeatureContribution
)

__all__ = [
    # Core
    "AnomalyDetector",
    "SecurityEvent",
    "AnomalyResult",
    "AnomalyType",
    "create_anomaly_detector",
    # Baseline
    "BaselineModeler",
    "UserBaseline",
    # Explainability
    "AnomalyExplainer",
    "Explanation",
    "FeatureContribution",
]
