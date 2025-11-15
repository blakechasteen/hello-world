"""
Isolation Forest Anomaly Detector

Fast, unsupervised anomaly detection using Isolation Forest algorithm.

Features:
- No labeled data required (unsupervised)
- Fast training and prediction
- Works well for high-dimensional data
- Detects outliers based on ease of isolation

Algorithm:
Isolation Forest isolates anomalies by randomly selecting a feature and
then randomly selecting a split value. Anomalies are easier to isolate
(require fewer splits), so they have shorter average path lengths in the
isolation trees.

Author: HoloLoom Security Team
Created: 2025-11-15
"""

import logging
from typing import List, Optional
import numpy as np

# Optional sklearn dependency (graceful degradation)
try:
    from sklearn.ensemble import IsolationForest as SKLearnIsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("⚠️  scikit-learn not installed. Isolation Forest unavailable.")

logger = logging.getLogger(__name__)


class IsolationForestDetector:
    """
    Isolation Forest detector for anomaly detection.

    Uses sklearn's IsolationForest implementation.

    Features:
    - Fast training (<100ms for 10k events)
    - Fast prediction (<1ms per event)
    - No deep learning required
    - Works well for high-dimensional data
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.05,  # Expected % of anomalies
        random_state: int = 42
    ):
        """
        Initialize Isolation Forest detector.

        Args:
            n_estimators: Number of isolation trees
            contamination: Expected proportion of anomalies (0.0-0.5)
            random_state: Random seed for reproducibility
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for Isolation Forest. "
                              "Install with: pip install scikit-learn")

        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state

        # Model (lazy-initialized)
        self._model: Optional[SKLearnIsolationForest] = None
        self._is_fitted = False

        logger.info(f"IsolationForestDetector initialized (n_estimators={n_estimators}, "
                    f"contamination={contamination})")

    async def fit(self, events: List['SecurityEvent']):
        """
        Train Isolation Forest on normal behavior.

        Args:
            events: List of security events (should be mostly normal)
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available. Skipping Isolation Forest training.")
            return

        logger.info(f"Training Isolation Forest on {len(events)} events")

        # Convert events to feature vectors
        X = np.array([event.to_feature_vector() for event in events])

        # Create and fit model
        self._model = SKLearnIsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1  # Use all CPU cores
        )

        self._model.fit(X)
        self._is_fitted = True

        logger.info(f"Isolation Forest training complete ({X.shape[0]} samples, "
                    f"{X.shape[1]} features)")

    async def predict(self, event: 'SecurityEvent') -> float:
        """
        Predict anomaly score for a single event.

        Args:
            event: Security event to score

        Returns:
            Anomaly score (0.0-1.0, higher = more anomalous)
        """
        if not self._is_fitted or not SKLEARN_AVAILABLE:
            return 0.0

        # Convert to feature vector
        X = event.to_feature_vector().reshape(1, -1)

        # Get anomaly score from sklearn
        # sklearn returns: -1 for anomalies, 1 for inliers
        # decision_function returns: negative for anomalies, positive for inliers
        raw_score = self._model.decision_function(X)[0]

        # Normalize to 0-1 range (higher = more anomalous)
        # Raw scores typically range from -0.5 to 0.5
        normalized_score = max(0.0, min(1.0, 0.5 - raw_score))

        return float(normalized_score)

    async def predict_batch(self, events: List['SecurityEvent']) -> List[float]:
        """
        Predict anomaly scores for multiple events (efficient batch processing).

        Args:
            events: List of security events

        Returns:
            List of anomaly scores (0.0-1.0)
        """
        if not self._is_fitted or not SKLEARN_AVAILABLE:
            return [0.0] * len(events)

        # Convert to feature matrix
        X = np.array([event.to_feature_vector() for event in events])

        # Get batch scores
        raw_scores = self._model.decision_function(X)

        # Normalize to 0-1 range
        normalized_scores = [
            max(0.0, min(1.0, 0.5 - score))
            for score in raw_scores
        ]

        return normalized_scores

    def get_feature_importances(self) -> Optional[np.ndarray]:
        """
        Get feature importances (not directly supported by Isolation Forest).

        Returns:
            None (Isolation Forest doesn't provide feature importances)
        """
        # Isolation Forest doesn't provide feature importances like Random Forest
        # Would need to compute permutation importance separately
        logger.warning("Feature importances not available for Isolation Forest")
        return None
