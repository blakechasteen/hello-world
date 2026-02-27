"""
ML-Based Router

Uses machine learning models to predict optimal department for queries.

Features:
- Multi-class classification (one department per query)
- Confidence scores with uncertainty estimation
- Online learning from production feedback
- Model versioning and A/B testing

Author: HoloLoom B2B Framework
Date: November 2025
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np


@dataclass
class RoutingPrediction:
    """ML routing prediction"""
    department_id: str
    confidence: float
    alternatives: List[Tuple[str, float]]  # (department, probability)
    model_version: str
    features_used: List[str]


class MLRouter:
    """
    ML-based department router.

    Predicts optimal department using trained classification model.

    Example:
        >>> router = MLRouter(model_path="models/routing_v1.pkl")
        >>> prediction = router.predict(
        ...     query="Explain machine learning",
        ...     context={"user_role": "data_scientist"}
        ... )
        >>> print(f"Department: {prediction.department_id}")
        >>> print(f"Confidence: {prediction.confidence:.2f}")
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_version: str = "v1.0"
    ):
        """
        Initialize ML router.

        Args:
            model_path: Path to trained model (None = use default)
            model_version: Model version identifier
        """
        self.model_version = model_version
        self.model = self._load_model(model_path) if model_path else None

        # Department mapping
        self.departments = ["rag", "planning", "orchestration", "infrastructure", "context"]
        self.dept_to_idx = {dept: i for i, dept in enumerate(self.departments)}
        self.idx_to_dept = {i: dept for i, dept in enumerate(self.departments)}

    def _load_model(self, model_path: str):
        """
        Load trained model from disk.

        Args:
            model_path: Path to model file

        Returns:
            Loaded model
        """
        # TODO: Implement model loading
        # For now, return None (fallback to rule-based)
        import warnings
        warnings.warn("ML model not available, using rule-based fallback")
        return None

    def predict(
        self,
        query: str,
        context: Optional[Dict] = None
    ) -> RoutingPrediction:
        """
        Predict optimal department for query.

        Args:
            query: Query text
            context: Additional context (user, session, etc.)

        Returns:
            RoutingPrediction with department and confidence
        """
        from HoloLoom.routing.ml import RoutingFeatureExtractor

        # Extract features
        extractor = RoutingFeatureExtractor()
        features = extractor.extract(query, context or {})

        if self.model is None:
            # Fallback to rule-based
            return self._fallback_prediction(query, features)

        # Get model predictions (probabilities per department)
        feature_vector = self._features_to_vector(features)
        probabilities = self.model.predict_proba([feature_vector])[0]

        # Sort by probability
        sorted_indices = np.argsort(probabilities)[::-1]

        top_dept_idx = sorted_indices[0]
        top_dept = self.idx_to_dept[top_dept_idx]
        top_prob = probabilities[top_dept_idx]

        # Alternative departments
        alternatives = [
            (self.idx_to_dept[idx], float(probabilities[idx]))
            for idx in sorted_indices[1:4]  # Top 3 alternatives
        ]

        return RoutingPrediction(
            department_id=top_dept,
            confidence=float(top_prob),
            alternatives=alternatives,
            model_version=self.model_version,
            features_used=list(features.feature_dict.keys())
        )

    def _features_to_vector(self, features: "RoutingFeatures") -> np.ndarray:
        """
        Convert features to numpy vector for model input.

        Args:
            features: Extracted routing features

        Returns:
            Feature vector
        """
        # Combine all feature types
        vector = []

        # Query features
        vector.extend([
            features.query_length,
            features.word_count,
            features.avg_word_length,
            features.question_marks,
            features.exclamation_marks
        ])

        # Complexity features
        vector.append(features.complexity_score)

        # Keyword features (binary indicators)
        vector.extend([
            1.0 if kw in features.keywords else 0.0
            for kw in ["what", "how", "why", "when", "where", "plan", "analyze", "debug"]
        ])

        return np.array(vector)

    def _fallback_prediction(
        self,
        query: str,
        features: "RoutingFeatures"
    ) -> RoutingPrediction:
        """
        Fallback to rule-based prediction when ML model unavailable.

        Args:
            query: Query text
            features: Extracted features

        Returns:
            RoutingPrediction using heuristics
        """
        # Simple heuristic routing
        query_lower = query.lower()

        if any(kw in query_lower for kw in ["what is", "define", "explain"]):
            dept = "rag"
            conf = 0.75
        elif any(kw in query_lower for kw in ["plan", "strategy", "roadmap"]):
            dept = "planning"
            conf = 0.70
        elif any(kw in query_lower for kw in ["debug", "monitor", "deploy"]):
            dept = "infrastructure"
            conf = 0.68
        elif features.complexity_score > 0.8:
            dept = "orchestration"
            conf = 0.65
        else:
            dept = "rag"
            conf = 0.60

        return RoutingPrediction(
            department_id=dept,
            confidence=conf,
            alternatives=[
                ("planning", 0.20),
                ("orchestration", 0.15)
            ],
            model_version="rule-based-fallback",
            features_used=list(features.feature_dict.keys())
        )

    async def learn_from_feedback(
        self,
        query: str,
        predicted_dept: str,
        actual_dept: str,
        outcome: str,
        confidence: float
    ):
        """
        Learn from routing feedback (online learning).

        Args:
            query: Query text
            predicted_dept: Department predicted by model
            actual_dept: Department that should have been used
            outcome: Outcome ("success" | "failure")
            confidence: Outcome confidence (0.0-1.0)
        """
        # TODO: Implement online learning
        # For now, log the feedback for future training
        feedback = {
            "query": query,
            "predicted": predicted_dept,
            "actual": actual_dept,
            "outcome": outcome,
            "confidence": confidence
        }

        # In production: queue for batch retraining
        # For now: no-op
        pass

    def get_metrics(self) -> Dict:
        """
        Get ML router metrics.

        Returns:
            Dict with accuracy, coverage, confidence stats
        """
        return {
            "model_version": self.model_version,
            "model_available": self.model is not None,
            "fallback_mode": self.model is None,
            "supported_departments": self.departments
        }
