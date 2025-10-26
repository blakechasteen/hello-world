#!/usr/bin/env python3
"""
Predictive Quality System

Predicts which queries will need retries and proactively adjusts:
- ML-based quality prediction
- Difficulty scoring
- Confidence calibration
- Preemptive configuration adjustment
- Early warning for low-quality responses

Usage:
    from HoloLoom.chatops.predictive_quality import PredictiveQualitySystem

    system = PredictiveQualitySystem()

    # Predict quality before generating
    prediction = await system.predict_quality(query, context)

    # Adjust config based on prediction
    config = system.get_optimal_config(prediction)

    # Learn from outcomes
    system.learn(query, predicted_quality, actual_quality)
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, deque
import re
import math


@dataclass
class QualityPrediction:
    """Predicted quality metrics"""
    query_id: str
    predicted_quality: float  # 0.0-1.0
    confidence: float  # 0.0-1.0
    difficulty_score: float  # 0.0-1.0
    predicted_retry_probability: float
    recommended_config: Dict[str, Any]

    # Features used
    features: Dict[str, float] = field(default_factory=dict)

    # Prediction metadata
    predicted_at: datetime = field(default_factory=datetime.now)
    model_version: str = "1.0"


@dataclass
class QueryFeatures:
    """Features extracted from query"""
    query: str

    # Length features
    char_count: int = 0
    word_count: int = 0
    sentence_count: int = 0

    # Complexity features
    avg_word_length: float = 0.0
    unique_word_ratio: float = 0.0
    technical_term_count: int = 0

    # Type features
    is_question: bool = False
    is_command: bool = False
    has_code: bool = False
    has_urls: bool = False

    # Domain features
    query_type: str = "general"
    topics: List[str] = field(default_factory=list)

    # Context features
    has_context: bool = False
    context_size: int = 0
    conversation_length: int = 0


@dataclass
class OutcomeRecord:
    """Actual outcome for learning"""
    query_id: str
    predicted_quality: float
    actual_quality: float
    needed_retry: bool
    retry_count: int
    response_time: float
    timestamp: datetime


class PredictiveQualitySystem:
    """
    ML-based system to predict query difficulty and quality.

    Predicts:
    - Expected quality score
    - Retry probability
    - Optimal configuration
    - Difficulty level
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        history_size: int = 1000
    ):
        self.storage_path = storage_path or Path("./chatops_data/predictive_quality")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.history_size = history_size

        # Historical data for learning
        self.outcome_history: deque = deque(maxlen=history_size)

        # Feature weights (learned)
        self.feature_weights = {
            "char_count": 0.1,
            "word_count": 0.15,
            "technical_term_count": 0.2,
            "unique_word_ratio": 0.15,
            "is_question": 0.1,
            "has_code": 0.1,
            "query_type_incident": 0.15,
            "conversation_length": 0.05
        }

        # Difficulty thresholds
        self.difficulty_thresholds = {
            "simple": 0.3,
            "medium": 0.6,
            "complex": 1.0
        }

        # Technical terms dictionary
        self.technical_terms = set([
            "api", "database", "server", "endpoint", "authentication",
            "deployment", "kubernetes", "docker", "ci/cd", "pipeline",
            "incident", "error", "exception", "stack trace", "logs",
            "performance", "latency", "throughput", "bottleneck",
            "security", "vulnerability", "ssl", "certificate", "oauth"
        ])

        # Statistics
        self.stats = {
            "predictions_made": 0,
            "predictions_accurate": 0,
            "avg_prediction_error": 0.0,
            "retry_predictions_correct": 0,
            "total_outcomes_recorded": 0
        }

        # Load model
        self._load_model()

        logging.info("PredictiveQualitySystem initialized")

    async def predict_quality(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List] = None
    ) -> QualityPrediction:
        """
        Predict quality metrics for a query before processing.

        Args:
            query: User query
            context: Additional context
            conversation_history: Previous messages

        Returns:
            Quality prediction with recommendations
        """

        # Extract features
        features = self._extract_features(query, context, conversation_history)

        # Calculate difficulty score
        difficulty = self._calculate_difficulty(features)

        # Predict quality using learned weights
        predicted_quality = self._predict_quality_score(features, difficulty)

        # Predict retry probability
        retry_prob = self._predict_retry_probability(features, predicted_quality)

        # Get optimal configuration
        recommended_config = self._get_optimal_config(
            predicted_quality,
            difficulty,
            retry_prob
        )

        # Calculate confidence based on feature coverage and historical accuracy
        confidence = self._calculate_confidence(features)

        # Create prediction
        prediction = QualityPrediction(
            query_id=self._generate_query_id(query),
            predicted_quality=predicted_quality,
            confidence=confidence,
            difficulty_score=difficulty,
            predicted_retry_probability=retry_prob,
            recommended_config=recommended_config,
            features=self._features_to_dict(features)
        )

        self.stats["predictions_made"] += 1

        return prediction

    def _extract_features(
        self,
        query: str,
        context: Optional[Dict],
        history: Optional[List]
    ) -> QueryFeatures:
        """Extract features from query"""

        # Length features
        char_count = len(query)
        words = query.split()
        word_count = len(words)
        sentences = re.split(r'[.!?]+', query)
        sentence_count = len([s for s in sentences if s.strip()])

        # Complexity features
        avg_word_length = sum(len(w) for w in words) / max(word_count, 1)
        unique_words = set(w.lower() for w in words)
        unique_word_ratio = len(unique_words) / max(word_count, 1)

        # Technical terms
        technical_term_count = sum(
            1 for word in words
            if word.lower() in self.technical_terms
        )

        # Type detection
        is_question = '?' in query or any(
            query.lower().startswith(w) for w in
            ['what', 'why', 'how', 'when', 'where', 'who', 'which']
        )
        is_command = query.strip().startswith('!')
        has_code = '```' in query or any(
            keyword in query.lower() for keyword in ['function', 'class', 'def ', 'const ']
        )
        has_urls = 'http://' in query or 'https://' in query

        # Domain classification
        query_type = self._classify_query_type(query)
        topics = self._extract_topics(query)

        # Context features
        has_context = context is not None and len(context) > 0
        context_size = len(str(context)) if context else 0
        conversation_length = len(history) if history else 0

        return QueryFeatures(
            query=query,
            char_count=char_count,
            word_count=word_count,
            sentence_count=sentence_count,
            avg_word_length=avg_word_length,
            unique_word_ratio=unique_word_ratio,
            technical_term_count=technical_term_count,
            is_question=is_question,
            is_command=is_command,
            has_code=has_code,
            has_urls=has_urls,
            query_type=query_type,
            topics=topics,
            has_context=has_context,
            context_size=context_size,
            conversation_length=conversation_length
        )

    def _calculate_difficulty(self, features: QueryFeatures) -> float:
        """Calculate difficulty score from features"""

        difficulty = 0.0

        # Length-based difficulty
        if features.word_count > 50:
            difficulty += 0.2
        elif features.word_count > 20:
            difficulty += 0.1

        # Complexity-based
        if features.avg_word_length > 7:
            difficulty += 0.1
        if features.technical_term_count > 5:
            difficulty += 0.2
        elif features.technical_term_count > 2:
            difficulty += 0.1

        # Type-based
        if features.has_code:
            difficulty += 0.15
        if not features.is_question and not features.is_command:
            difficulty += 0.1  # Open-ended queries harder

        # Domain-based
        if features.query_type == "incident":
            difficulty += 0.15
        elif features.query_type == "code_review":
            difficulty += 0.2

        # Context helps reduce difficulty
        if features.has_context:
            difficulty *= 0.9

        return min(difficulty, 1.0)

    def _predict_quality_score(
        self,
        features: QueryFeatures,
        difficulty: float
    ) -> float:
        """Predict quality score using features"""

        # Base score (inverse of difficulty)
        base_score = 1.0 - (difficulty * 0.5)

        # Feature-based adjustments
        feature_dict = self._features_to_dict(features)

        weighted_adjustment = 0.0
        for feature_name, weight in self.feature_weights.items():
            feature_value = feature_dict.get(feature_name, 0)

            # Normalize feature value
            if isinstance(feature_value, bool):
                feature_value = 1.0 if feature_value else 0.0
            elif isinstance(feature_value, int):
                feature_value = min(feature_value / 100, 1.0)

            weighted_adjustment += feature_value * weight

        # Combine
        predicted = base_score + weighted_adjustment * 0.2

        # Clamp to valid range
        return max(0.5, min(predicted, 0.95))

    def _predict_retry_probability(
        self,
        features: QueryFeatures,
        predicted_quality: float
    ) -> float:
        """Predict probability of needing retry"""

        # Base probability from predicted quality
        if predicted_quality >= 0.85:
            base_prob = 0.05
        elif predicted_quality >= 0.75:
            base_prob = 0.15
        elif predicted_quality >= 0.65:
            base_prob = 0.35
        else:
            base_prob = 0.55

        # Adjust based on features
        if features.query_type == "incident":
            base_prob *= 1.2
        if features.has_code:
            base_prob *= 1.15
        if not features.has_context:
            base_prob *= 1.1

        return min(base_prob, 0.9)

    def _get_optimal_config(
        self,
        predicted_quality: float,
        difficulty: float,
        retry_prob: float
    ) -> Dict[str, Any]:
        """Get optimal configuration based on prediction"""

        config = {}

        # Adjust based on predicted quality
        if predicted_quality < 0.7:
            # Low predicted quality - use stronger settings
            config["use_verification"] = True
            config["use_planning"] = True
            config["use_prompt_chaining"] = True
            config["temperature"] = 0.6  # More deterministic
        elif predicted_quality < 0.8:
            # Medium predicted quality
            config["use_verification"] = True
            config["use_planning"] = True
            config["temperature"] = 0.7
        else:
            # High predicted quality - standard settings
            config["temperature"] = 0.7

        # Adjust based on difficulty
        if difficulty > 0.7:
            config["max_tokens"] = 2048  # Allow longer responses
            config["use_verification"] = True
        elif difficulty > 0.4:
            config["max_tokens"] = 1536

        # Adjust based on retry probability
        if retry_prob > 0.4:
            config["auto_retry_threshold"] = 0.8  # Higher threshold
            config["max_retries"] = 3
        else:
            config["auto_retry_threshold"] = 0.75

        return config

    def _calculate_confidence(self, features: QueryFeatures) -> float:
        """Calculate prediction confidence"""

        confidence = 0.7  # Base confidence

        # More historical data = higher confidence
        if len(self.outcome_history) > 500:
            confidence += 0.15
        elif len(self.outcome_history) > 100:
            confidence += 0.1

        # More features = higher confidence
        feature_coverage = sum([
            features.has_context,
            features.conversation_length > 0,
            len(features.topics) > 0
        ])
        confidence += feature_coverage * 0.05

        # Historical accuracy
        if self.stats["predictions_made"] > 0:
            accuracy_rate = self.stats["predictions_accurate"] / self.stats["predictions_made"]
            confidence *= accuracy_rate

        return min(confidence, 0.95)

    def learn(
        self,
        query: str,
        predicted_quality: float,
        actual_quality: float,
        needed_retry: bool = False,
        retry_count: int = 0,
        response_time: float = 0.0
    ):
        """
        Learn from actual outcome.

        Args:
            query: Original query
            predicted_quality: What was predicted
            actual_quality: Actual quality score
            needed_retry: Whether retry was needed
            retry_count: Number of retries
            response_time: Response time in seconds
        """

        # Record outcome
        outcome = OutcomeRecord(
            query_id=self._generate_query_id(query),
            predicted_quality=predicted_quality,
            actual_quality=actual_quality,
            needed_retry=needed_retry,
            retry_count=retry_count,
            response_time=response_time,
            timestamp=datetime.now()
        )

        self.outcome_history.append(outcome)
        self.stats["total_outcomes_recorded"] += 1

        # Calculate prediction error
        error = abs(predicted_quality - actual_quality)

        # Update accuracy stats
        if error < 0.1:  # Within 10% = accurate
            self.stats["predictions_accurate"] += 1

        # Update average error
        prev_avg = self.stats["avg_prediction_error"]
        n = self.stats["total_outcomes_recorded"]
        self.stats["avg_prediction_error"] = (prev_avg * (n - 1) + error) / n

        # Update feature weights using simple gradient descent
        if len(self.outcome_history) >= 10:
            self._update_feature_weights()

        # Save periodically
        if len(self.outcome_history) % 50 == 0:
            self._save_model()

    def _update_feature_weights(self):
        """Update feature weights based on recent outcomes"""

        # Simple online learning - adjust weights based on recent errors
        recent = list(self.outcome_history)[-50:]

        for feature_name in self.feature_weights.keys():
            # Calculate correlation between feature and quality
            feature_values = []
            actual_qualities = []

            for outcome in recent:
                # Would need to store features with outcome
                # For now, skip detailed update
                pass

            # Adjust weight slightly based on performance
            # This is a simplified version - would use proper gradient descent
            learning_rate = 0.01
            error = self.stats["avg_prediction_error"]

            # Decrease weight if high error
            if error > 0.15:
                self.feature_weights[feature_name] *= (1 - learning_rate)

    def get_statistics(self) -> Dict[str, Any]:
        """Get prediction statistics"""

        stats = self.stats.copy()

        # Add derived metrics
        if stats["predictions_made"] > 0:
            stats["accuracy_rate"] = stats["predictions_accurate"] / stats["predictions_made"]

        # Recent performance
        if len(self.outcome_history) > 0:
            recent = list(self.outcome_history)[-20:]
            recent_errors = [abs(o.predicted_quality - o.actual_quality) for o in recent]
            stats["recent_avg_error"] = sum(recent_errors) / len(recent_errors)

        return stats

    def get_insights(self) -> Dict[str, Any]:
        """Get insights from prediction history"""

        if len(self.outcome_history) == 0:
            return {"message": "Not enough data yet"}

        outcomes = list(self.outcome_history)

        # Hardest query types
        type_errors = defaultdict(list)
        for outcome in outcomes:
            # Would need to store query type with outcome
            pass

        # When predictions are wrong
        large_errors = [o for o in outcomes if abs(o.predicted_quality - o.actual_quality) > 0.2]

        insights = {
            "total_predictions": len(outcomes),
            "large_errors": len(large_errors),
            "large_error_rate": len(large_errors) / len(outcomes),
            "avg_retry_rate": sum(1 for o in outcomes if o.needed_retry) / len(outcomes)
        }

        return insights

    def _classify_query_type(self, query: str) -> str:
        """Classify query type"""
        query_lower = query.lower()

        if any(word in query_lower for word in ["incident", "error", "down", "500"]):
            return "incident"
        elif any(word in query_lower for word in ["review", "pr", "code"]):
            return "code_review"
        elif any(word in query_lower for word in ["deploy", "release"]):
            return "deployment"
        else:
            return "general"

    def _extract_topics(self, query: str) -> List[str]:
        """Extract topics from query"""
        topics = []

        # Simple keyword-based extraction
        topic_keywords = {
            "authentication": ["auth", "login", "oauth", "jwt"],
            "database": ["database", "sql", "postgres", "mysql"],
            "performance": ["slow", "latency", "performance", "bottleneck"],
            "security": ["security", "vulnerability", "exploit"],
            "deployment": ["deploy", "release", "rollout"]
        }

        query_lower = query.lower()
        for topic, keywords in topic_keywords.items():
            if any(kw in query_lower for kw in keywords):
                topics.append(topic)

        return topics

    def _features_to_dict(self, features: QueryFeatures) -> Dict[str, Any]:
        """Convert features to dict"""
        return {
            "char_count": features.char_count,
            "word_count": features.word_count,
            "sentence_count": features.sentence_count,
            "avg_word_length": features.avg_word_length,
            "unique_word_ratio": features.unique_word_ratio,
            "technical_term_count": features.technical_term_count,
            "is_question": features.is_question,
            "is_command": features.is_command,
            "has_code": features.has_code,
            "has_urls": features.has_urls,
            "query_type_incident": features.query_type == "incident",
            "has_context": features.has_context,
            "context_size": features.context_size,
            "conversation_length": features.conversation_length
        }

    def _generate_query_id(self, query: str) -> str:
        """Generate unique query ID"""
        return f"query_{hash(query) % 1000000}_{datetime.now().strftime('%H%M%S')}"

    def _save_model(self):
        """Save model weights and history"""
        model_file = self.storage_path / "model.json"

        model_data = {
            "feature_weights": self.feature_weights,
            "stats": self.stats,
            "version": "1.0",
            "last_updated": datetime.now().isoformat()
        }

        with open(model_file, 'w') as f:
            json.dump(model_data, f, indent=2)

    def _load_model(self):
        """Load model weights"""
        model_file = self.storage_path / "model.json"

        if model_file.exists():
            try:
                with open(model_file, 'r') as f:
                    model_data = json.load(f)

                self.feature_weights = model_data.get("feature_weights", self.feature_weights)
                self.stats = model_data.get("stats", self.stats)

                logging.info("Loaded predictive quality model")
            except Exception as e:
                logging.error(f"Failed to load model: {e}")


# Demo
async def demo_predictive_quality():
    """Demonstrate predictive quality system"""

    print("🔮 Predictive Quality System Demo\n")

    system = PredictiveQualitySystem()

    # Test queries
    test_queries = [
        ("What is the API endpoint for user authentication?", "Simple question"),
        ("The production API is returning 500 errors with stack traces showing database connection timeouts", "Complex incident"),
        ("Review this code for security vulnerabilities:\n```python\ndef login(username, password):\n    query = f\"SELECT * FROM users WHERE username='{username}'\"\n```", "Code review with security issue"),
        ("How do I deploy?", "Vague question")
    ]

    for query, description in test_queries:
        print(f"Query: {description}")
        print(f"  Text: {query[:80]}...")

        prediction = await system.predict_quality(query)

        print(f"  Predicted Quality: {prediction.predicted_quality:.2f}")
        print(f"  Confidence: {prediction.confidence:.2f}")
        print(f"  Difficulty: {prediction.difficulty_score:.2f}")
        print(f"  Retry Probability: {prediction.predicted_retry_probability:.1%}")
        print(f"  Recommended Config: {prediction.recommended_config}")
        print()

        # Simulate learning
        actual_quality = prediction.predicted_quality + (hash(query) % 20 - 10) / 100
        system.learn(query, prediction.predicted_quality, actual_quality)

    # Statistics
    print("Statistics:")
    stats = system.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(demo_predictive_quality())
