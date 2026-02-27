"""
Feature Extraction for ML Routing

Extracts features from queries and context for ML model input.

Features extracted:
- Query linguistic features (length, complexity, keywords)
- Semantic features (embeddings, topics)
- Context features (user, session, history)
- Temporal features (time of day, day of week)

Author: HoloLoom B2B Framework
Date: November 2025
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import re


@dataclass
class RoutingFeatures:
    """
    Features extracted for routing ML model.

    Contains all features used for department prediction.
    """
    # Query linguistic features
    query_length: int
    word_count: int
    avg_word_length: float
    question_marks: int
    exclamation_marks: int

    # Complexity features
    complexity_score: float  # 0.0-1.0

    # Keyword features
    keywords: List[str] = field(default_factory=list)

    # Semantic features
    query_embedding: Optional[List[float]] = None
    topic_distribution: Optional[Dict[str, float]] = None

    # Context features
    user_role: Optional[str] = None
    session_length: int = 0
    query_history_size: int = 0

    # Temporal features
    hour_of_day: int = 0
    day_of_week: int = 0

    # Derived features
    feature_dict: Dict[str, Any] = field(default_factory=dict)


class RoutingFeatureExtractor:
    """
    Extract features from queries and context for ML routing.

    Example:
        >>> extractor = RoutingFeatureExtractor()
        >>> features = extractor.extract(
        ...     query="What is machine learning?",
        ...     context={"user_role": "student", "session_id": "s123"}
        ... )
        >>> print(f"Complexity: {features.complexity_score:.2f}")
        >>> print(f"Keywords: {features.keywords}")
    """

    def __init__(self):
        """Initialize feature extractor"""
        # Keyword categories for department routing
        self.keyword_categories = {
            "rag": ["what", "explain", "define", "describe", "tell me about"],
            "planning": ["plan", "strategy", "roadmap", "goal", "objective"],
            "orchestration": ["coordinate", "workflow", "pipeline", "sequence"],
            "infrastructure": ["deploy", "monitor", "scale", "resource", "performance"],
            "context": ["who", "when", "where", "history", "session"]
        }

    def extract(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> RoutingFeatures:
        """
        Extract features from query and context.

        Args:
            query: Query text
            context: Additional context (user, session, etc.)

        Returns:
            RoutingFeatures with all extracted features
        """
        # Linguistic features
        query_length = len(query)
        words = query.split()
        word_count = len(words)
        avg_word_length = sum(len(w) for w in words) / max(word_count, 1)
        question_marks = query.count("?")
        exclamation_marks = query.count("!")

        # Complexity score
        complexity = self._calculate_complexity(query, words)

        # Keywords
        keywords = self._extract_keywords(query)

        # Context features
        user_role = context.get("user_role")
        session_length = context.get("session_length", 0)
        query_history_size = len(context.get("history", []))

        # Temporal features
        import datetime
        now = datetime.datetime.now()
        hour_of_day = now.hour
        day_of_week = now.weekday()

        # Build feature dict for model
        feature_dict = {
            "query_length": query_length,
            "word_count": word_count,
            "avg_word_length": avg_word_length,
            "question_marks": question_marks,
            "exclamation_marks": exclamation_marks,
            "complexity_score": complexity,
            "has_keywords": len(keywords) > 0,
            "user_role": user_role,
            "session_length": session_length,
            "query_history_size": query_history_size,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week
        }

        return RoutingFeatures(
            query_length=query_length,
            word_count=word_count,
            avg_word_length=avg_word_length,
            question_marks=question_marks,
            exclamation_marks=exclamation_marks,
            complexity_score=complexity,
            keywords=keywords,
            user_role=user_role,
            session_length=session_length,
            query_history_size=query_history_size,
            hour_of_day=hour_of_day,
            day_of_week=day_of_week,
            feature_dict=feature_dict
        )

    def _calculate_complexity(self, query: str, words: List[str]) -> float:
        """
        Calculate query complexity score (0.0-1.0).

        Factors:
        - Query length
        - Word count
        - Rare words (simple heuristic: word length > 10)
        - Sentence structure

        Args:
            query: Query text
            words: Tokenized words

        Returns:
            Complexity score 0.0 (simple) to 1.0 (complex)
        """
        complexity = 0.0

        # Length factor (normalized to 0-0.3)
        length_score = min(len(query) / 200.0, 1.0) * 0.3
        complexity += length_score

        # Word count factor (normalized to 0-0.2)
        word_count_score = min(len(words) / 30.0, 1.0) * 0.2
        complexity += word_count_score

        # Rare/complex words (normalized to 0-0.3)
        rare_words = sum(1 for w in words if len(w) > 10)
        rare_word_score = min(rare_words / 5.0, 1.0) * 0.3
        complexity += rare_word_score

        # Multi-clause structure (normalized to 0-0.2)
        clause_markers = query.count(",") + query.count(";") + query.count("and") + query.count("or")
        clause_score = min(clause_markers / 5.0, 1.0) * 0.2
        complexity += clause_score

        return min(complexity, 1.0)

    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract routing-relevant keywords from query.

        Args:
            query: Query text

        Returns:
            List of detected keywords
        """
        query_lower = query.lower()
        keywords = []

        for category, category_keywords in self.keyword_categories.items():
            for keyword in category_keywords:
                if keyword in query_lower:
                    keywords.append(keyword)

        return keywords

    def extract_semantic_features(
        self,
        query: str,
        embedding_model: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Extract semantic features (embeddings, topics).

        Args:
            query: Query text
            embedding_model: Optional embedding model

        Returns:
            Dict with semantic features
        """
        # TODO: Implement semantic feature extraction
        # For now, return placeholder
        return {
            "query_embedding": None,  # Would use sentence-transformers
            "topic_distribution": None  # Would use topic modeling
        }
