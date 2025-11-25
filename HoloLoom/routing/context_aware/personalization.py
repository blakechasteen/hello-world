"""
Personalization Engine

Learns user preferences and adapts routing decisions accordingly.

Features:
- User profile management
- Preference learning from feedback
- Collaborative filtering (similar users)
- Cold start handling

Author: HoloLoom B2B Framework
Date: November 2025
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from collections import defaultdict
import numpy as np


@dataclass
class UserProfile:
    """User profile with learned preferences"""
    user_id: str
    preferred_departments: Dict[str, float] = field(default_factory=dict)  # dept → weight
    query_patterns: List[str] = field(default_factory=list)  # Common query types
    avg_confidence: float = 0.0
    total_queries: int = 0
    successful_queries: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class PersonalizationEngine:
    """
    Personalization engine for context-aware routing.

    Learns user preferences and adapts routing decisions.

    Example:
        >>> engine = PersonalizationEngine()
        >>> engine.update_preference("user123", "rag", outcome="success", confidence=0.92)
        >>> profile = engine.get_profile("user123")
        >>> print(profile.preferred_departments)  # {"rag": 0.05}
    """

    def __init__(self, enable_collaborative_filtering: bool = False):
        """
        Initialize personalization engine.

        Args:
            enable_collaborative_filtering: Enable collaborative filtering (similar users)
        """
        self.enable_collaborative_filtering = enable_collaborative_filtering

        # User profiles
        self._profiles: Dict[str, UserProfile] = {}

        # Collaborative filtering matrix (user × department)
        self._user_dept_matrix: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

    def get_profile(self, user_id: str) -> UserProfile:
        """
        Get or create user profile.

        Args:
            user_id: User ID

        Returns:
            UserProfile
        """
        if user_id not in self._profiles:
            self._profiles[user_id] = UserProfile(user_id=user_id)

        return self._profiles[user_id]

    def update_preference(
        self,
        user_id: str,
        department: str,
        outcome: str,  # "success" | "failure"
        confidence: float,
    ):
        """
        Update user preference based on routing outcome.

        Args:
            user_id: User ID
            department: Department that was used
            outcome: Outcome of query
            confidence: Confidence of outcome
        """
        profile = self.get_profile(user_id)

        # Update stats
        profile.total_queries += 1
        if outcome == "success":
            profile.successful_queries += 1

        # Update average confidence
        n = profile.total_queries
        profile.avg_confidence = (profile.avg_confidence * (n - 1) + confidence) / n

        # Update department preference
        if outcome == "success":
            # Increase preference (learning rate: 0.05)
            current = profile.preferred_departments.get(department, 0.0)
            profile.preferred_departments[department] = min(0.5, current + 0.05)
        else:
            # Decrease preference
            current = profile.preferred_departments.get(department, 0.0)
            profile.preferred_departments[department] = max(-0.2, current - 0.05)

        # Update collaborative filtering matrix
        if self.enable_collaborative_filtering:
            self._user_dept_matrix[user_id][department] = profile.preferred_departments[
                department
            ]

    def get_recommendation(
        self, user_id: str, top_k: int = 3
    ) -> List[tuple[str, float]]:
        """
        Get top-k department recommendations for user.

        Uses collaborative filtering if enabled, otherwise just user's own preferences.

        Args:
            user_id: User ID
            top_k: Number of recommendations

        Returns:
            List of (department, score) tuples
        """
        profile = self.get_profile(user_id)

        if self.enable_collaborative_filtering and profile.total_queries < 10:
            # Cold start: use collaborative filtering
            return self._collaborative_recommendation(user_id, top_k)
        else:
            # Use user's own preferences
            preferences = profile.preferred_departments
            sorted_prefs = sorted(preferences.items(), key=lambda x: x[1], reverse=True)
            return sorted_prefs[:top_k]

    def _collaborative_recommendation(
        self, user_id: str, top_k: int
    ) -> List[tuple[str, float]]:
        """
        Collaborative filtering recommendations.

        Finds similar users and recommends departments they prefer.

        Args:
            user_id: User ID
            top_k: Number of recommendations

        Returns:
            List of (department, score) tuples
        """
        # Find similar users (cosine similarity)
        user_vector = self._user_dept_matrix.get(user_id, {})

        similarities = []
        for other_user_id, other_vector in self._user_dept_matrix.items():
            if other_user_id == user_id:
                continue

            # Cosine similarity
            sim = self._cosine_similarity(user_vector, other_vector)
            if sim > 0.5:  # Threshold for similarity
                similarities.append((other_user_id, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Aggregate department scores from similar users
        dept_scores: Dict[str, float] = defaultdict(float)
        total_weight = 0.0

        for other_user_id, similarity in similarities[:10]:  # Top 10 similar users
            other_profile = self.get_profile(other_user_id)
            for dept, score in other_profile.preferred_departments.items():
                dept_scores[dept] += score * similarity
                total_weight += similarity

        # Normalize scores
        if total_weight > 0:
            for dept in dept_scores:
                dept_scores[dept] /= total_weight

        # Sort and return top-k
        sorted_depts = sorted(dept_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_depts[:top_k]

    def _cosine_similarity(
        self, vec1: Dict[str, float], vec2: Dict[str, float]
    ) -> float:
        """
        Compute cosine similarity between two department preference vectors.

        Args:
            vec1: First vector (department → score)
            vec2: Second vector (department → score)

        Returns:
            Cosine similarity (0.0-1.0)
        """
        # Get all departments
        all_depts = set(vec1.keys()) | set(vec2.keys())

        if not all_depts:
            return 0.0

        # Convert to numpy arrays
        v1 = np.array([vec1.get(dept, 0.0) for dept in all_depts])
        v2 = np.array([vec2.get(dept, 0.0) for dept in all_depts])

        # Cosine similarity
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def get_metrics(self) -> Dict[str, Any]:
        """Get personalization metrics"""
        total_users = len(self._profiles)
        active_users = sum(1 for p in self._profiles.values() if p.total_queries > 0)

        avg_queries_per_user = (
            sum(p.total_queries for p in self._profiles.values()) / total_users
            if total_users > 0
            else 0
        )

        avg_success_rate = (
            sum(
                p.successful_queries / p.total_queries if p.total_queries > 0 else 0
                for p in self._profiles.values()
            )
            / total_users
            if total_users > 0
            else 0
        )

        return {
            "total_users": total_users,
            "active_users": active_users,
            "avg_queries_per_user": avg_queries_per_user,
            "avg_success_rate": avg_success_rate,
            "collaborative_filtering_enabled": self.enable_collaborative_filtering,
        }
