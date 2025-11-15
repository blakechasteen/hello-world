"""
Adaptive Difficulty Engine

Thompson Sampling for optimal challenge selection.

**Key Algorithm**: Thompson Sampling (Bayesian bandit)
- Each objective = bandit arm
- Reward = student_success × engagement
- Sample from Beta(α, β) distributions
- Provably optimal regret bounds: O(√T)

**Implementation Date**: November 15, 2025
"""

from typing import List, Dict, Tuple
import random

try:
    from HoloLoom.policy.thompson_sampling import TSBandit
    HAS_THOMPSON = True
except ImportError:
    HAS_THOMPSON = False
    print("⚠️  Thompson Sampling not available - using random selection")

from .curriculum_graph import EdWINKnowledgeGraph, LearningObjective
from .student_model import EdWINStudentModel


class AdaptiveDifficultyEngine:
    """
    Thompson Sampling for Optimal Challenge

    Selects next learning objective to balance:
    - Exploration: Try new topics
    - Exploitation: Reinforce successful topics
    - Optimal challenge: difficulty ≈ current_skill + 0.1 (zone of proximal development)

    Algorithm:
    1. Estimate difficulty for each available objective
    2. Calculate optimal challenge (slight above current skill)
    3. Thompson sample to balance exploration/exploitation
    4. Return best objective

    Example:
        >>> engine = AdaptiveDifficultyEngine(student_model, curriculum_kg)
        >>>
        >>> # Select next objective
        >>> next_obj = engine.select_next_objective(
        ...     available_objectives=unlocked_objs
        ... )
        >>>
        >>> print(f"Recommended: {next_obj.title}")
        >>> print(f"Difficulty: {engine.estimate_difficulty(next_obj.id):.2f}")
    """

    def __init__(
        self,
        student_model: EdWINStudentModel,
        curriculum_kg: EdWINKnowledgeGraph,
        n_arms: int = 100
    ):
        """
        Initialize adaptive difficulty engine

        Args:
            student_model: Student profile
            curriculum_kg: Curriculum knowledge graph
            n_arms: Number of bandit arms (max candidate objectives)
        """
        self.student = student_model
        self.curriculum_kg = curriculum_kg
        self.n_arms = n_arms

        # Thompson Sampling bandit
        if HAS_THOMPSON:
            self.bandit = TSBandit(n_arms=n_arms)
        else:
            self.bandit = None

        # Track objective → arm mapping
        self.objective_to_arm: Dict[str, int] = {}
        self.arm_to_objective: Dict[int, str] = {}

        # Difficulty cache
        self._difficulty_cache: Dict[str, float] = {}

    def select_next_objective(
        self,
        available_objectives: List[LearningObjective]
    ) -> LearningObjective:
        """
        Select optimal next objective using Thompson Sampling

        Steps:
        1. Estimate difficulty for each available objective
        2. Calculate optimal challenge score
        3. Thompson sample (if available) or use heuristic
        4. Return best objective

        Args:
            available_objectives: List of unlocked objectives

        Returns:
            Selected learning objective
        """
        if not available_objectives:
            raise ValueError("No available objectives to select from")

        # Ensure objectives are registered in bandit
        self._register_objectives(available_objectives)

        # Calculate scores for each objective
        scored_objectives = []

        for obj in available_objectives:
            # Estimate difficulty
            difficulty = self.estimate_difficulty(obj.id)

            # Calculate optimal challenge score
            # Optimal = difficulty is slightly above current skill
            current_skill = self._estimate_current_skill()
            optimal_challenge = 1.0 - abs(difficulty - (current_skill + 0.1))

            # Thompson sample (if available)
            if HAS_THOMPSON and self.bandit:
                arm_idx = self.objective_to_arm[obj.id]
                thompson_score = self.bandit.sample_arm()
            else:
                # Fallback: random exploration
                thompson_score = random.random()

            # Combined score
            # 70% Thompson Sampling, 30% optimal challenge
            combined_score = 0.7 * thompson_score + 0.3 * optimal_challenge

            scored_objectives.append((combined_score, obj))

        # Select best
        scored_objectives.sort(key=lambda x: x[0], reverse=True)
        selected_obj = scored_objectives[0][1]

        return selected_obj

    def estimate_difficulty(self, objective_id: str) -> float:
        """
        Estimate difficulty of an objective

        Formula:
        difficulty = (0.5 * bloom_level / 6)  # Normalize to 0-1
                   + (0.3 * prerequisite_depth / max_depth)
                   + (0.2 * grade_delta / max_grade_delta)

        Args:
            objective_id: Learning objective ID

        Returns:
            Float 0.0 (easiest) - 1.0 (hardest)
        """
        # Check cache
        if objective_id in self._difficulty_cache:
            return self._difficulty_cache[objective_id]

        obj = self.curriculum_kg.get_objective(objective_id)
        if not obj:
            return 0.5  # Default

        # Component 1: Bloom level (0.5 weight)
        bloom_component = obj.bloom_level.value / 6.0  # Normalize to 0-1

        # Component 2: Prerequisite depth (0.3 weight)
        prereqs = self.curriculum_kg.get_prerequisites(objective_id)
        max_depth = 10  # Assume max 10 levels of prerequisites
        prereq_component = min(len(prereqs) / max_depth, 1.0)

        # Component 3: Grade delta (0.2 weight)
        grade_delta = abs(obj.grade_level - self.student.grade)
        max_grade_delta = 4  # Max 4 grades above/below
        grade_component = min(grade_delta / max_grade_delta, 1.0)

        # Combined difficulty
        difficulty = (
            0.5 * bloom_component +
            0.3 * prereq_component +
            0.2 * grade_component
        )

        # Cache result
        self._difficulty_cache[objective_id] = difficulty

        return difficulty

    def update_after_interaction(
        self,
        objective_id: str,
        success: bool,
        engagement: float
    ):
        """
        Update Thompson Sampling priors after student interaction

        Args:
            objective_id: Objective that was attempted
            success: Whether student succeeded
            engagement: Engagement score (0.0-1.0)
        """
        if not HAS_THOMPSON or not self.bandit:
            return

        # Calculate reward
        reward = (1.0 if success else 0.0) * engagement

        # Update bandit
        if objective_id in self.objective_to_arm:
            arm_idx = self.objective_to_arm[objective_id]
            self.bandit.update(arm_idx, reward)

    def expected_reward(self, objective_id: str) -> float:
        """
        Get expected reward for an objective

        Returns:
            Expected reward (0.0-1.0)
        """
        if not HAS_THOMPSON or not self.bandit:
            return 0.5

        if objective_id in self.objective_to_arm:
            arm_idx = self.objective_to_arm[objective_id]
            stats = self.bandit.get_stats()
            alpha = stats["alpha"][arm_idx]
            beta = stats["beta"][arm_idx]
            return alpha / (alpha + beta)

        return 0.5  # Default (uniform prior)

    def _register_objectives(self, objectives: List[LearningObjective]):
        """Register objectives in bandit (assign arm indices)"""
        for obj in objectives:
            if obj.id not in self.objective_to_arm:
                # Assign next available arm
                arm_idx = len(self.objective_to_arm) % self.n_arms
                self.objective_to_arm[obj.id] = arm_idx
                self.arm_to_objective[arm_idx] = obj.id

    def _estimate_current_skill(self) -> float:
        """
        Estimate student's current skill level

        Based on:
        - Average mastery scores
        - Bloom levels of mastered objectives
        - Grade progression

        Returns:
            Float 0.0-1.0
        """
        if not self.student.mastery_scores:
            return 0.0

        # Average mastery across all objectives
        avg_mastery = sum(self.student.mastery_scores.values()) / len(self.student.mastery_scores)

        # Average Bloom level of mastered objectives
        mastered_objs = [
            self.curriculum_kg.get_objective(obj_id)
            for obj_id in self.student.mastered_objectives
        ]
        mastered_objs = [obj for obj in mastered_objs if obj]

        if mastered_objs:
            avg_bloom = sum(obj.bloom_level.value for obj in mastered_objs) / len(mastered_objs)
            avg_bloom_normalized = avg_bloom / 6.0  # Normalize to 0-1
        else:
            avg_bloom_normalized = 0.0

        # Combined skill estimate
        skill = (0.6 * avg_mastery) + (0.4 * avg_bloom_normalized)

        return skill

    def get_statistics(self) -> Dict:
        """
        Get engine statistics

        Returns:
            Dict with statistics
        """
        if HAS_THOMPSON and self.bandit:
            bandit_stats = self.bandit.get_stats()
        else:
            bandit_stats = {}

        return {
            "current_skill_estimate": self._estimate_current_skill(),
            "difficulty_preference": self.student.difficulty_preference,
            "objectives_tracked": len(self.objective_to_arm),
            "bandit_stats": bandit_stats
        }


# Helper functions

def create_adaptive_engine(
    student_model: EdWINStudentModel,
    curriculum_kg: EdWINKnowledgeGraph
) -> AdaptiveDifficultyEngine:
    """
    Create adaptive difficulty engine

    Args:
        student_model: Student profile
        curriculum_kg: Curriculum knowledge graph

    Returns:
        AdaptiveDifficultyEngine
    """
    return AdaptiveDifficultyEngine(
        student_model=student_model,
        curriculum_kg=curriculum_kg
    )
