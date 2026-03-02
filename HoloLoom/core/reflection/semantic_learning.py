"""
Semantic Trajectory Learning
=============================
Extracts maximum learning signal from 244D semantic trajectories for policy training.

Core Insight:
------------
Semantic trajectories are RICH supervisory signals. Beyond scalar rewards, we can learn:
1. Which semantic dimensions predict success
2. How different tools affect semantic state
3. Optimal semantic paths for different goals
4. Tool -> Semantic Effect mappings
5. Semantic failure modes and recovery

This is "blob->job" conversion: turning rich semantic blobs into actionable training jobs.

Philosophy:
----------
"Every semantic trajectory is a lesson in what worked and what didn't, IF we know how to read it."

The 244D semantic space provides orders of magnitude more information than scalar rewards.
We extract multi-task learning signals, contrastive pairs, auxiliary predictions, and
dense curriculum sequences from semantic trajectories.

Author: Claude Code (with HoloLoom by Blake)
Date: 2025-10-27
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque

from HoloLoom.fabric.spacetime import Spacetime
from HoloLoom.semantic_calculus.dimensions import EXTENDED_244_DIMENSIONS

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SemanticLearningConfig:
    """Configuration for semantic trajectory learning."""

    # Multi-task learning
    enable_dimension_prediction: bool = True  # Predict semantic changes
    enable_tool_effect_learning: bool = True  # Learn tool->semantic mappings
    enable_goal_achievement: bool = True      # Predict goal achievement

    # Contrastive learning
    enable_contrastive_pairs: bool = True     # Learn from success/failure pairs
    contrastive_margin: float = 0.5           # Margin for contrastive loss

    # Auxiliary tasks
    enable_trajectory_forecasting: bool = True  # Predict future semantic states
    forecast_horizon: int = 3                    # How many steps ahead

    # Curriculum learning
    enable_semantic_curriculum: bool = True    # Stage goals by difficulty
    curriculum_stages: int = 5                 # Number of difficulty stages

    # Weights for multi-task losses
    dimension_prediction_weight: float = 0.2
    tool_effect_weight: float = 0.2
    goal_achievement_weight: float = 0.2
    contrastive_weight: float = 0.2
    forecasting_weight: float = 0.1
    policy_weight: float = 0.3  # Main policy loss


# ============================================================================
# Semantic Experience Structure
# ============================================================================

@dataclass
class SemanticExperience:
    """
    Rich experience tuple with semantic trajectory information.

    This is the "blob" - packed with learning signals.
    """
    # Standard RL tuple
    observation: Dict[str, Any]
    action: str
    reward: float
    next_observation: Dict[str, Any]
    done: bool

    # Semantic trajectory
    semantic_state: Dict[str, float]         # 244D position
    semantic_velocity: Dict[str, float]      # Rate of change
    semantic_categories: Dict[str, float]    # 16 categories

    next_semantic_state: Dict[str, float]    # After action
    next_semantic_velocity: Dict[str, float]
    next_semantic_categories: Dict[str, float]

    # Semantic goal context
    semantic_goal: Optional[Dict[str, float]] = None  # Target dimensions
    goal_alignment_before: float = 0.0
    goal_alignment_after: float = 0.0

    # Tool effect signature
    tool_semantic_delta: Dict[str, float] = field(default_factory=dict)  # Change per dimension

    # Metadata
    query_text: str = ""
    response_text: str = ""
    confidence: float = 0.0
    success: bool = False  # Did user accept/like response?

    # Trajectory context (for forecasting)
    semantic_history: List[Dict[str, float]] = field(default_factory=list)  # Past 5 states


# ============================================================================
# Semantic Trajectory Analyzer
# ============================================================================

class SemanticTrajectoryAnalyzer:
    """
    Analyzes semantic trajectories to extract learning signals.

    Converts semantic "blobs" into actionable "jobs" for the policy.
    """

    def __init__(self, config: SemanticLearningConfig):
        self.config = config

        # Tool effect statistics: tool -> dimension -> effect
        self.tool_semantic_effects: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # Success patterns: what semantic states lead to success?
        self.successful_semantic_states: List[Dict[str, float]] = []
        self.failed_semantic_states: List[Dict[str, float]] = []

        # Goal achievement patterns
        self.goal_trajectories: List[Dict[str, Any]] = []

        logger.info("SemanticTrajectoryAnalyzer initialized")

    def analyze_experience(
        self,
        experience: SemanticExperience
    ) -> Dict[str, Any]:
        """
        Analyze a semantic experience to extract learning signals.

        Args:
            experience: Rich semantic experience

        Returns:
            Dict with extracted learning signals
        """
        signals = {}

        # 1. Analyze tool semantic effect
        if self.config.enable_tool_effect_learning:
            tool_effect = self._analyze_tool_effect(experience)
            signals['tool_effect'] = tool_effect

        # 2. Analyze goal alignment change
        if self.config.enable_goal_achievement and experience.semantic_goal:
            goal_signal = self._analyze_goal_progress(experience)
            signals['goal_progress'] = goal_signal

        # 3. Identify semantic success patterns
        if experience.success:
            self.successful_semantic_states.append(experience.semantic_state)
        else:
            self.failed_semantic_states.append(experience.semantic_state)

        # 4. Extract dimension importance
        dimension_importance = self._compute_dimension_importance(experience)
        signals['dimension_importance'] = dimension_importance

        # 5. Detect semantic anomalies
        anomalies = self._detect_semantic_anomalies(experience)
        if anomalies:
            signals['anomalies'] = anomalies

        return signals

    def _analyze_tool_effect(
        self,
        experience: SemanticExperience
    ) -> Dict[str, float]:
        """
        Analyze how the tool affected semantic state.

        Returns:
            Dict mapping dimension -> effect magnitude
        """
        tool = experience.action
        effects = {}

        for dim_name in experience.semantic_state.keys():
            before = experience.semantic_state.get(dim_name, 0.0)
            after = experience.next_semantic_state.get(dim_name, 0.0)
            delta = after - before

            effects[dim_name] = delta

            # Update statistics
            self.tool_semantic_effects[tool][dim_name].append(delta)

        return effects

    def _analyze_goal_progress(
        self,
        experience: SemanticExperience
    ) -> Dict[str, Any]:
        """
        Analyze progress toward semantic goal.

        Returns:
            Dict with goal progress metrics
        """
        alignment_delta = (
            experience.goal_alignment_after - experience.goal_alignment_before
        )

        # Which dimensions improved most?
        dimension_improvements = {}
        for dim_name, target_value in experience.semantic_goal.items():
            before = experience.semantic_state.get(dim_name, 0.0)
            after = experience.next_semantic_state.get(dim_name, 0.0)

            dist_before = abs(before - target_value)
            dist_after = abs(after - target_value)
            improvement = dist_before - dist_after

            dimension_improvements[dim_name] = improvement

        return {
            'alignment_delta': alignment_delta,
            'dimension_improvements': dimension_improvements,
            'achieved': alignment_delta > 0,
            'magnitude': abs(alignment_delta)
        }

    def _compute_dimension_importance(
        self,
        experience: SemanticExperience
    ) -> Dict[str, float]:
        """
        Compute which dimensions were most important for this interaction.

        Based on:
        1. Magnitude of semantic velocity (active dimensions)
        2. Correlation with success/failure
        3. Alignment with goals
        """
        importance = {}

        for dim_name, velocity in experience.semantic_velocity.items():
            # Base importance: velocity magnitude
            base_importance = abs(velocity)

            # Boost if aligned with goal
            goal_boost = 0.0
            if experience.semantic_goal and dim_name in experience.semantic_goal:
                target = experience.semantic_goal[dim_name]
                current = experience.semantic_state.get(dim_name, 0.0)
                # Higher boost if moving toward target
                if (velocity > 0 and current < target) or (velocity < 0 and current > target):
                    goal_boost = 0.5

            # Boost if correlated with success
            success_boost = 0.3 if experience.success else 0.0

            importance[dim_name] = base_importance + goal_boost + success_boost

        return importance

    def _detect_semantic_anomalies(
        self,
        experience: SemanticExperience
    ) -> List[str]:
        """
        Detect unusual semantic patterns that might indicate issues.

        Returns:
            List of anomaly descriptions
        """
        anomalies = []

        # Check for extreme velocities (semantic "shocks")
        for dim_name, velocity in experience.semantic_velocity.items():
            if abs(velocity) > 1.0:  # Very large change
                anomalies.append(
                    f"Extreme velocity in {dim_name}: {velocity:.3f}"
                )

        # Check for goal misalignment with success
        if experience.success and experience.goal_alignment_after < 0.5:
            anomalies.append(
                f"Success despite low goal alignment ({experience.goal_alignment_after:.2f})"
            )

        # Check for contradictory dimensions
        # (e.g., high Complexity and high Simplicity simultaneously)
        contradictions = self._check_contradictory_dimensions(
            experience.semantic_state
        )
        anomalies.extend(contradictions)

        return anomalies

    def _check_contradictory_dimensions(
        self,
        semantic_state: Dict[str, float]
    ) -> List[str]:
        """Check for semantically contradictory dimension activations."""
        contradictions = []

        # Define contradictory pairs
        contradictory_pairs = [
            ('Complexity', 'Simplicity'),
            ('Warmth', 'Coldness'),
            ('Formality', 'Casualness'),
            ('Certainty', 'Uncertainty'),
            ('Directness', 'Indirectness')
        ]

        for dim1, dim2 in contradictory_pairs:
            val1 = semantic_state.get(dim1, 0.0)
            val2 = semantic_state.get(dim2, 0.0)

            # Both high (>0.5) is contradictory
            if val1 > 0.5 and val2 > 0.5:
                contradictions.append(
                    f"Contradictory: {dim1}={val1:.2f} and {dim2}={val2:.2f}"
                )

        return contradictions

    def get_tool_effect_model(self, tool: str) -> Dict[str, Tuple[float, float]]:
        """
        Get learned model of tool's semantic effects.

        Args:
            tool: Tool name

        Returns:
            Dict mapping dimension -> (mean_effect, std_effect)
        """
        if tool not in self.tool_semantic_effects:
            return {}

        model = {}
        for dim_name, effects in self.tool_semantic_effects[tool].items():
            if effects:
                model[dim_name] = (np.mean(effects), np.std(effects))

        return model

    def suggest_contrastive_pairs(
        self,
        min_pairs: int = 10
    ) -> List[Tuple[Dict[str, float], Dict[str, float]]]:
        """
        Suggest contrastive pairs for contrastive learning.

        Returns:
            List of (success_state, failure_state) pairs
        """
        pairs = []

        n_success = len(self.successful_semantic_states)
        n_failure = len(self.failed_semantic_states)

        if n_success == 0 or n_failure == 0:
            return pairs

        # Sample pairs
        n_pairs = min(min_pairs, min(n_success, n_failure))

        for _ in range(n_pairs):
            success_state = self.successful_semantic_states[
                np.random.randint(n_success)
            ]
            failure_state = self.failed_semantic_states[
                np.random.randint(n_failure)
            ]
            pairs.append((success_state, failure_state))

        return pairs


# ============================================================================
# Multi-Task Semantic Learner
# ============================================================================

class SemanticMultiTaskLearner:
    """
    Multi-task learning head for semantic trajectory prediction.

    Learns auxiliary tasks that provide dense supervision:
    1. Dimension prediction: Predict change in each dimension
    2. Tool effect prediction: Predict tool's semantic impact
    3. Goal achievement: Predict if goal will be achieved
    4. Trajectory forecasting: Predict future semantic states

    These auxiliary losses guide the policy toward semantic awareness.
    """

    def __init__(
        self,
        input_dim: int,
        n_dimensions: int = 244,
        n_tools: int = 10,
        config: Optional[SemanticLearningConfig] = None
    ):
        self.config = config or SemanticLearningConfig()

        # Dimension prediction head
        if self.config.enable_dimension_prediction:
            self.dimension_predictor = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Linear(512, n_dimensions)
            )

        # Tool effect prediction head
        if self.config.enable_tool_effect_learning:
            self.tool_effect_predictor = nn.Sequential(
                nn.Linear(input_dim + n_tools, 512),  # Input + tool one-hot
                nn.ReLU(),
                nn.Linear(512, n_dimensions)
            )

        # Goal achievement predictor
        if self.config.enable_goal_achievement:
            self.goal_achiever = nn.Sequential(
                nn.Linear(input_dim + n_dimensions, 256),  # Input + goal
                nn.ReLU(),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )

        # Trajectory forecaster
        if self.config.enable_trajectory_forecasting:
            self.forecaster = nn.LSTM(
                input_size=n_dimensions,
                hidden_size=512,
                num_layers=2,
                batch_first=True
            )
            self.forecaster_head = nn.Linear(512, n_dimensions)

        logger.info("SemanticMultiTaskLearner initialized")

    def compute_auxiliary_losses(
        self,
        policy_features: torch.Tensor,
        semantic_state: torch.Tensor,
        next_semantic_state: torch.Tensor,
        tool_onehot: torch.Tensor,
        semantic_goal: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute auxiliary losses from semantic trajectory.

        Args:
            policy_features: Policy network features [B, input_dim]
            semantic_state: Current semantic state [B, 244]
            next_semantic_state: Next semantic state [B, 244]
            tool_onehot: Selected tool (one-hot) [B, n_tools]
            semantic_goal: Optional goal dimensions [B, 244]

        Returns:
            Dict of losses
        """
        losses = {}

        # 1. Dimension prediction loss
        if self.config.enable_dimension_prediction:
            pred_next = self.dimension_predictor(policy_features)
            dimension_loss = F.mse_loss(pred_next, next_semantic_state)
            losses['dimension_prediction'] = (
                self.config.dimension_prediction_weight * dimension_loss
            )

        # 2. Tool effect prediction loss
        if self.config.enable_tool_effect_learning:
            tool_input = torch.cat([policy_features, tool_onehot], dim=-1)
            pred_effect = self.tool_effect_predictor(tool_input)
            actual_effect = next_semantic_state - semantic_state
            tool_effect_loss = F.mse_loss(pred_effect, actual_effect)
            losses['tool_effect'] = (
                self.config.tool_effect_weight * tool_effect_loss
            )

        # 3. Goal achievement prediction
        if self.config.enable_goal_achievement and semantic_goal is not None:
            goal_input = torch.cat([policy_features, semantic_goal], dim=-1)
            pred_achievement = self.goal_achiever(goal_input).squeeze(-1)

            # True achievement: distance to goal decreases
            dist_before = torch.abs(semantic_state - semantic_goal).mean(dim=-1)
            dist_after = torch.abs(next_semantic_state - semantic_goal).mean(dim=-1)
            actual_achievement = (dist_after < dist_before).float()

            goal_loss = F.binary_cross_entropy(pred_achievement, actual_achievement)
            losses['goal_achievement'] = (
                self.config.goal_achievement_weight * goal_loss
            )

        return losses


# ============================================================================
# Semantic Curriculum Designer
# ============================================================================

class SemanticCurriculumDesigner:
    """
    Designs curriculum learning stages based on semantic goal difficulty.

    Starts with easy goals (few dimensions, large targets) and gradually
    increases difficulty (more dimensions, tighter targets).
    """

    def __init__(self, n_stages: int = 5):
        self.n_stages = n_stages
        self.current_stage = 0

        logger.info(f"SemanticCurriculumDesigner with {n_stages} stages")

    def get_stage_goals(
        self,
        base_goals: Dict[str, float],
        stage: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Get semantic goals appropriate for current curriculum stage.

        Args:
            base_goals: Full goal specification
            stage: Stage index (uses current_stage if None)

        Returns:
            Subset of goals appropriate for stage
        """
        if stage is None:
            stage = self.current_stage

        # Stage 0: 1-2 dimensions, loose targets
        # Stage 4: All dimensions, tight targets

        n_goals = len(base_goals)
        goals_per_stage = max(1, n_goals // self.n_stages)
        n_goals_this_stage = min(n_goals, (stage + 1) * goals_per_stage)

        # Select most important dimensions for this stage
        sorted_goals = sorted(base_goals.items(), key=lambda x: abs(x[1]), reverse=True)
        stage_goals = dict(sorted_goals[:n_goals_this_stage])

        # Loosen targets for early stages
        tolerance = (self.n_stages - stage) * 0.1
        stage_goals = {
            dim: max(0.0, min(1.0, target - tolerance))
            for dim, target in stage_goals.items()
        }

        logger.info(
            f"Stage {stage}: {len(stage_goals)} goals with tolerance {tolerance:.2f}"
        )

        return stage_goals

    def should_advance_stage(
        self,
        success_rate: float,
        threshold: float = 0.7
    ) -> bool:
        """
        Determine if policy should advance to next stage.

        Args:
            success_rate: Recent success rate on current stage
            threshold: Required success rate to advance

        Returns:
            True if should advance
        """
        if success_rate >= threshold and self.current_stage < self.n_stages - 1:
            self.current_stage += 1
            logger.info(
                f"Advancing to stage {self.current_stage} "
                f"(success rate: {success_rate:.2%})"
            )
            return True
        return False


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'SemanticLearningConfig',
    'SemanticExperience',
    'SemanticTrajectoryAnalyzer',
    'SemanticMultiTaskLearner',
    'SemanticCurriculumDesigner',
]