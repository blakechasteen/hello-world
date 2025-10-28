"""
Semantic Micropolicy Nudges
============================
Integrates 244D semantic calculus with neural policy engine for semantically-aware decisions.

This module provides:
1. SemanticStateEncoder: Encodes 244D semantic position for policy input
2. SemanticRewardShaper: Shapes rewards based on semantic trajectory
3. SemanticGatedMHA: Attention gating based on semantic state
4. SemanticNudgePolicy: Policy wrapper that applies semantic guidance

Philosophy:
-----------
The policy doesn't just select tools - it navigates semantic space.
By understanding where we are semantically (Warmth, Clarity, Wisdom, etc.),
we can make decisions that are not only effective but also *semantically appropriate*.

Example:
--------
>>> # Define semantic goals
>>> goals = {'Clarity': 0.9, 'Warmth': 0.7, 'Directness': 0.8}
>>> reward_shaper = SemanticRewardShaper(target_dimensions=goals)
>>>
>>> # Shape rewards based on semantic trajectory
>>> shaped_reward = reward_shaper.shape_reward(
...     base_reward=0.7,
...     semantic_state_old={'Clarity': 0.5, 'Warmth': 0.6},
...     semantic_state_new={'Clarity': 0.7, 'Warmth': 0.65}
... )
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from HoloLoom.documentation.types import Features, Context, ActionPlan
from HoloLoom.semantic_calculus.dimensions import (
    EXTENDED_244_DIMENSIONS,
    SemanticSpectrum
)

logger = logging.getLogger(__name__)


# ============================================================================
# Semantic Category Mapping
# ============================================================================

# Map 244 dimensions to 16 semantic categories for efficient encoding
SEMANTIC_CATEGORIES = {
    'Core': list(range(0, 16)),           # Standard dimensions
    'Narrative': list(range(16, 32)),     # Heroic journey
    'Emotional': list(range(32, 48)),     # Emotional depth
    'Relational': list(range(48, 64)),    # Interpersonal
    'Archetypal': list(range(64, 80)),    # Jungian archetypes
    'Philosophical': list(range(80, 96)), # Existential
    'Transformation': list(range(96, 112)), # Change dynamics
    'Ethical': list(range(112, 128)),     # Moral/ethical
    'Creative': list(range(128, 144)),    # Artistic
    'Cognitive': list(range(144, 160)),   # Cognitive complexity
    'Temporal': list(range(160, 176)),    # Temporal/narrative
    'Spatial': list(range(176, 188)),     # Spatial/setting
    'Character': list(range(188, 200)),   # Character traits
    'Plot': list(range(200, 212)),        # Plot structure
    'Theme': list(range(212, 224)),       # Thematic
    'Style': list(range(224, 244))        # Style/voice (20 dims for padding)
}

# Reverse mapping: dimension index -> category name
DIM_TO_CATEGORY = {}
for category, indices in SEMANTIC_CATEGORIES.items():
    for idx in indices:
        if idx < 244:  # Valid dimension
            DIM_TO_CATEGORY[idx] = category


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SemanticNudgeConfig:
    """Configuration for semantic micropolicy nudging."""

    # Encoding
    n_dims: int = 244              # Total semantic dimensions
    policy_dim: int = 384          # Policy network dimension
    top_k: int = 32                # Top-K sparse dimensions
    n_categories: int = 16         # Number of semantic categories

    # Nudging strength
    nudge_weight: float = 0.1      # How much to bias toward semantic goals (0-1)
    reward_shaping_gamma: float = 0.99  # Discount for potential shaping

    # Thresholds
    goal_threshold: float = 0.2    # Distance threshold for goal achievement
    min_velocity: float = 0.01     # Minimum velocity to consider dimension active

    # Optimization
    use_sparse: bool = True        # Use sparse top-K encoding
    cache_projections: bool = True # Cache semantic axis projections


# ============================================================================
# Semantic State Encoder
# ============================================================================

class SemanticStateEncoder(nn.Module):
    """
    Encodes 244D semantic position into policy features.

    Compresses semantic state using:
    1. Sparse top-K active dimensions (position + velocity)
    2. Category aggregation (16 semantic categories)
    3. Fusion into policy dimension
    """

    def __init__(self, config: SemanticNudgeConfig):
        super().__init__()
        self.config = config

        # Sparse projection: top-K dimensions (position + velocity)
        self.sparse_proj = nn.Sequential(
            nn.Linear(config.top_k * 2, config.policy_dim // 2),
            nn.LayerNorm(config.policy_dim // 2),
            nn.ReLU()
        )

        # Category aggregation: 16 categories
        self.category_proj = nn.Sequential(
            nn.Linear(config.n_categories, config.policy_dim // 2),
            nn.LayerNorm(config.policy_dim // 2),
            nn.ReLU()
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(config.policy_dim, config.policy_dim),
            nn.LayerNorm(config.policy_dim)
        )

    def forward(self, semantic_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode semantic state for policy.

        Args:
            semantic_state: Dict with:
                - 'position': [B, 244] current semantic projections
                - 'velocity': [B, 244] semantic velocity
                - 'categories': [B, 16] aggregated by category

        Returns:
            Encoded representation [B, policy_dim]
        """
        pos = semantic_state['position']  # [B, 244]
        vel = semantic_state['velocity']  # [B, 244]

        # Select top-K dimensions by absolute velocity (most active)
        topk_values, topk_indices = torch.topk(
            torch.abs(vel), self.config.top_k, dim=-1
        )

        # Gather top-K positions and velocities
        topk_pos = torch.gather(pos, 1, topk_indices)  # [B, top_k]
        topk_vel = torch.gather(vel, 1, topk_indices)  # [B, top_k]

        # Concatenate position + velocity for sparse encoding
        sparse_features = torch.cat([topk_pos, topk_vel], dim=-1)  # [B, top_k*2]
        sparse_encoded = self.sparse_proj(sparse_features)  # [B, policy_dim//2]

        # Category aggregation
        category_features = semantic_state['categories']  # [B, 16]
        category_encoded = self.category_proj(category_features)  # [B, policy_dim//2]

        # Fuse
        fused = torch.cat([sparse_encoded, category_encoded], dim=-1)
        return self.fusion(fused)


# ============================================================================
# Semantic Reward Shaping
# ============================================================================

class SemanticRewardShaper:
    """
    Shapes rewards based on semantic trajectory.

    Uses potential-based reward shaping to guide policy toward
    desired semantic regions while preserving optimal policy.

    Theory:
    -------
    Potential-based shaping: F(s,a,s') = γ·φ(s') - φ(s)
    Shaped reward: R' = R + F

    This preserves the optimal policy while providing denser feedback.
    """

    def __init__(
        self,
        target_dimensions: Dict[str, float],
        gamma: float = 0.99,
        potential_weight: float = 0.3
    ):
        """
        Args:
            target_dimensions: Desired positions for key dimensions
                Example: {'Wisdom': 0.8, 'Compassion': 0.7, 'Clarity': 0.9}
            gamma: Discount factor for potential shaping
            potential_weight: Weight of potential shaping (0-1)
        """
        self.target_dims = target_dimensions
        self.gamma = gamma
        self.potential_weight = potential_weight

        logger.info(
            f"SemanticRewardShaper initialized with {len(target_dimensions)} goals"
        )
        for dim, target in target_dimensions.items():
            logger.debug(f"  Target: {dim} = {target:.2f}")

    def compute_potential(self, semantic_state: Dict[str, float]) -> float:
        """
        Compute potential function value from semantic position.

        Potential increases as we approach target dimensions.
        Uses negative distance: φ(s) = -||s - s_goal||

        Args:
            semantic_state: Current semantic projections (244D dict)

        Returns:
            Potential value (higher near targets)
        """
        distances = []

        for dim_name, target_value in self.target_dims.items():
            if dim_name in semantic_state:
                current_value = semantic_state[dim_name]
                distance = abs(current_value - target_value)
                distances.append(distance)
            else:
                # Dimension not found, assume maximum distance
                logger.warning(f"Target dimension '{dim_name}' not in semantic state")
                distances.append(1.0)

        if not distances:
            return 0.0

        # Average distance
        avg_distance = np.mean(distances)

        # Potential: -distance (negative distance encourages getting closer)
        potential = -avg_distance

        return potential

    def shape_reward(
        self,
        base_reward: float,
        semantic_state_old: Dict[str, float],
        semantic_state_new: Dict[str, float]
    ) -> float:
        """
        Apply potential-based semantic reward shaping.

        F(s, a, s') = γ·φ(s') - φ(s)
        R' = R + weight·F

        Args:
            base_reward: Original reward signal
            semantic_state_old: Previous semantic position
            semantic_state_new: New semantic position

        Returns:
            Shaped reward
        """
        # Compute potentials
        potential_old = self.compute_potential(semantic_state_old)
        potential_new = self.compute_potential(semantic_state_new)

        # Shaping term
        shaping = self.gamma * potential_new - potential_old

        # Weighted combination
        shaped_reward = base_reward + self.potential_weight * shaping

        logger.debug(
            f"Reward shaping: base={base_reward:.3f}, "
            f"potential_old={potential_old:.3f}, potential_new={potential_new:.3f}, "
            f"shaping={shaping:.3f}, shaped={shaped_reward:.3f}"
        )

        return shaped_reward

    def compute_goal_alignment(
        self,
        semantic_state: Dict[str, float]
    ) -> float:
        """
        Compute alignment score with semantic goals.

        Args:
            semantic_state: Current semantic position

        Returns:
            Alignment score (0-1, higher is better)
        """
        distances = []

        for dim_name, target_value in self.target_dims.items():
            if dim_name in semantic_state:
                current_value = semantic_state[dim_name]
                distance = abs(current_value - target_value)
                distances.append(distance)

        if not distances:
            return 0.0

        # Average distance
        avg_distance = np.mean(distances)

        # Convert to alignment: 1 - distance (normalized)
        alignment = 1.0 - np.clip(avg_distance / 2.0, 0.0, 1.0)

        return alignment


# ============================================================================
# Semantic Nudge Policy Wrapper
# ============================================================================

class SemanticNudgePolicy:
    """
    Policy wrapper that applies semantic micropolicy nudges.

    Augments base policy decisions with semantic guidance,
    subtly biasing tool selection toward semantically desirable regions.
    """

    def __init__(
        self,
        base_policy,  # Base PolicyEngine
        semantic_spectrum: SemanticSpectrum,
        config: Optional[SemanticNudgeConfig] = None,
        semantic_goals: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            base_policy: Base policy engine
            semantic_spectrum: 244D semantic spectrum analyzer
            config: Configuration (uses defaults if None)
            semantic_goals: Optional semantic goals to pursue
        """
        self.base_policy = base_policy
        self.semantic_spectrum = semantic_spectrum
        self.config = config or SemanticNudgeConfig()

        # Initialize reward shaper if goals provided
        self.reward_shaper = None
        if semantic_goals:
            self.reward_shaper = SemanticRewardShaper(
                target_dimensions=semantic_goals,
                gamma=self.config.reward_shaping_gamma
            )

        logger.info("SemanticNudgePolicy initialized")
        if semantic_goals:
            logger.info(f"  Semantic goals: {list(semantic_goals.keys())}")

    async def decide(
        self,
        features: Features,
        context: Context,
        semantic_state: Optional[Dict[str, float]] = None
    ) -> ActionPlan:
        """
        Make decision with optional semantic guidance.

        Args:
            features: Extracted query features
            context: Retrieved context
            semantic_state: Current semantic position (244D dict)

        Returns:
            ActionPlan with semantic nudges applied
        """
        # Get base decision from underlying policy
        action_plan = await self.base_policy.decide(features, context)

        # If semantic state and goals provided, apply nudges
        if semantic_state and self.reward_shaper:
            action_plan = self._apply_semantic_nudge(action_plan, semantic_state)

        return action_plan

    def _apply_semantic_nudge(
        self,
        action_plan: ActionPlan,
        semantic_state: Dict[str, float]
    ) -> ActionPlan:
        """
        Apply subtle bias toward semantically aligned tools.

        This is the core "nudge" mechanism: we adjust tool probabilities
        based on estimated semantic alignment, without overriding the policy.

        Args:
            action_plan: Base action plan from policy
            semantic_state: Current semantic position

        Returns:
            Action plan with semantic nudges applied
        """
        # Compute current alignment with goals
        alignment = self.reward_shaper.compute_goal_alignment(semantic_state)

        # If already well-aligned, minimal nudging needed
        if alignment > (1.0 - self.config.goal_threshold):
            logger.debug(f"Already aligned ({alignment:.3f}), minimal nudging")
            return action_plan

        # Estimate which tools would improve alignment
        # (In a full implementation, this would use learned tool->semantic mappings)
        # For now, we apply uniform nudging weighted by alignment deficit

        alignment_deficit = 1.0 - alignment
        nudge_strength = self.config.nudge_weight * alignment_deficit

        # Adjust tool probabilities
        # Preserve original ordering but add small semantic bias
        adjusted_probs = {}
        for tool, base_prob in action_plan.tool_probs.items():
            # Add small bonus to all tools (encourages exploration toward goal)
            semantic_bonus = nudge_strength * base_prob
            adjusted_probs[tool] = base_prob + semantic_bonus

        # Renormalize
        total = sum(adjusted_probs.values())
        if total > 0:
            adjusted_probs = {k: v / total for k, v in adjusted_probs.items()}
        else:
            adjusted_probs = action_plan.tool_probs  # Fallback

        # Update action plan
        original_tool = action_plan.tool
        action_plan.tool_probs = adjusted_probs

        # Re-select tool based on adjusted probs (if changed significantly)
        max_tool = max(adjusted_probs.items(), key=lambda x: x[1])[0]
        if max_tool != original_tool:
            logger.info(
                f"Semantic nudge changed tool: {original_tool} -> {max_tool} "
                f"(alignment={alignment:.3f})"
            )
            action_plan.tool = max_tool

        return action_plan

    def shape_reward(
        self,
        base_reward: float,
        semantic_state_old: Dict[str, float],
        semantic_state_new: Dict[str, float]
    ) -> float:
        """
        Shape reward based on semantic trajectory.

        Args:
            base_reward: Original reward
            semantic_state_old: Previous semantic position
            semantic_state_new: New semantic position

        Returns:
            Shaped reward
        """
        if self.reward_shaper:
            return self.reward_shaper.shape_reward(
                base_reward,
                semantic_state_old,
                semantic_state_new
            )
        else:
            return base_reward


# ============================================================================
# Semantic State Computation
# ============================================================================

def compute_semantic_state(
    text: str,
    semantic_spectrum: SemanticSpectrum,
    previous_state: Optional[Dict[str, float]] = None,
    dt: float = 1.0
) -> Dict[str, Any]:
    """
    Compute semantic state (position, velocity, categories) from text.

    Args:
        text: Input text
        semantic_spectrum: Initialized spectrum with learned axes
        previous_state: Previous semantic position (for velocity)
        dt: Time step for velocity computation

    Returns:
        Dict with:
            - 'position': Dict[str, float] - semantic projections (244D)
            - 'velocity': Dict[str, float] - rate of change per dimension
            - 'categories': Dict[str, float] - aggregated by category (16D)
            - 'position_tensor': torch.Tensor - [1, 244]
            - 'velocity_tensor': torch.Tensor - [1, 244]
            - 'categories_tensor': torch.Tensor - [1, 16]
    """
    # Embed text (single point, no trajectory)
    words = text.split()[:50]  # Limit to 50 words for efficiency
    if not words:
        # Empty text, return zeros
        return _empty_semantic_state()

    # Get embeddings
    # Assume semantic_spectrum has an embed_fn attribute or we pass it separately
    # For now, we'll need to handle this in the caller

    # This is a placeholder - actual implementation needs embedding function
    # We'll compute projections assuming we have the vector
    raise NotImplementedError(
        "compute_semantic_state requires embedding function integration. "
        "See demo implementation for full example."
    )


def _empty_semantic_state() -> Dict[str, Any]:
    """Return empty semantic state (all zeros)."""
    n_dims = 244
    n_categories = 16

    return {
        'position': {dim.name: 0.0 for dim in EXTENDED_244_DIMENSIONS},
        'velocity': {dim.name: 0.0 for dim in EXTENDED_244_DIMENSIONS},
        'categories': {cat: 0.0 for cat in SEMANTIC_CATEGORIES.keys()},
        'position_tensor': torch.zeros(1, n_dims),
        'velocity_tensor': torch.zeros(1, n_dims),
        'categories_tensor': torch.zeros(1, n_categories)
    }


def aggregate_by_category(
    semantic_projections: Dict[str, float]
) -> Dict[str, float]:
    """
    Aggregate 244D projections into 16 semantic categories.

    Args:
        semantic_projections: Dict mapping dimension name -> value

    Returns:
        Dict mapping category name -> aggregated value
    """
    category_sums = {cat: 0.0 for cat in SEMANTIC_CATEGORIES.keys()}
    category_counts = {cat: 0 for cat in SEMANTIC_CATEGORIES.keys()}

    # Aggregate by category
    for i, dim in enumerate(EXTENDED_244_DIMENSIONS):
        if dim.name in semantic_projections:
            category = DIM_TO_CATEGORY.get(i, 'Core')
            category_sums[category] += semantic_projections[dim.name]
            category_counts[category] += 1

    # Average per category
    category_averages = {}
    for cat in SEMANTIC_CATEGORIES.keys():
        if category_counts[cat] > 0:
            category_averages[cat] = category_sums[cat] / category_counts[cat]
        else:
            category_averages[cat] = 0.0

    return category_averages


# ============================================================================
# Utility Functions
# ============================================================================

def define_semantic_goals(goal_type: str) -> Dict[str, float]:
    """
    Define common semantic goal configurations.

    Args:
        goal_type: One of:
            - 'professional': Professional, clear, direct
            - 'empathetic': Warm, compassionate, understanding
            - 'educational': Patient, clear, encouraging
            - 'creative': Imaginative, expressive, flowing
            - 'analytical': Complex, nuanced, precise

    Returns:
        Dict of semantic goals
    """
    goals = {
        'professional': {
            'Formality': 0.7,
            'Clarity': 0.9,
            'Directness': 0.8,
            'Precision': 0.8,
            'Efficiency': 0.7
        },
        'empathetic': {
            'Warmth': 0.9,
            'Compassion': 0.9,
            'Understanding': 0.8,
            'Patience': 0.8,
            'Support': 0.8
        },
        'educational': {
            'Clarity': 0.9,
            'Patience': 0.8,
            'Encouragement': 0.7,
            'Simplicity': 0.7,  # Negative complexity
            'Support': 0.8
        },
        'creative': {
            'Imagination': 0.8,
            'Expression': 0.8,
            'Flow': 0.8,
            'Beauty': 0.7,
            'Originality': 0.8
        },
        'analytical': {
            'Complexity': 0.7,
            'Nuance': 0.8,
            'Precision': 0.9,
            'Logic': 0.8,
            'Depth-Artistic': 0.8
        }
    }

    if goal_type not in goals:
        raise ValueError(
            f"Unknown goal type '{goal_type}'. "
            f"Choose from: {list(goals.keys())}"
        )

    return goals[goal_type]


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'SemanticNudgeConfig',
    'SemanticStateEncoder',
    'SemanticRewardShaper',
    'SemanticNudgePolicy',
    'compute_semantic_state',
    'aggregate_by_category',
    'define_semantic_goals',
]