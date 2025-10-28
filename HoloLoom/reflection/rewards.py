"""
Reflection - Reward Signal Extraction
=====================================
Extracts scalar reward signals from Spacetime artifacts for PPO learning.

Philosophy:
The fabric of spacetime contains rich information about execution quality.
By extracting reward signals from confidence, timing, errors, and user feedback,
we create training signals that guide the policy toward better tool selection.

Reward Components:
- Base reward: Tool selection success (confidence-based)
- Quality bonus: User feedback and quality scores
- Efficiency bonus: Fast execution with minimal resource use
- Curiosity penalty: Errors and warnings reduce reward
- Sparse rewards: Optional user ratings override computed rewards
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

from HoloLoom.fabric.spacetime import Spacetime

logger = logging.getLogger(__name__)


# ============================================================================
# Reward Configuration
# ============================================================================

@dataclass
class RewardConfig:
    """
    Configuration for reward computation.

    Attributes:
        base_weight: Weight for confidence-based base reward (0.6)
        quality_weight: Weight for quality score bonus (0.3)
        efficiency_weight: Weight for timing efficiency (0.1)
        error_penalty: Penalty for errors (-0.5 per error)
        warning_penalty: Penalty for warnings (-0.1 per warning)
        timeout_penalty: Penalty for exceeding time budget (-0.3)
        min_reward: Minimum clipped reward (-1.0)
        max_reward: Maximum clipped reward (1.0)
        time_budget_ms: Expected execution time budget (1000ms)
    """
    base_weight: float = 0.6
    quality_weight: float = 0.3
    efficiency_weight: float = 0.1
    error_penalty: float = -0.5
    warning_penalty: float = -0.1
    timeout_penalty: float = -0.3
    min_reward: float = -1.0
    max_reward: float = 1.0
    time_budget_ms: float = 1000.0


# ============================================================================
# Reward Computation
# ============================================================================

class RewardExtractor:
    """
    Extracts scalar reward signals from Spacetime artifacts.

    Computes multi-component rewards based on:
    1. Base reward: Tool confidence (0-1)
    2. Quality bonus: User feedback quality score
    3. Efficiency bonus: Fast execution relative to time budget
    4. Error/warning penalties: Negative feedback for failures

    The final reward is a weighted sum, clipped to [-1, 1].
    """

    def __init__(self, config: Optional[RewardConfig] = None):
        """
        Initialize reward extractor.

        Args:
            config: Optional reward configuration (uses defaults if None)
        """
        self.config = config or RewardConfig()
        self.logger = logging.getLogger(__name__)

    def compute_reward(
        self,
        spacetime: Spacetime,
        user_feedback: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Compute scalar reward from spacetime artifact.

        Args:
            spacetime: Woven fabric with execution trace
            user_feedback: Optional user feedback with ratings

        Returns:
            Scalar reward in [-1, 1]
        """
        # User feedback overrides computed reward (sparse rewards)
        if user_feedback and 'rating' in user_feedback:
            rating = user_feedback['rating']  # Assume 0-5 scale
            return self._normalize_rating(rating)

        # Compute multi-component reward
        components = self._compute_components(spacetime)

        # Weighted sum
        reward = (
            self.config.base_weight * components['base'] +
            self.config.quality_weight * components['quality'] +
            self.config.efficiency_weight * components['efficiency']
        )

        # Add penalties
        reward += components['error_penalty']
        reward += components['warning_penalty']
        reward += components['timeout_penalty']

        # Clip to valid range
        reward = np.clip(reward, self.config.min_reward, self.config.max_reward)

        self.logger.debug(
            f"Reward computed: {reward:.3f} "
            f"(base={components['base']:.2f}, "
            f"quality={components['quality']:.2f}, "
            f"efficiency={components['efficiency']:.2f}, "
            f"errors={components['error_penalty']:.2f})"
        )

        return float(reward)

    def _compute_components(self, spacetime: Spacetime) -> Dict[str, float]:
        """
        Compute individual reward components.

        Args:
            spacetime: Woven fabric

        Returns:
            Dict with reward components
        """
        components = {}

        # 1. Base reward: Confidence (0-1)
        components['base'] = spacetime.confidence

        # 2. Quality bonus: Quality score if available (0-1)
        if spacetime.quality_score is not None:
            components['quality'] = spacetime.quality_score
        else:
            # Default to confidence if no quality score
            components['quality'] = spacetime.confidence

        # 3. Efficiency bonus: Fast execution
        duration = spacetime.trace.duration_ms
        if duration <= self.config.time_budget_ms:
            # Bonus for being under budget (0 to 1)
            components['efficiency'] = 1.0 - (duration / self.config.time_budget_ms)
        else:
            # Penalty for exceeding budget (0 to -0.5)
            overage = (duration - self.config.time_budget_ms) / self.config.time_budget_ms
            components['efficiency'] = -0.5 * min(overage, 1.0)

        # 4. Error penalty
        num_errors = len(spacetime.trace.errors)
        components['error_penalty'] = num_errors * self.config.error_penalty

        # 5. Warning penalty
        num_warnings = len(spacetime.trace.warnings)
        components['warning_penalty'] = num_warnings * self.config.warning_penalty

        # 6. Timeout penalty (if duration very long)
        if duration > 2 * self.config.time_budget_ms:
            components['timeout_penalty'] = self.config.timeout_penalty
        else:
            components['timeout_penalty'] = 0.0

        return components

    def _normalize_rating(self, rating: float) -> float:
        """
        Normalize user rating to [-1, 1] range.

        Args:
            rating: User rating (typically 0-5)

        Returns:
            Normalized reward in [-1, 1]
        """
        # Assume 5-point scale: 0=bad, 5=excellent
        # Map: 0 -> -1, 1 -> -0.6, 2 -> -0.2, 3 -> 0.2, 4 -> 0.6, 5 -> 1.0
        normalized = (rating / 5.0) * 2.0 - 1.0
        return np.clip(normalized, -1.0, 1.0)

    def extract_experience(
        self,
        spacetime: Spacetime,
        user_feedback: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract complete experience tuple for PPO training.

        Returns observation, action, reward, done flag.

        Args:
            spacetime: Woven fabric
            user_feedback: Optional user feedback

        Returns:
            Dict with experience components:
                - observation: Features dict (motifs, embeddings, context stats)
                - action: Tool selected (string)
                - reward: Scalar reward
                - done: Always True (episodic, single-step tasks)
                - info: Additional metadata
        """
        # Compute reward
        reward = self.compute_reward(spacetime, user_feedback)

        # Extract observation (feature representation)
        observation = {
            'motifs_detected': spacetime.trace.motifs_detected,
            'embedding_scales_used': spacetime.trace.embedding_scales_used,
            'context_shards_count': spacetime.trace.context_shards_count,
            'retrieval_mode': spacetime.trace.retrieval_mode,
            'threads_activated_count': len(spacetime.trace.threads_activated)
        }

        # Extract action
        action = spacetime.tool_used

        # Info dict with additional metadata
        info = {
            'confidence': spacetime.confidence,
            'quality_score': spacetime.quality_score,
            'duration_ms': spacetime.trace.duration_ms,
            'policy_adapter': spacetime.trace.policy_adapter,
            'num_errors': len(spacetime.trace.errors),
            'num_warnings': len(spacetime.trace.warnings),
            'bandit_statistics': spacetime.trace.bandit_statistics
        }

        return {
            'observation': observation,
            'action': action,
            'reward': reward,
            'done': True,  # Single-step episodic tasks
            'info': info
        }


# ============================================================================
# Reward Shaping Utilities
# ============================================================================

def compute_shaped_reward(
    base_reward: float,
    potential_old: float,
    potential_new: float,
    gamma: float = 0.99
) -> float:
    """
    Compute shaped reward using potential-based reward shaping.

    F(s, a, s') = gamma * phi(s') - phi(s)
    Shaped reward = R(s, a, s') + F(s, a, s')

    Args:
        base_reward: Original reward signal
        potential_old: Potential function value at old state
        potential_new: Potential function value at new state
        gamma: Discount factor

    Returns:
        Shaped reward preserving optimal policy
    """
    shaping = gamma * potential_new - potential_old
    return base_reward + shaping


def estimate_potential(spacetime: Spacetime) -> float:
    """
    Estimate potential function value from spacetime state.

    Potential is based on heuristic quality indicators:
    - High confidence suggests good state
    - More threads activated suggests richer context
    - Fewer errors suggests stable state

    Args:
        spacetime: Woven fabric

    Returns:
        Potential value (unbounded, typically 0-1)
    """
    # Confidence component (0-1)
    confidence_component = spacetime.confidence

    # Context richness component (0-1, scaled by typical max of 10 threads)
    context_component = min(len(spacetime.trace.threads_activated) / 10.0, 1.0)

    # Error component (penalty, 0 to -1)
    error_component = -min(len(spacetime.trace.errors) * 0.2, 1.0)

    # Weighted sum (0-1 typically)
    potential = 0.6 * confidence_component + 0.3 * context_component + 0.1 * error_component

    return potential


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from datetime import datetime
    from HoloLoom.fabric.spacetime import Spacetime, WeavingTrace

    print("="*80)
    print("Reward Extraction Demo")
    print("="*80 + "\n")

    # Create test spacetime
    trace = WeavingTrace(
        start_time=datetime.now(),
        end_time=datetime.now(),
        duration_ms=850.0,  # Under 1000ms budget
        motifs_detected=["ALGORITHM", "OPTIMIZATION"],
        embedding_scales_used=[96, 192, 384],
        threads_activated=["thread_001", "thread_002", "thread_003"],
        context_shards_count=3,
        retrieval_mode="fused",
        policy_adapter="fused_adapter",
        tool_selected="answer",
        tool_confidence=0.87,
        errors=[],
        warnings=[]
    )

    spacetime = Spacetime(
        query_text="What is Thompson Sampling?",
        response="Thompson Sampling is...",
        tool_used="answer",
        confidence=0.87,
        trace=trace
    )
    spacetime.add_quality_score(0.92)

    # Extract reward
    extractor = RewardExtractor()
    reward = extractor.compute_reward(spacetime)
    print(f"Computed reward: {reward:.3f}\n")

    # Extract full experience
    experience = extractor.extract_experience(spacetime)
    print("Experience tuple:")
    print(f"  Action: {experience['action']}")
    print(f"  Reward: {experience['reward']:.3f}")
    print(f"  Done: {experience['done']}")
    print(f"  Info: {experience['info']}\n")

    # Test with user feedback (sparse reward)
    user_feedback = {'rating': 5}  # Excellent!
    reward_with_feedback = extractor.compute_reward(spacetime, user_feedback)
    print(f"Reward with user rating=5: {reward_with_feedback:.3f}\n")

    # Test potential-based shaping
    potential = estimate_potential(spacetime)
    print(f"State potential: {potential:.3f}\n")

    print("✓ Demo complete!")
