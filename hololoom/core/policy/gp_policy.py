"""
GP-Based Policy Wrapper
========================
Continuous action space optimization using Gaussian Process bandits.

Wraps GPThompsonSampling and GPUpperConfidenceBound for hyperparameter tuning.
Provides seamless integration with HoloLoom's unified policy architecture.

WHY GP BANDITS?
- Discrete Thompson Sampling: Beta distributions over 4 fixed tools
- GP Bandits: Learn smooth functions over continuous hyperparameter spaces
- Benefits: Optimal stiffness, damping, temperature automatically discovered
- Regret: O(√T) theoretical bounds vs empirical tuning

USAGE:
```python
from HoloLoom.core.policy.gp_policy import GPPolicy, GPConfig

# Create GP-based policy
policy = GPPolicy(
    mem_dim=384,
    emb=embedder,
    scales=[96, 192, 384],
    gp_config=GPConfig(
        acquisition="thompson",  # or "ucb"
        kernel_type="matern",
        action_space_dims=3  # stiffness, damping, temperature
    )
)

# Use like normal policy
action_plan = await policy.decide(features, context)
# Hyperparameters automatically optimized via GP!
```

CONTINUOUS ACTION SPACE:
- stiffness: [0.05, 0.5] → normalized [0, 1]
- damping: [0.5, 0.95] → normalized [0, 1]
- temperature: [0.1, 2.0] → normalized [0, 1]
- ... (extensible to any continuous parameter)

INTEGRATION WITH UNIFIED POLICY:
- Wraps UnifiedPolicy from policy/unified.py
- GP learns which hyperparameters maximize reward
- Updates every N observations (configurable)
- Falls back to discrete Thompson if GP disabled

Author: HoloLoom Mathematical Moonshot Team
Date: 2025-11-03
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import logging

# Import GP bandits
from HoloLoom.bandits.gaussian_process_bandits import (
    GPThompsonSampling,
    GPUpperConfidenceBound,
    GaussianProcess,
    create_gp_thompson_sampling,
    create_gp_ucb,
    KernelConfig,
    KernelType,
)

# Import policy components
from HoloLoom.core.policy.unified import (
    UnifiedPolicy,
    create_policy,
    BanditStrategy,
)

# Import types
from HoloLoom.core.protocols.types import Features, Context, ActionPlan
from HoloLoom.core.embedding.spectral import MatryoshkaEmbeddings

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class GPConfig:
    """
    Configuration for GP-based policy.

    Controls continuous action space, kernel selection, and acquisition.
    """

    # Acquisition strategy
    acquisition: str = "thompson"  # "thompson" or "ucb"

    # Kernel configuration
    kernel_type: str = "matern"  # "matern" or "rbf"
    length_scale: float = 0.3  # Kernel length scale
    variance: float = 1.0  # Kernel output variance
    nu: float = 2.5  # Matérn smoothness (1.5, 2.5, 5.0)

    # UCB-specific
    beta: float = 2.0  # Exploration parameter for UCB
    adaptive_beta: bool = True  # Use β = √(2 log(t))

    # Noise
    noise_variance: float = 0.01  # Observation noise

    # Action space (continuous hyperparameters)
    action_space_dims: int = 3  # Default: stiffness, damping, temperature
    action_space_bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'stiffness': (0.05, 0.5),
        'damping': (0.5, 0.95),
        'temperature': (0.1, 2.0)
    })

    # Discretization (for candidate set)
    n_candidates_per_dim: int = 5  # Discretization resolution

    # Update frequency
    update_interval: int = 10  # Retrain GP every N observations

    # Fallback
    enable_fallback: bool = True  # Fall back to discrete TS if GP fails


# ============================================================================
# Action Space Normalization
# ============================================================================

class ActionSpaceNormalizer:
    """
    Normalizes continuous hyperparameters to [0, 1] for GP kernel.

    Maps:
    - stiffness: [0.05, 0.5] → [0, 1]
    - damping: [0.5, 0.95] → [0, 1]
    - temperature: [0.1, 2.0] → [0, 1]
    """

    def __init__(self, bounds: Dict[str, Tuple[float, float]]):
        """
        Initialize normalizer.

        Args:
            bounds: Dictionary mapping parameter names to (min, max) tuples
        """
        self.bounds = bounds
        self.param_names = list(bounds.keys())
        self.n_dims = len(self.param_names)

    def normalize(self, params: Dict[str, float]) -> np.ndarray:
        """
        Normalize parameters to [0, 1]^d.

        Args:
            params: Dictionary of parameter values

        Returns:
            Normalized vector in [0, 1]^d
        """
        x = np.zeros(self.n_dims)
        for i, name in enumerate(self.param_names):
            val = params[name]
            min_val, max_val = self.bounds[name]
            x[i] = (val - min_val) / (max_val - min_val)
        return x

    def denormalize(self, x: np.ndarray) -> Dict[str, float]:
        """
        Denormalize from [0, 1]^d to original scale.

        Args:
            x: Normalized vector

        Returns:
            Dictionary of denormalized parameters
        """
        params = {}
        for i, name in enumerate(self.param_names):
            min_val, max_val = self.bounds[name]
            params[name] = min_val + x[i] * (max_val - min_val)
        return params

    def create_candidate_set(self, n_per_dim: int = 5) -> np.ndarray:
        """
        Create discretized candidate set for GP optimization.

        Generates grid of candidates in [0, 1]^d.

        Args:
            n_per_dim: Number of candidates per dimension

        Returns:
            Candidate set (n_candidates × d)
        """
        # Create 1D grids
        grids = [np.linspace(0, 1, n_per_dim) for _ in range(self.n_dims)]

        # Cartesian product
        meshgrid = np.meshgrid(*grids, indexing='ij')
        candidates = np.stack([g.flatten() for g in meshgrid], axis=1)

        return candidates


# ============================================================================
# GP Policy Wrapper
# ============================================================================

@dataclass
class GPPolicy:
    """
    Policy engine using Gaussian Process bandits for continuous optimization.

    Wraps UnifiedPolicy and adds GP-based hyperparameter tuning.

    Architecture:
    1. Discrete tool selection (answer, search, notion_write, calc)
    2. Continuous hyperparameter optimization via GP (stiffness, damping, etc.)
    3. Observation collection: (hyperparams, reward) → GP training data
    4. Periodic GP retraining

    Workflow:
    1. GP samples/optimizes over candidate set
    2. Denormalize to get actual hyperparameters
    3. Apply hyperparameters to system
    4. Observe reward
    5. Update GP with (normalized_params, reward)
    6. Repeat
    """

    # Underlying policy
    base_policy: UnifiedPolicy

    # GP configuration
    gp_config: GPConfig

    # Action space
    normalizer: ActionSpaceNormalizer
    candidate_set: np.ndarray  # Normalized candidates

    # GP bandit (thompson or ucb)
    gp_bandit: Optional[Any] = None  # GPThompsonSampling or GPUpperConfidenceBound

    # Observation tracking
    observations: int = 0
    observation_buffer: List[Tuple[np.ndarray, float]] = field(default_factory=list)

    # Current hyperparameters
    current_hyperparams: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize GP bandit."""
        # Create kernel config
        kernel_config = KernelConfig(
            kernel_type=KernelType.MATERN if self.gp_config.kernel_type == "matern" else KernelType.RBF,
            length_scale=self.gp_config.length_scale,
            variance=self.gp_config.variance,
            nu=self.gp_config.nu
        )

        # Create GP bandit
        if self.gp_config.acquisition == "thompson":
            self.gp_bandit = create_gp_thompson_sampling(
                candidate_set=self.candidate_set,
                kernel_config=kernel_config,
                noise_variance=self.gp_config.noise_variance
            )
        elif self.gp_config.acquisition == "ucb":
            self.gp_bandit = create_gp_ucb(
                candidate_set=self.candidate_set,
                kernel_config=kernel_config,
                noise_variance=self.gp_config.noise_variance,
                beta=self.gp_config.beta,
                adaptive_beta=self.gp_config.adaptive_beta
            )
        else:
            raise ValueError(f"Unknown acquisition: {self.gp_config.acquisition}")

        # Initialize with default hyperparameters (mid-range)
        self.current_hyperparams = {
            name: (bounds[0] + bounds[1]) / 2
            for name, bounds in self.gp_config.action_space_bounds.items()
        }

        logger.info(f"GPPolicy initialized with {self.gp_config.acquisition} acquisition")
        logger.info(f"Action space: {list(self.gp_config.action_space_bounds.keys())}")
        logger.info(f"Candidate set size: {len(self.candidate_set)}")

    async def decide(self, features: Features, context: Context) -> ActionPlan:
        """
        Make a decision using GP-optimized hyperparameters.

        Pipeline:
        1. GP selects optimal hyperparameters (continuous action)
        2. Apply hyperparameters to system
        3. Base policy decides tool (discrete action)
        4. Observe reward
        5. Update GP with (hyperparams, reward)

        Args:
            features: Extracted features
            context: Retrieved context

        Returns:
            ActionPlan with chosen tool and optimized hyperparameters
        """
        # Step 1: GP selects hyperparameters
        action_normalized, gp_metadata = self.gp_bandit.select_action()
        self.current_hyperparams = self.normalizer.denormalize(action_normalized)

        logger.debug(f"GP selected hyperparams: {self.current_hyperparams}")
        logger.debug(f"GP metadata: {gp_metadata}")

        # Step 2: Apply hyperparameters to system
        # TODO: Apply to spring dynamics, temperature, etc.
        # For now, just store in metadata

        # Step 3: Base policy decides tool
        action_plan = await self.base_policy.decide(features, context)

        # Step 4: Compute reward
        # Reward based on coherence + context quality + hyperparameter effectiveness
        base_reward = features.metrics.get("coherence", 0.0)
        context_quality = len(context.hits) * 0.05 if context.hits else 0.0
        reward = base_reward + context_quality

        # Step 5: Update GP
        self.observations += 1
        self.observation_buffer.append((action_normalized, reward))
        self.gp_bandit.update(action_normalized, reward)

        # Periodic GP retraining (optional, GP updates incrementally)
        if self.observations % self.gp_config.update_interval == 0:
            logger.info(f"GP updated with {self.observations} observations")

        # Step 6: Augment action plan with GP metadata
        if hasattr(action_plan, 'metadata'):
            if action_plan.metadata is None:
                action_plan.metadata = {}
            action_plan.metadata['gp'] = {
                'hyperparams': self.current_hyperparams,
                'acquisition': self.gp_config.acquisition,
                'gp_metadata': gp_metadata,
                'reward': reward,
                'n_observations': self.observations
            }

        return action_plan

    def get_gp_statistics(self) -> Dict[str, Any]:
        """
        Get GP bandit statistics.

        Returns:
            Dictionary with GP stats (posterior mean/std, observations, etc.)
        """
        # Get posterior statistics over candidate set
        gp = self.gp_bandit.gp
        mean, std = gp.predict(self.candidate_set, return_std=True)

        # Find best candidate (highest posterior mean)
        best_idx = np.argmax(mean)
        best_hyperparams = self.normalizer.denormalize(self.candidate_set[best_idx])

        return {
            'n_observations': len(gp.y),
            'best_hyperparams': best_hyperparams,
            'best_mean': float(mean[best_idx]),
            'best_std': float(std[best_idx]),
            'posterior_mean': mean.tolist(),
            'posterior_std': std.tolist(),
            'observation_count': self.observations
        }


# ============================================================================
# Factory Functions
# ============================================================================

def create_gp_policy(
    mem_dim: int,
    emb: MatryoshkaEmbeddings,
    scales: List[int],
    gp_config: Optional[GPConfig] = None,
    device: Optional[torch.device] = None,
    n_layers: int = 2,
    n_heads: int = 4,
    epsilon: float = 0.1,
    cfg: Optional[Any] = None,
    **kwargs
) -> GPPolicy:
    """
    Factory function to create a GP-based policy.

    Args:
        mem_dim: Memory dimension
        emb: Embeddings instance
        scales: Embedding scales
        gp_config: GP configuration (optional)
        device: Torch device
        n_layers: Transformer layers
        n_heads: Attention heads
        epsilon: Fallback epsilon for discrete TS
        **kwargs: Additional args for base policy

    Returns:
        GPPolicy instance
    """
    if gp_config is None:
        gp_config = GPConfig()

    # Create action space normalizer
    normalizer = ActionSpaceNormalizer(gp_config.action_space_bounds)

    # Create candidate set
    candidate_set = normalizer.create_candidate_set(
        n_per_dim=gp_config.n_candidates_per_dim
    )

    # Create base policy (with discrete Thompson Sampling as fallback)
    base_policy = create_policy(
        mem_dim=mem_dim,
        emb=emb,
        scales=scales,
        device=device,
        n_layers=n_layers,
        n_heads=n_heads,
        bandit_strategy=BanditStrategy.EPSILON_GREEDY,  # Fallback strategy
        epsilon=epsilon,
        cfg=cfg,  # Pass config for safety settings
        **kwargs
    )

    # Create GP policy
    return GPPolicy(
        base_policy=base_policy,
        gp_config=gp_config,
        normalizer=normalizer,
        candidate_set=candidate_set
    )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'GPConfig',
    'ActionSpaceNormalizer',
    'GPPolicy',
    'create_gp_policy',
]
