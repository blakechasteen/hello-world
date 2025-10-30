"""
Learned Value Functions

Option C: Deep Enhancement - End-to-end neural decision making.

Replaces handcrafted features with learned value functions:
- V(s): State value - "How good is this state?"
- Q(s,a): Action-value - "How good is this action in this state?"
- Advantage A(s,a): Q(s,a) - V(s) - "How much better than average?"

Enables end-to-end learning from experience with policy gradients.

Research Alignment:
- Sutton & Barto (2018): Reinforcement Learning (Chapter 9-13)
- Mnih et al. (2015): Deep Q-Networks (DQN)
- Schulman et al. (2017): Proximal Policy Optimization (PPO)
- Lillicrap et al. (2015): Deep Deterministic Policy Gradient (DDPG)

Public API:
    ValueNetwork: State value function V(s)
    QNetwork: Action-value function Q(s,a)
    ActorCritic: Combined policy and value
    train_value_function: TD learning
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import logging
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)

# PyTorch support
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import optim
    PYTORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available for value functions")
    PYTORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Experience:
    """
    Single experience tuple for learning.

    Attributes:
        state: Current state observation
        action: Action taken
        reward: Reward received
        next_state: Resulting state
        done: Episode termination flag
    """
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


@dataclass
class ValueEstimate:
    """
    Value function estimate with uncertainty.

    Attributes:
        value: Estimated value
        confidence: Uncertainty estimate
        advantage: Advantage estimate (for actor-critic)
    """
    value: float
    confidence: float = 1.0
    advantage: Optional[float] = None


# ============================================================================
# Value Network (V-function)
# ============================================================================

if PYTORCH_AVAILABLE:
    class ValueNetworkPyTorch(nn.Module):
        """
        State value function V(s).

        Estimates expected return from state s:
        V(s) = E[G_t | S_t = s]

        where G_t = sum of discounted future rewards
        """

        def __init__(self,
                     state_dim: int,
                     hidden_dims: List[int],
                     activation: str = 'relu'):
            """
            Initialize value network.

            Args:
                state_dim: State space dimension
                hidden_dims: Hidden layer dimensions
                activation: Activation function ('relu', 'tanh')
            """
            super().__init__()

            self.state_dim = state_dim

            # Build network
            layers = []
            prev_dim = state_dim

            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU() if activation == 'relu' else nn.Tanh(),
                ])
                prev_dim = hidden_dim

            # Value head (scalar output)
            layers.append(nn.Linear(prev_dim, 1))

            self.network = nn.Sequential(*layers)

            logger.info(f"ValueNetwork: state_dim={state_dim}, hidden={hidden_dims}")

        def forward(self, state: torch.Tensor) -> torch.Tensor:
            """
            Estimate value of state.

            Args:
                state: State tensor (batch_size, state_dim)

            Returns:
                Value estimates (batch_size, 1)
            """
            return self.network(state)


# ============================================================================
# Q-Network (Action-Value Function)
# ============================================================================

if PYTORCH_AVAILABLE:
    class QNetworkPyTorch(nn.Module):
        """
        Action-value function Q(s,a).

        Estimates expected return from taking action a in state s:
        Q(s,a) = E[G_t | S_t = s, A_t = a]
        """

        def __init__(self,
                     state_dim: int,
                     action_dim: int,
                     hidden_dims: List[int],
                     dueling: bool = False):
            """
            Initialize Q-network.

            Args:
                state_dim: State space dimension
                action_dim: Action space dimension
                hidden_dims: Hidden layer dimensions
                dueling: Use dueling architecture (DQN improvement)
            """
            super().__init__()

            self.state_dim = state_dim
            self.action_dim = action_dim
            self.dueling = dueling

            # Shared feature extractor
            feature_layers = []
            prev_dim = state_dim

            for hidden_dim in hidden_dims[:-1]:
                feature_layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                ])
                prev_dim = hidden_dim

            self.features = nn.Sequential(*feature_layers)

            if dueling:
                # Dueling DQN: Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
                self.value_stream = nn.Sequential(
                    nn.Linear(prev_dim, hidden_dims[-1]),
                    nn.ReLU(),
                    nn.Linear(hidden_dims[-1], 1)
                )
                self.advantage_stream = nn.Sequential(
                    nn.Linear(prev_dim, hidden_dims[-1]),
                    nn.ReLU(),
                    nn.Linear(hidden_dims[-1], action_dim)
                )
            else:
                # Standard Q-network
                self.q_head = nn.Sequential(
                    nn.Linear(prev_dim, hidden_dims[-1]),
                    nn.ReLU(),
                    nn.Linear(hidden_dims[-1], action_dim)
                )

            logger.info(f"QNetwork: state_dim={state_dim}, action_dim={action_dim}, "
                       f"dueling={dueling}")

        def forward(self, state: torch.Tensor) -> torch.Tensor:
            """
            Estimate Q-values for all actions.

            Args:
                state: State tensor (batch_size, state_dim)

            Returns:
                Q-values (batch_size, action_dim)
            """
            features = self.features(state)

            if self.dueling:
                # Dueling architecture
                value = self.value_stream(features)
                advantage = self.advantage_stream(features)

                # Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
                q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
            else:
                q_values = self.q_head(features)

            return q_values


# ============================================================================
# Actor-Critic Network
# ============================================================================

if PYTORCH_AVAILABLE:
    class ActorCriticPyTorch(nn.Module):
        """
        Actor-Critic architecture.

        Actor: Policy π(a|s) - "What action to take?"
        Critic: Value V(s) - "How good is current state?"

        Used in policy gradient methods (A2C, PPO, etc.)
        """

        def __init__(self,
                     state_dim: int,
                     action_dim: int,
                     hidden_dims: List[int],
                     continuous: bool = False):
            """
            Initialize actor-critic.

            Args:
                state_dim: State space dimension
                action_dim: Action space dimension
                hidden_dims: Shared hidden layer dimensions
                continuous: Continuous action space (Gaussian policy)
            """
            super().__init__()

            self.state_dim = state_dim
            self.action_dim = action_dim
            self.continuous = continuous

            # Shared feature extractor
            feature_layers = []
            prev_dim = state_dim

            for hidden_dim in hidden_dims:
                feature_layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                ])
                prev_dim = hidden_dim

            self.shared_features = nn.Sequential(*feature_layers)

            # Actor head (policy)
            if continuous:
                # Gaussian policy: output mean and log_std
                self.actor_mean = nn.Linear(prev_dim, action_dim)
                self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
            else:
                # Categorical policy: output logits
                self.actor = nn.Linear(prev_dim, action_dim)

            # Critic head (value)
            self.critic = nn.Linear(prev_dim, 1)

            logger.info(f"ActorCritic: state_dim={state_dim}, action_dim={action_dim}, "
                       f"continuous={continuous}")

        def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Forward pass through actor-critic.

            Args:
                state: State tensor (batch_size, state_dim)

            Returns:
                Tuple of (action_logits/mean, value)
            """
            features = self.shared_features(state)

            # Actor output
            if self.continuous:
                action_mean = self.actor_mean(features)
                action_output = action_mean  # Return mean
            else:
                action_logits = self.actor(features)
                action_output = action_logits

            # Critic output
            value = self.critic(features)

            return action_output, value

        def get_action(self, state: torch.Tensor, deterministic: bool = False):
            """
            Sample action from policy.

            Args:
                state: State tensor
                deterministic: Use mean (no exploration)

            Returns:
                Tuple of (action, log_prob, value)
            """
            action_output, value = self.forward(state)

            if self.continuous:
                # Gaussian policy
                mean = action_output
                std = torch.exp(self.actor_log_std)

                if deterministic:
                    action = mean
                    log_prob = None
                else:
                    dist = torch.distributions.Normal(mean, std)
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum(dim=-1)
            else:
                # Categorical policy
                logits = action_output

                if deterministic:
                    action = torch.argmax(logits, dim=-1)
                    log_prob = None
                else:
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

            return action, log_prob, value


# ============================================================================
# Value Function Learner
# ============================================================================

class ValueFunctionLearner:
    """
    Learns value functions from experience.

    Supports:
    - TD(0) learning
    - Monte Carlo returns
    - Generalized Advantage Estimation (GAE)
    """

    def __init__(self,
                 network: Any,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 tau: float = 0.005):
        """
        Initialize value function learner.

        Args:
            network: Value network (V or Q)
            learning_rate: Learning rate
            gamma: Discount factor
            tau: Target network soft update rate
        """
        self.network = network
        self.gamma = gamma
        self.tau = tau

        if PYTORCH_AVAILABLE:
            self.optimizer = optim.Adam(network.parameters(), lr=learning_rate)
            self.target_network = None  # For Q-learning

        logger.info(f"ValueFunctionLearner: lr={learning_rate}, gamma={gamma}")

    def train_td(self,
                experiences: List[Experience],
                batch_size: int = 32) -> Dict[str, float]:
        """
        Train using TD(0) learning.

        TD target: r + γ * V(s')
        Loss: MSE between V(s) and TD target

        Args:
            experiences: List of experience tuples
            batch_size: Mini-batch size

        Returns:
            Training metrics
        """
        if not PYTORCH_AVAILABLE:
            return {}

        self.network.train()

        # Sample mini-batch
        indices = np.random.choice(len(experiences), min(batch_size, len(experiences)))
        batch = [experiences[i] for i in indices]

        # Prepare tensors
        states = torch.FloatTensor([exp.state for exp in batch])
        rewards = torch.FloatTensor([exp.reward for exp in batch])
        next_states = torch.FloatTensor([exp.next_state for exp in batch])
        dones = torch.FloatTensor([float(exp.done) for exp in batch])

        # Current value estimates
        values = self.network(states).squeeze()

        # TD targets
        with torch.no_grad():
            next_values = self.network(next_states).squeeze()
            td_targets = rewards + self.gamma * next_values * (1 - dones)

        # TD loss
        loss = F.mse_loss(values, td_targets)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'loss': loss.item(),
            'mean_value': values.mean().item(),
            'mean_td_error': (td_targets - values).abs().mean().item()
        }

    def estimate(self, state: np.ndarray) -> ValueEstimate:
        """
        Estimate value of state.

        Args:
            state: State array

        Returns:
            Value estimate
        """
        if not PYTORCH_AVAILABLE:
            return ValueEstimate(value=0.0)

        self.network.eval()

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            value = self.network(state_tensor).item()

        return ValueEstimate(value=value)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'ValueNetworkPyTorch',
    'QNetworkPyTorch',
    'ActorCriticPyTorch',
    'ValueFunctionLearner',
    'Experience',
    'ValueEstimate',
]
