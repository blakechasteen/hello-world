"""
Reflection - PPO Policy Trainer
================================
Trains the tool selection policy using PPO based on reflection buffer experience.

Philosophy:
The policy learns from its own weaving outcomes. By extracting experience from
the reflection buffer and applying PPO updates, the system gradually improves
its tool selection decisions over time.

Architecture:
- Extracts batched experience from ReflectionBuffer
- Computes advantages using Generalized Advantage Estimation (GAE)
- Updates policy using Proximal Policy Optimization (PPO)
- Tracks learning metrics for monitoring improvement

Integration:
- Works with existing NeuralCore policy in WeavingShuttle
- Uses ReflectionBuffer for experience storage
- Periodic updates during weaving cycles
- Compatible with Thompson Sampling exploration

Author: Claude Code (with HoloLoom by Blake)
Date: 2025-10-27
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

from HoloLoom.reflection.buffer import ReflectionBuffer

logger = logging.getLogger(__name__)


# ============================================================================
# PPO Configuration
# ============================================================================

@dataclass
class PPOConfig:
    """
    Configuration for PPO training.

    Attributes:
        learning_rate: Optimizer learning rate (3e-4)
        clip_epsilon: PPO clipping parameter (0.2)
        value_loss_coef: Value function loss weight (0.5)
        entropy_coef: Entropy bonus weight (0.01)
        max_grad_norm: Gradient clipping threshold (0.5)
        gamma: Discount factor for returns (0.99)
        gae_lambda: GAE lambda parameter (0.95)
        n_epochs: Number of optimization epochs per update (4)
        batch_size: Minibatch size for updates (64)
        target_kl: Early stopping KL divergence threshold (0.01)
    """
    learning_rate: float = 3e-4
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    n_epochs: int = 4
    batch_size: int = 64
    target_kl: float = 0.01


# ============================================================================
# PPO Trainer
# ============================================================================

class PPOTrainer:
    """
    Trains the tool selection policy using PPO from reflection buffer experience.

    The trainer:
    1. Extracts experience batches from reflection buffer
    2. Computes advantages using GAE
    3. Updates policy with clipped surrogate objective
    4. Updates value function with MSE loss
    5. Adds entropy bonus for exploration
    6. Tracks learning metrics

    Usage:
        trainer = PPOTrainer(policy=shuttle.policy.core, config=ppo_config)

        # Periodic training
        if len(reflection_buffer) >= min_batch_size:
            metrics = await trainer.train_on_buffer(reflection_buffer)
            print(f"Policy loss: {metrics['policy_loss']:.3f}")
    """

    def __init__(
        self,
        policy: nn.Module,
        config: Optional[PPOConfig] = None,
        device: str = 'cpu'
    ):
        """
        Initialize PPO trainer.

        Args:
            policy: Neural policy module (NeuralCore)
            config: Optional PPO configuration
            device: Torch device ('cpu' or 'cuda')
        """
        self.policy = policy
        self.config = config or PPOConfig()
        self.device = device

        # Move policy to device
        self.policy.to(device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.config.learning_rate
        )

        # Training state
        self.update_count = 0
        self.total_samples = 0

        # Metrics tracking
        self.metrics_history: List[Dict[str, float]] = []

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"PPOTrainer initialized (lr={self.config.learning_rate})")

    async def train_on_buffer(
        self,
        buffer: ReflectionBuffer,
        min_samples: int = 32
    ) -> Dict[str, float]:
        """
        Train policy on experience from reflection buffer.

        Args:
            buffer: ReflectionBuffer with stored experience
            min_samples: Minimum samples needed for training

        Returns:
            Dict with training metrics
        """
        # Extract batch from buffer
        batch = buffer.get_ppo_batch()

        if len(batch['rewards']) < min_samples:
            self.logger.debug(f"Not enough samples for training ({len(batch['rewards'])} < {min_samples})")
            return {}

        self.logger.info(f"Training on {len(batch['rewards'])} samples from reflection buffer")

        # Convert batch to tensors
        tensors = self._batch_to_tensors(batch)

        # Compute advantages using GAE
        advantages, returns = self._compute_advantages(
            tensors['rewards'],
            tensors['values'],
            tensors['dones']
        )

        # Perform PPO update
        metrics = self._ppo_update(
            tensors['observations'],
            tensors['actions'],
            tensors['log_probs_old'],
            advantages,
            returns
        )

        # Track metrics
        self.update_count += 1
        self.total_samples += len(batch['rewards'])
        self.metrics_history.append(metrics)

        self.logger.info(
            f"Update {self.update_count}: "
            f"policy_loss={metrics['policy_loss']:.3f}, "
            f"value_loss={metrics['value_loss']:.3f}, "
            f"entropy={metrics['entropy']:.3f}"
        )

        return metrics

    def _batch_to_tensors(self, batch: Dict[str, List]) -> Dict[str, torch.Tensor]:
        """
        Convert batch dict to tensors.

        Args:
            batch: Batch from reflection buffer

        Returns:
            Dict of tensors ready for training
        """
        # For now, we'll use a simplified approach where observations are feature dicts
        # In a full implementation, we'd encode these into feature vectors

        # Extract rewards and dones (already numeric)
        rewards = torch.tensor(batch['rewards'], dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch['dones'], dtype=torch.float32, device=self.device)

        # TODO: Encode observations into feature vectors
        # For now, create dummy tensors (this will be replaced with actual encoding)
        batch_size = len(batch['observations'])
        observations = torch.randn(batch_size, 128, device=self.device)  # Placeholder

        # TODO: Encode actions (tool names) to indices
        # For now, create dummy action indices
        actions = torch.randint(0, 5, (batch_size,), device=self.device)  # Placeholder

        # Compute current values (forward pass through policy)
        with torch.no_grad():
            policy_output = self.policy(observations)
            if isinstance(policy_output, dict) and 'value' in policy_output:
                values = policy_output['value']
            else:
                values = torch.zeros(batch_size, device=self.device)  # Placeholder

            # Get log probs of old actions (for importance sampling)
            if isinstance(policy_output, dict) and 'logits' in policy_output:
                logits = policy_output['logits']
                log_probs_old = F.log_softmax(logits, dim=-1)
                # Select log probs for taken actions
                log_probs_old = log_probs_old.gather(1, actions.unsqueeze(1)).squeeze(1)
            else:
                log_probs_old = torch.zeros(batch_size, device=self.device)  # Placeholder

        return {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'values': values,
            'log_probs_old': log_probs_old
        }

    def _compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages using Generalized Advantage Estimation (GAE).

        Args:
            rewards: Reward tensor (T,)
            values: Value estimates (T,)
            dones: Done flags (T,)

        Returns:
            Tuple of (advantages, returns)
        """
        T = rewards.size(0)
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        # Compute GAE
        gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0.0  # Terminal state
            else:
                next_value = values[t + 1]

            mask = 1.0 - dones[t]
            delta = rewards[t] + self.config.gamma * next_value * mask - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * mask * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        # Normalize advantages (helps training stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def _ppo_update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_probs_old: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor
    ) -> Dict[str, float]:
        """
        Perform PPO update on policy.

        Args:
            observations: Observation tensor (B, F)
            actions: Action indices (B,)
            log_probs_old: Old log probabilities (B,)
            advantages: Computed advantages (B,)
            returns: Computed returns (B,)

        Returns:
            Dict with loss metrics
        """
        # Accumulate metrics over epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0

        # Multiple optimization epochs
        for epoch in range(self.config.n_epochs):
            # Forward pass
            policy_output = self.policy(observations)

            # Extract logits and values
            if isinstance(policy_output, dict):
                logits = policy_output.get('logits', None)
                values_pred = policy_output.get('value', torch.zeros_like(advantages))
            else:
                self.logger.warning("Policy output is not a dict, using placeholders")
                logits = policy_output  # Assume raw logits
                values_pred = torch.zeros_like(advantages)

            # Compute log probs for taken actions
            log_probs = F.log_softmax(logits, dim=-1)
            log_probs_actions = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

            # Importance sampling ratio
            ratio = torch.exp(log_probs_actions - log_probs_old)

            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value function loss (MSE)
            value_loss = F.mse_loss(values_pred, returns)

            # Entropy bonus (encourage exploration)
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).mean()

            # Total loss
            loss = (
                policy_loss +
                self.config.value_loss_coef * value_loss -
                self.config.entropy_coef * entropy
            )

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            # Track metrics
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()

            # Approximate KL divergence
            with torch.no_grad():
                kl = (log_probs_old - log_probs_actions).mean()
                total_kl += kl.item()

            # Early stopping if KL divergence too high
            if kl.item() > self.config.target_kl:
                self.logger.debug(f"Early stopping at epoch {epoch+1} due to high KL: {kl.item():.4f}")
                break

        # Average over epochs
        n = self.config.n_epochs
        metrics = {
            'policy_loss': total_policy_loss / n,
            'value_loss': total_value_loss / n,
            'entropy': total_entropy / n,
            'kl_divergence': total_kl / n,
            'update_count': self.update_count
        }

        return metrics

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of training metrics.

        Returns:
            Dict with aggregated metrics
        """
        if not self.metrics_history:
            return {}

        # Compute averages over last N updates
        recent_metrics = self.metrics_history[-10:]  # Last 10 updates

        summary = {
            'total_updates': self.update_count,
            'total_samples': self.total_samples,
            'avg_policy_loss': np.mean([m['policy_loss'] for m in recent_metrics]),
            'avg_value_loss': np.mean([m['value_loss'] for m in recent_metrics]),
            'avg_entropy': np.mean([m['entropy'] for m in recent_metrics]),
            'avg_kl': np.mean([m['kl_divergence'] for m in recent_metrics])
        }

        return summary

    def save_checkpoint(self, path: str) -> None:
        """
        Save training checkpoint.

        Args:
            path: Checkpoint file path
        """
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'update_count': self.update_count,
            'total_samples': self.total_samples,
            'config': self.config.__dict__,
            'metrics_history': self.metrics_history
        }

        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """
        Load training checkpoint.

        Args:
            path: Checkpoint file path
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.update_count = checkpoint['update_count']
        self.total_samples = checkpoint['total_samples']
        self.metrics_history = checkpoint['metrics_history']

        self.logger.info(f"Checkpoint loaded from {path} (update {self.update_count})")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import asyncio
    from HoloLoom.reflection.buffer import ReflectionBuffer
    from HoloLoom.fabric.spacetime import Spacetime, WeavingTrace
    from datetime import datetime

    async def demo():
        print("="*80)
        print("PPO Trainer Demo")
        print("="*80 + "\n")

        # Create dummy policy (placeholder)
        class DummyPolicy(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(128, 5)  # 5 tools
                self.value_head = nn.Linear(128, 1)

            def forward(self, x):
                logits = self.fc(x)
                value = self.value_head(x).squeeze(-1)
                return {'logits': logits, 'value': value}

        policy = DummyPolicy()

        # Create trainer
        trainer = PPOTrainer(policy=policy)

        # Create reflection buffer with some mock data
        buffer = ReflectionBuffer(capacity=100)

        # Simulate weaving cycles
        for i in range(50):
            trace = WeavingTrace(
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration_ms=800 + np.random.randn() * 100,
                tool_selected=['answer', 'search', 'calc', 'notion_write', 'query'][i % 5],
                tool_confidence=0.6 + np.random.rand() * 0.3
            )

            spacetime = Spacetime(
                query_text=f"Query {i}",
                response=f"Response {i}",
                tool_used=trace.tool_selected,
                confidence=trace.tool_confidence,
                trace=trace
            )

            await buffer.store(spacetime, feedback={'helpful': True})

        print(f"Buffer filled with {len(buffer)} episodes\n")

        # Train on buffer
        print("Training policy on buffer experience...\n")
        metrics = await trainer.train_on_buffer(buffer, min_samples=32)

        print("Training metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        print()

        # Get summary
        summary = trainer.get_metrics_summary()
        print("Metrics summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

        print("\n✓ Demo complete!")

    asyncio.run(demo())
