"""
Online trainer for bandit models.

Handles periodic batch updates of the posterior (Bootstrap ensemble or MC-Dropout).
Uses simple MSE loss with Adam optimizer.
"""


import torch
import torch.nn as nn
import torch.optim as optim

from hololoom.bandits.neural_ts.posterior import BootstrapPosterior, MCDropoutPosterior
from hololoom.bandits.neural_ts.replay import ReplayBuffer


class BanditTrainer:
    """
    Online trainer for contextual bandit models.

    Performs periodic SGD updates using replay buffer data.
    Supports both Bootstrap (train each head on bootstrap sample)
    and MC-Dropout (train single model on full batch).

    Attributes:
        posterior: The posterior to train (Bootstrap or MC-Dropout)
        lr: Learning rate
        batch_size: Batch size for training
        device: Training device

    Example:
        >>> trainer = BanditTrainer(posterior, lr=1e-3, batch_size=256)
        >>> replay = ReplayBuffer(capacity=10000)
        >>> # ... fill replay buffer ...
        >>> trainer.fit_steps(replay, n_steps=100)
        >>> print(trainer.get_statistics())
    """

    def __init__(
        self,
        posterior: BootstrapPosterior | MCDropoutPosterior,
        lr: float = 1e-3,
        batch_size: int = 256,
        device: str = "cpu",
    ):
        """
        Initialize trainer.

        Args:
            posterior: Posterior to train
            lr: Learning rate
            batch_size: Batch size for each training step
            device: Device to train on
        """
        self.posterior = posterior
        self.lr = lr
        self.batch_size = batch_size
        self.device = device

        # Create optimizers
        if isinstance(posterior, BootstrapPosterior):
            # One optimizer per ensemble member
            self.optimizers = [
                optim.Adam(model.parameters(), lr=lr)
                for model in posterior.models
            ]
        elif isinstance(posterior, MCDropoutPosterior):
            # Single optimizer for MC-Dropout
            self.optimizers = [optim.Adam(posterior.model.parameters(), lr=lr)]
        else:
            raise TypeError(f"Unknown posterior type: {type(posterior)}")

        self.loss_fn = nn.MSELoss()

        # Statistics
        self._total_steps = 0
        self._total_loss = 0.0

    def fit_steps(self, replay: ReplayBuffer, n_steps: int) -> dict[str, float]:
        """
        Train for n_steps mini-batches.

        Args:
            replay: Replay buffer to sample from
            n_steps: Number of gradient steps

        Returns:
            Dict with training statistics (mean_loss, total_steps)

        Raises:
            ValueError: If replay buffer not ready

        Example:
            >>> stats = trainer.fit_steps(replay, n_steps=100)
            >>> print(f"Loss: {stats['mean_loss']:.4f}")
        """
        if not replay.is_ready():
            raise ValueError(
                f"Replay buffer not ready (size={len(replay)}, warmup={replay.warmup})"
            )

        losses = []

        if isinstance(self.posterior, BootstrapPosterior):
            # Bootstrap: train each head on independent bootstrap samples
            losses = self._fit_bootstrap(replay, n_steps)
        else:
            # MC-Dropout: train single model on regular batches
            losses = self._fit_mc_dropout(replay, n_steps)

        mean_loss = sum(losses) / len(losses) if losses else 0.0
        self._total_steps += n_steps
        self._total_loss += sum(losses)

        return {
            "mean_loss": mean_loss,
            "total_steps": self._total_steps,
        }

    def _fit_bootstrap(self, replay: ReplayBuffer, n_steps: int) -> list[float]:
        """Train Bootstrap ensemble."""
        losses = []

        for step in range(n_steps):
            # Each head trains on its own bootstrap sample
            for model, optimizer in zip(self.posterior.models, self.optimizers):
                # Sample with replacement
                features, rewards = replay.sample(
                    batch_size=self.batch_size,
                    bootstrap=True,
                )

                # Convert to tensors
                x = torch.from_numpy(features).float().to(self.device)
                y = torch.from_numpy(rewards).float().to(self.device)

                # Forward
                model.train()
                predictions = model(x)

                # Loss
                loss = self.loss_fn(predictions, y)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

        return losses

    def _fit_mc_dropout(self, replay: ReplayBuffer, n_steps: int) -> list[float]:
        """Train MC-Dropout model."""
        losses = []
        model = self.posterior.model
        optimizer = self.optimizers[0]

        for step in range(n_steps):
            # Regular sampling (no bootstrap for MC-Dropout)
            features, rewards = replay.sample(
                batch_size=self.batch_size,
                bootstrap=False,
            )

            # Convert to tensors
            x = torch.from_numpy(features).float().to(self.device)
            y = torch.from_numpy(rewards).float().to(self.device)

            # Forward (dropout ON during training)
            model.train()
            predictions = model(x)

            # Loss
            loss = self.loss_fn(predictions, y)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        return losses

    def get_statistics(self) -> dict[str, float]:
        """
        Get training statistics.

        Returns:
            Dict with total_steps and avg_loss
        """
        avg_loss = (
            self._total_loss / self._total_steps if self._total_steps > 0 else 0.0
        )

        return {
            "total_steps": self._total_steps,
            "avg_loss": avg_loss,
        }

    def save_checkpoint(self, path: str) -> None:
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint

        Note:
            Saves all ensemble members for Bootstrap, single model for MC-Dropout.
        """
        if isinstance(self.posterior, BootstrapPosterior):
            checkpoint = {
                "type": "bootstrap",
                "models": [model.state_dict() for model in self.posterior.models],
                "optimizers": [opt.state_dict() for opt in self.optimizers],
                "total_steps": self._total_steps,
                "total_loss": self._total_loss,
            }
        else:
            checkpoint = {
                "type": "mc_dropout",
                "model": self.posterior.model.state_dict(),
                "optimizer": self.optimizers[0].state_dict(),
                "total_steps": self._total_steps,
                "total_loss": self._total_loss,
            }

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint file

        Raises:
            ValueError: If checkpoint type doesn't match posterior type
        """
        checkpoint = torch.load(path, map_location=self.device)

        if isinstance(self.posterior, BootstrapPosterior):
            if checkpoint["type"] != "bootstrap":
                raise ValueError("Checkpoint type mismatch: expected bootstrap")

            for model, state_dict in zip(self.posterior.models, checkpoint["models"]):
                model.load_state_dict(state_dict)

            for opt, state_dict in zip(self.optimizers, checkpoint["optimizers"]):
                opt.load_state_dict(state_dict)

        else:
            if checkpoint["type"] != "mc_dropout":
                raise ValueError("Checkpoint type mismatch: expected mc_dropout")

            self.posterior.model.load_state_dict(checkpoint["model"])
            self.optimizers[0].load_state_dict(checkpoint["optimizer"])

        self._total_steps = checkpoint["total_steps"]
        self._total_loss = checkpoint["total_loss"]

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"BanditTrainer(lr={self.lr}, batch_size={self.batch_size}, "
            f"steps={stats['total_steps']}, avg_loss={stats['avg_loss']:.4f})"
        )
