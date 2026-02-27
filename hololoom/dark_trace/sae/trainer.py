"""
Research-Grade SAE Training

Implements training best practices from Anthropic's SAE research:
- L1 Annealing: Gradual sparsity increase for stable training
- Dead Feature Detection: Track and resample inactive features
- Ghost Gradients: Allow dead features to receive gradient signals
- Decoder Normalization: Prevent scale collapse

Based on:
- "Towards Monosemanticity" (Anthropic, 2023)
- "Scaling Monosemanticity" (Anthropic, 2024)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
import time
import math

from hololoom.dark_trace.sae.core import ResearchSAE


@dataclass
class TrainingConfig:
    """
    Configuration for research-grade SAE training.

    Key hyperparameters from Anthropic's methodology:
    - L1 warmup prevents early training collapse
    - Dead feature threshold triggers resampling
    - Ghost gradients help dead features recover
    """
    # L1 Annealing
    l1_warmup_steps: int = 1000              # Steps to linearly warmup L1
    l1_coefficient: float = 3e-4             # Target L1 coefficient
    l1_schedule: str = "linear"              # "linear", "cosine", or "constant"

    # Dead Feature Detection
    dead_feature_threshold: float = 1e-5     # Activation below this = dead
    dead_threshold_batches: int = 100        # Consecutive batches to mark dead
    enable_resampling: bool = True           # Whether to resample dead features
    resample_every_n_steps: int = 5000       # How often to resample
    resample_scale: float = 0.2              # Scale for reinitialized weights

    # Ghost Gradients
    enable_ghost_gradients: bool = True
    ghost_coefficient: float = 0.01

    # Auxk Loss (for dead feature recovery)
    enable_auxk_loss: bool = True
    auxk_coefficient: float = 1e-3
    auxk_k: int = 32                         # Top-k dead features to encourage

    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    grad_clip_norm: Optional[float] = 1.0    # Gradient clipping

    # Training
    batch_size: int = 32
    normalize_decoder: bool = True           # Normalize after each step

    # Logging
    log_every_n_steps: int = 100
    eval_every_n_steps: int = 500

    @classmethod
    def fast(cls) -> "TrainingConfig":
        """Fast training config for testing/development."""
        return cls(
            l1_warmup_steps=100,
            dead_threshold_batches=20,
            resample_every_n_steps=500,
            log_every_n_steps=20,
            eval_every_n_steps=100,
        )

    @classmethod
    def research(cls) -> "TrainingConfig":
        """Research config with full feature set."""
        return cls(
            l1_warmup_steps=2000,
            dead_threshold_batches=200,
            resample_every_n_steps=10000,
            enable_ghost_gradients=True,
            enable_auxk_loss=True,
            enable_resampling=True,
        )


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""
    step: int = 0
    epoch: int = 0
    total_loss: float = 0.0
    recon_loss: float = 0.0
    sparsity_loss: float = 0.0
    auxk_loss: float = 0.0
    l1_coefficient: float = 0.0
    n_dead_features: int = 0
    dead_fraction: float = 0.0
    mean_activation: float = 0.0
    l0_sparsity: float = 0.0
    grad_norm: float = 0.0
    learning_rate: float = 0.0
    samples_seen: int = 0
    elapsed_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "step": self.step,
            "epoch": self.epoch,
            "loss/total": self.total_loss,
            "loss/recon": self.recon_loss,
            "loss/sparsity": self.sparsity_loss,
            "loss/auxk": self.auxk_loss,
            "l1_coefficient": self.l1_coefficient,
            "features/n_dead": self.n_dead_features,
            "features/dead_fraction": self.dead_fraction,
            "activation/mean": self.mean_activation,
            "sparsity/l0": self.l0_sparsity,
            "training/grad_norm": self.grad_norm,
            "training/lr": self.learning_rate,
            "training/samples_seen": self.samples_seen,
            "training/elapsed_time": self.elapsed_time,
        }


class L1AnnealingScheduler:
    """
    L1 coefficient annealing scheduler.

    Implements gradual L1 warmup to prevent early training collapse.
    Without warmup, strong L1 penalty can push all features to zero
    before they have a chance to specialize.

    Schedules:
    - linear: l1 = min(step / warmup_steps, 1.0) * target_l1
    - cosine: l1 = target_l1 * (1 - cos(pi * min(step/warmup_steps, 1.0))) / 2
    - constant: l1 = target_l1 (no warmup)
    """

    def __init__(
        self,
        target_l1: float,
        warmup_steps: int,
        schedule: str = "linear",
    ):
        """
        Args:
            target_l1: Final L1 coefficient after warmup
            warmup_steps: Number of steps to warmup
            schedule: "linear", "cosine", or "constant"
        """
        self.target_l1 = target_l1
        self.warmup_steps = warmup_steps
        self.schedule = schedule

        if schedule not in ("linear", "cosine", "constant"):
            raise ValueError(f"Unknown schedule: {schedule}")

    def get_l1(self, step: int) -> float:
        """Get L1 coefficient for current step."""
        if self.schedule == "constant" or self.warmup_steps <= 0:
            return self.target_l1

        progress = min(step / self.warmup_steps, 1.0)

        if self.schedule == "linear":
            return progress * self.target_l1
        elif self.schedule == "cosine":
            # Cosine annealing from 0 to target
            return self.target_l1 * (1 - math.cos(math.pi * progress)) / 2
        else:
            return self.target_l1

    def is_warmup_complete(self, step: int) -> bool:
        """Check if warmup phase is complete."""
        return step >= self.warmup_steps


class DeadFeatureResampler:
    """
    Resamples dead features to give them a chance to become useful.

    Dead features are detected by the ActivationTracker when a feature
    doesn't activate above threshold for N consecutive batches.

    Resampling strategy (per Anthropic):
    1. Identify dead features from tracker
    2. Find high-loss examples in the current batch
    3. Reinitialize encoder weights to point toward high-loss inputs
    4. Reset decoder column to small random direction
    5. Reset bias terms
    """

    def __init__(
        self,
        resample_scale: float = 0.2,
        min_dead_to_resample: int = 1,
    ):
        """
        Args:
            resample_scale: Scale for reinitialized weights
            min_dead_to_resample: Minimum dead features to trigger resample
        """
        self.resample_scale = resample_scale
        self.min_dead_to_resample = min_dead_to_resample
        self.total_resampled = 0
        self.resample_history: List[Dict[str, Any]] = []

    def resample(
        self,
        sae: ResearchSAE,
        activations: torch.Tensor,
        losses: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Resample dead features.

        Args:
            sae: The SAE model
            activations: [batch_size, input_dim] batch of activations
            losses: [batch_size] per-sample reconstruction losses (optional)

        Returns:
            Dictionary with resampling statistics
        """
        dead_indices = sae.tracker.get_dead_features()

        if len(dead_indices) < self.min_dead_to_resample:
            return {
                "resampled": 0,
                "dead_before": len(dead_indices),
                "message": "Not enough dead features to resample",
            }

        device = sae.encoder.weight.device
        n_to_resample = len(dead_indices)

        # If we have per-sample losses, use high-loss examples
        if losses is not None:
            # Get indices of highest-loss samples
            _, high_loss_indices = losses.topk(min(n_to_resample, len(losses)))
            resample_directions = activations[high_loss_indices]
        else:
            # Random sample from batch
            perm = torch.randperm(len(activations))[:n_to_resample]
            resample_directions = activations[perm]

        # Normalize directions
        resample_directions = resample_directions / (
            resample_directions.norm(dim=-1, keepdim=True) + 1e-8
        )

        with torch.no_grad():
            for i, feat_idx in enumerate(dead_indices):
                if i >= len(resample_directions):
                    break

                direction = resample_directions[i]

                # Reinitialize encoder row to point toward high-loss input
                sae.encoder.weight.data[feat_idx] = direction * self.resample_scale

                # Reset encoder bias
                sae.encoder.bias.data[feat_idx] = 0.0

                # Reset decoder column to small random direction
                random_dir = torch.randn(sae.input_dim, device=device)
                random_dir = random_dir / (random_dir.norm() + 1e-8)
                sae.decoder.weight.data[:, feat_idx] = random_dir * self.resample_scale

                # Reset feature health tracking
                sae.tracker.feature_health[feat_idx].is_dead = False
                sae.tracker.feature_health[feat_idx].consecutive_dead_batches = 0
                sae.tracker.feature_health[feat_idx].activation_count = 0

        # Normalize decoder after resampling
        sae.normalize_decoder_weights()

        # Update dead mask
        sae._update_dead_mask()

        result = {
            "resampled": min(n_to_resample, len(resample_directions)),
            "dead_before": n_to_resample,
            "dead_indices": dead_indices[:10],  # First 10 for logging
        }

        self.total_resampled += result["resampled"]
        self.resample_history.append({
            "step": sae.tracker.current_step,
            **result,
        })

        return result


class ResearchTrainer:
    """
    Research-grade SAE trainer.

    Implements the full training pipeline with:
    - L1 annealing schedule
    - Dead feature detection and resampling
    - Ghost gradients for dead feature recovery
    - Auxk loss for encouraging dead features
    - Decoder normalization
    - Comprehensive metrics tracking
    """

    def __init__(
        self,
        sae: ResearchSAE,
        config: Optional[TrainingConfig] = None,
    ):
        """
        Args:
            sae: The ResearchSAE model to train
            config: Training configuration
        """
        self.sae = sae
        self.config = config or TrainingConfig()

        # Optimizer
        self.optimizer = optim.Adam(
            sae.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_eps,
            weight_decay=self.config.weight_decay,
        )

        # L1 Scheduler
        self.l1_scheduler = L1AnnealingScheduler(
            target_l1=self.config.l1_coefficient,
            warmup_steps=self.config.l1_warmup_steps,
            schedule=self.config.l1_schedule,
        )

        # Dead feature resampler
        self.resampler = DeadFeatureResampler(
            resample_scale=self.config.resample_scale,
        )

        # Training state
        self.step = 0
        self.epoch = 0
        self.samples_seen = 0
        self.start_time = time.time()

        # Metrics history
        self.metrics_history: List[TrainingMetrics] = []

    def train_step(
        self,
        activations: torch.Tensor,
    ) -> TrainingMetrics:
        """
        Execute one training step.

        Args:
            activations: [batch_size, input_dim] batch of activations

        Returns:
            TrainingMetrics for this step
        """
        self.sae.train()
        self.step += 1
        self.samples_seen += activations.size(0)

        # Get current L1 coefficient from scheduler
        current_l1 = self.l1_scheduler.get_l1(self.step)

        # Forward pass
        reconstruction, latents, latents_pre = self.sae.forward(
            activations, return_pre_activations=True
        )

        # Compute main losses
        losses = self.sae.compute_loss(
            activations, reconstruction, latents, l1_coefficient=current_l1
        )

        total_loss = losses["loss"]
        auxk_loss_value = 0.0

        # Auxk loss for dead feature recovery
        if self.config.enable_auxk_loss:
            auxk_loss = self.sae.compute_auxk_loss(latents, k=self.config.auxk_k)
            auxk_loss_value = auxk_loss.item()
            total_loss = total_loss + self.config.auxk_coefficient * auxk_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        grad_norm = 0.0
        if self.config.grad_clip_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.sae.parameters(), self.config.grad_clip_norm
            ).item()

        # Optimizer step
        self.optimizer.step()

        # Normalize decoder weights
        self.sae.normalize_decoder_weights()

        # Update activation tracking
        tracking_stats = self.sae.update_tracking(latents)

        # Check for resampling
        if (
            self.config.enable_resampling
            and self.step % self.config.resample_every_n_steps == 0
            and self.step > 0
        ):
            # Compute per-sample losses for resampling
            with torch.no_grad():
                per_sample_loss = ((reconstruction - activations) ** 2).mean(dim=-1)
            self.resampler.resample(self.sae, activations, per_sample_loss)

        # Build metrics
        health = self.sae.get_health_summary()
        metrics = TrainingMetrics(
            step=self.step,
            epoch=self.epoch,
            total_loss=total_loss.item(),
            recon_loss=losses["recon_loss"].item(),
            sparsity_loss=losses["sparsity_loss"].item(),
            auxk_loss=auxk_loss_value,
            l1_coefficient=current_l1,
            n_dead_features=health["n_dead"],
            dead_fraction=health["dead_fraction"],
            mean_activation=tracking_stats.get("mean_activation", 0.0),
            l0_sparsity=tracking_stats.get("l0_sparsity", 0.0),
            grad_norm=grad_norm,
            learning_rate=self.config.learning_rate,
            samples_seen=self.samples_seen,
            elapsed_time=time.time() - self.start_time,
        )

        self.metrics_history.append(metrics)

        return metrics

    def train_epoch(
        self,
        dataloader,
        callbacks: Optional[List[Callable[[TrainingMetrics], None]]] = None,
    ) -> List[TrainingMetrics]:
        """
        Train for one epoch.

        Args:
            dataloader: Iterable yielding batches of activations
            callbacks: Optional callbacks called after each step

        Returns:
            List of metrics for each step
        """
        self.epoch += 1
        epoch_metrics = []

        for batch in dataloader:
            # Handle different batch formats
            if isinstance(batch, (tuple, list)):
                activations = batch[0]
            else:
                activations = batch

            metrics = self.train_step(activations)
            epoch_metrics.append(metrics)

            # Execute callbacks
            if callbacks:
                for callback in callbacks:
                    callback(metrics)

            # Logging
            if self.step % self.config.log_every_n_steps == 0:
                self._log_metrics(metrics)

        return epoch_metrics

    def _log_metrics(self, metrics: TrainingMetrics):
        """Log metrics (override for custom logging)."""
        print(
            f"Step {metrics.step:6d} | "
            f"Loss: {metrics.total_loss:.4f} | "
            f"Recon: {metrics.recon_loss:.4f} | "
            f"L1: {metrics.l1_coefficient:.2e} | "
            f"Dead: {metrics.n_dead_features:4d} ({metrics.dead_fraction:.1%}) | "
            f"L0: {metrics.l0_sparsity:.3f}"
        )

    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training progress."""
        if not self.metrics_history:
            return {"status": "no training yet"}

        recent = self.metrics_history[-100:]
        return {
            "total_steps": self.step,
            "total_epochs": self.epoch,
            "samples_seen": self.samples_seen,
            "elapsed_time": time.time() - self.start_time,
            "l1_warmup_complete": self.l1_scheduler.is_warmup_complete(self.step),
            "current_l1": self.l1_scheduler.get_l1(self.step),
            "avg_loss": sum(m.total_loss for m in recent) / len(recent),
            "avg_recon_loss": sum(m.recon_loss for m in recent) / len(recent),
            "avg_dead_fraction": sum(m.dead_fraction for m in recent) / len(recent),
            "total_resampled": self.resampler.total_resampled,
            "health_summary": self.sae.get_health_summary(),
        }

    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        torch.save({
            "step": self.step,
            "epoch": self.epoch,
            "samples_seen": self.samples_seen,
            "sae_state": self.sae.state_dict_extended(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": self.config,
            "resampler_history": self.resampler.resample_history,
        }, path)

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path)
        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]
        self.samples_seen = checkpoint["samples_seen"]
        self.sae.load_state_dict(checkpoint["sae_state"]["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.resampler.resample_history = checkpoint.get("resampler_history", [])
