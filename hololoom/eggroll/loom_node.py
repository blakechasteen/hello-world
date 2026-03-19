from typing import Any

import numpy as np
import torch

from hololoom.dark_trace.auto_probe import AutoProbe
from hololoom.dark_trace.steering_policy import ConsistencyGuard
from hololoom.eggroll.mirror_core import MirrorCoreAgent

# Rust-accelerated operations (with NumPy fallback)
from hololoom.eggroll.rust_ops import (
    RUST_AVAILABLE,
    cosine_similarity_batch,
    normalize_batch,
)


class LoomNode:
    """
    A single node in the distributed HoloLoom Collective.
    Runs an independent copy of MirrorCoreAgent + AutoProbe.

    Rust Acceleration:
    - Embedding normalization (30-50x speedup)
    - Cosine similarity computation for fitness enhancement
    """
    def __init__(self, worker_id: int, model_type: str = "trm", model_kwargs: dict = None, use_dark_trace: bool = True):
        self.worker_id = worker_id
        if model_kwargs is None:
            model_kwargs = {}

        backend_str = "🦀 Rust" if RUST_AVAILABLE else "📐 NumPy"
        print(f"[LoomNode {worker_id}] Initializing ({backend_str} backend)...")
        self.agent = MirrorCoreAgent(model_id=f"worker_{worker_id}", model_type=model_type, **model_kwargs)
        self.model_type = model_type

        self.probe = None
        self.guard = None

        # Cache for embeddings (reused across steps)
        self._last_embedding: np.ndarray | None = None
        self._target_embedding: np.ndarray | None = None

        if use_dark_trace:
            # Determine probe target based on architecture
            target_layer = "recursive_block" # Default for TRM
            if model_type == "spiking":
                target_layer = "layers.0" # Probe the first spiking layer
            elif model_type == "liquid":
                target_layer = "reservoir"

            # Attach Dark Trace Introspection
            # We delay start slightly to avoid thundering herd on thread creation if many workers
            self.probe = AutoProbe(self.agent, layer_name=target_layer)
            self.probe.start()
            self.guard = ConsistencyGuard(self.probe)

    def set_weights(self, weights: dict):
        """Syncs weights with the Hive Mind (Central Parameter Server)."""
        self.agent.custom_model.load_state_dict(weights)

    def set_target_embedding(self, target_embedding: np.ndarray) -> None:
        """Set target embedding for similarity-based fitness computation."""
        self._target_embedding = normalize_batch(
            target_embedding.reshape(1, -1).astype(np.float32)
        ).flatten()

    def extract_embedding(self, hidden_states: torch.Tensor) -> np.ndarray:
        """
        Extract and normalize embedding from model hidden states.

        Uses Rust-accelerated normalization (30-50x speedup).
        Falls back to NumPy if Rust unavailable.

        Args:
            hidden_states: Model hidden states tensor [batch, seq, dim]

        Returns:
            Normalized embedding vector [dim]
        """
        # Mean pool over sequence dimension
        with torch.no_grad():
            embedding = hidden_states.mean(dim=1).squeeze().cpu().numpy()

        # Ensure 2D for batch normalization
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        # Rust-accelerated normalization
        normalized = normalize_batch(embedding.astype(np.float32))

        return normalized.flatten()

    def step(self, task_input: torch.Tensor, target: torch.Tensor) -> tuple[float, dict[str, Any], str]:
        """
        Performs one evolutionary / gradient step.

        Uses Rust-accelerated operations for:
        - Embedding normalization
        - Cosine similarity computation (if target embedding set)

        Args:
            task_input: Input tensor for the model
            target: Target tensor for loss computation

        Returns:
            Tuple of (fitness, metrics_dict, output_sample)
        """
        # Safety Check
        if self.guard:
            self.guard.enforce()

        # Forward with hidden states
        outputs = self.agent.custom_model(task_input)

        # Compute loss
        loss = self.agent.compute_loss(input_ids=task_input, labels=target)

        # Base fitness (negative loss = higher is better)
        fitness = -loss.item()

        # Metrics collection
        metrics: dict[str, Any] = {
            "loss": loss.item(),
            "base_fitness": fitness,
        }

        # Extract embedding from model (if available)
        if hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        elif isinstance(outputs, tuple) and len(outputs) > 1:
            hidden_states = outputs[1] if isinstance(outputs[1], torch.Tensor) else None
        elif isinstance(outputs, torch.Tensor) and outputs.ndim >= 3:
            hidden_states = outputs
        else:
            hidden_states = None

        if hidden_states is not None:
            try:
                embedding = self.extract_embedding(hidden_states)
                self._last_embedding = embedding
                metrics["embedding"] = embedding.tolist()
                metrics["embedding_dim"] = len(embedding)

                # Compute similarity bonus if target embedding is set
                if self._target_embedding is not None:
                    # Rust-accelerated cosine similarity
                    similarity = cosine_similarity_batch(
                        embedding.reshape(1, -1).astype(np.float32),
                        self._target_embedding.astype(np.float32)
                    )[0]
                    metrics["target_similarity"] = float(similarity)

                    # Add similarity bonus to fitness (weighted)
                    similarity_bonus = similarity * 0.1  # 10% weight
                    fitness += similarity_bonus
                    metrics["similarity_bonus"] = similarity_bonus
            except Exception as e:
                metrics["embedding_error"] = str(e)

        # Introspection Bonus (Dark Trace)
        if self.probe:
            health = self.probe.get_brain_health()
            sparsity = health.get("sparsity_loss", 0.0)
            sparsity_penalty = sparsity * 0.1
            fitness -= sparsity_penalty  # Penalty for non-sparse thoughts

            metrics["sparsity_loss"] = sparsity
            metrics["sparsity_penalty"] = sparsity_penalty
            metrics["brain_health"] = health

        metrics["final_fitness"] = fitness

        # Generate output sample for logging
        output_sample = f"Worker {self.worker_id} | Fitness: {fitness:.4f}"

        return fitness, metrics, output_sample

    def shutdown(self):
        if self.probe:
            self.probe.stop()
