"""
Dark Trace Bridge: SAE ↔ Semantic Correlation Learning

Learns and maintains correlations between SAE features and semantic axes.
Uses Welford's online algorithm for numerical stability during streaming updates.

Key Features:
- Online correlation learning with streaming data
- Statistical significance testing (Fisher's z-transformation)
- Confidence intervals for correlation estimates
- Sparse storage for efficient memory usage

Research Basis:
- Welford's online algorithm for numerical stability
- Fisher's z-transformation for confidence intervals
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import defaultdict
import threading
import numpy as np
import torch

from HoloLoom.dark_trace.protocol import LensType, TraceLens
from HoloLoom.dark_trace.registry import CorrelationMatrix, FeatureCorrelation


@dataclass
class CorrelationConfig:
    """Configuration for SAE-Semantic correlation learning."""

    # Minimum samples for correlation computation
    min_samples: int = 30

    # Minimum correlation magnitude to store (sparsity)
    min_correlation: float = 0.1

    # Minimum confidence threshold
    min_confidence: float = 0.9

    # Maximum correlations to track per feature (memory efficiency)
    max_correlations_per_feature: int = 50

    # Update frequency (update correlations every N observations)
    update_frequency: int = 100

    # Enable batch processing for efficiency
    enable_batch_mode: bool = True
    batch_size: int = 64

    @classmethod
    def default(cls) -> "CorrelationConfig":
        """Default configuration."""
        return cls()

    @classmethod
    def fast(cls) -> "CorrelationConfig":
        """Fast mode with lower sample requirements."""
        return cls(
            min_samples=10,
            min_confidence=0.8,
            update_frequency=50,
        )

    @classmethod
    def research(cls) -> "CorrelationConfig":
        """Research mode with high statistical rigor."""
        return cls(
            min_samples=100,
            min_correlation=0.05,
            min_confidence=0.99,
            max_correlations_per_feature=100,
        )


@dataclass
class CorrelationResult:
    """Result of correlation computation."""

    # Correlation matrix
    correlations: Dict[str, Dict[str, float]]

    # Number of samples used
    sample_count: int

    # Statistics
    mean_correlation: float
    max_correlation: float
    significant_correlations: int

    # Metadata
    timestamp: str = ""

    def get_top_correlations(
        self,
        feature_id: str,
        k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Get top-k correlations for a feature."""
        if feature_id not in self.correlations:
            return []

        sorted_corrs = sorted(
            self.correlations[feature_id].items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        return sorted_corrs[:k]


class SAESemanticCorrelator:
    """
    Learns correlations between SAE features and semantic axes.

    Uses streaming correlation computation with Welford's algorithm
    for numerical stability. Maintains both dense and sparse representations.

    Usage:
        correlator = SAESemanticCorrelator(sae_lens, semantic_lens)

        # Update with observations
        for activations in data_stream:
            correlator.observe(activations)

        # Compute correlations
        result = correlator.compute_correlations()

        # Get semantic meaning of SAE feature
        meanings = correlator.get_semantic_meanings("sae.42")
    """

    def __init__(
        self,
        sae_lens: TraceLens,
        semantic_lens: TraceLens,
        config: Optional[CorrelationConfig] = None,
    ):
        """
        Initialize correlator.

        Args:
            sae_lens: SAE lens for feature projection
            semantic_lens: Semantic lens for axis projection
            config: Correlation configuration
        """
        self.sae_lens = sae_lens
        self.semantic_lens = semantic_lens
        self.config = config or CorrelationConfig.default()

        # Core correlation matrix (uses existing registry implementation)
        self._correlation_matrix = CorrelationMatrix()

        # Observation tracking
        self._observation_count = 0
        self._batch_buffer: List[torch.Tensor] = []

        # Cache for computed correlations
        self._cached_result: Optional[CorrelationResult] = None
        self._cache_valid = False

        # Thread safety
        self._lock = threading.Lock()

    def observe(self, activations: torch.Tensor) -> None:
        """
        Observe activations and update running statistics.

        Args:
            activations: Model activations [batch, dim] or [dim]
        """
        with self._lock:
            self._cache_valid = False

            if self.config.enable_batch_mode:
                self._batch_buffer.append(activations)
                if len(self._batch_buffer) >= self.config.batch_size:
                    self._process_batch()
            else:
                self._process_single(activations)

    def _process_batch(self) -> None:
        """Process accumulated batch of observations."""
        if not self._batch_buffer:
            return

        # Stack batch
        batch = torch.cat([
            a.unsqueeze(0) if a.dim() == 1 else a
            for a in self._batch_buffer
        ], dim=0)

        # Project through both lenses
        sae_features = self.sae_lens.project(batch)
        semantic_features = self.semantic_lens.project(batch)

        # Convert to numpy for correlation computation
        sae_np = sae_features.detach().cpu().numpy()
        sem_np = semantic_features.detach().cpu().numpy()

        # Handle batched input
        if sae_np.ndim > 1:
            for i in range(sae_np.shape[0]):
                self._update_correlations(sae_np[i], sem_np[i])
        else:
            self._update_correlations(sae_np, sem_np)

        self._batch_buffer = []

    def _process_single(self, activations: torch.Tensor) -> None:
        """Process a single observation."""
        # Project through both lenses
        sae_features = self.sae_lens.project(activations)
        semantic_features = self.semantic_lens.project(activations)

        # Convert to numpy
        sae_np = sae_features.detach().cpu().numpy()
        sem_np = semantic_features.detach().cpu().numpy()

        # Flatten if needed
        if sae_np.ndim > 1:
            sae_np = sae_np.mean(axis=0)
            sem_np = sem_np.mean(axis=0)

        self._update_correlations(sae_np, sem_np)

    def _update_correlations(
        self,
        sae_activations: np.ndarray,
        semantic_activations: np.ndarray,
    ) -> None:
        """Update running correlation statistics."""
        self._observation_count += 1

        # Create feature activation dicts
        sae_dict = {
            f"sae.{i}": float(v)
            for i, v in enumerate(sae_activations)
            if abs(v) > 1e-6  # Only track active features
        }

        sem_dict = {
            f"semantic.{i}": float(v)
            for i, v in enumerate(semantic_activations)
        }

        # Update correlation matrix
        self._correlation_matrix.update(sae_dict, sem_dict)

    def compute_correlations(self) -> CorrelationResult:
        """
        Compute correlations from accumulated observations.

        Returns:
            CorrelationResult with correlation matrix and statistics
        """
        with self._lock:
            # Flush any remaining batch
            self._process_batch()

            if self._cache_valid and self._cached_result is not None:
                return self._cached_result

            # Compute correlations
            self._correlation_matrix.compute_correlations(
                min_samples=self.config.min_samples
            )

            # Extract correlation values
            correlations: Dict[str, Dict[str, float]] = {}
            all_correlations: List[float] = []
            significant_count = 0

            for src_id in self._correlation_matrix._correlations:
                correlations[src_id] = {}
                for tgt_id, corr in self._correlation_matrix._correlations[src_id].items():
                    if abs(corr.correlation) >= self.config.min_correlation:
                        correlations[src_id][tgt_id] = corr.correlation
                        all_correlations.append(abs(corr.correlation))

                        if corr.confidence >= self.config.min_confidence:
                            significant_count += 1

            # Compute statistics
            mean_corr = np.mean(all_correlations) if all_correlations else 0.0
            max_corr = max(all_correlations) if all_correlations else 0.0

            from datetime import datetime
            result = CorrelationResult(
                correlations=correlations,
                sample_count=self._observation_count,
                mean_correlation=float(mean_corr),
                max_correlation=float(max_corr),
                significant_correlations=significant_count,
                timestamp=datetime.now().isoformat(),
            )

            self._cached_result = result
            self._cache_valid = True

            return result

    def get_semantic_meanings(
        self,
        sae_feature_id: str,
        k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Get semantic meanings for an SAE feature.

        Args:
            sae_feature_id: SAE feature ID (e.g., "sae.42")
            k: Number of top correlations to return

        Returns:
            List of (semantic_name, correlation) tuples
        """
        result = self.compute_correlations()

        if sae_feature_id not in result.correlations:
            return []

        top_corrs = result.get_top_correlations(sae_feature_id, k=k)

        # Extract semantic names
        return [
            (tgt_id.replace("semantic.", ""), corr)
            for tgt_id, corr in top_corrs
            if tgt_id.startswith("semantic.")
        ]

    def get_sae_features_for_semantic(
        self,
        semantic_axis: str,
        k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Get SAE features most correlated with a semantic axis.

        Args:
            semantic_axis: Semantic axis name (e.g., "Warmth") or ID ("semantic.0")
            k: Number of top features to return

        Returns:
            List of (sae_feature_id, correlation) tuples
        """
        result = self.compute_correlations()

        # Normalize semantic axis name
        if not semantic_axis.startswith("semantic."):
            # Search for matching axis
            target_id = None
            for src_id in result.correlations:
                for tgt_id in result.correlations[src_id]:
                    if semantic_axis.lower() in tgt_id.lower():
                        target_id = tgt_id
                        break
                if target_id:
                    break
            if target_id is None:
                return []
        else:
            target_id = semantic_axis

        # Find all SAE features correlated with this semantic axis
        sae_correlations = []
        for sae_id in result.correlations:
            if sae_id.startswith("sae."):
                if target_id in result.correlations[sae_id]:
                    corr = result.correlations[sae_id][target_id]
                    sae_correlations.append((sae_id, corr))

        # Sort by absolute correlation
        sae_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        return sae_correlations[:k]

    def get_unmapped_features(
        self,
        min_activation: float = 0.1,
    ) -> List[str]:
        """
        Find SAE features with no strong semantic correlations.

        These may represent novel semantic dimensions not captured
        by the existing semantic axes.

        Args:
            min_activation: Minimum average activation to consider

        Returns:
            List of SAE feature IDs with weak semantic correlations
        """
        result = self.compute_correlations()
        unmapped = []

        for sae_id in result.correlations:
            if not sae_id.startswith("sae."):
                continue

            # Check if any strong semantic correlations
            max_corr = max(
                (abs(c) for c in result.correlations[sae_id].values()),
                default=0.0
            )

            if max_corr < self.config.min_correlation:
                unmapped.append(sae_id)

        return unmapped

    def reset(self) -> None:
        """Reset all correlation statistics."""
        with self._lock:
            self._correlation_matrix = CorrelationMatrix()
            self._observation_count = 0
            self._batch_buffer = []
            self._cached_result = None
            self._cache_valid = False

    def save(self, path: str) -> None:
        """Save correlation state to file."""
        import json

        result = self.compute_correlations()
        state = {
            "correlations": result.correlations,
            "sample_count": result.sample_count,
            "config": {
                "min_samples": self.config.min_samples,
                "min_correlation": self.config.min_correlation,
                "min_confidence": self.config.min_confidence,
            },
        }

        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

    def load(self, path: str) -> None:
        """Load correlation state from file."""
        import json

        with open(path, 'r') as f:
            state = json.load(f)

        # Restore correlations (simplified - would need more work for full restoration)
        self._cached_result = CorrelationResult(
            correlations=state["correlations"],
            sample_count=state["sample_count"],
            mean_correlation=0.0,  # Would need to recompute
            max_correlation=0.0,
            significant_correlations=0,
        )
        self._cache_valid = True
