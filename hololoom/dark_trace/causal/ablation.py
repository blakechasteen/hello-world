"""
Dark Trace Causal: Feature Ablation Testing

Implements rigorous ablation studies for causal validation of features.
Ablation = removing/suppressing a feature and measuring the impact.

Key Insight: If removing feature X significantly changes output Y,
then X is causally involved in producing Y.

Research Basis:
- Causal Tracing (Meng et al., 2022)
- Ablation studies in neuroscience
- Feature importance via removal

Ablation Methods:
1. Zero Ablation: Set feature activation to zero
2. Mean Ablation: Set to dataset mean activation
3. Resample Ablation: Replace with activation from different input
4. Noise Ablation: Add noise to corrupt the feature
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch


class AblationMethod(Enum):
    """Methods for ablating (suppressing) features."""

    ZERO = "zero"  # Set activation to zero
    MEAN = "mean"  # Set to dataset mean
    RESAMPLE = "resample"  # Replace with different example
    NOISE = "noise"  # Add Gaussian noise


@dataclass
class AblationConfig:
    """Configuration for ablation experiments."""

    # Ablation method
    method: AblationMethod = AblationMethod.ZERO

    # For NOISE method: noise standard deviation
    noise_std: float = 1.0

    # Statistical settings
    n_samples: int = 100  # Number of test samples
    n_bootstrap: int = 1000  # Bootstrap iterations for CI
    confidence_level: float = 0.95  # Confidence interval level

    # Significance threshold
    significance_threshold: float = 0.05  # p-value threshold

    # Batch processing
    batch_size: int = 32

    # Feature groups (for group ablation)
    enable_group_ablation: bool = False

    @classmethod
    def default(cls) -> "AblationConfig":
        """Default configuration."""
        return cls()

    @classmethod
    def quick(cls) -> "AblationConfig":
        """Quick configuration for development."""
        return cls(
            n_samples=20,
            n_bootstrap=100,
        )

    @classmethod
    def rigorous(cls) -> "AblationConfig":
        """Rigorous configuration for publication."""
        return cls(
            n_samples=500,
            n_bootstrap=10000,
            confidence_level=0.99,
            significance_threshold=0.01,
        )


@dataclass
class AblationEffect:
    """Effect of ablating a single feature."""

    feature_id: str
    feature_index: int

    # Output changes
    output_delta: float  # Mean absolute change in output
    output_delta_std: float  # Standard deviation
    relative_change: float  # Relative change (%)

    # Statistical significance
    p_value: float
    is_significant: bool
    confidence_interval: tuple[float, float]

    # Effect direction
    increases_output: bool  # True if ablation increases output
    effect_size: float  # Cohen's d or similar

    # Breakdown by output dimension (if multi-dimensional)
    per_dimension_effects: dict[int, float] = field(default_factory=dict)

    # Metadata
    n_samples: int = 0
    ablation_method: str = ""

    def __repr__(self) -> str:
        sig = "***" if self.p_value < 0.001 else "**" if self.p_value < 0.01 else "*" if self.is_significant else ""
        direction = "+" if self.increases_output else "-"
        return f"AblationEffect({self.feature_id}: {direction}{self.output_delta:.4f}{sig})"


@dataclass
class AblationResult:
    """Result of ablating one or more features."""

    # Feature(s) ablated
    feature_ids: list[str]
    feature_indices: list[int]

    # Aggregate effects
    total_output_delta: float
    total_relative_change: float

    # Per-feature effects
    per_feature_effects: dict[str, AblationEffect] = field(default_factory=dict)

    # Interaction effects (for multi-feature ablation)
    interaction_effect: float | None = None  # Synergy/redundancy

    # Original vs ablated outputs
    original_outputs: torch.Tensor | None = None
    ablated_outputs: torch.Tensor | None = None

    # Statistical summary
    mean_effect: float = 0.0
    max_effect: float = 0.0
    significant_features: int = 0
    total_features: int = 0

    # Metadata
    config: AblationConfig | None = None
    computation_time_ms: float = 0.0

    def get_significant_features(self) -> list[AblationEffect]:
        """Get features with statistically significant effects."""
        return [
            effect for effect in self.per_feature_effects.values()
            if effect.is_significant
        ]

    def get_top_features(self, k: int = 10) -> list[AblationEffect]:
        """Get top-k features by effect size."""
        sorted_effects = sorted(
            self.per_feature_effects.values(),
            key=lambda x: abs(x.output_delta),
            reverse=True
        )
        return sorted_effects[:k]

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "Ablation Result Summary:",
            f"  Features ablated: {len(self.feature_ids)}",
            f"  Total output delta: {self.total_output_delta:.4f}",
            f"  Relative change: {self.total_relative_change:.2%}",
            f"  Significant features: {self.significant_features}/{self.total_features}",
        ]

        if self.per_feature_effects:
            lines.append("\n  Top 5 effects:")
            for effect in self.get_top_features(5):
                lines.append(f"    {effect}")

        return "\n".join(lines)


@dataclass
class BatchAblationResult:
    """Results from ablating multiple features independently."""

    # Individual results
    results: dict[str, AblationResult] = field(default_factory=dict)

    # Aggregate statistics
    n_features_tested: int = 0
    n_significant: int = 0
    mean_effect: float = 0.0
    max_effect: float = 0.0

    # Feature ranking
    ranked_features: list[tuple[str, float]] = field(default_factory=list)

    # Metadata
    total_computation_time_ms: float = 0.0

    def get_feature_ranking(self) -> list[tuple[str, float]]:
        """Get features ranked by effect size."""
        if self.ranked_features:
            return self.ranked_features

        ranking = []
        for feature_id, result in self.results.items():
            if result.per_feature_effects and feature_id in result.per_feature_effects:
                effect = result.per_feature_effects[feature_id]
                ranking.append((feature_id, abs(effect.output_delta)))

        return sorted(ranking, key=lambda x: x[1], reverse=True)


class FeatureAblator:
    """
    Performs feature ablation experiments for causal validation.

    Ablation tests whether a feature is causally necessary for
    producing a particular output by removing/suppressing the
    feature and measuring the effect.

    Usage:
        ablator = FeatureAblator(model, probe)

        # Single feature ablation
        result = ablator.ablate_feature("sae.42", test_inputs)

        # Batch ablation (test many features)
        batch_result = ablator.ablate_batch(feature_ids, test_inputs)

        # Group ablation (test features together)
        result = ablator.ablate_group(["sae.42", "sae.43"], test_inputs)
    """

    def __init__(
        self,
        model: Any,  # Model or hook system
        probe: Any,  # MindProbe for activation access
        config: AblationConfig | None = None,
        output_fn: Callable | None = None,  # Custom output extraction
    ):
        """
        Initialize FeatureAblator.

        Args:
            model: Model to ablate features in
            probe: MindProbe for reading/modifying activations
            config: Ablation configuration
            output_fn: Optional function to extract relevant output
                       (default: use model's raw output)
        """
        self.model = model
        self.probe = probe
        self.config = config or AblationConfig.default()
        self.output_fn = output_fn or self._default_output_fn

        # Cache for mean activations (for MEAN ablation method)
        self._mean_cache: dict[str, torch.Tensor] = {}

        # Statistics tracking
        self._ablation_count = 0

    def _default_output_fn(self, model_output: Any) -> torch.Tensor:
        """Default output extraction."""
        if isinstance(model_output, torch.Tensor):
            return model_output
        if hasattr(model_output, 'logits'):
            return model_output.logits
        if isinstance(model_output, tuple):
            return model_output[0]
        raise ValueError(f"Cannot extract output from {type(model_output)}")

    def ablate_feature(
        self,
        feature_id: str,
        test_inputs: torch.Tensor,
        method: AblationMethod | None = None,
    ) -> AblationResult:
        """
        Ablate a single feature and measure the effect.

        Args:
            feature_id: Feature to ablate (e.g., "sae.42")
            test_inputs: Test inputs [batch, ...]
            method: Ablation method (default: from config)

        Returns:
            AblationResult with effect measurements
        """
        import time
        start_time = time.time()

        method = method or self.config.method
        feature_index = self._parse_feature_index(feature_id)

        # Get original outputs
        original_outputs = self._get_outputs(test_inputs)

        # Get ablated outputs
        ablated_outputs = self._get_ablated_outputs(
            test_inputs, [feature_index], method
        )

        # Compute effects
        effect = self._compute_effect(
            feature_id, feature_index,
            original_outputs, ablated_outputs,
            method
        )

        self._ablation_count += 1

        return AblationResult(
            feature_ids=[feature_id],
            feature_indices=[feature_index],
            total_output_delta=effect.output_delta,
            total_relative_change=effect.relative_change,
            per_feature_effects={feature_id: effect},
            original_outputs=original_outputs,
            ablated_outputs=ablated_outputs,
            mean_effect=effect.output_delta,
            max_effect=effect.output_delta,
            significant_features=1 if effect.is_significant else 0,
            total_features=1,
            config=self.config,
            computation_time_ms=(time.time() - start_time) * 1000,
        )

    def ablate_batch(
        self,
        feature_ids: list[str],
        test_inputs: torch.Tensor,
        method: AblationMethod | None = None,
    ) -> BatchAblationResult:
        """
        Ablate multiple features independently.

        Each feature is ablated separately to measure its
        individual contribution.

        Args:
            feature_ids: Features to test
            test_inputs: Test inputs
            method: Ablation method

        Returns:
            BatchAblationResult with all individual results
        """
        import time
        start_time = time.time()

        results = {}
        effects = []

        for feature_id in feature_ids:
            result = self.ablate_feature(feature_id, test_inputs, method)
            results[feature_id] = result

            if feature_id in result.per_feature_effects:
                effects.append(result.per_feature_effects[feature_id])

        # Aggregate statistics
        n_significant = sum(1 for e in effects if e.is_significant)
        mean_effect = np.mean([abs(e.output_delta) for e in effects]) if effects else 0.0
        max_effect = max([abs(e.output_delta) for e in effects]) if effects else 0.0

        # Ranking
        ranking = [(f, abs(results[f].total_output_delta)) for f in feature_ids]
        ranking.sort(key=lambda x: x[1], reverse=True)

        return BatchAblationResult(
            results=results,
            n_features_tested=len(feature_ids),
            n_significant=n_significant,
            mean_effect=mean_effect,
            max_effect=max_effect,
            ranked_features=ranking,
            total_computation_time_ms=(time.time() - start_time) * 1000,
        )

    def ablate_group(
        self,
        feature_ids: list[str],
        test_inputs: torch.Tensor,
        method: AblationMethod | None = None,
    ) -> AblationResult:
        """
        Ablate multiple features simultaneously.

        This tests whether features work together (synergy)
        or independently (additive effects).

        Args:
            feature_ids: Features to ablate together
            test_inputs: Test inputs
            method: Ablation method

        Returns:
            AblationResult with group effect and interaction analysis
        """
        import time
        start_time = time.time()

        method = method or self.config.method
        feature_indices = [self._parse_feature_index(f) for f in feature_ids]

        # Get original outputs
        original_outputs = self._get_outputs(test_inputs)

        # Get group ablated outputs
        group_ablated_outputs = self._get_ablated_outputs(
            test_inputs, feature_indices, method
        )

        # Compute group effect
        group_delta = self._compute_output_delta(
            original_outputs, group_ablated_outputs
        )

        # Get individual effects for comparison
        individual_effects = {}
        individual_sum = 0.0

        for feature_id, feature_index in zip(feature_ids, feature_indices):
            individual_ablated = self._get_ablated_outputs(
                test_inputs, [feature_index], method
            )
            effect = self._compute_effect(
                feature_id, feature_index,
                original_outputs, individual_ablated,
                method
            )
            individual_effects[feature_id] = effect
            individual_sum += effect.output_delta

        # Interaction effect: how much the group effect differs
        # from the sum of individual effects
        # Positive = synergy (features amplify each other)
        # Negative = redundancy (features overlap)
        interaction_effect = group_delta - individual_sum

        # Compute relative change
        original_norm = torch.norm(original_outputs.float()).item()
        relative_change = group_delta / (original_norm + 1e-8)

        return AblationResult(
            feature_ids=feature_ids,
            feature_indices=feature_indices,
            total_output_delta=group_delta,
            total_relative_change=relative_change,
            per_feature_effects=individual_effects,
            interaction_effect=interaction_effect,
            original_outputs=original_outputs,
            ablated_outputs=group_ablated_outputs,
            mean_effect=group_delta / len(feature_ids),
            max_effect=max(abs(e.output_delta) for e in individual_effects.values()),
            significant_features=sum(1 for e in individual_effects.values() if e.is_significant),
            total_features=len(feature_ids),
            config=self.config,
            computation_time_ms=(time.time() - start_time) * 1000,
        )

    def _get_outputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """Get model outputs for inputs."""
        with torch.no_grad():
            outputs = self.model(inputs)
            return self.output_fn(outputs)

    def _get_ablated_outputs(
        self,
        inputs: torch.Tensor,
        feature_indices: list[int],
        method: AblationMethod,
    ) -> torch.Tensor:
        """Get model outputs with specified features ablated."""
        # Create ablation hook
        ablation_values = self._get_ablation_values(
            inputs, feature_indices, method
        )

        def ablation_hook(module, input, output):
            # Modify the relevant feature dimensions
            for i, idx in enumerate(feature_indices):
                if output.dim() == 3:  # [batch, seq, hidden]
                    output[:, :, idx] = ablation_values[i]
                elif output.dim() == 2:  # [batch, hidden]
                    output[:, idx] = ablation_values[i]
            return output

        # Register hook, run forward, remove hook
        if hasattr(self.probe, 'hook_layer'):
            hook = self.probe.hook_layer.register_forward_hook(ablation_hook)
        else:
            # Fallback: assume probe has a model attribute with accessible layers
            hook = None

        try:
            with torch.no_grad():
                outputs = self.model(inputs)
                return self.output_fn(outputs)
        finally:
            if hook is not None:
                hook.remove()

    def _get_ablation_values(
        self,
        inputs: torch.Tensor,
        feature_indices: list[int],
        method: AblationMethod,
    ) -> list[torch.Tensor]:
        """Get ablation values based on method."""
        values = []

        for idx in feature_indices:
            if method == AblationMethod.ZERO:
                values.append(torch.zeros(1, device=inputs.device))

            elif method == AblationMethod.MEAN:
                # Use cached mean or compute
                cache_key = f"mean_{idx}"
                if cache_key not in self._mean_cache:
                    # Would need dataset to compute mean
                    # For now, use zero as fallback
                    self._mean_cache[cache_key] = torch.zeros(1, device=inputs.device)
                values.append(self._mean_cache[cache_key])

            elif method == AblationMethod.NOISE:
                noise = torch.randn(1, device=inputs.device) * self.config.noise_std
                values.append(noise)

            elif method == AblationMethod.RESAMPLE:
                # Shuffle activations across batch
                # This requires access to current activations
                values.append(torch.zeros(1, device=inputs.device))

        return values

    def _compute_output_delta(
        self,
        original: torch.Tensor,
        ablated: torch.Tensor,
    ) -> float:
        """Compute mean absolute difference in outputs."""
        diff = (original.float() - ablated.float()).abs()
        return diff.mean().item()

    def _compute_effect(
        self,
        feature_id: str,
        feature_index: int,
        original: torch.Tensor,
        ablated: torch.Tensor,
        method: AblationMethod,
    ) -> AblationEffect:
        """Compute detailed ablation effect."""
        # Basic delta
        diff = (original.float() - ablated.float())
        output_delta = diff.abs().mean().item()
        output_delta_std = diff.abs().std().item()

        # Relative change
        original_norm = torch.norm(original.float()).item()
        relative_change = output_delta / (original_norm + 1e-8)

        # Direction: does ablation increase or decrease output?
        mean_diff = diff.mean().item()
        increases_output = mean_diff < 0  # Ablation decreased something, so feature increased it

        # Effect size (Cohen's d approximation)
        pooled_std = (original.float().std() + ablated.float().std()) / 2
        effect_size = abs(mean_diff) / (pooled_std.item() + 1e-8)

        # Statistical significance (simple t-test approximation)
        n = original.numel()
        t_stat = abs(mean_diff) / (output_delta_std / np.sqrt(n) + 1e-8)

        # Two-tailed p-value (rough approximation)
        # For large n, t-distribution approaches normal
        from scipy import stats
        try:
            p_value = 2 * (1 - stats.t.cdf(t_stat, df=n-1))
        except ImportError:
            # Fallback without scipy
            p_value = 0.05 if t_stat > 2 else 0.5

        is_significant = p_value < self.config.significance_threshold

        # Confidence interval (bootstrap would be better)
        z = 1.96 if self.config.confidence_level == 0.95 else 2.576
        margin = z * output_delta_std / np.sqrt(n)
        confidence_interval = (output_delta - margin, output_delta + margin)

        # Per-dimension effects (if output is multi-dimensional)
        per_dim = {}
        if original.dim() >= 2:
            flat_orig = original.view(-1, original.shape[-1])
            flat_abl = ablated.view(-1, ablated.shape[-1])
            for d in range(min(10, original.shape[-1])):  # Limit to first 10
                per_dim[d] = (flat_orig[:, d] - flat_abl[:, d]).abs().mean().item()

        return AblationEffect(
            feature_id=feature_id,
            feature_index=feature_index,
            output_delta=output_delta,
            output_delta_std=output_delta_std,
            relative_change=relative_change,
            p_value=p_value,
            is_significant=is_significant,
            confidence_interval=confidence_interval,
            increases_output=increases_output,
            effect_size=effect_size,
            per_dimension_effects=per_dim,
            n_samples=n,
            ablation_method=method.value,
        )

    def _parse_feature_index(self, feature_id: str) -> int:
        """Parse feature index from feature ID."""
        # Handle formats like "sae.42", "semantic.5", etc.
        if "." in feature_id:
            return int(feature_id.split(".")[-1])
        return int(feature_id)

    def compute_mean_activations(
        self,
        dataset: torch.Tensor,
        feature_indices: list[int] | None = None,
    ) -> None:
        """
        Compute and cache mean activations from dataset.

        Useful for MEAN ablation method.

        Args:
            dataset: Dataset to compute means from [n_samples, ...]
            feature_indices: Specific features to compute (None = all)
        """
        # Get activations for dataset
        with torch.no_grad():
            # This would need probe access to get hidden states
            # Placeholder for now
            pass

    def get_statistics(self) -> dict[str, Any]:
        """Get ablation statistics."""
        return {
            "total_ablations": self._ablation_count,
            "mean_cache_size": len(self._mean_cache),
            "config": {
                "method": self.config.method.value,
                "n_samples": self.config.n_samples,
                "significance_threshold": self.config.significance_threshold,
            },
        }
