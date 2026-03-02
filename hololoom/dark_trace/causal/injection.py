"""
Dark Trace Causal: Feature Injection Testing

Implements controlled feature activation to measure causal effects.
Injection = force-activating a feature and measuring the behavioral change.

Key Insight: If injecting feature X produces effect Y, then X is
causally sufficient for producing Y.

Research Basis:
- Activation engineering (Turner et al., 2023)
- Steering vectors (Anthropic, 2024)
- Representation control

Injection Methods:
1. Additive: Add scaled activation vector
2. Replacement: Replace activation entirely
3. Clamping: Set activation to specific value
4. Steering: Add steering vector in feature direction
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from enum import Enum
import numpy as np
import torch
from collections import defaultdict


class InjectionMethod(Enum):
    """Methods for injecting feature activations."""

    ADDITIVE = "additive"  # Add to existing activation
    REPLACEMENT = "replacement"  # Replace activation entirely
    CLAMPING = "clamping"  # Clamp to specific value
    STEERING = "steering"  # Add steering vector


@dataclass
class InjectionConfig:
    """Configuration for injection experiments."""

    # Injection method
    method: InjectionMethod = InjectionMethod.ADDITIVE

    # Injection strength range for dose-response
    strength_range: Tuple[float, float, int] = (0.0, 2.0, 11)  # (min, max, n_steps)

    # Statistical settings
    n_samples: int = 100  # Number of test samples
    n_bootstrap: int = 1000  # Bootstrap iterations

    # Interaction detection
    detect_interactions: bool = True
    interaction_threshold: float = 0.1  # Synergy/antagonism threshold

    # Batch processing
    batch_size: int = 32

    @classmethod
    def default(cls) -> "InjectionConfig":
        """Default configuration."""
        return cls()

    @classmethod
    def quick(cls) -> "InjectionConfig":
        """Quick configuration for development."""
        return cls(
            strength_range=(0.0, 1.0, 5),
            n_samples=20,
            n_bootstrap=100,
        )

    @classmethod
    def detailed(cls) -> "InjectionConfig":
        """Detailed configuration for thorough analysis."""
        return cls(
            strength_range=(0.0, 3.0, 21),
            n_samples=200,
            n_bootstrap=5000,
        )


@dataclass
class DoseResponsePoint:
    """Single point in dose-response curve."""

    strength: float  # Injection strength
    mean_effect: float  # Mean effect on output
    std_effect: float  # Standard deviation
    confidence_interval: Tuple[float, float]


@dataclass
class DoseResponseCurve:
    """Dose-response relationship for a feature."""

    feature_id: str
    feature_index: int

    # Curve data
    points: List[DoseResponsePoint] = field(default_factory=list)

    # Fitted parameters (if linear/sigmoidal fit applied)
    slope: Optional[float] = None  # Linear coefficient
    intercept: Optional[float] = None
    saturation_point: Optional[float] = None  # Where effect plateaus
    ec50: Optional[float] = None  # Half-maximal effective concentration

    # Curve characteristics
    is_linear: bool = True  # Linear response?
    is_saturating: bool = False  # Does it plateau?
    is_monotonic: bool = True  # Always increasing/decreasing?

    # Effect direction
    positive_effect: bool = True  # Does increasing strength increase output?

    def get_strengths(self) -> List[float]:
        """Get all tested strengths."""
        return [p.strength for p in self.points]

    def get_effects(self) -> List[float]:
        """Get all mean effects."""
        return [p.mean_effect for p in self.points]

    def predict_effect(self, strength: float) -> float:
        """Predict effect at given strength using fitted curve."""
        if self.slope is not None and self.intercept is not None:
            return self.slope * strength + self.intercept
        # Fallback: linear interpolation
        strengths = self.get_strengths()
        effects = self.get_effects()
        return float(np.interp(strength, strengths, effects))

    def summary(self) -> str:
        """Generate summary string."""
        direction = "+" if self.positive_effect else "-"
        curve_type = "linear" if self.is_linear else "nonlinear"
        saturation = f", saturates at {self.saturation_point:.2f}" if self.is_saturating else ""
        return (
            f"DoseResponse({self.feature_id}: {direction}{curve_type}"
            f", slope={self.slope:.3f}{saturation})"
        )


@dataclass
class InteractionEffect:
    """Interaction between two features during injection."""

    feature_a: str
    feature_b: str

    # Interaction type
    is_synergistic: bool = False  # Combined > sum of parts
    is_antagonistic: bool = False  # Combined < sum of parts
    is_additive: bool = True  # Combined ≈ sum of parts

    # Quantification
    interaction_score: float = 0.0  # Deviation from additivity
    expected_combined: float = 0.0  # Sum of individual effects
    observed_combined: float = 0.0  # Actual combined effect

    # Statistical significance
    p_value: float = 1.0
    is_significant: bool = False


@dataclass
class InjectionResult:
    """Result of feature injection experiment."""

    # Feature(s) injected
    feature_ids: List[str]
    feature_indices: List[int]
    injection_strength: float

    # Measured effects
    output_delta: float  # Change in output
    relative_change: float  # Percentage change
    effect_direction: str  # "increase" or "decrease"

    # Dose-response (if computed)
    dose_response: Optional[DoseResponseCurve] = None

    # Original vs injected outputs
    original_outputs: Optional[torch.Tensor] = None
    injected_outputs: Optional[torch.Tensor] = None

    # Behavioral analysis
    behavioral_shift: Dict[str, float] = field(default_factory=dict)

    # Statistical confidence
    confidence: float = 0.0
    p_value: float = 1.0
    is_significant: bool = False

    # Metadata
    config: Optional[InjectionConfig] = None
    computation_time_ms: float = 0.0

    def summary(self) -> str:
        """Generate summary string."""
        sig = " (significant)" if self.is_significant else ""
        return (
            f"Injection: {self.feature_ids} @ strength={self.injection_strength:.2f}\n"
            f"  Effect: {self.effect_direction} by {self.relative_change:.2%}{sig}\n"
            f"  Output delta: {self.output_delta:.4f}"
        )


class FeatureInjector:
    """
    Performs feature injection experiments for causal validation.

    Injection tests whether a feature is causally sufficient for
    producing a particular effect by force-activating it and
    measuring the behavioral change.

    Usage:
        injector = FeatureInjector(model, probe)

        # Single injection at specific strength
        result = injector.inject_feature("sae.42", strength=0.8, test_inputs)

        # Dose-response curve
        curve = injector.compute_dose_response("sae.42", test_inputs)

        # Multi-feature injection with interaction detection
        result = injector.inject_multiple(
            ["sae.42", "sae.43"],
            strengths=[0.5, 0.5],
            test_inputs
        )
    """

    def __init__(
        self,
        model: Any,
        probe: Any,
        config: Optional[InjectionConfig] = None,
        output_fn: Optional[Callable] = None,
    ):
        """
        Initialize FeatureInjector.

        Args:
            model: Model to inject features into
            probe: MindProbe for activation access
            config: Injection configuration
            output_fn: Optional custom output extraction
        """
        self.model = model
        self.probe = probe
        self.config = config or InjectionConfig.default()
        self.output_fn = output_fn or self._default_output_fn

        # Cache for feature directions (steering vectors)
        self._direction_cache: Dict[str, torch.Tensor] = {}

        # Statistics
        self._injection_count = 0

    def _default_output_fn(self, model_output: Any) -> torch.Tensor:
        """Default output extraction."""
        if isinstance(model_output, torch.Tensor):
            return model_output
        if hasattr(model_output, 'logits'):
            return model_output.logits
        if isinstance(model_output, tuple):
            return model_output[0]
        raise ValueError(f"Cannot extract output from {type(model_output)}")

    def inject_feature(
        self,
        feature_id: str,
        strength: float,
        test_inputs: torch.Tensor,
        method: Optional[InjectionMethod] = None,
    ) -> InjectionResult:
        """
        Inject a feature at specified strength.

        Args:
            feature_id: Feature to inject
            strength: Injection strength (typically 0.0-2.0)
            test_inputs: Test inputs
            method: Injection method

        Returns:
            InjectionResult with effect measurements
        """
        import time
        start_time = time.time()

        method = method or self.config.method
        feature_index = self._parse_feature_index(feature_id)

        # Get original outputs
        original_outputs = self._get_outputs(test_inputs)

        # Get injected outputs
        injected_outputs = self._get_injected_outputs(
            test_inputs, [feature_index], [strength], method
        )

        # Compute effect
        output_delta, relative_change, direction = self._compute_injection_effect(
            original_outputs, injected_outputs
        )

        # Statistical test
        n = original_outputs.numel()
        diff = (injected_outputs.float() - original_outputs.float()).flatten()
        t_stat = diff.mean().abs() / (diff.std() / np.sqrt(n) + 1e-8)

        try:
            from scipy import stats
            p_value = 2 * (1 - stats.t.cdf(t_stat.item(), df=n-1))
        except ImportError:
            p_value = 0.05 if t_stat.item() > 2 else 0.5

        self._injection_count += 1

        return InjectionResult(
            feature_ids=[feature_id],
            feature_indices=[feature_index],
            injection_strength=strength,
            output_delta=output_delta,
            relative_change=relative_change,
            effect_direction=direction,
            original_outputs=original_outputs,
            injected_outputs=injected_outputs,
            confidence=1.0 - p_value,
            p_value=p_value,
            is_significant=p_value < 0.05,
            config=self.config,
            computation_time_ms=(time.time() - start_time) * 1000,
        )

    def compute_dose_response(
        self,
        feature_id: str,
        test_inputs: torch.Tensor,
        method: Optional[InjectionMethod] = None,
    ) -> DoseResponseCurve:
        """
        Compute dose-response curve for a feature.

        Tests the feature at multiple injection strengths to
        characterize the relationship between activation and effect.

        Args:
            feature_id: Feature to test
            test_inputs: Test inputs
            method: Injection method

        Returns:
            DoseResponseCurve with fitted parameters
        """
        method = method or self.config.method
        feature_index = self._parse_feature_index(feature_id)

        min_s, max_s, n_steps = self.config.strength_range
        strengths = np.linspace(min_s, max_s, n_steps)

        # Get original outputs for baseline
        original_outputs = self._get_outputs(test_inputs)
        original_norm = torch.norm(original_outputs.float()).item()

        # Test each strength
        points = []
        effects = []

        for strength in strengths:
            injected_outputs = self._get_injected_outputs(
                test_inputs, [feature_index], [strength], method
            )

            diff = (injected_outputs.float() - original_outputs.float()).flatten()
            mean_effect = diff.mean().item()
            std_effect = diff.std().item()

            # Confidence interval
            n = diff.numel()
            margin = 1.96 * std_effect / np.sqrt(n)
            ci = (mean_effect - margin, mean_effect + margin)

            points.append(DoseResponsePoint(
                strength=float(strength),
                mean_effect=mean_effect,
                std_effect=std_effect,
                confidence_interval=ci,
            ))
            effects.append(mean_effect)

        # Fit linear model
        effects_array = np.array(effects)
        strengths_array = np.array(strengths)

        # Linear regression
        slope, intercept = np.polyfit(strengths_array, effects_array, 1)

        # Check for saturation (diminishing returns at high strengths)
        second_half_slope = np.polyfit(
            strengths_array[n_steps//2:],
            effects_array[n_steps//2:],
            1
        )[0]
        is_saturating = abs(second_half_slope) < abs(slope) * 0.5

        # Check monotonicity
        diffs = np.diff(effects_array)
        is_monotonic = np.all(diffs >= 0) or np.all(diffs <= 0)

        # Check linearity (R^2)
        predicted = slope * strengths_array + intercept
        ss_res = np.sum((effects_array - predicted) ** 2)
        ss_tot = np.sum((effects_array - np.mean(effects_array)) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-8))
        is_linear = r_squared > 0.9

        # EC50 approximation (half-maximal effect)
        max_effect = np.max(np.abs(effects_array))
        half_max = max_effect / 2
        ec50_idx = np.argmin(np.abs(np.abs(effects_array) - half_max))
        ec50 = strengths[ec50_idx]

        # Saturation point
        saturation_point = None
        if is_saturating:
            # Find where effect reaches 90% of max
            threshold = 0.9 * max_effect
            sat_idx = np.argmax(np.abs(effects_array) >= threshold)
            saturation_point = strengths[sat_idx]

        return DoseResponseCurve(
            feature_id=feature_id,
            feature_index=feature_index,
            points=points,
            slope=float(slope),
            intercept=float(intercept),
            saturation_point=saturation_point,
            ec50=float(ec50),
            is_linear=is_linear,
            is_saturating=is_saturating,
            is_monotonic=is_monotonic,
            positive_effect=slope > 0,
        )

    def inject_multiple(
        self,
        feature_ids: List[str],
        strengths: List[float],
        test_inputs: torch.Tensor,
        method: Optional[InjectionMethod] = None,
    ) -> InjectionResult:
        """
        Inject multiple features simultaneously.

        Detects interaction effects (synergy/antagonism).

        Args:
            feature_ids: Features to inject
            strengths: Injection strength per feature
            test_inputs: Test inputs
            method: Injection method

        Returns:
            InjectionResult with interaction analysis
        """
        import time
        start_time = time.time()

        method = method or self.config.method
        feature_indices = [self._parse_feature_index(f) for f in feature_ids]

        # Get original outputs
        original_outputs = self._get_outputs(test_inputs)

        # Get combined injection
        combined_outputs = self._get_injected_outputs(
            test_inputs, feature_indices, strengths, method
        )

        # Compute combined effect
        combined_delta, combined_rel, direction = self._compute_injection_effect(
            original_outputs, combined_outputs
        )

        # Compute individual effects for interaction analysis
        individual_deltas = []
        for feature_id, feature_index, strength in zip(feature_ids, feature_indices, strengths):
            individual_output = self._get_injected_outputs(
                test_inputs, [feature_index], [strength], method
            )
            delta, _, _ = self._compute_injection_effect(
                original_outputs, individual_output
            )
            individual_deltas.append(delta)

        # Interaction: combined vs sum of individuals
        expected_combined = sum(individual_deltas)
        interaction_score = combined_delta - expected_combined

        behavioral_shift = {
            "combined_effect": combined_delta,
            "sum_of_individuals": expected_combined,
            "interaction": interaction_score,
            "is_synergistic": interaction_score > self.config.interaction_threshold,
            "is_antagonistic": interaction_score < -self.config.interaction_threshold,
        }

        self._injection_count += 1

        return InjectionResult(
            feature_ids=feature_ids,
            feature_indices=feature_indices,
            injection_strength=np.mean(strengths),
            output_delta=combined_delta,
            relative_change=combined_rel,
            effect_direction=direction,
            original_outputs=original_outputs,
            injected_outputs=combined_outputs,
            behavioral_shift=behavioral_shift,
            confidence=0.9 if abs(interaction_score) < self.config.interaction_threshold else 0.7,
            config=self.config,
            computation_time_ms=(time.time() - start_time) * 1000,
        )

    def detect_interactions(
        self,
        feature_ids: List[str],
        strength: float,
        test_inputs: torch.Tensor,
    ) -> List[InteractionEffect]:
        """
        Detect pairwise interactions between features.

        Args:
            feature_ids: Features to test
            strength: Injection strength for all
            test_inputs: Test inputs

        Returns:
            List of detected interaction effects
        """
        interactions = []
        original_outputs = self._get_outputs(test_inputs)

        # Get individual effects
        individual_effects = {}
        for feature_id in feature_ids:
            result = self.inject_feature(feature_id, strength, test_inputs)
            individual_effects[feature_id] = result.output_delta

        # Test all pairs
        for i, feat_a in enumerate(feature_ids):
            for feat_b in feature_ids[i+1:]:
                # Combined injection
                result = self.inject_multiple(
                    [feat_a, feat_b],
                    [strength, strength],
                    test_inputs
                )

                expected = individual_effects[feat_a] + individual_effects[feat_b]
                observed = result.output_delta
                interaction_score = observed - expected

                # Classify interaction
                is_synergistic = interaction_score > self.config.interaction_threshold
                is_antagonistic = interaction_score < -self.config.interaction_threshold
                is_additive = not is_synergistic and not is_antagonistic

                # Simple significance test
                is_significant = abs(interaction_score) > self.config.interaction_threshold

                interactions.append(InteractionEffect(
                    feature_a=feat_a,
                    feature_b=feat_b,
                    is_synergistic=is_synergistic,
                    is_antagonistic=is_antagonistic,
                    is_additive=is_additive,
                    interaction_score=interaction_score,
                    expected_combined=expected,
                    observed_combined=observed,
                    is_significant=is_significant,
                ))

        return interactions

    def _get_outputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """Get model outputs for inputs."""
        with torch.no_grad():
            outputs = self.model(inputs)
            return self.output_fn(outputs)

    def _get_injected_outputs(
        self,
        inputs: torch.Tensor,
        feature_indices: List[int],
        strengths: List[float],
        method: InjectionMethod,
    ) -> torch.Tensor:
        """Get model outputs with features injected."""

        def injection_hook(module, input, output):
            for idx, strength in zip(feature_indices, strengths):
                if method == InjectionMethod.ADDITIVE:
                    if output.dim() == 3:
                        output[:, :, idx] = output[:, :, idx] + strength
                    elif output.dim() == 2:
                        output[:, idx] = output[:, idx] + strength

                elif method == InjectionMethod.REPLACEMENT:
                    if output.dim() == 3:
                        output[:, :, idx] = strength
                    elif output.dim() == 2:
                        output[:, idx] = strength

                elif method == InjectionMethod.CLAMPING:
                    if output.dim() == 3:
                        output[:, :, idx] = torch.clamp(
                            output[:, :, idx], min=strength, max=strength
                        )
                    elif output.dim() == 2:
                        output[:, idx] = torch.clamp(
                            output[:, idx], min=strength, max=strength
                        )

                elif method == InjectionMethod.STEERING:
                    # Add steering vector in feature direction
                    direction = self._get_feature_direction(idx, output.shape[-1])
                    direction = direction.to(output.device)
                    if output.dim() == 3:
                        output = output + strength * direction.unsqueeze(0).unsqueeze(0)
                    else:
                        output = output + strength * direction.unsqueeze(0)

            return output

        # Register hook
        if hasattr(self.probe, 'hook_layer'):
            hook = self.probe.hook_layer.register_forward_hook(injection_hook)
        else:
            hook = None

        try:
            with torch.no_grad():
                outputs = self.model(inputs)
                return self.output_fn(outputs)
        finally:
            if hook is not None:
                hook.remove()

    def _get_feature_direction(
        self,
        feature_index: int,
        hidden_dim: int,
    ) -> torch.Tensor:
        """Get steering direction for a feature."""
        cache_key = f"{feature_index}_{hidden_dim}"

        if cache_key not in self._direction_cache:
            # Default: one-hot in feature direction
            direction = torch.zeros(hidden_dim)
            direction[feature_index] = 1.0
            self._direction_cache[cache_key] = direction

        return self._direction_cache[cache_key]

    def _compute_injection_effect(
        self,
        original: torch.Tensor,
        injected: torch.Tensor,
    ) -> Tuple[float, float, str]:
        """Compute effect of injection."""
        diff = injected.float() - original.float()
        output_delta = diff.abs().mean().item()

        original_norm = torch.norm(original.float()).item()
        relative_change = output_delta / (original_norm + 1e-8)

        mean_diff = diff.mean().item()
        direction = "increase" if mean_diff > 0 else "decrease"

        return output_delta, relative_change, direction

    def _parse_feature_index(self, feature_id: str) -> int:
        """Parse feature index from ID."""
        if "." in feature_id:
            return int(feature_id.split(".")[-1])
        return int(feature_id)

    def set_feature_direction(
        self,
        feature_id: str,
        direction: torch.Tensor,
    ) -> None:
        """
        Set custom steering direction for a feature.

        Useful for learned steering vectors.

        Args:
            feature_id: Feature ID
            direction: Steering direction vector
        """
        feature_index = self._parse_feature_index(feature_id)
        cache_key = f"{feature_index}_{direction.shape[0]}"
        self._direction_cache[cache_key] = direction

    def get_statistics(self) -> Dict[str, Any]:
        """Get injection statistics."""
        return {
            "total_injections": self._injection_count,
            "direction_cache_size": len(self._direction_cache),
            "config": {
                "method": self.config.method.value,
                "strength_range": self.config.strength_range,
            },
        }
