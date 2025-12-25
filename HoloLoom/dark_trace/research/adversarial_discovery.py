"""
Adversarial Discovery for Dark Trace

Techniques for finding adversarial inputs that activate, suppress, or
manipulate specific SAE features. Useful for understanding feature
sensitivity and robustness.

Based on:
- "Adversarial Examples for Semantic Segmentation" (Xie et al., 2017)
- "Feature Visualization" (Olah et al., 2017)
- "Scaling Monosemanticity" (Anthropic, 2024)

Author: HoloLoom Team
Created: December 2025
"""

from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json

import numpy as np


class AttackMethod(Enum):
    """Methods for adversarial discovery."""
    FGSM = "fgsm"                   # Fast Gradient Sign Method
    PGD = "pgd"                     # Projected Gradient Descent
    DEEPFOOL = "deepfool"          # DeepFool minimal perturbation
    CW = "cw"                       # Carlini-Wagner attack
    FEATURE_VIS = "feature_vis"    # Feature visualization (maximize activation)
    COUNTERFACTUAL = "counterfactual"  # Minimal change to flip classification


class TargetType(Enum):
    """What to target in adversarial discovery."""
    MAXIMIZE = "maximize"           # Maximize feature activation
    MINIMIZE = "minimize"           # Minimize feature activation
    TARGET_VALUE = "target_value"  # Achieve specific activation value
    FLIP = "flip"                   # Flip from active to inactive or vice versa


@dataclass
class AttackConfig:
    """Configuration for adversarial attacks."""
    method: AttackMethod = AttackMethod.PGD
    target_type: TargetType = TargetType.MAXIMIZE

    # Perturbation budget
    epsilon: float = 0.1            # Max perturbation magnitude (L-inf)
    epsilon_l2: Optional[float] = None  # Optional L2 budget

    # Optimization parameters
    step_size: float = 0.01
    num_steps: int = 100
    random_start: bool = True

    # Convergence criteria
    early_stop_threshold: float = 0.01
    patience: int = 10

    # Regularization
    tv_weight: float = 0.0          # Total variation regularization
    l2_weight: float = 0.0          # L2 regularization on perturbation

    # Target value (for TARGET_VALUE type)
    target_activation: float = 1.0


@dataclass
class AdversarialExample:
    """A discovered adversarial example."""
    original_input: np.ndarray
    adversarial_input: np.ndarray
    perturbation: np.ndarray

    feature_idx: int
    original_activation: float
    adversarial_activation: float

    perturbation_norm_linf: float
    perturbation_norm_l2: float

    method: AttackMethod
    num_iterations: int
    success: bool

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_idx": self.feature_idx,
            "original_activation": self.original_activation,
            "adversarial_activation": self.adversarial_activation,
            "perturbation_norm_linf": self.perturbation_norm_linf,
            "perturbation_norm_l2": self.perturbation_norm_l2,
            "method": self.method.value,
            "num_iterations": self.num_iterations,
            "success": self.success,
            "metadata": self.metadata,
        }


@dataclass
class DiscoveryResult:
    """Results from adversarial discovery."""
    feature_idx: int
    examples: List[AdversarialExample]

    success_rate: float
    avg_perturbation_linf: float
    avg_perturbation_l2: float
    avg_activation_change: float

    most_effective: Optional[AdversarialExample]  # Smallest perturbation
    strongest_effect: Optional[AdversarialExample]  # Largest activation change

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_idx": self.feature_idx,
            "success_rate": self.success_rate,
            "avg_perturbation_linf": self.avg_perturbation_linf,
            "avg_perturbation_l2": self.avg_perturbation_l2,
            "avg_activation_change": self.avg_activation_change,
            "num_examples": len(self.examples),
            "most_effective": self.most_effective.to_dict() if self.most_effective else None,
            "strongest_effect": self.strongest_effect.to_dict() if self.strongest_effect else None,
            "metadata": self.metadata,
        }


class AdversarialAttacker:
    """
    Find adversarial inputs that manipulate specific features.

    Uses gradient-based optimization to find minimal perturbations
    that cause desired changes in feature activations.
    """

    def __init__(self, config: Optional[AttackConfig] = None):
        self.config = config or AttackConfig()

    def attack(
        self,
        input_data: np.ndarray,
        feature_idx: int,
        activation_fn: Callable[[np.ndarray], np.ndarray],
        gradient_fn: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
    ) -> AdversarialExample:
        """
        Find adversarial example for a specific feature.

        Args:
            input_data: Original input (any shape)
            feature_idx: Index of feature to target
            activation_fn: Function that returns feature activations
            gradient_fn: Function that returns gradient w.r.t. input
                        If None, uses numerical gradient

        Returns:
            AdversarialExample with perturbation details
        """
        original_acts = activation_fn(input_data)
        original_activation = float(original_acts[feature_idx]) if feature_idx < len(original_acts) else 0.0

        # Select attack method
        if self.config.method == AttackMethod.FGSM:
            adv_input, iterations = self._fgsm_attack(
                input_data, feature_idx, activation_fn, gradient_fn
            )
        elif self.config.method == AttackMethod.PGD:
            adv_input, iterations = self._pgd_attack(
                input_data, feature_idx, activation_fn, gradient_fn
            )
        elif self.config.method == AttackMethod.FEATURE_VIS:
            adv_input, iterations = self._feature_vis_attack(
                input_data, feature_idx, activation_fn, gradient_fn
            )
        elif self.config.method == AttackMethod.COUNTERFACTUAL:
            adv_input, iterations = self._counterfactual_attack(
                input_data, feature_idx, activation_fn, gradient_fn
            )
        else:
            # Default to PGD
            adv_input, iterations = self._pgd_attack(
                input_data, feature_idx, activation_fn, gradient_fn
            )

        # Compute final activation
        adv_acts = activation_fn(adv_input)
        adv_activation = float(adv_acts[feature_idx]) if feature_idx < len(adv_acts) else 0.0

        # Compute perturbation metrics
        perturbation = adv_input - input_data
        norm_linf = float(np.max(np.abs(perturbation)))
        norm_l2 = float(np.linalg.norm(perturbation))

        # Determine success
        success = self._check_success(original_activation, adv_activation)

        return AdversarialExample(
            original_input=input_data,
            adversarial_input=adv_input,
            perturbation=perturbation,
            feature_idx=feature_idx,
            original_activation=original_activation,
            adversarial_activation=adv_activation,
            perturbation_norm_linf=norm_linf,
            perturbation_norm_l2=norm_l2,
            method=self.config.method,
            num_iterations=iterations,
            success=success,
        )

    def _check_success(self, original: float, adversarial: float) -> bool:
        """Check if attack achieved its goal."""
        if self.config.target_type == TargetType.MAXIMIZE:
            return adversarial > original + self.config.early_stop_threshold
        elif self.config.target_type == TargetType.MINIMIZE:
            return adversarial < original - self.config.early_stop_threshold
        elif self.config.target_type == TargetType.TARGET_VALUE:
            return abs(adversarial - self.config.target_activation) < self.config.early_stop_threshold
        elif self.config.target_type == TargetType.FLIP:
            # Flip from active (>0.5) to inactive (<0.5) or vice versa
            was_active = original > 0.5
            is_active = adversarial > 0.5
            return was_active != is_active
        return False

    def _numerical_gradient(
        self,
        x: np.ndarray,
        feature_idx: int,
        activation_fn: Callable,
        eps: float = 1e-4
    ) -> np.ndarray:
        """Compute numerical gradient for feature activation."""
        grad = np.zeros_like(x, dtype=np.float64)
        x_flat = x.flatten()

        for i in range(len(x_flat)):
            x_plus = x_flat.copy()
            x_minus = x_flat.copy()
            x_plus[i] += eps
            x_minus[i] -= eps

            act_plus = activation_fn(x_plus.reshape(x.shape))
            act_minus = activation_fn(x_minus.reshape(x.shape))

            f_plus = act_plus[feature_idx] if feature_idx < len(act_plus) else 0.0
            f_minus = act_minus[feature_idx] if feature_idx < len(act_minus) else 0.0

            grad.flat[i] = (f_plus - f_minus) / (2 * eps)

        return grad

    def _fgsm_attack(
        self,
        x: np.ndarray,
        feature_idx: int,
        activation_fn: Callable,
        gradient_fn: Optional[Callable],
    ) -> Tuple[np.ndarray, int]:
        """Fast Gradient Sign Method (single step)."""
        # Get gradient
        if gradient_fn is not None:
            grad = gradient_fn(x, feature_idx)
        else:
            grad = self._numerical_gradient(x, feature_idx, activation_fn)

        # Determine direction based on target type
        if self.config.target_type in [TargetType.MAXIMIZE, TargetType.TARGET_VALUE]:
            direction = np.sign(grad)  # Ascend
        else:
            direction = -np.sign(grad)  # Descend

        # Apply perturbation
        perturbation = self.config.epsilon * direction
        adv_x = x + perturbation

        return adv_x, 1

    def _pgd_attack(
        self,
        x: np.ndarray,
        feature_idx: int,
        activation_fn: Callable,
        gradient_fn: Optional[Callable],
    ) -> Tuple[np.ndarray, int]:
        """Projected Gradient Descent attack."""
        # Initialize with random start if configured
        if self.config.random_start:
            adv_x = x + np.random.uniform(
                -self.config.epsilon,
                self.config.epsilon,
                x.shape
            ).astype(x.dtype)
        else:
            adv_x = x.copy()

        best_adv_x = adv_x.copy()
        best_activation = float(activation_fn(adv_x)[feature_idx])
        no_improvement = 0

        for iteration in range(self.config.num_steps):
            # Get gradient
            if gradient_fn is not None:
                grad = gradient_fn(adv_x, feature_idx)
            else:
                grad = self._numerical_gradient(adv_x, feature_idx, activation_fn)

            # Determine direction
            if self.config.target_type == TargetType.MAXIMIZE:
                step = self.config.step_size * np.sign(grad)
            elif self.config.target_type == TargetType.MINIMIZE:
                step = -self.config.step_size * np.sign(grad)
            elif self.config.target_type == TargetType.TARGET_VALUE:
                current_act = activation_fn(adv_x)[feature_idx]
                if current_act < self.config.target_activation:
                    step = self.config.step_size * np.sign(grad)
                else:
                    step = -self.config.step_size * np.sign(grad)
            else:
                step = self.config.step_size * np.sign(grad)

            # Add regularization gradients
            if self.config.l2_weight > 0:
                step -= self.config.l2_weight * (adv_x - x)

            # Update
            adv_x = adv_x + step

            # Project back to epsilon ball
            perturbation = adv_x - x
            perturbation = np.clip(perturbation, -self.config.epsilon, self.config.epsilon)
            adv_x = x + perturbation

            # Track best
            current_activation = float(activation_fn(adv_x)[feature_idx])

            improved = False
            if self.config.target_type == TargetType.MAXIMIZE:
                if current_activation > best_activation:
                    best_activation = current_activation
                    best_adv_x = adv_x.copy()
                    improved = True
            elif self.config.target_type == TargetType.MINIMIZE:
                if current_activation < best_activation:
                    best_activation = current_activation
                    best_adv_x = adv_x.copy()
                    improved = True
            elif self.config.target_type == TargetType.TARGET_VALUE:
                target_dist = abs(current_activation - self.config.target_activation)
                best_dist = abs(best_activation - self.config.target_activation)
                if target_dist < best_dist:
                    best_activation = current_activation
                    best_adv_x = adv_x.copy()
                    improved = True

            # Early stopping
            if improved:
                no_improvement = 0
            else:
                no_improvement += 1

            if no_improvement >= self.config.patience:
                return best_adv_x, iteration + 1

        return best_adv_x, self.config.num_steps

    def _feature_vis_attack(
        self,
        x: np.ndarray,
        feature_idx: int,
        activation_fn: Callable,
        gradient_fn: Optional[Callable],
    ) -> Tuple[np.ndarray, int]:
        """
        Feature visualization - maximize activation without epsilon constraint.

        Used for understanding what activates a feature maximally.
        """
        adv_x = x.copy()

        for iteration in range(self.config.num_steps):
            # Get gradient
            if gradient_fn is not None:
                grad = gradient_fn(adv_x, feature_idx)
            else:
                grad = self._numerical_gradient(adv_x, feature_idx, activation_fn)

            # Normalize gradient for stable optimization
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-8:
                grad = grad / grad_norm

            # Update (always ascend for feature visualization)
            adv_x = adv_x + self.config.step_size * grad

            # Apply regularization
            if self.config.l2_weight > 0:
                adv_x = adv_x * (1 - self.config.l2_weight)

            if self.config.tv_weight > 0:
                # Total variation denoising (simplified)
                adv_x = self._apply_tv_regularization(adv_x, self.config.tv_weight)

        return adv_x, self.config.num_steps

    def _counterfactual_attack(
        self,
        x: np.ndarray,
        feature_idx: int,
        activation_fn: Callable,
        gradient_fn: Optional[Callable],
    ) -> Tuple[np.ndarray, int]:
        """
        Find minimal perturbation to flip feature activation.

        Uses adaptive step size to find smallest effective perturbation.
        """
        original_act = activation_fn(x)[feature_idx]
        was_active = original_act > 0.5

        adv_x = x.copy()
        step_size = self.config.step_size

        for iteration in range(self.config.num_steps):
            # Get gradient
            if gradient_fn is not None:
                grad = gradient_fn(adv_x, feature_idx)
            else:
                grad = self._numerical_gradient(adv_x, feature_idx, activation_fn)

            # Direction: decrease if active, increase if inactive
            if was_active:
                direction = -np.sign(grad)
            else:
                direction = np.sign(grad)

            # Adaptive step with L2 constraint
            adv_x = adv_x + step_size * direction

            # Check if flipped
            current_act = activation_fn(adv_x)[feature_idx]
            is_active = current_act > 0.5

            if was_active != is_active:
                # Success! Binary search for minimal perturbation
                return self._binary_search_minimal(
                    x, adv_x, feature_idx, activation_fn, was_active
                )

            # Decay step size if not making progress
            step_size *= 0.99

        return adv_x, self.config.num_steps

    def _binary_search_minimal(
        self,
        original: np.ndarray,
        adversarial: np.ndarray,
        feature_idx: int,
        activation_fn: Callable,
        was_active: bool,
        num_steps: int = 10,
    ) -> Tuple[np.ndarray, int]:
        """Binary search for minimal effective perturbation."""
        lo = 0.0
        hi = 1.0

        for _ in range(num_steps):
            mid = (lo + hi) / 2
            test_x = original + mid * (adversarial - original)
            test_act = activation_fn(test_x)[feature_idx]
            is_active = test_act > 0.5

            if was_active != is_active:
                hi = mid  # Still flipped, try smaller
            else:
                lo = mid  # Not flipped, need more perturbation

        # Return smallest perturbation that still works
        return original + hi * (adversarial - original), num_steps

    def _apply_tv_regularization(self, x: np.ndarray, weight: float) -> np.ndarray:
        """Apply total variation regularization (simplified 1D version)."""
        if x.ndim == 1:
            # Simple smoothing for 1D
            x_smooth = x.copy()
            x_smooth[1:-1] = x[1:-1] - weight * (
                2 * x[1:-1] - x[:-2] - x[2:]
            )
            return x_smooth
        return x  # No TV for higher dimensions in simple version


class FeatureSensitivityAnalyzer:
    """
    Analyze feature sensitivity to input perturbations.

    Discovers which input dimensions most strongly affect each feature.
    """

    def __init__(self, config: Optional[AttackConfig] = None):
        self.config = config or AttackConfig()
        self.attacker = AdversarialAttacker(config)

    def analyze_sensitivity(
        self,
        inputs: np.ndarray,
        feature_idx: int,
        activation_fn: Callable[[np.ndarray], np.ndarray],
        gradient_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Analyze sensitivity of a feature to input perturbations.

        Args:
            inputs: Batch of inputs (n_samples, input_dim)
            feature_idx: Feature to analyze
            activation_fn: Function returning activations
            gradient_fn: Optional gradient function

        Returns:
            Sensitivity analysis results
        """
        n_samples = len(inputs)
        input_dim = inputs.shape[1] if inputs.ndim > 1 else len(inputs[0])

        # Collect gradients across samples
        gradients = []
        activations = []

        for x in inputs:
            # Get activation
            act = activation_fn(x.reshape(1, -1) if x.ndim == 1 else x[np.newaxis])
            activations.append(float(act[0, feature_idx]) if act.ndim > 1 else float(act[feature_idx]))

            # Get gradient
            if gradient_fn is not None:
                grad = gradient_fn(x, feature_idx)
            else:
                grad = self.attacker._numerical_gradient(x, feature_idx, activation_fn)

            gradients.append(grad.flatten())

        gradients = np.array(gradients)
        activations = np.array(activations)

        # Compute sensitivity metrics
        mean_gradient = np.mean(np.abs(gradients), axis=0)
        max_gradient = np.max(np.abs(gradients), axis=0)
        gradient_variance = np.var(gradients, axis=0)

        # Find most sensitive dimensions
        sensitivity_ranking = np.argsort(mean_gradient)[::-1]

        # Correlation between gradient magnitude and activation
        grad_magnitudes = np.linalg.norm(gradients, axis=1)
        if np.std(grad_magnitudes) > 1e-8 and np.std(activations) > 1e-8:
            correlation = float(np.corrcoef(grad_magnitudes, activations)[0, 1])
        else:
            correlation = 0.0

        return {
            "feature_idx": feature_idx,
            "n_samples": n_samples,
            "mean_activation": float(np.mean(activations)),
            "std_activation": float(np.std(activations)),
            "mean_gradient_magnitude": float(np.mean(grad_magnitudes)),
            "top_sensitive_dims": sensitivity_ranking[:10].tolist(),
            "dim_sensitivities": mean_gradient.tolist(),
            "gradient_activation_correlation": correlation,
            "metadata": {
                "max_gradient_per_dim": max_gradient.tolist(),
                "gradient_variance_per_dim": gradient_variance.tolist(),
            }
        }

    def find_adversarial_directions(
        self,
        inputs: np.ndarray,
        feature_idx: int,
        activation_fn: Callable,
        n_directions: int = 5,
    ) -> List[np.ndarray]:
        """
        Find principal adversarial directions for a feature.

        Uses PCA on gradients to find most effective perturbation directions.
        """
        # Collect gradients
        gradients = []
        for x in inputs:
            grad = self.attacker._numerical_gradient(x, feature_idx, activation_fn)
            gradients.append(grad.flatten())

        gradients = np.array(gradients)

        # Center gradients
        gradients_centered = gradients - np.mean(gradients, axis=0)

        # SVD to find principal directions
        if gradients_centered.shape[0] > 1:
            U, S, Vh = np.linalg.svd(gradients_centered, full_matrices=False)

            # Top directions
            directions = []
            for i in range(min(n_directions, Vh.shape[0])):
                direction = Vh[i]
                direction = direction / (np.linalg.norm(direction) + 1e-8)
                directions.append(direction)

            return directions

        # Fallback: return mean gradient direction
        mean_grad = np.mean(gradients, axis=0)
        mean_grad = mean_grad / (np.linalg.norm(mean_grad) + 1e-8)
        return [mean_grad]


class AdversarialDiscoverer:
    """
    Main class for systematic adversarial discovery across features.

    Finds and catalogs adversarial examples for understanding
    feature robustness and sensitivity.
    """

    def __init__(self, config: Optional[AttackConfig] = None):
        self.config = config or AttackConfig()
        self.attacker = AdversarialAttacker(config)
        self.sensitivity_analyzer = FeatureSensitivityAnalyzer(config)
        self._discovered: Dict[int, DiscoveryResult] = {}

    def discover_for_feature(
        self,
        inputs: np.ndarray,
        feature_idx: int,
        activation_fn: Callable,
        gradient_fn: Optional[Callable] = None,
        num_examples: int = 10,
    ) -> DiscoveryResult:
        """
        Discover adversarial examples for a specific feature.

        Args:
            inputs: Candidate inputs to perturb
            feature_idx: Target feature index
            activation_fn: Function returning feature activations
            gradient_fn: Optional gradient function
            num_examples: Max examples to generate

        Returns:
            DiscoveryResult with adversarial examples
        """
        examples = []

        # Select diverse inputs
        selected_inputs = self._select_diverse_inputs(inputs, num_examples)

        for x in selected_inputs:
            example = self.attacker.attack(
                x, feature_idx, activation_fn, gradient_fn
            )
            examples.append(example)

        # Compute statistics
        successful = [e for e in examples if e.success]
        success_rate = len(successful) / len(examples) if examples else 0.0

        avg_linf = float(np.mean([e.perturbation_norm_linf for e in examples]))
        avg_l2 = float(np.mean([e.perturbation_norm_l2 for e in examples]))
        avg_change = float(np.mean([
            abs(e.adversarial_activation - e.original_activation)
            for e in examples
        ]))

        # Find most effective (smallest perturbation among successful)
        most_effective = None
        if successful:
            most_effective = min(successful, key=lambda e: e.perturbation_norm_l2)

        # Find strongest effect
        strongest = max(examples, key=lambda e: abs(
            e.adversarial_activation - e.original_activation
        ))

        result = DiscoveryResult(
            feature_idx=feature_idx,
            examples=examples,
            success_rate=success_rate,
            avg_perturbation_linf=avg_linf,
            avg_perturbation_l2=avg_l2,
            avg_activation_change=avg_change,
            most_effective=most_effective,
            strongest_effect=strongest,
        )

        self._discovered[feature_idx] = result
        return result

    def discover_all_features(
        self,
        inputs: np.ndarray,
        activation_fn: Callable,
        feature_indices: Optional[List[int]] = None,
        gradient_fn: Optional[Callable] = None,
        num_examples_per_feature: int = 5,
    ) -> Dict[int, DiscoveryResult]:
        """
        Discover adversarial examples for multiple features.

        Args:
            inputs: Candidate inputs
            activation_fn: Function returning feature activations
            feature_indices: Features to analyze (None = all)
            gradient_fn: Optional gradient function
            num_examples_per_feature: Examples per feature

        Returns:
            Dictionary mapping feature index to DiscoveryResult
        """
        # Determine features to analyze
        if feature_indices is None:
            sample_acts = activation_fn(inputs[0:1])
            n_features = sample_acts.shape[-1] if sample_acts.ndim > 1 else len(sample_acts)
            feature_indices = list(range(n_features))

        results = {}
        for feature_idx in feature_indices:
            result = self.discover_for_feature(
                inputs, feature_idx, activation_fn, gradient_fn,
                num_examples_per_feature
            )
            results[feature_idx] = result

        return results

    def find_universal_adversarial_perturbation(
        self,
        inputs: np.ndarray,
        feature_indices: List[int],
        activation_fn: Callable,
        target_type: TargetType = TargetType.MINIMIZE,
    ) -> np.ndarray:
        """
        Find a single perturbation that affects multiple features.

        This is input-agnostic and works across different inputs.
        """
        # Compute average gradient across inputs and features
        avg_grad = np.zeros(inputs[0].shape)

        for x in inputs:
            for feature_idx in feature_indices:
                grad = self.attacker._numerical_gradient(x, feature_idx, activation_fn)
                avg_grad += grad

        avg_grad /= (len(inputs) * len(feature_indices))

        # Normalize and scale
        if np.linalg.norm(avg_grad) > 1e-8:
            avg_grad = avg_grad / np.linalg.norm(avg_grad)

        # Direction based on target
        if target_type in [TargetType.MAXIMIZE]:
            universal_perturbation = self.config.epsilon * avg_grad
        else:
            universal_perturbation = -self.config.epsilon * avg_grad

        return universal_perturbation

    def _select_diverse_inputs(
        self,
        inputs: np.ndarray,
        n: int,
    ) -> np.ndarray:
        """Select diverse inputs for better coverage."""
        if len(inputs) <= n:
            return inputs

        # Simple diversity: select based on L2 distance
        selected_indices = [0]

        for _ in range(n - 1):
            max_min_dist = -1
            best_idx = 0

            for i in range(len(inputs)):
                if i in selected_indices:
                    continue

                # Distance to nearest selected
                min_dist = min(
                    np.linalg.norm(inputs[i] - inputs[j])
                    for j in selected_indices
                )

                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = i

            selected_indices.append(best_idx)

        return inputs[selected_indices]

    def get_robustness_summary(self) -> Dict[str, Any]:
        """Summarize robustness across all discovered features."""
        if not self._discovered:
            return {"error": "No features discovered yet"}

        success_rates = [r.success_rate for r in self._discovered.values()]
        avg_perturbations = [r.avg_perturbation_l2 for r in self._discovered.values()]

        # Find most and least robust features
        most_robust = max(
            self._discovered.items(),
            key=lambda x: x[1].avg_perturbation_l2
        )
        least_robust = min(
            self._discovered.items(),
            key=lambda x: x[1].avg_perturbation_l2
        )

        return {
            "n_features_analyzed": len(self._discovered),
            "avg_attack_success_rate": float(np.mean(success_rates)),
            "avg_perturbation_required": float(np.mean(avg_perturbations)),
            "most_robust_feature": {
                "idx": most_robust[0],
                "avg_perturbation": most_robust[1].avg_perturbation_l2,
            },
            "least_robust_feature": {
                "idx": least_robust[0],
                "avg_perturbation": least_robust[1].avg_perturbation_l2,
            },
        }

    def export_results(self) -> Dict[str, Any]:
        """Export all discovery results."""
        return {
            str(idx): result.to_dict()
            for idx, result in self._discovered.items()
        }


# Factory functions
def create_attacker(
    method: AttackMethod = AttackMethod.PGD,
    target: TargetType = TargetType.MAXIMIZE,
    epsilon: float = 0.1,
) -> AdversarialAttacker:
    """Create an adversarial attacker with common settings."""
    config = AttackConfig(
        method=method,
        target_type=target,
        epsilon=epsilon,
    )
    return AdversarialAttacker(config)


def create_discoverer(
    method: AttackMethod = AttackMethod.PGD,
    epsilon: float = 0.1,
) -> AdversarialDiscoverer:
    """Create an adversarial discoverer."""
    config = AttackConfig(method=method, epsilon=epsilon)
    return AdversarialDiscoverer(config)


def create_sensitivity_analyzer() -> FeatureSensitivityAnalyzer:
    """Create a feature sensitivity analyzer."""
    return FeatureSensitivityAnalyzer()
