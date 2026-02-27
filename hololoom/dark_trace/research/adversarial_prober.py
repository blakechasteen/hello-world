"""
Dark Trace SAE: Adversarial Probing Suite
==========================================

Systematic vulnerability testing for SAE features using gradient-based
and gradient-free probing methods. Integrates with HoloLoom's alignment
framework for risk-aware probing.

Key Components:
- AdversarialProber: Main prober combining gradient and genetic methods
- GradientProber: FGSM, PGD gradient-based probing
- VulnerabilityReport: Per-feature vulnerability assessment
- ProbeConfig: Configuration for probing behavior

Research Basis:
- "Towards Robust and Verified AI" (Anthropic, 2023)
- "Adversarial Examples in Deep Learning" (Goodfellow et al., 2015)
- "Certified Robustness to Adversarial Word Substitutions" (Jia et al., 2019)

Usage:
    from hololoom.dark_trace.research import (
        AdversarialProber,
        ProbeConfig,
        VulnerabilityReport,
    )

    # Create prober with safety guardrails
    config = ProbeConfig.balanced()
    prober = AdversarialProber(config, model_adapter)

    # Probe specific features
    report = prober.probe_feature(42, test_inputs)
    print(f"Feature 42 vulnerability: {report.vulnerability_score:.3f}")

    # Probe all features
    reports = prober.probe_all_features(test_inputs, top_k=100)
    vulnerable = [r for r in reports if r.is_vulnerable]

Created: 2025-12-28
"""

from typing import Any, Dict, List, Optional, Tuple, Callable, Union, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
import time

import numpy as np

if TYPE_CHECKING:
    from hololoom.alignment import SafetyGuardrails
    from hololoom.dark_trace.models import ModelAdapter


class ProbeMethod(Enum):
    """Methods for adversarial probing."""
    GRADIENT = "gradient"           # Gradient-based (FGSM, PGD)
    GENETIC = "genetic"             # Genetic algorithm search
    RANDOM = "random"               # Random perturbation baseline
    HYBRID = "hybrid"               # Gradient + Genetic combined
    TARGETED = "targeted"           # Target specific activation levels


class VulnerabilityLevel(Enum):
    """Feature vulnerability classification."""
    ROBUST = "robust"               # Resistant to adversarial perturbations
    LOW = "low"                     # Minor vulnerability
    MEDIUM = "medium"               # Moderate vulnerability
    HIGH = "high"                   # High vulnerability
    CRITICAL = "critical"           # Easily manipulated


@dataclass
class ProbeConfig:
    """Configuration for adversarial probing."""
    # Probing method
    method: ProbeMethod = ProbeMethod.HYBRID

    # Perturbation bounds
    epsilon: float = 0.1           # Max L-inf perturbation
    epsilon_l2: Optional[float] = None  # Optional L2 bound

    # Gradient-based parameters
    gradient_steps: int = 50       # Steps for PGD
    gradient_step_size: float = 0.01
    use_pgd: bool = True           # PGD vs FGSM

    # Genetic algorithm parameters
    population_size: int = 50
    generations: int = 30
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7

    # Convergence
    early_stop_threshold: float = 0.01
    patience: int = 5

    # Vulnerability thresholds
    vulnerability_threshold: float = 0.3  # Activation change for "vulnerable"
    critical_threshold: float = 0.7       # Activation change for "critical"

    # Safety integration
    enable_safety_checks: bool = True
    max_probes_per_feature: int = 20

    # Computational limits
    timeout_seconds: float = 60.0  # Per-feature timeout

    @classmethod
    def quick(cls) -> "ProbeConfig":
        """Fast probing for initial assessment."""
        return cls(
            method=ProbeMethod.GRADIENT,
            gradient_steps=20,
            population_size=20,
            generations=10,
            max_probes_per_feature=10,
        )

    @classmethod
    def balanced(cls) -> "ProbeConfig":
        """Balanced speed/thoroughness for production."""
        return cls(
            method=ProbeMethod.HYBRID,
            gradient_steps=50,
            population_size=50,
            generations=30,
            max_probes_per_feature=20,
        )

    @classmethod
    def thorough(cls) -> "ProbeConfig":
        """Thorough probing for security assessment."""
        return cls(
            method=ProbeMethod.HYBRID,
            gradient_steps=100,
            population_size=100,
            generations=50,
            max_probes_per_feature=50,
            timeout_seconds=120.0,
        )


@dataclass
class ProbeResult:
    """Result from a single probe attempt."""
    original_input: np.ndarray
    perturbed_input: np.ndarray
    original_activation: float
    perturbed_activation: float
    perturbation_norm_linf: float
    perturbation_norm_l2: float
    success: bool                   # Achieved target change
    method_used: ProbeMethod
    iterations: int

    @property
    def activation_change(self) -> float:
        """Absolute activation change."""
        return abs(self.perturbed_activation - self.original_activation)

    @property
    def relative_change(self) -> float:
        """Relative activation change."""
        if abs(self.original_activation) < 1e-8:
            return self.activation_change
        return self.activation_change / abs(self.original_activation)


@dataclass
class VulnerabilityReport:
    """Comprehensive vulnerability report for a feature."""
    feature_idx: int

    # Probe results
    probes: List[ProbeResult] = field(default_factory=list)

    # Summary statistics
    vulnerability_score: float = 0.0  # 0.0 (robust) to 1.0 (critical)
    vulnerability_level: VulnerabilityLevel = VulnerabilityLevel.ROBUST

    # Aggregated metrics
    avg_perturbation_linf: float = 0.0
    avg_perturbation_l2: float = 0.0
    avg_activation_change: float = 0.0
    success_rate: float = 0.0
    min_effective_perturbation: float = float('inf')

    # Most effective attacks
    most_effective_probe: Optional[ProbeResult] = None  # Smallest perturbation
    strongest_probe: Optional[ProbeResult] = None       # Largest effect

    # Timing
    probe_duration_seconds: float = 0.0

    # Metadata
    config_used: Optional[ProbeConfig] = None
    safety_blocked: bool = False
    error_message: Optional[str] = None

    @property
    def is_vulnerable(self) -> bool:
        """Check if feature is considered vulnerable."""
        return self.vulnerability_level in [
            VulnerabilityLevel.MEDIUM,
            VulnerabilityLevel.HIGH,
            VulnerabilityLevel.CRITICAL,
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary."""
        return {
            "feature_idx": self.feature_idx,
            "vulnerability_score": self.vulnerability_score,
            "vulnerability_level": self.vulnerability_level.value,
            "avg_perturbation_linf": self.avg_perturbation_linf,
            "avg_perturbation_l2": self.avg_perturbation_l2,
            "avg_activation_change": self.avg_activation_change,
            "success_rate": self.success_rate,
            "min_effective_perturbation": self.min_effective_perturbation,
            "num_probes": len(self.probes),
            "probe_duration_seconds": self.probe_duration_seconds,
            "is_vulnerable": self.is_vulnerable,
            "safety_blocked": self.safety_blocked,
        }


class GradientProber:
    """
    Gradient-based adversarial probing using FGSM and PGD.

    Efficient probing when gradients are available.
    """

    def __init__(self, config: ProbeConfig):
        self.config = config

    def probe(
        self,
        input_data: np.ndarray,
        feature_idx: int,
        activation_fn: Callable[[np.ndarray], np.ndarray],
        gradient_fn: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
        target_type: str = "maximize",  # "maximize", "minimize", "flip"
    ) -> ProbeResult:
        """
        Probe feature vulnerability using gradient-based methods.

        Args:
            input_data: Original input
            feature_idx: Feature to probe
            activation_fn: Returns feature activations
            gradient_fn: Returns gradient w.r.t. input (optional)
            target_type: What to achieve ("maximize", "minimize", "flip")

        Returns:
            ProbeResult with perturbation details
        """
        original_acts = activation_fn(input_data)
        original_activation = self._get_activation(original_acts, feature_idx)

        if self.config.use_pgd:
            perturbed, iterations = self._pgd_probe(
                input_data, feature_idx, activation_fn, gradient_fn, target_type
            )
        else:
            perturbed, iterations = self._fgsm_probe(
                input_data, feature_idx, activation_fn, gradient_fn, target_type
            )

        perturbed_acts = activation_fn(perturbed)
        perturbed_activation = self._get_activation(perturbed_acts, feature_idx)

        perturbation = perturbed - input_data
        norm_linf = float(np.max(np.abs(perturbation)))
        norm_l2 = float(np.linalg.norm(perturbation))

        success = self._check_success(
            original_activation, perturbed_activation, target_type
        )

        return ProbeResult(
            original_input=input_data,
            perturbed_input=perturbed,
            original_activation=original_activation,
            perturbed_activation=perturbed_activation,
            perturbation_norm_linf=norm_linf,
            perturbation_norm_l2=norm_l2,
            success=success,
            method_used=ProbeMethod.GRADIENT,
            iterations=iterations,
        )

    def _get_activation(self, activations: np.ndarray, idx: int) -> float:
        """Extract activation value safely."""
        if activations.ndim > 1:
            return float(activations[0, idx]) if idx < activations.shape[-1] else 0.0
        return float(activations[idx]) if idx < len(activations) else 0.0

    def _check_success(
        self, original: float, perturbed: float, target_type: str
    ) -> bool:
        """Check if probe achieved target."""
        change = perturbed - original
        threshold = self.config.vulnerability_threshold

        if target_type == "maximize":
            return change > threshold
        elif target_type == "minimize":
            return change < -threshold
        elif target_type == "flip":
            return (original > 0.5) != (perturbed > 0.5)
        return False

    def _numerical_gradient(
        self,
        x: np.ndarray,
        feature_idx: int,
        activation_fn: Callable,
        eps: float = 1e-4,
    ) -> np.ndarray:
        """Compute numerical gradient."""
        grad = np.zeros_like(x, dtype=np.float64)
        x_flat = x.flatten()

        for i in range(len(x_flat)):
            x_plus = x_flat.copy()
            x_minus = x_flat.copy()
            x_plus[i] += eps
            x_minus[i] -= eps

            act_plus = activation_fn(x_plus.reshape(x.shape))
            act_minus = activation_fn(x_minus.reshape(x.shape))

            f_plus = self._get_activation(act_plus, feature_idx)
            f_minus = self._get_activation(act_minus, feature_idx)

            grad.flat[i] = (f_plus - f_minus) / (2 * eps)

        return grad

    def _fgsm_probe(
        self,
        x: np.ndarray,
        feature_idx: int,
        activation_fn: Callable,
        gradient_fn: Optional[Callable],
        target_type: str,
    ) -> Tuple[np.ndarray, int]:
        """Fast Gradient Sign Method probe."""
        if gradient_fn is not None:
            grad = gradient_fn(x, feature_idx)
        else:
            grad = self._numerical_gradient(x, feature_idx, activation_fn)

        if target_type == "maximize":
            direction = np.sign(grad)
        elif target_type == "minimize":
            direction = -np.sign(grad)
        else:
            direction = np.sign(grad)

        perturbed = x + self.config.epsilon * direction
        return perturbed, 1

    def _pgd_probe(
        self,
        x: np.ndarray,
        feature_idx: int,
        activation_fn: Callable,
        gradient_fn: Optional[Callable],
        target_type: str,
    ) -> Tuple[np.ndarray, int]:
        """Projected Gradient Descent probe."""
        perturbed = x + np.random.uniform(
            -self.config.epsilon * 0.1,
            self.config.epsilon * 0.1,
            x.shape
        ).astype(x.dtype)

        best_perturbed = perturbed.copy()
        best_activation = self._get_activation(activation_fn(perturbed), feature_idx)
        no_improvement = 0

        for iteration in range(self.config.gradient_steps):
            if gradient_fn is not None:
                grad = gradient_fn(perturbed, feature_idx)
            else:
                grad = self._numerical_gradient(perturbed, feature_idx, activation_fn)

            if target_type == "maximize":
                step = self.config.gradient_step_size * np.sign(grad)
            elif target_type == "minimize":
                step = -self.config.gradient_step_size * np.sign(grad)
            else:
                current_act = self._get_activation(activation_fn(perturbed), feature_idx)
                if current_act > 0.5:
                    step = -self.config.gradient_step_size * np.sign(grad)
                else:
                    step = self.config.gradient_step_size * np.sign(grad)

            perturbed = perturbed + step

            perturbation = perturbed - x
            perturbation = np.clip(perturbation, -self.config.epsilon, self.config.epsilon)
            perturbed = x + perturbation

            current_activation = self._get_activation(activation_fn(perturbed), feature_idx)

            improved = False
            if target_type == "maximize" and current_activation > best_activation:
                best_activation = current_activation
                best_perturbed = perturbed.copy()
                improved = True
            elif target_type == "minimize" and current_activation < best_activation:
                best_activation = current_activation
                best_perturbed = perturbed.copy()
                improved = True
            elif target_type == "flip":
                original = self._get_activation(activation_fn(x), feature_idx)
                if (original > 0.5) != (current_activation > 0.5):
                    return perturbed, iteration + 1

            if improved:
                no_improvement = 0
            else:
                no_improvement += 1

            if no_improvement >= self.config.patience:
                return best_perturbed, iteration + 1

        return best_perturbed, self.config.gradient_steps


class AdversarialProber:
    """
    Main adversarial probing system combining gradient and genetic methods.

    Systematically tests features for adversarial vulnerabilities and
    integrates with HoloLoom's alignment framework for safety-aware probing.
    """

    def __init__(
        self,
        config: Optional[ProbeConfig] = None,
        model_adapter: Optional["ModelAdapter"] = None,
        safety_guardrails: Optional["SafetyGuardrails"] = None,
    ):
        """
        Initialize adversarial prober.

        Args:
            config: Probing configuration
            model_adapter: Model adapter for activation extraction
            safety_guardrails: Optional safety integration
        """
        self.config = config or ProbeConfig.balanced()
        self.model_adapter = model_adapter
        self.safety_guardrails = safety_guardrails

        self.gradient_prober = GradientProber(self.config)
        self._genetic_prober = None  # Lazy init

        self._reports: Dict[int, VulnerabilityReport] = {}

    @property
    def genetic_prober(self):
        """Lazy initialization of genetic prober."""
        if self._genetic_prober is None:
            from hololoom.dark_trace.research.genetic_prober import GeneticAdversarialSearch
            self._genetic_prober = GeneticAdversarialSearch(self.config)
        return self._genetic_prober

    def probe_feature(
        self,
        feature_idx: int,
        test_inputs: np.ndarray,
        activation_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        gradient_fn: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
    ) -> VulnerabilityReport:
        """
        Probe a specific feature for vulnerabilities.

        Args:
            feature_idx: Feature index to probe
            test_inputs: Test inputs (n_samples, input_dim)
            activation_fn: Optional custom activation function
            gradient_fn: Optional gradient function

        Returns:
            VulnerabilityReport with assessment
        """
        start_time = time.time()

        if self.config.enable_safety_checks and self.safety_guardrails:
            if not self._check_safety(feature_idx):
                return VulnerabilityReport(
                    feature_idx=feature_idx,
                    safety_blocked=True,
                    error_message="Probing blocked by safety guardrails",
                )

        if activation_fn is None:
            if self.model_adapter is None:
                return VulnerabilityReport(
                    feature_idx=feature_idx,
                    error_message="No activation function or model adapter provided",
                )
            activation_fn = lambda x: self.model_adapter.get_activations(x, layer="sae")

        probes: List[ProbeResult] = []
        n_inputs = min(len(test_inputs), self.config.max_probes_per_feature)

        for i in range(n_inputs):
            if time.time() - start_time > self.config.timeout_seconds:
                break

            x = test_inputs[i]

            if self.config.method in [ProbeMethod.GRADIENT, ProbeMethod.HYBRID]:
                probe_result = self.gradient_prober.probe(
                    x, feature_idx, activation_fn, gradient_fn, "maximize"
                )
                probes.append(probe_result)

                probe_result_min = self.gradient_prober.probe(
                    x, feature_idx, activation_fn, gradient_fn, "minimize"
                )
                probes.append(probe_result_min)

            if self.config.method in [ProbeMethod.GENETIC, ProbeMethod.HYBRID]:
                try:
                    genetic_result = self.genetic_prober.search(
                        x, feature_idx, activation_fn, "maximize"
                    )
                    probes.append(genetic_result)
                except Exception:
                    pass

            if self.config.method == ProbeMethod.RANDOM:
                random_result = self._random_probe(x, feature_idx, activation_fn)
                probes.append(random_result)

        report = self._compile_report(feature_idx, probes, start_time)
        self._reports[feature_idx] = report
        return report

    def probe_all_features(
        self,
        test_inputs: np.ndarray,
        feature_indices: Optional[List[int]] = None,
        activation_fn: Optional[Callable] = None,
        top_k: Optional[int] = None,
    ) -> List[VulnerabilityReport]:
        """
        Probe multiple features for vulnerabilities.

        Args:
            test_inputs: Test inputs
            feature_indices: Features to probe (None = infer from first input)
            activation_fn: Optional custom activation function
            top_k: Only probe top K features (by some criterion)

        Returns:
            List of VulnerabilityReports sorted by vulnerability score
        """
        if activation_fn is None and self.model_adapter:
            activation_fn = lambda x: self.model_adapter.get_activations(x, layer="sae")

        if feature_indices is None:
            sample_acts = activation_fn(test_inputs[0:1])
            n_features = sample_acts.shape[-1] if sample_acts.ndim > 1 else len(sample_acts)
            feature_indices = list(range(n_features))

        if top_k and top_k < len(feature_indices):
            feature_indices = feature_indices[:top_k]

        reports = []
        for feature_idx in feature_indices:
            report = self.probe_feature(
                feature_idx, test_inputs, activation_fn
            )
            reports.append(report)

        reports.sort(key=lambda r: r.vulnerability_score, reverse=True)
        return reports

    def get_vulnerability_summary(self) -> Dict[str, Any]:
        """
        Summarize vulnerabilities across all probed features.

        Returns:
            Summary statistics
        """
        if not self._reports:
            return {"error": "No features probed yet"}

        reports = list(self._reports.values())
        vulnerability_scores = [r.vulnerability_score for r in reports]

        level_counts = {
            VulnerabilityLevel.ROBUST: 0,
            VulnerabilityLevel.LOW: 0,
            VulnerabilityLevel.MEDIUM: 0,
            VulnerabilityLevel.HIGH: 0,
            VulnerabilityLevel.CRITICAL: 0,
        }
        for r in reports:
            level_counts[r.vulnerability_level] += 1

        most_vulnerable = max(reports, key=lambda r: r.vulnerability_score)
        most_robust = min(reports, key=lambda r: r.vulnerability_score)

        return {
            "n_features_probed": len(reports),
            "avg_vulnerability_score": float(np.mean(vulnerability_scores)),
            "std_vulnerability_score": float(np.std(vulnerability_scores)),
            "level_distribution": {k.value: v for k, v in level_counts.items()},
            "n_vulnerable": sum(1 for r in reports if r.is_vulnerable),
            "most_vulnerable": {
                "feature_idx": most_vulnerable.feature_idx,
                "score": most_vulnerable.vulnerability_score,
                "level": most_vulnerable.vulnerability_level.value,
            },
            "most_robust": {
                "feature_idx": most_robust.feature_idx,
                "score": most_robust.vulnerability_score,
                "level": most_robust.vulnerability_level.value,
            },
        }

    def _check_safety(self, feature_idx: int) -> bool:
        """Check if probing is allowed by safety guardrails."""
        if self.safety_guardrails is None:
            return True

        try:
            from hololoom.alignment import ActionRequest, RiskLevel

            request = ActionRequest(
                action="adversarial_probe",
                context={
                    "feature_idx": feature_idx,
                    "probe_method": self.config.method.value,
                },
            )
            decision = self.safety_guardrails.evaluate(request)
            return decision.risk_level != RiskLevel.CRITICAL
        except ImportError:
            return True

    def _random_probe(
        self,
        x: np.ndarray,
        feature_idx: int,
        activation_fn: Callable,
    ) -> ProbeResult:
        """Random perturbation baseline probe."""
        perturbation = np.random.uniform(
            -self.config.epsilon,
            self.config.epsilon,
            x.shape
        ).astype(x.dtype)

        perturbed = x + perturbation

        original_acts = activation_fn(x)
        perturbed_acts = activation_fn(perturbed)

        original_activation = self.gradient_prober._get_activation(original_acts, feature_idx)
        perturbed_activation = self.gradient_prober._get_activation(perturbed_acts, feature_idx)

        norm_linf = float(np.max(np.abs(perturbation)))
        norm_l2 = float(np.linalg.norm(perturbation))

        return ProbeResult(
            original_input=x,
            perturbed_input=perturbed,
            original_activation=original_activation,
            perturbed_activation=perturbed_activation,
            perturbation_norm_linf=norm_linf,
            perturbation_norm_l2=norm_l2,
            success=abs(perturbed_activation - original_activation) > self.config.vulnerability_threshold,
            method_used=ProbeMethod.RANDOM,
            iterations=1,
        )

    def _compile_report(
        self,
        feature_idx: int,
        probes: List[ProbeResult],
        start_time: float,
    ) -> VulnerabilityReport:
        """Compile probe results into vulnerability report."""
        duration = time.time() - start_time

        if not probes:
            return VulnerabilityReport(
                feature_idx=feature_idx,
                probe_duration_seconds=duration,
            )

        successful = [p for p in probes if p.success]
        success_rate = len(successful) / len(probes)

        avg_linf = float(np.mean([p.perturbation_norm_linf for p in probes]))
        avg_l2 = float(np.mean([p.perturbation_norm_l2 for p in probes]))
        avg_change = float(np.mean([p.activation_change for p in probes]))

        min_effective = float('inf')
        if successful:
            min_effective = min(p.perturbation_norm_l2 for p in successful)

        most_effective = None
        if successful:
            most_effective = min(successful, key=lambda p: p.perturbation_norm_l2)

        strongest = max(probes, key=lambda p: p.activation_change)

        vulnerability_score = self._compute_vulnerability_score(
            avg_change, min_effective, success_rate
        )
        vulnerability_level = self._classify_vulnerability(vulnerability_score)

        return VulnerabilityReport(
            feature_idx=feature_idx,
            probes=probes,
            vulnerability_score=vulnerability_score,
            vulnerability_level=vulnerability_level,
            avg_perturbation_linf=avg_linf,
            avg_perturbation_l2=avg_l2,
            avg_activation_change=avg_change,
            success_rate=success_rate,
            min_effective_perturbation=min_effective,
            most_effective_probe=most_effective,
            strongest_probe=strongest,
            probe_duration_seconds=duration,
            config_used=self.config,
        )

    def _compute_vulnerability_score(
        self,
        avg_change: float,
        min_perturbation: float,
        success_rate: float,
    ) -> float:
        """
        Compute vulnerability score from 0.0 (robust) to 1.0 (critical).

        Score is based on:
        - How much activation changes (higher = more vulnerable)
        - How small a perturbation is needed (smaller = more vulnerable)
        - What fraction of probes succeeded (higher = more vulnerable)
        """
        change_score = min(1.0, avg_change / self.config.critical_threshold)

        if min_perturbation == float('inf'):
            perturbation_score = 0.0
        else:
            perturbation_score = max(0.0, 1.0 - min_perturbation / self.config.epsilon)

        score = (
            0.4 * change_score +
            0.3 * perturbation_score +
            0.3 * success_rate
        )

        return min(1.0, max(0.0, score))

    def _classify_vulnerability(self, score: float) -> VulnerabilityLevel:
        """Classify vulnerability level from score."""
        if score < 0.2:
            return VulnerabilityLevel.ROBUST
        elif score < 0.4:
            return VulnerabilityLevel.LOW
        elif score < 0.6:
            return VulnerabilityLevel.MEDIUM
        elif score < 0.8:
            return VulnerabilityLevel.HIGH
        else:
            return VulnerabilityLevel.CRITICAL

    def export_reports(self) -> Dict[str, Any]:
        """Export all vulnerability reports."""
        return {
            str(idx): report.to_dict()
            for idx, report in self._reports.items()
        }


def create_prober(
    config: Optional[ProbeConfig] = None,
    safety_guardrails: Optional["SafetyGuardrails"] = None,
) -> AdversarialProber:
    """
    Factory function to create an adversarial prober.

    Args:
        config: Probing configuration (default: balanced)
        safety_guardrails: Optional safety integration

    Returns:
        Configured AdversarialProber
    """
    return AdversarialProber(
        config=config or ProbeConfig.balanced(),
        safety_guardrails=safety_guardrails,
    )
