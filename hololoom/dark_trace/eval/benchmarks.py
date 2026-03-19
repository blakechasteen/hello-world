"""
Dark Trace Evaluation: Benchmark Suite

Provides synthetic benchmarks with ground truth for validating interpretability claims.
Implements multiple benchmark types for comprehensive SAE evaluation.

Benchmark Types:
- SyntheticBenchmark: Ground truth features for recovery testing
- FeatureRecoveryBenchmark: Known concept detection
- InterventionBenchmark: Steering/intervention effectiveness
- ConsistencyBenchmark: Cross-model feature consistency

Research Basis:
- Scaling Monosemanticity (Anthropic 2024): SAE evaluation methodology
- Towards Monosemanticity (Anthropic 2023): Feature quality metrics
- ACDC (Conmy et al., 2023): Circuit evaluation benchmarks
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class BenchmarkType(Enum):
    """Types of benchmarks."""

    SYNTHETIC = "synthetic"
    FEATURE_RECOVERY = "feature_recovery"
    INTERVENTION = "intervention"
    CONSISTENCY = "consistency"


class DifficultyLevel(Enum):
    """Benchmark difficulty levels."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXTREME = "extreme"


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution.

    Attributes:
        n_samples: Number of test samples
        n_features: Number of ground truth features
        sparsity: Target sparsity level (0-1)
        noise_level: Noise added to activations
        difficulty: Overall difficulty level
        seed: Random seed for reproducibility
        timeout_seconds: Maximum benchmark runtime
        verbose: Enable detailed logging
    """

    n_samples: int = 1000
    n_features: int = 100
    sparsity: float = 0.05
    noise_level: float = 0.1
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    seed: int | None = 42
    timeout_seconds: float = 300.0
    verbose: bool = False

    @classmethod
    def quick(cls) -> BenchmarkConfig:
        """Quick benchmark config for fast iteration."""
        return cls(
            n_samples=100,
            n_features=50,
            difficulty=DifficultyLevel.EASY,
            timeout_seconds=30.0,
        )

    @classmethod
    def standard(cls) -> BenchmarkConfig:
        """Standard benchmark config."""
        return cls(
            n_samples=1000,
            n_features=100,
            difficulty=DifficultyLevel.MEDIUM,
            timeout_seconds=300.0,
        )

    @classmethod
    def comprehensive(cls) -> BenchmarkConfig:
        """Comprehensive benchmark for publication-quality results."""
        return cls(
            n_samples=10000,
            n_features=500,
            difficulty=DifficultyLevel.HARD,
            timeout_seconds=3600.0,
            verbose=True,
        )

    @classmethod
    def research(cls) -> BenchmarkConfig:
        """Research-grade benchmark with extreme difficulty."""
        return cls(
            n_samples=50000,
            n_features=1000,
            sparsity=0.01,
            noise_level=0.2,
            difficulty=DifficultyLevel.EXTREME,
            timeout_seconds=7200.0,
            verbose=True,
        )


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run.

    Attributes:
        benchmark_type: Type of benchmark
        name: Benchmark name
        passed: Whether benchmark passed threshold
        score: Overall score (0-1)
        metrics: Detailed metrics dictionary
        duration_seconds: Execution time
        config: Configuration used
        metadata: Additional metadata
    """

    benchmark_type: BenchmarkType
    name: str
    passed: bool
    score: float
    metrics: dict[str, float]
    duration_seconds: float
    config: BenchmarkConfig
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate score range."""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Score must be in [0, 1], got {self.score}")

    def summary(self) -> str:
        """Generate human-readable summary."""
        status = "✓ PASSED" if self.passed else "✗ FAILED"
        lines = [
            f"Benchmark: {self.name}",
            f"Status: {status}",
            f"Score: {self.score:.4f}",
            f"Duration: {self.duration_seconds:.2f}s",
            "",
            "Metrics:",
        ]
        for key, value in sorted(self.metrics.items()):
            lines.append(f"  {key}: {value:.4f}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "benchmark_type": self.benchmark_type.value,
            "name": self.name,
            "passed": self.passed,
            "score": self.score,
            "metrics": self.metrics,
            "duration_seconds": self.duration_seconds,
            "config": {
                "n_samples": self.config.n_samples,
                "n_features": self.config.n_features,
                "difficulty": self.config.difficulty.value,
            },
            "metadata": self.metadata,
        }


class SAEProtocol(Protocol):
    """Protocol for SAE models to be benchmarked."""

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input to latent features."""
        ...

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode latent features to reconstruction."""
        ...

    @property
    def n_features(self) -> int:
        """Number of latent features."""
        ...


class BaseBenchmark:
    """Base class for all benchmarks."""

    def __init__(self, config: BenchmarkConfig | None = None):
        """Initialize benchmark.

        Args:
            config: Benchmark configuration
        """
        self.config = config or BenchmarkConfig.standard()
        self._rng = np.random.default_rng(self.config.seed)

    def run(self, sae: SAEProtocol) -> BenchmarkResult:
        """Run the benchmark.

        Args:
            sae: SAE model to evaluate

        Returns:
            Benchmark result
        """
        raise NotImplementedError("Subclasses must implement run()")

    def _create_result(
        self,
        benchmark_type: BenchmarkType,
        name: str,
        score: float,
        metrics: dict[str, float],
        duration: float,
        threshold: float = 0.7,
        metadata: dict[str, Any] | None = None,
    ) -> BenchmarkResult:
        """Create a benchmark result.

        Args:
            benchmark_type: Type of benchmark
            name: Benchmark name
            score: Overall score
            metrics: Detailed metrics
            duration: Execution time
            threshold: Pass threshold
            metadata: Additional metadata

        Returns:
            BenchmarkResult instance
        """
        return BenchmarkResult(
            benchmark_type=benchmark_type,
            name=name,
            passed=score >= threshold,
            score=score,
            metrics=metrics,
            duration_seconds=duration,
            config=self.config,
            metadata=metadata or {},
        )


class SyntheticBenchmark(BaseBenchmark):
    """Benchmark with synthetic ground truth features.

    Creates known sparse features and tests recovery:
    - Generates ground truth feature activations
    - Creates synthetic data from features
    - Measures feature recovery accuracy
    - Tests reconstruction quality

    Key Metrics:
    - Feature recovery rate (% of ground truth features found)
    - False discovery rate (spurious features)
    - Reconstruction MSE
    - Sparsity match
    """

    def __init__(
        self,
        config: BenchmarkConfig | None = None,
        feature_type: str = "sparse",
    ):
        """Initialize synthetic benchmark.

        Args:
            config: Benchmark configuration
            feature_type: Type of features ('sparse', 'compositional', 'hierarchical')
        """
        super().__init__(config)
        self.feature_type = feature_type

    def _generate_ground_truth(
        self,
        n_features: int,
        input_dim: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate ground truth features and data.

        Args:
            n_features: Number of ground truth features
            input_dim: Input dimension

        Returns:
            Tuple of (feature_directions, activations, data)
        """
        # Generate orthogonal feature directions
        feature_directions = self._rng.standard_normal((n_features, input_dim))
        feature_directions = feature_directions / np.linalg.norm(
            feature_directions, axis=1, keepdims=True
        )

        # Generate sparse activations
        activations = np.zeros((self.config.n_samples, n_features))
        for i in range(self.config.n_samples):
            # Each sample activates ~sparsity * n_features features
            n_active = max(1, int(self.config.sparsity * n_features))
            active_idx = self._rng.choice(n_features, size=n_active, replace=False)
            activations[i, active_idx] = self._rng.exponential(1.0, size=n_active)

        # Generate data from features
        data = activations @ feature_directions

        # Add noise
        noise = self._rng.standard_normal(data.shape) * self.config.noise_level
        data = data + noise

        return feature_directions, activations, data

    def _match_features(
        self,
        ground_truth: np.ndarray,
        learned: np.ndarray,
        threshold: float = 0.8,
    ) -> tuple[list[tuple[int, int]], float]:
        """Match learned features to ground truth.

        Args:
            ground_truth: Ground truth feature directions [n_gt, dim]
            learned: Learned feature directions [n_learned, dim]
            threshold: Cosine similarity threshold for match

        Returns:
            Tuple of (matches, mean_similarity)
        """
        # Compute cosine similarity matrix
        gt_norm = ground_truth / np.linalg.norm(ground_truth, axis=1, keepdims=True)
        learned_norm = learned / np.linalg.norm(learned, axis=1, keepdims=True)
        similarity = np.abs(gt_norm @ learned_norm.T)  # Absolute for direction agnostic

        # Greedy matching
        matches = []
        used_learned = set()

        for gt_idx in range(len(ground_truth)):
            best_sim = -1
            best_learned = -1

            for learned_idx in range(len(learned)):
                if learned_idx in used_learned:
                    continue
                if similarity[gt_idx, learned_idx] > best_sim:
                    best_sim = similarity[gt_idx, learned_idx]
                    best_learned = learned_idx

            if best_sim >= threshold and best_learned >= 0:
                matches.append((gt_idx, best_learned))
                used_learned.add(best_learned)

        # Mean similarity for matches
        mean_sim = np.mean([similarity[gt, lr] for gt, lr in matches]) if matches else 0.0

        return matches, mean_sim

    def run(self, sae: SAEProtocol) -> BenchmarkResult:
        """Run synthetic feature recovery benchmark.

        Args:
            sae: SAE model to evaluate

        Returns:
            Benchmark result with recovery metrics
        """
        start_time = time.time()

        # Generate ground truth
        input_dim = sae.n_features  # Assume SAE encoder input matches decoder output
        gt_directions, gt_activations, data = self._generate_ground_truth(
            self.config.n_features, input_dim
        )

        # Encode data
        learned_activations = sae.encode(data)

        # Get learned feature directions (decoder weights)
        # Approximate by looking at what inputs maximally activate each feature
        learned_directions = np.zeros((sae.n_features, input_dim))
        for i in range(sae.n_features):
            # Features that activate on this data
            active_mask = learned_activations[:, i] > 0
            if np.any(active_mask):
                # Average input that activates this feature
                learned_directions[i] = np.mean(data[active_mask], axis=0)

        # Match features
        matches, mean_sim = self._match_features(gt_directions, learned_directions)

        # Compute metrics
        recovery_rate = len(matches) / len(gt_directions)
        false_discovery_rate = 1.0 - len(matches) / max(1, sae.n_features)

        # Reconstruction quality
        reconstructed = sae.decode(learned_activations)
        mse = np.mean((data - reconstructed) ** 2)

        # Sparsity comparison
        gt_sparsity = np.mean(gt_activations > 0)
        learned_sparsity = np.mean(learned_activations > 0)
        sparsity_error = abs(gt_sparsity - learned_sparsity)

        duration = time.time() - start_time

        # Overall score: weighted combination
        score = (
            0.4 * recovery_rate +
            0.2 * (1.0 - false_discovery_rate) +
            0.2 * mean_sim +
            0.1 * max(0, 1.0 - mse) +
            0.1 * (1.0 - sparsity_error)
        )

        return self._create_result(
            benchmark_type=BenchmarkType.SYNTHETIC,
            name=f"Synthetic Feature Recovery ({self.feature_type})",
            score=score,
            metrics={
                "recovery_rate": recovery_rate,
                "false_discovery_rate": false_discovery_rate,
                "mean_match_similarity": mean_sim,
                "reconstruction_mse": mse,
                "ground_truth_sparsity": gt_sparsity,
                "learned_sparsity": learned_sparsity,
                "sparsity_error": sparsity_error,
                "n_matches": float(len(matches)),
            },
            duration=duration,
            metadata={
                "feature_type": self.feature_type,
                "n_ground_truth": len(gt_directions),
                "n_learned": sae.n_features,
            },
        )


class FeatureRecoveryBenchmark(BaseBenchmark):
    """Benchmark for known concept detection.

    Tests whether SAE recovers known semantic features:
    - Predefined concept vectors (e.g., "positive sentiment")
    - Tests alignment with known concepts
    - Measures monosemanticity of concept features

    Key Metrics:
    - Concept alignment score
    - Feature purity (monosemanticity per concept)
    - Coverage (% of concepts with dedicated feature)
    """

    def __init__(
        self,
        config: BenchmarkConfig | None = None,
        concept_vectors: dict[str, np.ndarray] | None = None,
    ):
        """Initialize feature recovery benchmark.

        Args:
            config: Benchmark configuration
            concept_vectors: Dict mapping concept names to direction vectors
        """
        super().__init__(config)
        self.concept_vectors = concept_vectors or {}

    def add_concept(self, name: str, vector: np.ndarray) -> None:
        """Add a concept vector.

        Args:
            name: Concept name
            vector: Direction vector
        """
        self.concept_vectors[name] = vector / np.linalg.norm(vector)

    def _generate_synthetic_concepts(self, dim: int, n_concepts: int = 10) -> None:
        """Generate synthetic concept vectors for testing.

        Args:
            dim: Vector dimension
            n_concepts: Number of concepts to generate
        """
        concept_names = [
            "positive_sentiment", "negative_sentiment", "technical", "casual",
            "urgent", "formal", "emotional", "factual", "question", "command",
            "past_tense", "future_tense", "first_person", "third_person",
            "abstract", "concrete", "short", "long", "simple", "complex",
        ]

        for i, name in enumerate(concept_names[:n_concepts]):
            # Generate orthogonal concept vectors
            vec = np.zeros(dim)
            vec[i % dim] = 1.0
            # Add some noise for realism
            noise = self._rng.standard_normal(dim) * 0.1
            vec = vec + noise
            self.concept_vectors[name] = vec / np.linalg.norm(vec)

    def run(self, sae: SAEProtocol) -> BenchmarkResult:
        """Run feature recovery benchmark.

        Args:
            sae: SAE model to evaluate

        Returns:
            Benchmark result with concept alignment metrics
        """
        start_time = time.time()

        # Generate synthetic concepts if none provided
        if not self.concept_vectors:
            self._generate_synthetic_concepts(sae.n_features)

        # Generate concept-aligned test data
        n_per_concept = self.config.n_samples // len(self.concept_vectors)
        all_data = []
        all_labels = []

        for concept_name, concept_vec in self.concept_vectors.items():
            # Generate data aligned with concept
            concept_data = (
                np.outer(self._rng.exponential(1.0, n_per_concept), concept_vec) +
                self._rng.standard_normal((n_per_concept, len(concept_vec))) * self.config.noise_level
            )
            all_data.append(concept_data)
            all_labels.extend([concept_name] * n_per_concept)

        data = np.vstack(all_data)

        # Encode with SAE
        activations = sae.encode(data)

        # Find best matching feature for each concept
        concept_features = {}
        alignment_scores = {}

        for concept_name, concept_vec in self.concept_vectors.items():
            # Get activations for this concept's data
            concept_mask = np.array([l == concept_name for l in all_labels])
            concept_acts = activations[concept_mask]
            other_acts = activations[~concept_mask]

            # Find feature most selective for this concept
            best_feature = -1
            best_selectivity = -1

            for f_idx in range(sae.n_features):
                # Selectivity = mean(concept activations) / mean(other activations)
                concept_mean = np.mean(concept_acts[:, f_idx])
                other_mean = np.mean(other_acts[:, f_idx]) + 1e-10
                selectivity = concept_mean / other_mean

                if selectivity > best_selectivity:
                    best_selectivity = selectivity
                    best_feature = f_idx

            concept_features[concept_name] = best_feature
            alignment_scores[concept_name] = min(1.0, best_selectivity / 10.0)  # Normalize

        # Compute metrics
        mean_alignment = np.mean(list(alignment_scores.values()))

        # Check if concepts have unique features
        unique_features = len(set(concept_features.values()))
        coverage = unique_features / len(self.concept_vectors)

        # Feature purity: how exclusive is each feature to its concept
        purity_scores = []
        for concept_name, f_idx in concept_features.items():
            concept_mask = np.array([l == concept_name for l in all_labels])
            concept_acts = activations[concept_mask, f_idx]
            other_acts = activations[~concept_mask, f_idx]

            # Purity = P(concept | feature active)
            total_active = np.sum(activations[:, f_idx] > 0)
            concept_active = np.sum((activations[:, f_idx] > 0) & concept_mask)
            purity = concept_active / max(1, total_active)
            purity_scores.append(purity)

        mean_purity = np.mean(purity_scores)

        duration = time.time() - start_time

        # Overall score
        score = 0.4 * mean_alignment + 0.3 * coverage + 0.3 * mean_purity

        return self._create_result(
            benchmark_type=BenchmarkType.FEATURE_RECOVERY,
            name="Feature Recovery (Known Concepts)",
            score=score,
            metrics={
                "mean_alignment": mean_alignment,
                "coverage": coverage,
                "mean_purity": mean_purity,
                "n_concepts": float(len(self.concept_vectors)),
                "n_unique_features": float(unique_features),
            },
            duration=duration,
            metadata={
                "concepts": list(self.concept_vectors.keys()),
                "concept_feature_mapping": concept_features,
            },
        )


class InterventionBenchmark(BaseBenchmark):
    """Benchmark for steering/intervention effectiveness.

    Tests causal effectiveness of feature manipulation:
    - Activate feature → measure output change
    - Ablate feature → measure impact
    - Dose-response curves

    Key Metrics:
    - Steering accuracy (does intervention achieve goal?)
    - Side effect magnitude (unintended changes)
    - Controllability (linear response to intervention strength)
    """

    def __init__(
        self,
        config: BenchmarkConfig | None = None,
        target_features: list[int] | None = None,
    ):
        """Initialize intervention benchmark.

        Args:
            config: Benchmark configuration
            target_features: Features to test interventions on
        """
        super().__init__(config)
        self.target_features = target_features

    def run(self, sae: SAEProtocol) -> BenchmarkResult:
        """Run intervention effectiveness benchmark.

        Args:
            sae: SAE model to evaluate

        Returns:
            Benchmark result with intervention metrics
        """
        start_time = time.time()

        # Generate baseline data
        input_dim = sae.n_features
        baseline_data = self._rng.standard_normal((self.config.n_samples, input_dim))

        # Get baseline activations
        baseline_acts = sae.encode(baseline_data)
        baseline_recon = sae.decode(baseline_acts)

        # Select features to intervene on
        if self.target_features is None:
            # Select most active features
            feature_activity = np.mean(baseline_acts > 0, axis=0)
            active_features = np.argsort(feature_activity)[-10:]
            self.target_features = active_features.tolist()

        # Test interventions
        steering_accuracies = []
        side_effects = []
        controllabilities = []

        for feature_idx in self.target_features:
            # Intervention: set feature to specific value
            intervention_strengths = [0.5, 1.0, 2.0, 5.0]
            responses = []

            for strength in intervention_strengths:
                intervened_acts = baseline_acts.copy()
                intervened_acts[:, feature_idx] = strength

                # Decode with intervention
                intervened_recon = sae.decode(intervened_acts)

                # Measure response
                response = np.mean(np.abs(intervened_recon - baseline_recon))
                responses.append(response)

            # Steering accuracy: did intervention change output?
            steering_accuracy = min(1.0, responses[-1] / (responses[0] + 1e-10))
            steering_accuracies.append(steering_accuracy)

            # Side effects: changes in other features
            intervened_acts_final = baseline_acts.copy()
            intervened_acts_final[:, feature_idx] = 5.0
            re_encoded = sae.encode(sae.decode(intervened_acts_final))

            other_features = [i for i in range(sae.n_features) if i != feature_idx]
            side_effect = np.mean(np.abs(
                re_encoded[:, other_features] - baseline_acts[:, other_features]
            ))
            side_effects.append(side_effect)

            # Controllability: linear response to strength
            responses_arr = np.array(responses)
            strengths_arr = np.array(intervention_strengths)
            correlation = np.corrcoef(responses_arr, strengths_arr)[0, 1]
            controllabilities.append(max(0, correlation))

        # Aggregate metrics
        mean_steering = np.mean(steering_accuracies)
        mean_side_effect = np.mean(side_effects)
        mean_controllability = np.mean(controllabilities)

        duration = time.time() - start_time

        # Overall score (penalize side effects)
        score = (
            0.4 * mean_steering +
            0.3 * (1.0 - min(1.0, mean_side_effect)) +
            0.3 * mean_controllability
        )

        return self._create_result(
            benchmark_type=BenchmarkType.INTERVENTION,
            name="Intervention Effectiveness",
            score=score,
            metrics={
                "mean_steering_accuracy": mean_steering,
                "mean_side_effect": mean_side_effect,
                "mean_controllability": mean_controllability,
                "n_features_tested": float(len(self.target_features)),
            },
            duration=duration,
            metadata={
                "target_features": self.target_features,
            },
        )


class ConsistencyBenchmark(BaseBenchmark):
    """Benchmark for cross-model feature consistency.

    Tests whether features are consistent across:
    - Different random seeds
    - Different training data samples
    - Different model sizes

    Key Metrics:
    - Feature stability (same features across seeds)
    - Activation correlation (similar patterns)
    - Representation similarity (CKA, centered kernel alignment)
    """

    def __init__(
        self,
        config: BenchmarkConfig | None = None,
        reference_sae: SAEProtocol | None = None,
    ):
        """Initialize consistency benchmark.

        Args:
            config: Benchmark configuration
            reference_sae: Reference SAE for comparison
        """
        super().__init__(config)
        self.reference_sae = reference_sae

    def _compute_cka(
        self,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> float:
        """Compute Centered Kernel Alignment (CKA).

        Args:
            X: First representation [n_samples, dim1]
            Y: Second representation [n_samples, dim2]

        Returns:
            CKA similarity score
        """
        # Linear CKA
        X = X - X.mean(axis=0)
        Y = Y - Y.mean(axis=0)

        # Compute Gram matrices
        XX = X @ X.T
        YY = Y @ Y.T

        # CKA = HSIC(X, Y) / sqrt(HSIC(X, X) * HSIC(Y, Y))
        hsic_xy = np.sum(XX * YY)
        hsic_xx = np.sum(XX * XX)
        hsic_yy = np.sum(YY * YY)

        cka = hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10)
        return float(cka)

    def run(self, sae: SAEProtocol) -> BenchmarkResult:
        """Run consistency benchmark.

        Args:
            sae: SAE model to evaluate

        Returns:
            Benchmark result with consistency metrics
        """
        start_time = time.time()

        # Generate test data
        input_dim = sae.n_features
        test_data = self._rng.standard_normal((self.config.n_samples, input_dim))

        # Get activations from target SAE
        target_acts = sae.encode(test_data)

        if self.reference_sae is not None:
            # Compare to reference
            ref_acts = self.reference_sae.encode(test_data)

            # CKA similarity
            cka_score = self._compute_cka(target_acts, ref_acts)

            # Activation correlation (per-feature correlation average)
            n_compare = min(target_acts.shape[1], ref_acts.shape[1])
            correlations = []
            for i in range(n_compare):
                corr = np.corrcoef(target_acts[:, i], ref_acts[:, i])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
            mean_correlation = np.mean(correlations) if correlations else 0.0

            # Feature stability: how many features have a strong match
            strong_matches = sum(1 for c in correlations if c > 0.5)
            stability = strong_matches / n_compare if n_compare > 0 else 0.0

        else:
            # Self-consistency: compare two halves of data
            half = self.config.n_samples // 2
            acts1 = target_acts[:half]
            acts2 = target_acts[half:half*2]

            # Pad to same size
            min_samples = min(len(acts1), len(acts2))
            acts1 = acts1[:min_samples]
            acts2 = acts2[:min_samples]

            cka_score = self._compute_cka(acts1, acts2)

            # Per-feature correlation
            correlations = []
            for i in range(target_acts.shape[1]):
                corr = np.corrcoef(acts1[:, i], acts2[:, i])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
            mean_correlation = np.mean(correlations) if correlations else 0.0
            stability = 1.0  # Self-consistency baseline

        duration = time.time() - start_time

        # Overall score
        score = 0.4 * cka_score + 0.3 * mean_correlation + 0.3 * stability

        return self._create_result(
            benchmark_type=BenchmarkType.CONSISTENCY,
            name="Cross-Model Consistency",
            score=score,
            metrics={
                "cka_similarity": cka_score,
                "mean_feature_correlation": mean_correlation,
                "feature_stability": stability,
            },
            duration=duration,
            metadata={
                "has_reference": self.reference_sae is not None,
            },
        )


@dataclass
class BenchmarkSuiteResult:
    """Result from running a benchmark suite.

    Attributes:
        results: Individual benchmark results
        overall_score: Weighted average score
        passed: Whether all benchmarks passed
        duration_seconds: Total execution time
    """

    results: list[BenchmarkResult]
    overall_score: float
    passed: bool
    duration_seconds: float

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "BENCHMARK SUITE RESULTS",
            "=" * 60,
            "",
            f"Overall Score: {self.overall_score:.4f}",
            f"Status: {'PASSED' if self.passed else 'FAILED'}",
            f"Total Duration: {self.duration_seconds:.2f}s",
            "",
            "-" * 60,
        ]

        for result in self.results:
            status = "✓" if result.passed else "✗"
            lines.append(f"{status} {result.name}: {result.score:.4f}")

        lines.append("-" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "results": [r.to_dict() for r in self.results],
            "overall_score": self.overall_score,
            "passed": self.passed,
            "duration_seconds": self.duration_seconds,
        }

    def save(self, path: str | Path) -> None:
        """Save results to JSON file.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class BenchmarkSuite:
    """Suite of benchmarks for comprehensive SAE evaluation.

    Runs multiple benchmarks and aggregates results.

    Example:
        suite = BenchmarkSuite()
        suite.add(SyntheticBenchmark())
        suite.add(FeatureRecoveryBenchmark())
        suite.add(InterventionBenchmark())
        suite.add(ConsistencyBenchmark())

        results = suite.run(sae)
        print(results.summary())
    """

    def __init__(
        self,
        config: BenchmarkConfig | None = None,
        weights: dict[BenchmarkType, float] | None = None,
    ):
        """Initialize benchmark suite.

        Args:
            config: Default configuration for benchmarks
            weights: Weights for each benchmark type in overall score
        """
        self.config = config or BenchmarkConfig.standard()
        self.weights = weights or {
            BenchmarkType.SYNTHETIC: 0.3,
            BenchmarkType.FEATURE_RECOVERY: 0.25,
            BenchmarkType.INTERVENTION: 0.25,
            BenchmarkType.CONSISTENCY: 0.2,
        }
        self.benchmarks: list[BaseBenchmark] = []

    def add(self, benchmark: BaseBenchmark) -> BenchmarkSuite:
        """Add a benchmark to the suite.

        Args:
            benchmark: Benchmark to add

        Returns:
            Self for chaining
        """
        self.benchmarks.append(benchmark)
        return self

    def add_all_defaults(self) -> BenchmarkSuite:
        """Add all default benchmarks.

        Returns:
            Self for chaining
        """
        self.add(SyntheticBenchmark(self.config))
        self.add(FeatureRecoveryBenchmark(self.config))
        self.add(InterventionBenchmark(self.config))
        self.add(ConsistencyBenchmark(self.config))
        return self

    def run(self, sae: SAEProtocol) -> BenchmarkSuiteResult:
        """Run all benchmarks in the suite.

        Args:
            sae: SAE model to evaluate

        Returns:
            Suite result with all benchmark results
        """
        start_time = time.time()
        results = []

        for benchmark in self.benchmarks:
            try:
                result = benchmark.run(sae)
                results.append(result)
            except Exception as e:
                # Create failed result
                results.append(BenchmarkResult(
                    benchmark_type=BenchmarkType.SYNTHETIC,
                    name=f"Failed: {type(benchmark).__name__}",
                    passed=False,
                    score=0.0,
                    metrics={"error": str(e)},
                    duration_seconds=0.0,
                    config=benchmark.config,
                    metadata={"exception": str(e)},
                ))

        # Compute overall score
        weighted_sum = 0.0
        weight_sum = 0.0

        for result in results:
            weight = self.weights.get(result.benchmark_type, 0.25)
            weighted_sum += result.score * weight
            weight_sum += weight

        overall_score = weighted_sum / weight_sum if weight_sum > 0 else 0.0

        # Check if all passed
        all_passed = all(r.passed for r in results)

        duration = time.time() - start_time

        return BenchmarkSuiteResult(
            results=results,
            overall_score=overall_score,
            passed=all_passed,
            duration_seconds=duration,
        )


# Convenience functions

def create_benchmark_suite(
    config: BenchmarkConfig | None = None,
    include_all: bool = True,
) -> BenchmarkSuite:
    """Create a benchmark suite with default benchmarks.

    Args:
        config: Benchmark configuration
        include_all: Whether to include all default benchmarks

    Returns:
        Configured benchmark suite
    """
    suite = BenchmarkSuite(config)
    if include_all:
        suite.add_all_defaults()
    return suite


def run_quick_benchmark(sae: SAEProtocol) -> BenchmarkSuiteResult:
    """Run quick benchmark suite for fast validation.

    Args:
        sae: SAE model to evaluate

    Returns:
        Benchmark results
    """
    config = BenchmarkConfig.quick()
    suite = create_benchmark_suite(config, include_all=True)
    return suite.run(sae)


def run_comprehensive_benchmark(sae: SAEProtocol) -> BenchmarkSuiteResult:
    """Run comprehensive benchmark suite for publication-quality results.

    Args:
        sae: SAE model to evaluate

    Returns:
        Benchmark results
    """
    config = BenchmarkConfig.comprehensive()
    suite = create_benchmark_suite(config, include_all=True)
    return suite.run(sae)
