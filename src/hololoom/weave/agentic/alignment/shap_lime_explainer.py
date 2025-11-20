"""
SHAP/LIME Integration for HoloLoom Alignment

Provides model-agnostic feature attribution to explain policy decisions.
Implements both SHAP (Shapley Additive Explanations) and LIME (Local
Interpretable Model-agnostic Explanations) for different use cases.

Key Features:
- SHAP: Global feature importance with game-theoretic guarantees
- LIME: Local linear approximations for individual predictions
- Multi-scale attribution (motif, embedding, spectral features)
- Integration with HoloLoom's 244D semantic space
- Visualization generation for reports

References:
- SHAP: Lundberg & Lee (2017) "A Unified Approach to Interpreting Model Predictions"
- LIME: Ribeiro et al. (2016) "Why Should I Trust You?"
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import numpy as np
from pathlib import Path
import json
import asyncio


class ExplanationType(Enum):
    """Type of explanation to generate."""
    SHAP = "shap"
    LIME = "lime"
    BOTH = "both"


class FeatureType(Enum):
    """Type of feature being explained."""
    MOTIF = "motif"
    EMBEDDING = "embedding"
    SPECTRAL = "spectral"
    SEMANTIC = "semantic"
    ALL = "all"


@dataclass
class FeatureAttribution:
    """Attribution score for a single feature."""
    feature_name: str
    feature_type: FeatureType
    attribution_score: float  # Positive = increases confidence, Negative = decreases
    feature_value: float
    confidence_contribution: float  # How much this feature contributed to final confidence

    def __repr__(self) -> str:
        sign = "+" if self.attribution_score > 0 else ""
        return (f"{self.feature_name} ({self.feature_type.value}): "
                f"{sign}{self.attribution_score:.4f} "
                f"(contrib: {self.confidence_contribution:.2%})")


@dataclass
class SHAPExplanation:
    """SHAP-based explanation for a decision."""
    query_id: str
    query_text: str
    predicted_tool: str
    predicted_confidence: float

    # Feature attributions (Shapley values)
    attributions: List[FeatureAttribution] = field(default_factory=list)

    # Base value (expected value with no features)
    base_value: float = 0.5

    # Interaction effects (optional)
    interaction_effects: Optional[Dict[Tuple[str, str], float]] = None

    # Metadata
    timestamp: float = 0.0
    computation_time_ms: float = 0.0

    def top_positive_features(self, n: int = 5) -> List[FeatureAttribution]:
        """Get top N features that increased confidence."""
        return sorted(
            [a for a in self.attributions if a.attribution_score > 0],
            key=lambda x: x.attribution_score,
            reverse=True
        )[:n]

    def top_negative_features(self, n: int = 5) -> List[FeatureAttribution]:
        """Get top N features that decreased confidence."""
        return sorted(
            [a for a in self.attributions if a.attribution_score < 0],
            key=lambda x: x.attribution_score
        )[:n]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "predicted_tool": self.predicted_tool,
            "predicted_confidence": self.predicted_confidence,
            "base_value": self.base_value,
            "attributions": [
                {
                    "feature_name": a.feature_name,
                    "feature_type": a.feature_type.value,
                    "attribution_score": a.attribution_score,
                    "feature_value": a.feature_value,
                    "confidence_contribution": a.confidence_contribution
                }
                for a in self.attributions
            ],
            "timestamp": self.timestamp,
            "computation_time_ms": self.computation_time_ms
        }


@dataclass
class LIMEExplanation:
    """LIME-based local linear approximation."""
    query_id: str
    query_text: str
    predicted_tool: str
    predicted_confidence: float

    # Local linear model coefficients
    linear_coefficients: Dict[str, float] = field(default_factory=dict)

    # R² score of local approximation
    local_fidelity: float = 0.0

    # Number of samples used for approximation
    num_samples: int = 1000

    # Perturbation radius
    perturbation_radius: float = 0.1

    # Metadata
    timestamp: float = 0.0
    computation_time_ms: float = 0.0

    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top N features by absolute coefficient value."""
        return sorted(
            self.linear_coefficients.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:n]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "predicted_tool": self.predicted_tool,
            "predicted_confidence": self.predicted_confidence,
            "linear_coefficients": self.linear_coefficients,
            "local_fidelity": self.local_fidelity,
            "num_samples": self.num_samples,
            "perturbation_radius": self.perturbation_radius,
            "timestamp": self.timestamp,
            "computation_time_ms": self.computation_time_ms
        }


class SHAPExplainer:
    """
    SHAP explainer using Kernel SHAP algorithm.

    Computes Shapley values to attribute prediction to features.
    Game-theoretic guarantees: fairness, consistency, efficiency.
    """

    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], float],
        feature_names: List[str],
        feature_types: Optional[List[FeatureType]] = None,
        num_samples: int = 1000,
        background_data: Optional[np.ndarray] = None
    ):
        """
        Initialize SHAP explainer.

        Args:
            predict_fn: Function that takes feature vector and returns confidence
            feature_names: Names of all features
            feature_types: Type of each feature (motif, embedding, etc.)
            num_samples: Number of samples for Kernel SHAP approximation
            background_data: Background dataset for baseline (uses zeros if None)
        """
        self.predict_fn = predict_fn
        self.feature_names = feature_names
        self.feature_types = feature_types or [FeatureType.ALL] * len(feature_names)
        self.num_samples = num_samples

        # Background data (expected value baseline)
        if background_data is not None:
            self.background = background_data
        else:
            self.background = np.zeros((1, len(feature_names)))

    async def explain(
        self,
        query_id: str,
        query_text: str,
        features: np.ndarray,
        predicted_tool: str,
        predicted_confidence: float
    ) -> SHAPExplanation:
        """
        Generate SHAP explanation for a prediction.

        Uses Kernel SHAP algorithm (Lundberg & Lee 2017):
        1. Generate coalition samples (feature subsets)
        2. Evaluate model on coalitions
        3. Solve weighted linear regression for Shapley values

        Args:
            query_id: Unique query identifier
            query_text: Original query text
            features: Feature vector used for prediction
            predicted_tool: Tool that was predicted
            predicted_confidence: Confidence of prediction

        Returns:
            SHAPExplanation with feature attributions
        """
        import time
        start_time = time.time()

        # Base value (expected value with no features)
        base_value = self.predict_fn(self.background.mean(axis=0))

        # Generate coalition samples (feature subsets)
        # Each sample has some features from instance, some from background
        coalitions = self._generate_coalitions(features)

        # Evaluate model on coalitions
        predictions = np.array([self.predict_fn(coalition) for coalition in coalitions])

        # Compute Shapley values via weighted linear regression
        shapley_values = self._compute_shapley_values(coalitions, predictions, features)

        # Create feature attributions
        attributions = []
        for i, (name, shap_val, feat_type) in enumerate(
            zip(self.feature_names, shapley_values, self.feature_types)
        ):
            # Confidence contribution is Shapley value relative to total
            total_effect = abs(predicted_confidence - base_value)
            contribution = abs(shap_val) / total_effect if total_effect > 0 else 0.0

            attributions.append(FeatureAttribution(
                feature_name=name,
                feature_type=feat_type,
                attribution_score=shap_val,
                feature_value=features[i],
                confidence_contribution=contribution
            ))

        computation_time = (time.time() - start_time) * 1000

        return SHAPExplanation(
            query_id=query_id,
            query_text=query_text,
            predicted_tool=predicted_tool,
            predicted_confidence=predicted_confidence,
            attributions=attributions,
            base_value=base_value,
            timestamp=time.time(),
            computation_time_ms=computation_time
        )

    def _generate_coalitions(self, features: np.ndarray) -> np.ndarray:
        """
        Generate coalition samples for Kernel SHAP.

        Each coalition is a feature vector with some features from the instance
        and some from the background (masked).

        Returns array of shape (num_samples, num_features)
        """
        num_features = len(features)
        coalitions = []

        # Always include empty coalition (all background) and full coalition
        background_mean = self.background.mean(axis=0)
        coalitions.append(background_mean)
        coalitions.append(features)

        # Generate random coalitions
        for _ in range(self.num_samples - 2):
            # Random mask: 1 = use instance feature, 0 = use background
            mask = np.random.binomial(1, 0.5, num_features)
            coalition = mask * features + (1 - mask) * background_mean
            coalitions.append(coalition)

        return np.array(coalitions)

    def _compute_shapley_values(
        self,
        coalitions: np.ndarray,
        predictions: np.ndarray,
        features: np.ndarray
    ) -> np.ndarray:
        """
        Compute Shapley values via weighted linear regression.

        Kernel SHAP uses special weights based on coalition size to ensure
        Shapley value guarantees.

        Returns array of Shapley values (one per feature)
        """
        num_features = len(features)
        background_mean = self.background.mean(axis=0)

        # Create coalition matrix (binary: which features are active)
        coalition_matrix = np.zeros((len(coalitions), num_features))
        for i, coalition in enumerate(coalitions):
            coalition_matrix[i] = (np.abs(coalition - background_mean) > 1e-6).astype(float)

        # Kernel SHAP weights
        coalition_sizes = coalition_matrix.sum(axis=1)
        weights = self._shapley_kernel_weights(coalition_sizes, num_features)

        # Weighted linear regression: predictions = base + Σ(shap_i * z_i)
        # Where z_i is 1 if feature i is active, 0 otherwise
        # Solve: SHAP = (Z^T W Z)^-1 Z^T W (predictions - base)

        # Add small regularization for numerical stability
        ZtWZ = coalition_matrix.T @ (weights[:, None] * coalition_matrix) + 1e-6 * np.eye(num_features)
        ZtWy = coalition_matrix.T @ (weights * predictions)

        shapley_values = np.linalg.solve(ZtWZ, ZtWy)

        return shapley_values

    def _shapley_kernel_weights(self, coalition_sizes: np.ndarray, num_features: int) -> np.ndarray:
        """
        Compute Kernel SHAP weights based on coalition sizes.

        Weight function ensures Shapley value guarantees:
        w(z) = (M-1) / (binom(M, |z|) * |z| * (M - |z|))

        where M is number of features, |z| is coalition size.
        """
        weights = np.zeros_like(coalition_sizes, dtype=float)

        for i, size in enumerate(coalition_sizes):
            if size == 0 or size == num_features:
                # Empty and full coalitions get infinite weight (include always)
                weights[i] = 1e6
            else:
                # Shapley kernel weight
                from math import comb
                weights[i] = (num_features - 1) / (
                    comb(num_features, int(size)) * size * (num_features - size)
                )

        return weights


class LIMEExplainer:
    """
    LIME explainer for local linear approximations.

    Fits a simple linear model around a prediction to explain
    which features matter locally.
    """

    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], float],
        feature_names: List[str],
        num_samples: int = 1000,
        perturbation_radius: float = 0.1
    ):
        """
        Initialize LIME explainer.

        Args:
            predict_fn: Function that takes feature vector and returns confidence
            feature_names: Names of all features
            num_samples: Number of perturbed samples to generate
            perturbation_radius: Radius of perturbations (std dev)
        """
        self.predict_fn = predict_fn
        self.feature_names = feature_names
        self.num_samples = num_samples
        self.perturbation_radius = perturbation_radius

    async def explain(
        self,
        query_id: str,
        query_text: str,
        features: np.ndarray,
        predicted_tool: str,
        predicted_confidence: float
    ) -> LIMEExplanation:
        """
        Generate LIME explanation by fitting local linear model.

        Algorithm:
        1. Generate perturbed samples around instance
        2. Get model predictions for perturbed samples
        3. Weight samples by proximity to instance
        4. Fit weighted linear regression
        5. Return linear coefficients as explanations

        Args:
            query_id: Unique query identifier
            query_text: Original query text
            features: Feature vector used for prediction
            predicted_tool: Tool that was predicted
            predicted_confidence: Confidence of prediction

        Returns:
            LIMEExplanation with local linear model
        """
        import time
        start_time = time.time()

        # Generate perturbed samples
        perturbed_samples = self._generate_perturbed_samples(features)

        # Get predictions for perturbed samples
        predictions = np.array([self.predict_fn(sample) for sample in perturbed_samples])

        # Compute proximity weights (closer samples weighted higher)
        distances = np.linalg.norm(perturbed_samples - features, axis=1)
        weights = self._proximity_kernel(distances)

        # Fit weighted linear regression
        coefficients, fidelity = self._fit_linear_model(
            perturbed_samples, predictions, weights
        )

        computation_time = (time.time() - start_time) * 1000

        return LIMEExplanation(
            query_id=query_id,
            query_text=query_text,
            predicted_tool=predicted_tool,
            predicted_confidence=predicted_confidence,
            linear_coefficients=dict(zip(self.feature_names, coefficients)),
            local_fidelity=fidelity,
            num_samples=self.num_samples,
            perturbation_radius=self.perturbation_radius,
            timestamp=time.time(),
            computation_time_ms=computation_time
        )

    def _generate_perturbed_samples(self, features: np.ndarray) -> np.ndarray:
        """
        Generate perturbed samples around instance.

        Uses Gaussian perturbations with configurable radius.
        """
        num_features = len(features)
        noise = np.random.randn(self.num_samples, num_features) * self.perturbation_radius
        return features + noise

    def _proximity_kernel(self, distances: np.ndarray, kernel_width: float = 0.25) -> np.ndarray:
        """
        Compute proximity weights using exponential kernel.

        Closer samples get higher weight in linear regression.
        """
        return np.exp(-(distances ** 2) / (kernel_width ** 2))

    def _fit_linear_model(
        self,
        samples: np.ndarray,
        predictions: np.ndarray,
        weights: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Fit weighted linear regression.

        Returns:
            coefficients: Linear coefficients (one per feature)
            fidelity: R² score of local approximation
        """
        # Weighted least squares: β = (X^T W X)^-1 X^T W y
        XtWX = samples.T @ (weights[:, None] * samples) + 1e-6 * np.eye(samples.shape[1])
        XtWy = samples.T @ (weights * predictions)

        coefficients = np.linalg.solve(XtWX, XtWy)

        # Compute R² (local fidelity)
        predicted = samples @ coefficients
        residuals = predictions - predicted
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((predictions - predictions.mean()) ** 2)
        fidelity = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return coefficients, fidelity


class UnifiedExplainer:
    """
    Unified explainer that combines SHAP and LIME.

    Provides both global feature importance (SHAP) and local
    linear interpretability (LIME) in a single interface.
    """

    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], float],
        feature_names: List[str],
        feature_types: Optional[List[FeatureType]] = None,
        shap_samples: int = 1000,
        lime_samples: int = 1000,
        perturbation_radius: float = 0.1
    ):
        """
        Initialize unified explainer.

        Args:
            predict_fn: Function that takes feature vector and returns confidence
            feature_names: Names of all features
            feature_types: Type of each feature
            shap_samples: Number of samples for SHAP
            lime_samples: Number of samples for LIME
            perturbation_radius: LIME perturbation radius
        """
        self.shap_explainer = SHAPExplainer(
            predict_fn, feature_names, feature_types, shap_samples
        )
        self.lime_explainer = LIMEExplainer(
            predict_fn, feature_names, lime_samples, perturbation_radius
        )

    async def explain(
        self,
        query_id: str,
        query_text: str,
        features: np.ndarray,
        predicted_tool: str,
        predicted_confidence: float,
        explanation_type: ExplanationType = ExplanationType.BOTH
    ) -> Tuple[Optional[SHAPExplanation], Optional[LIMEExplanation]]:
        """
        Generate explanations using SHAP and/or LIME.

        Args:
            query_id: Unique query identifier
            query_text: Original query text
            features: Feature vector used for prediction
            predicted_tool: Tool that was predicted
            predicted_confidence: Confidence of prediction
            explanation_type: Which explanation(s) to generate

        Returns:
            Tuple of (SHAP explanation, LIME explanation)
            Either may be None depending on explanation_type
        """
        shap_result = None
        lime_result = None

        if explanation_type in (ExplanationType.SHAP, ExplanationType.BOTH):
            shap_result = await self.shap_explainer.explain(
                query_id, query_text, features, predicted_tool, predicted_confidence
            )

        if explanation_type in (ExplanationType.LIME, ExplanationType.BOTH):
            lime_result = await self.lime_explainer.explain(
                query_id, query_text, features, predicted_tool, predicted_confidence
            )

        return shap_result, lime_result

    async def batch_explain(
        self,
        queries: List[Tuple[str, str, np.ndarray, str, float]],
        explanation_type: ExplanationType = ExplanationType.BOTH,
        max_concurrent: int = 5
    ) -> List[Tuple[Optional[SHAPExplanation], Optional[LIMEExplanation]]]:
        """
        Generate explanations for multiple queries concurrently.

        Args:
            queries: List of (query_id, query_text, features, tool, confidence)
            explanation_type: Which explanation(s) to generate
            max_concurrent: Maximum concurrent explanations

        Returns:
            List of (SHAP, LIME) explanation tuples
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def explain_with_semaphore(query_data):
            async with semaphore:
                query_id, query_text, features, tool, confidence = query_data
                return await self.explain(
                    query_id, query_text, features, tool, confidence, explanation_type
                )

        return await asyncio.gather(*[explain_with_semaphore(q) for q in queries])


def save_explanations(
    explanations: List[Tuple[Optional[SHAPExplanation], Optional[LIMEExplanation]]],
    output_dir: Path
):
    """
    Save explanations to disk.

    Args:
        explanations: List of (SHAP, LIME) explanation tuples
        output_dir: Directory to save explanations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shap_file = output_dir / "shap_explanations.jsonl"
    lime_file = output_dir / "lime_explanations.jsonl"

    with open(shap_file, 'w') as f_shap, open(lime_file, 'w') as f_lime:
        for shap_exp, lime_exp in explanations:
            if shap_exp is not None:
                f_shap.write(json.dumps(shap_exp.to_dict()) + '\n')
            if lime_exp is not None:
                f_lime.write(json.dumps(lime_exp.to_dict()) + '\n')


if __name__ == "__main__":
    print("SHAP/LIME Explainer - Feature Attribution for HoloLoom Alignment")
    print("=" * 70)
    print("\nComponents:")
    print("  - SHAPExplainer: Game-theoretic feature attribution")
    print("  - LIMEExplainer: Local linear approximations")
    print("  - UnifiedExplainer: Combined SHAP + LIME interface")
    print("\nUsage:")
    print("  from HoloLoom.alignment.shap_lime_explainer import UnifiedExplainer")
    print("  explainer = UnifiedExplainer(predict_fn, feature_names)")
    print("  shap, lime = await explainer.explain(query_id, text, features, tool, conf)")