"""
Feature Attribution - Which features drove the decision?

Implements SHAP-style and LIME-style explanations for understanding
which input features contributed most to a model's decision.

Research:
- Lundberg & Lee (2017): SHAP - SHapley Additive exPlanations
- Ribeiro et al. (2016): LIME - Local Interpretable Model-Agnostic Explanations
- Shapley (1953): A value for n-person games (Nobel Prize foundation)
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable, Tuple
from enum import Enum
import itertools

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class AttributionMethod(Enum):
    """Attribution methods for feature importance"""
    SHAPLEY = "shapley"  # Exact Shapley values (exponential time)
    KERNEL_SHAP = "kernel_shap"  # Approximate Shapley (polynomial time)
    LIME = "lime"  # Local linear approximation
    INTEGRATED_GRADIENTS = "integrated_gradients"  # Gradient-based (neural nets)
    ATTENTION_WEIGHTS = "attention_weights"  # Direct attention scores
    ABLATION = "ablation"  # Feature removal impact


@dataclass
class FeatureImportance:
    """Feature importance attribution"""
    feature_name: str
    importance: float  # Contribution to decision (-inf, +inf)
    confidence: float  # Confidence in this attribution [0, 1]
    method: AttributionMethod

    # Additional metadata
    rank: Optional[int] = None  # Rank among all features
    percentile: Optional[float] = None  # Percentile importance
    is_positive: bool = True  # Positive or negative contribution

    def __repr__(self) -> str:
        sign = "+" if self.is_positive else "-"
        return f"{self.feature_name}: {sign}{abs(self.importance):.4f} (rank={self.rank})"


class FeatureAttributor:
    """
    Attribute decisions to input features using multiple methods.

    Works with any black-box model that outputs predictions.
    """

    def __init__(
        self,
        model: Optional[Callable] = None,
        method: AttributionMethod = AttributionMethod.KERNEL_SHAP,
        num_samples: int = 100,
    ):
        """
        Args:
            model: Black-box model (features -> prediction)
            method: Attribution method to use
            num_samples: Number of samples for approximation methods
        """
        self.model = model
        self.method = method
        self.num_samples = num_samples

    def attribute(
        self,
        features: Dict[str, Any],
        prediction: Any,
        baseline: Optional[Dict[str, Any]] = None,
    ) -> List[FeatureImportance]:
        """
        Attribute prediction to input features.

        Args:
            features: Input features that led to prediction
            prediction: Model's prediction (for reference)
            baseline: Baseline/reference features for comparison

        Returns:
            List of feature importances, sorted by magnitude
        """
        if self.method == AttributionMethod.SHAPLEY:
            return self._shapley_values(features, baseline)
        elif self.method == AttributionMethod.KERNEL_SHAP:
            return self._kernel_shap(features, baseline)
        elif self.method == AttributionMethod.LIME:
            return self._lime_approximation(features)
        elif self.method == AttributionMethod.ABLATION:
            return self._ablation_analysis(features, prediction)
        elif self.method == AttributionMethod.ATTENTION_WEIGHTS:
            return self._attention_weights(features)
        else:
            # Default: simple feature scoring
            return self._simple_attribution(features)

    def _shapley_values(
        self,
        features: Dict[str, Any],
        baseline: Optional[Dict[str, Any]] = None
    ) -> List[FeatureImportance]:
        """
        Exact Shapley values (exponential complexity).

        Shapley value φ_i = Σ [|S|! (|N| - |S| - 1)! / |N|!] [f(S ∪ {i}) - f(S)]

        For each feature i, average its marginal contribution over all subsets S.
        """
        if self.model is None:
            return self._simple_attribution(features)

        feature_names = list(features.keys())
        n = len(feature_names)
        shapley_values = {name: 0.0 for name in feature_names}

        baseline = baseline or {name: 0 for name in feature_names}

        # Iterate over all subsets (2^n - exponential!)
        for subset_size in range(n + 1):
            for subset in itertools.combinations(feature_names, subset_size):
                subset_features = {name: features[name] if name in subset else baseline[name]
                                   for name in feature_names}

                # f(S)
                pred_without = self._evaluate_model(subset_features)

                # For each feature not in subset, compute f(S ∪ {i})
                for feature_name in feature_names:
                    if feature_name not in subset:
                        # f(S ∪ {i})
                        subset_with_feature = subset_features.copy()
                        subset_with_feature[feature_name] = features[feature_name]
                        pred_with = self._evaluate_model(subset_with_feature)

                        # Marginal contribution
                        marginal = pred_with - pred_without

                        # Shapley weight: |S|! (|N| - |S| - 1)! / |N|!
                        weight = self._factorial(subset_size) * self._factorial(n - subset_size - 1) / self._factorial(n)
                        shapley_values[feature_name] += weight * marginal

        # Convert to FeatureImportance objects
        importances = []
        for rank, (name, value) in enumerate(sorted(shapley_values.items(), key=lambda x: abs(x[1]), reverse=True)):
            importances.append(FeatureImportance(
                feature_name=name,
                importance=value,
                confidence=1.0,  # Exact method
                method=AttributionMethod.SHAPLEY,
                rank=rank + 1,
                is_positive=value >= 0,
            ))

        return importances

    def _kernel_shap(
        self,
        features: Dict[str, Any],
        baseline: Optional[Dict[str, Any]] = None
    ) -> List[FeatureImportance]:
        """
        Kernel SHAP - approximate Shapley values (polynomial complexity).

        Uses weighted linear regression to approximate Shapley values.
        Much faster than exact Shapley for large feature sets.
        """
        if self.model is None:
            return self._simple_attribution(features)

        feature_names = list(features.keys())
        n = len(feature_names)
        baseline = baseline or {name: 0 for name in feature_names}

        # Sample random subsets
        import random
        samples = []
        weights = []
        predictions = []

        for _ in range(self.num_samples):
            # Random subset
            subset_size = random.randint(0, n)
            subset = set(random.sample(feature_names, subset_size))

            # Create feature vector (1 if included, 0 if baseline)
            binary_vec = [1 if name in subset else 0 for name in feature_names]
            samples.append(binary_vec)

            # SHAP kernel weight
            if subset_size == 0 or subset_size == n:
                weight = 1000  # High weight for empty/full sets
            else:
                weight = (n - 1) / (subset_size * (n - subset_size))
            weights.append(weight)

            # Evaluate model
            subset_features = {name: features[name] if name in subset else baseline[name]
                               for name in feature_names}
            pred = self._evaluate_model(subset_features)
            predictions.append(pred)

        # Weighted linear regression: pred ~ w1*f1 + w2*f2 + ... + wn*fn
        if NUMPY_AVAILABLE:
            X = np.array(samples)
            y = np.array(predictions)
            w = np.array(weights)

            # Weighted least squares: (X^T W X)^{-1} X^T W y
            try:
                W = np.diag(w)
                coefficients = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
            except np.linalg.LinAlgError:
                # Fallback: unweighted
                coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
        else:
            # Simple fallback: average prediction changes
            coefficients = [0.0] * n
            for i, name in enumerate(feature_names):
                with_feature = sum(p for s, p in zip(samples, predictions) if s[i] == 1)
                without_feature = sum(p for s, p in zip(samples, predictions) if s[i] == 0)
                count_with = sum(1 for s in samples if s[i] == 1)
                count_without = sum(1 for s in samples if s[i] == 0)

                if count_with > 0 and count_without > 0:
                    coefficients[i] = (with_feature / count_with) - (without_feature / count_without)

        # Convert to FeatureImportance
        importances = []
        importance_dict = {name: coef for name, coef in zip(feature_names, coefficients)}

        for rank, (name, value) in enumerate(sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)):
            importances.append(FeatureImportance(
                feature_name=name,
                importance=float(value),
                confidence=0.9,  # Approximate method
                method=AttributionMethod.KERNEL_SHAP,
                rank=rank + 1,
                is_positive=value >= 0,
            ))

        return importances

    def _lime_approximation(
        self,
        features: Dict[str, Any]
    ) -> List[FeatureImportance]:
        """
        LIME - Local Interpretable Model-Agnostic Explanations.

        Fits a simple linear model locally around the prediction.
        """
        if self.model is None:
            return self._simple_attribution(features)

        feature_names = list(features.keys())

        # Perturb features locally
        import random
        samples = []
        predictions = []
        distances = []

        for _ in range(self.num_samples):
            # Small perturbations
            perturbed = {}
            distance = 0.0

            for name, value in features.items():
                if isinstance(value, (int, float)):
                    # Numeric: Gaussian noise
                    noise = random.gauss(0, 0.1 * abs(value) if value != 0 else 0.1)
                    perturbed[name] = value + noise
                    distance += noise ** 2
                else:
                    # Non-numeric: keep or drop randomly
                    if random.random() > 0.3:
                        perturbed[name] = value
                    else:
                        perturbed[name] = 0
                        distance += 1.0

            samples.append([perturbed[name] if isinstance(perturbed[name], (int, float)) else 1
                            for name in feature_names])
            predictions.append(self._evaluate_model(perturbed))
            distances.append(distance)

        # Exponential kernel: weight = exp(-d^2 / σ^2)
        sigma = 1.0
        weights = [pow(2.718281828, -d / (sigma ** 2)) for d in distances]

        # Weighted linear regression
        if NUMPY_AVAILABLE:
            X = np.array(samples)
            y = np.array(predictions)
            w = np.array(weights)

            try:
                W = np.diag(w)
                coefficients = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
            except np.linalg.LinAlgError:
                coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
        else:
            # Fallback
            coefficients = [0.0] * len(feature_names)

        # Convert to FeatureImportance
        importances = []
        importance_dict = {name: coef for name, coef in zip(feature_names, coefficients)}

        for rank, (name, value) in enumerate(sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)):
            importances.append(FeatureImportance(
                feature_name=name,
                importance=float(value),
                confidence=0.8,  # Local approximation
                method=AttributionMethod.LIME,
                rank=rank + 1,
                is_positive=value >= 0,
            ))

        return importances

    def _ablation_analysis(
        self,
        features: Dict[str, Any],
        prediction: Any
    ) -> List[FeatureImportance]:
        """
        Ablation analysis: Remove each feature and measure impact.

        Simple but effective: shows what happens when feature is missing.
        """
        if self.model is None:
            return self._simple_attribution(features)

        baseline_pred = self._evaluate_model(features)
        importances = []

        for rank, (name, value) in enumerate(features.items()):
            # Remove feature
            ablated = features.copy()
            ablated[name] = 0  # Zero out feature

            # Measure impact
            ablated_pred = self._evaluate_model(ablated)
            impact = baseline_pred - ablated_pred  # Drop in prediction

            importances.append(FeatureImportance(
                feature_name=name,
                importance=impact,
                confidence=1.0,  # Direct measurement
                method=AttributionMethod.ABLATION,
                rank=rank + 1,
                is_positive=impact >= 0,
            ))

        # Sort by magnitude
        importances.sort(key=lambda x: abs(x.importance), reverse=True)
        for i, imp in enumerate(importances):
            imp.rank = i + 1

        return importances

    def _attention_weights(
        self,
        features: Dict[str, Any]
    ) -> List[FeatureImportance]:
        """
        Use attention weights as feature importance (if available).

        For transformer-based models with attention mechanisms.
        """
        # If features contain attention weights, use them directly
        if 'attention_weights' in features:
            weights = features['attention_weights']
            importances = []

            for rank, (name, weight) in enumerate(sorted(weights.items(), key=lambda x: x[1], reverse=True)):
                importances.append(FeatureImportance(
                    feature_name=name,
                    importance=weight,
                    confidence=0.95,
                    method=AttributionMethod.ATTENTION_WEIGHTS,
                    rank=rank + 1,
                    is_positive=True,
                ))

            return importances
        else:
            return self._simple_attribution(features)

    def _simple_attribution(
        self,
        features: Dict[str, Any]
    ) -> List[FeatureImportance]:
        """
        Simple heuristic attribution based on feature magnitudes.

        Fallback when no model is available.
        """
        importances = []

        for rank, (name, value) in enumerate(features.items()):
            # Use magnitude as importance
            if isinstance(value, (int, float)):
                importance = abs(value)
            elif isinstance(value, (list, tuple)) and len(value) > 0:
                importance = len(value)
            elif isinstance(value, str):
                importance = len(value) / 100.0  # Text length
            else:
                importance = 1.0

            importances.append(FeatureImportance(
                feature_name=name,
                importance=importance,
                confidence=0.5,  # Low confidence heuristic
                method=AttributionMethod.ABLATION,
                rank=rank + 1,
                is_positive=True,
            ))

        # Sort by magnitude
        importances.sort(key=lambda x: x.importance, reverse=True)
        for i, imp in enumerate(importances):
            imp.rank = i + 1

        return importances

    def _evaluate_model(self, features: Dict[str, Any]) -> float:
        """Evaluate model on features, return scalar prediction"""
        if self.model is None:
            return 0.0

        result = self.model(features)

        # Convert to scalar
        if isinstance(result, (int, float)):
            return float(result)
        elif TORCH_AVAILABLE and isinstance(result, torch.Tensor):
            return float(result.item() if result.numel() == 1 else result.mean())
        elif NUMPY_AVAILABLE and isinstance(result, np.ndarray):
            return float(result.item() if result.size == 1 else result.mean())
        else:
            return 0.0

    @staticmethod
    def _factorial(n: int) -> int:
        """Compute n!"""
        if n <= 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result


class ShapleyValues:
    """Convenience class for computing exact Shapley values"""

    @staticmethod
    def compute(
        model: Callable,
        features: Dict[str, Any],
        baseline: Optional[Dict[str, Any]] = None
    ) -> List[FeatureImportance]:
        """Compute exact Shapley values"""
        attributor = FeatureAttributor(model=model, method=AttributionMethod.SHAPLEY)
        return attributor.attribute(features, prediction=None, baseline=baseline)


class LimeExplainer:
    """Convenience class for LIME explanations"""

    @staticmethod
    def explain(
        model: Callable,
        features: Dict[str, Any],
        num_samples: int = 100
    ) -> List[FeatureImportance]:
        """Generate LIME explanation"""
        attributor = FeatureAttributor(model=model, method=AttributionMethod.LIME, num_samples=num_samples)
        return attributor.attribute(features, prediction=None)
