"""
Counterfactual Generation - "What if we changed X?"

Generates counterfactual explanations: minimal changes to inputs that
would lead to a different decision. Leverages twin networks from Layer 4.

Research:
- Wachter et al. (2017): Counterfactual Explanations without Opening the Black Box
- Mothilal et al. (2020): DiCE - Diverse Counterfactual Explanations
- Van Looveren & Klaise (2021): Interpretable Counterfactual Explanations
- Pearl (2009): Causality - Counterfactual reasoning foundations
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable, Set
from enum import Enum

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class CounterfactualMethod(Enum):
    """Counterfactual generation methods"""
    MINIMAL_EDIT = "minimal_edit"  # Find minimum changes
    DIVERSE = "diverse"  # Generate diverse counterfactuals
    FEASIBLE = "feasible"  # Only feasible/realistic changes
    CAUSAL = "causal"  # Respect causal structure
    TWIN_NETWORK = "twin_network"  # Use twin networks (Layer 4)


@dataclass
class Counterfactual:
    """A counterfactual explanation"""
    original: Dict[str, Any]  # Original input
    counterfactual: Dict[str, Any]  # Modified input
    changes: Dict[str, tuple]  # Changes: {feature: (old_value, new_value)}

    # Predictions
    original_prediction: Any
    counterfactual_prediction: Any

    # Metadata
    num_changes: int = 0
    distance: float = 0.0  # Distance from original
    feasibility: float = 1.0  # How feasible is this change [0, 1]
    confidence: float = 1.0  # Confidence in this counterfactual

    def __post_init__(self):
        """Compute statistics"""
        self.num_changes = len(self.changes)

    def __repr__(self) -> str:
        return f"Counterfactual({self.num_changes} changes, distance={self.distance:.3f})"

    def explain(self) -> str:
        """Generate natural language explanation"""
        if not self.changes:
            return "No changes needed."

        explanation = []
        explanation.append(f"To change the prediction from '{self.original_prediction}' to '{self.counterfactual_prediction}', ")
        explanation.append(f"you would need to make {self.num_changes} change(s):\n")

        for feature, (old_val, new_val) in self.changes.items():
            explanation.append(f"  • Change {feature} from {old_val} to {new_val}")

        return "".join(explanation)


@dataclass
class MinimalEdit:
    """Represents a minimal edit to achieve different prediction"""
    feature: str
    old_value: Any
    new_value: Any
    impact: float  # Impact on prediction

    def __repr__(self) -> str:
        return f"{self.feature}: {self.old_value} → {self.new_value} (impact={self.impact:.3f})"


class CounterfactualGenerator:
    """
    Generate counterfactual explanations for model predictions.

    Finds minimal changes to inputs that would flip the prediction,
    helping users understand decision boundaries.
    """

    def __init__(
        self,
        model: Optional[Callable] = None,
        method: CounterfactualMethod = CounterfactualMethod.MINIMAL_EDIT,
        max_changes: int = 3,
        twin_network: Optional[Any] = None  # From Layer 4
    ):
        """
        Args:
            model: Black-box model (features -> prediction)
            method: Counterfactual generation method
            max_changes: Maximum number of features to change
            twin_network: Twin network from Layer 4 (for exact counterfactuals)
        """
        self.model = model
        self.method = method
        self.max_changes = max_changes
        self.twin_network = twin_network

    def generate(
        self,
        features: Dict[str, Any],
        target_prediction: Any,
        current_prediction: Optional[Any] = None,
        num_counterfactuals: int = 1
    ) -> List[Counterfactual]:
        """
        Generate counterfactual explanations.

        Args:
            features: Original input features
            target_prediction: Desired prediction
            current_prediction: Current model prediction (optional)
            num_counterfactuals: Number of diverse counterfactuals to generate

        Returns:
            List of counterfactual explanations
        """
        if self.method == CounterfactualMethod.TWIN_NETWORK and self.twin_network:
            return self._twin_network_counterfactual(features, target_prediction, num_counterfactuals)
        elif self.method == CounterfactualMethod.MINIMAL_EDIT:
            return self._minimal_edit_counterfactual(features, target_prediction, num_counterfactuals)
        elif self.method == CounterfactualMethod.DIVERSE:
            return self._diverse_counterfactuals(features, target_prediction, num_counterfactuals)
        else:
            return self._greedy_search_counterfactual(features, target_prediction)

    def _twin_network_counterfactual(
        self,
        features: Dict[str, Any],
        target_prediction: Any,
        num_counterfactuals: int
    ) -> List[Counterfactual]:
        """
        Use twin networks (Layer 4) for exact counterfactual reasoning.

        Twin networks can compute P(Y_1 | do(X_0 = x_1)) exactly,
        answering "what if we had taken a different action?"
        """
        if not hasattr(self.twin_network, 'counterfactual'):
            # Fallback to minimal edit
            return self._minimal_edit_counterfactual(features, target_prediction, num_counterfactuals)

        counterfactuals = []

        # Use twin network to find interventions
        # Twin network: (actual_state, counterfactual_state) -> (actual_outcome, counterfactual_outcome)
        for i in range(num_counterfactuals):
            # Generate different interventions
            intervention = self._generate_intervention(features, i)

            # Query twin network
            actual_outcome, cf_outcome = self.twin_network.counterfactual(
                actual_state=features,
                counterfactual_state=intervention
            )

            # Check if this achieves target
            if cf_outcome == target_prediction:
                changes = {}
                for key in intervention.keys():
                    if intervention[key] != features.get(key):
                        changes[key] = (features.get(key), intervention[key])

                counterfactual = Counterfactual(
                    original=features,
                    counterfactual=intervention,
                    changes=changes,
                    original_prediction=actual_outcome,
                    counterfactual_prediction=cf_outcome,
                    distance=self._compute_distance(features, intervention),
                    feasibility=0.95,  # Twin networks guarantee causal consistency
                    confidence=0.99  # Exact counterfactual
                )
                counterfactuals.append(counterfactual)

        return counterfactuals

    def _minimal_edit_counterfactual(
        self,
        features: Dict[str, Any],
        target_prediction: Any,
        num_counterfactuals: int = 1
    ) -> List[Counterfactual]:
        """
        Find minimal edits (fewest feature changes) to flip prediction.

        Uses greedy search: change features one-by-one to maximize impact.
        """
        current_pred = self._evaluate_model(features) if self.model else None
        counterfactuals = []

        for _ in range(num_counterfactuals):
            # Greedy search
            modified = features.copy()
            changes = {}
            changed_features = set()

            for step in range(self.max_changes):
                # Try changing each unchanged feature
                best_feature = None
                best_value = None
                best_impact = 0.0

                for feature_name, current_value in features.items():
                    if feature_name in changed_features:
                        continue

                    # Try different values
                    for new_value in self._get_candidate_values(feature_name, current_value):
                        test_features = modified.copy()
                        test_features[feature_name] = new_value

                        test_pred = self._evaluate_model(test_features)

                        # Measure impact (how much closer to target)
                        impact = self._prediction_similarity(test_pred, target_prediction)

                        if impact > best_impact:
                            best_impact = impact
                            best_feature = feature_name
                            best_value = new_value

                # Apply best change
                if best_feature:
                    modified[best_feature] = best_value
                    changes[best_feature] = (features[best_feature], best_value)
                    changed_features.add(best_feature)

                    # Check if we reached target
                    new_pred = self._evaluate_model(modified)
                    if self._predictions_match(new_pred, target_prediction):
                        break
                else:
                    break

            # Create counterfactual
            cf_pred = self._evaluate_model(modified)
            counterfactual = Counterfactual(
                original=features,
                counterfactual=modified,
                changes=changes,
                original_prediction=current_pred,
                counterfactual_prediction=cf_pred,
                distance=self._compute_distance(features, modified),
                feasibility=0.8,  # Greedy may not find most feasible
                confidence=0.7
            )
            counterfactuals.append(counterfactual)

            # Generate diverse next counterfactual
            num_counterfactuals -= 1

        return counterfactuals

    def _diverse_counterfactuals(
        self,
        features: Dict[str, Any],
        target_prediction: Any,
        num_counterfactuals: int
    ) -> List[Counterfactual]:
        """
        Generate diverse counterfactuals (DiCE approach).

        Returns multiple counterfactuals that are different from each other,
        showing various ways to achieve the target.
        """
        counterfactuals = []
        used_features = set()

        for i in range(num_counterfactuals):
            # Generate counterfactual avoiding already-used features
            cf = self._generate_single_diverse_cf(
                features,
                target_prediction,
                avoid_features=used_features
            )
            counterfactuals.append(cf)

            # Track which features were changed
            for feature in cf.changes.keys():
                used_features.add(feature)

        return counterfactuals

    def _generate_single_diverse_cf(
        self,
        features: Dict[str, Any],
        target_prediction: Any,
        avoid_features: Set[str]
    ) -> Counterfactual:
        """Generate single diverse counterfactual avoiding certain features"""
        modified = features.copy()
        changes = {}

        # Select features not already used
        available_features = [f for f in features.keys() if f not in avoid_features]

        # Randomly change available features
        import random
        num_to_change = min(self.max_changes, len(available_features))
        features_to_change = random.sample(available_features, num_to_change)

        for feature_name in features_to_change:
            current_value = features[feature_name]
            new_value = random.choice(self._get_candidate_values(feature_name, current_value))
            modified[feature_name] = new_value
            changes[feature_name] = (current_value, new_value)

        current_pred = self._evaluate_model(features) if self.model else None
        cf_pred = self._evaluate_model(modified) if self.model else None

        return Counterfactual(
            original=features,
            counterfactual=modified,
            changes=changes,
            original_prediction=current_pred,
            counterfactual_prediction=cf_pred,
            distance=self._compute_distance(features, modified),
            feasibility=0.7,  # Random changes may not be most feasible
            confidence=0.6
        )

    def _greedy_search_counterfactual(
        self,
        features: Dict[str, Any],
        target_prediction: Any
    ) -> List[Counterfactual]:
        """Simple greedy search fallback"""
        return self._minimal_edit_counterfactual(features, target_prediction, num_counterfactuals=1)

    def _generate_intervention(
        self,
        features: Dict[str, Any],
        seed: int
    ) -> Dict[str, Any]:
        """Generate intervention for twin network"""
        import random
        random.seed(seed)

        intervention = features.copy()
        num_changes = random.randint(1, self.max_changes)
        features_to_change = random.sample(list(features.keys()), num_changes)

        for feature_name in features_to_change:
            current_value = features[feature_name]
            candidates = self._get_candidate_values(feature_name, current_value)
            intervention[feature_name] = random.choice(candidates)

        return intervention

    def _get_candidate_values(
        self,
        feature_name: str,
        current_value: Any
    ) -> List[Any]:
        """Get candidate values for a feature"""
        if isinstance(current_value, bool):
            return [not current_value]
        elif isinstance(current_value, int):
            # Try nearby values
            return [current_value - 1, current_value + 1, 0, current_value * 2]
        elif isinstance(current_value, float):
            # Try nearby values
            return [current_value * 0.5, current_value * 0.8, current_value * 1.2, current_value * 1.5]
        elif isinstance(current_value, str):
            # Categorical: try common alternatives
            return ["alternative_1", "alternative_2", "other"]
        else:
            return [current_value]  # Can't change

    def _evaluate_model(self, features: Dict[str, Any]) -> Any:
        """Evaluate model on features"""
        if self.model is None:
            return None

        result = self.model(features)

        # Convert to comparable type
        if TORCH_AVAILABLE and isinstance(result, torch.Tensor):
            if result.numel() == 1:
                return float(result.item())
            else:
                return result.argmax().item()
        elif isinstance(result, (list, tuple)):
            return result[0] if len(result) > 0 else None
        else:
            return result

    def _compute_distance(
        self,
        features1: Dict[str, Any],
        features2: Dict[str, Any]
    ) -> float:
        """Compute distance between two feature sets"""
        distance = 0.0
        num_features = len(features1)

        for key in features1.keys():
            val1 = features1.get(key)
            val2 = features2.get(key)

            if val1 != val2:
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # Numeric: normalized difference
                    distance += abs(val1 - val2) / (abs(val1) + abs(val2) + 1e-10)
                else:
                    # Categorical: binary distance
                    distance += 1.0

        return distance / num_features

    def _prediction_similarity(
        self,
        pred1: Any,
        pred2: Any
    ) -> float:
        """Measure how similar two predictions are"""
        if pred1 == pred2:
            return 1.0
        elif isinstance(pred1, (int, float)) and isinstance(pred2, (int, float)):
            # Numeric predictions: inverse distance
            diff = abs(pred1 - pred2)
            return 1.0 / (1.0 + diff)
        else:
            return 0.0

    def _predictions_match(
        self,
        pred1: Any,
        pred2: Any
    ) -> bool:
        """Check if predictions match"""
        if isinstance(pred1, float) and isinstance(pred2, float):
            return abs(pred1 - pred2) < 0.01
        else:
            return pred1 == pred2


def find_counterfactuals(
    model: Callable,
    features: Dict[str, Any],
    target_prediction: Any,
    method: CounterfactualMethod = CounterfactualMethod.MINIMAL_EDIT,
    num_counterfactuals: int = 3
) -> List[Counterfactual]:
    """
    Convenience function to find counterfactuals.

    Args:
        model: Black-box model
        features: Original input
        target_prediction: Desired prediction
        method: Counterfactual generation method
        num_counterfactuals: Number of diverse counterfactuals

    Returns:
        List of counterfactual explanations
    """
    generator = CounterfactualGenerator(model=model, method=method)
    return generator.generate(features, target_prediction, num_counterfactuals=num_counterfactuals)
