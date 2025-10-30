"""
Decision Tree Extraction - Extract interpretable rules from neural networks

Approximates neural network behavior with decision trees/rules
that are human-interpretable.

Research:
- Craven & Shavlik (1996): Extracting Tree-Structured Representations
- Tan et al. (2018): Tree Space Prototypes
- Bastani et al. (2018): Interpreting Blackbox Models via Model Extraction
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Set
from enum import Enum


class SplitCriterion(Enum):
    """Splitting criteria for decision trees"""
    INFORMATION_GAIN = "information_gain"
    GINI = "gini"
    VARIANCE = "variance"


@dataclass
class DecisionNode:
    """Node in a decision tree"""
    feature: Optional[str] = None  # Feature to split on (None for leaf)
    threshold: Optional[float] = None  # Split threshold
    left: Optional['DecisionNode'] = None  # Left child (feature <= threshold)
    right: Optional['DecisionNode'] = None  # Right child (feature > threshold)
    prediction: Optional[Any] = None  # Prediction (for leaf nodes)
    samples: int = 0  # Number of samples at this node
    impurity: float = 1.0  # Node impurity
    depth: int = 0

    def is_leaf(self) -> bool:
        """Check if this is a leaf node"""
        return self.feature is None

    def __repr__(self) -> str:
        if self.is_leaf():
            return f"Leaf(predict={self.prediction}, samples={self.samples})"
        else:
            return f"Node(feature={self.feature}, threshold={self.threshold:.3f}, depth={self.depth})"


@dataclass
class Rule:
    """A decision rule extracted from tree"""
    conditions: List[str]  # List of conditions (e.g., ["age > 30", "income <= 50000"])
    prediction: Any  # Prediction if all conditions met
    confidence: float = 1.0  # Rule confidence
    support: int = 0  # Number of samples supporting this rule

    def __repr__(self) -> str:
        conditions_str = " AND ".join(self.conditions)
        return f"IF {conditions_str} THEN predict={self.prediction} (conf={self.confidence:.2f}, support={self.support})"


@dataclass
class RuleSet:
    """Collection of decision rules"""
    rules: List[Rule] = field(default_factory=list)

    def add_rule(self, rule: Rule):
        """Add a rule to the set"""
        self.rules.append(rule)

    def predict(self, features: Dict[str, Any]) -> Optional[Any]:
        """Predict using rules (first matching rule wins)"""
        for rule in self.rules:
            if self._matches_rule(features, rule):
                return rule.prediction
        return None

    def _matches_rule(self, features: Dict[str, Any], rule: Rule) -> bool:
        """Check if features satisfy all rule conditions"""
        for condition in rule.conditions:
            if not self._evaluate_condition(features, condition):
                return False
        return True

    def _evaluate_condition(self, features: Dict[str, Any], condition: str) -> bool:
        """Evaluate a single condition"""
        # Parse condition: "feature_name > threshold" or "feature_name <= threshold"
        for op in [" > ", " <= ", " >= ", " < ", " == ", " != "]:
            if op in condition:
                feature_name, value_str = condition.split(op)
                feature_name = feature_name.strip()
                value_str = value_str.strip()

                if feature_name not in features:
                    return False

                feature_value = features[feature_name]

                try:
                    threshold = float(value_str)
                except ValueError:
                    threshold = value_str  # String comparison

                # Evaluate
                if op == " > ":
                    return feature_value > threshold
                elif op == " <= ":
                    return feature_value <= threshold
                elif op == " >= ":
                    return feature_value >= threshold
                elif op == " < ":
                    return feature_value < threshold
                elif op == " == ":
                    return feature_value == threshold
                elif op == " != ":
                    return feature_value != threshold

        return False

    def __repr__(self) -> str:
        return f"RuleSet({len(self.rules)} rules)"


class DecisionTreeExtractor:
    """
    Extract decision tree from neural network.

    Approximates the neural network's decision boundary
    using an interpretable decision tree.
    """

    def __init__(
        self,
        model: Optional[Callable] = None,
        max_depth: int = 5,
        min_samples_split: int = 10,
        criterion: SplitCriterion = SplitCriterion.INFORMATION_GAIN
    ):
        """
        Args:
            model: Black-box model to extract tree from
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split a node
            criterion: Splitting criterion
        """
        self.model = model
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.tree: Optional[DecisionNode] = None

    def extract(
        self,
        training_data: List[Dict[str, Any]],
        feature_names: Optional[List[str]] = None
    ) -> DecisionNode:
        """
        Extract decision tree from model.

        Args:
            training_data: Training examples (will query model on these)
            feature_names: Names of features to consider

        Returns:
            Root node of extracted decision tree
        """
        if not training_data:
            raise ValueError("Need training data to extract tree")

        # Infer feature names
        if feature_names is None:
            feature_names = list(training_data[0].keys())

        # Query model on training data to get labels
        labels = []
        for example in training_data:
            prediction = self._query_model(example)
            labels.append(prediction)

        # Build tree
        self.tree = self._build_tree(
            training_data,
            labels,
            feature_names,
            depth=0
        )

        return self.tree

    def _build_tree(
        self,
        data: List[Dict[str, Any]],
        labels: List[Any],
        feature_names: List[str],
        depth: int
    ) -> DecisionNode:
        """Recursively build decision tree"""
        num_samples = len(data)

        # Base cases
        if depth >= self.max_depth or num_samples < self.min_samples_split:
            # Create leaf
            prediction = self._majority_class(labels)
            return DecisionNode(
                prediction=prediction,
                samples=num_samples,
                impurity=self._compute_impurity(labels),
                depth=depth
            )

        # All same label -> leaf
        if len(set(labels)) == 1:
            return DecisionNode(
                prediction=labels[0],
                samples=num_samples,
                impurity=0.0,
                depth=depth
            )

        # Find best split
        best_feature, best_threshold, best_gain = self._find_best_split(
            data,
            labels,
            feature_names
        )

        if best_feature is None or best_gain <= 0:
            # No good split -> leaf
            prediction = self._majority_class(labels)
            return DecisionNode(
                prediction=prediction,
                samples=num_samples,
                impurity=self._compute_impurity(labels),
                depth=depth
            )

        # Split data
        left_data, left_labels, right_data, right_labels = self._split_data(
            data,
            labels,
            best_feature,
            best_threshold
        )

        # Recursively build children
        left_child = self._build_tree(left_data, left_labels, feature_names, depth + 1)
        right_child = self._build_tree(right_data, right_labels, feature_names, depth + 1)

        return DecisionNode(
            feature=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child,
            samples=num_samples,
            impurity=self._compute_impurity(labels),
            depth=depth
        )

    def _find_best_split(
        self,
        data: List[Dict[str, Any]],
        labels: List[Any],
        feature_names: List[str]
    ) -> tuple:
        """Find best feature and threshold to split on"""
        best_feature = None
        best_threshold = None
        best_gain = -float('inf')

        current_impurity = self._compute_impurity(labels)

        for feature in feature_names:
            # Get feature values
            values = [example.get(feature, 0) for example in data]

            if not all(isinstance(v, (int, float)) for v in values):
                continue  # Skip non-numeric features

            # Try different thresholds
            unique_values = sorted(set(values))
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2.0

                # Split
                left_labels = [labels[j] for j in range(len(data)) if values[j] <= threshold]
                right_labels = [labels[j] for j in range(len(data)) if values[j] > threshold]

                if not left_labels or not right_labels:
                    continue

                # Compute information gain
                left_impurity = self._compute_impurity(left_labels)
                right_impurity = self._compute_impurity(right_labels)

                left_weight = len(left_labels) / len(labels)
                right_weight = len(right_labels) / len(labels)

                weighted_impurity = left_weight * left_impurity + right_weight * right_impurity
                gain = current_impurity - weighted_impurity

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _split_data(
        self,
        data: List[Dict[str, Any]],
        labels: List[Any],
        feature: str,
        threshold: float
    ) -> tuple:
        """Split data on feature <= threshold"""
        left_data = []
        left_labels = []
        right_data = []
        right_labels = []

        for i, example in enumerate(data):
            value = example.get(feature, 0)
            if value <= threshold:
                left_data.append(example)
                left_labels.append(labels[i])
            else:
                right_data.append(example)
                right_labels.append(labels[i])

        return left_data, left_labels, right_data, right_labels

    def _compute_impurity(self, labels: List[Any]) -> float:
        """Compute impurity (entropy or Gini)"""
        if not labels:
            return 0.0

        # Count classes
        class_counts = {}
        for label in labels:
            class_counts[label] = class_counts.get(label, 0) + 1

        total = len(labels)

        if self.criterion == SplitCriterion.GINI:
            # Gini impurity: 1 - Σ p_i^2
            impurity = 1.0
            for count in class_counts.values():
                prob = count / total
                impurity -= prob ** 2
            return impurity

        else:  # Information gain (entropy)
            # Entropy: -Σ p_i log p_i
            import math
            entropy = 0.0
            for count in class_counts.values():
                prob = count / total
                if prob > 0:
                    entropy -= prob * math.log2(prob)
            return entropy

    def _majority_class(self, labels: List[Any]) -> Any:
        """Return most common label"""
        if not labels:
            return None

        class_counts = {}
        for label in labels:
            class_counts[label] = class_counts.get(label, 0) + 1

        return max(class_counts.items(), key=lambda x: x[1])[0]

    def _query_model(self, features: Dict[str, Any]) -> Any:
        """Query model for prediction"""
        if self.model is None:
            return 0

        result = self.model(features)

        # Convert to simple type
        if isinstance(result, (int, float, str, bool)):
            return result
        else:
            return str(result)

    def extract_rules(self) -> RuleSet:
        """
        Extract rules from tree.

        Converts tree into IF-THEN rules.
        """
        if self.tree is None:
            return RuleSet()

        rules = RuleSet()
        self._extract_rules_recursive(self.tree, [], rules)
        return rules

    def _extract_rules_recursive(
        self,
        node: DecisionNode,
        conditions: List[str],
        ruleset: RuleSet
    ):
        """Recursively extract rules from tree"""
        if node.is_leaf():
            # Create rule
            rule = Rule(
                conditions=conditions.copy(),
                prediction=node.prediction,
                confidence=1.0 - node.impurity,  # Low impurity = high confidence
                support=node.samples
            )
            ruleset.add_rule(rule)
        else:
            # Traverse left
            left_condition = f"{node.feature} <= {node.threshold:.3f}"
            self._extract_rules_recursive(
                node.left,
                conditions + [left_condition],
                ruleset
            )

            # Traverse right
            right_condition = f"{node.feature} > {node.threshold:.3f}"
            self._extract_rules_recursive(
                node.right,
                conditions + [right_condition],
                ruleset
            )


def extract_rules(
    model: Callable,
    training_data: List[Dict[str, Any]],
    max_depth: int = 5
) -> RuleSet:
    """
    Convenience function to extract rules from model.

    Args:
        model: Black-box model
        training_data: Training examples
        max_depth: Maximum tree depth

    Returns:
        RuleSet with extracted rules
    """
    extractor = DecisionTreeExtractor(model=model, max_depth=max_depth)
    extractor.extract(training_data)
    return extractor.extract_rules()


def visualize_tree(
    node: DecisionNode,
    indent: str = ""
) -> str:
    """
    Visualize decision tree as text.

    Args:
        node: Root node of tree
        indent: Indentation string (for recursion)

    Returns:
        Text representation of tree
    """
    if node.is_leaf():
        return f"{indent}└─ Predict: {node.prediction} (samples={node.samples}, impurity={node.impurity:.3f})\n"
    else:
        text = []
        text.append(f"{indent}├─ {node.feature} <= {node.threshold:.3f}?\n")
        text.append(visualize_tree(node.left, indent + "│  "))
        text.append(f"{indent}└─ {node.feature} > {node.threshold:.3f}?\n")
        text.append(visualize_tree(node.right, indent + "   "))
        return "".join(text)
