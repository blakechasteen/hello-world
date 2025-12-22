"""
Lens Composer: Multi-Domain Lens Composition

This module provides tools for composing multiple domain lenses into
unified interpretability systems, with routing and weighting strategies.

Key Capabilities:
- Compose multiple lenses with configurable weights
- Route queries to appropriate domains based on content
- Aggregate projections across domains
- Handle domain-specific normalization

Design Philosophy:
- Real applications span multiple domains
- Composition should be flexible and configurable
- Routing enables efficient multi-domain handling
- Aggregation preserves interpretability

Usage:
    # Basic composition
    composer = LensComposer()
    composer.add(MedicalDomain().lens, weight=0.6, name="medical")
    composer.add(SafetyDomain().lens, weight=0.4, name="safety")
    composed = composer.compose()

    projection = composed.project(activations)
    # {"medical.severity": 0.8, "medical.urgency": 0.5, "safety.harm": 0.3, ...}

    # Routing-based composition
    router = DomainRouter()
    router.add_rule(RoutingRule(pattern="medical|health|patient", domain="medical"))
    router.add_rule(RoutingRule(pattern="legal|law|court", domain="legal"))

    domain = router.route("What is the patient's diagnosis?")
    # Returns "medical"
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union
from enum import Enum
import numpy as np
import re


class LensProtocol(Protocol):
    """Protocol for lens objects that can be composed."""

    def project(self, activations: np.ndarray) -> Dict[str, float]:
        """Project activations onto semantic space."""
        ...

    def get_axis_names(self) -> List[str]:
        """Get names of all axes."""
        ...


class CompositionStrategy(Enum):
    """Strategy for combining projections from multiple lenses."""

    CONCATENATE = "concatenate"  # Prefix with lens name, keep all
    WEIGHTED_AVERAGE = "weighted_average"  # Average overlapping axes
    MAX_POOL = "max_pool"  # Take max for overlapping axes
    UNION = "union"  # Keep all, rename duplicates
    INTERSECTION = "intersection"  # Keep only common axes


@dataclass
class RoutingRule:
    """Rule for routing queries to domains."""

    pattern: str  # Regex pattern
    domain: str  # Target domain name
    priority: int = 0  # Higher priority wins ties
    min_match_score: float = 0.0  # Minimum score to trigger

    def matches(self, text: str) -> Tuple[bool, float]:
        """
        Check if text matches this rule.

        Returns (matches, score) tuple.
        """
        try:
            matches = re.findall(self.pattern, text.lower())
            if matches:
                # Score based on number and length of matches
                score = sum(len(m) for m in matches) / len(text)
                return score >= self.min_match_score, score
            return False, 0.0
        except re.error:
            return False, 0.0


@dataclass
class CompositionResult:
    """Result of lens composition."""

    projection: Dict[str, float]
    source_projections: Dict[str, Dict[str, float]]
    weights_used: Dict[str, float]
    strategy: CompositionStrategy
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_domain_projection(self, domain: str) -> Dict[str, float]:
        """Get projection for a specific domain."""
        return self.source_projections.get(domain, {})

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Composition ({self.strategy.value}):",
            f"  Domains: {list(self.source_projections.keys())}",
            f"  Total axes: {len(self.projection)}",
        ]

        # Top axes by absolute value
        sorted_axes = sorted(
            self.projection.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]

        lines.append("  Top axes:")
        for name, value in sorted_axes:
            lines.append(f"    {name}: {value:.3f}")

        return "\n".join(lines)


class ComposedLens:
    """
    A lens composed from multiple source lenses.

    Provides unified projection interface while maintaining
    access to individual domain projections.
    """

    def __init__(
        self,
        lenses: Dict[str, LensProtocol],
        weights: Dict[str, float],
        strategy: CompositionStrategy = CompositionStrategy.CONCATENATE,
    ):
        """
        Initialize composed lens.

        Args:
            lenses: Dict mapping domain names to lens objects
            weights: Dict mapping domain names to weights
            strategy: Strategy for combining projections
        """
        self.lenses = lenses
        self.weights = weights
        self.strategy = strategy

        # Validate
        for name in lenses:
            if name not in weights:
                weights[name] = 1.0

    def project(self, activations: np.ndarray) -> Dict[str, float]:
        """
        Project activations through all lenses.

        Returns combined projection based on strategy.
        """
        result = self.project_all(activations)
        return result.projection

    def project_all(self, activations: np.ndarray) -> CompositionResult:
        """
        Project through all lenses with full metadata.

        Returns CompositionResult with source projections.
        """
        source_projections = {}

        for name, lens in self.lenses.items():
            projection = lens.project(activations)
            source_projections[name] = projection

        # Combine based on strategy
        combined = self._combine(source_projections)

        return CompositionResult(
            projection=combined,
            source_projections=source_projections,
            weights_used=self.weights.copy(),
            strategy=self.strategy,
        )

    def _combine(
        self, source_projections: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Combine projections using configured strategy."""
        if self.strategy == CompositionStrategy.CONCATENATE:
            return self._combine_concatenate(source_projections)
        elif self.strategy == CompositionStrategy.WEIGHTED_AVERAGE:
            return self._combine_weighted_average(source_projections)
        elif self.strategy == CompositionStrategy.MAX_POOL:
            return self._combine_max_pool(source_projections)
        elif self.strategy == CompositionStrategy.UNION:
            return self._combine_union(source_projections)
        elif self.strategy == CompositionStrategy.INTERSECTION:
            return self._combine_intersection(source_projections)
        else:
            return self._combine_concatenate(source_projections)

    def _combine_concatenate(
        self, source_projections: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Concatenate with domain prefix."""
        combined = {}
        for domain, projection in source_projections.items():
            weight = self.weights.get(domain, 1.0)
            for axis, value in projection.items():
                key = f"{domain}.{axis}"
                combined[key] = value * weight
        return combined

    def _combine_weighted_average(
        self, source_projections: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Weighted average for overlapping axes."""
        axis_values: Dict[str, List[Tuple[float, float]]] = {}

        for domain, projection in source_projections.items():
            weight = self.weights.get(domain, 1.0)
            for axis, value in projection.items():
                if axis not in axis_values:
                    axis_values[axis] = []
                axis_values[axis].append((value, weight))

        combined = {}
        for axis, values in axis_values.items():
            total_weight = sum(w for _, w in values)
            if total_weight > 0:
                combined[axis] = sum(v * w for v, w in values) / total_weight
            else:
                combined[axis] = 0.0

        return combined

    def _combine_max_pool(
        self, source_projections: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Take max absolute value for overlapping axes."""
        axis_values: Dict[str, List[float]] = {}

        for domain, projection in source_projections.items():
            weight = self.weights.get(domain, 1.0)
            for axis, value in projection.items():
                if axis not in axis_values:
                    axis_values[axis] = []
                axis_values[axis].append(value * weight)

        combined = {}
        for axis, values in axis_values.items():
            # Max by absolute value, keep sign
            max_idx = np.argmax([abs(v) for v in values])
            combined[axis] = values[max_idx]

        return combined

    def _combine_union(
        self, source_projections: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Keep all axes, rename duplicates."""
        combined = {}
        seen = set()

        for domain, projection in source_projections.items():
            weight = self.weights.get(domain, 1.0)
            for axis, value in projection.items():
                key = axis
                if key in seen:
                    key = f"{domain}_{axis}"
                combined[key] = value * weight
                seen.add(axis)

        return combined

    def _combine_intersection(
        self, source_projections: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Keep only axes present in all domains."""
        if not source_projections:
            return {}

        # Find common axes
        axis_sets = [set(proj.keys()) for proj in source_projections.values()]
        common = axis_sets[0]
        for s in axis_sets[1:]:
            common = common.intersection(s)

        # Average common axes
        combined = {}
        for axis in common:
            values = []
            weights = []
            for domain, projection in source_projections.items():
                values.append(projection[axis])
                weights.append(self.weights.get(domain, 1.0))

            total_weight = sum(weights)
            if total_weight > 0:
                combined[axis] = sum(v * w for v, w in zip(values, weights)) / total_weight

        return combined

    def get_axis_names(self) -> List[str]:
        """Get all axis names across all lenses."""
        names = set()
        for lens in self.lenses.values():
            names.update(lens.get_axis_names())
        return sorted(names)

    def get_domain_names(self) -> List[str]:
        """Get names of all composed domains."""
        return list(self.lenses.keys())


class LensComposer:
    """
    Builder for composing multiple lenses.

    Fluent interface for constructing ComposedLens objects.
    """

    def __init__(self, strategy: CompositionStrategy = CompositionStrategy.CONCATENATE):
        """
        Initialize composer.

        Args:
            strategy: Default composition strategy
        """
        self.strategy = strategy
        self._lenses: Dict[str, LensProtocol] = {}
        self._weights: Dict[str, float] = {}

    def add(
        self,
        lens: LensProtocol,
        name: Optional[str] = None,
        weight: float = 1.0,
    ) -> "LensComposer":
        """
        Add a lens to the composition.

        Args:
            lens: Lens object to add
            name: Optional name (auto-generated if not provided)
            weight: Weight for this lens (default 1.0)

        Returns:
            Self for chaining
        """
        if name is None:
            name = f"lens_{len(self._lenses)}"

        self._lenses[name] = lens
        self._weights[name] = weight
        return self

    def set_weight(self, name: str, weight: float) -> "LensComposer":
        """Set weight for a lens."""
        if name in self._lenses:
            self._weights[name] = weight
        return self

    def set_strategy(self, strategy: CompositionStrategy) -> "LensComposer":
        """Set composition strategy."""
        self.strategy = strategy
        return self

    def remove(self, name: str) -> "LensComposer":
        """Remove a lens from the composition."""
        if name in self._lenses:
            del self._lenses[name]
            del self._weights[name]
        return self

    def compose(self) -> ComposedLens:
        """
        Create the composed lens.

        Returns:
            ComposedLens with all added lenses
        """
        return ComposedLens(
            lenses=self._lenses.copy(),
            weights=self._weights.copy(),
            strategy=self.strategy,
        )

    def reset(self) -> "LensComposer":
        """Clear all added lenses."""
        self._lenses.clear()
        self._weights.clear()
        return self


class DomainRouter:
    """
    Routes queries/inputs to appropriate domains.

    Uses pattern matching and optional classifiers to
    determine which domain lens to use.
    """

    def __init__(self, default_domain: Optional[str] = None):
        """
        Initialize router.

        Args:
            default_domain: Default domain when no rules match
        """
        self.default_domain = default_domain
        self._rules: List[RoutingRule] = []
        self._classifiers: Dict[str, Callable] = {}

    def add_rule(self, rule: RoutingRule) -> "DomainRouter":
        """
        Add a routing rule.

        Args:
            rule: RoutingRule to add

        Returns:
            Self for chaining
        """
        self._rules.append(rule)
        # Sort by priority (higher first)
        self._rules.sort(key=lambda r: r.priority, reverse=True)
        return self

    def add_classifier(
        self,
        name: str,
        classifier: Callable[[str], Tuple[str, float]],
    ) -> "DomainRouter":
        """
        Add a classifier function for domain detection.

        Args:
            name: Name for this classifier
            classifier: Function(text) -> (domain, confidence)

        Returns:
            Self for chaining
        """
        self._classifiers[name] = classifier
        return self

    def route(self, text: str) -> Optional[str]:
        """
        Route text to a domain.

        Args:
            text: Input text to route

        Returns:
            Domain name or None if no match
        """
        result = self.route_with_score(text)
        return result[0] if result[0] is not None else self.default_domain

    def route_with_score(self, text: str) -> Tuple[Optional[str], float]:
        """
        Route text with confidence score.

        Returns (domain, score) tuple.
        """
        best_domain = None
        best_score = 0.0

        # Check rules
        for rule in self._rules:
            matches, score = rule.matches(text)
            if matches and score > best_score:
                best_domain = rule.domain
                best_score = score

        # Check classifiers
        for name, classifier in self._classifiers.items():
            try:
                domain, score = classifier(text)
                if score > best_score:
                    best_domain = domain
                    best_score = score
            except Exception:
                pass

        return best_domain, best_score

    def route_multiple(
        self, text: str, top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Get multiple domain matches with scores.

        Returns list of (domain, score) tuples sorted by score.
        """
        scores: Dict[str, float] = {}

        # Check rules
        for rule in self._rules:
            matches, score = rule.matches(text)
            if matches:
                if rule.domain not in scores or score > scores[rule.domain]:
                    scores[rule.domain] = score

        # Check classifiers
        for name, classifier in self._classifiers.items():
            try:
                domain, score = classifier(text)
                if domain not in scores or score > scores[domain]:
                    scores[domain] = score
            except Exception:
                pass

        # Sort by score
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def route_and_project(
        self,
        text: str,
        activations: np.ndarray,
        lenses: Dict[str, LensProtocol],
    ) -> Tuple[str, Dict[str, float]]:
        """
        Route and project in one step.

        Args:
            text: Text to route
            activations: Activations to project
            lenses: Dict mapping domain names to lenses

        Returns:
            (domain, projection) tuple
        """
        domain = self.route(text)
        if domain is None or domain not in lenses:
            if self.default_domain and self.default_domain in lenses:
                domain = self.default_domain
            else:
                return None, {}

        lens = lenses[domain]
        projection = lens.project(activations)
        return domain, projection


def create_router_from_config(
    rules: List[Dict[str, Any]],
    default_domain: Optional[str] = None,
) -> DomainRouter:
    """
    Create router from configuration dict.

    Args:
        rules: List of rule dicts with pattern, domain, priority, min_match_score
        default_domain: Default domain

    Returns:
        Configured DomainRouter
    """
    router = DomainRouter(default_domain=default_domain)

    for rule_dict in rules:
        rule = RoutingRule(
            pattern=rule_dict["pattern"],
            domain=rule_dict["domain"],
            priority=rule_dict.get("priority", 0),
            min_match_score=rule_dict.get("min_match_score", 0.0),
        )
        router.add_rule(rule)

    return router


def compose_with_routing(
    lenses: Dict[str, LensProtocol],
    router: DomainRouter,
    text: str,
    activations: np.ndarray,
    fallback_strategy: CompositionStrategy = CompositionStrategy.CONCATENATE,
) -> CompositionResult:
    """
    Compose lenses with routing-based weighting.

    Routes to primary domain, uses it with higher weight.

    Args:
        lenses: Dict of domain -> lens
        router: Router for domain selection
        text: Text for routing
        activations: Activations to project
        fallback_strategy: Strategy when routing fails

    Returns:
        CompositionResult with routing-aware weights
    """
    # Get routing results
    matches = router.route_multiple(text, top_k=len(lenses))

    if not matches:
        # No matches: equal weights
        weights = {name: 1.0 for name in lenses}
    else:
        # Weight by routing scores
        total_score = sum(score for _, score in matches) or 1.0
        weights = {}
        for domain, score in matches:
            if domain in lenses:
                weights[domain] = score / total_score

        # Give remaining domains minimal weight
        for name in lenses:
            if name not in weights:
                weights[name] = 0.1

    # Compose
    composed = ComposedLens(lenses, weights, fallback_strategy)
    return composed.project_all(activations)
