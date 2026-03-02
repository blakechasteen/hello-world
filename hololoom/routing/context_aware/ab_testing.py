"""
A/B Testing for Routing Strategies

Enables experimentation with different routing strategies to find optimal approach.

Features:
- Multiple routing variants with traffic splits
- Statistical significance testing
- Automatic winner selection
- Gradual rollout (champion/challenger pattern)

Author: HoloLoom B2B Framework
Date: November 2025
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import random
from collections import defaultdict


class RoutingVariant(Enum):
    """Routing strategy variants for A/B testing"""
    CONTROL = "control"           # Current production strategy
    VARIANT_A = "variant_a"       # Test strategy A
    VARIANT_B = "variant_b"       # Test strategy B
    VARIANT_C = "variant_c"       # Test strategy C


@dataclass
class ABTestConfig:
    """Configuration for A/B test"""
    test_name: str
    variants: Dict[RoutingVariant, float]  # variant → traffic %
    control_variant: RoutingVariant = RoutingVariant.CONTROL
    min_sample_size: int = 100
    confidence_threshold: float = 0.95
    auto_promote_winner: bool = False


@dataclass
class VariantMetrics:
    """Metrics for a single variant"""
    variant: RoutingVariant
    total_routes: int = 0
    successful_routes: int = 0
    total_confidence: float = 0.0
    avg_latency_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_routes == 0:
            return 0.0
        return self.successful_routes / self.total_routes

    @property
    def avg_confidence(self) -> float:
        """Calculate average confidence"""
        if self.total_routes == 0:
            return 0.0
        return self.total_confidence / self.total_routes


class ABTestRouter:
    """
    A/B testing router for routing strategies.

    Splits traffic between routing variants, tracks metrics,
    and determines statistical significance.

    Example:
        >>> config = ABTestConfig(
        ...     test_name="rule_vs_ml",
        ...     variants={
        ...         RoutingVariant.CONTROL: 0.5,    # 50% rule-based
        ...         RoutingVariant.VARIANT_A: 0.5   # 50% ML-based
        ...     },
        ...     min_sample_size=100,
        ...     auto_promote_winner=True
        ... )
        >>> ab_router = ABTestRouter(config)
        >>>
        >>> # Route query
        >>> variant = ab_router.assign_variant(user_id="user123")
        >>> # ... route using variant strategy ...
        >>> ab_router.record_outcome(variant, confidence=0.92, latency_ms=150)
        >>>
        >>> # Check results
        >>> results = ab_router.get_results()
        >>> print(f"Winner: {results['winner']}")
        >>> print(f"Confidence: {results['confidence']:.1%}")
    """

    def __init__(self, config: ABTestConfig):
        """
        Initialize A/B test router.

        Args:
            config: A/B test configuration
        """
        self.config = config

        # Validate traffic splits sum to 1.0
        total_traffic = sum(config.variants.values())
        if abs(total_traffic - 1.0) > 0.01:
            raise ValueError(f"Traffic splits must sum to 1.0, got {total_traffic}")

        # Metrics by variant
        self._metrics: Dict[RoutingVariant, VariantMetrics] = {
            variant: VariantMetrics(variant=variant)
            for variant in config.variants.keys()
        }

        # User assignments (sticky assignments for consistency)
        self._user_assignments: Dict[str, RoutingVariant] = {}

    def assign_variant(self, user_id: Optional[str] = None) -> RoutingVariant:
        """
        Assign user to routing variant.

        Uses sticky assignment (same user always gets same variant).

        Args:
            user_id: Optional user ID for sticky assignment

        Returns:
            Assigned routing variant
        """
        # Check sticky assignment
        if user_id and user_id in self._user_assignments:
            return self._user_assignments[user_id]

        # Weighted random selection
        rand = random.random()
        cumulative = 0.0

        for variant, traffic in self.config.variants.items():
            cumulative += traffic
            if rand <= cumulative:
                # Store sticky assignment
                if user_id:
                    self._user_assignments[user_id] = variant
                return variant

        # Fallback to control (shouldn't reach here)
        return self.config.control_variant

    def record_outcome(
        self,
        variant: RoutingVariant,
        confidence: float,
        latency_ms: float,
        success: bool = True
    ):
        """
        Record routing outcome for variant.

        Args:
            variant: Routing variant used
            confidence: Routing confidence (0.0-1.0)
            latency_ms: Routing latency in milliseconds
            success: Whether routing was successful
        """
        metrics = self._metrics[variant]

        metrics.total_routes += 1
        if success:
            metrics.successful_routes += 1

        # Update running averages
        n = metrics.total_routes
        metrics.total_confidence += confidence
        metrics.avg_latency_ms = (
            metrics.avg_latency_ms * (n - 1) + latency_ms
        ) / n

    def get_results(self) -> Dict[str, Any]:
        """
        Get A/B test results with statistical analysis.

        Returns:
            Dict with test results:
            - winner: Winning variant (if significant)
            - confidence: Statistical confidence in winner
            - metrics: Metrics by variant
            - ready_for_decision: Whether enough data collected
        """
        # Check if enough data
        total_samples = sum(m.total_routes for m in self._metrics.values())
        ready = total_samples >= self.config.min_sample_size

        # Find best variant
        best_variant = max(
            self._metrics.values(),
            key=lambda m: m.avg_confidence
        )

        # Compare to control
        control_metrics = self._metrics[self.config.control_variant]

        # Simple statistical test (z-test approximation)
        # In production, use proper statistical libraries
        confidence = self._calculate_confidence(
            best_variant,
            control_metrics
        )

        winner = None
        if ready and confidence >= self.config.confidence_threshold:
            winner = best_variant.variant

        return {
            "test_name": self.config.test_name,
            "total_samples": total_samples,
            "ready_for_decision": ready,
            "winner": winner,
            "confidence": confidence,
            "metrics": {
                variant: {
                    "success_rate": metrics.success_rate,
                    "avg_confidence": metrics.avg_confidence,
                    "avg_latency_ms": metrics.avg_latency_ms,
                    "sample_size": metrics.total_routes
                }
                for variant, metrics in self._metrics.items()
            }
        }

    def _calculate_confidence(
        self,
        variant_a: VariantMetrics,
        variant_b: VariantMetrics
    ) -> float:
        """
        Calculate statistical confidence that variant A is better than B.

        Uses simplified z-test. In production, use scipy.stats.

        Args:
            variant_a: First variant metrics
            variant_b: Second variant metrics

        Returns:
            Confidence level (0.0-1.0)
        """
        # Need minimum samples
        if variant_a.total_routes < 30 or variant_b.total_routes < 30:
            return 0.0

        # Calculate difference in means
        mean_diff = variant_a.avg_confidence - variant_b.avg_confidence

        # If no difference, 50% confidence
        if abs(mean_diff) < 0.01:
            return 0.5

        # Estimate standard error (simplified)
        # In production, calculate proper pooled standard error
        se_a = 0.1 / (variant_a.total_routes ** 0.5)
        se_b = 0.1 / (variant_b.total_routes ** 0.5)
        se_diff = (se_a ** 2 + se_b ** 2) ** 0.5

        # Z-score
        z = abs(mean_diff) / se_diff if se_diff > 0 else 0

        # Convert to confidence (approximate)
        # z=1.96 → 95%, z=2.58 → 99%
        if z >= 2.58:
            return 0.99
        elif z >= 1.96:
            return 0.95
        elif z >= 1.64:
            return 0.90
        else:
            return 0.5 + (z / 1.96) * 0.45  # Linear interpolation

    def promote_winner(self) -> Optional[RoutingVariant]:
        """
        Promote winning variant if test is conclusive.

        Only promotes if:
        - Enough samples collected
        - Statistical significance achieved
        - auto_promote_winner enabled

        Returns:
            Winning variant if promoted, None otherwise
        """
        if not self.config.auto_promote_winner:
            return None

        results = self.get_results()

        if results["ready_for_decision"] and results["winner"]:
            winner = results["winner"]

            # Update config to send 100% traffic to winner
            self.config.variants = {winner: 1.0}
            self.config.control_variant = winner

            return winner

        return None

    def reset(self):
        """Reset A/B test (clear all data)"""
        self._metrics = {
            variant: VariantMetrics(variant=variant)
            for variant in self.config.variants.keys()
        }
        self._user_assignments.clear()

    def export_results(self) -> Dict[str, Any]:
        """
        Export full A/B test results for analysis.

        Returns:
            Complete results with all metrics and configuration
        """
        results = self.get_results()

        return {
            "config": {
                "test_name": self.config.test_name,
                "variants": {
                    variant.value: traffic
                    for variant, traffic in self.config.variants.items()
                },
                "control_variant": self.config.control_variant.value,
                "min_sample_size": self.config.min_sample_size,
                "confidence_threshold": self.config.confidence_threshold
            },
            "results": results,
            "raw_metrics": {
                variant.value: {
                    "total_routes": metrics.total_routes,
                    "successful_routes": metrics.successful_routes,
                    "success_rate": metrics.success_rate,
                    "avg_confidence": metrics.avg_confidence,
                    "avg_latency_ms": metrics.avg_latency_ms
                }
                for variant, metrics in self._metrics.items()
            },
            "user_assignments": len(self._user_assignments)
        }
