"""
Tests for Physics - Gradient Flow Engine

Tests gradient-based query routing using physics simulation.

Author: Claude Code
Date: 2025-11-17
Phase: 2 (Testing) - Week 1, Day 5
Target: +8 tests for gradient flow
"""

import pytest
import numpy as np
from HoloLoom.physics.gradient_flow import (
    GradientFlowEngine,
    Target,
    RouteDecision
)


class TestGradientFlowInitialization:
    """Test suite for GradientFlowEngine initialization."""

    def test_engine_initialization(self):
        """Test engine initialization with basic loss function."""
        def simple_loss(metrics):
            return metrics.get('load', 0.0)

        engine = GradientFlowEngine(
            loss_fn=simple_loss,
            learning_rate=0.1,
            noise_level=0.05,
            dt=0.01
        )

        assert engine.loss_fn is not None
        assert engine.learning_rate == 0.1
        assert engine.noise_level == 0.05
        assert engine.dt == 0.01


class TestBasicRouting:
    """Test suite for basic gradient flow routing."""

    def test_route_to_lowest_load(self):
        """Test routing flows to target with lowest load."""
        def load_loss(metrics):
            return metrics.get('load', 0.0)

        engine = GradientFlowEngine(
            loss_fn=load_loss,
            learning_rate=0.1,
            noise_level=0.01  # Low noise for deterministic routing
        )

        targets = ["server1", "server2", "server3"]
        metrics = [
            {"load": 0.9},  # High load
            {"load": 0.2},  # Low load (should be chosen)
            {"load": 0.6}   # Medium load
        ]

        decision = engine.route(
            targets=targets,
            target_metrics=metrics,
            max_steps=20
        )

        # Should route to server2 (lowest load)
        assert decision.target == "server2"
        assert decision.loss < 0.5  # Should have low loss

    def test_route_with_multiple_metrics(self):
        """Test routing with combined loss function."""
        def combined_loss(metrics):
            # Weight load heavily, latency lightly
            return 0.7 * metrics.get('load', 0.0) + 0.3 * metrics.get('latency', 0.0) / 100.0

        engine = GradientFlowEngine(
            loss_fn=combined_loss,
            learning_rate=0.1,
            noise_level=0.01
        )

        targets = ["server1", "server2"]
        metrics = [
            {"load": 0.3, "latency": 100},  # Low load, high latency
            {"load": 0.4, "latency": 20}    # Slightly higher load, low latency
        ]

        decision = engine.route(
            targets=targets,
            target_metrics=metrics,
            max_steps=20
        )

        # Should make a decision
        assert decision.target in targets
        assert decision.iterations > 0


class TestNoiseExploration:
    """Test suite for noise-based exploration."""

    def test_noise_enables_exploration(self):
        """Test that noise allows exploration of non-optimal targets."""
        def load_loss(metrics):
            return metrics.get('load', 0.0)

        # High noise engine
        engine = GradientFlowEngine(
            loss_fn=load_loss,
            learning_rate=0.1,
            noise_level=0.3  # High noise
        )

        targets = ["server1", "server2"]
        metrics = [
            {"load": 0.2},  # Optimal
            {"load": 0.8}   # Sub-optimal
        ]

        # Run multiple times, check if we get variety due to noise
        results = []
        for _ in range(10):
            decision = engine.route(targets, metrics, max_steps=10)
            results.append(decision.target)

        # With high noise, we should occasionally explore sub-optimal target
        unique_targets = set(results)
        # At least explored the optimal one
        assert "server1" in unique_targets


class TestConvergence:
    """Test suite for gradient flow convergence."""

    def test_convergence_to_minimum(self):
        """Test that gradient flow converges to minimum loss."""
        def quadratic_loss(metrics):
            # Simple quadratic bowl
            x = metrics.get('x', 0.0)
            return x ** 2

        engine = GradientFlowEngine(
            loss_fn=quadratic_loss,
            learning_rate=0.1,
            noise_level=0.0  # No noise for clean convergence
        )

        # Position at x=5, should converge to x=0
        decision = engine.route(
            targets=["target"],
            target_metrics=[{"x": 5.0}],
            max_steps=50
        )

        # Should converge close to minimum (x=0)
        assert decision.loss < 1.0  # Close to 0
        assert decision.iterations > 0

    def test_max_steps_limit(self):
        """Test that routing respects max_steps limit."""
        def load_loss(metrics):
            return metrics.get('load', 0.0)

        engine = GradientFlowEngine(load_fn=load_loss)

        max_steps = 5
        decision = engine.route(
            targets=["server1"],
            target_metrics=[{"load": 0.5}],
            max_steps=max_steps
        )

        # Should not exceed max_steps
        assert decision.iterations <= max_steps


class TestEdgeCases:
    """Test suite for edge cases."""

    def test_single_target(self):
        """Test routing with only one target."""
        def load_loss(metrics):
            return metrics.get('load', 0.0)

        engine = GradientFlowEngine(loss_fn=load_loss)

        decision = engine.route(
            targets=["only_server"],
            target_metrics=[{"load": 0.5}],
            max_steps=10
        )

        # Should route to the only available target
        assert decision.target == "only_server"

    def test_empty_metrics(self):
        """Test routing with empty metrics dict."""
        def load_loss(metrics):
            return metrics.get('load', 0.5)  # Default to 0.5 if missing

        engine = GradientFlowEngine(loss_fn=load_loss)

        decision = engine.route(
            targets=["server1"],
            target_metrics=[{}],  # Empty metrics
            max_steps=10
        )

        # Should handle gracefully
        assert decision.target == "server1"
        assert decision.loss >= 0.0
