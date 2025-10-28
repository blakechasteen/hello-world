"""
Memory Routing System - Learnable Backend Selection
===================================================

Modular routing strategies that can be A/B tested and optimized.

Architecture:
- RoutingStrategy protocol (define WHAT, not HOW)
- Multiple implementations (rule-based, ML-based, hybrid)
- A/B testing framework
- Learning/feedback loop
- Performance metrics

Usage:
    # Rule-based routing
    router = RuleBasedRouter()
    backend = router.select_backend(query, available_backends)

    # ML-based routing (learns from outcomes)
    router = LearnedRouter()
    backend = router.select_backend(query, available_backends)

    # A/B test two strategies
    experiment = ABTest(strategy_a=rule_based, strategy_b=learned)
    backend = experiment.select_backend(query, available_backends)
"""

from .protocol import (
    RoutingStrategy,
    RoutingDecision,
    RoutingOutcome,
    BackendType,
    QueryType
)
from .rule_based import RuleBasedRouter
from .learned import LearnedRouter
from .ab_test import ABTestRouter, RoutingExperiment

__all__ = [
    'RoutingStrategy',
    'RoutingDecision',
    'RoutingOutcome',
    'BackendType',
    'QueryType',
    'RuleBasedRouter',
    'LearnedRouter',
    'ABTestRouter',
    'RoutingExperiment',
]
