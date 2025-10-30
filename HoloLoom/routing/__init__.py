"""
HoloLoom Routing Module

Intelligent routing system that learns optimal backend selection from performance data.
Uses Thompson Sampling (multi-armed bandit) for exploration/exploitation balance.
"""

from .learned import ThompsonBandit, LearnedRouter
from .metrics import RoutingMetrics, MetricsCollector
from .ab_test import ABTestRouter, StrategyVariant

__all__ = [
    'ThompsonBandit',
    'LearnedRouter',
    'RoutingMetrics',
    'MetricsCollector',
    'ABTestRouter',
    'StrategyVariant'
]
