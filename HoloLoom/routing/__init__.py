"""
HoloLoom Routing Module

Intelligent routing system with two approaches:

1. **Learned Routing** (Thompson Sampling):
   - Learns optimal backend selection from performance data
   - Multi-armed bandit for exploration/exploitation

2. **Gradient Flow Routing** (Physics-Based):
   - Routes queries by following gradients downhill
   - Natural load balancing, no learning required
   - Datacenter downhill flow metaphor
"""

# Learned routing (existing)
from .learned import ThompsonBandit, LearnedRouter
from .metrics import RoutingMetrics, MetricsCollector
from .ab_test import ABTestRouter, StrategyVariant

# Gradient flow routing (Phase 1)
from .flow_router import (
    FlowRouter,
    ServerRouter,
    ToolRouter,
    ServerConfig,
    ToolConfig,
    create_flow_router,
    create_server_router,
    create_tool_router
)

__all__ = [
    # Learned routing
    'ThompsonBandit',
    'LearnedRouter',
    'RoutingMetrics',
    'MetricsCollector',
    'ABTestRouter',
    'StrategyVariant',

    # Gradient flow routing
    'FlowRouter',
    'ServerRouter',
    'ToolRouter',
    'ServerConfig',
    'ToolConfig',
    'create_flow_router',
    'create_server_router',
    'create_tool_router',
]
