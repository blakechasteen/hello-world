"""
Flow Router - Query Routing Using Gradient Flow

High-level API for routing queries using gradient descent.

Applications:
    - Load balancing across servers
    - Tool selection for queries
    - Resource allocation
    - Multi-agent coordination

Usage:
    router = create_flow_router(
        servers=['server1', 'server2', 'server3'],
        loss_fn='combined'
    )

    target = await router.route(
        query="What is Thompson Sampling?",
        current_loads={'server1': 0.9, 'server2': 0.3, 'server3': 0.6}
    )
"""

from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
import asyncio

from ..physics.gradient_flow import (
    GradientFlowEngine,
    RouteDecision,
    combined_loss,
    create_tool_selection_loss,
    load_loss,
    latency_loss,
    cost_loss
)


@dataclass
class ServerConfig:
    """Configuration for a server."""
    name: str
    max_load: float = 1.0
    base_latency: float = 50.0  # ms
    cost_per_query: float = 0.01


@dataclass
class ToolConfig:
    """Configuration for a tool."""
    name: str
    cost: float
    quality: float
    latency: float  # ms


class FlowRouter:
    """
    Query router using gradient flow.

    Routes queries to optimal targets by following gradients downhill.

    Usage:
        router = FlowRouter(
            targets=['server1', 'server2', 'server3'],
            loss_fn=combined_loss()
        )

        decision = await router.route(
            current_metrics={
                'server1': {'load': 0.9, 'latency': 50},
                'server2': {'load': 0.3, 'latency': 30},
                'server3': {'load': 0.6, 'latency': 40}
            }
        )

        print(f"Route to: {decision.target}")
    """

    def __init__(
        self,
        targets: List[str],
        loss_fn: Callable[[Dict[str, float]], float],
        learning_rate: float = 0.1,
        noise_level: float = 0.05,
        max_steps: int = 10
    ):
        """
        Initialize flow router.

        Args:
            targets: List of target names (servers, tools, etc.)
            loss_fn: Loss function for routing decisions
            learning_rate: Gradient descent learning rate
            noise_level: Exploration noise amplitude
            max_steps: Max gradient descent iterations
        """
        self.targets = targets
        self.engine = GradientFlowEngine(
            loss_fn=loss_fn,
            learning_rate=learning_rate,
            noise_level=noise_level
        )
        self.max_steps = max_steps
        self.route_history: List[RouteDecision] = []

    async def route(
        self,
        current_metrics: Dict[str, Dict[str, float]],
        query: Optional[str] = None
    ) -> RouteDecision:
        """
        Route query to optimal target.

        Args:
            current_metrics: Dict mapping target name -> metrics
            query: Optional query text (for logging)

        Returns:
            RouteDecision with selected target
        """
        # Extract metrics in target order
        target_metrics = [
            current_metrics[target]
            for target in self.targets
        ]

        # Route using gradient flow
        decision = self.engine.route(
            targets=self.targets,
            target_metrics=target_metrics,
            max_steps=self.max_steps
        )

        # Store in history
        self.route_history.append(decision)

        return decision

    def get_statistics(self) -> Dict[str, Any]:
        """Get routing statistics."""
        if not self.route_history:
            return {
                "total_routes": 0,
                "targets": {},
                "avg_loss": 0.0
            }

        # Count routes per target
        target_counts = {}
        for decision in self.route_history:
            target_counts[decision.target] = target_counts.get(decision.target, 0) + 1

        # Average loss
        avg_loss = sum(d.loss for d in self.route_history) / len(self.route_history)

        return {
            "total_routes": len(self.route_history),
            "targets": target_counts,
            "avg_loss": avg_loss
        }


class ServerRouter(FlowRouter):
    """
    Router for load balancing across servers.

    Automatically manages server configurations and load tracking.

    Usage:
        router = ServerRouter(
            servers=[
                ServerConfig("server1", max_load=1.0),
                ServerConfig("server2", max_load=1.0),
                ServerConfig("server3", max_load=1.0)
            ]
        )

        # Route query
        decision = await router.route_query("What is Thompson Sampling?")

        # Router automatically tracks load
        print(f"Current loads: {router.get_loads()}")
    """

    def __init__(
        self,
        servers: List[ServerConfig],
        load_weight: float = 0.5,
        latency_weight: float = 0.3,
        cost_weight: float = 0.2
    ):
        """
        Initialize server router.

        Args:
            servers: List of server configurations
            load_weight: Weight for load in loss function
            latency_weight: Weight for latency
            cost_weight: Weight for cost
        """
        self.servers = {s.name: s for s in servers}
        self.current_loads = {s.name: 0.0 for s in servers}

        # Create combined loss function
        loss_fn = combined_loss(load_weight, latency_weight, cost_weight)

        super().__init__(
            targets=[s.name for s in servers],
            loss_fn=loss_fn
        )

    async def route_query(self, query: str) -> RouteDecision:
        """
        Route query to least-loaded server.

        Automatically updates load tracking.

        Args:
            query: Query text

        Returns:
            RouteDecision with selected server
        """
        # Build current metrics
        metrics = {}
        for name, server in self.servers.items():
            metrics[name] = {
                "load": self.current_loads[name],
                "latency": server.base_latency,
                "cost": server.cost_per_query
            }

        # Route
        decision = await self.route(metrics, query)

        # Update load (simplified - just increment)
        self.current_loads[decision.target] += 0.1
        self.current_loads[decision.target] = min(
            self.current_loads[decision.target],
            self.servers[decision.target].max_load
        )

        return decision

    def get_loads(self) -> Dict[str, float]:
        """Get current server loads."""
        return dict(self.current_loads)

    def reset_loads(self):
        """Reset all server loads to 0."""
        self.current_loads = {name: 0.0 for name in self.servers.keys()}


class ToolRouter(FlowRouter):
    """
    Router for tool selection.

    Balances cost, quality, and latency.

    Usage:
        router = ToolRouter(
            tools=[
                ToolConfig("search", cost=0.5, quality=0.7, latency=100),
                ToolConfig("answer", cost=0.2, quality=0.6, latency=50),
                ToolConfig("reason", cost=0.8, quality=0.95, latency=200)
            ]
        )

        decision = await router.select_tool("Complex reasoning task")
        print(f"Use tool: {decision.target}")
    """

    def __init__(
        self,
        tools: List[ToolConfig],
        cost_weight: float = 0.3,
        quality_weight: float = 0.5,
        latency_weight: float = 0.2
    ):
        """
        Initialize tool router.

        Args:
            tools: List of tool configurations
            cost_weight: Weight for cost
            quality_weight: Weight for quality (inverted)
            latency_weight: Weight for latency
        """
        self.tools = {t.name: t for t in tools}

        # Create tool selection loss
        loss_fn = create_tool_selection_loss(
            cost_weight, quality_weight, latency_weight
        )

        super().__init__(
            targets=[t.name for t in tools],
            loss_fn=loss_fn
        )

    async def select_tool(self, query: str) -> RouteDecision:
        """
        Select optimal tool for query.

        Args:
            query: Query or task description

        Returns:
            RouteDecision with selected tool
        """
        # Build metrics from tool configs
        metrics = {}
        for name, tool in self.tools.items():
            metrics[name] = {
                "cost": tool.cost,
                "quality": tool.quality,
                "latency": tool.latency
            }

        return await self.route(metrics, query)


# Factory functions

def create_flow_router(
    servers: Optional[List[str]] = None,
    tools: Optional[List[str]] = None,
    loss_fn: str = "combined",
    **kwargs
) -> FlowRouter:
    """
    Create a flow router.

    Args:
        servers: List of server names (for load balancing)
        tools: List of tool names (for tool selection)
        loss_fn: Loss function type ("combined", "load", "latency", "cost")
        **kwargs: Additional arguments for router

    Returns:
        Configured FlowRouter
    """
    if servers:
        targets = servers
    elif tools:
        targets = tools
    else:
        raise ValueError("Must provide either servers or tools")

    # Select loss function
    if loss_fn == "combined":
        loss = combined_loss()
    elif loss_fn == "load":
        loss = load_loss
    elif loss_fn == "latency":
        loss = latency_loss
    elif loss_fn == "cost":
        loss = cost_loss
    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")

    return FlowRouter(targets=targets, loss_fn=loss, **kwargs)


def create_server_router(
    server_names: List[str],
    max_loads: Optional[List[float]] = None,
    **kwargs
) -> ServerRouter:
    """
    Create a server router for load balancing.

    Args:
        server_names: List of server names
        max_loads: Optional max loads for each server
        **kwargs: Additional router arguments

    Returns:
        Configured ServerRouter
    """
    if max_loads is None:
        max_loads = [1.0] * len(server_names)

    servers = [
        ServerConfig(name=name, max_load=load)
        for name, load in zip(server_names, max_loads)
    ]

    return ServerRouter(servers=servers, **kwargs)


def create_tool_router(
    tool_configs: List[Dict[str, float]],
    **kwargs
) -> ToolRouter:
    """
    Create a tool router.

    Args:
        tool_configs: List of tool config dicts with keys:
                      name, cost, quality, latency
        **kwargs: Additional router arguments

    Returns:
        Configured ToolRouter
    """
    tools = [
        ToolConfig(
            name=cfg["name"],
            cost=cfg.get("cost", 0.5),
            quality=cfg.get("quality", 0.7),
            latency=cfg.get("latency", 100.0)
        )
        for cfg in tool_configs
    ]

    return ToolRouter(tools=tools, **kwargs)
