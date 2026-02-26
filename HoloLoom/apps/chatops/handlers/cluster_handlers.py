"""
Cluster Handlers for Matrix ChatOps
====================================

Eggroll distributed cluster management commands.

Commands:
- !cluster status - Show cluster health overview
- !cluster nodes - List connected nodes with details
- !cluster balance - Show load distribution
- !cluster help - Show cluster command help

Features:
- Multi-backend support (Local, Ray)
- Health monitoring with auto-respawn
- Load balancing visualization
- Worker metrics tracking

Usage:
    from HoloLoom.apps.chatops.handlers.cluster_handlers import (
        register_cluster_handlers,
        ClusterHandlers
    )

    # In run_chatops.py:
    register_cluster_handlers(bot, backend)

Created: 2025-12-09
"""

import logging
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# Matrix Dependencies (optional)
# ============================================================================

try:
    from nio import MatrixRoom, RoomMessageText
    MATRIX_AVAILABLE = True
except ImportError:
    MATRIX_AVAILABLE = False
    MatrixRoom = Any
    RoomMessageText = Any


# ============================================================================
# Handler Registry (optional)
# ============================================================================

try:
    from HoloLoom.apps.chatops.handlers.handler_registry import (
        HandlerRegistry, HandlerCategory, chatops_handler
    )
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False
    HandlerRegistry = None
    HandlerCategory = None


# ============================================================================
# Eggroll Dependencies (optional)
# ============================================================================

try:
    from HoloLoom.eggroll.distributed_backend import (
        DistributedBackend,
        LocalBackend,
        RayBackend
    )
    EGGROLL_AVAILABLE = True
except ImportError:
    EGGROLL_AVAILABLE = False
    DistributedBackend = None
    LocalBackend = None
    RayBackend = None

try:
    from HoloLoom.eggroll import EggrollIntegration
    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False
    EggrollIntegration = None


# ============================================================================
# Node Status Types
# ============================================================================

class NodeStatus(Enum):
    """Status of a cluster node."""
    HEALTHY = "healthy"
    DEAD = "dead"
    UNRESPONSIVE = "unresponsive"
    INITIALIZING = "initializing"
    UNKNOWN = "unknown"


@dataclass
class NodeInfo:
    """Information about a cluster node."""
    node_id: int
    status: NodeStatus
    model_type: str = "standard"
    last_health_check: Optional[datetime] = None
    tasks_completed: int = 0
    current_load: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClusterInfo:
    """Information about the entire cluster."""
    backend_type: str
    num_workers: int
    nodes: List[NodeInfo]
    total_tasks_completed: int = 0
    avg_load: float = 0.0
    started_at: Optional[datetime] = None


# ============================================================================
# Cluster Manager
# ============================================================================

class ClusterManager:
    """
    Manages cluster state and provides status information.

    Works with LocalBackend and RayBackend.
    """

    def __init__(self, backend: Optional['DistributedBackend'] = None):
        """
        Initialize cluster manager.

        Args:
            backend: DistributedBackend instance (LocalBackend or RayBackend)
        """
        self.backend = backend
        self._node_info: Dict[int, NodeInfo] = {}
        self._task_counts: Dict[int, int] = {}
        self._started_at: Optional[datetime] = None

        if backend is not None:
            self._started_at = datetime.now()

    def set_backend(self, backend: 'DistributedBackend') -> None:
        """Set or update the backend."""
        self.backend = backend
        self._started_at = datetime.now()

    def get_backend_type(self) -> str:
        """Get the type of backend."""
        if self.backend is None:
            return "none"
        elif hasattr(self.backend, 'ray'):
            return "ray"
        else:
            return "local"

    def get_cluster_info(self) -> ClusterInfo:
        """Get comprehensive cluster information."""
        if not self.backend:
            return ClusterInfo(
                backend_type="none",
                num_workers=0,
                nodes=[]
            )

        # Get node health
        health_status = self._get_health_status()

        # Build node info list
        nodes = []
        for node_id, status_str in health_status.items():
            status = NodeStatus(status_str) if status_str in [s.value for s in NodeStatus] else NodeStatus.UNKNOWN

            # Get cached info or create new
            if node_id in self._node_info:
                node = self._node_info[node_id]
                node.status = status
                node.last_health_check = datetime.now()
            else:
                node = NodeInfo(
                    node_id=node_id,
                    status=status,
                    last_health_check=datetime.now()
                )
                self._node_info[node_id] = node

            nodes.append(node)

        # Calculate aggregate metrics
        total_tasks = sum(self._task_counts.values())
        avg_load = sum(n.current_load for n in nodes) / len(nodes) if nodes else 0.0

        return ClusterInfo(
            backend_type=self.get_backend_type(),
            num_workers=len(nodes),
            nodes=nodes,
            total_tasks_completed=total_tasks,
            avg_load=avg_load,
            started_at=self._started_at
        )

    def _get_health_status(self) -> Dict[int, str]:
        """Get health status from backend."""
        if not self.backend:
            return {}

        try:
            return self.backend.check_worker_health()
        except Exception as e:
            logger.error(f"Failed to check health: {e}")
            return {}

    def record_task_completion(self, node_id: int) -> None:
        """Record that a node completed a task."""
        self._task_counts[node_id] = self._task_counts.get(node_id, 0) + 1
        if node_id in self._node_info:
            self._node_info[node_id].tasks_completed = self._task_counts[node_id]


# Global cluster manager
_cluster_manager: Optional[ClusterManager] = None


def get_cluster_manager() -> ClusterManager:
    """Get or create global cluster manager."""
    global _cluster_manager
    if _cluster_manager is None:
        _cluster_manager = ClusterManager()
    return _cluster_manager


def set_cluster_manager(backend: 'DistributedBackend') -> ClusterManager:
    """Set the cluster manager with a backend."""
    global _cluster_manager
    _cluster_manager = ClusterManager(backend)
    return _cluster_manager


# ============================================================================
# Command Handlers
# ============================================================================

async def handle_cluster_status(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str
) -> str:
    """
    Handle !cluster status command - Show cluster health overview.

    Shows:
        - Backend type (local/ray)
        - Total workers and healthy count
        - Uptime
        - Total tasks completed
        - Average load

    Args:
        room: Matrix room
        event: Message event
        args: Optional arguments

    Returns:
        Cluster status overview
    """
    manager = get_cluster_manager()
    cluster = manager.get_cluster_info()

    if cluster.backend_type == "none":
        return (
            "## Cluster Status\n\n"
            "**Status**: Not initialized\n\n"
            "No distributed backend is configured. "
            "Start a cluster with LocalBackend or RayBackend to enable distributed processing."
        )

    # Count healthy nodes
    healthy = sum(1 for n in cluster.nodes if n.status == NodeStatus.HEALTHY)
    dead = sum(1 for n in cluster.nodes if n.status == NodeStatus.DEAD)

    # Status indicator
    if healthy == cluster.num_workers:
        health_icon = "+"
        health_status = "All nodes healthy"
    elif healthy > 0:
        health_icon = "~"
        health_status = f"{healthy}/{cluster.num_workers} nodes healthy"
    else:
        health_icon = "-"
        health_status = "No healthy nodes"

    # Uptime
    uptime_str = "Unknown"
    if cluster.started_at:
        uptime = datetime.now() - cluster.started_at
        hours = int(uptime.total_seconds() // 3600)
        minutes = int((uptime.total_seconds() % 3600) // 60)
        uptime_str = f"{hours}h {minutes}m"

    response = f"## Cluster Status [{health_icon}]\n\n"
    response += f"**Backend**: {cluster.backend_type.upper()}\n"
    response += f"**Status**: {health_status}\n"
    response += f"**Workers**: {cluster.num_workers}\n"
    response += f"**Healthy**: {healthy} / {cluster.num_workers}\n"

    if dead > 0:
        response += f"**Dead**: {dead} (auto-respawning)\n"

    response += f"**Uptime**: {uptime_str}\n"
    response += f"**Total Tasks**: {cluster.total_tasks_completed}\n"

    if cluster.avg_load > 0:
        response += f"**Avg Load**: {cluster.avg_load:.1%}\n"

    return response


async def handle_cluster_nodes(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str
) -> str:
    """
    Handle !cluster nodes command - List connected nodes.

    Shows detailed information about each node:
        - Node ID and status
        - Model type
        - Tasks completed
        - Current load
        - Last health check

    Args:
        room: Matrix room
        event: Message event
        args: Optional arguments (--verbose for more details)

    Returns:
        Node listing
    """
    manager = get_cluster_manager()
    cluster = manager.get_cluster_info()

    if cluster.backend_type == "none":
        return (
            "## Cluster Nodes\n\n"
            "_No cluster configured. Start a backend to see nodes._"
        )

    if not cluster.nodes:
        return (
            "## Cluster Nodes\n\n"
            "_No nodes registered yet._"
        )

    verbose = '--verbose' in args if args else False

    response = f"## Cluster Nodes ({cluster.num_workers})\n\n"
    response += f"Backend: {cluster.backend_type.upper()}\n\n"

    # Status icons
    status_icons = {
        NodeStatus.HEALTHY: "+",
        NodeStatus.DEAD: "x",
        NodeStatus.UNRESPONSIVE: "?",
        NodeStatus.INITIALIZING: "...",
        NodeStatus.UNKNOWN: "?"
    }

    for node in sorted(cluster.nodes, key=lambda n: n.node_id):
        icon = status_icons.get(node.status, "?")
        status_str = node.status.value

        response += f"**Node {node.node_id}** [{icon}]\n"
        response += f"  Status: {status_str}\n"
        response += f"  Model: {node.model_type}\n"
        response += f"  Tasks: {node.tasks_completed}\n"

        if verbose:
            if node.current_load > 0:
                response += f"  Load: {node.current_load:.1%}\n"

            if node.last_health_check:
                ago = datetime.now() - node.last_health_check
                response += f"  Last Check: {int(ago.total_seconds())}s ago\n"

            if node.metrics:
                response += f"  Metrics:\n"
                for key, value in node.metrics.items():
                    if isinstance(value, float):
                        response += f"    {key}: {value:.4f}\n"
                    else:
                        response += f"    {key}: {value}\n"

        response += "\n"

    if not verbose:
        response += "_Use `!cluster nodes --verbose` for more details_\n"

    return response


async def handle_cluster_balance(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str
) -> str:
    """
    Handle !cluster balance command - Show load distribution.

    Shows:
        - Task distribution across nodes
        - Load balance visualization
        - Recommendations if imbalanced

    Args:
        room: Matrix room
        event: Message event
        args: Optional arguments

    Returns:
        Load distribution visualization
    """
    manager = get_cluster_manager()
    cluster = manager.get_cluster_info()

    if cluster.backend_type == "none":
        return (
            "## Load Balance\n\n"
            "_No cluster configured._"
        )

    if not cluster.nodes:
        return (
            "## Load Balance\n\n"
            "_No nodes to balance._"
        )

    response = f"## Load Balance\n\n"
    response += f"Backend: {cluster.backend_type.upper()}\n"
    response += f"Workers: {cluster.num_workers}\n\n"

    # Calculate task distribution
    total_tasks = sum(n.tasks_completed for n in cluster.nodes)
    if total_tasks == 0:
        response += "_No tasks completed yet._\n"
        return response

    # Task distribution
    response += "### Task Distribution\n\n"

    max_tasks = max(n.tasks_completed for n in cluster.nodes) or 1
    bar_width = 20

    for node in sorted(cluster.nodes, key=lambda n: n.node_id):
        tasks = node.tasks_completed
        pct = (tasks / total_tasks * 100) if total_tasks > 0 else 0
        bar_fill = int((tasks / max_tasks) * bar_width)
        bar = "#" * bar_fill + "-" * (bar_width - bar_fill)

        status_icon = "+" if node.status == NodeStatus.HEALTHY else "x"

        response += f"Node {node.node_id} [{status_icon}]: [{bar}] {tasks} ({pct:.1f}%)\n"

    response += f"\n**Total Tasks**: {total_tasks}\n"

    # Balance analysis
    if cluster.nodes:
        avg_tasks = total_tasks / len(cluster.nodes)
        max_deviation = max(abs(n.tasks_completed - avg_tasks) for n in cluster.nodes)
        balance_score = 1 - (max_deviation / max_tasks) if max_tasks > 0 else 1.0

        response += f"**Balance Score**: {balance_score:.2f}\n"

        if balance_score < 0.7:
            response += "\n**Recommendation**: Load is imbalanced. Consider:\n"
            response += "- Checking node health\n"
            response += "- Reviewing task distribution strategy\n"
            response += "- Restarting underperforming nodes\n"
        elif balance_score >= 0.9:
            response += "\n*Load is well balanced across nodes.*\n"

    return response


async def handle_cluster_help(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str
) -> str:
    """Handle !cluster help command."""
    return """## Cluster Management Commands

**Commands:**
- `!cluster status` - Show cluster health overview
- `!cluster nodes` - List connected nodes
- `!cluster nodes --verbose` - Detailed node information
- `!cluster balance` - Show load distribution

**Backend Types:**
- `local` - Multiprocessing-based (development)
- `ray` - Ray-based distributed (production)

**Node Status:**
- `[+]` healthy - Node is operational
- `[x]` dead - Node failed (auto-respawning)
- `[?]` unresponsive - Node not responding
- `[...]` initializing - Node starting up

**Load Balancing:**
- Tasks are distributed across healthy nodes
- Dead nodes are automatically respawned
- Balance score indicates distribution quality

**Example Usage:**
```
!cluster status
!cluster nodes --verbose
!cluster balance
```

**Integration:**
```python
from HoloLoom.eggroll.distributed_backend import LocalBackend
from HoloLoom.apps.chatops.handlers.cluster_handlers import set_cluster_manager

# Initialize backend
backend = LocalBackend()
backend.initialize(num_workers=4, worker_cls=LoomNode)

# Register with cluster manager
set_cluster_manager(backend)
```
"""


# ============================================================================
# Registration
# ============================================================================

def register_cluster_handlers(
    bot,
    backend: Optional['DistributedBackend'] = None
) -> None:
    """
    Register all cluster command handlers.

    Args:
        bot: Matrix bot instance
        backend: Optional DistributedBackend instance
    """
    if not MATRIX_AVAILABLE:
        logger.warning("Matrix not available, cluster handlers not registered")
        return

    logger.info("Registering cluster command handlers")

    # Initialize cluster manager if backend provided
    if backend:
        set_cluster_manager(backend)

    # Define command map
    command_map = {
        "cluster status": handle_cluster_status,
        "cluster nodes": handle_cluster_nodes,
        "cluster balance": handle_cluster_balance,
        "cluster help": handle_cluster_help,
        "cluster": handle_cluster_help,  # Default to help
    }

    # Register with bot
    for cmd, handler in command_map.items():
        bot.register_command(cmd, handler)

    logger.info(f"Registered {len(command_map)} cluster commands")


# ============================================================================
# Decorator-Based Handler Class
# ============================================================================

class ClusterHandlers:
    """
    Decorator-based ChatOps handlers for cluster management.

    Usage:
        from HoloLoom.apps.chatops.handlers.cluster_handlers import ClusterHandlers

        handlers = ClusterHandlers(backend=distributed_backend)
        registry.register_instance(handlers)
    """

    def __init__(self, backend: Optional['DistributedBackend'] = None):
        """
        Initialize with optional DistributedBackend instance.

        Args:
            backend: DistributedBackend instance
        """
        if backend:
            set_cluster_manager(backend)

    if REGISTRY_AVAILABLE:
        @HandlerRegistry.register(
            command="cluster status",
            description="Show cluster health overview",
            usage="!cluster status",
            category=HandlerCategory.SYSTEM,
            aliases=["cstatus"]
        )
        async def handle_status(self, room, event, args: str) -> str:
            """Handle !cluster status command."""
            return await handle_cluster_status(room, event, args)

        @HandlerRegistry.register(
            command="cluster nodes",
            description="List connected cluster nodes",
            usage="!cluster nodes [--verbose]",
            category=HandlerCategory.SYSTEM,
            aliases=["cnodes"]
        )
        async def handle_nodes(self, room, event, args: str) -> str:
            """Handle !cluster nodes command."""
            return await handle_cluster_nodes(room, event, args)

        @HandlerRegistry.register(
            command="cluster balance",
            description="Show load distribution across nodes",
            usage="!cluster balance",
            category=HandlerCategory.SYSTEM,
            aliases=["cbalance"]
        )
        async def handle_balance(self, room, event, args: str) -> str:
            """Handle !cluster balance command."""
            return await handle_cluster_balance(room, event, args)

        @HandlerRegistry.register(
            command="cluster help",
            description="Show cluster command help",
            usage="!cluster help",
            category=HandlerCategory.SYSTEM,
            hidden=True
        )
        async def handle_help(self, room, event, args: str) -> str:
            """Handle !cluster help command."""
            return await handle_cluster_help(room, event, args)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("Cluster Handlers Demo")
    print("=" * 80)
    print()

    # Create mock cluster manager
    manager = ClusterManager()

    # Simulate nodes
    manager._node_info = {
        0: NodeInfo(node_id=0, status=NodeStatus.HEALTHY, tasks_completed=42, current_load=0.65),
        1: NodeInfo(node_id=1, status=NodeStatus.HEALTHY, tasks_completed=38, current_load=0.58),
        2: NodeInfo(node_id=2, status=NodeStatus.HEALTHY, tasks_completed=45, current_load=0.72),
        3: NodeInfo(node_id=3, status=NodeStatus.DEAD, tasks_completed=10, current_load=0.0),
    }
    manager._task_counts = {0: 42, 1: 38, 2: 45, 3: 10}
    manager._started_at = datetime.now()

    # Mock backend type
    class MockBackend:
        def check_worker_health(self):
            return {0: "healthy", 1: "healthy", 2: "healthy", 3: "dead"}

    manager.backend = MockBackend()

    print("Cluster Info:")
    cluster = manager.get_cluster_info()
    print(f"  Backend: {cluster.backend_type}")
    print(f"  Workers: {cluster.num_workers}")
    print(f"  Total Tasks: {cluster.total_tasks_completed}")
    print()

    print("Nodes:")
    for node in cluster.nodes:
        print(f"  Node {node.node_id}: {node.status.value} ({node.tasks_completed} tasks)")

    print("\n" + "=" * 80)
    print("Demo complete!")
