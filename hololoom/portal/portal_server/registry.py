"""
Node Registry - In-memory storage for registered nodes.

Thread-safe registry with heartbeat tracking and status management.
"""

import asyncio
from datetime import datetime, timedelta

from ..shared.types import LoomStatus, NodeCapabilities, NodeRecord, NodeStatus


class NodeRegistry:
    """
    In-memory registry of compute nodes.

    Thread-safe via asyncio locks. Nodes are marked offline
    if they don't send heartbeats within the timeout period.
    """

    def __init__(self, loom_id: str, node_timeout_seconds: int = 90):
        self.loom_id = loom_id
        self.node_timeout_seconds = node_timeout_seconds
        self._nodes: dict[str, NodeRecord] = {}
        self._lock = asyncio.Lock()
        self._start_time = datetime.utcnow()

    async def register(
        self,
        node_id: str,
        address: str,
        capabilities: NodeCapabilities
    ) -> NodeRecord:
        """
        Register or update a node.

        Args:
            node_id: Unique node identifier
            address: Node HTTP address (host:port)
            capabilities: Node hardware capabilities

        Returns:
            Updated NodeRecord
        """
        async with self._lock:
            if node_id in self._nodes:
                # Update existing node
                node = self._nodes[node_id]
                node.address = address
                node.capabilities = capabilities
                node.last_heartbeat = datetime.utcnow()
                node.status = NodeStatus.ONLINE
            else:
                # Register new node
                node = NodeRecord(
                    node_id=node_id,
                    address=address,
                    capabilities=capabilities,
                    status=NodeStatus.ONLINE,
                    last_heartbeat=datetime.utcnow(),
                )
                self._nodes[node_id] = node

            return node

    async def heartbeat(self, node_id: str) -> NodeRecord | None:
        """
        Update node heartbeat timestamp.

        Args:
            node_id: Node identifier

        Returns:
            Updated NodeRecord or None if not found
        """
        async with self._lock:
            if node_id in self._nodes:
                node = self._nodes[node_id]
                node.last_heartbeat = datetime.utcnow()
                node.status = NodeStatus.ONLINE
                return node
            return None

    async def get(self, node_id: str) -> NodeRecord | None:
        """Get a node by ID."""
        async with self._lock:
            return self._nodes.get(node_id)

    async def list_all(self) -> list[NodeRecord]:
        """List all registered nodes with updated status."""
        await self._update_status()
        async with self._lock:
            return list(self._nodes.values())

    async def list_online(self) -> list[NodeRecord]:
        """List only online nodes."""
        await self._update_status()
        async with self._lock:
            return [n for n in self._nodes.values() if n.status == NodeStatus.ONLINE]

    async def unregister(self, node_id: str) -> bool:
        """
        Remove a node from the registry.

        Returns:
            True if node was removed, False if not found
        """
        async with self._lock:
            if node_id in self._nodes:
                del self._nodes[node_id]
                return True
            return False

    async def get_status(self) -> LoomStatus:
        """Get overall Loom status."""
        await self._update_status()

        async with self._lock:
            nodes = list(self._nodes.values())
            online = [n for n in nodes if n.status == NodeStatus.ONLINE]
            offline = [n for n in nodes if n.status == NodeStatus.OFFLINE]

            return LoomStatus(
                loom_id=self.loom_id,
                nodes_online=len(online),
                nodes_offline=len(offline),
                total_nodes=len(nodes),
                total_cpu_cores=sum(n.capabilities.cpu_cores for n in online),
                total_memory_mb=sum(n.capabilities.memory_mb for n in online),
                jobs_in_progress=sum(n.current_jobs for n in online),
                uptime_seconds=(datetime.utcnow() - self._start_time).total_seconds(),
            )

    async def _update_status(self) -> None:
        """Update status of nodes based on heartbeat timeout."""
        cutoff = datetime.utcnow() - timedelta(seconds=self.node_timeout_seconds)

        async with self._lock:
            for node in self._nodes.values():
                if node.last_heartbeat < cutoff:
                    node.status = NodeStatus.OFFLINE
                elif node.status == NodeStatus.OFFLINE:
                    # Node came back online
                    node.status = NodeStatus.ONLINE
