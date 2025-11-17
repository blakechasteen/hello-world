"""
Distributed Wool Storage Node
==============================

Distributed storage node that extends WoolStorage with cluster capabilities.

This module implements the core distributed node that:
- Uses consistent hashing for file placement
- Routes requests to appropriate nodes
- Handles local vs remote storage
- Integrates with gossip protocol for membership

Architecture:
    ┌─────────────────────────────────────┐
    │      DistributedWoolNode             │
    ├─────────────────────────────────────┤
    │  + store(data) → file_id             │
    │  + read(ref) → memoryview            │
    │  + get_responsible_nodes(file_id)    │
    │  + sync_file(file_id, nodes)         │
    └─────────────────────────────────────┘
              │         │         │
              ▼         ▼         ▼
    ┌─────────────┐ ┌──────────┐ ┌──────────────┐
    │ Local       │ │ Hash     │ │ Network      │
    │ WoolStorage │ │ Ring     │ │ Protocol     │
    └─────────────┘ └──────────┘ └──────────────┘

Author: Claude Code
Date: November 17, 2025 (Phase 6.2)
"""

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set, Dict, Any
from enum import Enum

from HoloLoom.wool import WoolStorage, WoolReference
from HoloLoom.wool.distributed.ring import ConsistentHashRing


logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Node health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"  # Partially unavailable
    UNREACHABLE = "unreachable"
    UNKNOWN = "unknown"


@dataclass
class NodeInfo:
    """Information about a cluster node."""
    node_id: str
    host: str
    port: int
    status: NodeStatus = NodeStatus.UNKNOWN
    last_seen: float = 0.0
    storage_used_bytes: int = 0
    storage_capacity_bytes: int = 0

    @property
    def storage_utilization(self) -> float:
        """Storage utilization (0.0-1.0)."""
        if self.storage_capacity_bytes == 0:
            return 0.0
        return self.storage_used_bytes / self.storage_capacity_bytes

    @property
    def address(self) -> str:
        """Network address."""
        return f"{self.host}:{self.port}"


@dataclass
class FileLocation:
    """Location information for a distributed file."""
    file_id: str
    primary_nodes: List[str]
    replica_nodes: List[str]
    local: bool  # Whether file is stored locally

    @property
    def all_nodes(self) -> List[str]:
        """All nodes (primary + replicas)."""
        return self.primary_nodes + self.replica_nodes

    @property
    def replication_factor(self) -> int:
        """Current replication factor."""
        return len(self.all_nodes)


class DistributedWoolNode:
    """
    Distributed wool storage node with cluster awareness.

    Features:
    - Consistent hashing for file placement
    - Local and remote file operations
    - Health tracking and monitoring
    - Replication support

    Usage:
        node = DistributedWoolNode(
            node_id="node-1",
            host="localhost",
            port=9000,
            peers=["localhost:9001", "localhost:9002"]
        )

        async with node:
            # Store file (automatically routed to correct nodes)
            file_id = await node.store(b"content")

            # Read file (local or remote)
            data = await node.read(WoolReference(file_id, 0, 7))

            # Check file location
            location = node.get_file_location(file_id)
            print(f"Stored on: {location.all_nodes}")
    """

    def __init__(
        self,
        node_id: str,
        host: str = "localhost",
        port: int = 9000,
        base_path: Optional[Path] = None,
        peers: Optional[List[str]] = None,
        num_replicas: int = 3,
        virtual_nodes: int = 150
    ):
        """
        Initialize distributed wool node.

        Args:
            node_id: Unique node identifier
            host: Node host address
            port: Node port
            base_path: Base path for local storage
            peers: List of peer addresses ["host:port", ...]
            num_replicas: Number of replicas to maintain
            virtual_nodes: Virtual nodes per physical node
        """
        self.node_id = node_id
        self.host = host
        self.port = port
        self.num_replicas = num_replicas

        # Local storage
        if base_path is None:
            base_path = Path(f"./data/wool/distributed/{node_id}")
        self.local_storage = WoolStorage(base_path=base_path)

        # Cluster state
        self.peers: Dict[str, NodeInfo] = {}
        self.hash_ring: Optional[ConsistentHashRing] = None

        # Initialize peers
        if peers:
            for peer_addr in peers:
                host, port_str = peer_addr.split(":")
                peer_id = f"{host}:{port_str}"  # Use address as ID for now
                self.peers[peer_id] = NodeInfo(
                    node_id=peer_id,
                    host=host,
                    port=int(port_str),
                    status=NodeStatus.UNKNOWN
                )

        # Add self to peers
        self.peers[node_id] = NodeInfo(
            node_id=node_id,
            host=host,
            port=port,
            status=NodeStatus.HEALTHY
        )

        # Build hash ring
        self._rebuild_hash_ring()

        # Network protocol (will be injected)
        self.network_protocol = None  # Set externally

        # Statistics
        self.stats = {
            'local_stores': 0,
            'remote_stores': 0,
            'local_reads': 0,
            'remote_reads': 0,
            'sync_operations': 0,
            'errors': 0
        }

        logger.info(f"Initialized distributed wool node: {node_id} at {host}:{port}")
        logger.info(f"Cluster size: {len(self.peers)} nodes, replicas: {num_replicas}")

    def _rebuild_hash_ring(self):
        """Rebuild consistent hash ring from current peer list."""
        healthy_nodes = [
            peer_id for peer_id, info in self.peers.items()
            if info.status in (NodeStatus.HEALTHY, NodeStatus.DEGRADED)
        ]

        if not healthy_nodes:
            healthy_nodes = [self.node_id]  # At minimum, include self

        self.hash_ring = ConsistentHashRing(
            nodes=healthy_nodes,
            virtual_nodes_per_physical=150,
            num_replicas=self.num_replicas
        )

        logger.debug(f"Rebuilt hash ring with {len(healthy_nodes)} nodes")

    def get_responsible_nodes(self, file_id: str) -> List[str]:
        """
        Get nodes responsible for storing a file.

        Args:
            file_id: File identifier (SHA-256 hash)

        Returns:
            List of node IDs responsible for this file
        """
        if self.hash_ring is None:
            return [self.node_id]

        return self.hash_ring.get_nodes_for_file(file_id)

    def get_file_location(self, file_id: str) -> FileLocation:
        """
        Get location information for a file.

        Args:
            file_id: File identifier

        Returns:
            FileLocation with node assignments
        """
        responsible_nodes = self.get_responsible_nodes(file_id)

        # First node is primary, rest are replicas
        primary = responsible_nodes[:1] if responsible_nodes else []
        replicas = responsible_nodes[1:] if len(responsible_nodes) > 1 else []

        # Check if file is stored locally
        local_path = self.local_storage._get_file_path(file_id)
        local = local_path.exists()

        return FileLocation(
            file_id=file_id,
            primary_nodes=primary,
            replica_nodes=replicas,
            local=local
        )

    def is_responsible_for(self, file_id: str) -> bool:
        """
        Check if this node is responsible for storing a file.

        Args:
            file_id: File identifier

        Returns:
            True if this node should store the file
        """
        responsible = self.get_responsible_nodes(file_id)
        return self.node_id in responsible

    async def store(
        self,
        data: bytes,
        content_type: str = "application/octet-stream",
        replicate: bool = True
    ) -> str:
        """
        Store data in distributed wool storage.

        Args:
            data: Data to store
            content_type: MIME type
            replicate: Whether to replicate to other nodes

        Returns:
            File identifier (SHA-256 hash)
        """
        # Compute file ID
        file_id = hashlib.sha256(data).hexdigest()

        # Get responsible nodes
        location = self.get_file_location(file_id)

        # Store locally if this node is responsible
        if self.is_responsible_for(file_id):
            self.local_storage.store(data, content_type)
            self.stats['local_stores'] += 1
            logger.debug(f"Stored {file_id[:12]}... locally")

        # Replicate to other nodes
        if replicate and len(location.all_nodes) > 1:
            await self._replicate_to_nodes(
                file_id=file_id,
                data=data,
                content_type=content_type,
                target_nodes=[n for n in location.all_nodes if n != self.node_id]
            )

        return file_id

    async def read(self, ref: WoolReference) -> memoryview:
        """
        Read data from distributed wool storage.

        Args:
            ref: Wool reference to read

        Returns:
            Memory view of data

        Raises:
            FileNotFoundError: If file not found on any node
        """
        location = self.get_file_location(ref.file_id)

        # Try local first
        if location.local:
            self.stats['local_reads'] += 1
            return self.local_storage.read(ref)

        # Try remote nodes
        if self.network_protocol:
            for node_id in location.all_nodes:
                if node_id == self.node_id:
                    continue

                try:
                    data = await self.network_protocol.read_remote(
                        node_id=node_id,
                        ref=ref
                    )
                    self.stats['remote_reads'] += 1
                    logger.debug(f"Read {ref.file_id[:12]}... from {node_id}")
                    return memoryview(data)
                except Exception as e:
                    logger.warning(f"Failed to read from {node_id}: {e}")
                    continue

        # Not found anywhere
        self.stats['errors'] += 1
        raise FileNotFoundError(f"File {ref.file_id} not found on any node")

    async def _replicate_to_nodes(
        self,
        file_id: str,
        data: bytes,
        content_type: str,
        target_nodes: List[str]
    ):
        """
        Replicate file to target nodes.

        Args:
            file_id: File identifier
            data: File data
            content_type: MIME type
            target_nodes: List of target node IDs
        """
        if not self.network_protocol or not target_nodes:
            return

        # Send to all targets concurrently
        tasks = []
        for node_id in target_nodes:
            task = self.network_protocol.replicate_to_node(
                node_id=node_id,
                file_id=file_id,
                data=data,
                content_type=content_type
            )
            tasks.append(task)

        # Wait for all replications
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log results
        success = sum(1 for r in results if not isinstance(r, Exception))
        self.stats['remote_stores'] += success

        if success < len(target_nodes):
            logger.warning(
                f"Replicated {file_id[:12]}... to {success}/{len(target_nodes)} nodes"
            )
        else:
            logger.debug(f"Replicated {file_id[:12]}... to {success} nodes")

    async def sync_file(self, file_id: str) -> bool:
        """
        Synchronize a file across all responsible nodes.

        Args:
            file_id: File to synchronize

        Returns:
            True if sync successful
        """
        location = self.get_file_location(file_id)

        # Read file from any available source
        ref = WoolReference(file_id=file_id, offset=0, length=0)  # Full file

        try:
            # Try local first
            if location.local:
                data = bytes(self.local_storage.read(ref))
            else:
                # Try remote nodes
                data = None
                for node_id in location.all_nodes:
                    if node_id == self.node_id:
                        continue
                    try:
                        data = await self.network_protocol.read_remote(node_id, ref)
                        break
                    except:
                        continue

                if data is None:
                    logger.error(f"Cannot sync {file_id[:12]}...: not found anywhere")
                    return False

            # Replicate to all responsible nodes
            await self._replicate_to_nodes(
                file_id=file_id,
                data=data,
                content_type="application/octet-stream",
                target_nodes=[n for n in location.all_nodes if n != self.node_id]
            )

            self.stats['sync_operations'] += 1
            logger.info(f"Synced {file_id[:12]}... across {len(location.all_nodes)} nodes")
            return True

        except Exception as e:
            logger.error(f"Sync failed for {file_id[:12]}...: {e}")
            self.stats['errors'] += 1
            return False

    def add_peer(self, node_id: str, host: str, port: int):
        """
        Add a new peer to the cluster.

        Args:
            node_id: Node identifier
            host: Node host
            port: Node port
        """
        self.peers[node_id] = NodeInfo(
            node_id=node_id,
            host=host,
            port=port,
            status=NodeStatus.UNKNOWN
        )
        self._rebuild_hash_ring()
        logger.info(f"Added peer: {node_id} at {host}:{port}")

    def remove_peer(self, node_id: str):
        """
        Remove a peer from the cluster.

        Args:
            node_id: Node identifier
        """
        if node_id in self.peers:
            del self.peers[node_id]
            self._rebuild_hash_ring()
            logger.info(f"Removed peer: {node_id}")

    def update_peer_status(self, node_id: str, status: NodeStatus):
        """
        Update peer health status.

        Args:
            node_id: Node identifier
            status: New status
        """
        if node_id in self.peers:
            old_status = self.peers[node_id].status
            self.peers[node_id].status = status

            if old_status != status:
                logger.info(f"Peer {node_id} status: {old_status.value} → {status.value}")
                self._rebuild_hash_ring()  # Rebuild if health changed

    def get_stats(self) -> Dict[str, Any]:
        """Get node statistics."""
        return {
            'node_id': self.node_id,
            'address': f"{self.host}:{self.port}",
            'cluster_size': len(self.peers),
            'healthy_nodes': sum(
                1 for p in self.peers.values()
                if p.status == NodeStatus.HEALTHY
            ),
            'local_storage': {
                'files': len(list(self.local_storage.base_path.rglob("*"))),
                'stats': self.local_storage.get_stats()
            },
            'operations': self.stats.copy()
        }

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.close()

    def close(self):
        """Close node and cleanup resources."""
        self.local_storage.close()
        logger.info(f"Closed distributed wool node: {self.node_id}")
