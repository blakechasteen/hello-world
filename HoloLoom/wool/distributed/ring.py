"""
Consistent Hash Ring
====================

Consistent hashing for distributed file placement.

Uses virtual nodes to ensure even distribution across cluster.

Algorithm:
    1. Hash file_id to ring position (0 - 2^160)
    2. Find first virtual node >= position (clockwise)
    3. Map virtual node → physical node
    4. Return physical node ID

Author: Claude Code
Date: November 17, 2025 (Phase 6)
"""

import hashlib
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from bisect import bisect_right


@dataclass
class VirtualNode:
    """Virtual node on the hash ring."""
    position: int  # Ring position (0 - 2^160)
    physical_node_id: str  # Physical node this virtual node belongs to
    replica_index: int = 0  # Which replica (0 = primary, 1+ = replicas)

    def __lt__(self, other):
        """Sort by position."""
        return self.position < other.position


class ConsistentHashRing:
    """
    Consistent hash ring for distributed file placement.

    Features:
    - Virtual nodes for even distribution
    - Replication support (primary + N replicas)
    - Add/remove nodes without full re-hash
    - Deterministic placement (same file_id → same nodes)

    Example:
        ring = ConsistentHashRing(
            nodes=['node1', 'node2', 'node3'],
            virtual_nodes_per_physical=150,
            num_replicas=3
        )

        # Find nodes for file
        nodes = ring.get_nodes_for_file('e3b0c44298fc1c...')
        # → ['node1', 'node2', 'node3']  (primary + 2 replicas)
    """

    def __init__(
        self,
        nodes: List[str],
        virtual_nodes_per_physical: int = 150,
        num_replicas: int = 3
    ):
        """
        Initialize consistent hash ring.

        Args:
            nodes: List of physical node IDs
            virtual_nodes_per_physical: Virtual nodes per physical (default: 150)
            num_replicas: Number of replicas (including primary, default: 3)
        """
        self.virtual_nodes_per_physical = virtual_nodes_per_physical
        self.num_replicas = num_replicas

        # Ring: sorted list of virtual nodes
        self.ring: List[VirtualNode] = []

        # Positions: for fast bisect search
        self._positions: List[int] = []

        # Physical nodes
        self.physical_nodes: set = set()

        # Add initial nodes
        for node_id in nodes:
            self.add_node(node_id)

    def _hash_to_ring_position(self, key: str) -> int:
        """
        Hash key to ring position (0 - 2^160).

        Uses SHA-1 for ring hashing (industry standard for consistent hashing).

        Args:
            key: String to hash

        Returns:
            Ring position (0 to 2^160 - 1)
        """
        h = hashlib.sha1(key.encode('utf-8')).digest()
        # Take first 20 bytes, interpret as big-endian integer
        return int.from_bytes(h, byteorder='big')

    def add_node(self, node_id: str):
        """
        Add physical node to ring.

        Creates virtual_nodes_per_physical virtual nodes.

        Args:
            node_id: Physical node ID
        """
        if node_id in self.physical_nodes:
            return  # Already added

        self.physical_nodes.add(node_id)

        # Create virtual nodes
        for i in range(self.virtual_nodes_per_physical):
            # Hash: node_id:virtual_index
            vnode_key = f"{node_id}:vnode{i}"
            position = self._hash_to_ring_position(vnode_key)

            vnode = VirtualNode(
                position=position,
                physical_node_id=node_id,
                replica_index=0  # Primary
            )

            self.ring.append(vnode)

        # Re-sort ring
        self.ring.sort()
        self._positions = [vnode.position for vnode in self.ring]

    def remove_node(self, node_id: str):
        """
        Remove physical node from ring.

        Args:
            node_id: Physical node ID
        """
        if node_id not in self.physical_nodes:
            return

        self.physical_nodes.remove(node_id)

        # Remove all virtual nodes for this physical node
        self.ring = [vnode for vnode in self.ring if vnode.physical_node_id != node_id]
        self._positions = [vnode.position for vnode in self.ring]

    def get_node_for_position(self, position: int) -> Optional[str]:
        """
        Get physical node responsible for position.

        Args:
            position: Ring position

        Returns:
            Physical node ID, or None if ring empty
        """
        if not self.ring:
            return None

        # Find first virtual node >= position (clockwise)
        idx = bisect_right(self._positions, position)

        # Wrap around if past end
        if idx >= len(self.ring):
            idx = 0

        return self.ring[idx].physical_node_id

    def get_nodes_for_file(self, file_id: str) -> List[str]:
        """
        Get nodes responsible for file (primary + replicas).

        Args:
            file_id: SHA-256 hash of file

        Returns:
            List of node IDs (length = num_replicas)
        """
        if not self.ring:
            return []

        # Hash file to ring position
        position = self._hash_to_ring_position(file_id)

        # Find first virtual node >= position
        idx = bisect_right(self._positions, position)
        if idx >= len(self.ring):
            idx = 0

        # Collect unique physical nodes (walk clockwise)
        nodes = []
        seen = set()
        current_idx = idx

        while len(nodes) < self.num_replicas and len(seen) < len(self.physical_nodes):
            vnode = self.ring[current_idx]
            physical = vnode.physical_node_id

            if physical not in seen:
                nodes.append(physical)
                seen.add(physical)

            # Move clockwise
            current_idx = (current_idx + 1) % len(self.ring)

        return nodes

    def get_primary_node(self, file_id: str) -> Optional[str]:
        """
        Get primary node for file.

        Args:
            file_id: SHA-256 hash of file

        Returns:
            Primary node ID
        """
        nodes = self.get_nodes_for_file(file_id)
        return nodes[0] if nodes else None

    def get_replica_nodes(self, file_id: str) -> List[str]:
        """
        Get replica nodes (excluding primary).

        Args:
            file_id: SHA-256 hash of file

        Returns:
            List of replica node IDs
        """
        nodes = self.get_nodes_for_file(file_id)
        return nodes[1:] if len(nodes) > 1 else []

    def get_ring_state(self) -> Dict:
        """
        Get ring state for debugging.

        Returns:
            Dictionary with ring statistics
        """
        return {
            'physical_nodes': len(self.physical_nodes),
            'virtual_nodes': len(self.ring),
            'replicas': self.num_replicas,
            'virtual_per_physical': self.virtual_nodes_per_physical,
            'nodes': sorted(list(self.physical_nodes))
        }

    def rebalance_required(self, file_id: str, current_nodes: List[str]) -> bool:
        """
        Check if file needs rebalancing.

        Args:
            file_id: SHA-256 hash of file
            current_nodes: Current nodes storing file

        Returns:
            True if rebalancing needed
        """
        expected_nodes = set(self.get_nodes_for_file(file_id))
        actual_nodes = set(current_nodes)

        # Rebalance if mismatch
        return expected_nodes != actual_nodes

    def get_rebalance_plan(
        self,
        file_id: str,
        current_nodes: List[str]
    ) -> Tuple[List[str], List[str]]:
        """
        Get rebalancing plan for file.

        Args:
            file_id: SHA-256 hash of file
            current_nodes: Current nodes storing file

        Returns:
            (nodes_to_add, nodes_to_remove)
        """
        expected = set(self.get_nodes_for_file(file_id))
        actual = set(current_nodes)

        nodes_to_add = list(expected - actual)
        nodes_to_remove = list(actual - expected)

        return (nodes_to_add, nodes_to_remove)
