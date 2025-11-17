"""
Distributed Wool Storage (Phase 6)
===================================

Distributed content-addressable storage with replication and fault tolerance.

Architecture:
    Local Node → Ring (Consistent Hash) → Replica Nodes
        ↓ Store              ↓ Placement         ↓ Replication
    SHA-256 Hash    Virtual Nodes        N Replicas

Features:
- Consistent hashing for node placement
- Configurable replication (default: 3)
- Automatic failover and recovery
- Gossip protocol for node discovery
- Zero-copy network transfer (sendfile)

Author: Claude Code
Date: November 17, 2025 (Phase 6 Design)
"""

from .ring import ConsistentHashRing, VirtualNode
from .node import DistributedWoolNode, NodeConfig, NodeStatus
from .replication import ReplicationManager, ReplicationStrategy
from .network import WoolNetworkProtocol, FileTransfer
from .gossip import GossipProtocol, ClusterState

__all__ = [
    'ConsistentHashRing',
    'VirtualNode',
    'DistributedWoolNode',
    'NodeConfig',
    'NodeStatus',
    'ReplicationManager',
    'ReplicationStrategy',
    'WoolNetworkProtocol',
    'FileTransfer',
    'GossipProtocol',
    'ClusterState',
]
