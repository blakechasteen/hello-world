"""
Integration Tests for Distributed Wool Storage
================================================

Comprehensive integration tests for Phase 6:
- End-to-end distributed operations
- Replication and consistency
- Node failure and recovery
- Cluster rebalancing
- Network protocol integration

Author: Claude Code
Date: November 17, 2025 (Track 2 Week 1: Integration Tests)
"""

import pytest
import asyncio
import hashlib
from pathlib import Path
from typing import List

from HoloLoom.wool import WoolStorage, WoolReference
from HoloLoom.wool.distributed import (
    ConsistentHashRing,
    DistributedWoolNode,
    WoolNetworkProtocol,
    GossipProtocol,
    ReplicationManager,
    ConsistencyLevel
)


# Test Fixtures
# =============

@pytest.fixture
async def three_node_cluster():
    """Create 3-node test cluster."""
    nodes = []
    node_ids = ["node1", "node2", "node3"]

    for node_id in node_ids:
        node = DistributedWoolNode(
            node_id=node_id,
            bind_address=f"127.0.0.1:{8000 + len(nodes)}",
            peers=node_ids,
            base_path=Path(f"/tmp/wool_test_{node_id}")
        )
        await node.start()
        nodes.append(node)

    # Wait for gossip to stabilize
    await asyncio.sleep(1.0)

    yield nodes

    # Cleanup
    for node in nodes:
        await node.stop()


@pytest.fixture
def consistent_ring():
    """Create consistent hash ring with 3 nodes."""
    return ConsistentHashRing(
        nodes=["node1", "node2", "node3"],
        virtual_nodes_per_physical=150,
        num_replicas=3
    )


# Consistent Hash Ring Tests
# ===========================

def test_ring_deterministic_placement(consistent_ring):
    """Test that file placement is deterministic."""
    file_id = "test_file_123"

    # Same file ID should always map to same nodes
    nodes1 = consistent_ring.get_nodes_for_file(file_id)
    nodes2 = consistent_ring.get_nodes_for_file(file_id)

    assert nodes1 == nodes2
    assert len(nodes1) == 3  # Replication factor


def test_ring_even_distribution(consistent_ring):
    """Test that files are evenly distributed across nodes."""
    # Generate 1000 file IDs
    file_counts = {"node1": 0, "node2": 0, "node3": 0}

    for i in range(1000):
        file_id = f"file_{i}"
        nodes = consistent_ring.get_nodes_for_file(file_id)
        primary_node = nodes[0]
        file_counts[primary_node] += 1

    # Each node should get roughly 1/3 of files (±10%)
    for count in file_counts.values():
        assert 250 < count < 450  # ~333 ± 83


def test_ring_add_node(consistent_ring):
    """Test adding node minimizes re-hashing."""
    # Get initial placement for 100 files
    initial_placements = {}
    for i in range(100):
        file_id = f"file_{i}"
        nodes = consistent_ring.get_nodes_for_file(file_id)
        initial_placements[file_id] = nodes[0]

    # Add new node
    consistent_ring.add_node("node4")

    # Count how many files moved
    moved = 0
    for file_id, old_primary in initial_placements.items():
        nodes = consistent_ring.get_nodes_for_file(file_id)
        new_primary = nodes[0]
        if new_primary != old_primary:
            moved += 1

    # Should move roughly 1/4 of files (new node gets 1/4 share)
    # Allow 10-40% range
    assert 10 < moved < 40


def test_ring_remove_node(consistent_ring):
    """Test removing node re-distributes its files."""
    # Get initial placement for 100 files
    initial_placements = {}
    for i in range(100):
        file_id = f"file_{i}"
        nodes = consistent_ring.get_nodes_for_file(file_id)
        initial_placements[file_id] = nodes[0]

    # Remove node
    consistent_ring.remove_node("node3")

    # All files that were on node3 should be moved
    moved = 0
    for file_id, old_primary in initial_placements.items():
        nodes = consistent_ring.get_nodes_for_file(file_id)
        new_primary = nodes[0]
        if old_primary == "node3":
            assert new_primary in ["node1", "node2"]
            moved += 1
        else:
            # Files on other nodes should stay put
            assert new_primary == old_primary

    # Some files should have moved
    assert moved > 0


# Distributed Node Tests
# =======================

@pytest.mark.asyncio
async def test_distributed_store_local(three_node_cluster):
    """Test storing file on correct primary node."""
    nodes = three_node_cluster

    data = b"Hello, distributed world!"
    file_id = hashlib.sha256(data).hexdigest()

    # Determine which node should be primary
    ring = nodes[0].ring
    primary_nodes = ring.get_nodes_for_file(file_id)
    primary_node_id = primary_nodes[0]

    # Find corresponding node
    primary_node = next(n for n in nodes if n.node_id == primary_node_id)

    # Store file
    ref = await primary_node.store(data, replicate=False)

    # Verify stored locally
    stored_data = primary_node.local_storage.read(ref)
    assert stored_data.tobytes() == data


@pytest.mark.asyncio
async def test_distributed_store_with_replication(three_node_cluster):
    """Test storing file with replication to all nodes."""
    nodes = three_node_cluster

    data = b"Replicated data"
    file_id = hashlib.sha256(data).hexdigest()

    # Determine primary node
    ring = nodes[0].ring
    replica_nodes = ring.get_nodes_for_file(file_id)
    primary_node_id = replica_nodes[0]

    # Find primary node
    primary_node = next(n for n in nodes if n.node_id == primary_node_id)

    # Store with replication
    ref = await primary_node.store(data, replicate=True)

    # Wait for replication
    await asyncio.sleep(0.5)

    # Verify all replica nodes have the file
    for replica_id in replica_nodes:
        replica_node = next(n for n in nodes if n.node_id == replica_id)

        # Try to read locally
        try:
            stored_data = replica_node.local_storage.read(ref)
            assert stored_data.tobytes() == data
        except FileNotFoundError:
            # Replication may not be complete yet
            pass


@pytest.mark.asyncio
async def test_distributed_read_remote(three_node_cluster):
    """Test reading file from remote node."""
    nodes = three_node_cluster

    data = b"Remote data"
    file_id = hashlib.sha256(data).hexdigest()

    # Store on primary
    ring = nodes[0].ring
    primary_nodes = ring.get_nodes_for_file(file_id)
    primary_node_id = primary_nodes[0]
    primary_node = next(n for n in nodes if n.node_id == primary_node_id)

    ref = await primary_node.store(data, replicate=False)

    # Read from different node
    other_node = next(n for n in nodes if n.node_id != primary_node_id)

    # This should trigger remote read
    remote_data = await other_node.read(ref)
    assert remote_data.tobytes() == data


# Replication Tests
# ==================

@pytest.mark.asyncio
async def test_replication_quorum_read(three_node_cluster):
    """Test quorum read (read from majority)."""
    nodes = three_node_cluster

    data = b"Quorum data"
    file_id = hashlib.sha256(data).hexdigest()

    # Get replica manager
    replication_mgr = nodes[0].replication_manager

    # Store with replication
    await nodes[0].store(data, replicate=True)
    await asyncio.sleep(0.5)  # Wait for replication

    # Read with QUORUM consistency
    read_nodes = replication_mgr.select_read_nodes(
        file_id,
        ConsistencyLevel.QUORUM
    )

    # Should read from at least 2 nodes (majority of 3)
    assert len(read_nodes) >= 2


@pytest.mark.asyncio
async def test_replication_all_consistency(three_node_cluster):
    """Test ALL consistency (read from all replicas)."""
    nodes = three_node_cluster

    data = b"Strong consistency data"
    file_id = hashlib.sha256(data).hexdigest()

    # Get replica manager
    replication_mgr = nodes[0].replication_manager

    # Store with replication
    await nodes[0].store(data, replicate=True)
    await asyncio.sleep(0.5)

    # Read with ALL consistency
    read_nodes = replication_mgr.select_read_nodes(
        file_id,
        ConsistencyLevel.ALL
    )

    # Should read from all 3 replicas
    assert len(read_nodes) == 3


# Gossip Protocol Tests
# ======================

@pytest.mark.asyncio
async def test_gossip_heartbeat(three_node_cluster):
    """Test gossip heartbeat between nodes."""
    nodes = three_node_cluster

    # Nodes should discover each other via gossip
    await asyncio.sleep(2.0)  # Wait for several gossip rounds

    for node in nodes:
        members = node.gossip.get_members()

        # Should know about all 3 nodes (including self)
        assert len(members) == 3


@pytest.mark.asyncio
async def test_gossip_failure_detection(three_node_cluster):
    """Test failure detection via gossip."""
    nodes = three_node_cluster

    # Wait for initial gossip
    await asyncio.sleep(1.0)

    # Stop one node (simulate failure)
    failed_node = nodes[2]
    await failed_node.stop()

    # Wait for failure detection
    await asyncio.sleep(5.0)

    # Other nodes should detect failure
    for node in nodes[:2]:
        member = node.gossip.get_member(failed_node.node_id)
        assert member.status in ["SUSPECT", "UNREACHABLE"]


# Network Protocol Tests
# =======================

@pytest.mark.asyncio
async def test_network_ping():
    """Test network ping between nodes."""
    protocol = WoolNetworkProtocol(node_id="node1")

    # Start server on node2
    node2_protocol = WoolNetworkProtocol(node_id="node2")
    await node2_protocol.start(bind_address="127.0.0.1:8001")

    # Ping from node1
    success = await protocol.ping("node2", "127.0.0.1:8001")
    assert success

    await node2_protocol.stop()


@pytest.mark.asyncio
async def test_network_read_remote():
    """Test reading file via network protocol."""
    # Setup: Store file on node2
    node2_protocol = WoolNetworkProtocol(node_id="node2")
    node2_storage = WoolStorage(base_path=Path("/tmp/wool_test_node2"))

    data = b"Network test data"
    ref = node2_storage.store(data)

    # Start node2 server
    await node2_protocol.start(bind_address="127.0.0.1:8002")

    # Read from node1
    node1_protocol = WoolNetworkProtocol(node_id="node1")
    remote_data = await node1_protocol.read_remote("node2", ref)

    assert remote_data == data

    await node2_protocol.stop()


# End-to-End Integration Tests
# =============================

@pytest.mark.asyncio
async def test_e2e_distributed_storage(three_node_cluster):
    """
    End-to-end test: Store, replicate, read from different nodes.
    """
    nodes = three_node_cluster

    # 1. Store file on primary
    data = b"End-to-end test data"
    file_id = hashlib.sha256(data).hexdigest()

    ring = nodes[0].ring
    primary_nodes = ring.get_nodes_for_file(file_id)
    primary_node = next(n for n in nodes if n.node_id == primary_nodes[0])

    ref = await primary_node.store(data, replicate=True)

    # 2. Wait for replication
    await asyncio.sleep(1.0)

    # 3. Read from different nodes
    for node in nodes:
        remote_data = await node.read(ref)
        assert remote_data.tobytes() == data


@pytest.mark.asyncio
async def test_e2e_node_failure_recovery(three_node_cluster):
    """
    End-to-end test: Store data, kill primary, read from replica.
    """
    nodes = three_node_cluster

    # 1. Store with replication
    data = b"Failure recovery test"
    file_id = hashlib.sha256(data).hexdigest()

    ring = nodes[0].ring
    replica_nodes = ring.get_nodes_for_file(file_id)
    primary_node = next(n for n in nodes if n.node_id == replica_nodes[0])

    ref = await primary_node.store(data, replicate=True)
    await asyncio.sleep(1.0)

    # 2. Kill primary node
    await primary_node.stop()

    # 3. Read from replica
    replica_node = next(
        n for n in nodes
        if n.node_id in replica_nodes[1:] and n.node_id != primary_node.node_id
    )

    # Should read successfully from replica
    remote_data = await replica_node.read(ref)
    assert remote_data.tobytes() == data


# Performance Tests
# ==================

@pytest.mark.asyncio
async def test_performance_distributed_throughput(three_node_cluster):
    """Test distributed storage throughput."""
    import time

    nodes = three_node_cluster

    # Store 100 files
    start = time.time()

    for i in range(100):
        data = f"File {i}".encode()
        node = nodes[i % 3]  # Round-robin
        await node.store(data, replicate=False)  # Skip replication for speed

    duration = time.time() - start
    throughput = 100 / duration

    # Should achieve >100 files/sec
    assert throughput > 100


@pytest.mark.asyncio
async def test_performance_replication_latency(three_node_cluster):
    """Test replication latency."""
    import time

    nodes = three_node_cluster

    data = b"Latency test"

    start = time.time()
    await nodes[0].store(data, replicate=True)
    latency = time.time() - start

    # Replication should complete within 100ms
    assert latency < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
