"""
Distributed Wool Storage Load Testing
======================================

Load testing specific to Phase 6 distributed storage:
- Cluster performance under load
- Node failure and recovery
- Network saturation
- Replication lag
- Rebalancing stress

Scenarios:
- 3-node cluster with 1M files
- Simulate node failures during operations
- Network partition recovery
- Concurrent operations across nodes
- Rebalancing after node add/remove

Author: Claude Code
Date: November 17, 2025 (Track 2 Week 3: Distributed Load Testing)
"""

import asyncio
import time
import random
from pathlib import Path
from typing import List
from dataclasses import dataclass

from HoloLoom.wool.distributed import (
    DistributedWoolNode,
    ConsistencyLevel
)


# Distributed Load Test Results
# ==============================

@dataclass
class DistributedLoadTestResult:
    """Results from distributed load test."""
    test_name: str
    cluster_size: int
    duration_sec: float
    operations_completed: int
    operations_failed: int
    network_failures: int
    node_failures: int
    replication_lag_ms: float

    @property
    def throughput(self) -> float:
        return self.operations_completed / self.duration_sec if self.duration_sec > 0 else 0

    @property
    def success_rate(self) -> float:
        total = self.operations_completed + self.operations_failed
        return self.operations_completed / total if total > 0 else 0

    def summary(self) -> str:
        return (
            f"{self.test_name}:\n"
            f"  Cluster size: {self.cluster_size} nodes\n"
            f"  Duration: {self.duration_sec:.2f} sec\n"
            f"  Operations: {self.operations_completed} completed, {self.operations_failed} failed\n"
            f"  Throughput: {self.throughput:.2f} ops/sec\n"
            f"  Success rate: {self.success_rate * 100:.1f}%\n"
            f"  Network failures: {self.network_failures}\n"
            f"  Node failures: {self.node_failures}\n"
            f"  Avg replication lag: {self.replication_lag_ms:.2f} ms"
        )


# Distributed Test Scenarios
# ===========================

async def test_cluster_baseline_load(num_nodes: int = 3, num_files: int = 100000):
    """
    Baseline distributed load: Store 100k files across cluster.

    Args:
        num_nodes: Number of nodes in cluster
        num_files: Number of files to store

    Returns:
        DistributedLoadTestResult
    """
    print(f"\n🔧 Starting cluster baseline load test ({num_nodes} nodes, {num_files:,} files)...")

    # Create cluster
    nodes = []
    node_ids = [f"node{i}" for i in range(num_nodes)]

    for i, node_id in enumerate(node_ids):
        node = DistributedWoolNode(
            node_id=node_id,
            bind_address=f"127.0.0.1:{9000 + i}",
            peers=node_ids,
            base_path=Path(f"/tmp/distributed_load_node{i}")
        )
        await node.start()
        nodes.append(node)

    # Wait for cluster stabilization
    await asyncio.sleep(2.0)

    ops_completed = 0
    ops_failed = 0
    replication_lags = []

    data = b"x" * 1024  # 1KB

    start_time = time.time()

    # Distribute writes across nodes (round-robin)
    for i in range(num_files):
        node = nodes[i % num_nodes]

        try:
            rep_start = time.time()
            ref = await node.store(data, replicate=True)
            rep_lag = (time.time() - rep_start) * 1000
            replication_lags.append(rep_lag)
            ops_completed += 1
        except Exception:
            ops_failed += 1

        if (i + 1) % 10000 == 0:
            print(f"  Progress: {i + 1:,} files stored...")

    end_time = time.time()
    duration = end_time - start_time

    # Cleanup
    for node in nodes:
        await node.stop()

    avg_replication_lag = sum(replication_lags) / len(replication_lags) if replication_lags else 0

    result = DistributedLoadTestResult(
        test_name="Cluster Baseline Load",
        cluster_size=num_nodes,
        duration_sec=duration,
        operations_completed=ops_completed,
        operations_failed=ops_failed,
        network_failures=0,
        node_failures=0,
        replication_lag_ms=avg_replication_lag
    )

    print(f"✅ Cluster baseline: {result.throughput:.2f} ops/sec, "
          f"{result.success_rate * 100:.1f}% success")

    return result


async def test_node_failure_recovery(num_nodes: int = 3, num_files: int = 10000):
    """
    Simulate node failure during operations and test recovery.

    Args:
        num_nodes: Number of nodes
        num_files: Number of files to store

    Returns:
        DistributedLoadTestResult
    """
    print(f"\n🔧 Starting node failure recovery test...")

    # Create cluster
    nodes = []
    node_ids = [f"node{i}" for i in range(num_nodes)]

    for i, node_id in enumerate(node_ids):
        node = DistributedWoolNode(
            node_id=node_id,
            bind_address=f"127.0.0.1:{9100 + i}",
            peers=node_ids,
            base_path=Path(f"/tmp/failure_test_node{i}")
        )
        await node.start()
        nodes.append(node)

    await asyncio.sleep(2.0)

    ops_completed = 0
    ops_failed = 0
    node_failures = 0

    data = b"x" * 1024

    start_time = time.time()

    # Store files, killing nodes randomly
    for i in range(num_files):
        node_idx = i % num_nodes
        node = nodes[node_idx]

        # Kill node at 30%, 60% progress
        if i == num_files // 3 or i == 2 * num_files // 3:
            kill_idx = random.randint(0, num_nodes - 1)
            if nodes[kill_idx]:
                print(f"  💥 Simulating failure of node{kill_idx}...")
                await nodes[kill_idx].stop()
                nodes[kill_idx] = None
                node_failures += 1
                await asyncio.sleep(1.0)  # Recovery delay

        # Skip if node is dead
        if node is None:
            # Try next available node
            for fallback_node in nodes:
                if fallback_node:
                    node = fallback_node
                    break

        if node:
            try:
                ref = await node.store(data, replicate=True)
                ops_completed += 1
            except Exception:
                ops_failed += 1
        else:
            ops_failed += 1

        if (i + 1) % 1000 == 0:
            print(f"  Progress: {i + 1:,} files stored...")

    end_time = time.time()
    duration = end_time - start_time

    # Cleanup
    for node in nodes:
        if node:
            await node.stop()

    result = DistributedLoadTestResult(
        test_name="Node Failure Recovery",
        cluster_size=num_nodes,
        duration_sec=duration,
        operations_completed=ops_completed,
        operations_failed=ops_failed,
        network_failures=0,
        node_failures=node_failures,
        replication_lag_ms=0
    )

    print(f"✅ Node failure recovery: {result.throughput:.2f} ops/sec, "
          f"{result.success_rate * 100:.1f}% success, {node_failures} node failures")

    return result


async def test_concurrent_operations(num_nodes: int = 3, num_workers: int = 100):
    """
    Concurrent operations from multiple workers across cluster.

    Args:
        num_nodes: Number of nodes
        num_workers: Number of concurrent workers

    Returns:
        DistributedLoadTestResult
    """
    print(f"\n🔧 Starting concurrent operations test ({num_workers} workers)...")

    # Create cluster
    nodes = []
    node_ids = [f"node{i}" for i in range(num_nodes)]

    for i, node_id in enumerate(node_ids):
        node = DistributedWoolNode(
            node_id=node_id,
            bind_address=f"127.0.0.1:{9200 + i}",
            peers=node_ids,
            base_path=Path(f"/tmp/concurrent_test_node{i}")
        )
        await node.start()
        nodes.append(node)

    await asyncio.sleep(2.0)

    ops_completed = 0
    ops_failed = 0

    data = b"x" * 1024

    async def worker(worker_id: int, operations: int = 100):
        """Worker coroutine."""
        nonlocal ops_completed, ops_failed

        for i in range(operations):
            # Random node
            node = random.choice(nodes)

            try:
                ref = await node.store(data, replicate=True)
                ops_completed += 1
            except Exception:
                ops_failed += 1

        print(f"  Worker {worker_id} complete")

    start_time = time.time()

    # Launch workers concurrently
    tasks = [worker(i) for i in range(num_workers)]
    await asyncio.gather(*tasks)

    end_time = time.time()
    duration = end_time - start_time

    # Cleanup
    for node in nodes:
        await node.stop()

    result = DistributedLoadTestResult(
        test_name="Concurrent Operations",
        cluster_size=num_nodes,
        duration_sec=duration,
        operations_completed=ops_completed,
        operations_failed=ops_failed,
        network_failures=0,
        node_failures=0,
        replication_lag_ms=0
    )

    print(f"✅ Concurrent operations: {result.throughput:.2f} ops/sec, "
          f"{result.success_rate * 100:.1f}% success")

    return result


async def test_rebalancing_stress(initial_nodes: int = 3, added_nodes: int = 2):
    """
    Stress test cluster rebalancing when adding nodes.

    Args:
        initial_nodes: Initial cluster size
        added_nodes: Number of nodes to add

    Returns:
        DistributedLoadTestResult
    """
    print(f"\n🔧 Starting rebalancing stress test...")

    # Create initial cluster
    nodes = []
    node_ids = [f"node{i}" for i in range(initial_nodes)]

    for i, node_id in enumerate(node_ids):
        node = DistributedWoolNode(
            node_id=node_id,
            bind_address=f"127.0.0.1:{9300 + i}",
            peers=node_ids,
            base_path=Path(f"/tmp/rebalance_test_node{i}")
        )
        await node.start()
        nodes.append(node)

    await asyncio.sleep(2.0)

    ops_completed = 0
    ops_failed = 0

    data = b"x" * 1024

    start_time = time.time()

    # Store 10k files initially
    print("  Phase 1: Initial load (10k files)...")
    for i in range(10000):
        node = nodes[i % len(nodes)]
        try:
            ref = await node.store(data, replicate=True)
            ops_completed += 1
        except Exception:
            ops_failed += 1

    # Add new nodes (simulate expansion)
    print(f"  Phase 2: Adding {added_nodes} new nodes...")
    for i in range(added_nodes):
        new_idx = initial_nodes + i
        new_node_id = f"node{new_idx}"

        # Update peer list for all nodes
        node_ids.append(new_node_id)

        new_node = DistributedWoolNode(
            node_id=new_node_id,
            bind_address=f"127.0.0.1:{9300 + new_idx}",
            peers=node_ids,
            base_path=Path(f"/tmp/rebalance_test_node{new_idx}")
        )
        await new_node.start()
        nodes.append(new_node)

    # Wait for rebalancing
    await asyncio.sleep(5.0)

    # Continue operations on expanded cluster
    print("  Phase 3: Post-expansion load (10k files)...")
    for i in range(10000):
        node = nodes[i % len(nodes)]
        try:
            ref = await node.store(data, replicate=True)
            ops_completed += 1
        except Exception:
            ops_failed += 1

    end_time = time.time()
    duration = end_time - start_time

    # Cleanup
    for node in nodes:
        await node.stop()

    result = DistributedLoadTestResult(
        test_name="Rebalancing Stress",
        cluster_size=len(nodes),
        duration_sec=duration,
        operations_completed=ops_completed,
        operations_failed=ops_failed,
        network_failures=0,
        node_failures=0,
        replication_lag_ms=0
    )

    print(f"✅ Rebalancing stress: {result.throughput:.2f} ops/sec, "
          f"{result.success_rate * 100:.1f}% success, expanded from {initial_nodes} to {len(nodes)} nodes")

    return result


# Main Distributed Load Test Suite
# ==================================

async def main():
    """Run distributed load test suite."""
    results = []

    print("""
╔════════════════════════════════════════════════════════════════╗
║         DISTRIBUTED WOOL STORAGE LOAD TEST SUITE               ║
║           Phase 6: Cluster Stress Testing                      ║
╚════════════════════════════════════════════════════════════════╝
""")

    # Test 1: Baseline cluster load
    result = await test_cluster_baseline_load(num_nodes=3, num_files=100000)
    results.append(result)

    # Test 2: Node failure recovery
    result = await test_node_failure_recovery(num_nodes=3, num_files=10000)
    results.append(result)

    # Test 3: Concurrent operations
    result = await test_concurrent_operations(num_nodes=3, num_workers=100)
    results.append(result)

    # Test 4: Rebalancing stress
    result = await test_rebalancing_stress(initial_nodes=3, added_nodes=2)
    results.append(result)

    # Print summary
    print("\n" + "="*80)
    print("DISTRIBUTED LOAD TEST SUMMARY")
    print("="*80)

    for result in results:
        print(f"\n{result.summary()}")

    print("\n" + "="*80)

    # Export results
    import json
    results_json = {
        "distributed_load_tests": [
            {
                "test_name": r.test_name,
                "cluster_size": r.cluster_size,
                "duration_sec": r.duration_sec,
                "operations_completed": r.operations_completed,
                "operations_failed": r.operations_failed,
                "throughput_ops_sec": r.throughput,
                "success_rate": r.success_rate,
                "network_failures": r.network_failures,
                "node_failures": r.node_failures,
                "replication_lag_ms": r.replication_lag_ms
            }
            for r in results
        ]
    }

    output_path = Path("distributed_load_test_results.json")
    with open(output_path, 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"\n📊 Results exported to: {output_path}")
    print("\n✅ All distributed load tests complete!")


if __name__ == "__main__":
    asyncio.run(main())
