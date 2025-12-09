"""
Integration Tests for Portal Production Hardening Features.

Tests component interactions and end-to-end workflows:
1. Load balancer + node selection
2. Job queue + metrics
3. Version manager + CAS storage
4. PKI + secure communication
5. Full system integration
"""

import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import List
import pytest

# Import all Portal components
from HoloLoom.portal.shared.load_balancer import (
    LoadBalancer,
    LoadBalanceStrategy,
)
from HoloLoom.portal.shared.types import (
    NodeRecord,
    NodeCapabilities,
    NodeStatus,
)
from HoloLoom.portal.shared.metrics import (
    MetricsRegistry,
)
from HoloLoom.portal.shared.pki import (
    PKIManager,
)
from HoloLoom.portal.shared.cas_storage import (
    CASStorage,
)


# =============================================================================
# Test Fixtures
# =============================================================================


def create_node(
    node_id: str,
    cpu_cores: int = 4,
    memory_mb: int = 8192,
    gpu: bool = False,
    current_jobs: int = 0,
    status: NodeStatus = NodeStatus.ONLINE,
) -> NodeRecord:
    """Create a node record for testing."""
    return NodeRecord(
        node_id=node_id,
        address=f"192.168.1.{node_id[-1]}:8080",
        capabilities=NodeCapabilities(
            cpu_cores=cpu_cores,
            memory_mb=memory_mb,
            gpu=gpu,
        ),
        current_jobs=current_jobs,
        status=status,
        last_heartbeat=datetime.utcnow(),
    )


@pytest.fixture
def sample_nodes() -> List[NodeRecord]:
    """Create sample node records for testing."""
    return [
        create_node("node-1", cpu_cores=4, memory_mb=8192, current_jobs=2),
        create_node("node-2", cpu_cores=8, memory_mb=16384, gpu=True, current_jobs=5),
        create_node("node-3", cpu_cores=2, memory_mb=4096, current_jobs=4),
    ]


@pytest.fixture
def temp_storage_dir():
    """Create temporary directory for storage tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_certs_dir():
    """Create temporary directory for PKI tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def fresh_metrics():
    """Create fresh metrics registry for each test."""
    registry = MetricsRegistry()
    registry.reset_all()
    return registry


# =============================================================================
# Load Balancer Integration
# =============================================================================


class TestLoadBalancerIntegration:
    """Test load balancer with node selection strategies."""

    def test_least_loaded_selection(self, sample_nodes):
        """Least loaded strategy selects node with fewest jobs."""
        lb = LoadBalancer(strategy=LoadBalanceStrategy.LEAST_LOADED)
        result = lb.select(sample_nodes)

        assert result.node is not None
        # node-1 has 2 jobs, node-2 has 5, node-3 has 4
        assert result.node.node_id == "node-1"
        assert result.strategy_used == LoadBalanceStrategy.LEAST_LOADED

    def test_weighted_selection(self, sample_nodes):
        """Weighted strategy considers capacity and load."""
        lb = LoadBalancer(strategy=LoadBalanceStrategy.WEIGHTED)
        result = lb.select(sample_nodes)

        assert result.node is not None
        # node-2 has highest capacity (8 cores, 16GB)
        # but also highest load, so weighted score varies
        assert result.candidates_considered == 3

    def test_round_robin_selection(self, sample_nodes):
        """Round robin cycles through nodes."""
        lb = LoadBalancer(strategy=LoadBalanceStrategy.ROUND_ROBIN)

        selections = []
        for _ in range(6):
            result = lb.select(sample_nodes)
            if result.node:
                selections.append(result.node.node_id)

        # Should cycle: 0, 1, 2, 0, 1, 2
        assert selections[0] == selections[3]
        assert selections[1] == selections[4]
        assert selections[2] == selections[5]

    def test_capability_filtering_gpu(self, sample_nodes):
        """Capability strategy filters by GPU requirement."""
        lb = LoadBalancer(strategy=LoadBalanceStrategy.CAPABILITY)

        # Request GPU
        result = lb.select(sample_nodes, job_requirements={"gpu": True})

        assert result.node is not None
        # Only node-2 has GPU
        assert result.node.node_id == "node-2"

    def test_capability_filtering_memory(self, sample_nodes):
        """Capability strategy filters by memory requirement."""
        lb = LoadBalancer(strategy=LoadBalanceStrategy.CAPABILITY)

        # Request high memory
        result = lb.select(sample_nodes, job_requirements={"min_memory_mb": 10000})

        assert result.node is not None
        # Only node-2 has >10GB
        assert result.node.node_id == "node-2"

    def test_empty_node_pool(self):
        """Handle empty node pool gracefully."""
        lb = LoadBalancer()
        result = lb.select([])

        assert result.node is None
        assert result.candidates_considered == 0

    def test_no_matching_capabilities(self, sample_nodes):
        """Fallback when no nodes match requirements."""
        lb = LoadBalancer(strategy=LoadBalanceStrategy.CAPABILITY)

        # Request impossible memory
        result = lb.select(sample_nodes, job_requirements={"min_memory_mb": 1000000})

        # Should fallback to all nodes
        assert result.node is not None

    def test_stats_tracking(self, sample_nodes):
        """Load balancer tracks selection statistics."""
        lb = LoadBalancer(strategy=LoadBalanceStrategy.LEAST_LOADED)

        for _ in range(10):
            lb.select(sample_nodes)

        stats = lb.get_stats()
        assert stats["selections_total"] == 10
        assert "least_loaded" in stats["selections_by_strategy"]
        assert stats["selections_by_strategy"]["least_loaded"] == 10


# =============================================================================
# Metrics Integration
# =============================================================================


class TestMetricsIntegration:
    """Test metrics tracking across operations."""

    def test_job_lifecycle_metrics(self, fresh_metrics):
        """Complete job lifecycle updates all relevant metrics."""
        # Submit job
        fresh_metrics.jobs_submitted.inc()

        # Job queued (no nodes available)
        fresh_metrics.jobs_queued.inc()
        fresh_metrics.queue_depth.set(1)

        # Job dispatched
        fresh_metrics.queue_depth.set(0)

        # Job completed
        fresh_metrics.jobs_completed.inc()
        fresh_metrics.job_duration.observe(150.0)

        # Verify final state
        assert fresh_metrics.jobs_submitted.value == 1
        assert fresh_metrics.jobs_completed.value == 1
        assert fresh_metrics.jobs_queued.value == 1
        assert fresh_metrics.queue_depth.value == 0
        assert fresh_metrics.job_duration.count == 1
        assert fresh_metrics.job_duration.sum == 150.0

    def test_failed_job_metrics(self, fresh_metrics):
        """Failed jobs increment failure counter."""
        fresh_metrics.jobs_submitted.inc()
        fresh_metrics.jobs_failed.inc()

        assert fresh_metrics.jobs_submitted.value == 1
        assert fresh_metrics.jobs_failed.value == 1
        assert fresh_metrics.jobs_completed.value == 0

    def test_queue_wait_time_histogram(self, fresh_metrics):
        """Queue wait times recorded in histogram."""
        wait_times = [50, 100, 250, 500, 1000, 2500]
        for wait in wait_times:
            fresh_metrics.queue_wait_time.observe(wait)

        assert fresh_metrics.queue_wait_time.count == 6
        assert fresh_metrics.queue_wait_time.sum == sum(wait_times)

        # Check bucket distribution
        buckets = fresh_metrics.queue_wait_time.get_bucket_counts()
        assert buckets[100] >= 2  # 50, 100 are <= 100
        assert buckets[500] >= 4  # 50, 100, 250, 500 are <= 500

    def test_node_metrics(self, fresh_metrics):
        """Node metrics track online/total counts."""
        fresh_metrics.nodes_online.set(3)
        fresh_metrics.nodes_total.set(5)

        assert fresh_metrics.nodes_online.value == 3
        assert fresh_metrics.nodes_total.value == 5

    def test_per_node_metrics(self, fresh_metrics):
        """Per-node labeled metrics work correctly."""
        fresh_metrics.node_cpu.labels(node_id="node-1").set(45.5)
        fresh_metrics.node_cpu.labels(node_id="node-2").set(72.3)
        fresh_metrics.node_memory.labels(node_id="node-1").set(65.0)

        all_cpu = fresh_metrics.node_cpu.get_all()
        assert len(all_cpu) == 2

        all_memory = fresh_metrics.node_memory.get_all()
        assert len(all_memory) == 1

    def test_prometheus_export(self, fresh_metrics):
        """Prometheus export includes all metrics."""
        fresh_metrics.jobs_submitted.inc(5)
        fresh_metrics.jobs_completed.inc(3)
        fresh_metrics.nodes_online.set(2)

        output = fresh_metrics.export_prometheus()

        assert "portal_jobs_submitted_total" in output
        assert "portal_jobs_completed_total" in output
        assert "portal_nodes_online" in output
        assert "5" in output
        assert "3" in output

    def test_json_export(self, fresh_metrics):
        """JSON export provides structured data."""
        fresh_metrics.jobs_submitted.inc(10)
        fresh_metrics.jobs_completed.inc(8)
        fresh_metrics.job_duration.observe(100)
        fresh_metrics.job_duration.observe(200)

        data = fresh_metrics.to_dict()

        assert data["jobs"]["submitted"] == 10
        assert data["jobs"]["completed"] == 8
        assert data["jobs"]["duration"]["count"] == 2
        assert data["jobs"]["duration"]["avg_ms"] == 150.0


# =============================================================================
# CAS Storage Integration
# =============================================================================


class TestCASStorageIntegration:
    """Test content-addressed storage operations."""

    def test_deduplication_across_versions(self, temp_storage_dir):
        """Same content in different versions stored once."""
        storage = CASStorage(temp_storage_dir)

        content = b"(module (func (export \"run\")))"

        # Store same content for multiple versions
        hash1 = storage.store(content, "my-module", "v1.0.0")
        hash2 = storage.store(content, "my-module", "v1.0.1")
        hash3 = storage.store(content, "my-module", "v1.1.0")

        # All should return same hash
        assert hash1 == hash2 == hash3

        # Stats show deduplication
        stats = storage.get_stats()
        assert stats.unique_objects == 1
        assert stats.total_references == 3
        assert stats.deduplication_ratio == 3.0

    def test_cross_module_deduplication(self, temp_storage_dir):
        """Different modules share common content."""
        storage = CASStorage(temp_storage_dir)

        content = b"(module (func (export \"common\")))"

        hash1 = storage.store(content, "module-a", "v1.0.0")
        hash2 = storage.store(content, "module-b", "v1.0.0")

        assert hash1 == hash2

        stats = storage.get_stats()
        assert stats.unique_objects == 1
        assert stats.total_references == 2

    def test_version_retrieval(self, temp_storage_dir):
        """Each version retrieves correct content."""
        storage = CASStorage(temp_storage_dir)

        content_v1 = b"(module (func (export \"v1\")))"
        content_v2 = b"(module (func (export \"v2\")))"

        storage.store(content_v1, "my-module", "v1.0.0")
        storage.store(content_v2, "my-module", "v2.0.0")

        assert storage.get("my-module", "v1.0.0") == content_v1
        assert storage.get("my-module", "v2.0.0") == content_v2

    def test_garbage_collection(self, temp_storage_dir):
        """Unreferenced content is garbage collected."""
        storage = CASStorage(temp_storage_dir)

        content = b"(module (func (export \"temp\")))"
        content_hash = storage.store(content, "temp-module", "v1.0.0")

        assert storage.exists_hash(content_hash)

        # Delete reference
        storage.delete_ref("temp-module", "v1.0.0")

        # Run GC
        deleted = storage.garbage_collect()
        assert deleted == 1
        assert not storage.exists_hash(content_hash)

    def test_partial_deletion_preserves_content(self, temp_storage_dir):
        """Deleting one reference preserves shared content."""
        storage = CASStorage(temp_storage_dir)

        content = b"(module (func (export \"shared\")))"
        storage.store(content, "module-a", "v1.0.0")
        storage.store(content, "module-b", "v1.0.0")

        storage.delete_ref("module-a", "v1.0.0")
        deleted = storage.garbage_collect()
        assert deleted == 0

        # Still accessible via other reference
        assert storage.get("module-b", "v1.0.0") == content

    def test_integrity_verification(self, temp_storage_dir):
        """Content integrity can be verified."""
        storage = CASStorage(temp_storage_dir)

        content = b"(module (func (export \"verified\")))"
        content_hash = storage.store(content, "my-module", "v1.0.0")

        assert storage.verify_integrity(content_hash)


# =============================================================================
# PKI Integration
# =============================================================================


class TestPKIIntegration:
    """Test PKI certificate management."""

    def test_ca_and_node_certificates(self, temp_certs_dir):
        """CA generates valid node certificates."""
        pki = PKIManager(temp_certs_dir)
        pki.generate_ca(common_name="Test CA")

        cert1, key1 = pki.issue_node_certificate("node-1")
        cert2, key2 = pki.issue_node_certificate("node-2")

        assert cert1.exists()
        assert key1.exists()
        assert cert2.exists()
        assert key2.exists()

        # Certificates are different
        assert cert1.read_bytes() != cert2.read_bytes()

    def test_certificate_contains_node_id(self, temp_certs_dir):
        """Node certificate contains node ID."""
        pki = PKIManager(temp_certs_dir)
        pki.generate_ca()

        cert_path, _ = pki.issue_node_certificate("my-special-node")
        cert_pem = cert_path.read_bytes()
        info = pki.verify_client_certificate(cert_pem)

        assert info is not None
        assert info.is_valid
        assert info.node_id == "my-special-node"

    def test_ssl_context_creation(self, temp_certs_dir):
        """SSL context created from certificates."""
        pki = PKIManager(temp_certs_dir)
        pki.generate_ca()
        cert_path, key_path = pki.issue_node_certificate("server")

        context = pki.create_client_ssl_context(cert_path, key_path)

        import ssl
        assert context is not None
        assert context.verify_mode == ssl.CERT_REQUIRED

    def test_malformed_certificate_rejected(self, temp_certs_dir):
        """Malformed certificates return None."""
        pki = PKIManager(temp_certs_dir)
        pki.generate_ca()

        info = pki.verify_client_certificate(b"not a certificate")
        assert info is None


# =============================================================================
# Full System Integration
# =============================================================================


class TestFullSystemIntegration:
    """End-to-end tests combining all components."""

    def test_complete_job_lifecycle(
        self, sample_nodes, temp_storage_dir, fresh_metrics
    ):
        """Complete job submission through execution."""
        # Setup
        lb = LoadBalancer(strategy=LoadBalanceStrategy.LEAST_LOADED)
        storage = CASStorage(temp_storage_dir)

        # Step 1: Store WASM module
        wasm_content = b"(module (func (export \"run\") (result i32) (i32.const 42)))"
        storage.store(wasm_content, "job-module", "v1.0.0")
        fresh_metrics.modules_loaded.set(1)

        # Step 2: Submit job
        fresh_metrics.jobs_submitted.inc()
        fresh_metrics.jobs_by_module.labels(module_id="job-module").inc()

        # Step 3: Select node
        result = lb.select(sample_nodes)
        assert result.node is not None
        fresh_metrics.scheduler_selections.inc()

        # Step 4: Simulate execution
        start_time = time.time()
        time.sleep(0.01)  # 10ms
        duration_ms = (time.time() - start_time) * 1000

        # Step 5: Record completion
        fresh_metrics.jobs_completed.inc()
        fresh_metrics.job_duration.observe(duration_ms)

        # Verify
        assert fresh_metrics.jobs_submitted.value == 1
        assert fresh_metrics.jobs_completed.value == 1
        assert storage.exists("job-module", "v1.0.0")

    def test_module_update_workflow(self, temp_storage_dir, fresh_metrics):
        """Module update with deduplication."""
        storage = CASStorage(temp_storage_dir)

        # Deploy v1.0.0
        v1_content = b"(module (func (export \"run\") (result i32) (i32.const 1)))"
        storage.store(v1_content, "updatable-module", "v1.0.0")
        fresh_metrics.modules_loaded.inc()

        # Deploy v1.0.1 with same content (hotfix)
        storage.store(v1_content, "updatable-module", "v1.0.1")

        # Deploy v2.0.0 with different content
        v2_content = b"(module (func (export \"run\") (result i32) (i32.const 2)))"
        storage.store(v2_content, "updatable-module", "v2.0.0")
        fresh_metrics.modules_loaded.inc()

        # Verify deduplication
        stats = storage.get_stats()
        assert stats.unique_objects == 2
        assert stats.total_references == 3

    def test_load_balancer_with_metrics(self, sample_nodes, fresh_metrics):
        """Load balancer operations update metrics."""
        lb = LoadBalancer()

        for _ in range(10):
            result = lb.select(sample_nodes)
            fresh_metrics.scheduler_selections.inc()

        assert fresh_metrics.scheduler_selections.value == 10
        assert lb.get_stats()["selections_total"] == 10

    def test_metrics_export_all_components(self, fresh_metrics):
        """Prometheus export includes all component metrics."""
        fresh_metrics.jobs_submitted.inc(10)
        fresh_metrics.jobs_completed.inc(8)
        fresh_metrics.jobs_failed.inc(1)
        fresh_metrics.nodes_online.set(3)
        fresh_metrics.queue_depth.set(2)
        fresh_metrics.modules_loaded.set(5)
        fresh_metrics.health_checks.inc(100)

        output = fresh_metrics.export_prometheus()

        expected_metrics = [
            "portal_jobs_submitted_total",
            "portal_jobs_completed_total",
            "portal_jobs_failed_total",
            "portal_nodes_online",
            "portal_queue_depth",
            "portal_modules_loaded",
            "portal_health_checks_total",
        ]

        for metric in expected_metrics:
            assert metric in output

    def test_concurrent_metrics_updates(self, fresh_metrics):
        """Metrics handle concurrent updates."""
        import threading

        def update_metrics(count):
            for _ in range(count):
                fresh_metrics.jobs_submitted.inc()
                fresh_metrics.jobs_completed.inc()

        threads = [
            threading.Thread(target=update_metrics, args=(100,))
            for _ in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert fresh_metrics.jobs_submitted.value == 500
        assert fresh_metrics.jobs_completed.value == 500


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_empty_content_storage(self, temp_storage_dir):
        """CAS handles empty content."""
        storage = CASStorage(temp_storage_dir)
        storage.store(b"", "empty-module", "v1.0.0")

        assert storage.exists("empty-module", "v1.0.0")
        assert storage.get("empty-module", "v1.0.0") == b""

    def test_nonexistent_module(self, temp_storage_dir):
        """Nonexistent module returns None."""
        storage = CASStorage(temp_storage_dir)
        assert storage.get("nonexistent", "v1.0.0") is None

    def test_metrics_zero_division(self, fresh_metrics):
        """Metrics handle zero values."""
        data = fresh_metrics.to_dict()

        assert data["jobs"]["duration"]["avg_ms"] == 0.0
        assert data["queue"]["wait_time"]["avg_ms"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
