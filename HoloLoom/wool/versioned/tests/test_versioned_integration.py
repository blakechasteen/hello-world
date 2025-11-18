"""
Integration Tests for Versioned Wool Storage
=============================================

Comprehensive integration tests for Phase 8:
- End-to-end versioning workflows
- Time-travel queries (as_of, between, history)
- Delta encoding and reconstruction
- Branch/merge operations
- Conflict detection and resolution
- Version lineage and ancestry

Author: Claude Code
Date: November 17, 2025 (Track 2 Week 1: Integration Tests)
"""

import pytest
import time
from pathlib import Path

from HoloLoom.wool.versioned import (
    VersionedWoolStorage,
    VersionedWoolReference,
    DeltaAlgorithm,
    MergeStrategy,
    MergeResult
)


# Test Fixtures
# =============

@pytest.fixture
def versioned_storage():
    """Create versioned wool storage."""
    storage = VersionedWoolStorage(
        base_path=Path("/tmp/wool_test_versioned"),
        enable_cache=True,
        enable_delta_encoding=True,
        delta_algorithm=None,  # Auto-select
        snapshot_interval=10
    )
    yield storage
    # Cleanup
    storage.close()


@pytest.fixture
def sample_versions():
    """Generate sample version data."""
    return [
        b"Version 1: Initial content",
        b"Version 2: Modified content",
        b"Version 3: Further modifications",
        b"Version 4: Even more changes",
        b"Version 5: Final version"
    ]


# Basic Versioning Tests
# =======================

def test_create_initial_version(versioned_storage):
    """Test creating initial version."""
    data = b"Initial version"

    ref = versioned_storage.store_versioned(
        data,
        content_type="text/plain",
        author="test",
        message="Initial commit"
    )

    assert isinstance(ref, VersionedWoolReference)
    assert ref.version_id == 1
    assert ref.parent_version is None
    assert not ref.is_delta  # First version is snapshot


def test_create_version_chain(versioned_storage, sample_versions):
    """Test creating version chain."""
    refs = []
    parent = None

    for i, data in enumerate(sample_versions):
        ref = versioned_storage.store_versioned(
            data,
            content_type="text/plain",
            parent=parent,
            author="test",
            message=f"Version {i+1}"
        )
        refs.append(ref)
        parent = ref

    # Verify version IDs
    for i, ref in enumerate(refs):
        assert ref.version_id == i + 1

    # Verify parent chain
    for i in range(1, len(refs)):
        assert refs[i].parent_version == refs[i-1].version_id


def test_read_latest_version(versioned_storage, sample_versions):
    """Test reading latest version."""
    parent = None

    for data in sample_versions:
        ref = versioned_storage.store_versioned(
            data,
            parent=parent,
            author="test"
        )
        parent = ref

    # Read latest
    latest_data = versioned_storage.read(ref)
    assert latest_data.tobytes() == sample_versions[-1]


# Time-Travel Query Tests
# ========================

def test_as_of_query(versioned_storage, sample_versions):
    """Test point-in-time query."""
    refs = []
    timestamps = []
    parent = None

    for data in sample_versions:
        ref = versioned_storage.store_versioned(
            data,
            parent=parent,
            author="test"
        )
        refs.append(ref)
        timestamps.append(ref.timestamp)
        parent = ref
        time.sleep(0.01)  # Ensure distinct timestamps

    # Query at different timestamps
    file_id = refs[0].file_id

    # Query at v2 timestamp
    ref_v2 = versioned_storage.as_of(file_id, timestamps[1])
    assert ref_v2.version_id == refs[1].version_id

    # Query at v4 timestamp
    ref_v4 = versioned_storage.as_of(file_id, timestamps[3])
    assert ref_v4.version_id == refs[3].version_id


def test_between_query(versioned_storage, sample_versions):
    """Test range query."""
    refs = []
    timestamps = []
    parent = None

    for data in sample_versions:
        ref = versioned_storage.store_versioned(
            data,
            parent=parent,
            author="test"
        )
        refs.append(ref)
        timestamps.append(ref.timestamp)
        parent = ref
        time.sleep(0.01)

    file_id = refs[0].file_id

    # Query between v2 and v4
    versions_in_range = versioned_storage.between(
        file_id,
        start_time=timestamps[1],
        end_time=timestamps[3]
    )

    # Should return v2, v3, v4
    assert len(versions_in_range) == 3


def test_get_history(versioned_storage, sample_versions):
    """Test getting complete version history."""
    parent = None

    for data in sample_versions:
        ref = versioned_storage.store_versioned(
            data,
            parent=parent,
            author="test"
        )
        parent = ref

    file_id = ref.file_id

    # Get complete history
    history = versioned_storage.get_history(file_id)

    # Should have all 5 versions
    assert len(history) == len(sample_versions)


# Delta Encoding Tests
# =====================

def test_delta_creation(versioned_storage):
    """Test delta encoding between versions."""
    v1_data = b"Hello, world!"
    v2_data = b"Hello, universe!"

    # Store v1 (snapshot)
    ref_v1 = versioned_storage.store_versioned(
        v1_data,
        author="test",
        message="v1"
    )

    # Store v2 (delta)
    ref_v2 = versioned_storage.store_versioned(
        v2_data,
        parent=ref_v1,
        author="test",
        message="v2"
    )

    # v2 should be stored as delta
    assert ref_v2.is_delta
    assert ref_v2.delta_from == ref_v1.file_id


def test_delta_reconstruction(versioned_storage):
    """Test reconstructing version from deltas."""
    v1_data = b"Line 1\nLine 2\nLine 3\n"
    v2_data = b"Line 1\nLine 2 modified\nLine 3\n"
    v3_data = b"Line 1\nLine 2 modified\nLine 3\nLine 4\n"

    # Store version chain
    ref_v1 = versioned_storage.store_versioned(v1_data, message="v1")
    ref_v2 = versioned_storage.store_versioned(v2_data, parent=ref_v1, message="v2")
    ref_v3 = versioned_storage.store_versioned(v3_data, parent=ref_v2, message="v3")

    # Reconstruct v3 (should apply delta chain)
    reconstructed = versioned_storage.reconstruct_version(ref_v3)

    assert reconstructed == v3_data


def test_snapshot_interval(versioned_storage):
    """Test periodic snapshot creation."""
    parent = None

    for i in range(15):
        data = f"Version {i+1}".encode()
        ref = versioned_storage.store_versioned(
            data,
            parent=parent,
            author="test"
        )
        parent = ref

        # Every 10th version should be snapshot
        if (i + 1) % 10 == 0:
            assert not ref.is_delta
        elif i > 0:
            # Others should be deltas (except v1)
            assert ref.is_delta


def test_delta_compression_ratio(versioned_storage):
    """Test delta encoding provides storage savings."""
    # Similar versions
    v1_data = ("Hello, world! " * 100).encode()
    v2_data = ("Hello, world! " * 100 + "Extra line").encode()

    ref_v1 = versioned_storage.store_versioned(v1_data)
    ref_v2 = versioned_storage.store_versioned(v2_data, parent=ref_v1)

    # Get stats
    stats = versioned_storage.get_version_stats()

    # Delta should save storage
    assert stats['bytes_stored'] < stats['bytes_versioned']
    assert stats['storage_savings'] > 0


# Version Lineage Tests
# ======================

def test_get_lineage(versioned_storage):
    """Test getting version lineage."""
    # Create linear version chain
    refs = []
    parent = None

    for i in range(5):
        data = f"Version {i+1}".encode()
        ref = versioned_storage.store_versioned(
            data,
            parent=parent,
            author="test"
        )
        refs.append(ref)
        parent = ref

    # Get lineage of v5
    lineage = versioned_storage.get_lineage(refs[4].version_id)

    # Should be [1, 2, 3, 4, 5]
    assert lineage == [1, 2, 3, 4, 5]


# Branch Operations Tests
# ========================

def test_create_branch(versioned_storage):
    """Test creating a branch."""
    # Create initial version
    ref_v1 = versioned_storage.store_versioned(
        b"Initial",
        message="v1"
    )

    # Create branch
    versioned_storage.create_branch(
        "feature-x",
        from_version=ref_v1.version_id,
        author="test",
        description="Feature X development"
    )

    # Verify branch exists
    branches = versioned_storage.list_branches()
    branch_names = [b.name for b in branches]

    assert "main" in branch_names
    assert "feature-x" in branch_names


def test_switch_branch(versioned_storage):
    """Test switching branches."""
    # Create version on main
    ref_v1 = versioned_storage.store_versioned(b"Main branch")

    # Create and switch to feature branch
    versioned_storage.create_branch("feature")
    versioned_storage.switch_branch("feature")

    assert versioned_storage.current_branch == "feature"


def test_branch_isolation(versioned_storage):
    """Test that branches have independent HEADs."""
    # v1 on main
    ref_v1 = versioned_storage.store_versioned(
        b"Main v1",
        message="Main v1"
    )

    # Create feature branch
    versioned_storage.create_branch("feature", from_version=1)
    versioned_storage.switch_branch("feature")

    # v2 on feature
    ref_v2_feature = versioned_storage.store_versioned(
        b"Feature v2",
        parent=ref_v1,
        message="Feature v2"
    )

    # Switch back to main
    versioned_storage.switch_branch("main")

    # v2 on main (different from feature)
    ref_v2_main = versioned_storage.store_versioned(
        b"Main v2",
        parent=ref_v1,
        message="Main v2"
    )

    # Verify different HEADs
    branches = versioned_storage.list_branches()
    main_branch = next(b for b in branches if b.name == "main")
    feature_branch = next(b for b in branches if b.name == "feature")

    assert main_branch.head_version == ref_v2_main.version_id
    assert feature_branch.head_version == ref_v2_feature.version_id


# Merge Tests
# ============

def test_merge_fast_forward(versioned_storage):
    """Test fast-forward merge."""
    # v1 on main
    ref_v1 = versioned_storage.store_versioned(b"v1", message="v1")

    # Create feature branch
    versioned_storage.create_branch("feature", from_version=1)
    versioned_storage.switch_branch("feature")

    # v2, v3 on feature
    ref_v2 = versioned_storage.store_versioned(b"v2", parent=ref_v1, message="v2")
    ref_v3 = versioned_storage.store_versioned(b"v3", parent=ref_v2, message="v3")

    # Merge feature into main (should be fast-forward)
    result = versioned_storage.merge(source="feature", target="main")

    assert result.success
    assert result.fast_forward
    assert result.merged_version == ref_v3.version_id


def test_merge_no_conflict(versioned_storage):
    """Test merge without conflicts."""
    # v1 on main
    ref_v1 = versioned_storage.store_versioned(b"v1", message="v1")

    # Create feature branch
    versioned_storage.create_branch("feature", from_version=1)

    # v2 on main
    ref_v2_main = versioned_storage.store_versioned(
        b"v2 main",
        parent=ref_v1,
        message="v2 main"
    )

    # v2 on feature (same content as main, no conflict)
    versioned_storage.switch_branch("feature")
    ref_v2_feature = versioned_storage.store_versioned(
        b"v2 main",  # Same content
        parent=ref_v1,
        message="v2 feature"
    )

    # Merge feature into main
    result = versioned_storage.merge(
        source="feature",
        target="main",
        strategy=MergeStrategy.OURS
    )

    assert result.success
    assert not result.fast_forward


def test_merge_with_conflict_ours(versioned_storage):
    """Test merge with conflict using OURS strategy."""
    # v1 on main
    ref_v1 = versioned_storage.store_versioned(b"v1", message="v1")

    # Create feature branch
    versioned_storage.create_branch("feature", from_version=1)

    # v2 on main
    ref_v2_main = versioned_storage.store_versioned(
        b"v2 main content",
        parent=ref_v1,
        message="v2 main"
    )

    # v2 on feature (different content)
    versioned_storage.switch_branch("feature")
    ref_v2_feature = versioned_storage.store_versioned(
        b"v2 feature content",
        parent=ref_v1,
        message="v2 feature"
    )

    # Merge with OURS strategy (keep main's version)
    result = versioned_storage.merge(
        source="feature",
        target="main",
        strategy=MergeStrategy.OURS
    )

    assert result.success
    assert len(result.conflicts) > 0  # Conflict detected
    assert result.strategy_used == MergeStrategy.OURS

    # Verify merged version uses main's content
    versioned_storage.switch_branch("main")
    branches = versioned_storage.list_branches()
    main_branch = next(b for b in branches if b.name == "main")
    merged_ref = versioned_storage._version_metadata[main_branch.head_version]

    merged_data = versioned_storage.reconstruct_version(merged_ref)
    assert merged_data == b"v2 main content"


def test_merge_with_conflict_theirs(versioned_storage):
    """Test merge with conflict using THEIRS strategy."""
    # v1 on main
    ref_v1 = versioned_storage.store_versioned(b"v1", message="v1")

    # Create feature branch
    versioned_storage.create_branch("feature", from_version=1)

    # v2 on main
    ref_v2_main = versioned_storage.store_versioned(
        b"v2 main content",
        parent=ref_v1,
        message="v2 main"
    )

    # v2 on feature (different content)
    versioned_storage.switch_branch("feature")
    ref_v2_feature = versioned_storage.store_versioned(
        b"v2 feature content",
        parent=ref_v1,
        message="v2 feature"
    )

    # Merge with THEIRS strategy (keep feature's version)
    result = versioned_storage.merge(
        source="feature",
        target="main",
        strategy=MergeStrategy.THEIRS
    )

    assert result.success

    # Verify merged version uses feature's content
    versioned_storage.switch_branch("main")
    branches = versioned_storage.list_branches()
    main_branch = next(b for b in branches if b.name == "main")
    merged_ref = versioned_storage._version_metadata[main_branch.head_version]

    merged_data = versioned_storage.reconstruct_version(merged_ref)
    assert merged_data == b"v2 feature content"


def test_merge_manual_conflict_detection(versioned_storage):
    """Test manual merge conflict detection."""
    # v1 on main
    ref_v1 = versioned_storage.store_versioned(b"v1", message="v1")

    # Create feature branch
    versioned_storage.create_branch("feature", from_version=1)

    # Divergent changes
    ref_v2_main = versioned_storage.store_versioned(
        b"main changes",
        parent=ref_v1,
        message="main"
    )

    versioned_storage.switch_branch("feature")
    ref_v2_feature = versioned_storage.store_versioned(
        b"feature changes",
        parent=ref_v1,
        message="feature"
    )

    # Merge with MANUAL strategy (should fail)
    result = versioned_storage.merge(
        source="feature",
        target="main",
        strategy=MergeStrategy.MANUAL
    )

    assert not result.success
    assert len(result.conflicts) > 0


# Statistics Tests
# =================

def test_versioning_statistics(versioned_storage, sample_versions):
    """Test versioning statistics tracking."""
    parent = None

    for data in sample_versions:
        ref = versioned_storage.store_versioned(
            data,
            parent=parent,
            author="test"
        )
        parent = ref

    stats = versioned_storage.get_version_stats()

    assert stats['versions_created'] == len(sample_versions)
    assert stats['snapshots_created'] >= 1
    assert stats['bytes_versioned'] > 0


def test_branch_statistics(versioned_storage):
    """Test branch statistics tracking."""
    # Create version
    ref_v1 = versioned_storage.store_versioned(b"v1")

    # Create multiple branches
    versioned_storage.create_branch("feature1", from_version=1)
    versioned_storage.create_branch("feature2", from_version=1)
    versioned_storage.create_branch("feature3", from_version=1)

    # Merge one branch
    versioned_storage.switch_branch("feature1")
    ref_v2 = versioned_storage.store_versioned(b"v2", parent=ref_v1)
    versioned_storage.merge(source="feature1", target="main")

    stats = versioned_storage.get_stats()
    branch_stats = stats['branches']

    assert branch_stats['total_branches'] == 4  # main + 3 features
    assert branch_stats['merges_attempted'] >= 1
    assert branch_stats['merges_succeeded'] >= 1


# End-to-End Integration Tests
# =============================

def test_e2e_versioning_workflow(versioned_storage):
    """
    End-to-end test: Complete versioning workflow
    - Create versions
    - Time-travel queries
    - Delta encoding
    - Branch/merge
    """
    # 1. Create version chain on main
    ref_v1 = versioned_storage.store_versioned(
        b"Version 1",
        message="Initial"
    )
    t1 = ref_v1.timestamp

    time.sleep(0.01)

    ref_v2 = versioned_storage.store_versioned(
        b"Version 2",
        parent=ref_v1,
        message="Update"
    )
    t2 = ref_v2.timestamp

    # 2. Create feature branch
    versioned_storage.create_branch("feature", from_version=1)
    versioned_storage.switch_branch("feature")

    ref_v3_feature = versioned_storage.store_versioned(
        b"Version 3 (feature)",
        parent=ref_v1,
        message="Feature work"
    )

    # 3. More work on main
    versioned_storage.switch_branch("main")
    ref_v3_main = versioned_storage.store_versioned(
        b"Version 3 (main)",
        parent=ref_v2,
        message="Main work"
    )

    # 4. Time-travel query
    file_id = ref_v1.file_id
    ref_at_t1 = versioned_storage.as_of(file_id, t1)
    assert ref_at_t1.version_id == 1

    # 5. Merge feature into main
    result = versioned_storage.merge(
        source="feature",
        target="main",
        strategy=MergeStrategy.OURS
    )
    assert result.success

    # 6. Verify statistics
    stats = versioned_storage.get_stats()
    assert stats['versioning']['versions_created'] >= 4
    assert stats['branches']['merges_succeeded'] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
