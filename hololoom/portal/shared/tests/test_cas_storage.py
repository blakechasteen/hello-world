"""
Tests for Content-Addressed Storage (CAS) module.

Tests cover:
- CASEntry and CASStats dataclasses
- Store and retrieve operations
- Deduplication (same content, different references)
- Reference deletion and garbage collection
- Integrity verification
- Thread safety
- Edge cases and error handling
"""

import hashlib
import tempfile
import threading
import time
from pathlib import Path
from unittest import mock

import pytest

from hololoom.portal.shared.cas_storage import (
    CASEntry,
    CASReference,
    CASStats,
    CASStorage,
)


# =============================================================================
# CASEntry Tests
# =============================================================================


class TestCASEntry:
    """Tests for CASEntry dataclass."""

    def test_create_entry(self):
        """CASEntry can be created with required fields."""
        entry = CASEntry(
            content_hash="abc123def456",
            size_bytes=1024,
            path=Path("/some/path"),
        )
        assert entry.content_hash == "abc123def456"
        assert entry.size_bytes == 1024
        assert entry.path == Path("/some/path")
        assert entry.reference_count == 1  # Default

    def test_entry_defaults(self):
        """CASEntry has sensible defaults."""
        entry = CASEntry(
            content_hash="abc123",
            size_bytes=512,
            path=Path("/tmp/test"),
        )
        assert entry.reference_count == 1
        assert entry.created_at > 0
        assert entry.last_accessed > 0

    def test_entry_to_dict(self):
        """CASEntry.to_dict() returns serializable dictionary."""
        test_path = Path("some") / "path"
        entry = CASEntry(
            content_hash="abc123def456",
            size_bytes=1024,
            path=test_path,
            reference_count=3,
        )
        d = entry.to_dict()
        assert d["content_hash"] == "abc123def456"
        assert d["size_bytes"] == 1024
        assert d["path"] == str(test_path)  # Platform-agnostic
        assert d["reference_count"] == 3
        assert "created_at" in d
        assert "last_accessed" in d


# =============================================================================
# CASStats Tests
# =============================================================================


class TestCASStats:
    """Tests for CASStats dataclass."""

    def test_create_stats(self):
        """CASStats can be created with default values."""
        stats = CASStats()
        assert stats.unique_objects == 0
        assert stats.total_references == 0
        assert stats.total_size_bytes == 0
        assert stats.deduplication_ratio == 1.0

    def test_stats_with_values(self):
        """CASStats accepts custom values."""
        stats = CASStats(
            unique_objects=10,
            total_references=25,
            total_size_bytes=1024000,
            deduplication_ratio=2.5,
        )
        assert stats.unique_objects == 10
        assert stats.total_references == 25
        assert stats.total_size_bytes == 1024000
        assert stats.deduplication_ratio == 2.5

    def test_stats_to_dict(self):
        """CASStats.to_dict() returns serializable dictionary."""
        stats = CASStats(
            unique_objects=5,
            total_references=15,
            total_size_bytes=5000,
            deduplication_ratio=3.0,
            objects_by_size={"< 1KB": 3, "1KB - 1MB": 2},
        )
        d = stats.to_dict()
        assert d["unique_objects"] == 5
        assert d["total_references"] == 15
        assert d["total_size_bytes"] == 5000
        assert d["deduplication_ratio"] == 3.0
        assert d["size_distribution"]["< 1KB"] == 3


# =============================================================================
# CASReference Tests
# =============================================================================


class TestCASReference:
    """Tests for CASReference dataclass."""

    def test_create_reference(self):
        """CASReference can be created with required fields."""
        ref = CASReference(
            module_id="my-module",
            version="v1.0.0",
            content_hash="abc123def456",
        )
        assert ref.module_id == "my-module"
        assert ref.version == "v1.0.0"
        assert ref.content_hash == "abc123def456"
        assert ref.created_at > 0

    def test_reference_to_dict(self):
        """CASReference.to_dict() returns serializable dictionary."""
        ref = CASReference(
            module_id="test-module",
            version="v2.0.0",
            content_hash="xyz789",
        )
        d = ref.to_dict()
        assert d["module_id"] == "test-module"
        assert d["version"] == "v2.0.0"
        assert d["content_hash"] == "xyz789"
        assert "created_at" in d


# =============================================================================
# CASStorage Basic Tests
# =============================================================================


class TestCASStorageBasic:
    """Basic tests for CASStorage operations."""

    def test_create_storage(self):
        """CASStorage creates directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = CASStorage(Path(tmpdir))
            assert storage.objects_dir.exists()
            assert storage.refs_dir.exists()

    def test_store_and_get(self):
        """Store content and retrieve it by reference."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = CASStorage(Path(tmpdir))

            content = b"Hello, World!"
            content_hash = storage.store(content, "my-module", "v1.0.0")

            # Verify hash is SHA256
            expected_hash = hashlib.sha256(content).hexdigest()
            assert content_hash == expected_hash

            # Retrieve by reference
            retrieved = storage.get("my-module", "v1.0.0")
            assert retrieved == content

    def test_get_by_hash(self):
        """Retrieve content directly by hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = CASStorage(Path(tmpdir))

            content = b"Test content for hash retrieval"
            content_hash = storage.store(content, "test-mod", "v1.0.0")

            # Get by hash
            retrieved = storage.get_by_hash(content_hash)
            assert retrieved == content

    def test_get_nonexistent_returns_none(self):
        """Getting non-existent content returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = CASStorage(Path(tmpdir))

            # Non-existent reference
            assert storage.get("no-module", "v1.0.0") is None

            # Non-existent hash
            assert storage.get_by_hash("nonexistent123") is None

    def test_get_hash(self):
        """Get hash for module/version without reading content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = CASStorage(Path(tmpdir))

            content = b"Some content"
            expected_hash = storage.store(content, "mod", "v1")

            # Get hash only
            retrieved_hash = storage.get_hash("mod", "v1")
            assert retrieved_hash == expected_hash

            # Non-existent
            assert storage.get_hash("no-mod", "v1") is None


# =============================================================================
# CASStorage Deduplication Tests
# =============================================================================


class TestCASStorageDeduplication:
    """Tests for content deduplication."""

    def test_identical_content_same_hash(self):
        """Identical content produces same hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = CASStorage(Path(tmpdir))

            content = b"Identical content"
            hash1 = storage.store(content, "module1", "v1.0.0")
            hash2 = storage.store(content, "module2", "v1.0.0")

            assert hash1 == hash2

    def test_deduplication_increments_reference_count(self):
        """Storing duplicate content increments reference count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = CASStorage(Path(tmpdir))

            content = b"Duplicate content"
            hash1 = storage.store(content, "mod1", "v1")

            entry = storage.get_entry(hash1)
            assert entry.reference_count == 1

            # Store same content again
            storage.store(content, "mod2", "v1")

            entry = storage.get_entry(hash1)
            assert entry.reference_count == 2

    def test_deduplication_ratio(self):
        """Deduplication ratio reflects actual savings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = CASStorage(Path(tmpdir))

            content = b"Shared content"
            storage.store(content, "mod1", "v1")
            storage.store(content, "mod2", "v1")
            storage.store(content, "mod3", "v1")

            stats = storage.get_stats()
            assert stats.unique_objects == 1
            assert stats.total_references == 3
            assert stats.deduplication_ratio == 3.0

    def test_different_content_different_hash(self):
        """Different content produces different hashes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = CASStorage(Path(tmpdir))

            hash1 = storage.store(b"Content A", "mod", "v1")
            hash2 = storage.store(b"Content B", "mod", "v2")

            assert hash1 != hash2


# =============================================================================
# CASStorage Reference Management Tests
# =============================================================================


class TestCASStorageReferenceManagement:
    """Tests for reference management and updates."""

    def test_exists(self):
        """exists() correctly reports reference existence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = CASStorage(Path(tmpdir))

            assert not storage.exists("mod", "v1")

            storage.store(b"content", "mod", "v1")

            assert storage.exists("mod", "v1")
            assert not storage.exists("mod", "v2")

    def test_exists_hash(self):
        """exists_hash() correctly reports content existence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = CASStorage(Path(tmpdir))

            content_hash = storage.store(b"content", "mod", "v1")

            assert storage.exists_hash(content_hash)
            assert not storage.exists_hash("nonexistent123")

    def test_list_modules(self):
        """list_modules() returns all module IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = CASStorage(Path(tmpdir))

            storage.store(b"c1", "module-a", "v1")
            storage.store(b"c2", "module-b", "v1")
            storage.store(b"c3", "module-a", "v2")

            modules = storage.list_modules()
            assert set(modules) == {"module-a", "module-b"}

    def test_list_versions(self):
        """list_versions() returns all versions of a module."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = CASStorage(Path(tmpdir))

            storage.store(b"c1", "my-module", "v1.0.0")
            storage.store(b"c2", "my-module", "v1.1.0")
            storage.store(b"c3", "my-module", "v2.0.0")
            storage.store(b"c4", "other", "v1.0.0")

            versions = storage.list_versions("my-module")
            assert set(versions) == {"v1.0.0", "v1.1.0", "v2.0.0"}

            # Non-existent module
            assert storage.list_versions("no-module") == []

    def test_get_reference(self):
        """get_reference() returns reference info."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = CASStorage(Path(tmpdir))

            content_hash = storage.store(b"content", "mod", "v1")

            ref = storage.get_reference("mod", "v1")
            assert ref is not None
            assert ref.module_id == "mod"
            assert ref.version == "v1"
            assert ref.content_hash == content_hash

            # Non-existent
            assert storage.get_reference("no-mod", "v1") is None

    def test_update_reference_to_different_content(self):
        """Updating reference to different content adjusts ref counts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = CASStorage(Path(tmpdir))

            content1 = b"Original content"
            content2 = b"Updated content"

            hash1 = storage.store(content1, "mod", "v1")
            entry1 = storage.get_entry(hash1)
            assert entry1.reference_count == 1

            # Update to different content
            hash2 = storage.store(content2, "mod", "v1")

            # Old content ref count should decrease
            entry1 = storage.get_entry(hash1)
            assert entry1.reference_count == 0

            # New content ref count should be 1
            entry2 = storage.get_entry(hash2)
            assert entry2.reference_count == 1


# =============================================================================
# CASStorage Delete and GC Tests
# =============================================================================


class TestCASStorageDeleteAndGC:
    """Tests for reference deletion and garbage collection."""

    def test_delete_ref(self):
        """delete_ref() removes reference and decrements ref count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = CASStorage(Path(tmpdir))

            content = b"Content to delete"
            content_hash = storage.store(content, "mod", "v1")

            # Verify exists
            assert storage.exists("mod", "v1")
            assert storage.get_entry(content_hash).reference_count == 1

            # Delete reference
            result = storage.delete_ref("mod", "v1")
            assert result is True

            # Reference gone
            assert not storage.exists("mod", "v1")
            assert storage.get_entry(content_hash).reference_count == 0

    def test_delete_ref_nonexistent(self):
        """delete_ref() returns False for non-existent reference."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = CASStorage(Path(tmpdir))

            result = storage.delete_ref("no-mod", "v1")
            assert result is False

    def test_garbage_collect(self):
        """garbage_collect() removes unreferenced objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = CASStorage(Path(tmpdir))

            content = b"Content for GC"
            content_hash = storage.store(content, "mod", "v1")

            # Delete reference
            storage.delete_ref("mod", "v1")

            # Content still exists but ref count is 0
            assert storage.exists_hash(content_hash)

            # Run garbage collection
            deleted = storage.garbage_collect()
            assert deleted == 1

            # Content no longer exists
            assert not storage.exists_hash(content_hash)

    def test_gc_preserves_referenced_content(self):
        """garbage_collect() preserves content with references."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = CASStorage(Path(tmpdir))

            content = b"Shared content"
            content_hash = storage.store(content, "mod1", "v1")
            storage.store(content, "mod2", "v1")

            # Delete one reference
            storage.delete_ref("mod1", "v1")

            # GC should not delete (still has one reference)
            deleted = storage.garbage_collect()
            assert deleted == 0
            assert storage.exists_hash(content_hash)

    def test_delete_module_directory_when_empty(self):
        """Deleting last version of module removes module directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = CASStorage(Path(tmpdir))

            storage.store(b"c1", "mod", "v1")

            module_dir = storage.refs_dir / "mod"
            assert module_dir.exists()

            storage.delete_ref("mod", "v1")

            # Module directory should be removed
            assert not module_dir.exists()


# =============================================================================
# CASStorage Integrity Tests
# =============================================================================


class TestCASStorageIntegrity:
    """Tests for content integrity verification."""

    def test_verify_integrity_valid(self):
        """verify_integrity() returns True for valid content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = CASStorage(Path(tmpdir))

            content = b"Valid content"
            content_hash = storage.store(content, "mod", "v1")

            assert storage.verify_integrity(content_hash) is True

    def test_verify_integrity_corrupted(self):
        """verify_integrity() returns False for corrupted content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = CASStorage(Path(tmpdir))

            content = b"Original content"
            content_hash = storage.store(content, "mod", "v1")

            # Corrupt the file
            obj_path = storage._object_path(content_hash)
            with open(obj_path, "wb") as f:
                f.write(b"Corrupted!")

            assert storage.verify_integrity(content_hash) is False

    def test_verify_integrity_missing(self):
        """verify_integrity() returns False for missing content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = CASStorage(Path(tmpdir))

            assert storage.verify_integrity("nonexistent123") is False

    def test_verify_all(self):
        """verify_all() checks all objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = CASStorage(Path(tmpdir))

            hash1 = storage.store(b"Content 1", "mod1", "v1")
            hash2 = storage.store(b"Content 2", "mod2", "v1")

            # All valid
            results = storage.verify_all()
            assert results[hash1] is True
            assert results[hash2] is True

            # Corrupt one
            obj_path = storage._object_path(hash2)
            with open(obj_path, "wb") as f:
                f.write(b"Bad!")

            results = storage.verify_all()
            assert results[hash1] is True
            assert results[hash2] is False


# =============================================================================
# CASStorage Statistics Tests
# =============================================================================


class TestCASStorageStats:
    """Tests for storage statistics."""

    def test_get_stats_empty(self):
        """get_stats() returns zeros for empty storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = CASStorage(Path(tmpdir))

            stats = storage.get_stats()
            assert stats.unique_objects == 0
            assert stats.total_references == 0
            assert stats.total_size_bytes == 0
            assert stats.deduplication_ratio == 1.0

    def test_get_stats_with_content(self):
        """get_stats() returns correct values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = CASStorage(Path(tmpdir))

            storage.store(b"A" * 100, "mod1", "v1")
            storage.store(b"A" * 100, "mod2", "v1")  # Duplicate
            storage.store(b"B" * 2000, "mod3", "v1")  # Different

            stats = storage.get_stats()
            assert stats.unique_objects == 2
            assert stats.total_references == 3
            assert stats.total_size_bytes == 2100
            assert stats.deduplication_ratio == 1.5

    def test_stats_size_distribution(self):
        """get_stats() categorizes objects by size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = CASStorage(Path(tmpdir))

            storage.store(b"A" * 500, "mod1", "v1")  # < 1KB
            storage.store(b"B" * 5000, "mod2", "v1")  # 1KB - 1MB
            storage.store(b"C" * 2000000, "mod3", "v1")  # 1MB - 10MB

            stats = storage.get_stats()
            assert "< 1KB" in stats.objects_by_size
            assert "1KB - 1MB" in stats.objects_by_size
            assert "1MB - 10MB" in stats.objects_by_size


# =============================================================================
# CASStorage Persistence Tests
# =============================================================================


class TestCASStoragePersistence:
    """Tests for storage persistence and recovery."""

    def test_scan_existing_on_init(self):
        """CASStorage scans existing storage on initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create first storage, add content
            storage1 = CASStorage(Path(tmpdir))
            storage1.store(b"Persistent content", "mod", "v1")

            # Create new storage instance on same directory
            storage2 = CASStorage(Path(tmpdir))

            # Content should be available
            content = storage2.get("mod", "v1")
            assert content == b"Persistent content"

    def test_scan_rebuilds_reference_counts(self):
        """Scanning existing storage rebuilds reference counts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage1 = CASStorage(Path(tmpdir))
            content = b"Shared content"
            storage1.store(content, "mod1", "v1")
            storage1.store(content, "mod2", "v1")

            # New instance should have correct ref counts
            storage2 = CASStorage(Path(tmpdir))
            content_hash = storage2.get_hash("mod1", "v1")
            entry = storage2.get_entry(content_hash)

            assert entry.reference_count == 2


# =============================================================================
# CASStorage Thread Safety Tests
# =============================================================================


class TestCASStorageThreadSafety:
    """Tests for thread-safe operations."""

    def test_concurrent_stores(self):
        """Concurrent stores don't corrupt data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = CASStorage(Path(tmpdir))
            errors = []

            def store_content(i):
                try:
                    content = f"Content {i}".encode()
                    storage.store(content, f"mod{i}", "v1")
                except Exception as e:
                    errors.append(e)

            # Run concurrent stores
            threads = [threading.Thread(target=store_content, args=(i,)) for i in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0

            # Verify all content stored correctly
            for i in range(20):
                content = storage.get(f"mod{i}", "v1")
                assert content == f"Content {i}".encode()

    def test_concurrent_reads(self):
        """Concurrent reads don't fail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = CASStorage(Path(tmpdir))
            storage.store(b"Shared content", "mod", "v1")
            errors = []
            results = []

            def read_content():
                try:
                    content = storage.get("mod", "v1")
                    results.append(content)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=read_content) for _ in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0
            assert len(results) == 20
            assert all(r == b"Shared content" for r in results)


# =============================================================================
# CASStorage Edge Cases Tests
# =============================================================================


class TestCASStorageEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_content(self):
        """Empty content is stored correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = CASStorage(Path(tmpdir))

            content_hash = storage.store(b"", "mod", "v1")
            retrieved = storage.get("mod", "v1")

            assert retrieved == b""
            expected_hash = hashlib.sha256(b"").hexdigest()
            assert content_hash == expected_hash

    def test_large_content(self):
        """Large content (>100KB) is handled correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = CASStorage(Path(tmpdir))

            # 100KB of content (reduced for faster tests)
            content = b"X" * (100 * 1024)
            content_hash = storage.store(content, "large-mod", "v1")

            retrieved = storage.get("large-mod", "v1")
            assert retrieved == content
            assert len(retrieved) == 100 * 1024

    def test_special_characters_in_module_id(self):
        """Module IDs with underscores and dashes work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = CASStorage(Path(tmpdir))

            storage.store(b"content", "my-module_v2", "v1.0.0-beta")

            content = storage.get("my-module_v2", "v1.0.0-beta")
            assert content == b"content"

    def test_hash_prefix_directory_structure(self):
        """Objects are stored with hash prefix directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = CASStorage(Path(tmpdir), hash_prefix_length=2)

            content = b"Test content"
            content_hash = storage.store(content, "mod", "v1")

            # Check directory structure
            prefix = content_hash[:2]
            suffix = content_hash[2:]
            obj_path = storage.objects_dir / prefix / suffix

            assert obj_path.exists()
            assert obj_path.read_bytes() == content

    def test_custom_hash_prefix_length(self):
        """Custom hash prefix length works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = CASStorage(Path(tmpdir), hash_prefix_length=4)

            content = b"Test"
            content_hash = storage.store(content, "mod", "v1")

            prefix = content_hash[:4]
            suffix = content_hash[4:]
            obj_path = storage.objects_dir / prefix / suffix

            assert obj_path.exists()

    def test_to_dict(self):
        """to_dict() exports full storage state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = CASStorage(Path(tmpdir))

            storage.store(b"c1", "mod1", "v1.0.0")
            storage.store(b"c2", "mod1", "v2.0.0")
            storage.store(b"c3", "mod2", "v1.0.0")

            d = storage.to_dict()

            assert "storage_dir" in d
            assert "stats" in d
            assert "modules" in d
            assert "mod1" in d["modules"]
            assert "v1.0.0" in d["modules"]["mod1"]
            assert "v2.0.0" in d["modules"]["mod1"]
            assert "mod2" in d["modules"]


# =============================================================================
# CASStorage Access Time Tests
# =============================================================================


class TestCASStorageAccessTime:
    """Tests for access time tracking."""

    def test_last_accessed_updated_on_get(self):
        """last_accessed is updated when content is retrieved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = CASStorage(Path(tmpdir))

            content_hash = storage.store(b"content", "mod", "v1")
            entry = storage.get_entry(content_hash)
            initial_time = entry.last_accessed

            # Wait a bit and access
            time.sleep(0.1)
            storage.get_by_hash(content_hash)

            entry = storage.get_entry(content_hash)
            assert entry.last_accessed > initial_time

    def test_last_accessed_updated_on_store(self):
        """last_accessed is updated when duplicate is stored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = CASStorage(Path(tmpdir))

            content = b"shared"
            content_hash = storage.store(content, "mod1", "v1")
            entry = storage.get_entry(content_hash)
            initial_time = entry.last_accessed

            # Wait and store duplicate
            time.sleep(0.1)
            storage.store(content, "mod2", "v1")

            entry = storage.get_entry(content_hash)
            assert entry.last_accessed > initial_time
