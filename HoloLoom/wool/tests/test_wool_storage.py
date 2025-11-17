"""
Wool Storage Integration Tests
===============================

Tests for WoolStorage, TextReference, ZeroCopyMemoryShard, and HybridKG.

Author: Claude Code
Date: November 17, 2025
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import List

from HoloLoom.Documentation.types import MemoryShard
from HoloLoom.wool.storage import WoolStorage, WoolReference
from HoloLoom.wool.text_reference import TextReference
from HoloLoom.wool.zerocopy_shard import ZeroCopyMemoryShard, convert_memory_shard_to_zerocopy
from HoloLoom.wool.hybrid_kg import HybridKG, HybridKGStats


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def wool_storage(temp_dir):
    """Create WoolStorage instance."""
    storage = WoolStorage(base_path=temp_dir / "wool")
    yield storage
    storage.close()


@pytest.fixture
def hybrid_kg(wool_storage):
    """Create HybridKG instance."""
    return HybridKG(wool_storage=wool_storage)


# ============================================================================
# Test: WoolStorage Basic Operations
# ============================================================================

class TestWoolStorage:
    """Test WoolStorage basic operations."""

    def test_store_and_read(self, wool_storage):
        """Test basic store and read operations."""
        # Store data
        data = b"Hello, World!"
        ref = wool_storage.store(data, content_type='text/plain')

        assert ref.file_id is not None
        assert len(ref.file_id) == 64  # SHA-256 hash
        assert ref.offset == 0
        assert ref.length == len(data)

        # Read data
        view = wool_storage.read(ref)
        assert view.tobytes() == data

    def test_deduplication(self, wool_storage):
        """Test automatic deduplication."""
        data = b"Duplicate data"

        # Store same data twice
        ref1 = wool_storage.store(data)
        ref2 = wool_storage.store(data)

        # Should have same file_id
        assert ref1.file_id == ref2.file_id

        # Check stats
        stats = wool_storage.get_stats()
        assert stats['files_stored'] == 1
        assert stats['deduplications'] == 1
        assert stats['dedup_rate'] == 0.5  # 1 dedup / 2 total

    def test_read_text(self, wool_storage):
        """Test text reading with encoding."""
        text = "Hello, Unicode! 你好 🌍"
        data = text.encode('utf-8')
        ref = wool_storage.store(data, content_type='text/plain')

        # Read as text
        decoded = wool_storage.read_text(ref, encoding='utf-8')
        assert decoded == text

    def test_file_path_sharding(self, wool_storage):
        """Test directory sharding for filesystem performance."""
        data = b"Test data"
        ref = wool_storage.store(data)

        # Path should be sharded: [first 3]/[next 3]/[full hash]
        path = wool_storage.get_path(ref.file_id)
        parts = path.parts

        assert parts[-3] == ref.file_id[:3]  # First 3 chars
        assert parts[-2] == ref.file_id[3:6]  # Next 3 chars
        assert parts[-1] == ref.file_id  # Full hash

    def test_cache_performance(self, wool_storage):
        """Test mmap cache performance."""
        data = b"Cache test data"
        ref = wool_storage.store(data)

        # Clear cache
        wool_storage.clear_cache()

        # Cold read
        _ = wool_storage.read(ref)
        stats_cold = wool_storage.get_stats()

        # Warm read
        _ = wool_storage.read(ref)
        stats_warm = wool_storage.get_stats()

        # Cache should have hit
        assert stats_warm['cache_hits'] > stats_cold['cache_hits']

    def test_exists(self, wool_storage):
        """Test file existence checking."""
        data = b"Test"
        ref = wool_storage.store(data)

        assert wool_storage.exists(ref.file_id)
        assert not wool_storage.exists("nonexistent" + "0" * 56)

    def test_get_size(self, wool_storage):
        """Test file size retrieval."""
        data = b"Size test data"
        ref = wool_storage.store(data)

        size = wool_storage.get_size(ref.file_id)
        assert size == len(data)


# ============================================================================
# Test: TextReference
# ============================================================================

class TestTextReference:
    """Test TextReference functionality."""

    def test_create_and_serialize(self):
        """Test TextReference creation and serialization."""
        ref = TextReference(
            file_id="a" * 64,
            offset=100,
            length=500,
            encoding='utf-8'
        )

        # Serialize
        data = ref.to_dict()
        assert data['file_id'] == "a" * 64
        assert data['offset'] == 100
        assert data['length'] == 500
        assert data['encoding'] == 'utf-8'

        # Deserialize
        ref2 = TextReference.from_dict(data)
        assert ref2.file_id == ref.file_id
        assert ref2.offset == ref.offset
        assert ref2.length == ref.length

    def test_validation(self):
        """Test field validation."""
        # Invalid file_id length
        with pytest.raises(ValueError):
            TextReference(file_id="short", offset=0, length=10)

        # Invalid offset
        with pytest.raises(ValueError):
            TextReference(file_id="a" * 64, offset=-1, length=10)

        # Invalid length
        with pytest.raises(ValueError):
            TextReference(file_id="a" * 64, offset=0, length=-1)

    def test_resolve(self, wool_storage):
        """Test text resolution."""
        text = "Hello, TextReference!"
        data = text.encode('utf-8')
        wool_ref = wool_storage.store(data)

        # Create TextReference
        text_ref = TextReference(
            file_id=wool_ref.file_id,
            offset=0,
            length=len(data)
        )

        # Resolve
        resolved = text_ref.resolve(wool_storage)
        assert resolved == text

    def test_sizeof(self):
        """Test memory footprint reporting."""
        ref = TextReference(file_id="a" * 64, offset=0, length=100)
        assert ref.__sizeof__() == 88  # Expected size


# ============================================================================
# Test: ZeroCopyMemoryShard
# ============================================================================

class TestZeroCopyMemoryShard:
    """Test ZeroCopyMemoryShard functionality."""

    def test_create_and_get_text(self, wool_storage):
        """Test shard creation and text retrieval."""
        text = "Zero-copy shard text"
        data = text.encode('utf-8')
        wool_ref = wool_storage.store(data)

        text_ref = TextReference(
            file_id=wool_ref.file_id,
            offset=0,
            length=len(data)
        )

        shard = ZeroCopyMemoryShard(
            id="test_shard",
            episode="ep_001",
            text_ref=text_ref,
            entities=["entity1", "entity2"],
            motifs=["motif1"],
            metadata={"source": "test"}
        )

        # Get text (lazy)
        retrieved = shard.get_text(wool_storage)
        assert retrieved == text

        # Second call should use cache
        retrieved2 = shard.get_text(wool_storage)
        assert retrieved2 == text

    def test_conversion_from_legacy(self, wool_storage):
        """Test conversion from legacy MemoryShard."""
        legacy = MemoryShard(
            id="legacy_shard",
            episode="ep_002",
            text="Legacy shard text",
            entities=["ent1"],
            motifs=["mot1"],
            metadata={"type": "legacy"}
        )

        # Convert
        zerocopy = convert_memory_shard_to_zerocopy(legacy, wool_storage)

        assert zerocopy.id == legacy.id
        assert zerocopy.episode == legacy.episode
        assert zerocopy.entities == legacy.entities
        assert zerocopy.motifs == legacy.motifs
        assert zerocopy.metadata == legacy.metadata

        # Text should match
        assert zerocopy.get_text(wool_storage) == legacy.text


# ============================================================================
# Test: HybridKG
# ============================================================================

class TestHybridKG:
    """Test HybridKG functionality."""

    def test_add_legacy_shard(self, hybrid_kg):
        """Test adding legacy MemoryShard."""
        shard = MemoryShard(
            id="legacy_1",
            episode="ep_001",
            text="Legacy text",
            entities=["ent"],
            motifs=["mot"]
        )

        hybrid_kg.add_shard(shard)

        # Check stats
        stats = hybrid_kg.get_stats()
        assert stats.legacy_nodes == 1
        assert stats.zerocopy_nodes == 0
        assert stats.total_nodes == 1

        # Retrieve text
        text = hybrid_kg.get_text("legacy_1")
        assert text == "Legacy text"

    def test_add_zerocopy_shard(self, hybrid_kg, wool_storage):
        """Test adding ZeroCopyMemoryShard."""
        text = "Zero-copy text"
        data = text.encode('utf-8')
        wool_ref = wool_storage.store(data)

        text_ref = TextReference(
            file_id=wool_ref.file_id,
            offset=0,
            length=len(data)
        )

        shard = ZeroCopyMemoryShard(
            id="zerocopy_1",
            episode="ep_002",
            text_ref=text_ref,
            entities=["ent"],
            motifs=["mot"]
        )

        hybrid_kg.add_shard(shard)

        # Check stats
        stats = hybrid_kg.get_stats()
        assert stats.legacy_nodes == 0
        assert stats.zerocopy_nodes == 1
        assert stats.total_nodes == 1

        # Retrieve text
        text = hybrid_kg.get_text("zerocopy_1")
        assert text == "Zero-copy text"

    def test_auto_convert(self, hybrid_kg):
        """Test auto-conversion from legacy to zero-copy."""
        shard = MemoryShard(
            id="convert_1",
            episode="ep_003",
            text="Convert me!",
            entities=["ent"],
            motifs=["mot"]
        )

        # Add with auto_convert=True
        hybrid_kg.add_shard(shard, auto_convert=True)

        # Should be zero-copy
        stats = hybrid_kg.get_stats()
        assert stats.legacy_nodes == 0
        assert stats.zerocopy_nodes == 1

        # Text should still work
        text = hybrid_kg.get_text("convert_1")
        assert text == "Convert me!"

    def test_migrate_node(self, hybrid_kg):
        """Test single node migration."""
        # Add legacy node
        shard = MemoryShard(
            id="migrate_1",
            episode="ep_004",
            text="Migrate this text",
            entities=["ent"],
            motifs=["mot"]
        )
        hybrid_kg.add_shard(shard)

        # Migrate
        result = hybrid_kg.migrate_node("migrate_1")
        assert result is True

        # Check stats
        stats = hybrid_kg.get_stats()
        assert stats.legacy_nodes == 0
        assert stats.zerocopy_nodes == 1
        assert stats.migrated_nodes == 1

        # Text should still work
        text = hybrid_kg.get_text("migrate_1")
        assert text == "Migrate this text"

    def test_migrate_all(self, hybrid_kg):
        """Test batch migration."""
        # Add 10 legacy nodes
        for i in range(10):
            shard = MemoryShard(
                id=f"batch_{i}",
                episode="ep_005",
                text=f"Text {i}",
                entities=["ent"],
                motifs=["mot"]
            )
            hybrid_kg.add_shard(shard)

        # Migrate all
        migrated = hybrid_kg.migrate_all(batch_size=5)
        assert migrated == 10

        # Check stats
        stats = hybrid_kg.get_stats()
        assert stats.legacy_nodes == 0
        assert stats.zerocopy_nodes == 10
        assert stats.migrated_nodes == 10

        # All text should still work
        for i in range(10):
            text = hybrid_kg.get_text(f"batch_{i}")
            assert text == f"Text {i}"

    def test_mixed_nodes(self, hybrid_kg, wool_storage):
        """Test hybrid graph with both legacy and zero-copy nodes."""
        # Add legacy
        legacy = MemoryShard(
            id="mixed_legacy",
            episode="ep_006",
            text="Legacy in hybrid",
            entities=["ent"],
            motifs=["mot"]
        )
        hybrid_kg.add_shard(legacy)

        # Add zero-copy
        text = "Zero-copy in hybrid"
        data = text.encode('utf-8')
        wool_ref = wool_storage.store(data)
        text_ref = TextReference(
            file_id=wool_ref.file_id,
            offset=0,
            length=len(data)
        )
        zerocopy = ZeroCopyMemoryShard(
            id="mixed_zerocopy",
            episode="ep_006",
            text_ref=text_ref,
            entities=["ent"],
            motifs=["mot"]
        )
        hybrid_kg.add_shard(zerocopy)

        # Check stats
        stats = hybrid_kg.get_stats()
        assert stats.legacy_nodes == 1
        assert stats.zerocopy_nodes == 1
        assert stats.total_nodes == 2

        # Both should work
        assert hybrid_kg.get_text("mixed_legacy") == "Legacy in hybrid"
        assert hybrid_kg.get_text("mixed_zerocopy") == "Zero-copy in hybrid"

    def test_search(self, hybrid_kg):
        """Test search functionality."""
        # Add nodes
        for i in range(5):
            shard = MemoryShard(
                id=f"search_{i}",
                episode="ep_007",
                text=f"Text {i}",
                entities=["search_entity", f"ent_{i}"],
                motifs=["mot"]
            )
            hybrid_kg.add_shard(shard)

        # Search
        results = hybrid_kg.search("search_entity", limit=3)
        assert len(results) == 3
        assert all('search_entity' in r['entities'] for r in results)

    def test_statistics(self, hybrid_kg):
        """Test statistics tracking."""
        # Add legacy nodes
        for i in range(3):
            shard = MemoryShard(
                id=f"stats_{i}",
                episode="ep_008",
                text="A" * 1000,  # 1KB text
                entities=["ent"],
                motifs=["mot"]
            )
            hybrid_kg.add_shard(shard)

        stats = hybrid_kg.get_stats()
        assert stats.legacy_nodes == 3
        assert stats.zerocopy_nodes == 0
        assert stats.legacy_bytes == 3000  # 3 * 1KB
        assert stats.zerocopy_percentage == 0.0

        # Migrate
        hybrid_kg.migrate_all()

        stats = hybrid_kg.get_stats()
        assert stats.legacy_nodes == 0
        assert stats.zerocopy_nodes == 3
        assert stats.zerocopy_percentage == 100.0
        assert stats.memory_savings_mb > 0


# ============================================================================
# Test: Thread Safety
# ============================================================================

class TestThreadSafety:
    """Test thread safety of WoolStorage."""

    def test_concurrent_reads(self, wool_storage):
        """Test concurrent reads from cache."""
        import threading

        data = b"Concurrent read test"
        ref = wool_storage.store(data)

        # Concurrent reads
        results = []
        errors = []

        def read_worker():
            try:
                view = wool_storage.read(ref)
                results.append(view.tobytes())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should succeed
        assert len(errors) == 0
        assert len(results) == 10
        assert all(r == data for r in results)


# ============================================================================
# Test: Error Handling
# ============================================================================

class TestErrorHandling:
    """Test error handling."""

    def test_read_nonexistent_file(self, wool_storage):
        """Test reading non-existent file."""
        ref = WoolReference(
            file_id="0" * 64,
            offset=0,
            length=100
        )

        with pytest.raises(FileNotFoundError):
            wool_storage.read(ref)

    def test_invalid_range(self, wool_storage):
        """Test invalid read range."""
        data = b"Short data"
        ref = wool_storage.store(data)

        # Try to read beyond file
        invalid_ref = WoolReference(
            file_id=ref.file_id,
            offset=0,
            length=1000  # Larger than file
        )

        with pytest.raises(ValueError):
            wool_storage.read(invalid_ref)

    def test_get_nonexistent_node(self, hybrid_kg):
        """Test getting non-existent node."""
        text = hybrid_kg.get_text("nonexistent")
        assert text is None

    def test_migrate_nonexistent_node(self, hybrid_kg):
        """Test migrating non-existent node."""
        result = hybrid_kg.migrate_node("nonexistent")
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
