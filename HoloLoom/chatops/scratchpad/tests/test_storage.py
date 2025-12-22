"""
ChatOps Scratch Pad - Storage Layer Tests
==========================================
Tests for ContentStorage (CAS), MetadataStorage (SQLite),
AuditLogStorage, and ScratchPadStorage.

Created: December 2025
"""

import pytest
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

from ..storage import (
    ContentStorage,
    CASEntry,
    MetadataStorage,
    AuditLogStorage,
    ScratchPadStorage,
)
from ..types import (
    ScratchArtifact,
    ArtifactReference,
    ArtifactScope,
    ArtifactType,
    ArtifactStatus,
    ScratchPadConfig,
    AuditEvent,
    AuditEventType,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def content_storage(temp_dir):
    """Create ContentStorage instance."""
    return ContentStorage(temp_dir / "content")


@pytest.fixture
def metadata_storage(temp_dir):
    """Create MetadataStorage instance."""
    return MetadataStorage(temp_dir / "metadata.db")


@pytest.fixture
def audit_storage(temp_dir):
    """Create AuditLogStorage instance."""
    return AuditLogStorage(temp_dir / "audit.jsonl")


@pytest.fixture
def test_config(temp_dir):
    """Create test configuration."""
    return ScratchPadConfig(
        metadata_db_path=str(temp_dir / "metadata.db"),
        cas_storage_path=str(temp_dir / "content"),
        audit_log_path=str(temp_dir / "audit.jsonl"),
        max_artifact_size_bytes=1024 * 1024,  # 1MB
        max_artifacts_per_user=10,
        default_ttl_days=7,
        enable_audit=True,
    )


@pytest.fixture
def scratchpad_storage(test_config):
    """Create ScratchPadStorage instance."""
    return ScratchPadStorage(test_config)


@pytest.fixture
def test_user_id():
    return "@testuser:example.com"


@pytest.fixture
def test_room_id():
    return "!testroom:example.com"


# ============================================================================
# ContentStorage Tests
# ============================================================================

class TestContentStorageStore:
    """Tests for ContentStorage.store()."""

    def test_store_new_content(self, content_storage):
        """Should store new content and return hash."""
        content = b"Hello, World!"
        content_hash, deduplicated = content_storage.store(content)

        assert content_hash is not None
        assert len(content_hash) == 64  # SHA256 hex
        assert deduplicated is False

    def test_store_duplicate_content(self, content_storage):
        """Should detect duplicate content."""
        content = b"Hello, World!"
        hash1, dedup1 = content_storage.store(content)
        hash2, dedup2 = content_storage.store(content)

        assert hash1 == hash2
        assert dedup1 is False
        assert dedup2 is True  # Second store is deduplicated

    def test_store_different_content(self, content_storage):
        """Should store different content separately."""
        hash1, _ = content_storage.store(b"Content A")
        hash2, _ = content_storage.store(b"Content B")

        assert hash1 != hash2

    def test_store_creates_sharded_directory(self, content_storage):
        """Should create sharded directory structure."""
        content = b"Test content"
        content_hash, _ = content_storage.store(content)

        # Check directory structure
        prefix = content_hash[:2]
        suffix = content_hash[2:]
        expected_path = content_storage.storage_dir / prefix / suffix
        assert expected_path.exists()


class TestContentStorageGet:
    """Tests for ContentStorage.get()."""

    def test_get_existing_content(self, content_storage):
        """Should retrieve stored content."""
        original = b"Test content"
        content_hash, _ = content_storage.store(original)

        retrieved = content_storage.get(content_hash)
        assert retrieved == original

    def test_get_nonexistent_content(self, content_storage):
        """Should return None for nonexistent content."""
        fake_hash = "0" * 64
        result = content_storage.get(fake_hash)
        assert result is None

    def test_get_updates_access_time(self, content_storage):
        """Should update last_accessed on get."""
        content = b"Test content"
        content_hash, _ = content_storage.store(content)

        # Get initial access time
        initial_access = content_storage._entries[content_hash].last_accessed

        time.sleep(0.01)  # Small delay
        content_storage.get(content_hash)

        final_access = content_storage._entries[content_hash].last_accessed
        assert final_access > initial_access


class TestContentStorageExists:
    """Tests for ContentStorage.exists()."""

    def test_exists_true(self, content_storage):
        """Should return True for existing content."""
        content_hash, _ = content_storage.store(b"Test")
        assert content_storage.exists(content_hash) is True

    def test_exists_false(self, content_storage):
        """Should return False for nonexistent content."""
        assert content_storage.exists("0" * 64) is False


class TestContentStorageReferences:
    """Tests for reference counting."""

    def test_add_reference(self, content_storage):
        """Should increment reference count."""
        content_hash, _ = content_storage.store(b"Test")
        initial_count = content_storage._entries[content_hash].reference_count

        result = content_storage.add_reference(content_hash)

        assert result is True
        assert content_storage._entries[content_hash].reference_count == initial_count + 1

    def test_add_reference_nonexistent(self, content_storage):
        """Should return False for nonexistent content."""
        result = content_storage.add_reference("0" * 64)
        assert result is False

    def test_remove_reference(self, content_storage):
        """Should decrement reference count."""
        content_hash, _ = content_storage.store(b"Test")
        initial_count = content_storage._entries[content_hash].reference_count

        new_count = content_storage.remove_reference(content_hash)

        assert new_count == initial_count - 1

    def test_remove_reference_nonexistent(self, content_storage):
        """Should return -1 for nonexistent content."""
        result = content_storage.remove_reference("0" * 64)
        assert result == -1


class TestContentStorageGarbageCollect:
    """Tests for garbage collection."""

    def test_garbage_collect_removes_unreferenced(self, content_storage):
        """Should remove content with zero references."""
        content_hash, _ = content_storage.store(b"Test")

        # Remove reference to make it eligible for GC
        content_storage.remove_reference(content_hash)

        deleted = content_storage.garbage_collect()

        assert deleted == 1
        assert content_hash not in content_storage._entries

    def test_garbage_collect_keeps_referenced(self, content_storage):
        """Should keep content with references."""
        content_hash, _ = content_storage.store(b"Test")

        deleted = content_storage.garbage_collect()

        assert deleted == 0
        assert content_hash in content_storage._entries


class TestContentStorageStats:
    """Tests for storage statistics."""

    def test_get_stats_empty(self, content_storage):
        """Should return stats for empty storage."""
        stats = content_storage.get_stats()

        assert stats['unique_objects'] == 0
        assert stats['total_references'] == 0
        assert stats['total_size_bytes'] == 0
        assert stats['deduplication_ratio'] == 1.0

    def test_get_stats_with_content(self, content_storage):
        """Should return accurate stats."""
        content_storage.store(b"Content A")
        content_storage.store(b"Content B")
        content_storage.store(b"Content A")  # Duplicate

        stats = content_storage.get_stats()

        assert stats['unique_objects'] == 2
        assert stats['total_references'] == 3
        assert stats['total_size_bytes'] == len(b"Content A") + len(b"Content B")


class TestContentStorageIntegrity:
    """Tests for integrity verification."""

    def test_verify_integrity_valid(self, content_storage):
        """Should return True for valid content."""
        content_hash, _ = content_storage.store(b"Test content")
        assert content_storage.verify_integrity(content_hash) is True

    def test_verify_integrity_nonexistent(self, content_storage):
        """Should return False for nonexistent content."""
        assert content_storage.verify_integrity("0" * 64) is False


# ============================================================================
# MetadataStorage Tests
# ============================================================================

class TestMetadataStorageStore:
    """Tests for MetadataStorage.store()."""

    def test_store_artifact(self, metadata_storage, test_user_id, test_room_id):
        """Should store artifact metadata."""
        artifact = ScratchArtifact(
            artifact_id=ScratchArtifact.generate_id(),
            name="test_artifact",
            content_hash="a" * 64,
            artifact_type=ArtifactType.TEXT,
            size_bytes=100,
            owner_user_id=test_user_id,
            owner_room_id=test_room_id,
        )

        result = metadata_storage.store(artifact)
        assert result is True

    def test_store_and_retrieve(self, metadata_storage, test_user_id, test_room_id):
        """Should store and retrieve artifact."""
        artifact = ScratchArtifact(
            artifact_id=ScratchArtifact.generate_id(),
            name="test_artifact",
            content_hash="a" * 64,
            artifact_type=ArtifactType.CODE,
            size_bytes=200,
            owner_user_id=test_user_id,
            owner_room_id=test_room_id,
            metadata={'language': 'python'},
        )

        metadata_storage.store(artifact)
        retrieved = metadata_storage.get(artifact.artifact_id)

        assert retrieved is not None
        assert retrieved.name == "test_artifact"
        assert retrieved.artifact_type == ArtifactType.CODE
        assert retrieved.metadata == {'language': 'python'}


class TestMetadataStorageGet:
    """Tests for MetadataStorage.get()."""

    def test_get_nonexistent(self, metadata_storage):
        """Should return None for nonexistent artifact."""
        result = metadata_storage.get("nonexistent_id")
        assert result is None

    def test_get_by_name(self, metadata_storage, test_user_id, test_room_id):
        """Should get artifact by name."""
        artifact = ScratchArtifact(
            artifact_id=ScratchArtifact.generate_id(),
            name="named_artifact",
            content_hash="b" * 64,
            artifact_type=ArtifactType.TEXT,
            size_bytes=50,
            owner_user_id=test_user_id,
            owner_room_id=test_room_id,
        )
        metadata_storage.store(artifact)

        result = metadata_storage.get_by_name("named_artifact", test_user_id, test_room_id)

        assert result is not None
        assert result.artifact_id == artifact.artifact_id

    def test_get_by_name_wrong_user(self, metadata_storage, test_user_id, test_room_id):
        """Should not find artifact for different user."""
        artifact = ScratchArtifact(
            artifact_id=ScratchArtifact.generate_id(),
            name="private_artifact",
            content_hash="c" * 64,
            artifact_type=ArtifactType.TEXT,
            size_bytes=50,
            owner_user_id=test_user_id,
            owner_room_id=test_room_id,
        )
        metadata_storage.store(artifact)

        result = metadata_storage.get_by_name("private_artifact", "@other:example.com", test_room_id)
        assert result is None


class TestMetadataStorageList:
    """Tests for MetadataStorage listing functions."""

    def test_list_accessible_empty(self, metadata_storage, test_user_id, test_room_id):
        """Should return empty list when no artifacts."""
        result = metadata_storage.list_accessible(test_user_id, test_room_id)
        assert result == []

    def test_list_accessible_user_scope(self, metadata_storage, test_user_id, test_room_id):
        """Should list user's own artifacts."""
        artifact = ScratchArtifact(
            artifact_id=ScratchArtifact.generate_id(),
            name="user_artifact",
            content_hash="d" * 64,
            artifact_type=ArtifactType.TEXT,
            size_bytes=50,
            owner_user_id=test_user_id,
            owner_room_id=test_room_id,
            scope=ArtifactScope.USER,
        )
        metadata_storage.store(artifact)

        result = metadata_storage.list_accessible(
            test_user_id, test_room_id, scope_filter=ArtifactScope.USER
        )

        assert len(result) == 1
        assert result[0].name == "user_artifact"

    def test_list_accessible_room_scope(self, metadata_storage, test_user_id, test_room_id):
        """Should list room-shared artifacts."""
        artifact = ScratchArtifact(
            artifact_id=ScratchArtifact.generate_id(),
            name="room_artifact",
            content_hash="e" * 64,
            artifact_type=ArtifactType.TEXT,
            size_bytes=50,
            owner_user_id=test_user_id,
            owner_room_id=test_room_id,
            scope=ArtifactScope.ROOM,
        )
        metadata_storage.store(artifact)

        # Different user in same room should see it
        result = metadata_storage.list_accessible(
            "@other:example.com", test_room_id, scope_filter=ArtifactScope.ROOM
        )

        assert len(result) == 1
        assert result[0].name == "room_artifact"

    def test_list_by_session(self, metadata_storage, test_user_id, test_room_id):
        """Should list artifacts by session."""
        session_id = "session-123"
        artifact = ScratchArtifact(
            artifact_id=ScratchArtifact.generate_id(),
            name="session_artifact",
            content_hash="f" * 64,
            artifact_type=ArtifactType.TEXT,
            size_bytes=50,
            owner_user_id=test_user_id,
            owner_room_id=test_room_id,
            source_session_id=session_id,
        )
        metadata_storage.store(artifact)

        result = metadata_storage.list_by_session(session_id, test_user_id, test_room_id)

        assert len(result) == 1
        assert result[0].name == "session_artifact"


class TestMetadataStorageUpdate:
    """Tests for MetadataStorage update functions."""

    def test_update_access(self, metadata_storage, test_user_id, test_room_id):
        """Should update access timestamp and count."""
        artifact = ScratchArtifact(
            artifact_id=ScratchArtifact.generate_id(),
            name="access_test",
            content_hash="g" * 64,
            artifact_type=ArtifactType.TEXT,
            size_bytes=50,
            owner_user_id=test_user_id,
            owner_room_id=test_room_id,
            access_count=0,
        )
        metadata_storage.store(artifact)

        result = metadata_storage.update_access(artifact.artifact_id)
        assert result is True

        updated = metadata_storage.get(artifact.artifact_id)
        assert updated.access_count == 1

    def test_update_scope(self, metadata_storage, test_user_id, test_room_id):
        """Should update artifact scope."""
        artifact = ScratchArtifact(
            artifact_id=ScratchArtifact.generate_id(),
            name="scope_test",
            content_hash="h" * 64,
            artifact_type=ArtifactType.TEXT,
            size_bytes=50,
            owner_user_id=test_user_id,
            owner_room_id=test_room_id,
            scope=ArtifactScope.USER,
        )
        metadata_storage.store(artifact)

        result = metadata_storage.update_scope(artifact.artifact_id, ArtifactScope.ROOM)
        assert result is True

        updated = metadata_storage.get(artifact.artifact_id)
        assert updated.scope == ArtifactScope.ROOM

    def test_update_status(self, metadata_storage, test_user_id, test_room_id):
        """Should update artifact status."""
        artifact = ScratchArtifact(
            artifact_id=ScratchArtifact.generate_id(),
            name="status_test",
            content_hash="i" * 64,
            artifact_type=ArtifactType.TEXT,
            size_bytes=50,
            owner_user_id=test_user_id,
            owner_room_id=test_room_id,
        )
        metadata_storage.store(artifact)

        result = metadata_storage.update_status(artifact.artifact_id, ArtifactStatus.ARCHIVED)
        assert result is True

        updated = metadata_storage.get(artifact.artifact_id)
        assert updated.status == ArtifactStatus.ARCHIVED


class TestMetadataStorageDelete:
    """Tests for MetadataStorage.delete()."""

    def test_delete_existing(self, metadata_storage, test_user_id, test_room_id):
        """Should delete existing artifact."""
        content_hash = "j" * 64
        artifact = ScratchArtifact(
            artifact_id=ScratchArtifact.generate_id(),
            name="delete_test",
            content_hash=content_hash,
            artifact_type=ArtifactType.TEXT,
            size_bytes=50,
            owner_user_id=test_user_id,
            owner_room_id=test_room_id,
        )
        metadata_storage.store(artifact)

        result = metadata_storage.delete(artifact.artifact_id)

        assert result == content_hash
        assert metadata_storage.get(artifact.artifact_id) is None

    def test_delete_nonexistent(self, metadata_storage):
        """Should return None for nonexistent artifact."""
        result = metadata_storage.delete("nonexistent_id")
        assert result is None


class TestMetadataStorageQuotas:
    """Tests for quota-related functions."""

    def test_count_by_user(self, metadata_storage, test_user_id, test_room_id):
        """Should count user's artifacts."""
        for i in range(3):
            artifact = ScratchArtifact(
                artifact_id=ScratchArtifact.generate_id(),
                name=f"artifact_{i}",
                content_hash=f"{i}" * 64,
                artifact_type=ArtifactType.TEXT,
                size_bytes=50,
                owner_user_id=test_user_id,
                owner_room_id=test_room_id,
            )
            metadata_storage.store(artifact)

        count = metadata_storage.count_by_user(test_user_id, test_room_id)
        assert count == 3

    def test_total_size_by_user(self, metadata_storage, test_user_id, test_room_id):
        """Should sum user's artifact sizes."""
        sizes = [100, 200, 300]
        for i, size in enumerate(sizes):
            artifact = ScratchArtifact(
                artifact_id=ScratchArtifact.generate_id(),
                name=f"artifact_{i}",
                content_hash=f"{i}" * 64,
                artifact_type=ArtifactType.TEXT,
                size_bytes=size,
                owner_user_id=test_user_id,
                owner_room_id=test_room_id,
            )
            metadata_storage.store(artifact)

        total = metadata_storage.total_size_by_user(test_user_id, test_room_id)
        assert total == sum(sizes)


class TestMetadataStorageExpiration:
    """Tests for expiration functions."""

    def test_find_expired(self, metadata_storage, test_user_id, test_room_id):
        """Should find expired artifacts."""
        # Create expired artifact
        expired_artifact = ScratchArtifact(
            artifact_id=ScratchArtifact.generate_id(),
            name="expired",
            content_hash="k" * 64,
            artifact_type=ArtifactType.TEXT,
            size_bytes=50,
            owner_user_id=test_user_id,
            owner_room_id=test_room_id,
            expires_at=datetime.now() - timedelta(days=1),  # Expired yesterday
        )
        metadata_storage.store(expired_artifact)

        # Create non-expired artifact
        active_artifact = ScratchArtifact(
            artifact_id=ScratchArtifact.generate_id(),
            name="active",
            content_hash="l" * 64,
            artifact_type=ArtifactType.TEXT,
            size_bytes=50,
            owner_user_id=test_user_id,
            owner_room_id=test_room_id,
            expires_at=datetime.now() + timedelta(days=7),  # Expires next week
        )
        metadata_storage.store(active_artifact)

        expired = metadata_storage.find_expired(datetime.now())

        assert len(expired) == 1
        assert expired[0] == expired_artifact.artifact_id


# ============================================================================
# AuditLogStorage Tests
# ============================================================================

class TestAuditLogStorageLog:
    """Tests for AuditLogStorage.log()."""

    def test_log_event(self, audit_storage):
        """Should log audit event."""
        event = AuditEvent(
            event_type=AuditEventType.ARTIFACT_CREATED,
            timestamp=datetime.now(),
            user_id="@user:example.com",
            room_id="!room:example.com",
            artifact_id="artifact-123",
            artifact_name="test_artifact",
            success=True,
        )

        audit_storage.log(event)

        # File should exist
        assert audit_storage.log_path.exists()


class TestAuditLogStorageQuery:
    """Tests for AuditLogStorage.query()."""

    def test_query_empty(self, audit_storage):
        """Should return empty list when no events."""
        result = audit_storage.query()
        assert result == []

    def test_query_all(self, audit_storage):
        """Should return all events."""
        for i in range(3):
            event = AuditEvent(
                event_type=AuditEventType.ARTIFACT_CREATED,
                timestamp=datetime.now(),
                user_id="@user:example.com",
                room_id="!room:example.com",
                artifact_id=f"artifact-{i}",
            )
            audit_storage.log(event)

        result = audit_storage.query()
        assert len(result) == 3

    def test_query_by_event_type(self, audit_storage):
        """Should filter by event type."""
        audit_storage.log(AuditEvent(
            event_type=AuditEventType.ARTIFACT_CREATED,
            timestamp=datetime.now(),
            user_id="@user:example.com",
            room_id="!room:example.com",
        ))
        audit_storage.log(AuditEvent(
            event_type=AuditEventType.ARTIFACT_DELETED,
            timestamp=datetime.now(),
            user_id="@user:example.com",
            room_id="!room:example.com",
        ))

        result = audit_storage.query(event_type=AuditEventType.ARTIFACT_CREATED)

        assert len(result) == 1
        assert result[0].event_type == AuditEventType.ARTIFACT_CREATED

    def test_query_by_user(self, audit_storage):
        """Should filter by user ID."""
        audit_storage.log(AuditEvent(
            event_type=AuditEventType.ARTIFACT_CREATED,
            timestamp=datetime.now(),
            user_id="@user1:example.com",
            room_id="!room:example.com",
        ))
        audit_storage.log(AuditEvent(
            event_type=AuditEventType.ARTIFACT_CREATED,
            timestamp=datetime.now(),
            user_id="@user2:example.com",
            room_id="!room:example.com",
        ))

        result = audit_storage.query(user_id="@user1:example.com")

        assert len(result) == 1
        assert result[0].user_id == "@user1:example.com"

    def test_query_with_limit(self, audit_storage):
        """Should respect limit parameter."""
        for i in range(10):
            audit_storage.log(AuditEvent(
                event_type=AuditEventType.ARTIFACT_ACCESSED,
                timestamp=datetime.now(),
                user_id="@user:example.com",
                room_id="!room:example.com",
            ))

        result = audit_storage.query(limit=5)
        assert len(result) == 5


# ============================================================================
# ScratchPadStorage Tests
# ============================================================================

class TestScratchPadStorageStore:
    """Tests for ScratchPadStorage.store_artifact()."""

    def test_store_artifact(self, scratchpad_storage, test_user_id, test_room_id):
        """Should store artifact successfully."""
        artifact, error = scratchpad_storage.store_artifact(
            name="test_artifact",
            content=b"Test content",
            artifact_type=ArtifactType.TEXT,
            user_id=test_user_id,
            room_id=test_room_id,
        )

        assert artifact is not None
        assert error is None
        assert artifact.name == "test_artifact"

    def test_store_artifact_size_limit(self, scratchpad_storage, test_user_id, test_room_id):
        """Should reject oversized content."""
        # Create content larger than limit
        large_content = b"x" * (scratchpad_storage.config.max_artifact_size_bytes + 1)

        artifact, error = scratchpad_storage.store_artifact(
            name="large_artifact",
            content=large_content,
            artifact_type=ArtifactType.BINARY,
            user_id=test_user_id,
            room_id=test_room_id,
        )

        assert artifact is None
        assert "exceeds maximum size" in error

    def test_store_artifact_quota_limit(self, scratchpad_storage, test_user_id, test_room_id):
        """Should enforce quota limit."""
        # Fill up quota
        for i in range(scratchpad_storage.config.max_artifacts_per_user):
            scratchpad_storage.store_artifact(
                name=f"artifact_{i}",
                content=f"Content {i}".encode(),
                artifact_type=ArtifactType.TEXT,
                user_id=test_user_id,
                room_id=test_room_id,
            )

        # Try to add one more
        artifact, error = scratchpad_storage.store_artifact(
            name="over_quota",
            content=b"Extra content",
            artifact_type=ArtifactType.TEXT,
            user_id=test_user_id,
            room_id=test_room_id,
        )

        assert artifact is None
        assert "Quota exceeded" in error


class TestScratchPadStorageGet:
    """Tests for ScratchPadStorage.get_artifact()."""

    def test_get_artifact(self, scratchpad_storage, test_user_id, test_room_id):
        """Should retrieve artifact with content."""
        original_content = b"Test content"
        artifact, _ = scratchpad_storage.store_artifact(
            name="retrieve_test",
            content=original_content,
            artifact_type=ArtifactType.TEXT,
            user_id=test_user_id,
            room_id=test_room_id,
        )

        retrieved, content, error = scratchpad_storage.get_artifact(
            artifact.artifact_id, test_user_id, test_room_id
        )

        assert retrieved is not None
        assert content == original_content
        assert error is None

    def test_get_artifact_access_denied(self, scratchpad_storage, test_user_id, test_room_id):
        """Should deny access to unauthorized user."""
        artifact, _ = scratchpad_storage.store_artifact(
            name="private_artifact",
            content=b"Private",
            artifact_type=ArtifactType.TEXT,
            user_id=test_user_id,
            room_id=test_room_id,
            scope=ArtifactScope.USER,
        )

        # Different user tries to access
        retrieved, content, error = scratchpad_storage.get_artifact(
            artifact.artifact_id, "@other:example.com", "!otherroom:example.com"
        )

        assert retrieved is None
        assert content is None
        assert error == "Access denied"

    def test_get_by_name(self, scratchpad_storage, test_user_id, test_room_id):
        """Should get artifact by name."""
        original_content = b"Named content"
        artifact, _ = scratchpad_storage.store_artifact(
            name="named_artifact",
            content=original_content,
            artifact_type=ArtifactType.TEXT,
            user_id=test_user_id,
            room_id=test_room_id,
        )

        retrieved, content, error = scratchpad_storage.get_by_name(
            "named_artifact", test_user_id, test_room_id
        )

        assert retrieved is not None
        assert content == original_content


class TestScratchPadStorageList:
    """Tests for ScratchPadStorage listing functions."""

    def test_list_artifacts(self, scratchpad_storage, test_user_id, test_room_id):
        """Should list user's artifacts."""
        for i in range(3):
            scratchpad_storage.store_artifact(
                name=f"artifact_{i}",
                content=f"Content {i}".encode(),
                artifact_type=ArtifactType.TEXT,
                user_id=test_user_id,
                room_id=test_room_id,
            )

        result = scratchpad_storage.list_artifacts(test_user_id, test_room_id)

        assert len(result) == 3

    def test_list_session_artifacts(self, scratchpad_storage, test_user_id, test_room_id):
        """Should list artifacts by session."""
        session_id = "session-456"

        scratchpad_storage.store_artifact(
            name="session_artifact",
            content=b"Session content",
            artifact_type=ArtifactType.TEXT,
            user_id=test_user_id,
            room_id=test_room_id,
            source_session_id=session_id,
        )

        result = scratchpad_storage.list_session_artifacts(
            session_id, test_user_id, test_room_id
        )

        assert len(result) == 1


class TestScratchPadStorageDelete:
    """Tests for ScratchPadStorage.delete_artifact()."""

    def test_delete_artifact(self, scratchpad_storage, test_user_id, test_room_id):
        """Should delete artifact."""
        artifact, _ = scratchpad_storage.store_artifact(
            name="delete_me",
            content=b"Delete content",
            artifact_type=ArtifactType.TEXT,
            user_id=test_user_id,
            room_id=test_room_id,
        )

        success, error = scratchpad_storage.delete_artifact(
            artifact.artifact_id, test_user_id, test_room_id
        )

        assert success is True
        assert error is None

    def test_delete_artifact_permission_denied(self, scratchpad_storage, test_user_id, test_room_id):
        """Should deny delete to non-owner."""
        artifact, _ = scratchpad_storage.store_artifact(
            name="owned_artifact",
            content=b"Owned content",
            artifact_type=ArtifactType.TEXT,
            user_id=test_user_id,
            room_id=test_room_id,
        )

        # Different user tries to delete
        success, error = scratchpad_storage.delete_artifact(
            artifact.artifact_id, "@other:example.com", test_room_id
        )

        assert success is False
        assert error == "Permission denied"


class TestScratchPadStorageShare:
    """Tests for ScratchPadStorage.share_artifact()."""

    def test_share_artifact(self, scratchpad_storage, test_user_id, test_room_id):
        """Should share artifact with room."""
        artifact, _ = scratchpad_storage.store_artifact(
            name="share_me",
            content=b"Shared content",
            artifact_type=ArtifactType.TEXT,
            user_id=test_user_id,
            room_id=test_room_id,
            scope=ArtifactScope.USER,
        )

        success, error = scratchpad_storage.share_artifact(
            artifact.artifact_id, test_user_id, test_room_id
        )

        assert success is True
        assert error is None

        # Verify scope changed
        updated, _, _ = scratchpad_storage.get_artifact(
            artifact.artifact_id, test_user_id, test_room_id
        )
        assert updated.scope == ArtifactScope.ROOM

    def test_share_already_shared(self, scratchpad_storage, test_user_id, test_room_id):
        """Should reject sharing already-shared artifact."""
        artifact, _ = scratchpad_storage.store_artifact(
            name="already_shared",
            content=b"Shared content",
            artifact_type=ArtifactType.TEXT,
            user_id=test_user_id,
            room_id=test_room_id,
            scope=ArtifactScope.ROOM,  # Already room scope
        )

        success, error = scratchpad_storage.share_artifact(
            artifact.artifact_id, test_user_id, test_room_id
        )

        assert success is False
        assert "Already shared" in error


class TestScratchPadStorageCleanup:
    """Tests for cleanup operations."""

    def test_cleanup_expired(self, scratchpad_storage, test_user_id, test_room_id):
        """Should clean up expired artifacts."""
        # Store an artifact that's already expired
        artifact, _ = scratchpad_storage.store_artifact(
            name="expired_artifact",
            content=b"Expired content",
            artifact_type=ArtifactType.TEXT,
            user_id=test_user_id,
            room_id=test_room_id,
        )

        # Manually update expiration to past
        scratchpad_storage.metadata._get_connection().execute("""
            UPDATE artifacts SET expires_at = ? WHERE artifact_id = ?
        """, ((datetime.now() - timedelta(days=1)).isoformat(), artifact.artifact_id))
        scratchpad_storage.metadata._get_connection().commit()

        deleted = scratchpad_storage.cleanup_expired()

        assert deleted == 1


class TestScratchPadStorageStats:
    """Tests for ScratchPadStorage.get_stats()."""

    def test_get_stats(self, scratchpad_storage, test_user_id, test_room_id):
        """Should return storage statistics."""
        scratchpad_storage.store_artifact(
            name="stats_test",
            content=b"Test content",
            artifact_type=ArtifactType.TEXT,
            user_id=test_user_id,
            room_id=test_room_id,
        )

        stats = scratchpad_storage.get_stats()

        assert 'content' in stats
        assert 'config' in stats
        assert stats['content']['unique_objects'] == 1
