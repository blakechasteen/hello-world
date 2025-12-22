"""
ChatOps Scratch Pad - Integration Tests
========================================
End-to-end tests that verify the complete flow from
manager operations through storage and back.

Created: December 2025
"""

import pytest
import tempfile
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from ..types import (
    ScratchArtifact,
    ArtifactScope,
    ArtifactType,
    ArtifactStatus,
    ScratchPadConfig,
    SessionContext,
    AuditEventType,
)
from ..manager import (
    ScratchPadManager,
    create_scratchpad_manager,
    create_and_start_manager,
)
from ..storage import ScratchPadStorage


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def test_config(temp_dir):
    """Create test configuration."""
    return ScratchPadConfig(
        metadata_db_path=str(temp_dir / "metadata.db"),
        cas_storage_path=str(temp_dir / "content"),
        audit_log_path=str(temp_dir / "audit.jsonl"),
        max_artifact_size_bytes=1024 * 1024,  # 1MB
        max_artifacts_per_user=100,
        default_ttl_days=7,
        rate_limit_requests=100,
        rate_limit_window_seconds=60,
        enable_audit=True,
    )


@pytest.fixture
def manager(test_config):
    """Create ScratchPadManager instance."""
    return ScratchPadManager(test_config)


@pytest.fixture
def test_user_id():
    return "@testuser:example.com"


@pytest.fixture
def test_room_id():
    return "!testroom:example.com"


@pytest.fixture
def second_user_id():
    return "@seconduser:example.com"


# ============================================================================
# Store and Retrieve Flow
# ============================================================================

class TestStoreRetrieveFlow:
    """Tests for complete store → retrieve flow."""

    @pytest.mark.asyncio
    async def test_store_and_get_by_id(self, manager, test_user_id, test_room_id):
        """Should store and retrieve by ID."""
        # Store
        artifact = await manager.store(
            name="flow_test",
            content="Test content for flow",
            user_id=test_user_id,
            room_id=test_room_id,
        )

        assert artifact is not None
        assert artifact.name == "flow_test"

        # Retrieve by ID
        retrieved, content = await manager.get(
            artifact.artifact_id, test_user_id, test_room_id
        )

        assert retrieved is not None
        assert retrieved.artifact_id == artifact.artifact_id
        assert content == "Test content for flow"

    @pytest.mark.asyncio
    async def test_store_and_get_by_name(self, manager, test_user_id, test_room_id):
        """Should store and retrieve by name."""
        # Store
        artifact = await manager.store(
            name="named_flow_test",
            content="Named content",
            user_id=test_user_id,
            room_id=test_room_id,
        )

        # Retrieve by name
        retrieved, content = await manager.get(
            "named_flow_test", test_user_id, test_room_id
        )

        assert retrieved is not None
        assert retrieved.name == "named_flow_test"
        assert content == "Named content"

    @pytest.mark.asyncio
    async def test_store_binary_and_retrieve(self, manager, test_user_id, test_room_id):
        """Should handle binary content correctly."""
        binary_content = b"\x00\x01\x02\xff\xfe\xfd"

        artifact = await manager.store(
            name="binary_test",
            content=binary_content,
            user_id=test_user_id,
            room_id=test_room_id,
        )

        retrieved, content = await manager.get(
            artifact.artifact_id, test_user_id, test_room_id
        )

        assert content == binary_content


# ============================================================================
# Multi-User Access Control
# ============================================================================

class TestMultiUserAccess:
    """Tests for multi-user access control scenarios."""

    @pytest.mark.asyncio
    async def test_user_scope_isolation(
        self, manager, test_user_id, test_room_id, second_user_id
    ):
        """User-scoped artifacts should be isolated."""
        # User 1 stores
        artifact = await manager.store(
            name="private_artifact",
            content="Private to user 1",
            user_id=test_user_id,
            room_id=test_room_id,
            scope=ArtifactScope.USER,
        )

        # User 2 cannot retrieve
        retrieved, content = await manager.get(
            artifact.artifact_id, second_user_id, test_room_id
        )

        assert retrieved is None

    @pytest.mark.asyncio
    async def test_room_scope_sharing(
        self, manager, test_user_id, test_room_id, second_user_id
    ):
        """Room-scoped artifacts should be accessible by room members."""
        # User 1 stores with ROOM scope
        artifact = await manager.store(
            name="shared_artifact",
            content="Shared with room",
            user_id=test_user_id,
            room_id=test_room_id,
            scope=ArtifactScope.ROOM,
        )

        # User 2 in same room can retrieve
        retrieved, content = await manager.get(
            artifact.artifact_id, second_user_id, test_room_id
        )

        assert retrieved is not None
        assert content == "Shared with room"

    @pytest.mark.asyncio
    async def test_share_flow(
        self, manager, test_user_id, test_room_id, second_user_id
    ):
        """Should properly share artifacts from USER to ROOM scope."""
        # User 1 stores private
        artifact = await manager.store(
            name="to_share",
            content="Initially private",
            user_id=test_user_id,
            room_id=test_room_id,
            scope=ArtifactScope.USER,
        )

        # User 2 cannot access yet
        retrieved, _ = await manager.get(
            artifact.artifact_id, second_user_id, test_room_id
        )
        assert retrieved is None

        # User 1 shares
        success = await manager.share(
            artifact.artifact_id, test_user_id, test_room_id
        )
        assert success is True

        # Now user 2 can access
        retrieved, content = await manager.get(
            artifact.artifact_id, second_user_id, test_room_id
        )
        assert retrieved is not None
        assert content == "Initially private"


# ============================================================================
# Session Context Flow
# ============================================================================

class TestSessionContextFlow:
    """Tests for session context accumulation and retrieval."""

    @pytest.mark.asyncio
    async def test_session_context_building(self, manager, test_user_id, test_room_id):
        """Should build session context from multiple artifacts."""
        session_id = "session-integration-test"

        # Store multiple artifacts in session
        for i in range(3):
            await manager.store(
                name=f"session_artifact_{i}",
                content=f"Session content {i}",
                user_id=test_user_id,
                room_id=test_room_id,
                session_id=session_id,
            )

        # Get session context
        context = await manager.get_session_context(
            session_id, test_user_id, test_room_id
        )

        assert context is not None
        assert len(context.artifacts) == 3
        assert session_id in context.context_summary

    @pytest.mark.asyncio
    async def test_session_context_different_types(
        self, manager, test_user_id, test_room_id
    ):
        """Should handle different artifact types in session."""
        session_id = "multi-type-session"

        # Text artifact
        await manager.store(
            name="text.txt",
            content="Text content",
            user_id=test_user_id,
            room_id=test_room_id,
            session_id=session_id,
        )

        # Code artifact
        await manager.store(
            name="code.py",
            content="def hello(): pass",
            user_id=test_user_id,
            room_id=test_room_id,
            session_id=session_id,
        )

        # JSON artifact
        await manager.store(
            name="data.json",
            content='{"key": "value"}',
            user_id=test_user_id,
            room_id=test_room_id,
            session_id=session_id,
        )

        context = await manager.get_session_context(
            session_id, test_user_id, test_room_id
        )

        assert len(context.artifacts) == 3
        types = {a.artifact_type for a in context.artifacts}
        assert ArtifactType.TEXT in types
        assert ArtifactType.CODE in types
        assert ArtifactType.JSON in types


# ============================================================================
# Deduplication Flow
# ============================================================================

class TestDeduplicationFlow:
    """Tests for content deduplication."""

    @pytest.mark.asyncio
    async def test_identical_content_deduplicated(
        self, manager, test_user_id, test_room_id
    ):
        """Identical content should be deduplicated."""
        content = "This exact content appears multiple times"

        artifact1 = await manager.store(
            name="artifact_1",
            content=content,
            user_id=test_user_id,
            room_id=test_room_id,
        )

        artifact2 = await manager.store(
            name="artifact_2",
            content=content,
            user_id=test_user_id,
            room_id=test_room_id,
        )

        # Different artifacts
        assert artifact1.artifact_id != artifact2.artifact_id

        # Same content hash
        assert artifact1.content_hash == artifact2.content_hash

    @pytest.mark.asyncio
    async def test_deduplication_stats(self, manager, test_user_id, test_room_id):
        """Storage stats should reflect deduplication."""
        content = "Repeated content for stats test"

        # Store same content 3 times
        for i in range(3):
            await manager.store(
                name=f"repeated_{i}",
                content=content,
                user_id=test_user_id,
                room_id=test_room_id,
            )

        stats = await manager.get_stats()

        assert stats['content']['unique_objects'] == 1
        assert stats['content']['total_references'] == 3


# ============================================================================
# Delete and Cleanup Flow
# ============================================================================

class TestDeleteCleanupFlow:
    """Tests for deletion and cleanup operations."""

    @pytest.mark.asyncio
    async def test_delete_removes_from_list(self, manager, test_user_id, test_room_id):
        """Deleted artifacts should not appear in listings."""
        artifact = await manager.store(
            name="to_delete",
            content="Delete me",
            user_id=test_user_id,
            room_id=test_room_id,
        )

        # Verify it's in the list
        before_list = await manager.list(test_user_id, test_room_id)
        assert any(a.artifact_id == artifact.artifact_id for a in before_list)

        # Delete it
        success = await manager.delete(
            artifact.artifact_id, test_user_id, test_room_id
        )
        assert success is True

        # Verify it's gone
        after_list = await manager.list(test_user_id, test_room_id)
        assert not any(a.artifact_id == artifact.artifact_id for a in after_list)

    @pytest.mark.asyncio
    async def test_delete_only_by_owner(
        self, manager, test_user_id, test_room_id, second_user_id
    ):
        """Only owner should be able to delete."""
        artifact = await manager.store(
            name="owner_only",
            content="Owner's content",
            user_id=test_user_id,
            room_id=test_room_id,
        )

        # Non-owner cannot delete
        success = await manager.delete(
            artifact.artifact_id, second_user_id, test_room_id
        )
        assert success is False

        # Owner can delete
        success = await manager.delete(
            artifact.artifact_id, test_user_id, test_room_id
        )
        assert success is True


# ============================================================================
# Export Flow
# ============================================================================

class TestExportFlow:
    """Tests for artifact export functionality."""

    @pytest.mark.asyncio
    async def test_export_text_artifact(
        self, manager, test_user_id, test_room_id, temp_dir
    ):
        """Should export text artifact to file."""
        content = "Export this text content"
        artifact = await manager.store(
            name="export_text.txt",
            content=content,
            user_id=test_user_id,
            room_id=test_room_id,
        )

        export_path = temp_dir / "exported.txt"
        success = await manager.export_to_file(
            artifact.artifact_id, test_user_id, test_room_id, export_path
        )

        assert success is True
        assert export_path.exists()
        assert export_path.read_text() == content

    @pytest.mark.asyncio
    async def test_export_binary_artifact(
        self, manager, test_user_id, test_room_id, temp_dir
    ):
        """Should export binary artifact to file."""
        content = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100  # Fake PNG header
        artifact = await manager.store(
            name="export_binary.png",
            content=content,
            user_id=test_user_id,
            room_id=test_room_id,
        )

        export_path = temp_dir / "exported.png"
        success = await manager.export_to_file(
            artifact.artifact_id, test_user_id, test_room_id, export_path
        )

        assert success is True
        assert export_path.read_bytes() == content

    @pytest.mark.asyncio
    async def test_export_for_matrix(self, manager, test_user_id, test_room_id):
        """Should prepare export data for Matrix upload."""
        content = "Matrix export content"
        artifact = await manager.store(
            name="matrix_export.txt",
            content=content,
            user_id=test_user_id,
            room_id=test_room_id,
        )

        export_data = await manager.export_for_matrix(
            artifact.artifact_id, test_user_id, test_room_id
        )

        assert export_data is not None
        assert 'content' in export_data
        assert 'filename' in export_data
        assert 'mimetype' in export_data


# ============================================================================
# Quota and Limit Flow
# ============================================================================

class TestQuotaLimitFlow:
    """Tests for quota and limit enforcement."""

    @pytest.mark.asyncio
    async def test_quota_enforcement(self, temp_dir, test_user_id, test_room_id):
        """Should enforce artifact count quota."""
        # Config with low quota
        config = ScratchPadConfig(
            metadata_db_path=str(temp_dir / "metadata.db"),
            cas_storage_path=str(temp_dir / "content"),
            max_artifacts_per_user=3,
        )
        manager = ScratchPadManager(config)

        # Store up to quota
        for i in range(3):
            artifact = await manager.store(
                name=f"quota_{i}",
                content=f"Content {i}",
                user_id=test_user_id,
                room_id=test_room_id,
            )
            assert artifact is not None

        # Next should fail
        artifact = await manager.store(
            name="over_quota",
            content="Over quota content",
            user_id=test_user_id,
            room_id=test_room_id,
        )
        assert artifact is None

    @pytest.mark.asyncio
    async def test_size_limit_enforcement(self, temp_dir, test_user_id, test_room_id):
        """Should enforce artifact size limit."""
        # Config with small size limit
        config = ScratchPadConfig(
            metadata_db_path=str(temp_dir / "metadata.db"),
            cas_storage_path=str(temp_dir / "content"),
            max_artifact_size_bytes=100,  # 100 bytes max
        )
        manager = ScratchPadManager(config)

        # Small content should work
        small = await manager.store(
            name="small",
            content="Small",
            user_id=test_user_id,
            room_id=test_room_id,
        )
        assert small is not None

        # Large content should fail
        large = await manager.store(
            name="large",
            content="x" * 200,  # 200 bytes
            user_id=test_user_id,
            room_id=test_room_id,
        )
        assert large is None


# ============================================================================
# Lifecycle Management Flow
# ============================================================================

class TestLifecycleManagementFlow:
    """Tests for manager lifecycle."""

    @pytest.mark.asyncio
    async def test_start_stop_cleanup(self, test_config):
        """Should properly start and stop cleanup background task."""
        manager = ScratchPadManager(test_config)

        # Start should succeed
        await manager.start()
        assert manager._running is True

        # Stop should succeed
        await manager.stop()
        assert manager._running is False

    @pytest.mark.asyncio
    async def test_factory_function(self, test_config):
        """Factory function should create working manager."""
        manager = create_scratchpad_manager(test_config)

        assert manager is not None
        assert isinstance(manager, ScratchPadManager)

    @pytest.mark.asyncio
    async def test_factory_with_start(self, test_config):
        """Factory with start should create and start manager."""
        manager = await create_and_start_manager(test_config)

        try:
            assert manager is not None
            assert manager._running is True
        finally:
            await manager.stop()


# ============================================================================
# Audit Trail Flow
# ============================================================================

class TestAuditTrailFlow:
    """Tests for audit trail integration."""

    @pytest.mark.asyncio
    async def test_store_creates_audit_event(
        self, manager, test_user_id, test_room_id
    ):
        """Storing artifact should create audit event."""
        await manager.store(
            name="audited",
            content="Audited content",
            user_id=test_user_id,
            room_id=test_room_id,
        )

        # Check audit log exists and has entry
        audit_path = Path(manager.config.audit_log_path)
        assert audit_path.exists()

        # Read and verify content
        content = audit_path.read_text()
        assert "ARTIFACT_CREATED" in content

    @pytest.mark.asyncio
    async def test_access_denied_creates_audit_event(
        self, manager, test_user_id, test_room_id, second_user_id
    ):
        """Access denial should create audit event."""
        artifact = await manager.store(
            name="private",
            content="Private content",
            user_id=test_user_id,
            room_id=test_room_id,
            scope=ArtifactScope.USER,
        )

        # Trigger access denied
        await manager.get(
            artifact.artifact_id, second_user_id, "!otherroom:example.com"
        )

        # Check audit log
        content = Path(manager.config.audit_log_path).read_text()
        assert "ACCESS_DENIED" in content


# ============================================================================
# Type Detection Flow
# ============================================================================

class TestTypeDetectionFlow:
    """Tests for automatic type detection."""

    @pytest.mark.asyncio
    async def test_detect_python_code(self, manager, test_user_id, test_room_id):
        """Should detect Python code."""
        code = '''def hello():
    print("Hello, World!")

class MyClass:
    pass
'''
        artifact = await manager.store(
            name="script.py",
            content=code,
            user_id=test_user_id,
            room_id=test_room_id,
        )

        assert artifact.artifact_type == ArtifactType.CODE

    @pytest.mark.asyncio
    async def test_detect_json(self, manager, test_user_id, test_room_id):
        """Should detect JSON data."""
        json_content = '{"name": "test", "values": [1, 2, 3]}'
        artifact = await manager.store(
            name="data.json",
            content=json_content,
            user_id=test_user_id,
            room_id=test_room_id,
        )

        assert artifact.artifact_type == ArtifactType.JSON

    @pytest.mark.asyncio
    async def test_detect_image(self, manager, test_user_id, test_room_id):
        """Should detect image from magic bytes."""
        # PNG magic bytes
        png_content = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        artifact = await manager.store(
            name="image.png",
            content=png_content,
            user_id=test_user_id,
            room_id=test_room_id,
        )

        assert artifact.artifact_type == ArtifactType.IMAGE


# ============================================================================
# Concurrent Access Flow
# ============================================================================

class TestConcurrentAccessFlow:
    """Tests for concurrent access scenarios."""

    @pytest.mark.asyncio
    async def test_concurrent_stores(self, manager, test_user_id, test_room_id):
        """Should handle concurrent store operations."""
        async def store_artifact(i):
            return await manager.store(
                name=f"concurrent_{i}",
                content=f"Concurrent content {i}",
                user_id=test_user_id,
                room_id=test_room_id,
            )

        # Store 10 artifacts concurrently
        tasks = [store_artifact(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(r is not None for r in results)

        # All should have unique IDs
        ids = [r.artifact_id for r in results]
        assert len(set(ids)) == 10

    @pytest.mark.asyncio
    async def test_concurrent_reads(self, manager, test_user_id, test_room_id):
        """Should handle concurrent read operations."""
        # Create artifact
        artifact = await manager.store(
            name="read_test",
            content="Concurrent read content",
            user_id=test_user_id,
            room_id=test_room_id,
        )

        async def read_artifact():
            return await manager.get(
                artifact.artifact_id, test_user_id, test_room_id
            )

        # Read 10 times concurrently
        tasks = [read_artifact() for _ in range(10)]
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(r[0] is not None for r in results)
        assert all(r[1] == "Concurrent read content" for r in results)
