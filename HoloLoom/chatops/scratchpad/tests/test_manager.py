"""
ChatOps Scratch Pad - Manager Unit Tests
========================================
Unit tests for ScratchPadManager and related components.

Tests cover:
- Store operations (with rate limiting, validation)
- Get operations (by ID and name)
- List operations (with scope filtering)
- Delete operations
- Share operations
- Session context management
- Rate limiting behavior
- Name validation

Created: December 2025
"""

import asyncio
import os
import tempfile
import pytest
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, AsyncMock

# Import from scratchpad module
from HoloLoom.chatops.scratchpad.types import (
    ScratchArtifact,
    ArtifactScope,
    ArtifactType,
    ArtifactStatus,
    ArtifactReference,
    ScratchPadConfig,
    SessionArtifactContext,
    StoreResult,
    RetrieveResult,
    ListResult,
    DeleteResult,
)
from HoloLoom.chatops.scratchpad.manager import (
    ScratchPadManager,
    RateLimiter,
    NameValidator,
    create_scratchpad_manager,
    create_and_start_manager,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def test_config(temp_dir):
    """Create test configuration."""
    return ScratchPadConfig(
        metadata_db_path=str(temp_dir / "metadata.db"),
        cas_storage_path=str(temp_dir / "content"),
        max_artifact_size_bytes=1024 * 1024,  # 1MB
        max_artifacts_per_user=100,
        default_ttl_days=7,
        rate_limit_requests=100,  # High limit for tests
        rate_limit_window_seconds=60,
        auto_store_artifacts=True,
    )


@pytest.fixture
def manager(test_config):
    """Create a ScratchPadManager for tests."""
    return ScratchPadManager(test_config)


@pytest.fixture
def test_user_id():
    """Test user ID."""
    return "@testuser:example.com"


@pytest.fixture
def test_room_id():
    """Test room ID."""
    return "!testroom:example.com"


# ============================================================================
# Rate Limiter Tests
# ============================================================================

class TestRateLimiter:
    """Tests for RateLimiter."""

    def test_allows_within_limit(self):
        """Test that requests within limit are allowed."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)

        for i in range(5):
            allowed, remaining = limiter.check("user1")
            assert allowed is True
            assert remaining == 5 - i
            limiter.record("user1")

    def test_blocks_over_limit(self):
        """Test that requests over limit are blocked."""
        limiter = RateLimiter(max_requests=3, window_seconds=60)

        for _ in range(3):
            limiter.record("user1")

        allowed, remaining = limiter.check("user1")
        assert allowed is False
        assert remaining == 0

    def test_different_users_independent(self):
        """Test that rate limits are per-user."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)

        # User1 hits limit
        limiter.record("user1")
        limiter.record("user1")
        allowed1, _ = limiter.check("user1")
        assert allowed1 is False

        # User2 should still be allowed
        allowed2, remaining2 = limiter.check("user2")
        assert allowed2 is True
        assert remaining2 == 2

    def test_window_expiry(self):
        """Test that requests expire after window."""
        limiter = RateLimiter(max_requests=1, window_seconds=1)

        limiter.record("user1")
        allowed1, _ = limiter.check("user1")
        assert allowed1 is False

        # Wait for window to expire
        time.sleep(1.1)

        allowed2, remaining = limiter.check("user1")
        assert allowed2 is True
        assert remaining == 1

    def test_get_retry_after(self):
        """Test retry-after calculation."""
        limiter = RateLimiter(max_requests=1, window_seconds=60)
        limiter.record("user1")

        retry_after = limiter.get_retry_after("user1")
        assert 0 <= retry_after <= 60

        # Unknown user should return 0
        retry_after_unknown = limiter.get_retry_after("unknown")
        assert retry_after_unknown == 0


# ============================================================================
# Name Validator Tests
# ============================================================================

class TestNameValidator:
    """Tests for NameValidator."""

    def test_valid_names(self):
        """Test that valid names are accepted."""
        validator = NameValidator()

        valid_names = [
            "test",
            "test-file",
            "test_file",
            "test.txt",
            "MyFile123",
            "a",
            "A" * 100,
        ]

        for name in valid_names:
            valid, error = validator.validate(name)
            assert valid is True, f"Name '{name}' should be valid, got error: {error}"
            assert error is None

    def test_invalid_empty(self):
        """Test that empty names are rejected."""
        validator = NameValidator()
        valid, error = validator.validate("")
        assert valid is False
        assert "empty" in error.lower()

    def test_invalid_too_long(self):
        """Test that long names are rejected."""
        validator = NameValidator(max_length=10)
        valid, error = validator.validate("A" * 20)
        assert valid is False
        assert "long" in error.lower()

    def test_invalid_characters(self):
        """Test that invalid characters are rejected."""
        validator = NameValidator()

        invalid_names = [
            "test file",  # space
            "test/file",  # slash
            "test:file",  # colon
            "test<file",  # angle bracket
            "test>file",  # angle bracket
            "test|file",  # pipe
            "test*file",  # asterisk
        ]

        for name in invalid_names:
            valid, error = validator.validate(name)
            assert valid is False, f"Name '{name}' should be invalid"
            assert "only contain" in error.lower()

    def test_reserved_names(self):
        """Test that reserved names are rejected."""
        validator = NameValidator()

        reserved_names = ["..", ".", "null", "undefined", "none", "nil"]

        for name in reserved_names:
            valid, error = validator.validate(name)
            assert valid is False, f"Reserved name '{name}' should be invalid"
            assert "reserved" in error.lower()


# ============================================================================
# ScratchPadManager Store Tests
# ============================================================================

class TestManagerStore:
    """Tests for ScratchPadManager.store()."""

    @pytest.mark.asyncio
    async def test_store_text_content(self, manager, test_user_id, test_room_id):
        """Test storing text content."""
        result = await manager.store(
            name="test.txt",
            content="Hello, World!",
            user_id=test_user_id,
            room_id=test_room_id,
        )

        assert result.success is True
        assert result.artifact is not None
        assert result.artifact.name == "test.txt"
        assert result.artifact.artifact_type == ArtifactType.TEXT
        assert result.artifact.owner_user_id == test_user_id
        assert result.artifact.owner_room_id == test_room_id
        assert result.artifact.scope == ArtifactScope.USER

    @pytest.mark.asyncio
    async def test_store_binary_content(self, manager, test_user_id, test_room_id):
        """Test storing binary content."""
        binary_content = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        result = await manager.store(
            name="image.png",
            content=binary_content,
            user_id=test_user_id,
            room_id=test_room_id,
        )

        assert result.success is True
        assert result.artifact.artifact_type == ArtifactType.IMAGE

    @pytest.mark.asyncio
    async def test_store_with_explicit_type(self, manager, test_user_id, test_room_id):
        """Test storing with explicit artifact type."""
        result = await manager.store(
            name="data",
            content='{"key": "value"}',
            user_id=test_user_id,
            room_id=test_room_id,
            artifact_type=ArtifactType.DATA,
        )

        assert result.success is True
        assert result.artifact.artifact_type == ArtifactType.DATA

    @pytest.mark.asyncio
    async def test_store_with_metadata(self, manager, test_user_id, test_room_id):
        """Test storing with custom metadata."""
        metadata = {"source": "test", "version": "1.0"}

        result = await manager.store(
            name="test",
            content="content",
            user_id=test_user_id,
            room_id=test_room_id,
            metadata=metadata,
        )

        assert result.success is True
        assert result.artifact.metadata == metadata

    @pytest.mark.asyncio
    async def test_store_invalid_name(self, manager, test_user_id, test_room_id):
        """Test storing with invalid name fails."""
        result = await manager.store(
            name="invalid/name",
            content="content",
            user_id=test_user_id,
            room_id=test_room_id,
        )

        assert result.success is False
        assert result.error is not None
        assert "only contain" in result.error.lower()

    @pytest.mark.asyncio
    async def test_store_empty_name(self, manager, test_user_id, test_room_id):
        """Test storing with empty name fails."""
        result = await manager.store(
            name="",
            content="content",
            user_id=test_user_id,
            room_id=test_room_id,
        )

        assert result.success is False
        assert "empty" in result.error.lower()

    @pytest.mark.asyncio
    async def test_store_rate_limited(self, temp_dir, test_user_id, test_room_id):
        """Test that rate limiting works."""
        config = ScratchPadConfig(
            metadata_db_path=str(temp_dir / "metadata.db"),
            cas_storage_path=str(temp_dir / "content"),
            rate_limit_requests=2,
            rate_limit_window_seconds=60,
        )
        manager = ScratchPadManager(config)

        # First two should succeed
        await manager.store(name="test1", content="a", user_id=test_user_id, room_id=test_room_id)
        await manager.store(name="test2", content="b", user_id=test_user_id, room_id=test_room_id)

        # Third should be rate limited
        result = await manager.store(
            name="test3",
            content="c",
            user_id=test_user_id,
            room_id=test_room_id,
        )

        assert result.success is False
        assert "rate limited" in result.error.lower()

    @pytest.mark.asyncio
    async def test_store_with_scope(self, manager, test_user_id, test_room_id):
        """Test storing with different scopes."""
        # Store with ROOM scope
        result = await manager.store(
            name="shared",
            content="shared content",
            user_id=test_user_id,
            room_id=test_room_id,
            scope=ArtifactScope.ROOM,
        )

        assert result.success is True
        assert result.artifact.scope == ArtifactScope.ROOM

    @pytest.mark.asyncio
    async def test_store_with_session(self, manager, test_user_id, test_room_id):
        """Test storing with session ID updates context."""
        session_id = "session-123"

        result = await manager.store(
            name="session_artifact",
            content="content",
            user_id=test_user_id,
            room_id=test_room_id,
            source_session_id=session_id,
        )

        assert result.success is True
        assert result.artifact.source_session_id == session_id

        # Check session context was updated
        context = manager.get_session_context(session_id, test_user_id, test_room_id)
        assert len(context.artifact_references) > 0


# ============================================================================
# ScratchPadManager Get Tests
# ============================================================================

class TestManagerGet:
    """Tests for ScratchPadManager.get()."""

    @pytest.mark.asyncio
    async def test_get_by_id(self, manager, test_user_id, test_room_id):
        """Test getting artifact by ID."""
        # Store first
        store_result = await manager.store(
            name="test",
            content="content",
            user_id=test_user_id,
            room_id=test_room_id,
        )
        artifact_id = store_result.artifact.artifact_id

        # Get by ID
        get_result = await manager.get(artifact_id, test_user_id, test_room_id)

        assert get_result.success is True
        assert get_result.artifact.artifact_id == artifact_id
        assert get_result.content == b"content"

    @pytest.mark.asyncio
    async def test_get_by_name(self, manager, test_user_id, test_room_id):
        """Test getting artifact by name."""
        await manager.store(
            name="myfile",
            content="content",
            user_id=test_user_id,
            room_id=test_room_id,
        )

        result = await manager.get("myfile", test_user_id, test_room_id)

        assert result.success is True
        assert result.artifact.name == "myfile"
        assert result.content == b"content"

    @pytest.mark.asyncio
    async def test_get_not_found(self, manager, test_user_id, test_room_id):
        """Test getting non-existent artifact."""
        result = await manager.get("nonexistent", test_user_id, test_room_id)

        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_get_access_control_user(self, manager, test_user_id, test_room_id):
        """Test that USER scoped artifacts are only accessible by owner."""
        # Store as user1
        await manager.store(
            name="private",
            content="secret",
            user_id=test_user_id,
            room_id=test_room_id,
            scope=ArtifactScope.USER,
        )

        # Get as different user should fail
        result = await manager.get("private", "@other:example.com", test_room_id)

        assert result.success is False

    @pytest.mark.asyncio
    async def test_get_access_control_room(self, manager, test_user_id, test_room_id):
        """Test that ROOM scoped artifacts are accessible by room members."""
        # Store with ROOM scope
        await manager.store(
            name="shared",
            content="shared content",
            user_id=test_user_id,
            room_id=test_room_id,
            scope=ArtifactScope.ROOM,
        )

        # Get as different user in same room should succeed
        result = await manager.get("shared", "@other:example.com", test_room_id)

        assert result.success is True


# ============================================================================
# ScratchPadManager List Tests
# ============================================================================

class TestManagerList:
    """Tests for ScratchPadManager.list()."""

    @pytest.mark.asyncio
    async def test_list_empty(self, manager, test_user_id, test_room_id):
        """Test listing with no artifacts."""
        result = await manager.list(test_user_id, test_room_id)

        assert result.success is True
        assert len(result.artifacts) == 0
        assert result.total_count == 0

    @pytest.mark.asyncio
    async def test_list_user_artifacts(self, manager, test_user_id, test_room_id):
        """Test listing user's artifacts."""
        # Store multiple artifacts
        for i in range(3):
            await manager.store(
                name=f"artifact{i}",
                content=f"content{i}",
                user_id=test_user_id,
                room_id=test_room_id,
            )

        result = await manager.list(test_user_id, test_room_id)

        assert result.success is True
        assert result.total_count == 3
        assert len(result.artifacts) == 3

    @pytest.mark.asyncio
    async def test_list_scope_filter(self, manager, test_user_id, test_room_id):
        """Test listing with scope filter."""
        # Store USER scope
        await manager.store(
            name="private",
            content="private",
            user_id=test_user_id,
            room_id=test_room_id,
            scope=ArtifactScope.USER,
        )

        # Store ROOM scope
        await manager.store(
            name="shared",
            content="shared",
            user_id=test_user_id,
            room_id=test_room_id,
            scope=ArtifactScope.ROOM,
        )

        # List only USER scope
        result = await manager.list(
            test_user_id, test_room_id,
            scope_filter=ArtifactScope.USER
        )

        assert result.success is True
        assert result.total_count == 1
        assert result.artifacts[0].name == "private"

    @pytest.mark.asyncio
    async def test_list_respects_limit(self, manager, test_user_id, test_room_id):
        """Test that listing respects limit."""
        # Store 10 artifacts
        for i in range(10):
            await manager.store(
                name=f"artifact{i}",
                content=f"content{i}",
                user_id=test_user_id,
                room_id=test_room_id,
            )

        result = await manager.list(test_user_id, test_room_id, limit=5)

        assert result.success is True
        assert len(result.artifacts) == 5


# ============================================================================
# ScratchPadManager Delete Tests
# ============================================================================

class TestManagerDelete:
    """Tests for ScratchPadManager.delete()."""

    @pytest.mark.asyncio
    async def test_delete_by_id(self, manager, test_user_id, test_room_id):
        """Test deleting artifact by ID."""
        # Store
        store_result = await manager.store(
            name="to_delete",
            content="content",
            user_id=test_user_id,
            room_id=test_room_id,
        )
        artifact_id = store_result.artifact.artifact_id

        # Delete
        delete_result = await manager.delete(artifact_id, test_user_id, test_room_id)

        assert delete_result.success is True
        assert delete_result.artifact_id == artifact_id

        # Verify deleted
        get_result = await manager.get(artifact_id, test_user_id, test_room_id)
        assert get_result.success is False

    @pytest.mark.asyncio
    async def test_delete_by_name(self, manager, test_user_id, test_room_id):
        """Test deleting artifact by name."""
        await manager.store(
            name="to_delete",
            content="content",
            user_id=test_user_id,
            room_id=test_room_id,
        )

        delete_result = await manager.delete("to_delete", test_user_id, test_room_id)

        assert delete_result.success is True

    @pytest.mark.asyncio
    async def test_delete_not_found(self, manager, test_user_id, test_room_id):
        """Test deleting non-existent artifact."""
        result = await manager.delete("nonexistent", test_user_id, test_room_id)

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_delete_access_control(self, manager, test_user_id, test_room_id):
        """Test that only owner can delete."""
        await manager.store(
            name="owned",
            content="content",
            user_id=test_user_id,
            room_id=test_room_id,
        )

        # Different user tries to delete
        result = await manager.delete("owned", "@other:example.com", test_room_id)

        assert result.success is False


# ============================================================================
# ScratchPadManager Share Tests
# ============================================================================

class TestManagerShare:
    """Tests for ScratchPadManager.share()."""

    @pytest.mark.asyncio
    async def test_share_artifact(self, manager, test_user_id, test_room_id):
        """Test sharing an artifact with room."""
        # Store as USER scope
        await manager.store(
            name="to_share",
            content="content",
            user_id=test_user_id,
            room_id=test_room_id,
            scope=ArtifactScope.USER,
        )

        # Share
        success, error = await manager.share("to_share", test_user_id, test_room_id)

        assert success is True
        assert error is None

        # Verify now accessible by others
        result = await manager.get("to_share", "@other:example.com", test_room_id)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_share_not_found(self, manager, test_user_id, test_room_id):
        """Test sharing non-existent artifact."""
        success, error = await manager.share("nonexistent", test_user_id, test_room_id)

        assert success is False
        assert error is not None


# ============================================================================
# Session Context Tests
# ============================================================================

class TestSessionContext:
    """Tests for session context management."""

    @pytest.mark.asyncio
    async def test_get_session_context_new(self, manager, test_user_id, test_room_id):
        """Test getting new session context."""
        context = manager.get_session_context("session-new", test_user_id, test_room_id)

        assert context is not None
        assert context.session_id == "session-new"
        assert context.user_id == test_user_id
        assert context.room_id == test_room_id
        assert len(context.artifact_references) == 0

    @pytest.mark.asyncio
    async def test_session_context_accumulates(self, manager, test_user_id, test_room_id):
        """Test that session context accumulates artifacts."""
        session_id = "session-accum"

        # Store multiple artifacts with session
        for i in range(3):
            await manager.store(
                name=f"art{i}",
                content=f"content{i}",
                user_id=test_user_id,
                room_id=test_room_id,
                source_session_id=session_id,
            )

        context = manager.get_session_context(session_id, test_user_id, test_room_id)

        assert len(context.artifact_references) == 3

    def test_clear_session_context(self, manager, test_user_id, test_room_id):
        """Test clearing session context."""
        session_id = "session-clear"

        # Get context to create it
        manager.get_session_context(session_id, test_user_id, test_room_id)

        # Clear it
        manager.clear_session_context(session_id, test_user_id, test_room_id)

        # Getting again should create new empty context
        context = manager.get_session_context(session_id, test_user_id, test_room_id)
        assert len(context.artifact_references) == 0

    @pytest.mark.asyncio
    async def test_session_context_summary(self, manager, test_user_id, test_room_id):
        """Test session context summary generation."""
        session_id = "session-summary"

        await manager.store(
            name="code.py",
            content="print('hello')",
            user_id=test_user_id,
            room_id=test_room_id,
            source_session_id=session_id,
            artifact_type=ArtifactType.CODE,
        )

        context = manager.get_session_context(session_id, test_user_id, test_room_id)
        summary = context.get_references_summary()

        assert "code.py" in summary
        assert "code" in summary.lower()


# ============================================================================
# Export Tests
# ============================================================================

class TestExport:
    """Tests for export functionality."""

    @pytest.mark.asyncio
    async def test_export_to_file(self, manager, test_user_id, test_room_id, temp_dir):
        """Test exporting artifact to file."""
        await manager.store(
            name="export_test",
            content="export content",
            user_id=test_user_id,
            room_id=test_room_id,
        )

        output_path = temp_dir / "exported.txt"
        success, error = await manager.export_to_file(
            "export_test",
            test_user_id,
            test_room_id,
            output_path
        )

        assert success is True
        assert output_path.exists()
        assert output_path.read_bytes() == b"export content"

    @pytest.mark.asyncio
    async def test_get_export_data(self, manager, test_user_id, test_room_id):
        """Test getting export data with filename and mimetype."""
        await manager.store(
            name="data.json",
            content='{"key": "value"}',
            user_id=test_user_id,
            room_id=test_room_id,
            artifact_type=ArtifactType.DATA,
        )

        content, filename, mimetype = await manager.get_export_data(
            "data.json",
            test_user_id,
            test_room_id
        )

        assert content == b'{"key": "value"}'
        assert filename == "data.json"
        assert mimetype == "application/json"


# ============================================================================
# Artifact Type Detection Tests
# ============================================================================

class TestTypeDetection:
    """Tests for artifact type detection."""

    @pytest.mark.asyncio
    async def test_detect_by_extension_py(self, manager, test_user_id, test_room_id):
        """Test Python file detection."""
        result = await manager.store(
            name="script.py",
            content="print('hello')",
            user_id=test_user_id,
            room_id=test_room_id,
        )
        assert result.artifact.artifact_type == ArtifactType.CODE

    @pytest.mark.asyncio
    async def test_detect_by_extension_json(self, manager, test_user_id, test_room_id):
        """Test JSON file detection."""
        result = await manager.store(
            name="data.json",
            content='{}',
            user_id=test_user_id,
            room_id=test_room_id,
        )
        assert result.artifact.artifact_type == ArtifactType.DATA

    @pytest.mark.asyncio
    async def test_detect_by_magic_png(self, manager, test_user_id, test_room_id):
        """Test PNG detection by magic bytes."""
        png_header = b'\x89PNG\r\n\x1a\n' + b'\x00' * 100
        result = await manager.store(
            name="image_no_ext",
            content=png_header,
            user_id=test_user_id,
            room_id=test_room_id,
        )
        assert result.artifact.artifact_type == ArtifactType.IMAGE

    @pytest.mark.asyncio
    async def test_detect_code_content(self, manager, test_user_id, test_room_id):
        """Test code detection by content analysis."""
        result = await manager.store(
            name="code_file",
            content="def hello():\n    return 'world'",
            user_id=test_user_id,
            room_id=test_room_id,
        )
        assert result.artifact.artifact_type == ArtifactType.CODE


# ============================================================================
# Factory Function Tests
# ============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_scratchpad_manager(self, test_config):
        """Test create_scratchpad_manager factory."""
        manager = create_scratchpad_manager(test_config)
        assert manager is not None
        assert isinstance(manager, ScratchPadManager)

    def test_create_manager_default_config(self, temp_dir):
        """Test creating manager with default config."""
        # Temporarily change working directory for default paths
        with patch.object(ScratchPadConfig, '__post_init__', lambda self: None):
            config = ScratchPadConfig(
                metadata_db_path=str(temp_dir / "meta.db"),
                cas_storage_path=str(temp_dir / "content"),
            )
            manager = create_scratchpad_manager(config)
            assert manager is not None

    @pytest.mark.asyncio
    async def test_create_and_start_manager(self, test_config):
        """Test create_and_start_manager factory."""
        manager = await create_and_start_manager(test_config)
        assert manager is not None

        # Stop to clean up
        await manager.stop()


# ============================================================================
# Statistics Tests
# ============================================================================

class TestStatistics:
    """Tests for statistics collection."""

    @pytest.mark.asyncio
    async def test_get_stats(self, manager, test_user_id, test_room_id):
        """Test getting manager statistics."""
        # Store some artifacts
        await manager.store(
            name="test1",
            content="content1",
            user_id=test_user_id,
            room_id=test_room_id,
        )
        await manager.store(
            name="test2",
            content="content2",
            user_id=test_user_id,
            room_id=test_room_id,
        )

        stats = manager.get_stats()

        assert "metadata" in stats
        assert "content" in stats
        assert stats["metadata"]["total_artifacts"] >= 2


# ============================================================================
# Lifecycle Tests
# ============================================================================

class TestLifecycle:
    """Tests for manager lifecycle."""

    @pytest.mark.asyncio
    async def test_start_stop(self, test_config):
        """Test starting and stopping manager."""
        manager = ScratchPadManager(test_config)

        await manager.start()
        # Should be running

        await manager.stop()
        # Should be stopped

    @pytest.mark.asyncio
    async def test_stop_without_start(self, manager):
        """Test stopping manager that wasn't started."""
        # Should not raise
        await manager.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
