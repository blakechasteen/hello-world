"""
Tapestry System Tests

Comprehensive tests for the session continuity system.

Tests:
- Protocol/data classes
- JSON backend persistence
- Signal registry
- FabricInspector aggregation
- Warper goal decomposition
- LoomKeeper orchestration
- End-to-end workflow

Created: December 2025
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from datetime import datetime

from hololoom.tapestry.protocol import (
    ThreadStatus,
    Thread,
    Tapestry,
    SignalResult,
    FabricCheckResult,
    NoTapestryError
)
from hololoom.tapestry.backends.json_backend import JsonTapestryBackend
from hololoom.tapestry.factory import create_tapestry_backend
from hololoom.tapestry.signals.registry import SignalRegistry
from hololoom.tapestry.inspector import FabricInspector
from hololoom.tapestry.warper import Warper
from hololoom.tapestry.keeper import LoomKeeper, SessionContext
from hololoom.tapestry.git import GitIntegration


# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def tapestry_path(temp_dir):
    """Get path for test tapestry."""
    return str(temp_dir / "test_tapestry.json")


@pytest.fixture
def sample_tapestry():
    """Create sample tapestry for testing."""
    return Tapestry.create(
        goal="Implement user authentication",
        thread_descriptions=[
            "Research authentication patterns",
            "Implement login endpoint",
            "Add password hashing",
            "Write tests"
        ]
    )


# ─────────────────────────────────────────────────────────────────
# Protocol Tests
# ─────────────────────────────────────────────────────────────────

class TestThreadStatus:
    """Tests for ThreadStatus enum."""

    def test_all_statuses_defined(self):
        """Verify all expected statuses exist."""
        assert ThreadStatus.UNWOVEN.value == "unwoven"
        assert ThreadStatus.WEAVING.value == "weaving"
        assert ThreadStatus.WOVEN.value == "woven"
        assert ThreadStatus.TANGLED.value == "tangled"
        assert ThreadStatus.UNRAVELED.value == "unraveled"

    def test_status_count(self):
        """Verify exactly 5 statuses."""
        assert len(ThreadStatus) == 5


class TestThread:
    """Tests for Thread data class."""

    def test_thread_creation(self):
        """Test basic thread creation."""
        thread = Thread(
            id="1",
            description="Test thread",
            status=ThreadStatus.UNWOVEN,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        assert thread.id == "1"
        assert thread.status == ThreadStatus.UNWOVEN
        assert thread.commit_hash is None

    def test_thread_immutable(self):
        """Verify thread is frozen (immutable)."""
        thread = Thread(
            id="1",
            description="Test",
            status=ThreadStatus.UNWOVEN,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            thread.status = ThreadStatus.WOVEN

    def test_thread_with_status(self):
        """Test immutable status update."""
        thread = Thread(
            id="1",
            description="Test",
            status=ThreadStatus.UNWOVEN,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        new_thread = thread.with_status(ThreadStatus.WOVEN, commit_hash="abc123")

        # Original unchanged
        assert thread.status == ThreadStatus.UNWOVEN
        assert thread.commit_hash is None

        # New thread updated
        assert new_thread.status == ThreadStatus.WOVEN
        assert new_thread.commit_hash == "abc123"
        assert new_thread.id == thread.id


class TestTapestry:
    """Tests for Tapestry data class."""

    def test_tapestry_create(self):
        """Test factory creation."""
        tapestry = Tapestry.create(
            goal="Test goal",
            thread_descriptions=["Task 1", "Task 2", "Task 3"]
        )

        assert tapestry.goal == "Test goal"
        assert len(tapestry.threads) == 3
        assert all(t.status == ThreadStatus.UNWOVEN for t in tapestry.threads)
        assert tapestry.loom_id.startswith("loom_")

    def test_next_unwoven(self, sample_tapestry):
        """Test next_unwoven() returns first unwoven thread."""
        next_t = sample_tapestry.next_unwoven()
        assert next_t is not None
        assert next_t.id == "1"

    def test_next_unwoven_respects_order(self, sample_tapestry):
        """Test next_unwoven() respects thread order."""
        # Mark first thread as woven
        sample_tapestry.update_thread("1", ThreadStatus.WOVEN)

        next_t = sample_tapestry.next_unwoven()
        assert next_t.id == "2"

    def test_next_unwoven_none_when_complete(self, sample_tapestry):
        """Test next_unwoven() returns None when all woven."""
        for thread in sample_tapestry.threads:
            sample_tapestry.update_thread(thread.id, ThreadStatus.WOVEN)

        assert sample_tapestry.next_unwoven() is None

    def test_to_dict_from_dict(self, sample_tapestry):
        """Test serialization round-trip."""
        data = sample_tapestry.to_dict()
        restored = Tapestry.from_dict(data)

        assert restored.loom_id == sample_tapestry.loom_id
        assert restored.goal == sample_tapestry.goal
        assert len(restored.threads) == len(sample_tapestry.threads)

    def test_is_complete(self, sample_tapestry):
        """Test is_complete() method."""
        assert not sample_tapestry.is_complete()

        for thread in sample_tapestry.threads:
            sample_tapestry.update_thread(thread.id, ThreadStatus.WOVEN)

        assert sample_tapestry.is_complete()


# ─────────────────────────────────────────────────────────────────
# Backend Tests
# ─────────────────────────────────────────────────────────────────

class TestJsonTapestryBackend:
    """Tests for JSON backend."""

    @pytest.mark.asyncio
    async def test_save_and_load(self, tapestry_path, sample_tapestry):
        """Test save and load round-trip."""
        backend = JsonTapestryBackend(tapestry_path)

        await backend.save(sample_tapestry)
        loaded = await backend.load()

        assert loaded is not None
        assert loaded.loom_id == sample_tapestry.loom_id
        assert loaded.goal == sample_tapestry.goal

    @pytest.mark.asyncio
    async def test_exists(self, tapestry_path, sample_tapestry):
        """Test exists() method."""
        backend = JsonTapestryBackend(tapestry_path)

        assert not await backend.exists()
        await backend.save(sample_tapestry)
        assert await backend.exists()

    @pytest.mark.asyncio
    async def test_delete(self, tapestry_path, sample_tapestry):
        """Test delete() method."""
        backend = JsonTapestryBackend(tapestry_path)

        await backend.save(sample_tapestry)
        await backend.delete()

        assert not await backend.exists()

    @pytest.mark.asyncio
    async def test_load_nonexistent(self, tapestry_path):
        """Test loading nonexistent file returns None."""
        backend = JsonTapestryBackend(tapestry_path)
        loaded = await backend.load()
        assert loaded is None

    @pytest.mark.asyncio
    async def test_atomic_write(self, tapestry_path, sample_tapestry):
        """Verify atomic write creates backup."""
        backend = JsonTapestryBackend(tapestry_path)

        # First save
        await backend.save(sample_tapestry)

        # Modify and save again
        sample_tapestry.update_thread("1", ThreadStatus.WOVEN)
        await backend.save(sample_tapestry)

        # Check backup exists
        backup_path = Path(tapestry_path).with_suffix('.json.bak')
        assert backup_path.exists()


# ─────────────────────────────────────────────────────────────────
# Factory Tests
# ─────────────────────────────────────────────────────────────────

class TestFactory:
    """Tests for backend factory."""

    @pytest.mark.asyncio
    async def test_create_json_backend(self, tapestry_path):
        """Test creating JSON backend."""
        backend = await create_tapestry_backend("json", tapestry_path)
        assert isinstance(backend, JsonTapestryBackend)

    @pytest.mark.asyncio
    async def test_auto_detect(self, tapestry_path):
        """Test auto-detection defaults to JSON."""
        backend = await create_tapestry_backend(None, tapestry_path)
        assert isinstance(backend, JsonTapestryBackend)

    @pytest.mark.asyncio
    async def test_unknown_backend_fallback(self, tapestry_path):
        """Test unknown backend falls back to JSON."""
        backend = await create_tapestry_backend("unknown", tapestry_path)
        assert isinstance(backend, JsonTapestryBackend)


# ─────────────────────────────────────────────────────────────────
# Signal Registry Tests
# ─────────────────────────────────────────────────────────────────

class TestSignalRegistry:
    """Tests for signal registry."""

    def test_builtin_signals_registered(self):
        """Verify 6 built-in signals are registered."""
        # Import builtins to trigger registration
        from hololoom.tapestry.signals import builtins

        assert SignalRegistry.count() >= 6
        assert SignalRegistry.is_registered("tests")
        assert SignalRegistry.is_registered("quality")
        assert SignalRegistry.is_registered("alignment")
        assert SignalRegistry.is_registered("architecture")
        assert SignalRegistry.is_registered("documentation")
        assert SignalRegistry.is_registered("integration")

    def test_total_weight_approximately_one(self):
        """Verify total weight is approximately 1.0."""
        from hololoom.tapestry.signals import builtins

        total = SignalRegistry.total_weight()
        assert 0.99 <= total <= 1.01

    def test_get_signal(self):
        """Test getting signal by name."""
        from hololoom.tapestry.signals import builtins

        signal = SignalRegistry.get("tests")
        assert signal is not None
        assert signal.name == "tests"
        assert signal.weight == 0.20

    def test_get_all(self):
        """Test getting all signals."""
        from hololoom.tapestry.signals import builtins

        signals = SignalRegistry.get_all()
        assert len(signals) >= 6


# ─────────────────────────────────────────────────────────────────
# Inspector Tests
# ─────────────────────────────────────────────────────────────────

class TestFabricInspector:
    """Tests for FabricInspector."""

    @pytest.mark.asyncio
    async def test_inspect_no_signals(self):
        """Test inspection with no signals."""
        inspector = FabricInspector(signals=[])
        thread = Thread(
            id="1",
            description="Test",
            status=ThreadStatus.WEAVING,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )

        result = await inspector.inspect(thread, {})

        assert result.passed
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_inspect_with_mock_signal(self):
        """Test inspection with mock signal."""
        class MockSignal:
            name = "mock"
            weight = 1.0

            async def check(self, thread, context):
                return SignalResult(
                    signal_name="mock",
                    passed=True,
                    score=0.8
                )

        inspector = FabricInspector(signals=[MockSignal()])
        thread = Thread(
            id="1",
            description="Test",
            status=ThreadStatus.WEAVING,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )

        result = await inspector.inspect(thread, {})

        assert result.passed
        assert result.confidence == 0.8

    @pytest.mark.asyncio
    async def test_blocker_fails_inspection(self):
        """Test that blocker signal fails inspection."""
        class BlockerSignal:
            name = "blocker"
            weight = 0.5

            async def check(self, thread, context):
                return SignalResult(
                    signal_name="blocker",
                    passed=False,
                    score=0.0,
                    is_blocker=True,
                    details={"error": "Test blocker"}
                )

        inspector = FabricInspector(signals=[BlockerSignal()])
        thread = Thread(
            id="1",
            description="Test",
            status=ThreadStatus.WEAVING,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )

        result = await inspector.inspect(thread, {})

        assert not result.passed
        assert result.confidence == 0.0
        assert len(result.blockers) == 1


# ─────────────────────────────────────────────────────────────────
# Warper Tests
# ─────────────────────────────────────────────────────────────────

class TestWarper:
    """Tests for Warper."""

    @pytest.mark.asyncio
    async def test_basic_decomposition(self, tapestry_path):
        """Test basic goal decomposition."""
        backend = JsonTapestryBackend(tapestry_path)
        warper = Warper(backend, use_llm_planning=False)

        tapestry = await warper.setup("Implement feature X")

        assert tapestry is not None
        assert len(tapestry.threads) >= 1

    @pytest.mark.asyncio
    async def test_explicit_threads(self, tapestry_path):
        """Test setup with explicit threads."""
        backend = JsonTapestryBackend(tapestry_path)
        warper = Warper(backend, use_llm_planning=False)

        threads = ["Step 1", "Step 2", "Step 3"]
        tapestry = await warper.setup("Test goal", threads=threads)

        assert len(tapestry.threads) == 3
        assert tapestry.threads[0].description == "Step 1"

    @pytest.mark.asyncio
    async def test_resume_existing(self, tapestry_path):
        """Test resuming existing tapestry."""
        backend = JsonTapestryBackend(tapestry_path)
        warper = Warper(backend, use_llm_planning=False)

        # Create tapestry
        original = await warper.setup("Test goal", threads=["Task 1"])

        # Resume
        resumed = await warper.resume()

        assert resumed is not None
        assert resumed.loom_id == original.loom_id


# ─────────────────────────────────────────────────────────────────
# LoomKeeper Tests
# ─────────────────────────────────────────────────────────────────

class TestLoomKeeper:
    """Tests for LoomKeeper."""

    @pytest.mark.asyncio
    async def test_start_session(self, tapestry_path):
        """Test starting a new session."""
        keeper = LoomKeeper(path=tapestry_path)

        tapestry = await keeper.start(
            "Test goal",
            threads=["Task 1", "Task 2"]
        )

        assert tapestry is not None
        assert len(tapestry.threads) == 2

    @pytest.mark.asyncio
    async def test_resume_session(self, tapestry_path):
        """Test resuming session."""
        keeper = LoomKeeper(path=tapestry_path)

        # Start
        await keeper.start("Test", threads=["Task 1"])

        # Resume
        result = await keeper.resume()

        assert result is not None
        tapestry, next_thread = result
        assert next_thread is not None

    @pytest.mark.asyncio
    async def test_weave_thread(self, tapestry_path):
        """Test weaving a thread."""
        keeper = LoomKeeper(path=tapestry_path)

        # Use a mock inspector that always passes
        class MockInspector:
            async def inspect(self, thread, context):
                return FabricCheckResult(
                    passed=True,
                    confidence=0.9
                )

        keeper.inspector = MockInspector()

        tapestry = await keeper.start("Test", threads=["Task 1"])
        thread = tapestry.next_unwoven()

        async def executor(t):
            return "done"

        tapestry = await keeper.weave_thread(tapestry, thread, executor)

        assert tapestry._get_thread("1").status == ThreadStatus.WOVEN

    @pytest.mark.asyncio
    async def test_session_context_manager(self, tapestry_path):
        """Test session context manager."""
        keeper = LoomKeeper(path=tapestry_path)

        # Mock inspector
        class MockInspector:
            async def inspect(self, thread, context):
                return FabricCheckResult(passed=True, confidence=0.9)

        keeper.inspector = MockInspector()

        async def executor(t):
            return "done"

        async with keeper.session("Test goal") as ctx:
            assert ctx.next_thread is not None
            await ctx.weave(executor)

    @pytest.mark.asyncio
    async def test_session_no_tapestry_error(self, tapestry_path):
        """Test error when resuming with no tapestry."""
        keeper = LoomKeeper(path=tapestry_path)

        with pytest.raises(NoTapestryError):
            async with keeper.session() as ctx:
                pass


# ─────────────────────────────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────────────────────────────

class TestIntegration:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_full_workflow(self, tapestry_path):
        """Test complete workflow: start → weave → complete."""
        keeper = LoomKeeper(path=tapestry_path)

        # Mock inspector
        class MockInspector:
            async def inspect(self, thread, context):
                return FabricCheckResult(passed=True, confidence=0.85)

        keeper.inspector = MockInspector()

        async def executor(thread):
            return f"Completed: {thread.description}"

        # Start session
        async with keeper.session("Test workflow") as ctx:
            # Weave all threads
            while ctx.next_thread:
                success, check = await ctx.weave(executor)
                assert success
                assert check.confidence >= 0.8

        # Verify complete
        status = await keeper.get_status()
        assert status is not None
        assert status["is_complete"]

    @pytest.mark.asyncio
    async def test_resume_after_partial(self, tapestry_path):
        """Test resuming after partial completion."""
        keeper = LoomKeeper(path=tapestry_path)

        class MockInspector:
            async def inspect(self, thread, context):
                return FabricCheckResult(passed=True, confidence=0.9)

        keeper.inspector = MockInspector()

        async def executor(t):
            return "done"

        # Start and weave first thread only
        async with keeper.session("Multi-step task") as ctx:
            await ctx.weave(executor)
            # Don't complete all threads

        # Resume and complete
        async with keeper.session() as ctx:  # Resume
            while ctx.next_thread:
                await ctx.weave(executor)

        status = await keeper.get_status()
        assert status["is_complete"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
