"""
Jenny Week 4 Integration Tests
===============================
Tests for JennyRuntime unified orchestrator.

Week 4 MVP - December 2025

Tests:
1. Runtime lifecycle (start/stop, context manager)
2. Basic ask() flow
3. Action execution (pin, dismiss, why)
4. Panel management (list, get)
5. Provenance and history
6. Configuration options
7. Error handling
8. Statistics tracking
9. Full integration flow
"""

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime
from typing import List, Dict, Any

from HoloLoom.visualization.jenny_runtime import (
    JennyRuntime,
    JennyConfig,
    JennyPanel,
    create_runtime,
    jenny_session,
)
from HoloLoom.visualization.jenny_spec import (
    LifecycleStage,
    BindingMode,
    PanelTypeJenny,
)
from HoloLoom.visualization.jenny_actions import ActionStatus


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def config():
    """Default test configuration."""
    return JennyConfig(
        enable_ledger=True,
        auto_cleanup=False,  # Don't run cleanup in tests
        enable_streaming=True,
    )


@pytest_asyncio.fixture
async def runtime(config):
    """Provide a started JennyRuntime."""
    jenny = JennyRuntime(config)
    await jenny.start()
    yield jenny
    await jenny.stop()


# ============================================================================
# Test: Runtime Lifecycle
# ============================================================================

class TestRuntimeLifecycle:
    """Test runtime start/stop and context manager."""

    @pytest.mark.asyncio
    async def test_manual_start_stop(self, config):
        """Test manual start() and stop()."""
        jenny = JennyRuntime(config)

        # Not started yet
        assert jenny._started == False

        # Start
        result = await jenny.start()
        assert result is jenny  # Returns self
        assert jenny._started == True
        assert jenny._compiler_version is not None  # Version string, not compiler object
        assert jenny._lifecycle is not None

        # Stop
        await jenny.stop()
        assert jenny._started == False

    @pytest.mark.asyncio
    async def test_context_manager(self, config):
        """Test async context manager."""
        async with JennyRuntime(config) as jenny:
            assert jenny._started == True
            panel = await jenny.ask("Test question")
            assert panel is not None

        # Should be stopped after exit
        assert jenny._started == False

    @pytest.mark.asyncio
    async def test_jenny_session_helper(self):
        """Test jenny_session() convenience function."""
        async with jenny_session() as jenny:
            assert isinstance(jenny, JennyRuntime)
            assert jenny._started == True

    @pytest.mark.asyncio
    async def test_double_start_is_safe(self, config):
        """Starting twice should be safe."""
        jenny = JennyRuntime(config)
        await jenny.start()
        await jenny.start()  # Should not error
        assert jenny._started == True
        await jenny.stop()

    @pytest.mark.asyncio
    async def test_double_stop_is_safe(self, config):
        """Stopping twice should be safe."""
        jenny = JennyRuntime(config)
        await jenny.start()
        await jenny.stop()
        await jenny.stop()  # Should not error
        assert jenny._started == False


# ============================================================================
# Test: Basic Ask Flow
# ============================================================================

class TestAskFlow:
    """Test the ask() method."""

    @pytest.mark.asyncio
    async def test_ask_returns_panel(self, runtime):
        """ask() should return a JennyPanel."""
        panel = await runtime.ask("What is Python?")

        assert isinstance(panel, JennyPanel)
        assert panel.id is not None
        assert len(panel.id) > 0

    @pytest.mark.asyncio
    async def test_panel_has_title(self, runtime):
        """Panel should have a title."""
        panel = await runtime.ask("Explain recursion")

        assert panel.title is not None
        assert len(panel.title) > 0

    @pytest.mark.asyncio
    async def test_panel_has_content(self, runtime):
        """Panel should have content."""
        panel = await runtime.ask("What is a variable?")

        assert panel.content is not None
        assert isinstance(panel.content, dict)

    @pytest.mark.asyncio
    async def test_panel_has_html_rendering(self, runtime):
        """Panel should have HTML rendering."""
        panel = await runtime.ask("What is HTML?")

        assert panel.html is not None
        assert "<" in panel.html  # Contains HTML tags

    @pytest.mark.asyncio
    async def test_panel_has_terminal_rendering(self, runtime):
        """Panel should have terminal rendering."""
        panel = await runtime.ask("What is a terminal?")

        assert panel.terminal is not None
        assert len(panel.terminal) > 0

    @pytest.mark.asyncio
    async def test_panel_has_json_data(self, runtime):
        """Panel should have JSON data."""
        panel = await runtime.ask("What is JSON?")

        assert panel.json_data is not None
        assert isinstance(panel.json_data, dict)

    @pytest.mark.asyncio
    async def test_panel_starts_nascent(self, runtime):
        """New panels should start in NASCENT state."""
        panel = await runtime.ask("What is a lifecycle?")

        assert panel.lifecycle == LifecycleStage.NASCENT

    @pytest.mark.asyncio
    async def test_panel_has_actions(self, runtime):
        """Panel should have available actions."""
        panel = await runtime.ask("What are actions?")

        assert panel.actions is not None
        assert len(panel.actions) > 0

    @pytest.mark.asyncio
    async def test_ask_auto_starts_runtime(self):
        """ask() should auto-start if not started."""
        jenny = JennyRuntime()
        # Not started
        assert jenny._started == False

        panel = await jenny.ask("Auto start test")
        assert jenny._started == True
        assert panel is not None

        await jenny.stop()

    @pytest.mark.asyncio
    async def test_ask_with_context(self, runtime):
        """ask() should accept context."""
        panel = await runtime.ask(
            "Explain this code",
            context={"language": "python", "code": "def foo(): pass"}
        )

        assert panel is not None


# ============================================================================
# Test: Action Execution
# ============================================================================

class TestActionExecution:
    """Test the act() method."""

    @pytest.mark.asyncio
    async def test_pin_action_succeeds(self, runtime):
        """Pin action should transition NASCENT → STABLE."""
        panel = await runtime.ask("Test pin")
        assert panel.lifecycle == LifecycleStage.NASCENT

        result = await runtime.act(panel, "pin")

        assert result.status == ActionStatus.SUCCESS
        assert result.previous_lifecycle == LifecycleStage.NASCENT
        assert result.new_lifecycle == LifecycleStage.STABLE

    @pytest.mark.asyncio
    async def test_dismiss_action_succeeds(self, runtime):
        """Dismiss action should transition to DISSOLVING."""
        panel = await runtime.ask("Test dismiss")

        result = await runtime.act(panel, "dismiss")

        assert result.status == ActionStatus.SUCCESS
        assert result.new_lifecycle == LifecycleStage.DISSOLVING

    @pytest.mark.asyncio
    async def test_why_action_returns_data(self, runtime):
        """Why action should return explanation data."""
        panel = await runtime.ask("Test why")

        result = await runtime.act(panel, "why")

        assert result.status == ActionStatus.SUCCESS
        assert "why_panel_data" in result.data
        why_data = result.data["why_panel_data"]
        assert "reasoning" in why_data
        assert "alternatives_considered" in why_data

    @pytest.mark.asyncio
    async def test_copy_action_returns_content(self, runtime):
        """Copy action should return copyable content."""
        panel = await runtime.ask("Test copy")

        result = await runtime.act(panel, "copy")

        assert result.status == ActionStatus.SUCCESS
        assert "copied_text" in result.data

    @pytest.mark.asyncio
    async def test_act_with_panel_id_string(self, runtime):
        """act() should accept panel ID as string."""
        panel = await runtime.ask("Test string id")

        result = await runtime.act(panel.id, "pin")

        assert result.status == ActionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_act_on_missing_panel_fails(self, runtime):
        """act() should fail gracefully for missing panel."""
        result = await runtime.act("nonexistent-id", "pin")

        assert result.status == ActionStatus.FAILED
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_expand_action(self, runtime):
        """Expand action should work."""
        panel = await runtime.ask("Test expand")

        result = await runtime.act(panel, "expand")

        assert result.status == ActionStatus.SUCCESS
        assert result.data.get("new_size") == "xlarge"

    @pytest.mark.asyncio
    async def test_collapse_action(self, runtime):
        """Collapse action should work."""
        panel = await runtime.ask("Test collapse")

        result = await runtime.act(panel, "collapse")

        assert result.status == ActionStatus.SUCCESS
        assert result.data.get("new_size") == "small"


# ============================================================================
# Test: Panel Management
# ============================================================================

class TestPanelManagement:
    """Test panel listing and retrieval."""

    @pytest.mark.asyncio
    async def test_list_panels_empty_initially(self, runtime):
        """list_panels() should return empty initially."""
        panels = await runtime.list_panels()
        assert panels == []

    @pytest.mark.asyncio
    async def test_list_panels_returns_created(self, runtime):
        """list_panels() should return created panels."""
        await runtime.ask("Panel 1")
        await runtime.ask("Panel 2")
        await runtime.ask("Panel 3")

        panels = await runtime.list_panels()

        assert len(panels) == 3

    @pytest.mark.asyncio
    async def test_list_panels_filter_by_lifecycle(self, runtime):
        """list_panels() should filter by lifecycle."""
        p1 = await runtime.ask("NASCENT panel")
        p2 = await runtime.ask("Will be STABLE")
        await runtime.act(p2, "pin")

        nascent = await runtime.list_panels(lifecycle=LifecycleStage.NASCENT)
        stable = await runtime.list_panels(lifecycle=LifecycleStage.STABLE)

        assert len(nascent) == 1
        assert len(stable) == 1

    @pytest.mark.asyncio
    async def test_get_panel_by_id(self, runtime):
        """get_panel() should retrieve by ID."""
        original = await runtime.ask("Get me later")

        retrieved = await runtime.get_panel(original.id)

        assert retrieved is not None
        assert retrieved.id == original.id
        assert retrieved.title == original.title

    @pytest.mark.asyncio
    async def test_get_missing_panel_returns_none(self, runtime):
        """get_panel() should return None for missing ID."""
        panel = await runtime.get_panel("nonexistent")

        assert panel is None

    @pytest.mark.asyncio
    async def test_panel_reflects_lifecycle_changes(self, runtime):
        """Retrieved panel should reflect lifecycle changes."""
        panel = await runtime.ask("Watch my lifecycle")
        assert panel.lifecycle == LifecycleStage.NASCENT

        await runtime.act(panel, "pin")
        updated = await runtime.get_panel(panel.id)

        assert updated.lifecycle == LifecycleStage.STABLE


# ============================================================================
# Test: Provenance and History
# ============================================================================

class TestProvenance:
    """Test history and audit trail."""

    @pytest.mark.asyncio
    async def test_history_empty_initially(self, runtime):
        """history() should be empty initially."""
        history = runtime.history()
        assert history == []

    @pytest.mark.asyncio
    async def test_history_records_panel_creation(self, runtime):
        """history() should record panel creation."""
        await runtime.ask("Track me")

        history = runtime.history()

        assert len(history) >= 1
        # Should have a creation event
        events = [h.get("event_type") for h in history]
        assert any("creation" in str(e) for e in events)

    @pytest.mark.asyncio
    async def test_history_records_lifecycle_transitions(self, runtime):
        """history() should record lifecycle transitions."""
        panel = await runtime.ask("Track transitions")
        await runtime.act(panel, "pin")

        history = runtime.history()

        events = [h.get("event_type") for h in history]
        assert any("transition" in str(e) for e in events)

    @pytest.mark.asyncio
    async def test_history_limit_parameter(self, runtime):
        """history() should respect limit parameter."""
        for i in range(5):
            await runtime.ask(f"Panel {i}")

        history = runtime.history(limit=2)

        assert len(history) <= 2

    @pytest.mark.asyncio
    async def test_action_history_tracks_actions(self, runtime):
        """action_history() should track executed actions."""
        panel = await runtime.ask("Action tracking")
        await runtime.act(panel, "pin")
        await runtime.act(panel, "copy")

        history = runtime.action_history()

        assert len(history) >= 2
        handlers = [h.get("handler") for h in history]
        assert "pin_panel" in handlers
        assert "copy_content" in handlers


# ============================================================================
# Test: Configuration
# ============================================================================

class TestConfiguration:
    """Test configuration options."""

    @pytest.mark.asyncio
    async def test_default_config(self):
        """Runtime should work with default config."""
        async with JennyRuntime() as jenny:
            panel = await jenny.ask("Default config test")
            assert panel is not None

    @pytest.mark.asyncio
    async def test_custom_compiler_version(self):
        """Custom compiler version should be used."""
        config = JennyConfig(compiler_version="custom-v2.0.0")

        async with JennyRuntime(config) as jenny:
            panel = await jenny.ask("Custom version")
            assert panel.spec.compiler_version == "custom-v2.0.0"

    @pytest.mark.asyncio
    async def test_ledger_disabled(self):
        """Ledger can be disabled."""
        config = JennyConfig(enable_ledger=False)

        async with JennyRuntime(config) as jenny:
            await jenny.ask("No ledger")
            history = jenny.history()
            assert history == []

    @pytest.mark.asyncio
    async def test_streaming_disabled(self):
        """Streaming can be disabled."""
        config = JennyConfig(enable_streaming=False)

        async with JennyRuntime(config) as jenny:
            assert jenny._streaming is None


# ============================================================================
# Test: Statistics
# ============================================================================

class TestStatistics:
    """Test statistics tracking."""

    @pytest.mark.asyncio
    async def test_stats_tracks_panels_created(self, runtime):
        """Stats should track panels created."""
        await runtime.ask("Panel 1")
        await runtime.ask("Panel 2")

        stats = runtime.get_stats()

        assert stats["panels_created"] == 2

    @pytest.mark.asyncio
    async def test_stats_tracks_actions_executed(self, runtime):
        """Stats should track actions executed."""
        panel = await runtime.ask("Stats test")
        await runtime.act(panel, "pin")
        await runtime.act(panel, "copy")

        stats = runtime.get_stats()

        assert stats["actions_executed"] == 2

    @pytest.mark.asyncio
    async def test_stats_tracks_queries_processed(self, runtime):
        """Stats should track queries processed."""
        await runtime.ask("Query 1")
        await runtime.ask("Query 2")
        await runtime.ask("Query 3")

        stats = runtime.get_stats()

        assert stats["queries_processed"] == 3

    @pytest.mark.asyncio
    async def test_stats_includes_start_time(self, runtime):
        """Stats should include start time."""
        stats = runtime.get_stats()

        assert stats["start_time"] is not None
        assert isinstance(stats["start_time"], datetime)


# ============================================================================
# Test: Error Handling
# ============================================================================

class TestErrorHandling:
    """Test error handling."""

    @pytest.mark.asyncio
    async def test_invalid_action_name_handled(self, runtime):
        """Invalid action name should be handled gracefully."""
        panel = await runtime.ask("Error test")

        # Unknown action - should not crash
        result = await runtime.act(panel, "nonexistent_action")

        # Should fail but not crash
        assert result.status == ActionStatus.FAILED

    @pytest.mark.asyncio
    async def test_get_panel_not_started_returns_none(self):
        """get_panel() before start should return None."""
        jenny = JennyRuntime()
        # Not started

        panel = await jenny.get_panel("any-id")

        assert panel is None

    @pytest.mark.asyncio
    async def test_list_panels_not_started_returns_empty(self):
        """list_panels() before start should return empty."""
        jenny = JennyRuntime()

        panels = await jenny.list_panels()

        assert panels == []


# ============================================================================
# Test: Full Integration
# ============================================================================

class TestFullIntegration:
    """Full integration test scenarios."""

    @pytest.mark.asyncio
    async def test_complete_user_journey(self):
        """Test complete user journey."""
        async with jenny_session() as jenny:
            # User asks a question
            panel = await jenny.ask("What is Thompson Sampling?")
            assert panel.lifecycle == LifecycleStage.NASCENT

            # User likes it and pins
            result = await jenny.act(panel, "pin")
            assert result.status == ActionStatus.SUCCESS

            # User asks why this UI
            why_result = await jenny.act(panel, "why")
            assert "reasoning" in why_result.data.get("why_panel_data", {})

            # User copies content
            copy_result = await jenny.act(panel, "copy")
            assert copy_result.status == ActionStatus.SUCCESS

            # Check history captured everything
            history = jenny.history()
            assert len(history) >= 3

            # Verify stats
            stats = jenny.get_stats()
            assert stats["panels_created"] == 1
            assert stats["actions_executed"] == 3

    @pytest.mark.asyncio
    async def test_multiple_panels_workflow(self):
        """Test workflow with multiple panels."""
        async with jenny_session() as jenny:
            # Create multiple panels
            panels = []
            for i in range(5):
                p = await jenny.ask(f"Question {i}")
                panels.append(p)

            # Pin some
            await jenny.act(panels[0], "pin")
            await jenny.act(panels[2], "pin")

            # Dismiss one
            await jenny.act(panels[4], "dismiss")

            # List by state
            nascent = await jenny.list_panels(lifecycle=LifecycleStage.NASCENT)
            stable = await jenny.list_panels(lifecycle=LifecycleStage.STABLE)
            dissolving = await jenny.list_panels(lifecycle=LifecycleStage.DISSOLVING)

            # 5 created, 2 pinned, 1 dismissed, 2 still nascent
            assert len(stable) == 2
            assert len(dissolving) == 1
            assert len(nascent) == 2

    def test_sync_helpers(self):
        """Test sync helper methods."""
        # Create new event loop for this test (avoid pytest-asyncio conflict)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            jenny = JennyRuntime()

            # Sync ask
            panel = jenny.ask_sync("Sync test question")
            assert panel is not None
            assert panel.lifecycle == LifecycleStage.NASCENT

            # Sync act
            result = jenny.act_sync(panel, "pin")
            assert result.status == ActionStatus.SUCCESS

            # Clean up
            loop.run_until_complete(jenny.stop())
        finally:
            loop.close()


# ============================================================================
# Test: Factory Functions
# ============================================================================

class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_runtime(self):
        """create_runtime() should create JennyRuntime."""
        jenny = create_runtime()

        assert isinstance(jenny, JennyRuntime)
        assert jenny._started == False

    def test_create_runtime_with_config(self):
        """create_runtime() should accept config."""
        config = JennyConfig(compiler_version="test-v1.0")
        jenny = create_runtime(config)

        assert jenny._config.compiler_version == "test-v1.0"


# ============================================================================
# Test: Panel Representation
# ============================================================================

class TestPanelRepresentation:
    """Test JennyPanel __repr__ and accessors."""

    @pytest.mark.asyncio
    async def test_panel_repr(self, runtime):
        """Panel should have nice repr."""
        panel = await runtime.ask("Repr test")

        repr_str = repr(panel)

        assert "JennyPanel" in repr_str
        assert panel.id[:8] in repr_str

    @pytest.mark.asyncio
    async def test_panel_accessors(self, runtime):
        """Panel accessors should work."""
        panel = await runtime.ask("Accessor test")

        assert panel.id == panel.spec.spec_id
        assert panel.title == panel.spec.title
        assert panel.lifecycle == panel.spec.lifecycle
        assert panel.panel_type == panel.spec.panel_type
        assert panel.content == panel.spec.content
        assert panel.actions == panel.spec.actions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
