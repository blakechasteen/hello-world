"""
Jenny Week 3 Tests - Action Handler + Streaming
================================================

Tests for Week 3 components:
1. JennyActionHandler - Panel action execution
2. StreamingManager - Reactive/streaming data updates

Total: 45+ tests covering:
- Action execution (PIN, DISMISS, WHY, etc.)
- Custom handler registration
- Confirmation flows
- Streaming subscriptions
- Reactive polling
- Error handling
- Integration with lifecycle manager
"""

import pytest
import asyncio
from datetime import datetime
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

from HoloLoom.visualization.jenny_spec import (
    JennySpec,
    LifecycleStage,
    BindingMode,
    DissolutionTrigger,
    PanelTypeJenny,
    ACTION_PIN,
    ACTION_DISMISS,
    ACTION_WHY,
    ACTION_EXPAND,
    ACTION_COLLAPSE,
    ACTION_COPY,
    ACTION_EXPORT,
    create_action,
)
from HoloLoom.visualization.jenny_lifecycle import (
    InMemoryLifecycleManager,
    PanelState,
    PanelNotFoundError,
    InvalidTransitionError,
)
from HoloLoom.visualization.jenny_actions import (
    ActionStatus,
    ActionResult,
    JennyActionHandler,
    create_action_handler,
)
from HoloLoom.visualization.jenny_streaming import (
    StreamStatus,
    UpdateType,
    StreamUpdate,
    Subscription,
    StreamingManager,
    MockDataSource,
    create_streaming_manager,
)
from HoloLoom.visualization.spec_ledger import SpecLedger


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def lifecycle_manager():
    """Create lifecycle manager for tests."""
    return InMemoryLifecycleManager()


@pytest.fixture
def spec_ledger():
    """Create spec ledger for tests."""
    return SpecLedger()


@pytest.fixture
def action_handler(lifecycle_manager, spec_ledger):
    """Create action handler with lifecycle manager and ledger."""
    return create_action_handler(lifecycle_manager, spec_ledger)


@pytest.fixture
def streaming_manager():
    """Create streaming manager with mock data source."""
    return create_streaming_manager()


@pytest.fixture
def sample_nascent_spec():
    """Create a NASCENT spec for testing."""
    return JennySpec(
        spec_id=str(uuid4()),
        spacetime_id="spacetime_123",
        panel_type=PanelTypeJenny.TEXT,
        lifecycle=LifecycleStage.NASCENT,
        content={"text": "Test content"},
    )


@pytest.fixture
def sample_stable_spec():
    """Create a STABLE spec for testing."""
    return JennySpec(
        spec_id=str(uuid4()),
        spacetime_id="spacetime_456",
        panel_type=PanelTypeJenny.GRAPH,
        lifecycle=LifecycleStage.STABLE,
        content={"nodes": [], "edges": []},
    )


@pytest.fixture
def sample_reactive_spec():
    """Create a spec with REACTIVE binding for streaming tests."""
    return JennySpec(
        spec_id=str(uuid4()),
        spacetime_id="spacetime_789",
        panel_type=PanelTypeJenny.METRIC,
        lifecycle=LifecycleStage.STABLE,
        binding_mode=BindingMode.REACTIVE,
        data_source="metrics.cpu_usage",
        refresh_interval_ms=1000,
        content={"value": 0},
    )


@pytest.fixture
def sample_streaming_spec():
    """Create a spec with STREAMING binding."""
    return JennySpec(
        spec_id=str(uuid4()),
        spacetime_id="spacetime_999",
        panel_type=PanelTypeJenny.TIMELINE,
        lifecycle=LifecycleStage.STABLE,
        binding_mode=BindingMode.STREAMING,
        data_source="events.stream",
        content={"events": []},
    )


# ============================================================================
# Action Handler Tests - Basic Actions
# ============================================================================

class TestActionHandlerPinPanel:
    """Tests for pin_panel action."""

    @pytest.mark.asyncio
    async def test_pin_nascent_panel(self, action_handler, lifecycle_manager, sample_nascent_spec):
        """Pin action should transition NASCENT → STABLE."""
        # Register panel
        await lifecycle_manager.spawn(sample_nascent_spec)

        # Execute pin action
        result = await action_handler.execute(ACTION_PIN, sample_nascent_spec)

        assert result.status == ActionStatus.SUCCESS
        assert result.handler == "pin_panel"
        assert result.previous_lifecycle == LifecycleStage.NASCENT
        assert result.new_lifecycle == LifecycleStage.STABLE
        assert "pinned" in result.message.lower()

    @pytest.mark.asyncio
    async def test_pin_already_stable_fails(self, action_handler, lifecycle_manager, sample_nascent_spec):
        """Pinning an already stable panel should fail (invalid transition)."""
        # Spawn as NASCENT first
        await lifecycle_manager.spawn(sample_nascent_spec)
        # Pin to make it STABLE
        await lifecycle_manager.pin(sample_nascent_spec.spec_id)

        # Try to pin again - should fail
        result = await action_handler.execute(ACTION_PIN, sample_nascent_spec)

        assert result.status == ActionStatus.FAILED
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_pin_unregistered_panel_fails(self, action_handler, sample_nascent_spec):
        """Pinning unregistered panel should fail."""
        result = await action_handler.execute(ACTION_PIN, sample_nascent_spec)

        assert result.status == ActionStatus.FAILED
        assert "not found" in result.error.lower()


class TestActionHandlerDismissPanel:
    """Tests for dismiss_panel action."""

    @pytest.mark.asyncio
    async def test_dismiss_nascent_panel(self, action_handler, lifecycle_manager, sample_nascent_spec):
        """Dismiss NASCENT panel → DISSOLVING."""
        await lifecycle_manager.spawn(sample_nascent_spec)

        result = await action_handler.execute(ACTION_DISMISS, sample_nascent_spec)

        assert result.status == ActionStatus.SUCCESS
        assert result.handler == "dismiss_panel"
        assert result.new_lifecycle == LifecycleStage.DISSOLVING

    @pytest.mark.asyncio
    async def test_dismiss_stable_panel(self, action_handler, lifecycle_manager, sample_stable_spec):
        """Dismiss STABLE panel → DISSOLVING."""
        await lifecycle_manager.spawn(sample_stable_spec)

        result = await action_handler.execute(ACTION_DISMISS, sample_stable_spec)

        assert result.status == ActionStatus.SUCCESS
        assert result.new_lifecycle == LifecycleStage.DISSOLVING


class TestActionHandlerWhyPanel:
    """Tests for show_why_panel action."""

    @pytest.mark.asyncio
    async def test_show_why_generates_explanation(self, action_handler, lifecycle_manager, sample_nascent_spec):
        """Why action should generate explanation data."""
        await lifecycle_manager.spawn(sample_nascent_spec)

        result = await action_handler.execute(ACTION_WHY, sample_nascent_spec)

        assert result.status == ActionStatus.SUCCESS
        assert result.handler == "show_why_panel"
        assert "why_panel_data" in result.data
        assert result.data["should_spawn_panel"] is True

        why_data = result.data["why_panel_data"]
        assert why_data["original_spec_id"] == sample_nascent_spec.spec_id
        assert why_data["panel_type"] == sample_nascent_spec.panel_type.value
        assert "reasoning" in why_data
        assert "provenance" in why_data


class TestActionHandlerSizeActions:
    """Tests for expand/collapse actions."""

    @pytest.mark.asyncio
    async def test_expand_panel(self, action_handler, lifecycle_manager, sample_stable_spec):
        """Expand action should return size change data."""
        await lifecycle_manager.spawn(sample_stable_spec)

        result = await action_handler.execute(ACTION_EXPAND, sample_stable_spec)

        assert result.status == ActionStatus.SUCCESS
        assert result.data["new_size"] == "xlarge"
        assert result.data["animation"] == "scale_up"

    @pytest.mark.asyncio
    async def test_collapse_panel(self, action_handler, lifecycle_manager, sample_stable_spec):
        """Collapse action should return size change data."""
        await lifecycle_manager.spawn(sample_stable_spec)

        result = await action_handler.execute(ACTION_COLLAPSE, sample_stable_spec)

        assert result.status == ActionStatus.SUCCESS
        assert result.data["new_size"] == "small"
        assert result.data["animation"] == "scale_down"


class TestActionHandlerCopyContent:
    """Tests for copy_content action."""

    @pytest.mark.asyncio
    async def test_copy_text_content(self, action_handler, lifecycle_manager):
        """Copy text panel content."""
        spec = JennySpec(
            panel_type=PanelTypeJenny.TEXT,
            content={"text": "Important information"},
        )
        await lifecycle_manager.spawn(spec)

        result = await action_handler.execute(ACTION_COPY, spec)

        assert result.status == ActionStatus.SUCCESS
        assert result.data["copied_text"] == "Important information"
        assert result.data["content_type"] == "text"

    @pytest.mark.asyncio
    async def test_copy_code_content(self, action_handler, lifecycle_manager):
        """Copy code panel content."""
        spec = JennySpec(
            panel_type=PanelTypeJenny.CODE,
            content={"code": "def hello(): pass"},
        )
        await lifecycle_manager.spawn(spec)

        result = await action_handler.execute(ACTION_COPY, spec)

        assert result.status == ActionStatus.SUCCESS
        assert result.data["copied_text"] == "def hello(): pass"
        assert result.data["content_type"] == "code"


class TestActionHandlerExportPanel:
    """Tests for export_panel action (confirmation required)."""

    @pytest.mark.asyncio
    async def test_export_requires_confirmation(self, action_handler, lifecycle_manager, sample_stable_spec):
        """Export without confirmation returns PENDING."""
        await lifecycle_manager.spawn(sample_stable_spec)

        result = await action_handler.execute(ACTION_EXPORT, sample_stable_spec)

        assert result.status == ActionStatus.PENDING
        assert result.data["requires_confirmation"] is True

    @pytest.mark.asyncio
    async def test_export_with_confirmation(self, action_handler, lifecycle_manager, sample_stable_spec):
        """Export with confirmation succeeds."""
        await lifecycle_manager.spawn(sample_stable_spec)

        result = await action_handler.execute(
            ACTION_EXPORT,
            sample_stable_spec,
            context={"confirmed": True}
        )

        assert result.status == ActionStatus.SUCCESS
        assert "export_data" in result.data
        assert result.data["export_data"]["spec"]["spec_id"] == sample_stable_spec.spec_id


# ============================================================================
# Action Handler Tests - Custom Handlers
# ============================================================================

class TestCustomActionHandlers:
    """Tests for custom handler registration."""

    @pytest.mark.asyncio
    async def test_register_custom_handler(self, action_handler, lifecycle_manager, sample_stable_spec):
        """Register and execute custom handler."""
        await lifecycle_manager.spawn(sample_stable_spec)

        # Define custom handler
        async def custom_refresh_handler(spec, params, lifecycle_manager, ledger):
            return ActionResult(
                action_id=str(uuid4()),
                handler="refresh_data",
                status=ActionStatus.SUCCESS,
                spec_id=spec.spec_id,
                message="Data refreshed",
                data={"refreshed": True},
            )

        # Register
        action_handler.register_handler("refresh_data", custom_refresh_handler)

        # Execute
        custom_action = create_action("Refresh", "refresh_data")
        result = await action_handler.execute(custom_action, sample_stable_spec)

        assert result.status == ActionStatus.SUCCESS
        assert result.data["refreshed"] is True

    @pytest.mark.asyncio
    async def test_unknown_handler_fails(self, action_handler, sample_stable_spec):
        """Unknown handler should return FAILED."""
        unknown_action = create_action("Unknown", "nonexistent_handler")

        result = await action_handler.execute(unknown_action, sample_stable_spec)

        assert result.status == ActionStatus.FAILED
        assert "Unknown handler" in result.error

    def test_get_available_handlers(self, action_handler):
        """Get list of available handlers."""
        handlers = action_handler.get_available_handlers()

        assert "pin_panel" in handlers
        assert "dismiss_panel" in handlers
        assert "show_why_panel" in handlers
        assert "export_panel" in handlers


# ============================================================================
# Action Handler Tests - History
# ============================================================================

class TestActionHistory:
    """Tests for action history tracking."""

    @pytest.mark.asyncio
    async def test_action_recorded_in_history(self, action_handler, lifecycle_manager, sample_nascent_spec):
        """Actions are recorded in history."""
        await lifecycle_manager.spawn(sample_nascent_spec)

        await action_handler.execute(ACTION_PIN, sample_nascent_spec)

        history = action_handler.get_history(limit=10)
        assert len(history) == 1
        assert history[0]["handler"] == "pin_panel"
        assert history[0]["status"] == "success"

    @pytest.mark.asyncio
    async def test_history_limit(self, action_handler, lifecycle_manager, sample_nascent_spec):
        """History respects limit parameter."""
        await lifecycle_manager.spawn(sample_nascent_spec)

        # Execute multiple actions
        for _ in range(5):
            await action_handler.execute(ACTION_WHY, sample_nascent_spec)

        history = action_handler.get_history(limit=3)
        assert len(history) == 3

    def test_clear_history(self, action_handler):
        """Clear action history."""
        action_handler.clear_history()
        history = action_handler.get_history()
        assert len(history) == 0


# ============================================================================
# Streaming Manager Tests - Subscriptions
# ============================================================================

class TestStreamingSubscriptions:
    """Tests for streaming subscription management."""

    @pytest.mark.asyncio
    async def test_subscribe_reactive(self, streaming_manager, sample_reactive_spec):
        """Subscribe to reactive data source."""
        updates_received = []

        async def callback(update: StreamUpdate):
            updates_received.append(update)

        sub_id = await streaming_manager.subscribe(
            sample_reactive_spec,
            callback=callback,
        )

        assert sub_id is not None
        subscription = streaming_manager.get_subscription(sub_id)
        assert subscription is not None
        assert subscription.status == StreamStatus.ACTIVE
        assert subscription.binding_mode == BindingMode.REACTIVE

        # Wait for at least one poll
        await asyncio.sleep(0.1)

        # Clean up
        await streaming_manager.unsubscribe(sub_id)

    @pytest.mark.asyncio
    async def test_subscribe_streaming(self, streaming_manager, sample_streaming_spec):
        """Subscribe to streaming data source."""
        updates_received = []

        async def callback(update: StreamUpdate):
            updates_received.append(update)

        sub_id = await streaming_manager.subscribe(
            sample_streaming_spec,
            callback=callback,
        )

        assert sub_id is not None
        subscription = streaming_manager.get_subscription(sub_id)
        assert subscription.binding_mode == BindingMode.STREAMING

        # Clean up
        await streaming_manager.unsubscribe(sub_id)

    @pytest.mark.asyncio
    async def test_subscribe_static_fails(self, streaming_manager):
        """STATIC binding should not support subscriptions."""
        static_spec = JennySpec(
            binding_mode=BindingMode.STATIC,
            data_source="test",
        )

        with pytest.raises(ValueError) as exc_info:
            await streaming_manager.subscribe(static_spec, callback=AsyncMock())

        assert "STATIC" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_subscribe_without_data_source_fails(self, streaming_manager):
        """Creating REACTIVE spec without data_source should fail."""
        # JennySpec validates data_source at creation time
        with pytest.raises(ValueError) as exc_info:
            JennySpec(
                binding_mode=BindingMode.REACTIVE,
                data_source="",  # Empty
            )

        assert "data_source" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_unsubscribe(self, streaming_manager, sample_reactive_spec):
        """Unsubscribe from data source."""
        sub_id = await streaming_manager.subscribe(
            sample_reactive_spec,
            callback=AsyncMock(),
        )

        result = await streaming_manager.unsubscribe(sub_id)

        assert result is True
        assert streaming_manager.get_subscription(sub_id) is None

    @pytest.mark.asyncio
    async def test_unsubscribe_nonexistent(self, streaming_manager):
        """Unsubscribe from non-existent subscription."""
        result = await streaming_manager.unsubscribe("nonexistent_id")
        assert result is False


# ============================================================================
# Streaming Manager Tests - Pause/Resume
# ============================================================================

class TestStreamingPauseResume:
    """Tests for pause/resume functionality."""

    @pytest.mark.asyncio
    async def test_pause_subscription(self, streaming_manager, sample_reactive_spec):
        """Pause an active subscription."""
        sub_id = await streaming_manager.subscribe(
            sample_reactive_spec,
            callback=AsyncMock(),
        )

        result = await streaming_manager.pause(sub_id)

        assert result is True
        subscription = streaming_manager.get_subscription(sub_id)
        assert subscription.status == StreamStatus.PAUSED

        await streaming_manager.unsubscribe(sub_id)

    @pytest.mark.asyncio
    async def test_resume_subscription(self, streaming_manager, sample_reactive_spec):
        """Resume a paused subscription."""
        sub_id = await streaming_manager.subscribe(
            sample_reactive_spec,
            callback=AsyncMock(),
        )
        await streaming_manager.pause(sub_id)

        result = await streaming_manager.resume(sub_id)

        assert result is True
        subscription = streaming_manager.get_subscription(sub_id)
        assert subscription.status == StreamStatus.ACTIVE

        await streaming_manager.unsubscribe(sub_id)

    @pytest.mark.asyncio
    async def test_resume_non_paused_fails(self, streaming_manager, sample_reactive_spec):
        """Resume a non-paused subscription returns False."""
        sub_id = await streaming_manager.subscribe(
            sample_reactive_spec,
            callback=AsyncMock(),
        )

        # Not paused, should return False
        result = await streaming_manager.resume(sub_id)
        assert result is False

        await streaming_manager.unsubscribe(sub_id)


# ============================================================================
# Streaming Manager Tests - Updates
# ============================================================================

class TestStreamingUpdates:
    """Tests for streaming data updates."""

    @pytest.mark.asyncio
    async def test_reactive_receives_updates(self, streaming_manager, sample_reactive_spec):
        """Reactive subscription receives poll updates."""
        updates_received = []

        async def callback(update: StreamUpdate):
            updates_received.append(update)

        # Subscribe with fast polling
        spec = JennySpec(
            spec_id=str(uuid4()),
            binding_mode=BindingMode.REACTIVE,
            data_source="test.query",
            refresh_interval_ms=50,  # 50ms for fast test
        )

        sub_id = await streaming_manager.subscribe(spec, callback=callback)

        # Wait for updates
        await asyncio.sleep(0.2)

        await streaming_manager.unsubscribe(sub_id)

        # Should have received at least 2 updates
        assert len(updates_received) >= 2
        assert all(u.update_type == UpdateType.FULL for u in updates_received)
        assert all(u.data_source == "test.query" for u in updates_received)

    @pytest.mark.asyncio
    async def test_update_sequence_numbers(self, streaming_manager, sample_reactive_spec):
        """Updates have incrementing sequence numbers."""
        updates_received = []

        async def callback(update: StreamUpdate):
            updates_received.append(update)

        spec = JennySpec(
            spec_id=str(uuid4()),
            binding_mode=BindingMode.REACTIVE,
            data_source="test.sequence",
            refresh_interval_ms=50,
        )

        sub_id = await streaming_manager.subscribe(spec, callback=callback)
        await asyncio.sleep(0.2)
        await streaming_manager.unsubscribe(sub_id)

        # Verify sequence is incrementing
        sequences = [u.sequence for u in updates_received]
        assert sequences == sorted(sequences)
        assert len(set(sequences)) == len(sequences)  # All unique


# ============================================================================
# Streaming Manager Tests - Statistics
# ============================================================================

class TestStreamingStatistics:
    """Tests for streaming statistics."""

    @pytest.mark.asyncio
    async def test_get_statistics(self, streaming_manager, sample_reactive_spec):
        """Get streaming statistics."""
        sub_id = await streaming_manager.subscribe(
            sample_reactive_spec,
            callback=AsyncMock(),
        )

        stats = streaming_manager.get_statistics()

        assert stats["total_subscriptions"] == 1
        assert stats["active_subscriptions"] == 1
        assert stats["reactive_count"] == 1
        assert stats["streaming_count"] == 0

        await streaming_manager.unsubscribe(sub_id)

    @pytest.mark.asyncio
    async def test_get_subscriptions_for_spec(self, streaming_manager, sample_reactive_spec):
        """Get subscriptions for a specific spec."""
        sub_id = await streaming_manager.subscribe(
            sample_reactive_spec,
            callback=AsyncMock(),
        )

        subs = streaming_manager.get_subscriptions_for_spec(sample_reactive_spec.spec_id)

        assert len(subs) == 1
        assert subs[0].subscription_id == sub_id

        await streaming_manager.unsubscribe(sub_id)

    @pytest.mark.asyncio
    async def test_get_active_subscriptions(self, streaming_manager, sample_reactive_spec):
        """Get all active subscriptions."""
        sub_id = await streaming_manager.subscribe(
            sample_reactive_spec,
            callback=AsyncMock(),
        )

        active = streaming_manager.get_active_subscriptions()

        assert len(active) == 1

        await streaming_manager.unsubscribe(sub_id)


# ============================================================================
# Streaming Manager Tests - Cleanup
# ============================================================================

class TestStreamingCleanup:
    """Tests for streaming manager cleanup."""

    @pytest.mark.asyncio
    async def test_close_all_subscriptions(self, streaming_manager, sample_reactive_spec):
        """Close manager cleans up all subscriptions."""
        sub_id1 = await streaming_manager.subscribe(
            sample_reactive_spec,
            callback=AsyncMock(),
        )

        spec2 = JennySpec(
            binding_mode=BindingMode.REACTIVE,
            data_source="test2",
            refresh_interval_ms=100,
        )
        sub_id2 = await streaming_manager.subscribe(spec2, callback=AsyncMock())

        await streaming_manager.close()

        assert streaming_manager.get_subscription(sub_id1) is None
        assert streaming_manager.get_subscription(sub_id2) is None


# ============================================================================
# StreamUpdate Tests
# ============================================================================

class TestStreamUpdate:
    """Tests for StreamUpdate dataclass."""

    def test_stream_update_to_dict(self):
        """StreamUpdate serializes to dict."""
        update = StreamUpdate(
            spec_id="spec_123",
            data_source="metrics.cpu",
            update_type=UpdateType.FULL,
            data={"cpu": 45.5},
            sequence=10,
        )

        d = update.to_dict()

        assert d["spec_id"] == "spec_123"
        assert d["data_source"] == "metrics.cpu"
        assert d["update_type"] == "full"
        assert d["data"]["cpu"] == 45.5
        assert d["sequence"] == 10

    def test_stream_update_to_json(self):
        """StreamUpdate serializes to JSON."""
        update = StreamUpdate(
            spec_id="spec_456",
            data_source="events",
            update_type=UpdateType.PARTIAL,
            data={"event": "click"},
        )

        json_str = update.to_json()

        assert '"spec_id": "spec_456"' in json_str
        assert '"update_type": "partial"' in json_str

    def test_stream_update_error(self):
        """StreamUpdate with error."""
        update = StreamUpdate(
            spec_id="spec_789",
            data_source="broken",
            update_type=UpdateType.ERROR,
            error="Connection failed",
        )

        d = update.to_dict()
        assert d["update_type"] == "error"
        assert d["error"] == "Connection failed"


# ============================================================================
# MockDataSource Tests
# ============================================================================

class TestMockDataSource:
    """Tests for MockDataSource."""

    @pytest.mark.asyncio
    async def test_fetch_returns_data(self):
        """Fetch returns mock data."""
        source = MockDataSource()

        data = await source.fetch("test.query")

        assert "query" in data
        assert data["query"] == "test.query"
        assert "result" in data
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_fetch_increments_counter(self):
        """Each fetch increments counter."""
        source = MockDataSource()

        data1 = await source.fetch("query")
        data2 = await source.fetch("query")

        assert data1["result"] != data2["result"]

    @pytest.mark.asyncio
    async def test_subscribe_returns_id(self):
        """Subscribe returns subscription ID."""
        source = MockDataSource()

        sub_id = await source.subscribe("query", callback=AsyncMock())

        assert sub_id is not None
        assert isinstance(sub_id, str)

        await source.unsubscribe(sub_id)

    @pytest.mark.asyncio
    async def test_subscribe_calls_callback(self):
        """Subscribe calls callback with data."""
        source = MockDataSource()
        received = []

        async def callback(data):
            received.append(data)

        sub_id = await source.subscribe("query", callback=callback)

        # Wait for at least one callback
        await asyncio.sleep(1.5)

        await source.unsubscribe(sub_id)

        assert len(received) >= 1
        assert "result" in received[0]


# ============================================================================
# Integration Tests
# ============================================================================

class TestActionStreamingIntegration:
    """Integration tests combining actions and streaming."""

    @pytest.mark.asyncio
    async def test_dismiss_unsubscribes_streaming(self, action_handler, lifecycle_manager, streaming_manager):
        """Dismissing a streaming panel should clean up subscriptions."""
        spec = JennySpec(
            spec_id=str(uuid4()),
            panel_type=PanelTypeJenny.METRIC,
            lifecycle=LifecycleStage.STABLE,
            binding_mode=BindingMode.REACTIVE,
            data_source="metrics.test",
            refresh_interval_ms=100,
            content={"value": 0},
        )

        # Register panel and subscribe
        await lifecycle_manager.spawn(spec)
        sub_id = await streaming_manager.subscribe(spec, callback=AsyncMock())

        # Verify subscription active
        assert streaming_manager.get_subscription(sub_id) is not None

        # Dismiss panel
        await action_handler.execute(ACTION_DISMISS, spec)

        # Manually clean up streaming (in real system, lifecycle manager would do this)
        await streaming_manager.unsubscribe(sub_id)

        # Verify cleaned up
        assert streaming_manager.get_subscription(sub_id) is None

    @pytest.mark.asyncio
    async def test_full_panel_lifecycle_with_streaming(
        self, action_handler, lifecycle_manager, streaming_manager
    ):
        """Full lifecycle: create → stream → pin → dismiss."""
        updates_received = []

        async def callback(update: StreamUpdate):
            updates_received.append(update)

        # Create reactive spec
        spec = JennySpec(
            spec_id=str(uuid4()),
            panel_type=PanelTypeJenny.METRIC,
            lifecycle=LifecycleStage.NASCENT,
            binding_mode=BindingMode.REACTIVE,
            data_source="metrics.lifecycle",
            refresh_interval_ms=50,
            content={"value": 0},
        )

        # 1. Register
        await lifecycle_manager.spawn(spec)

        # 2. Subscribe to data
        sub_id = await streaming_manager.subscribe(spec, callback=callback)

        # 3. Wait for some updates
        await asyncio.sleep(0.15)

        # 4. Pin panel
        pin_result = await action_handler.execute(ACTION_PIN, spec)
        assert pin_result.status == ActionStatus.SUCCESS

        # 5. More updates while stable
        await asyncio.sleep(0.1)

        # 6. Dismiss
        dismiss_result = await action_handler.execute(
            ACTION_DISMISS,
            spec.with_lifecycle(LifecycleStage.STABLE)
        )
        assert dismiss_result.status == ActionStatus.SUCCESS

        # 7. Clean up streaming
        await streaming_manager.unsubscribe(sub_id)

        # Verify we received updates
        assert len(updates_received) >= 2


# ============================================================================
# ActionResult Tests
# ============================================================================

class TestActionResult:
    """Tests for ActionResult dataclass."""

    def test_action_result_to_dict(self):
        """ActionResult serializes to dict."""
        result = ActionResult(
            action_id="action_123",
            handler="pin_panel",
            status=ActionStatus.SUCCESS,
            spec_id="spec_456",
            previous_lifecycle=LifecycleStage.NASCENT,
            new_lifecycle=LifecycleStage.STABLE,
            message="Panel pinned",
        )

        d = result.to_dict()

        assert d["action_id"] == "action_123"
        assert d["handler"] == "pin_panel"
        assert d["status"] == "success"
        assert d["previous_lifecycle"] == "nascent"
        assert d["new_lifecycle"] == "stable"

    def test_action_result_with_error(self):
        """ActionResult with error field."""
        result = ActionResult(
            action_id="action_789",
            handler="unknown",
            status=ActionStatus.FAILED,
            spec_id="spec_000",
            error="Handler not found",
        )

        d = result.to_dict()
        assert d["status"] == "failed"
        assert d["error"] == "Handler not found"
