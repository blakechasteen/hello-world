"""
Integration tests for WebSocket progress streaming.

Tests the JobProgressManager and JobProgressMessage for
real-time job progress updates via WebSocket.
"""

import pytest

pytest.importorskip("websockets")

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from hololoom.apps.chatops.handlers.websocket_progress import (
    JobMessageType,
    JobProgressMessage,
    JobProgressManager,
    JobProgressBroadcaster,
    get_global_manager,
    get_global_broadcaster,
)


class TestJobProgressMessage:
    """Tests for the JobProgressMessage dataclass."""

    def test_message_serialization(self):
        """Messages should serialize to dict correctly."""
        msg = JobProgressMessage(
            type=JobMessageType.JOB_PROGRESS,
            job_id="test-123",
            current_step=3,
            total_steps=9,
            step_name="Retrieval",
            message="Retrieving context..."
        )
        data = msg.to_dict()

        assert data["type"] == "job_progress"
        assert data["job_id"] == "test-123"
        assert data["current_step"] == 3
        assert data["total_steps"] == 9
        assert data["step_name"] == "Retrieval"
        assert "timestamp" in data

    def test_message_to_json(self):
        """Messages should serialize to JSON correctly."""
        msg = JobProgressMessage(
            type=JobMessageType.JOB_COMPLETED,
            job_id="abc",
            duration_ms=1500.0,
            confidence=0.92
        )
        json_str = msg.to_json()

        assert '"type": "job_completed"' in json_str
        assert '"job_id": "abc"' in json_str
        assert '"duration_ms": 1500.0' in json_str
        assert '"confidence": 0.92' in json_str

    def test_none_values_excluded(self):
        """None values should be excluded from dict."""
        msg = JobProgressMessage(
            type=JobMessageType.JOB_STARTED,
            job_id="test",
            tool_used=None,
            error=None
        )
        data = msg.to_dict()

        # None values should not be in dict
        assert "tool_used" not in data
        assert "error" not in data

    def test_message_types(self):
        """All message types should be valid."""
        types = [
            JobMessageType.JOB_QUEUED,
            JobMessageType.JOB_STARTED,
            JobMessageType.JOB_PROGRESS,
            JobMessageType.JOB_COMPLETED,
            JobMessageType.JOB_FAILED,
            JobMessageType.JOB_CANCELLED,
            JobMessageType.HEARTBEAT,
        ]

        for msg_type in types:
            msg = JobProgressMessage(type=msg_type, job_id="test")
            assert msg.type == msg_type
            assert msg.to_dict()["type"] == msg_type.value


class TestJobProgressManager:
    """Tests for the JobProgressManager component."""

    @pytest.mark.asyncio
    async def test_connect_and_disconnect(self):
        """Connections should be tracked correctly."""
        manager = JobProgressManager()
        mock_ws = AsyncMock()

        # Connect
        await manager.connect(mock_ws, "conn1")
        assert "conn1" in manager._connections
        mock_ws.accept.assert_called_once()

        # Disconnect
        await manager.disconnect("conn1")
        assert "conn1" not in manager._connections

    @pytest.mark.asyncio
    async def test_subscribe_patterns(self):
        """Subscriptions should be tracked by pattern."""
        manager = JobProgressManager()
        mock_ws = AsyncMock()

        await manager.connect(mock_ws, "conn1")
        manager.subscribe("conn1", "job:test-123")

        assert "job:test-123" in manager._subscriptions["conn1"]
        assert "conn1" in manager._pattern_subscribers["job:test-123"]

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        """Unsubscribe should remove patterns correctly."""
        manager = JobProgressManager()
        mock_ws = AsyncMock()

        await manager.connect(mock_ws, "conn1")
        manager.subscribe("conn1", "job:test-123")
        manager.unsubscribe("conn1", "job:test-123")

        assert "job:test-123" not in manager._subscriptions["conn1"]
        assert "job:test-123" not in manager._pattern_subscribers

    @pytest.mark.asyncio
    async def test_broadcast_to_subscribers(self):
        """Broadcasts should only go to subscribers."""
        manager = JobProgressManager()
        conn1 = AsyncMock()
        conn2 = AsyncMock()

        await manager.connect(conn1, "conn1")
        await manager.connect(conn2, "conn2")
        manager.subscribe("conn1", "job:test-123")
        # conn2 not subscribed to job:test-123

        count = await manager.broadcast_progress(
            job_id="test-123",
            message_type=JobMessageType.JOB_PROGRESS,
            current_step=3,
            step_name="Testing"
        )

        # conn1 should receive welcome + broadcast, conn2 only welcome
        # (connect() sends a welcome message)
        assert conn1.send_json.call_count == 2  # welcome + broadcast
        assert conn2.send_json.call_count == 1  # welcome only

        # Verify the broadcast message was sent to conn1
        broadcast_call = conn1.send_json.call_args_list[-1]
        assert broadcast_call[0][0]["type"] == "job_progress"
        assert broadcast_call[0][0]["job_id"] == "test-123"
        assert count == 1

    @pytest.mark.asyncio
    async def test_wildcard_subscription(self):
        """Wildcard '*' should receive all messages."""
        manager = JobProgressManager()
        mock_ws = AsyncMock()

        await manager.connect(mock_ws, "conn1")
        manager.subscribe("conn1", "*")

        await manager.broadcast_progress(
            job_id="any-job-id",
            message_type=JobMessageType.JOB_STARTED
        )

        # connect() sends welcome, then broadcast sends job_started
        assert mock_ws.send_json.call_count == 2

        # Verify the broadcast message
        broadcast_call = mock_ws.send_json.call_args_list[-1]
        assert broadcast_call[0][0]["type"] == "job_started"

    @pytest.mark.asyncio
    async def test_room_subscription(self):
        """Room subscription should receive room's jobs."""
        manager = JobProgressManager()
        mock_ws = AsyncMock()

        await manager.connect(mock_ws, "conn1")
        manager.subscribe("conn1", "room:!abc123:matrix.org")

        await manager.broadcast_progress(
            job_id="job1",
            message_type=JobMessageType.JOB_PROGRESS,
            room_id="!abc123:matrix.org"
        )

        # connect() sends welcome, then broadcast sends job_progress
        assert mock_ws.send_json.call_count == 2

        # Verify the broadcast message
        broadcast_call = mock_ws.send_json.call_args_list[-1]
        assert broadcast_call[0][0]["type"] == "job_progress"
        assert broadcast_call[0][0]["room_id"] == "!abc123:matrix.org"

    @pytest.mark.asyncio
    async def test_user_subscription(self):
        """User subscription should receive user's jobs."""
        manager = JobProgressManager()
        mock_ws = AsyncMock()

        await manager.connect(mock_ws, "conn1")
        manager.subscribe("conn1", "user:@alice:matrix.org")

        await manager.broadcast_progress(
            job_id="job1",
            message_type=JobMessageType.JOB_PROGRESS,
            user_id="@alice:matrix.org"
        )

        # connect() sends welcome, then broadcast sends job_progress
        assert mock_ws.send_json.call_count == 2

        # Verify the broadcast message
        broadcast_call = mock_ws.send_json.call_args_list[-1]
        assert broadcast_call[0][0]["type"] == "job_progress"
        assert broadcast_call[0][0]["user_id"] == "@alice:matrix.org"

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Stats should reflect current manager state."""
        manager = JobProgressManager()
        mock_ws = AsyncMock()

        await manager.connect(mock_ws, "conn1")
        manager.subscribe("conn1", "job:test-123")
        manager.subscribe("conn1", "room:!abc:matrix.org")

        stats = manager.get_stats()

        assert stats["active_connections"] == 1
        assert stats["total_subscriptions"] == 2
        assert "job:test-123" in stats["patterns"]
        assert "room:!abc:matrix.org" in stats["patterns"]

    @pytest.mark.asyncio
    async def test_clear_job_buffer(self):
        """Clear buffer should remove job's buffered messages."""
        manager = JobProgressManager()

        # Simulate buffered messages
        manager._message_buffer["job:test-123"] = [
            JobProgressMessage(type=JobMessageType.JOB_PROGRESS, job_id="test-123")
        ]

        manager.clear_job_buffer("test-123")

        assert "job:test-123" not in manager._message_buffer


class TestJobProgressBroadcaster:
    """Tests for the JobProgressBroadcaster helper."""

    @pytest.mark.asyncio
    async def test_job_queued(self):
        """job_queued should broadcast correct message type."""
        manager = JobProgressManager()
        mock_ws = AsyncMock()
        await manager.connect(mock_ws, "conn1")
        manager.subscribe("conn1", "*")

        broadcaster = JobProgressBroadcaster(manager)
        await broadcaster.job_queued("job1", "test query", "!room:mx", "@user:mx")

        # Verify message was sent
        mock_ws.send_json.assert_called()
        call_args = mock_ws.send_json.call_args[0][0]
        assert call_args["type"] == "job_queued"
        assert call_args["job_id"] == "job1"

    @pytest.mark.asyncio
    async def test_job_started(self):
        """job_started should broadcast correct message type."""
        manager = JobProgressManager()
        mock_ws = AsyncMock()
        await manager.connect(mock_ws, "conn1")
        manager.subscribe("conn1", "*")

        broadcaster = JobProgressBroadcaster(manager)
        await broadcaster.job_started("job1", "test query", "!room:mx", "@user:mx")

        call_args = mock_ws.send_json.call_args[0][0]
        assert call_args["type"] == "job_started"

    @pytest.mark.asyncio
    async def test_job_progress(self):
        """job_progress should broadcast step info."""
        manager = JobProgressManager()
        mock_ws = AsyncMock()
        await manager.connect(mock_ws, "conn1")
        manager.subscribe("conn1", "*")

        broadcaster = JobProgressBroadcaster(manager)
        await broadcaster.job_progress(
            job_id="job1",
            current_step=5,
            step_name="Decision",
            message="Making decision..."
        )

        call_args = mock_ws.send_json.call_args[0][0]
        assert call_args["type"] == "job_progress"
        assert call_args["current_step"] == 5
        assert call_args["step_name"] == "Decision"

    @pytest.mark.asyncio
    async def test_job_completed(self):
        """job_completed should broadcast results and clear buffer."""
        manager = JobProgressManager()
        mock_ws = AsyncMock()
        await manager.connect(mock_ws, "conn1")
        manager.subscribe("conn1", "*")

        # Add to buffer
        manager._message_buffer["job:job1"] = [
            JobProgressMessage(type=JobMessageType.JOB_PROGRESS, job_id="job1")
        ]

        broadcaster = JobProgressBroadcaster(manager)
        await broadcaster.job_completed(
            job_id="job1",
            tool_used="answer",
            confidence=0.92,
            duration_ms=1500.0
        )

        call_args = mock_ws.send_json.call_args[0][0]
        assert call_args["type"] == "job_completed"
        assert call_args["tool_used"] == "answer"
        assert call_args["confidence"] == 0.92
        assert call_args["duration_ms"] == 1500.0

        # Buffer should be cleared
        assert "job:job1" not in manager._message_buffer

    @pytest.mark.asyncio
    async def test_job_failed(self):
        """job_failed should broadcast error and clear buffer."""
        manager = JobProgressManager()
        mock_ws = AsyncMock()
        await manager.connect(mock_ws, "conn1")
        manager.subscribe("conn1", "*")

        broadcaster = JobProgressBroadcaster(manager)
        await broadcaster.job_failed("job1", "ValueError: test error")

        call_args = mock_ws.send_json.call_args[0][0]
        assert call_args["type"] == "job_failed"
        assert "ValueError" in call_args["error"]

    @pytest.mark.asyncio
    async def test_job_cancelled(self):
        """job_cancelled should broadcast and clear buffer."""
        manager = JobProgressManager()
        mock_ws = AsyncMock()
        await manager.connect(mock_ws, "conn1")
        manager.subscribe("conn1", "*")

        broadcaster = JobProgressBroadcaster(manager)
        await broadcaster.job_cancelled("job1")

        call_args = mock_ws.send_json.call_args[0][0]
        assert call_args["type"] == "job_cancelled"


class TestGlobalManager:
    """Tests for global manager singleton."""

    def test_get_global_manager(self):
        """get_global_manager should return singleton."""
        manager1 = get_global_manager()
        manager2 = get_global_manager()

        assert manager1 is manager2
        assert isinstance(manager1, JobProgressManager)

    def test_get_global_broadcaster(self):
        """get_global_broadcaster should use global manager."""
        broadcaster = get_global_broadcaster()

        assert isinstance(broadcaster, JobProgressBroadcaster)
        assert broadcaster.manager is get_global_manager()


class TestMessageBuffering:
    """Tests for message buffering for late subscribers."""

    @pytest.mark.asyncio
    async def test_messages_buffered(self):
        """Messages should be buffered for late subscribers."""
        manager = JobProgressManager()

        # Broadcast without subscribers (should buffer)
        await manager.broadcast_progress(
            job_id="test-123",
            message_type=JobMessageType.JOB_PROGRESS,
            current_step=3
        )

        assert "job:test-123" in manager._message_buffer
        assert len(manager._message_buffer["job:test-123"]) == 1

    @pytest.mark.asyncio
    async def test_buffer_limit(self):
        """Buffer should respect max size."""
        manager = JobProgressManager()
        manager._buffer_size = 10

        # Broadcast more than buffer size
        for i in range(15):
            await manager.broadcast_progress(
                job_id="test-123",
                message_type=JobMessageType.JOB_PROGRESS,
                current_step=i
            )

        # Should only keep last 10
        assert len(manager._message_buffer["job:test-123"]) == 10
