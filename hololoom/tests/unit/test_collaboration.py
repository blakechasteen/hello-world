"""
Comprehensive Unit Tests for Collaboration Modules.

Tests Phase 3: Multi-User Collaboration infrastructure:
- session.py - Session lifecycle management
- presence.py - Real-time presence tracking
- sync.py - State synchronization (CRDT-inspired)
- attribution.py - Contribution tracking
- voice.py - Voice room management

Created: December 2025
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Session module
from hololoom.collaboration.session import (
    Session, SessionManager, Participant, SessionSettings, JoinRequest,
    SessionState, SessionType, ParticipantRole,
    create_session_manager
)

# Presence module
from hololoom.collaboration.presence import (
    PresenceManager, UserPresence, CursorPosition, SelectionState, TypingIndicator,
    ActivityStatus, FocusType,
    create_presence_manager
)

# Sync module
from hololoom.collaboration.sync import (
    StateSynchronizer, Operation, Conflict, SyncState, OperationBuffer,
    OperationType, ConflictResolution,
    create_state_synchronizer
)

# Attribution module
from hololoom.collaboration.attribution import (
    AttributionManager, Contribution, UserContributionStats, AttributionContext,
    ContributionType, QualityRating,
    create_attribution_manager
)

# Voice module
from hololoom.collaboration.voice import (
    VoiceRoom, VoiceManager, VoiceRoomParticipant, VoiceRoomSettings,
    PeerConnection, MediaTrack, SignalingMessage,
    MediaType, StreamQuality, ConnectionState, SignalingType,
    create_voice_room, create_voice_manager
)


# ==============================================================================
# Session Module Tests
# ==============================================================================

class TestSession:
    """Tests for Session class."""

    def test_session_creation(self):
        """Test basic session creation."""
        session = Session(
            session_id="test_session_001",
            name="Test Session",
            session_type=SessionType.KNOWLEDGE_BASE,
            owner_id="user_001"
        )

        assert session.session_id == "test_session_001"
        assert session.name == "Test Session"
        assert session.session_type == SessionType.KNOWLEDGE_BASE
        assert session.owner_id == "user_001"
        assert session.state == SessionState.INITIALIZING
        assert len(session.participants) == 0

    def test_add_participant(self):
        """Test adding participants to session."""
        session = Session(
            session_id="test_session_002",
            name="Participant Test",
            session_type=SessionType.WHITEBOARD,
            owner_id="owner"
        )

        participant = Participant(
            user_id="user_001",
            display_name="Test User",
            role=ParticipantRole.EDITOR
        )

        result = session.add_participant(participant)

        assert result is True
        assert "user_001" in session.participants
        assert session.participants["user_001"].display_name == "Test User"

    def test_participant_max_limit(self):
        """Test session respects max participant limit."""
        settings = SessionSettings(max_participants=2)
        session = Session(
            session_id="test_session_003",
            name="Limited Session",
            session_type=SessionType.RESEARCH,
            owner_id="owner",
            settings=settings
        )

        # Add first participant
        p1 = Participant(user_id="user_001", display_name="User 1", role=ParticipantRole.EDITOR)
        assert session.add_participant(p1) is True

        # Add second participant
        p2 = Participant(user_id="user_002", display_name="User 2", role=ParticipantRole.EDITOR)
        assert session.add_participant(p2) is True

        # Third should fail
        p3 = Participant(user_id="user_003", display_name="User 3", role=ParticipantRole.EDITOR)
        assert session.add_participant(p3) is False

    def test_remove_participant(self):
        """Test removing participants."""
        session = Session(
            session_id="test_session_004",
            name="Remove Test",
            session_type=SessionType.REVIEW,
            owner_id="owner"
        )

        p1 = Participant(user_id="user_001", display_name="User 1", role=ParticipantRole.EDITOR)
        session.add_participant(p1)

        result = session.remove_participant("user_001")
        assert result is True
        assert "user_001" not in session.participants

    def test_ownership_transfer_on_leave(self):
        """Test ownership transfers when owner leaves."""
        session = Session(
            session_id="test_session_005",
            name="Ownership Test",
            session_type=SessionType.PRESENTATION,
            owner_id="owner"
        )

        # Add owner and another user
        owner = Participant(user_id="owner", display_name="Owner", role=ParticipantRole.OWNER)
        user = Participant(user_id="user_001", display_name="User", role=ParticipantRole.EDITOR)
        session.add_participant(owner)
        session.add_participant(user)

        # Owner leaves
        session.remove_participant("owner")

        # Ownership should transfer
        assert session.owner_id == "user_001"
        assert session.participants["user_001"].role == ParticipantRole.OWNER

    def test_role_hierarchy_permissions(self):
        """Test role-based permission checking."""
        session = Session(
            session_id="test_session_006",
            name="Permission Test",
            session_type=SessionType.KNOWLEDGE_BASE,
            owner_id="owner"
        )

        viewer = Participant(user_id="viewer", display_name="Viewer", role=ParticipantRole.VIEWER)
        editor = Participant(user_id="editor", display_name="Editor", role=ParticipantRole.EDITOR)
        admin = Participant(user_id="admin", display_name="Admin", role=ParticipantRole.ADMIN)

        session.add_participant(viewer)
        session.add_participant(editor)
        session.add_participant(admin)

        # Check permissions
        assert session.has_permission("viewer", ParticipantRole.VIEWER) is True
        assert session.has_permission("viewer", ParticipantRole.EDITOR) is False

        assert session.has_permission("editor", ParticipantRole.VIEWER) is True
        assert session.has_permission("editor", ParticipantRole.EDITOR) is True
        assert session.has_permission("editor", ParticipantRole.ADMIN) is False

        assert session.has_permission("admin", ParticipantRole.ADMIN) is True

    def test_join_request_flow(self):
        """Test join request with approval."""
        settings = SessionSettings(require_approval=True)
        session = Session(
            session_id="test_session_007",
            name="Approval Test",
            session_type=SessionType.RESEARCH,
            owner_id="owner",
            settings=settings
        )

        request = JoinRequest(
            user_id="user_001",
            display_name="New User",
            requested_role=ParticipantRole.EDITOR,
            message="Please let me join"
        )

        # Add request (should NOT auto-join when require_approval=True)
        session.add_join_request(request)
        assert "user_001" in session.join_requests
        assert "user_001" not in session.participants

        # Approve request
        session.approve_join_request("user_001")
        assert "user_001" not in session.join_requests
        assert "user_001" in session.participants

    def test_session_serialization(self):
        """Test session to_dict serialization."""
        session = Session(
            session_id="test_session_008",
            name="Serialize Test",
            session_type=SessionType.WHITEBOARD,
            owner_id="owner",
            description="Test description",
            tags=["test", "unit"]
        )

        p1 = Participant(user_id="user_001", display_name="User", role=ParticipantRole.EDITOR)
        session.add_participant(p1)

        data = session.to_dict()

        assert data["session_id"] == "test_session_008"
        assert data["name"] == "Serialize Test"
        assert data["session_type"] == "whiteboard"
        assert "user_001" in data["participants"]
        assert data["description"] == "Test description"


class TestSessionManager:
    """Tests for SessionManager class."""

    @pytest.mark.asyncio
    async def test_create_session(self):
        """Test creating a session via manager."""
        manager = create_session_manager()

        session = await manager.create_session(
            name="Manager Test",
            owner_id="owner",
            owner_name="Owner Name",
            session_type=SessionType.KNOWLEDGE_BASE
        )

        assert session is not None
        assert session.state == SessionState.ACTIVE
        assert "owner" in session.participants
        assert session.participants["owner"].role == ParticipantRole.OWNER

    @pytest.mark.asyncio
    async def test_join_session(self):
        """Test joining an existing session."""
        manager = create_session_manager()

        session = await manager.create_session(
            name="Join Test",
            owner_id="owner",
            owner_name="Owner"
        )

        result = await manager.join_session(
            session_id=session.session_id,
            user_id="joiner",
            display_name="Joiner Name"
        )

        assert result is not None
        assert "joiner" in result.participants

    @pytest.mark.asyncio
    async def test_leave_session(self):
        """Test leaving a session."""
        manager = create_session_manager()

        session = await manager.create_session(
            name="Leave Test",
            owner_id="owner",
            owner_name="Owner"
        )

        await manager.join_session(session.session_id, "user", "User")
        result = await manager.leave_session(session.session_id, "user")

        assert result is True
        assert "user" not in session.participants

    @pytest.mark.asyncio
    async def test_close_session(self):
        """Test closing a session."""
        manager = create_session_manager()

        session = await manager.create_session(
            name="Close Test",
            owner_id="owner",
            owner_name="Owner"
        )
        session_id = session.session_id

        result = await manager.close_session(session_id, "owner")

        assert result is True
        assert session_id not in manager.sessions

    @pytest.mark.asyncio
    async def test_list_public_sessions(self):
        """Test listing available sessions."""
        manager = create_session_manager()

        await manager.create_session("Session 1", "owner1", "Owner 1", SessionType.RESEARCH)
        await manager.create_session("Session 2", "owner2", "Owner 2", SessionType.WHITEBOARD)
        await manager.create_session("Session 3", "owner3", "Owner 3", SessionType.RESEARCH)

        # All sessions
        all_sessions = await manager.list_public_sessions()
        assert len(all_sessions) == 3

        # Filter by type
        research = await manager.list_public_sessions(session_type=SessionType.RESEARCH)
        assert len(research) == 2


# ==============================================================================
# Presence Module Tests
# ==============================================================================

class TestUserPresence:
    """Tests for UserPresence class."""

    def test_presence_creation(self):
        """Test creating user presence."""
        presence = UserPresence(
            user_id="user_001",
            display_name="Test User"
        )

        assert presence.user_id == "user_001"
        assert presence.status == ActivityStatus.ONLINE
        assert presence.focus == FocusType.NONE
        assert presence.cursor is None

    def test_cursor_update(self):
        """Test cursor position updates."""
        presence = UserPresence(user_id="user_001", display_name="Test")

        presence.update_cursor(100.0, 200.0, z=0.0, viewport_id="main")

        assert presence.cursor is not None
        assert presence.cursor.x == 100.0
        assert presence.cursor.y == 200.0
        assert presence.cursor.viewport_id == "main"
        assert presence.status == ActivityStatus.ACTIVE

    def test_focus_update(self):
        """Test focus tracking."""
        presence = UserPresence(user_id="user_001", display_name="Test")

        presence.update_focus(FocusType.KNOWLEDGE_GRAPH, element_id="node_123")

        assert presence.focus == FocusType.KNOWLEDGE_GRAPH
        assert presence.focus_element_id == "node_123"

    def test_typing_indicator(self):
        """Test typing indicator lifecycle."""
        presence = UserPresence(user_id="user_001", display_name="Test")

        presence.start_typing("query")

        assert presence.typing is not None
        assert presence.typing.context == "query"
        assert presence.typing.is_active()

        presence.stop_typing()
        assert presence.typing is None

    def test_idle_detection(self):
        """Test idle status detection."""
        presence = UserPresence(user_id="user_001", display_name="Test")
        presence.status = ActivityStatus.ACTIVE

        # Fake old activity
        presence.last_activity = datetime.now() - timedelta(minutes=10)

        result = presence.check_idle(idle_threshold=timedelta(minutes=5))

        assert result is True
        assert presence.status == ActivityStatus.IDLE

    def test_manual_status_override(self):
        """Test manual status setting doesn't get auto-changed."""
        presence = UserPresence(user_id="user_001", display_name="Test")

        presence.set_status(ActivityStatus.DO_NOT_DISTURB, "In a meeting")

        assert presence.status == ActivityStatus.DO_NOT_DISTURB
        assert presence.custom_status == "In a meeting"

        # Activity shouldn't change DND status
        presence.update_cursor(50, 50)
        assert presence.status == ActivityStatus.DO_NOT_DISTURB


class TestPresenceManager:
    """Tests for PresenceManager class."""

    def test_manager_creation(self):
        """Test presence manager creation."""
        manager = create_presence_manager("session_001")

        assert manager.session_id == "session_001"
        assert len(manager.presences) == 0

    def test_add_user(self):
        """Test adding user to presence tracking."""
        manager = create_presence_manager("session_001")

        presence = manager.add_user("user_001", "Test User")

        assert presence is not None
        assert presence.user_id == "user_001"
        assert presence.cursor_color is not None  # Should have assigned color
        assert "user_001" in manager.presences

    def test_remove_user(self):
        """Test removing user from presence."""
        manager = create_presence_manager("session_001")
        manager.add_user("user_001", "Test")

        result = manager.remove_user("user_001")

        assert result is True
        assert "user_001" not in manager.presences

    def test_cursor_updates(self):
        """Test cursor position updates via manager."""
        manager = create_presence_manager("session_001")
        manager.add_user("user_001", "Test")

        result = manager.update_cursor("user_001", 150.0, 250.0)

        assert result is True
        presence = manager.get_presence("user_001")
        assert presence.cursor.x == 150.0
        assert presence.cursor.y == 250.0

    def test_focus_tracking(self):
        """Test focus updates via manager."""
        manager = create_presence_manager("session_001")
        manager.add_user("user_001", "Test")

        manager.update_focus("user_001", FocusType.WHITEBOARD, "wb_001")

        presence = manager.get_presence("user_001")
        assert presence.focus == FocusType.WHITEBOARD
        assert presence.focus_element_id == "wb_001"

    def test_typing_state(self):
        """Test typing indicator management."""
        manager = create_presence_manager("session_001")
        manager.add_user("user_001", "Test")

        manager.set_typing("user_001", "comment", True)
        typists = manager.get_active_typists()
        assert ("user_001", "comment") in typists

        manager.set_typing("user_001", "comment", False)
        typists = manager.get_active_typists()
        assert len(typists) == 0

    def test_cursor_overlays(self):
        """Test getting cursor overlay data."""
        manager = create_presence_manager("session_001")
        manager.add_user("user_001", "User 1")
        manager.add_user("user_002", "User 2")

        manager.update_cursor("user_001", 100, 100)
        manager.update_cursor("user_002", 200, 200)

        overlays = manager.get_cursor_overlays()

        assert len(overlays) == 2
        assert all("cursor" in o for o in overlays)

    def test_color_palette_rotation(self):
        """Test unique colors assigned to users."""
        manager = create_presence_manager("session_001")

        colors = set()
        for i in range(5):
            p = manager.add_user(f"user_{i}", f"User {i}")
            colors.add(p.cursor_color)

        # All colors should be unique (if palette size >= 5)
        assert len(colors) == 5


# ==============================================================================
# Sync Module Tests
# ==============================================================================

class TestOperation:
    """Tests for Operation class."""

    def test_operation_creation(self):
        """Test creating an operation."""
        op = Operation(
            op_id="op_001",
            op_type=OperationType.INSERT,
            target_type="node",
            target_id="node_001",
            user_id="user_001",
            timestamp=datetime.now(),
            data={"content": "Test content"}
        )

        assert op.op_id == "op_001"
        assert op.op_type == OperationType.INSERT
        assert op.applied is False
        assert op.confirmed is False

    def test_operation_serialization(self):
        """Test operation to_dict and from_dict."""
        op = Operation(
            op_id="op_002",
            op_type=OperationType.UPDATE,
            target_type="edge",
            target_id="edge_001",
            user_id="user_001",
            timestamp=datetime.now(),
            data={"weight": 0.5},
            sequence_number=42,
            vector_clock={"user_001": 42}
        )

        data = op.to_dict()
        restored = Operation.from_dict(data)

        assert restored.op_id == op.op_id
        assert restored.op_type == op.op_type
        assert restored.sequence_number == 42


class TestOperationBuffer:
    """Tests for OperationBuffer class."""

    def test_buffer_add_and_confirm(self):
        """Test adding and confirming operations."""
        buffer = OperationBuffer(max_size=100)

        op = Operation(
            op_id="op_001",
            op_type=OperationType.INSERT,
            target_type="node",
            target_id="node_001",
            user_id="user",
            timestamp=datetime.now(),
            data={}
        )

        buffer.add(op)
        assert "op_001" in buffer.pending

        confirmed = buffer.confirm("op_001")
        assert confirmed is not None
        assert confirmed.confirmed is True
        assert "op_001" not in buffer.pending

    def test_buffer_max_size(self):
        """Test buffer respects max size."""
        buffer = OperationBuffer(max_size=2)

        for i in range(3):
            op = Operation(
                op_id=f"op_{i:03d}",
                op_type=OperationType.UPDATE,
                target_type="test",
                target_id="test",
                user_id="user",
                timestamp=datetime.now(),
                data={}
            )
            buffer.add(op)

        # Should only have 2 operations (oldest evicted)
        assert len(buffer.pending) == 2


class TestStateSynchronizer:
    """Tests for StateSynchronizer class."""

    def test_synchronizer_creation(self):
        """Test creating a synchronizer."""
        sync = create_state_synchronizer("session_001", "user_001")

        assert sync.session_id == "session_001"
        assert sync.user_id == "user_001"
        assert sync.sequence_number == 0

    def test_create_operation(self):
        """Test operation creation with vector clock."""
        sync = create_state_synchronizer("session_001", "user_001")

        op = sync.create_operation(
            op_type=OperationType.INSERT,
            target_type="node",
            target_id="node_001",
            data={"content": "Test"}
        )

        assert op.sequence_number == 1
        assert op.user_id == "user_001"
        assert op.vector_clock["user_001"] == 1

    def test_lock_acquisition(self):
        """Test locking mechanism."""
        sync = create_state_synchronizer("session_001", "user_001")

        # Acquire lock
        result = sync.acquire_lock("target_001", "node", duration_seconds=30)
        assert result is True

        # Check state
        state = sync.sync_states.get("target_001")
        assert state is not None
        assert state.locked_by == "user_001"
        assert state.is_locked() is True

    def test_lock_release(self):
        """Test lock release."""
        sync = create_state_synchronizer("session_001", "user_001")

        sync.acquire_lock("target_001", "node")
        result = sync.release_lock("target_001")

        assert result is True
        state = sync.sync_states.get("target_001")
        assert state.is_locked() is False

    def test_conflict_detection(self):
        """Test conflict detection between operations."""
        sync = create_state_synchronizer("session_001", "user_001")

        # Create two conflicting operations (same target, overlapping fields)
        op_a = Operation(
            op_id="op_a",
            op_type=OperationType.UPDATE,
            target_type="node",
            target_id="node_001",
            user_id="user_001",
            timestamp=datetime.now(),
            data={"field1": "value_a"}
        )

        op_b = Operation(
            op_id="op_b",
            op_type=OperationType.UPDATE,
            target_type="node",
            target_id="node_001",
            user_id="user_002",
            timestamp=datetime.now(),
            data={"field1": "value_b"}  # Same field!
        )

        conflicts = sync._detect_conflicts(op_a, [op_b])

        assert len(conflicts) == 1
        assert conflicts[0].target_id == "node_001"


# ==============================================================================
# Attribution Module Tests
# ==============================================================================

class TestContribution:
    """Tests for Contribution class."""

    def test_contribution_creation(self):
        """Test creating a contribution."""
        contrib = Contribution(
            contribution_id="contrib_001",
            contribution_type=ContributionType.KNOWLEDGE_ADD,
            user_id="user_001",
            target_type="memory",
            target_id="mem_001",
            content_preview="Test content",
            content_size=100
        )

        assert contrib.contribution_id == "contrib_001"
        assert contrib.contribution_type == ContributionType.KNOWLEDGE_ADD
        assert contrib.use_count == 0

    def test_impact_score_calculation(self):
        """Test impact score calculation."""
        contrib = Contribution(
            contribution_id="contrib_002",
            contribution_type=ContributionType.KNOWLEDGE_ADD,
            user_id="user_001",
            target_type="memory",
            target_id="mem_001",
            use_count=50,
            citation_count=25,
            upvotes=10,
            downvotes=2,
            view_count=200
        )

        impact = contrib.calculate_impact_score()

        assert 0.0 <= impact <= 1.0
        assert impact > 0.3  # Should have reasonable impact with these stats


class TestAttributionManager:
    """Tests for AttributionManager class."""

    def test_manager_creation(self):
        """Test creating attribution manager."""
        manager = create_attribution_manager("kb_001")

        assert manager.knowledge_base_id == "kb_001"
        assert len(manager.contributions) == 0

    def test_record_contribution(self):
        """Test recording a contribution."""
        manager = create_attribution_manager()

        contrib = manager.record_contribution(
            contribution_type=ContributionType.KNOWLEDGE_ADD,
            user_id="user_001",
            user_name="Test User",
            target_type="memory",
            target_id="mem_001",
            content_preview="Test content",
            content_size=150
        )

        assert contrib is not None
        assert contrib.contribution_id in manager.contributions
        assert "user_001" in manager.user_contributions

    def test_rate_contribution(self):
        """Test rating a contribution."""
        manager = create_attribution_manager()

        contrib = manager.record_contribution(
            contribution_type=ContributionType.KNOWLEDGE_ADD,
            user_id="user_001",
            user_name="User",
            target_type="memory",
            target_id="mem_001"
        )

        result = manager.rate_contribution(
            contrib.contribution_id,
            QualityRating.EXCELLENT,
            "rater_001"
        )

        assert result is True
        assert contrib.quality_rating == QualityRating.EXCELLENT

    def test_vote_contribution(self):
        """Test upvoting/downvoting."""
        manager = create_attribution_manager()

        contrib = manager.record_contribution(
            contribution_type=ContributionType.ANNOTATION,
            user_id="user_001",
            user_name="User",
            target_type="note",
            target_id="note_001"
        )

        manager.vote_contribution(contrib.contribution_id, "voter_001", upvote=True)
        manager.vote_contribution(contrib.contribution_id, "voter_002", upvote=True)
        manager.vote_contribution(contrib.contribution_id, "voter_003", upvote=False)

        assert contrib.upvotes == 2
        assert contrib.downvotes == 1

    def test_get_attribution(self):
        """Test getting attribution for a target."""
        manager = create_attribution_manager()

        # Multiple contributions to same target
        manager.record_contribution(
            ContributionType.KNOWLEDGE_ADD, "user_001", "User 1",
            "memory", "mem_001"
        )
        manager.record_contribution(
            ContributionType.KNOWLEDGE_EDIT, "user_002", "User 2",
            "memory", "mem_001"
        )

        attribution = manager.get_attribution("mem_001")

        assert attribution is not None
        assert attribution.created_by == "user_001"
        assert attribution.last_modified_by == "user_002"
        assert attribution.contributor_count == 2
        assert attribution.edit_count == 1

    def test_user_stats(self):
        """Test getting user contribution statistics."""
        manager = create_attribution_manager()

        # Record several contributions
        for i in range(5):
            manager.record_contribution(
                ContributionType.KNOWLEDGE_ADD, "user_001", "User",
                "memory", f"mem_{i:03d}",
                content_size=100
            )

        manager.record_contribution(
            ContributionType.ANNOTATION, "user_001", "User",
            "note", "note_001"
        )

        stats = manager.get_user_stats("user_001")

        assert stats.total_contributions() == 6
        assert stats.contribution_counts["knowledge_add"] == 5
        assert stats.contribution_counts["annotation"] == 1
        assert stats.unique_targets == 6

    def test_leaderboard(self):
        """Test leaderboard generation."""
        manager = create_attribution_manager()

        # User 1: 5 contributions
        for i in range(5):
            manager.record_contribution(
                ContributionType.KNOWLEDGE_ADD, "user_001", "User 1",
                "memory", f"mem_a_{i}"
            )

        # User 2: 3 contributions
        for i in range(3):
            manager.record_contribution(
                ContributionType.KNOWLEDGE_ADD, "user_002", "User 2",
                "memory", f"mem_b_{i}"
            )

        # User 3: 7 contributions
        for i in range(7):
            manager.record_contribution(
                ContributionType.KNOWLEDGE_ADD, "user_003", "User 3",
                "memory", f"mem_c_{i}"
            )

        leaderboard = manager.get_leaderboard(metric="contributions", limit=3)

        assert len(leaderboard) == 3
        assert leaderboard[0]["user_id"] == "user_003"
        assert leaderboard[0]["score"] == 7
        assert leaderboard[0]["rank"] == 1


# ==============================================================================
# Voice Module Tests
# ==============================================================================

class TestMediaTrack:
    """Tests for MediaTrack class."""

    def test_track_creation(self):
        """Test creating a media track."""
        track = MediaTrack(
            track_id="track_001",
            media_type=MediaType.AUDIO,
            user_id="user_001",
            quality=StreamQuality.HIGH
        )

        assert track.track_id == "track_001"
        assert track.media_type == MediaType.AUDIO
        assert track.enabled is True
        assert track.muted is False


class TestVoiceRoomParticipant:
    """Tests for VoiceRoomParticipant class."""

    def test_participant_creation(self):
        """Test creating voice room participant."""
        participant = VoiceRoomParticipant(
            user_id="user_001",
            display_name="Test User"
        )

        assert participant.user_id == "user_001"
        assert participant.is_muted is False
        assert participant.is_speaking is False


class TestVoiceRoom:
    """Tests for VoiceRoom class."""

    def test_room_creation(self):
        """Test creating a voice room."""
        settings = VoiceRoomSettings(max_participants=10)
        room = create_voice_room("session_001", settings)

        assert room.session_id == "session_001"
        assert room.settings.max_participants == 10

    def test_add_participant(self):
        """Test adding participant to voice room."""
        room = create_voice_room("session_001")

        participant = room.add_participant("user_001", "Test User")

        assert participant is not None
        assert participant.user_id == "user_001"
        assert "user_001" in room.participants

    def test_participant_limit(self):
        """Test voice room participant limit."""
        settings = VoiceRoomSettings(max_participants=2)
        room = create_voice_room("session_001", settings)

        room.add_participant("user_001", "User 1")
        room.add_participant("user_002", "User 2")
        result = room.add_participant("user_003", "User 3")

        assert result is None
        assert len(room.participants) == 2

    @pytest.mark.asyncio
    async def test_mute_toggle(self):
        """Test mute toggling."""
        room = create_voice_room("session_001")
        room.add_participant("user_001", "Test")

        await room.toggle_mute("user_001", muted=True)
        assert room.participants["user_001"].is_muted is True

        await room.toggle_mute("user_001")  # Toggle
        assert room.participants["user_001"].is_muted is False

    @pytest.mark.asyncio
    async def test_video_toggle(self):
        """Test video toggling."""
        settings = VoiceRoomSettings(allow_video=True, default_video_off=True)
        room = create_voice_room("session_001", settings)
        room.add_participant("user_001", "Test")

        await room.toggle_video("user_001", video_on=True)
        assert room.participants["user_001"].is_video_on is True

    @pytest.mark.asyncio
    async def test_screen_share(self):
        """Test screen sharing."""
        settings = VoiceRoomSettings(allow_screen_share=True)
        room = create_voice_room("session_001", settings)
        room.add_participant("user_001", "Test")

        result = await room.start_screen_share("user_001")
        assert result is True
        assert room.participants["user_001"].is_screen_sharing is True

        await room.stop_screen_share("user_001")
        assert room.participants["user_001"].is_screen_sharing is False

    @pytest.mark.asyncio
    async def test_only_one_screen_share(self):
        """Test only one user can screen share."""
        settings = VoiceRoomSettings(allow_screen_share=True)
        room = create_voice_room("session_001", settings)
        room.add_participant("user_001", "User 1")
        room.add_participant("user_002", "User 2")

        await room.start_screen_share("user_001")
        result = await room.start_screen_share("user_002")

        assert result is False
        assert room.participants["user_002"].is_screen_sharing is False

    def test_speaking_update(self):
        """Test speaking volume updates."""
        room = create_voice_room("session_001")
        room.add_participant("user_001", "Test")

        room.update_speaking("user_001", 0.5)

        assert room.participants["user_001"].is_speaking is True
        assert room.participants["user_001"].speaking_volume == 0.5

    @pytest.mark.asyncio
    async def test_peer_connection_creation(self):
        """Test creating peer connection."""
        room = create_voice_room("session_001")
        room.add_participant("user_001", "User 1")
        room.add_participant("user_002", "User 2")

        conn = await room.create_connection("user_001", "user_002")

        assert conn is not None
        assert conn.local_user_id == "user_001"
        assert conn.remote_user_id == "user_002"
        assert conn.state == ConnectionState.NEW


class TestVoiceManager:
    """Tests for VoiceManager class."""

    @pytest.mark.asyncio
    async def test_create_room(self):
        """Test creating room via manager."""
        manager = create_voice_manager()

        room = await manager.create_room("session_001")

        assert room is not None
        assert room.room_id in manager.rooms

    @pytest.mark.asyncio
    async def test_join_and_leave_room(self):
        """Test joining and leaving rooms."""
        manager = create_voice_manager()

        room = await manager.create_room("session_001")

        participant = await manager.join_room(room.room_id, "user_001", "User")
        assert participant is not None
        assert "user_001" in manager.user_rooms

        result = await manager.leave_room(room.room_id, "user_001")
        assert result is True
        assert "user_001" not in manager.user_rooms

    @pytest.mark.asyncio
    async def test_close_room(self):
        """Test closing a voice room."""
        manager = create_voice_manager()

        room = await manager.create_room("session_001")
        await manager.join_room(room.room_id, "user_001", "User")

        room_id = room.room_id
        result = await manager.close_room(room_id)

        assert result is True
        assert room_id not in manager.rooms
        assert "user_001" not in manager.user_rooms


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestCollaborationIntegration:
    """Integration tests across collaboration modules."""

    @pytest.mark.asyncio
    async def test_full_session_lifecycle(self):
        """Test complete session lifecycle with all modules."""
        # Create session
        session_manager = create_session_manager()
        session = await session_manager.create_session(
            name="Integration Test",
            owner_id="owner",
            owner_name="Owner"
        )

        # Create presence manager
        presence_manager = create_presence_manager(session.session_id)
        presence_manager.add_user("owner", "Owner")

        # Create attribution manager
        attribution_manager = create_attribution_manager(session.knowledge_base_id)

        # Create sync manager
        sync_manager = create_state_synchronizer(session.session_id, "owner")

        # Simulate activity
        presence_manager.update_cursor("owner", 100, 200)
        presence_manager.update_focus("owner", FocusType.KNOWLEDGE_GRAPH)

        # Record contribution
        contribution = attribution_manager.record_contribution(
            ContributionType.KNOWLEDGE_ADD,
            "owner",
            "Owner",
            "memory",
            "mem_001",
            content_preview="Test knowledge"
        )

        # Create sync operation
        op = sync_manager.create_operation(
            OperationType.INSERT,
            "memory",
            "mem_001",
            {"content": "Test knowledge"}
        )

        # Verify state
        assert session.state == SessionState.ACTIVE
        assert presence_manager.get_presence("owner").focus == FocusType.KNOWLEDGE_GRAPH
        assert contribution is not None
        assert op.sequence_number == 1

        # Close session
        await session_manager.close_session(session.session_id, "owner")
        assert session.session_id not in session_manager.sessions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
