"""
Collaboration Integration Tests
================================
Comprehensive tests for multi-user collaborative features.

Tests:
- Session management
- Presence tracking
- CRDT synchronization
- Permissions
- Activity logging
- Multi-user scenarios
- Conflict resolution
"""

import pytest
import asyncio
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from collaboration.session import SessionManager, CollaborativeSession
from collaboration.presence import PresenceTracker, CursorPosition, PresenceStatus
from collaboration.sync import CRDTEngine, Delta, OperationType, StateVector
from collaboration.permissions import PermissionManager, Role, Permission
from collaboration.activity import ActivityLogger, EventType


# ============================================================================
# Session Management Tests
# ============================================================================

def test_session_creation():
    """Test creating collaborative session."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = SessionManager(storage_dir=Path(tmpdir))

        session = manager.create_session(
            name="Test Session",
            owner_id="user1",
            owner_username="Alice"
        )

        assert session.name == "Test Session"
        assert session.owner_id == "user1"
        assert len(session.users) == 1
        assert session.users["user1"].role == "owner"


def test_session_user_management():
    """Test adding/removing users from session."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = SessionManager(storage_dir=Path(tmpdir))

        session = manager.create_session(
            name="Test",
            owner_id="user1",
            owner_username="Alice"
        )

        # Add users
        manager.join_session(session.session_id, "user2", "Bob", role="editor")
        manager.join_session(session.session_id, "user3", "Charlie", role="viewer")

        assert len(session.users) == 3

        # Remove user
        manager.leave_session(session.session_id, "user3")
        assert len(session.users) == 2
        assert "user3" not in session.users


def test_session_persistence():
    """Test saving and loading sessions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = SessionManager(storage_dir=Path(tmpdir))

        # Create session
        session = manager.create_session(
            name="Test Session",
            owner_id="user1",
            owner_username="Alice"
        )

        session.update_state("key", "value")

        # Save
        manager.save_session(session.session_id)

        # Load in new manager
        manager2 = SessionManager(storage_dir=Path(tmpdir))
        loaded = manager2.load_session(session.session_id)

        assert loaded is not None
        assert loaded.name == "Test Session"
        assert loaded.state["key"] == "value"


def test_ownership_transfer():
    """Test transferring session ownership."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = SessionManager(storage_dir=Path(tmpdir))

        session = manager.create_session(
            name="Test",
            owner_id="user1",
            owner_username="Alice"
        )

        manager.join_session(session.session_id, "user2", "Bob", role="editor")

        # Transfer ownership
        success = session.transfer_ownership("user2")

        assert success
        assert session.owner_id == "user2"
        assert session.users["user2"].role == "owner"
        assert session.users["user1"].role == "admin"


# ============================================================================
# Presence Tracking Tests
# ============================================================================

@pytest.mark.asyncio
async def test_presence_tracking():
    """Test user presence tracking."""
    tracker = PresenceTracker(idle_threshold_seconds=2, timeout_seconds=5)

    session_id = "test-session"

    # Add users
    tracker.add_user(session_id, "user1", "Alice")
    tracker.add_user(session_id, "user2", "Bob")

    presences = tracker.get_session_presences(session_id)
    assert len(presences) == 2

    # Check online
    online = tracker.get_online_users(session_id)
    assert len(online) == 2


@pytest.mark.asyncio
async def test_presence_heartbeat():
    """Test heartbeat mechanism."""
    tracker = PresenceTracker(idle_threshold_seconds=1, timeout_seconds=2)

    session_id = "test-session"
    tracker.add_user(session_id, "user1", "Alice")

    # Send heartbeat
    tracker.update_heartbeat(session_id, "user1")

    # Check online
    presence = tracker.get_user_presence(session_id, "user1")
    assert presence.is_online(timeout_seconds=2)

    # Wait and check idle
    await asyncio.sleep(1.5)
    tracker.update_all_statuses(session_id)

    assert presence.status == PresenceStatus.IDLE

    # Wait and check offline
    await asyncio.sleep(2)
    tracker.update_all_statuses(session_id)

    assert presence.status == PresenceStatus.OFFLINE


@pytest.mark.asyncio
async def test_cursor_tracking():
    """Test cursor position tracking."""
    tracker = PresenceTracker()

    session_id = "test-session"
    tracker.add_user(session_id, "user1", "Alice")

    # Update cursor
    cursor = CursorPosition(line=10, column=5)
    tracker.update_cursor(session_id, "user1", cursor)

    # Get presence
    presence = tracker.get_user_presence(session_id, "user1")
    assert presence.cursor is not None
    assert presence.cursor.line == 10
    assert presence.cursor.column == 5


# ============================================================================
# CRDT Synchronization Tests
# ============================================================================

def test_crdt_basic_operations():
    """Test basic CRDT operations."""
    engine = CRDTEngine(user_id="user1")

    # Set values
    engine.set_value("topic", "AI Research")
    engine.set_value("temperature", 0.7)

    assert engine.get_value("topic") == "AI Research"
    assert engine.get_value("temperature") == 0.7


def test_crdt_conflict_resolution():
    """Test CRDT conflict resolution (Last-Write-Wins)."""
    alice = CRDTEngine(user_id="alice")
    bob = CRDTEngine(user_id="bob")

    # Both set same value concurrently
    alice_delta = alice.set_value("topic", "Neural Networks")
    bob_delta = bob.set_value("topic", "Deep Learning")

    # Synchronize: Alice -> Bob
    bob.apply_delta(alice_delta)

    # Synchronize: Bob -> Alice
    alice.apply_delta(bob_delta)

    # Both should converge to same value (later timestamp wins)
    alice_topic = alice.get_value("topic")
    bob_topic = bob.get_value("topic")

    assert alice_topic == bob_topic  # Converged


def test_crdt_set_operations():
    """Test CRDT set operations (G-Set)."""
    alice = CRDTEngine(user_id="alice")
    bob = CRDTEngine(user_id="bob")

    # Add to set
    alice_delta = alice.add_to_set("participants", "alice")
    bob_delta = bob.add_to_set("participants", "bob")

    # Synchronize
    alice.apply_delta(bob_delta)
    bob.apply_delta(alice_delta)

    # Both should have same set
    alice_set = set(alice.get_set("participants"))
    bob_set = set(bob.get_set("participants"))

    assert alice_set == bob_set
    assert alice_set == {"alice", "bob"}


def test_crdt_multi_user_convergence():
    """Test CRDT convergence with 3+ users."""
    users = {
        "alice": CRDTEngine(user_id="alice"),
        "bob": CRDTEngine(user_id="bob"),
        "charlie": CRDTEngine(user_id="charlie")
    }

    # Each user sets values
    deltas = []
    deltas.append(users["alice"].set_value("a", 1))
    deltas.append(users["bob"].set_value("b", 2))
    deltas.append(users["charlie"].set_value("c", 3))

    # Broadcast all deltas to all users
    for delta in deltas:
        for engine in users.values():
            engine.apply_delta(delta)

    # Check convergence
    for engine in users.values():
        assert engine.get_value("a") == 1
        assert engine.get_value("b") == 2
        assert engine.get_value("c") == 3


def test_crdt_delta_compression():
    """Test delta compression."""
    engine = CRDTEngine(user_id="user1")

    # Create multiple deltas for same path
    deltas = []
    deltas.append(engine.set_value("x", 1))
    deltas.append(engine.set_value("x", 2))
    deltas.append(engine.set_value("x", 3))

    # Compress
    compressed = engine.compress_deltas(deltas)

    # Should keep only latest
    assert len(compressed) == 1
    assert compressed[0].value == 3


# ============================================================================
# Permission Tests
# ============================================================================

def test_role_hierarchy():
    """Test role hierarchy."""
    assert Role.VIEWER < Role.EDITOR
    assert Role.EDITOR < Role.ADMIN
    assert Role.ADMIN < Role.OWNER

    assert Role.OWNER > Role.ADMIN
    assert Role.ADMIN >= Role.EDITOR


def test_basic_permissions():
    """Test basic permission checking."""
    manager = PermissionManager()

    # Viewer can read
    assert manager.has_permission("user1", Role.VIEWER, Permission.READ_SESSION)

    # Viewer cannot edit
    assert not manager.has_permission("user1", Role.VIEWER, Permission.EDIT_MESSAGE)

    # Editor can edit
    assert manager.has_permission("user2", Role.EDITOR, Permission.EDIT_MESSAGE)

    # Only owner can delete session
    assert not manager.has_permission("user3", Role.ADMIN, Permission.DELETE_SESSION)
    assert manager.has_permission("user4", Role.OWNER, Permission.DELETE_SESSION)


def test_permission_actions():
    """Test permission action checking."""
    manager = PermissionManager()

    # Editor can send message
    check = manager.can_perform_action("user1", Role.EDITOR, "send_message")
    assert check.allowed

    # Viewer cannot edit
    check = manager.can_perform_action("user2", Role.VIEWER, "edit_message")
    assert not check.allowed


def test_user_modification_permissions():
    """Test permissions for modifying users."""
    manager = PermissionManager()

    # Admin can modify editor
    check = manager.can_modify_user("admin", Role.ADMIN, "editor", Role.EDITOR)
    assert check.allowed

    # Editor cannot modify admin
    check = manager.can_modify_user("editor", Role.EDITOR, "admin", Role.ADMIN)
    assert not check.allowed


def test_custom_permissions():
    """Test custom permission grants."""
    manager = PermissionManager()

    session_id = "test-session"

    # Viewer normally cannot edit
    assert not manager.has_permission(
        "viewer1",
        Role.VIEWER,
        Permission.EDIT_MESSAGE,
        session_id
    )

    # Grant custom permission
    manager.grant_permission(session_id, "viewer1", Permission.EDIT_MESSAGE)

    # Now can edit
    assert manager.has_permission(
        "viewer1",
        Role.VIEWER,
        Permission.EDIT_MESSAGE,
        session_id
    )

    # Revoke
    manager.revoke_permission(session_id, "viewer1", Permission.EDIT_MESSAGE)

    # Cannot edit again
    assert not manager.has_permission(
        "viewer1",
        Role.VIEWER,
        Permission.EDIT_MESSAGE,
        session_id
    )


# ============================================================================
# Activity Logging Tests
# ============================================================================

def test_activity_logging():
    """Test basic activity logging."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ActivityLogger(storage_dir=Path(tmpdir))

        session_id = "test-session"

        # Log events
        logger.log_event(
            EventType.SESSION_CREATED,
            session_id,
            "user1",
            "Alice"
        )

        logger.log_event(
            EventType.USER_JOINED,
            session_id,
            "user2",
            "Bob"
        )

        # Get events
        events = logger.get_events(session_id)
        assert len(events) == 2


def test_activity_filtering():
    """Test filtering activity events."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ActivityLogger(storage_dir=Path(tmpdir))

        session_id = "test-session"

        # Log various events
        logger.log_event(EventType.USER_JOINED, session_id, "user1", "Alice")
        logger.log_event(EventType.USER_JOINED, session_id, "user2", "Bob")
        logger.log_event(EventType.MESSAGE_SENT, session_id, "user1", "Alice")

        # Filter by type
        join_events = logger.get_events(session_id, event_type=EventType.USER_JOINED)
        assert len(join_events) == 2

        # Filter by user
        alice_events = logger.get_events(session_id, user_id="user1")
        assert len(alice_events) == 2


def test_version_history():
    """Test version snapshots."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ActivityLogger(storage_dir=Path(tmpdir))

        session_id = "test-session"

        # Log event
        event = logger.log_event(
            EventType.STATE_UPDATED,
            session_id,
            "user1",
            "Alice"
        )

        # Create version
        state = {"messages": ["Hello"], "topic": "Test"}
        version = logger.create_version(session_id, state, event.event_id)

        assert version.session_id == session_id
        assert version.state == state


def test_rollback():
    """Test state rollback."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ActivityLogger(storage_dir=Path(tmpdir))

        session_id = "test-session"

        # Create versions
        event1 = logger.log_event(EventType.STATE_UPDATED, session_id, "user1", "Alice")
        state1 = {"version": 1}
        version1 = logger.create_version(session_id, state1, event1.event_id)

        event2 = logger.log_event(EventType.STATE_UPDATED, session_id, "user1", "Alice")
        state2 = {"version": 2}
        version2 = logger.create_version(session_id, state2, event2.event_id)

        # Rollback to version 1
        rolled_back = logger.rollback_to_version(session_id, version1.version_id)

        assert rolled_back is not None
        assert rolled_back["version"] == 1


def test_diff_computation():
    """Test state diff computation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ActivityLogger(storage_dir=Path(tmpdir))

        before = {"a": 1, "b": 2}
        after = {"a": 1, "b": 3, "c": 4}

        diff = logger.compute_diff(before, after)

        assert "c" in diff["added_keys"]
        assert "b" in diff["modified_keys"]
        assert diff["total_changes"] == 2


def test_activity_persistence():
    """Test saving and loading activity."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ActivityLogger(storage_dir=Path(tmpdir))

        session_id = "test-session"

        # Log events
        logger.log_event(EventType.USER_JOINED, session_id, "user1", "Alice")
        logger.log_event(EventType.MESSAGE_SENT, session_id, "user1", "Alice")

        # Save
        logger.save_session_activity(session_id)

        # Load in new logger
        logger2 = ActivityLogger(storage_dir=Path(tmpdir))
        logger2.load_session_activity(session_id)

        events = logger2.get_events(session_id)
        assert len(events) == 2


# ============================================================================
# Multi-User Scenario Tests
# ============================================================================

@pytest.mark.asyncio
async def test_multi_user_collaboration():
    """Test complete multi-user collaboration scenario."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup
        session_manager = SessionManager(storage_dir=Path(tmpdir))
        presence_tracker = PresenceTracker()
        permission_manager = PermissionManager()
        activity_logger = ActivityLogger(storage_dir=Path(tmpdir))

        # Create session
        session = session_manager.create_session(
            name="Collaboration Test",
            owner_id="alice",
            owner_username="Alice"
        )

        session_id = session.session_id

        # Add users
        session_manager.join_session(session_id, "bob", "Bob", role="editor")
        session_manager.join_session(session_id, "charlie", "Charlie", role="viewer")

        # Track presence
        presence_tracker.add_user(session_id, "alice", "Alice")
        presence_tracker.add_user(session_id, "bob", "Bob")
        presence_tracker.add_user(session_id, "charlie", "Charlie")

        # Create CRDT engines
        engines = {
            "alice": CRDTEngine(user_id="alice"),
            "bob": CRDTEngine(user_id="bob"),
            "charlie": CRDTEngine(user_id="charlie")
        }

        # Simulate collaboration
        # Alice sets topic
        delta1 = engines["alice"].set_value("topic", "AI Safety")
        activity_logger.log_event(
            EventType.STATE_UPDATED,
            session_id,
            "alice",
            "Alice",
            data={"delta": delta1.to_dict()}
        )

        # Bob adds message
        delta2 = engines["bob"].add_to_set("messages", "Hello everyone!")

        # Synchronize deltas
        for user_id, engine in engines.items():
            engine.apply_delta(delta1)
            engine.apply_delta(delta2)

        # Check convergence
        for engine in engines.values():
            assert engine.get_value("topic") == "AI Safety"
            assert "Hello everyone!" in engine.get_set("messages")

        # Check permissions
        # Charlie (viewer) cannot edit
        check = permission_manager.can_perform_action(
            "charlie",
            Role.VIEWER,
            "edit_message",
            session_id
        )
        assert not check.allowed

        # Bob (editor) can edit
        check = permission_manager.can_perform_action(
            "bob",
            Role.EDITOR,
            "send_message",
            session_id
        )
        assert check.allowed

        # Verify activity log
        events = activity_logger.get_events(session_id)
        assert len(events) >= 1


@pytest.mark.asyncio
async def test_concurrent_updates():
    """Test handling concurrent updates from multiple users."""
    # Create 10 users
    num_users = 10
    engines = {f"user{i}": CRDTEngine(user_id=f"user{i}") for i in range(num_users)}

    # Each user performs operations
    all_deltas = []

    for i, (user_id, engine) in enumerate(engines.items()):
        delta = engine.set_value(f"key{i}", i * 10)
        all_deltas.append(delta)

    # Broadcast all deltas to all users
    for delta in all_deltas:
        for engine in engines.values():
            engine.apply_delta(delta)

    # Verify convergence
    for i in range(num_users):
        for engine in engines.values():
            assert engine.get_value(f"key{i}") == i * 10


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Collaboration Integration Tests")
    print("="*80 + "\n")

    # Run sync tests
    print("Running synchronous tests...")

    test_session_creation()
    print("✓ Session creation")

    test_session_user_management()
    print("✓ Session user management")

    test_session_persistence()
    print("✓ Session persistence")

    test_ownership_transfer()
    print("✓ Ownership transfer")

    test_crdt_basic_operations()
    print("✓ CRDT basic operations")

    test_crdt_conflict_resolution()
    print("✓ CRDT conflict resolution")

    test_crdt_set_operations()
    print("✓ CRDT set operations")

    test_crdt_multi_user_convergence()
    print("✓ CRDT multi-user convergence")

    test_crdt_delta_compression()
    print("✓ CRDT delta compression")

    test_role_hierarchy()
    print("✓ Role hierarchy")

    test_basic_permissions()
    print("✓ Basic permissions")

    test_permission_actions()
    print("✓ Permission actions")

    test_user_modification_permissions()
    print("✓ User modification permissions")

    test_custom_permissions()
    print("✓ Custom permissions")

    test_activity_logging()
    print("✓ Activity logging")

    test_activity_filtering()
    print("✓ Activity filtering")

    test_version_history()
    print("✓ Version history")

    test_rollback()
    print("✓ Rollback")

    test_diff_computation()
    print("✓ Diff computation")

    test_activity_persistence()
    print("✓ Activity persistence")

    # Run async tests
    print("\nRunning asynchronous tests...")

    asyncio.run(test_presence_tracking())
    print("✓ Presence tracking")

    asyncio.run(test_presence_heartbeat())
    print("✓ Presence heartbeat")

    asyncio.run(test_cursor_tracking())
    print("✓ Cursor tracking")

    asyncio.run(test_multi_user_collaboration())
    print("✓ Multi-user collaboration")

    asyncio.run(test_concurrent_updates())
    print("✓ Concurrent updates (10 users)")

    print("\n" + "="*80)
    print("All tests passed! ✓")
    print("="*80)
