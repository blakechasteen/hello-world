"""
Tests for session management system.

Tests cover:
- Session creation and retrieval
- Conversation history accumulation
- World flag persistence
- NPC relationship tracking
- Session serialization/deserialization
- Both storage backends (in-memory and file-based)
"""

import pytest
import tempfile
import json
from pathlib import Path

from apps.elle_game_engine.session import (
    GameSession,
    ConversationExchange,
    NPCRelationship,
    InMemorySessionStore,
    JSONSessionStore,
)


# ============================================================================
# Session Model Tests
# ============================================================================

def test_session_creation():
    """Test basic session creation."""
    session = GameSession(player_id="player_123")

    assert session.session_id is not None
    assert session.player_id == "player_123"
    assert len(session.conversation_history) == 0
    assert len(session.world_flags) == 0
    assert len(session.npc_relationships) == 0


def test_session_add_exchange():
    """Test adding conversation exchanges."""
    session = GameSession()

    session.add_exchange(
        player_query="Hello!",
        elle_response="Greetings, traveler!",
        npc_id="innkeeper",
    )

    assert len(session.conversation_history) == 1
    exchange = session.conversation_history[0]
    assert exchange.player_query == "Hello!"
    assert exchange.elle_response == "Greetings, traveler!"
    assert exchange.npc_id == "innkeeper"


def test_session_conversation_history_circular_buffer():
    """Test that conversation history maintains max size."""
    session = GameSession(max_history_size=3)

    # Add 5 exchanges
    for i in range(5):
        session.add_exchange(
            player_query=f"Query {i}",
            elle_response=f"Response {i}",
        )

    # Should only keep last 3
    assert len(session.conversation_history) == 3
    assert session.conversation_history[0].player_query == "Query 2"
    assert session.conversation_history[-1].player_query == "Query 4"


def test_session_get_conversation_context():
    """Test conversation context formatting."""
    session = GameSession()

    session.add_exchange("How are you?", "I am well, thank you!", npc_id="npc1")
    session.add_exchange("Tell me a secret.", "The castle is haunted.", npc_id="npc2")

    context = session.get_conversation_context()

    assert "How are you?" in context
    assert "I am well, thank you!" in context
    assert "npc1" in context
    assert "Tell me a secret." in context
    assert "The castle is haunted." in context
    assert "npc2" in context


def test_session_get_conversation_context_max_exchanges():
    """Test limiting conversation context."""
    session = GameSession()

    # Add 5 exchanges
    for i in range(5):
        session.add_exchange(f"Query {i}", f"Response {i}")

    # Get only last 2
    context = session.get_conversation_context(max_exchanges=2)

    assert "Query 3" in context
    assert "Query 4" in context
    assert "Query 0" not in context
    assert "Query 1" not in context


def test_session_update_world_flags():
    """Test world flag updates."""
    session = GameSession()

    session.update_world_flags({
        "quest_started": True,
        "gate_opened": True,
    })

    assert session.world_flags["quest_started"] is True
    assert session.world_flags["gate_opened"] is True

    # Update with new flags
    session.update_world_flags({
        "gate_opened": False,
        "treasure_found": True,
    })

    assert session.world_flags["quest_started"] is True
    assert session.world_flags["gate_opened"] is False
    assert session.world_flags["treasure_found"] is True


def test_session_update_npc_relationship():
    """Test NPC relationship tracking."""
    session = GameSession()

    # First interaction
    session.update_npc_relationship(
        npc_id="merchant",
        reputation_delta=10,
        mood="grateful",
    )

    assert "merchant" in session.npc_relationships
    rel = session.npc_relationships["merchant"]
    assert rel.reputation == 10
    assert rel.interactions == 1
    assert rel.last_mood == "grateful"

    # Second interaction
    session.update_npc_relationship(
        npc_id="merchant",
        reputation_delta=-5,
        mood="annoyed",
        custom_flags={"gave_quest": True},
    )

    rel = session.npc_relationships["merchant"]
    assert rel.reputation == 5  # 10 - 5
    assert rel.interactions == 2
    assert rel.last_mood == "annoyed"
    assert rel.custom_flags["gave_quest"] is True


def test_session_npc_relationship_clamping():
    """Test reputation clamping to [-100, 100]."""
    session = GameSession()

    # Try to go above 100
    session.update_npc_relationship("npc1", reputation_delta=150)
    assert session.npc_relationships["npc1"].reputation == 100

    # Try to go below -100
    session.update_npc_relationship("npc2", reputation_delta=-150)
    assert session.npc_relationships["npc2"].reputation == -100


def test_session_serialization():
    """Test session to_dict and from_dict."""
    session = GameSession(player_id="player_456")

    session.add_exchange("Hello", "Hi there", npc_id="guard")
    session.update_world_flags({"flag1": True})
    session.update_npc_relationship("guard", reputation_delta=5, mood="neutral")

    # Serialize
    data = session.to_dict()

    # Deserialize
    restored = GameSession.from_dict(data)

    assert restored.session_id == session.session_id
    assert restored.player_id == session.player_id
    assert len(restored.conversation_history) == 1
    assert restored.conversation_history[0].player_query == "Hello"
    assert restored.world_flags["flag1"] is True
    assert "guard" in restored.npc_relationships
    assert restored.npc_relationships["guard"].reputation == 5


# ============================================================================
# InMemorySessionStore Tests
# ============================================================================

def test_inmemory_store_create_session():
    """Test creating sessions in memory."""
    store = InMemorySessionStore()

    session = store.create_session(player_id="player_789")

    assert session.session_id is not None
    assert session.player_id == "player_789"
    assert len(store) == 1


def test_inmemory_store_get_session():
    """Test retrieving sessions."""
    store = InMemorySessionStore()

    session = store.create_session()
    retrieved = store.get_session(session.session_id)

    assert retrieved is not None
    assert retrieved.session_id == session.session_id


def test_inmemory_store_get_nonexistent_session():
    """Test retrieving non-existent session returns None."""
    store = InMemorySessionStore()

    assert store.get_session("nonexistent") is None


def test_inmemory_store_update_session():
    """Test updating sessions."""
    store = InMemorySessionStore()

    session = store.create_session()
    session.add_exchange("Test", "Response")

    store.update_session(session)

    retrieved = store.get_session(session.session_id)
    assert len(retrieved.conversation_history) == 1


def test_inmemory_store_delete_session():
    """Test deleting sessions."""
    store = InMemorySessionStore()

    session = store.create_session()
    assert len(store) == 1

    deleted = store.delete_session(session.session_id)
    assert deleted is True
    assert len(store) == 0


def test_inmemory_store_delete_nonexistent():
    """Test deleting non-existent session."""
    store = InMemorySessionStore()

    deleted = store.delete_session("nonexistent")
    assert deleted is False


# ============================================================================
# JSONSessionStore Tests
# ============================================================================

def test_json_store_create_session():
    """Test creating sessions with file storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = JSONSessionStore(storage_path=tmpdir)

        session = store.create_session(player_id="player_abc")

        assert session.session_id is not None
        assert len(store) == 1

        # Check file was created
        session_file = Path(tmpdir) / f"{session.session_id}.json"
        assert session_file.exists()


def test_json_store_get_session():
    """Test retrieving sessions from file storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = JSONSessionStore(storage_path=tmpdir)

        session = store.create_session()
        retrieved = store.get_session(session.session_id)

        assert retrieved is not None
        assert retrieved.session_id == session.session_id


def test_json_store_persistence():
    """Test sessions persist across store instances."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create session in first store
        store1 = JSONSessionStore(storage_path=tmpdir)
        session = store1.create_session(player_id="persistent_player")
        session.add_exchange("Test", "Response")
        session.update_world_flags({"persistent_flag": True})
        store1.update_session(session)

        # Create new store pointing to same directory
        store2 = JSONSessionStore(storage_path=tmpdir)

        # Should load existing session
        retrieved = store2.get_session(session.session_id)
        assert retrieved is not None
        assert retrieved.player_id == "persistent_player"
        assert len(retrieved.conversation_history) == 1
        assert retrieved.world_flags["persistent_flag"] is True


def test_json_store_update_session():
    """Test updating sessions writes to disk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = JSONSessionStore(storage_path=tmpdir)

        session = store.create_session()
        session.add_exchange("Update test", "Response")

        store.update_session(session)

        # Read file directly
        session_file = Path(tmpdir) / f"{session.session_id}.json"
        with open(session_file) as f:
            data = json.load(f)

        assert len(data["conversation_history"]) == 1
        assert data["conversation_history"][0]["player_query"] == "Update test"


def test_json_store_delete_session():
    """Test deleting sessions removes file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = JSONSessionStore(storage_path=tmpdir)

        session = store.create_session()
        session_file = Path(tmpdir) / f"{session.session_id}.json"

        assert session_file.exists()

        store.delete_session(session.session_id)

        assert not session_file.exists()
        assert len(store) == 0


def test_json_store_save():
    """Test manual save writes all sessions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = JSONSessionStore(storage_path=tmpdir)

        # Create multiple sessions
        session1 = store.create_session()
        session2 = store.create_session()

        session1.add_exchange("Test 1", "Response 1")
        session2.add_exchange("Test 2", "Response 2")

        # Update cache but don't save yet
        store._cache[session1.session_id] = session1
        store._cache[session2.session_id] = session2

        # Save all
        store.save()

        # Verify files exist and have correct content
        file1 = Path(tmpdir) / f"{session1.session_id}.json"
        file2 = Path(tmpdir) / f"{session2.session_id}.json"

        assert file1.exists()
        assert file2.exists()


def test_json_store_corrupted_file_handling():
    """Test store handles corrupted session files gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write a corrupted JSON file
        corrupted_file = Path(tmpdir) / "corrupted_session.json"
        with open(corrupted_file, 'w') as f:
            f.write("{ invalid json }")

        # Should not crash on initialization
        store = JSONSessionStore(storage_path=tmpdir)

        # Should return None for corrupted session
        assert store.get_session("corrupted_session") is None


# ============================================================================
# Integration Tests
# ============================================================================

def test_session_full_workflow():
    """Test complete session workflow."""
    store = InMemorySessionStore()

    # Create session
    session = store.create_session(player_id="integration_test_player")

    # Simulate multiple exchanges
    exchanges = [
        ("Hello", "Greetings!", "npc1"),
        ("What's your name?", "I am Bob.", "npc1"),
        ("Can you help me?", "Of course!", "npc1"),
    ]

    for player_query, elle_response, npc_id in exchanges:
        session.add_exchange(player_query, elle_response, npc_id)

    # Update world flags
    session.update_world_flags({"quest_started": True, "npc_met": True})

    # Update NPC relationship
    session.update_npc_relationship("npc1", reputation_delta=15, mood="grateful")

    # Save session
    store.update_session(session)

    # Retrieve and verify
    retrieved = store.get_session(session.session_id)

    assert len(retrieved.conversation_history) == 3
    assert retrieved.world_flags["quest_started"] is True
    assert retrieved.npc_relationships["npc1"].reputation == 15
    assert retrieved.npc_relationships["npc1"].interactions == 1


def test_session_context_continuity():
    """Test conversation context maintains continuity."""
    session = GameSession()

    # Simulate a conversation
    session.add_exchange("Do you sell potions?", "Yes, I have healing potions.", "merchant")
    session.add_exchange("How much?", "50 gold each.", "merchant")
    session.add_exchange("I'll take one.", "Here you go!", "merchant")

    context = session.get_conversation_context()

    # Context should show the flow
    assert "Do you sell potions?" in context
    assert "Yes, I have healing potions." in context
    assert "How much?" in context
    assert "50 gold each." in context
    assert "I'll take one." in context
    assert "Here you go!" in context


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
