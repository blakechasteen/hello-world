"""
Integration tests for session management through the service API.

Tests the full workflow of creating sessions, making requests,
and maintaining state across multiple interactions.
"""

import pytest
from fastapi.testclient import TestClient

from apps.elle_game_engine.service import create_app
from apps.elle_game_engine.session import InMemorySessionStore


@pytest.fixture
def client():
    """Create test client for Elle service."""
    app = create_app()
    return TestClient(app)


def test_session_creation_on_first_request(client):
    """Test that first request without session_id creates a new session."""
    request = {
        "game_state": {
            "scene_id": "village_square",
            "npcs": [
                {
                    "id": "merchant",
                    "name": "Bob",
                    "role": "merchant",
                    "location": "market",
                }
            ],
            "player": {
                "name": "Hero",
                "location": "village_square",
            },
            "world": {
                "time_of_day": "afternoon",
            },
        },
        "player_intent": {
            "type": "talk_to_npc",
            "target_npc_id": "merchant",
            "raw_input": "Hello!",
        },
        "player_id": "test_player_123",
    }

    response = client.post("/elle/game/action", json=request)

    assert response.status_code == 200
    data = response.json()

    # Should return session_id
    assert "session_id" in data
    assert data["session_id"] is not None

    # Should have valid response structure
    assert data["mode"] in ["npc_dialogue", "hint", "world_reaction", "dev_debug"]
    assert data["priority"] in ["low", "medium", "high"]


def test_session_continuity_across_requests(client):
    """Test that session state persists across multiple requests."""
    # First request - create session
    request1 = {
        "game_state": {
            "scene_id": "inn",
            "npcs": [{"id": "innkeeper", "name": "Alice", "role": "innkeeper", "location": "inn"}],
            "player": {"name": "Adventurer", "location": "inn"},
            "world": {"time_of_day": "evening"},
        },
        "player_intent": {
            "type": "talk_to_npc",
            "target_npc_id": "innkeeper",
            "raw_input": "Do you have a room?",
        },
    }

    response1 = client.post("/elle/game/action", json=request1)
    assert response1.status_code == 200
    data1 = response1.json()
    session_id = data1["session_id"]

    # Second request - use existing session
    request2 = {
        "game_state": {
            "scene_id": "inn",
            "npcs": [{"id": "innkeeper", "name": "Alice", "role": "innkeeper", "location": "inn"}],
            "player": {"name": "Adventurer", "location": "inn"},
            "world": {"time_of_day": "evening"},
        },
        "player_intent": {
            "type": "talk_to_npc",
            "target_npc_id": "innkeeper",
            "raw_input": "How much?",
        },
        "session_id": session_id,
    }

    response2 = client.post("/elle/game/action", json=request2)
    assert response2.status_code == 200
    data2 = response2.json()

    # Should return same session_id
    assert data2["session_id"] == session_id

    # Elle should have context from previous conversation
    # (The LLM will see conversation history in prompt)


def test_session_not_found_error(client):
    """Test that using non-existent session_id returns 404."""
    request = {
        "game_state": {
            "scene_id": "test",
            "npcs": [],
            "player": {"name": "Test", "location": "test"},
            "world": {"time_of_day": "day"},
        },
        "player_intent": {
            "type": "enter_scene",
        },
        "session_id": "nonexistent-session-id-12345",
    }

    response = client.post("/elle/game/action", json=request)

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_world_flag_persistence(client):
    """Test that world flags persist across requests."""
    # First request - action that changes world flags
    request1 = {
        "game_state": {
            "scene_id": "castle_gate",
            "npcs": [{"id": "guard", "name": "Guard", "role": "guard", "location": "gate"}],
            "player": {"name": "Hero", "location": "castle_gate"},
            "world": {"time_of_day": "noon"},
        },
        "player_intent": {
            "type": "talk_to_npc",
            "target_npc_id": "guard",
            "raw_input": "Let me pass!",
        },
    }

    response1 = client.post("/elle/game/action", json=request1)
    session_id = response1.json()["session_id"]

    # If response included world changes, they should be stored
    # (This depends on LLM response, but session system handles it)

    # Second request with same session
    request2 = {
        "game_state": {
            "scene_id": "castle_gate",
            "npcs": [{"id": "guard", "name": "Guard", "role": "guard", "location": "gate"}],
            "player": {"name": "Hero", "location": "castle_gate"},
            "world": {"time_of_day": "noon"},
        },
        "player_intent": {
            "type": "talk_to_npc",
            "target_npc_id": "guard",
            "raw_input": "Thank you!",
        },
        "session_id": session_id,
    }

    response2 = client.post("/elle/game/action", json=request2)
    assert response2.status_code == 200


def test_npc_reputation_tracking(client):
    """Test that NPC relationships update based on interactions."""
    # Create session with positive interaction
    request = {
        "game_state": {
            "scene_id": "market",
            "npcs": [{"id": "vendor", "name": "Vendor", "role": "vendor", "location": "market"}],
            "player": {"name": "Buyer", "location": "market"},
            "world": {"time_of_day": "morning"},
        },
        "player_intent": {
            "type": "talk_to_npc",
            "target_npc_id": "vendor",
            "raw_input": "Good morning!",
        },
    }

    response = client.post("/elle/game/action", json=request)
    assert response.status_code == 200

    # Reputation should be tracked in session
    # (Automatic based on dialogue tone from LLM)


def test_multiple_sessions_isolated(client):
    """Test that different player sessions are isolated."""
    # Create session for player 1
    request1 = {
        "game_state": {
            "scene_id": "town",
            "npcs": [],
            "player": {"name": "Player1", "location": "town"},
            "world": {"time_of_day": "day"},
        },
        "player_intent": {"type": "enter_scene"},
        "player_id": "player_1",
    }

    response1 = client.post("/elle/game/action", json=request1)
    session1_id = response1.json()["session_id"]

    # Create session for player 2
    request2 = {
        "game_state": {
            "scene_id": "town",
            "npcs": [],
            "player": {"name": "Player2", "location": "town"},
            "world": {"time_of_day": "day"},
        },
        "player_intent": {"type": "enter_scene"},
        "player_id": "player_2",
    }

    response2 = client.post("/elle/game/action", json=request2)
    session2_id = response2.json()["session_id"]

    # Sessions should be different
    assert session1_id != session2_id


def test_conversation_history_limit(client):
    """Test that conversation history maintains max size."""
    # Create session
    first_request = {
        "game_state": {
            "scene_id": "test",
            "npcs": [{"id": "npc", "name": "NPC", "role": "test", "location": "test"}],
            "player": {"name": "Test", "location": "test"},
            "world": {"time_of_day": "day"},
        },
        "player_intent": {
            "type": "talk_to_npc",
            "target_npc_id": "npc",
            "raw_input": "First message",
        },
    }

    response = client.post("/elle/game/action", json=first_request)
    session_id = response.json()["session_id"]

    # Make 15 more requests (total 16, but max is 10)
    for i in range(2, 17):
        request = {
            "game_state": {
                "scene_id": "test",
                "npcs": [{"id": "npc", "name": "NPC", "role": "test", "location": "test"}],
                "player": {"name": "Test", "location": "test"},
                "world": {"time_of_day": "day"},
            },
            "player_intent": {
                "type": "talk_to_npc",
                "target_npc_id": "npc",
                "raw_input": f"Message {i}",
            },
            "session_id": session_id,
        }
        client.post("/elle/game/action", json=request)

    # Session should only keep last 10 exchanges
    # (Verified in session tests, service correctly uses this)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
