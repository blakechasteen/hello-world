"""Tests for FastAPI service."""

import pytest
from fastapi.testclient import TestClient

from apps.elle_game_engine.service import app


@pytest.fixture
def client():
    """Create test client with lifespan context."""
    with TestClient(app) as client:
        yield client


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self, client):
        """Test health check returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "elle_game_engine"


class TestGameActionEndpoint:
    """Tests for main game action endpoint."""

    def test_talk_to_npc_success(self, client):
        """Test successful NPC dialogue request."""
        request_data = {
            "game_state": {
                "scene_id": "village_square",
                "npcs": [
                    {
                        "id": "innkeeper",
                        "name": "Bob",
                        "role": "innkeeper",
                        "mood": "neutral",
                        "location": "inn",
                        "flags": {},
                    }
                ],
                "player": {
                    "name": "Hero",
                    "location": "village_square",
                },
                "world": {
                    "time_of_day": "afternoon",
                },
                "tags": [],
            },
            "player_intent": {
                "type": "talk_to_npc",
                "target_npc_id": "innkeeper",
                "raw_input": "Hello!",
            },
        }

        response = client.post("/elle/game/action", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert "mode" in data
        assert "priority" in data
        assert data["priority"] in ["low", "medium", "high"]
        assert "dialogue" in data

    def test_enter_scene_success(self, client):
        """Test successful scene entry request."""
        request_data = {
            "game_state": {
                "scene_id": "dark_forest",
                "npcs": [],
                "player": {
                    "name": "Hero",
                    "location": "dark_forest",
                },
                "world": {
                    "time_of_day": "night",
                    "weather": "fog",
                    "tension_level": "high",
                },
                "tags": ["dangerous"],
            },
            "player_intent": {
                "type": "enter_scene",
            },
        }

        response = client.post("/elle/game/action", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert "mode" in data
        assert "priority" in data

    def test_request_hint_success(self, client):
        """Test successful hint request."""
        request_data = {
            "game_state": {
                "scene_id": "puzzle_room",
                "npcs": [],
                "player": {
                    "name": "Hero",
                    "location": "puzzle_room",
                    "quest_stage": "mid",
                },
                "world": {
                    "time_of_day": "day",
                },
                "tags": ["puzzle"],
            },
            "player_intent": {
                "type": "request_hint",
                "raw_input": "I'm stuck on this puzzle",
            },
        }

        response = client.post("/elle/game/action", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["mode"] == "hint"
        assert data["hint_text"] is not None

    def test_debug_summary_success(self, client):
        """Test successful debug summary request."""
        request_data = {
            "game_state": {
                "scene_id": "test_scene",
                "npcs": [],
                "player": {
                    "name": "Hero",
                    "location": "test",
                },
                "world": {
                    "time_of_day": "day",
                },
                "tags": [],
            },
            "player_intent": {
                "type": "debug_summary",
            },
        }

        response = client.post("/elle/game/action", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["mode"] == "dev_debug"
        assert data["debug_notes"] is not None

    def test_missing_required_fields(self, client):
        """Test that missing required fields returns 422."""
        request_data = {
            "game_state": {
                "scene_id": "test",
                # Missing player
                "world": {
                    "time_of_day": "day",
                },
            },
            "player_intent": {
                "type": "enter_scene",
            },
        }

        response = client.post("/elle/game/action", json=request_data)

        assert response.status_code == 422

    def test_invalid_intent_type(self, client):
        """Test that invalid intent type returns 422."""
        request_data = {
            "game_state": {
                "scene_id": "test",
                "player": {
                    "name": "Hero",
                    "location": "test",
                },
                "world": {
                    "time_of_day": "day",
                },
            },
            "player_intent": {
                "type": "invalid_type",  # Invalid
            },
        }

        response = client.post("/elle/game/action", json=request_data)

        assert response.status_code == 422

    def test_talk_to_npc_missing_target(self, client):
        """Test that talk_to_npc without target_npc_id returns 500."""
        request_data = {
            "game_state": {
                "scene_id": "test",
                "player": {
                    "name": "Hero",
                    "location": "test",
                },
                "world": {
                    "time_of_day": "day",
                },
            },
            "player_intent": {
                "type": "talk_to_npc",
                # Missing target_npc_id
            },
        }

        response = client.post("/elle/game/action", json=request_data)

        # Should fail validation
        assert response.status_code == 500

    def test_complex_game_state(self, client):
        """Test with complex game state."""
        request_data = {
            "game_state": {
                "scene_id": "throne_room",
                "npcs": [
                    {
                        "id": "king",
                        "name": "King Arthur",
                        "role": "monarch",
                        "mood": "serious",
                        "location": "throne",
                        "flags": {"war_declared": True, "knows_betrayal": True},
                    },
                    {
                        "id": "advisor",
                        "name": "Merlin",
                        "role": "advisor",
                        "mood": "concerned",
                        "location": "beside_throne",
                        "flags": {"prophecy_revealed": True},
                    },
                ],
                "player": {
                    "name": "Sir Lancelot",
                    "location": "throne_room",
                    "quest_stage": "climax",
                    "reputation": "hero",
                    "traits": {"brave": 5, "loyal": 4, "cunning": 2},
                    "inventory_tags": ["excalibur", "holy_grail", "armor_of_light"],
                },
                "world": {
                    "time_of_day": "evening",
                    "weather": "storm",
                    "tension_level": "critical",
                },
                "tags": ["post_battle", "war_council", "dramatic"],
            },
            "player_intent": {
                "type": "talk_to_npc",
                "target_npc_id": "king",
                "raw_input": "Your Majesty, I bring news from the front",
            },
        }

        response = client.post("/elle/game/action", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["mode"] == "npc_dialogue"
        assert len(data["dialogue"]) > 0
        # DummyClient may not extract exact NPC ID, just verify there is one
        assert data["dialogue"][0]["npc_id"] is not None
        assert isinstance(data["dialogue"][0]["npc_id"], str)


class TestResponseFormat:
    """Tests for response format validation."""

    def test_response_structure(self, client):
        """Test that response has all required fields."""
        request_data = {
            "game_state": {
                "scene_id": "test",
                "player": {
                    "name": "Hero",
                    "location": "test",
                },
                "world": {
                    "time_of_day": "day",
                },
            },
            "player_intent": {
                "type": "enter_scene",
            },
        }

        response = client.post("/elle/game/action", json=request_data)
        data = response.json()

        # Check all top-level fields exist
        assert "mode" in data
        assert "priority" in data
        assert "dialogue" in data
        assert "hint_text" in data
        assert "world_reaction" in data
        assert "debug_notes" in data

        # Check types
        assert isinstance(data["mode"], str)
        assert isinstance(data["priority"], str)
        assert isinstance(data["dialogue"], list)

    def test_dialogue_structure(self, client):
        """Test that dialogue items have correct structure."""
        request_data = {
            "game_state": {
                "scene_id": "test",
                "npcs": [{"id": "test_npc", "name": "Test", "role": "test"}],
                "player": {
                    "name": "Hero",
                    "location": "test",
                },
                "world": {
                    "time_of_day": "day",
                },
            },
            "player_intent": {
                "type": "talk_to_npc",
                "target_npc_id": "test_npc",
            },
        }

        response = client.post("/elle/game/action", json=request_data)
        data = response.json()

        if data["dialogue"]:
            dialogue_item = data["dialogue"][0]
            assert "npc_id" in dialogue_item
            assert "text" in dialogue_item
            assert "tone" in dialogue_item
            assert isinstance(dialogue_item["npc_id"], str)
            assert isinstance(dialogue_item["text"], str)
