"""Tests for game policy engine."""

import pytest
from apps.elle_game_engine.models import (
    GameStateSnapshot,
    PlayerState,
    WorldState,
    NPCState,
    PlayerIntent,
    PlayerIntentType,
    ActionMode,
)
from apps.elle_game_engine.llm_client import DummyLLMClient
from apps.elle_game_engine.policy import GamePolicy, PolicyError


@pytest.fixture
def dummy_client():
    """Create dummy LLM client for testing."""
    return DummyLLMClient()


@pytest.fixture
def game_policy(dummy_client):
    """Create game policy with dummy client."""
    return GamePolicy(dummy_client)


@pytest.fixture
def sample_game_state():
    """Create sample game state for testing."""
    return GameStateSnapshot(
        scene_id="village_square",
        npcs=[
            NPCState(
                id="innkeeper",
                name="Bob the Innkeeper",
                role="innkeeper",
                mood="nervous",
                location="inn",
                flags={"recent_theft": True},
            ),
            NPCState(
                id="guard",
                name="Captain Williams",
                role="guard",
                mood="alert",
                location="town_gate",
            ),
        ],
        player=PlayerState(
            name="Hero",
            location="village_square",
            quest_stage="intro",
            reputation="outsider",
            traits={"brave": 2, "kind": 3},
            inventory_tags=["sword", "healing_potion"],
        ),
        world=WorldState(
            time_of_day="afternoon",
            weather="clear",
            tension_level="calm",
        ),
        tags=["first_visit", "peaceful"],
    )


class TestGamePolicy:
    """Tests for GamePolicy class."""

    @pytest.mark.asyncio
    async def test_decide_talk_to_npc(self, game_policy, sample_game_state):
        """Test decision for talking to NPC."""
        intent = PlayerIntent(
            type=PlayerIntentType.TALK_TO_NPC,
            target_npc_id="innkeeper",
            raw_input="Hello!",
        )

        action = await game_policy.decide(sample_game_state, intent)

        assert action is not None
        assert action.mode == ActionMode.NPC_DIALOGUE
        assert len(action.dialogue) > 0
        assert action.dialogue[0].npc_id == "innkeeper"

    @pytest.mark.asyncio
    async def test_decide_enter_scene(self, game_policy, sample_game_state):
        """Test decision for entering scene."""
        intent = PlayerIntent(type=PlayerIntentType.ENTER_SCENE)

        action = await game_policy.decide(sample_game_state, intent)

        assert action is not None
        assert action.mode == ActionMode.WORLD_REACTION

    @pytest.mark.asyncio
    async def test_decide_request_hint(self, game_policy, sample_game_state):
        """Test decision for requesting hint."""
        intent = PlayerIntent(
            type=PlayerIntentType.REQUEST_HINT,
            raw_input="I'm stuck",
        )

        action = await game_policy.decide(sample_game_state, intent)

        assert action is not None
        assert action.mode == ActionMode.HINT
        assert action.hint_text is not None

    @pytest.mark.asyncio
    async def test_decide_debug_summary(self, game_policy, sample_game_state):
        """Test decision for debug summary."""
        intent = PlayerIntent(type=PlayerIntentType.DEBUG_SUMMARY)

        action = await game_policy.decide(sample_game_state, intent)

        assert action is not None
        assert action.mode == ActionMode.DEV_DEBUG
        assert action.debug_notes is not None

    def test_build_user_prompt(self, game_policy, sample_game_state):
        """Test prompt building."""
        intent = PlayerIntent(
            type=PlayerIntentType.TALK_TO_NPC,
            target_npc_id="innkeeper",
        )

        prompt = game_policy._build_user_prompt(sample_game_state, intent)

        # Verify key elements are in prompt
        assert "village_square" in prompt
        assert "Bob the Innkeeper" in prompt
        assert "Hero" in prompt
        assert "afternoon" in prompt
        assert "talk_to_npc" in prompt

    def test_parse_response_valid(self, game_policy):
        """Test parsing valid JSON response."""
        json_response = """
        {
            "mode": "npc_dialogue",
            "priority": "medium",
            "dialogue": [
                {
                    "npc_id": "guard",
                    "text": "Halt! State your business.",
                    "tone": "stern"
                }
            ],
            "hint_text": null,
            "world_reaction": null,
            "debug_notes": "Guard is on high alert"
        }
        """

        action = game_policy._parse_response(json_response)

        assert action.mode == ActionMode.NPC_DIALOGUE
        assert action.priority == "medium"
        assert len(action.dialogue) == 1
        assert action.dialogue[0].npc_id == "guard"
        assert action.dialogue[0].tone == "stern"
        assert action.debug_notes == "Guard is on high alert"

    def test_parse_response_with_markdown(self, game_policy):
        """Test parsing response wrapped in markdown code fence."""
        json_response = """```json
        {
            "mode": "hint",
            "priority": "low",
            "dialogue": [],
            "hint_text": "Try looking around",
            "world_reaction": null,
            "debug_notes": null
        }
        ```"""

        action = game_policy._parse_response(json_response)

        assert action.mode == ActionMode.HINT
        assert action.hint_text == "Try looking around"

    def test_parse_response_invalid_json(self, game_policy):
        """Test that invalid JSON raises error."""
        invalid_json = "This is not JSON"

        with pytest.raises(PolicyError, match="Invalid JSON"):
            game_policy._parse_response(invalid_json)

    def test_parse_response_missing_mode(self, game_policy):
        """Test that missing mode raises error."""
        json_response = """
        {
            "priority": "medium",
            "dialogue": []
        }
        """

        with pytest.raises(PolicyError, match="mode"):
            game_policy._parse_response(json_response)

    def test_parse_response_with_world_change(self, game_policy):
        """Test parsing response with world changes."""
        json_response = """
        {
            "mode": "world_reaction",
            "priority": "medium",
            "dialogue": [],
            "hint_text": null,
            "world_reaction": {
                "description": "The door creaks open",
                "flag_changes": {"door_opened": true}
            },
            "debug_notes": null
        }
        """

        action = game_policy._parse_response(json_response)

        assert action.mode == ActionMode.WORLD_REACTION
        assert action.world_reaction is not None
        assert action.world_reaction.description == "The door creaks open"
        assert action.world_reaction.flag_changes["door_opened"] is True


class TestDummyClientIntegration:
    """Integration tests with DummyLLMClient."""

    @pytest.mark.asyncio
    async def test_dummy_client_produces_valid_actions(self):
        """Test that dummy client produces valid actions."""
        client = DummyLLMClient()
        policy = GamePolicy(client)

        game_state = GameStateSnapshot(
            scene_id="test",
            player=PlayerState(name="Test", location="test"),
            world=WorldState(time_of_day="day"),
        )

        # Test all intent types
        intent_types = [
            PlayerIntentType.TALK_TO_NPC,
            PlayerIntentType.ENTER_SCENE,
            PlayerIntentType.REQUEST_HINT,
            PlayerIntentType.DEBUG_SUMMARY,
        ]

        for intent_type in intent_types:
            if intent_type == PlayerIntentType.TALK_TO_NPC:
                intent = PlayerIntent(
                    type=intent_type,
                    target_npc_id="test_npc",
                )
            else:
                intent = PlayerIntent(type=intent_type)

            action = await policy.decide(game_state, intent)
            assert action is not None
            assert action.priority in ["low", "medium", "high"]

    @pytest.mark.asyncio
    async def test_client_call_tracking(self, dummy_client, game_policy):
        """Test that client tracks calls correctly."""
        assert dummy_client.call_count == 0

        game_state = GameStateSnapshot(
            scene_id="test",
            player=PlayerState(name="Test", location="test"),
            world=WorldState(time_of_day="day"),
        )
        intent = PlayerIntent(type=PlayerIntentType.ENTER_SCENE)

        await game_policy.decide(game_state, intent)

        assert dummy_client.call_count == 1
        assert dummy_client.last_prompt is not None

        # Reset and verify
        dummy_client.reset()
        assert dummy_client.call_count == 0
        assert dummy_client.last_prompt is None
