"""Tests for game engine data models."""

import pytest
from apps.elle_game_engine.models import (
    NPCState,
    PlayerState,
    WorldState,
    GameStateSnapshot,
    PlayerIntent,
    PlayerIntentType,
    ElleGameAction,
    ActionMode,
    DialogueLine,
    WorldChange,
)


class TestNPCState:
    """Tests for NPCState model."""

    def test_create_npc_minimal(self):
        """Test creating NPC with minimal required fields."""
        npc = NPCState(id="guard_1", name="Guard", role="guard")
        assert npc.id == "guard_1"
        assert npc.name == "Guard"
        assert npc.role == "guard"
        assert npc.mood is None
        assert npc.location == ""
        assert npc.flags == {}

    def test_create_npc_full(self):
        """Test creating NPC with all fields."""
        npc = NPCState(
            id="innkeeper_1",
            name="Innkeeper Bob",
            role="innkeeper",
            mood="nervous",
            location="inn",
            flags={"knows_secret": True, "quest_given": False},
        )
        assert npc.id == "innkeeper_1"
        assert npc.mood == "nervous"
        assert npc.location == "inn"
        assert npc.flags["knows_secret"] is True

    def test_npc_validation_empty_id(self):
        """Test that empty ID raises error."""
        with pytest.raises(ValueError, match="id cannot be empty"):
            NPCState(id="", name="Test", role="test")

    def test_npc_validation_empty_name(self):
        """Test that empty name raises error."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            NPCState(id="test", name="", role="test")

    def test_npc_validation_empty_role(self):
        """Test that empty role raises error."""
        with pytest.raises(ValueError, match="role cannot be empty"):
            NPCState(id="test", name="Test", role="")


class TestPlayerState:
    """Tests for PlayerState model."""

    def test_create_player_minimal(self):
        """Test creating player with minimal fields."""
        player = PlayerState(name="Hero", location="village")
        assert player.name == "Hero"
        assert player.location == "village"
        assert player.quest_stage is None
        assert player.reputation is None
        assert player.traits == {}
        assert player.inventory_tags == []

    def test_create_player_full(self):
        """Test creating player with all fields."""
        player = PlayerState(
            name="Hero",
            location="village_square",
            quest_stage="mid",
            reputation="hero",
            traits={"brave": 3, "kind": 2},
            inventory_tags=["sword", "healing_potion"],
        )
        assert player.quest_stage == "mid"
        assert player.reputation == "hero"
        assert player.traits["brave"] == 3
        assert "sword" in player.inventory_tags

    def test_player_validation_empty_name(self):
        """Test that empty name raises error."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            PlayerState(name="", location="test")

    def test_player_validation_empty_location(self):
        """Test that empty location raises error."""
        with pytest.raises(ValueError, match="location cannot be empty"):
            PlayerState(name="Test", location="")


class TestWorldState:
    """Tests for WorldState model."""

    def test_create_world_minimal(self):
        """Test creating world with minimal fields."""
        world = WorldState(time_of_day="morning")
        assert world.time_of_day == "morning"
        assert world.weather is None
        assert world.tension_level is None

    def test_create_world_full(self):
        """Test creating world with all fields."""
        world = WorldState(
            time_of_day="night",
            weather="storm",
            tension_level="high",
        )
        assert world.time_of_day == "night"
        assert world.weather == "storm"
        assert world.tension_level == "high"

    def test_world_validation_empty_time(self):
        """Test that empty time_of_day raises error."""
        with pytest.raises(ValueError, match="time_of_day cannot be empty"):
            WorldState(time_of_day="")


class TestGameStateSnapshot:
    """Tests for GameStateSnapshot model."""

    def test_create_snapshot_minimal(self):
        """Test creating snapshot with minimal fields."""
        snapshot = GameStateSnapshot(scene_id="test_scene")
        assert snapshot.scene_id == "test_scene"
        assert snapshot.npcs == []
        assert snapshot.player.name == "Player"
        assert snapshot.world.time_of_day == "day"
        assert snapshot.tags == []

    def test_create_snapshot_full(self):
        """Test creating snapshot with all fields."""
        npcs = [
            NPCState(id="guard_1", name="Guard", role="guard", mood="alert"),
            NPCState(id="merchant_1", name="Merchant", role="merchant", mood="friendly"),
        ]
        player = PlayerState(name="Hero", location="village_square")
        world = WorldState(time_of_day="afternoon", weather="clear")

        snapshot = GameStateSnapshot(
            scene_id="village_square",
            npcs=npcs,
            player=player,
            world=world,
            tags=["first_visit", "peaceful"],
        )

        assert snapshot.npc_count == 2
        assert snapshot.has_tag("first_visit")
        assert not snapshot.has_tag("combat")

    def test_get_npc_by_id(self):
        """Test finding NPC by ID."""
        npcs = [
            NPCState(id="guard_1", name="Guard", role="guard"),
            NPCState(id="merchant_1", name="Merchant", role="merchant"),
        ]
        snapshot = GameStateSnapshot(scene_id="test", npcs=npcs)

        guard = snapshot.get_npc_by_id("guard_1")
        assert guard is not None
        assert guard.name == "Guard"

        nonexistent = snapshot.get_npc_by_id("missing")
        assert nonexistent is None

    def test_snapshot_validation_empty_scene(self):
        """Test that empty scene_id raises error."""
        with pytest.raises(ValueError, match="scene_id cannot be empty"):
            GameStateSnapshot(scene_id="")


class TestPlayerIntent:
    """Tests for PlayerIntent model."""

    def test_create_intent_talk_to_npc(self):
        """Test creating TALK_TO_NPC intent."""
        intent = PlayerIntent(
            type=PlayerIntentType.TALK_TO_NPC,
            target_npc_id="innkeeper_1",
            raw_input="Hello there!",
        )
        assert intent.type == PlayerIntentType.TALK_TO_NPC
        assert intent.target_npc_id == "innkeeper_1"
        assert intent.raw_input == "Hello there!"

    def test_create_intent_enter_scene(self):
        """Test creating ENTER_SCENE intent."""
        intent = PlayerIntent(type=PlayerIntentType.ENTER_SCENE)
        assert intent.type == PlayerIntentType.ENTER_SCENE
        assert intent.target_npc_id is None

    def test_create_intent_request_hint(self):
        """Test creating REQUEST_HINT intent."""
        intent = PlayerIntent(
            type=PlayerIntentType.REQUEST_HINT,
            raw_input="I'm stuck on this puzzle",
        )
        assert intent.type == PlayerIntentType.REQUEST_HINT
        assert intent.raw_input == "I'm stuck on this puzzle"

    def test_intent_validation_talk_requires_npc(self):
        """Test that TALK_TO_NPC requires target_npc_id."""
        with pytest.raises(ValueError, match="target_npc_id required"):
            PlayerIntent(type=PlayerIntentType.TALK_TO_NPC)


class TestDialogueLine:
    """Tests for DialogueLine model."""

    def test_create_dialogue_minimal(self):
        """Test creating dialogue with minimal fields."""
        dialogue = DialogueLine(npc_id="guard_1", text="Hello traveler")
        assert dialogue.npc_id == "guard_1"
        assert dialogue.text == "Hello traveler"
        assert dialogue.tone is None

    def test_create_dialogue_with_tone(self):
        """Test creating dialogue with tone."""
        dialogue = DialogueLine(
            npc_id="innkeeper_1",
            text="Welcome to my inn!",
            tone="warm",
        )
        assert dialogue.tone == "warm"

    def test_dialogue_validation_empty_npc_id(self):
        """Test that empty npc_id raises error."""
        with pytest.raises(ValueError, match="npc_id cannot be empty"):
            DialogueLine(npc_id="", text="test")

    def test_dialogue_validation_empty_text(self):
        """Test that empty text raises error."""
        with pytest.raises(ValueError, match="text cannot be empty"):
            DialogueLine(npc_id="test", text="")


class TestWorldChange:
    """Tests for WorldChange model."""

    def test_create_world_change_minimal(self):
        """Test creating world change with minimal fields."""
        change = WorldChange(description="The door creaks open")
        assert change.description == "The door creaks open"
        assert change.flag_changes == {}

    def test_create_world_change_with_flags(self):
        """Test creating world change with flag changes."""
        change = WorldChange(
            description="You discover a secret passage",
            flag_changes={"secret_passage_found": True, "door_locked": False},
        )
        assert change.flag_changes["secret_passage_found"] is True
        assert change.flag_changes["door_locked"] is False

    def test_world_change_validation_empty_description(self):
        """Test that empty description raises error."""
        with pytest.raises(ValueError, match="description cannot be empty"):
            WorldChange(description="")


class TestElleGameAction:
    """Tests for ElleGameAction model."""

    def test_create_action_npc_dialogue(self):
        """Test creating NPC dialogue action."""
        dialogue = [
            DialogueLine(npc_id="guard_1", text="Halt!", tone="stern")
        ]
        action = ElleGameAction(
            mode=ActionMode.NPC_DIALOGUE,
            priority="high",
            dialogue=dialogue,
        )
        assert action.mode == ActionMode.NPC_DIALOGUE
        assert action.priority == "high"
        assert action.has_dialogue
        assert not action.has_world_changes

    def test_create_action_hint(self):
        """Test creating hint action."""
        action = ElleGameAction(
            mode=ActionMode.HINT,
            priority="medium",
            hint_text="Try examining the bookshelf more carefully",
        )
        assert action.mode == ActionMode.HINT
        assert action.hint_text is not None
        assert not action.has_dialogue

    def test_create_action_world_reaction(self):
        """Test creating world reaction action."""
        change = WorldChange(
            description="The room grows darker",
            flag_changes={"lights_out": True},
        )
        action = ElleGameAction(
            mode=ActionMode.WORLD_REACTION,
            priority="medium",
            world_reaction=change,
        )
        assert action.mode == ActionMode.WORLD_REACTION
        assert action.has_world_changes

    def test_action_validation_invalid_priority(self):
        """Test that invalid priority raises error."""
        with pytest.raises(ValueError, match="Invalid priority"):
            ElleGameAction(
                mode=ActionMode.HINT,
                priority="extreme",
                hint_text="test",
            )

    def test_action_validation_dialogue_mode_requires_dialogue(self):
        """Test that NPC_DIALOGUE mode requires dialogue."""
        with pytest.raises(ValueError, match="requires dialogue"):
            ElleGameAction(
                mode=ActionMode.NPC_DIALOGUE,
                priority="medium",
            )

    def test_action_validation_hint_mode_requires_text(self):
        """Test that HINT mode requires hint_text."""
        with pytest.raises(ValueError, match="requires hint_text"):
            ElleGameAction(
                mode=ActionMode.HINT,
                priority="medium",
            )

    def test_action_to_dict(self):
        """Test converting action to dictionary."""
        dialogue = [DialogueLine(npc_id="guard_1", text="Halt!")]
        action = ElleGameAction(
            mode=ActionMode.NPC_DIALOGUE,
            priority="high",
            dialogue=dialogue,
            debug_notes="Test note",
        )
        data = action.to_dict()

        assert data["mode"] == "npc_dialogue"
        assert data["priority"] == "high"
        assert len(data["dialogue"]) == 1
        assert data["dialogue"][0]["text"] == "Halt!"
        assert data["debug_notes"] == "Test note"
