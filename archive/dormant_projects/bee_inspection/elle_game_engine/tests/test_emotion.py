"""
Tests for emotion modeling system.

Tests EmotionalState and EmotionEngine functionality.
"""

import pytest
import time
from apps.elle_game_engine.emotion import (
    EmotionalState,
    EmotionEngine,
    EmotionDimension,
    EMOTION_LABELS,
)


class TestEmotionalState:
    """Tests for EmotionalState class."""

    def test_default_emotional_state(self):
        """Test default emotional state initialization."""
        state = EmotionalState()

        assert state.valence == 0.0
        assert state.arousal == 0.4
        assert state.dominance == 0.5
        assert state.trust == 0.5

    def test_emotional_state_clamping(self):
        """Test that emotional dimensions are clamped to valid ranges."""
        state = EmotionalState(
            valence=2.0,  # Should clamp to 1.0
            arousal=-0.5,  # Should clamp to 0.0
            dominance=1.5,  # Should clamp to 1.0
            trust=-1.0,  # Should clamp to 0.0
        )

        assert state.valence == 1.0
        assert state.arousal == 0.0
        assert state.dominance == 1.0
        assert state.trust == 0.0

    def test_emotional_state_update(self):
        """Test updating emotional state with deltas."""
        state = EmotionalState()

        state.update(
            valence_delta=0.5,
            arousal_delta=0.2,
            dominance_delta=-0.1,
            trust_delta=0.3,
        )

        assert state.valence == pytest.approx(0.5)
        assert state.arousal == pytest.approx(0.6)
        assert state.dominance == pytest.approx(0.4)
        assert state.trust == pytest.approx(0.8)

    def test_emotional_state_decay(self):
        """Test emotional decay toward baseline."""
        state = EmotionalState(valence=0.8, arousal=0.9)

        # Decay over 10 seconds
        state.decay(time_elapsed_seconds=10.0, decay_rate=0.1)

        # Should have decayed toward baseline
        assert state.valence < 0.8
        assert state.arousal < 0.9
        assert state.arousal > state.baseline_arousal

    def test_get_emotion_label_happy(self):
        """Test emotion label detection for happy state."""
        state = EmotionalState(valence=0.8, arousal=0.6, dominance=0.5)
        assert state.get_emotion_label() == "happy"

    def test_get_emotion_label_angry(self):
        """Test emotion label detection for angry state."""
        state = EmotionalState(valence=-0.6, arousal=0.8, dominance=0.7)
        assert state.get_emotion_label() == "angry"

    def test_get_emotion_label_sad(self):
        """Test emotion label detection for sad state."""
        state = EmotionalState(valence=-0.6, arousal=0.3, dominance=0.3)
        assert state.get_emotion_label() == "sad"

    def test_get_tone(self):
        """Test tone generation from emotional state."""
        happy_state = EmotionalState(valence=0.8, arousal=0.6)
        assert happy_state.get_tone() == "warm"

        angry_state = EmotionalState(valence=-0.6, arousal=0.8, dominance=0.7)
        assert angry_state.get_tone() == "stern"

        neutral_state = EmotionalState()
        assert neutral_state.get_tone() == "neutral"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        state = EmotionalState(valence=0.5, arousal=0.6, trust=0.8)
        data = state.to_dict()

        assert "valence" in data
        assert "arousal" in data
        assert "dominance" in data
        assert "trust" in data
        assert "emotion_label" in data
        assert "tone" in data
        assert data["valence"] == 0.5

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "valence": 0.7,
            "arousal": 0.8,
            "dominance": 0.6,
            "trust": 0.9,
        }

        state = EmotionalState.from_dict(data)

        assert state.valence == 0.7
        assert state.arousal == 0.8
        assert state.dominance == 0.6
        assert state.trust == 0.9

    def test_from_emotion_label(self):
        """Test creating emotional state from label."""
        happy_state = EmotionalState.from_emotion_label("happy")
        assert happy_state.valence > 0.5
        assert happy_state.get_emotion_label() == "happy"

        angry_state = EmotionalState.from_emotion_label("angry")
        assert angry_state.valence < 0.0
        assert angry_state.get_emotion_label() == "angry"


class TestEmotionEngine:
    """Tests for EmotionEngine class."""

    def test_engine_initialization(self):
        """Test emotion engine initialization."""
        engine = EmotionEngine()
        assert engine.decay_rate == 0.1
        assert engine.decay_check_interval == 10.0
        assert len(engine.emotion_history) == 0

    def test_apply_decay(self):
        """Test applying decay to emotional state."""
        engine = EmotionEngine()
        state = EmotionalState(valence=0.8, arousal=0.9)

        # Set last_update to 10 seconds ago
        state.last_update = time.time() - 10.0

        updated_state = engine.apply_decay(state)

        # Should have decayed
        assert updated_state.valence < 0.8
        assert updated_state.arousal < 0.9

    def test_process_player_action_help(self):
        """Test processing 'help' action."""
        engine = EmotionEngine()
        state = EmotionalState()

        updated_state = engine.process_player_action(
            state,
            action_type="help",
            npc_id="npc_1",
        )

        # Help should increase valence and trust
        assert updated_state.valence > 0.0
        assert updated_state.trust > 0.5

    def test_process_player_action_insult(self):
        """Test processing 'insult' action."""
        engine = EmotionEngine()
        state = EmotionalState()

        updated_state = engine.process_player_action(
            state,
            action_type="insult",
            npc_id="npc_1",
        )

        # Insult should decrease valence and trust
        assert updated_state.valence < 0.0
        assert updated_state.trust < 0.5

    def test_process_player_action_intensity(self):
        """Test action intensity scaling."""
        engine = EmotionEngine()
        state1 = EmotionalState()
        state2 = EmotionalState()

        # Low intensity
        updated1 = engine.process_player_action(
            state1,
            action_type="help",
            intensity=0.5,
        )

        # High intensity
        updated2 = engine.process_player_action(
            state2,
            action_type="help",
            intensity=2.0,
        )

        # Higher intensity should have larger effect
        assert updated2.valence > updated1.valence

    def test_track_emotion_history(self):
        """Test emotion history tracking."""
        engine = EmotionEngine()
        state = EmotionalState()

        # Process several actions
        for i in range(5):
            engine.process_player_action(state, "help", npc_id="npc_1")
            time.sleep(0.01)  # Small delay to ensure different timestamps

        history = engine.get_emotion_history("npc_1", limit=10)
        assert len(history) == 5

    def test_emotion_history_limit(self):
        """Test emotion history size limit."""
        engine = EmotionEngine()
        state = EmotionalState()

        # Process many actions (more than 100)
        for i in range(150):
            engine.process_player_action(state, "help", npc_id="npc_1")

        history = engine.get_emotion_history("npc_1", limit=200)
        # Should be limited to 100
        assert len(history) <= 100

    def test_generate_emotion_context(self):
        """Test generating emotion context for prompts."""
        engine = EmotionEngine()

        happy_state = EmotionalState(valence=0.8, arousal=0.6, trust=0.9)
        context = engine.generate_emotion_context(happy_state)

        assert "happy" in context
        assert "trusts the player" in context.lower()
        assert "warm" in context

    def test_generate_emotion_context_angry(self):
        """Test emotion context for angry NPC."""
        engine = EmotionEngine()

        angry_state = EmotionalState(valence=-0.6, arousal=0.8, trust=0.1)
        context = engine.generate_emotion_context(angry_state)

        assert "angry" in context
        assert ("distrusts" in context.lower() or "wary" in context.lower())
        assert "stern" in context

    def test_get_emotion_modifiers(self):
        """Test getting emotion-based modifiers."""
        engine = EmotionEngine()

        # Happy, trusting NPC
        happy_state = EmotionalState(valence=0.8, trust=0.9)
        modifiers = engine.get_emotion_modifiers(happy_state)

        assert "price_multiplier" in modifiers
        assert "quest_difficulty_modifier" in modifiers
        assert "hint_generosity" in modifiers

        # Price should be lower (high trust)
        assert modifiers["price_multiplier"] < 1.0

    def test_get_emotion_modifiers_angry(self):
        """Test modifiers for angry NPC."""
        engine = EmotionEngine()

        # Angry, distrusting NPC
        angry_state = EmotionalState(valence=-0.6, trust=0.2)
        modifiers = engine.get_emotion_modifiers(angry_state)

        # Price should be higher (low trust)
        assert modifiers["price_multiplier"] > 1.0

        # Quests should be harder (negative valence, low trust)
        assert modifiers["quest_difficulty_modifier"] > 1.0

        # Less helpful
        assert modifiers["hint_generosity"] < 0.5


@pytest.mark.asyncio
class TestEmotionIntegration:
    """Integration tests for emotion system."""

    def test_emotion_lifecycle(self):
        """Test complete emotion lifecycle."""
        engine = EmotionEngine()
        state = EmotionalState()

        # Player helps NPC
        state = engine.process_player_action(state, "help", npc_id="npc_1")
        assert state.valence > 0.0

        # Wait and apply decay
        time.sleep(0.1)
        state = engine.apply_decay(state)

        # Should have decayed slightly
        assert state.valence > 0.0

        # Player insults NPC
        state = engine.process_player_action(state, "insult", npc_id="npc_1")
        assert state.valence < 0.0

    def test_multiple_actions_accumulate(self):
        """Test that multiple actions accumulate."""
        engine = EmotionEngine()
        state = EmotionalState()

        # Multiple help actions
        for _ in range(5):
            state = engine.process_player_action(state, "help")

        # Valence should be high
        assert state.valence > 0.5
        assert state.trust > 0.7
