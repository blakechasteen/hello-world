"""
Tests for quest generation system.

Tests Quest, QuestObjective, QuestReward, and QuestGenerator.
"""

import pytest
import time
from apps.elle_game_engine.quest import (
    Quest,
    QuestObjective,
    QuestReward,
    QuestGenerator,
    QuestStatus,
    QuestDifficulty,
    QuestGenerationError,
)
from apps.elle_game_engine.emotion import EmotionalState
from apps.elle_game_engine.llm_client import DummyLLMClient


class TestQuestObjective:
    """Tests for QuestObjective class."""

    def test_objective_initialization(self):
        """Test quest objective initialization."""
        obj = QuestObjective(
            id="obj_1",
            description="Collect 5 herbs",
            target=5,
        )

        assert obj.id == "obj_1"
        assert obj.description == "Collect 5 herbs"
        assert obj.target == 5
        assert obj.progress == 0
        assert not obj.completed

    def test_objective_is_complete(self):
        """Test objective completion detection."""
        obj = QuestObjective(
            id="obj_1",
            description="Collect 5 herbs",
            target=5,
            progress=5,
        )

        assert obj.is_complete()

    def test_objective_is_complete_explicit(self):
        """Test explicit completion flag."""
        obj = QuestObjective(
            id="obj_1",
            description="Talk to guard",
            completed=True,
        )

        assert obj.is_complete()

    def test_objective_to_dict(self):
        """Test objective serialization."""
        obj = QuestObjective(
            id="obj_1",
            description="Collect 5 herbs",
            target=5,
            progress=3,
        )

        data = obj.to_dict()

        assert data["id"] == "obj_1"
        assert data["description"] == "Collect 5 herbs"
        assert data["target"] == 5
        assert data["progress"] == 3


class TestQuestReward:
    """Tests for QuestReward class."""

    def test_reward_initialization(self):
        """Test quest reward initialization."""
        reward = QuestReward(
            description="100 gold and a magic sword",
            xp=250,
            gold=100,
            items=["magic_sword"],
        )

        assert reward.description == "100 gold and a magic sword"
        assert reward.xp == 250
        assert reward.gold == 100
        assert "magic_sword" in reward.items

    def test_reward_to_dict(self):
        """Test reward serialization."""
        reward = QuestReward(
            description="Gold and XP",
            xp=100,
            gold=50,
        )

        data = reward.to_dict()

        assert data["description"] == "Gold and XP"
        assert data["xp"] == 100
        assert data["gold"] == 50


class TestQuest:
    """Tests for Quest class."""

    def test_quest_initialization(self):
        """Test quest initialization."""
        quest = Quest(
            id="quest_1",
            title="Fetch Quest",
            description="Bring me 5 herbs",
            giver_npc_id="npc_merchant",
            difficulty=QuestDifficulty.EASY,
        )

        assert quest.id == "quest_1"
        assert quest.title == "Fetch Quest"
        assert quest.status == QuestStatus.AVAILABLE
        assert quest.difficulty == QuestDifficulty.EASY

    def test_quest_validation_empty_id(self):
        """Test that empty quest ID raises error."""
        with pytest.raises(ValueError, match="Quest ID cannot be empty"):
            Quest(
                id="",
                title="Test",
                description="Test",
                giver_npc_id="npc_1",
                difficulty=QuestDifficulty.NORMAL,
            )

    def test_quest_accept(self):
        """Test accepting a quest."""
        quest = Quest(
            id="quest_1",
            title="Test Quest",
            description="Test",
            giver_npc_id="npc_1",
            difficulty=QuestDifficulty.NORMAL,
        )

        quest.accept()

        assert quest.status == QuestStatus.ACTIVE
        assert quest.accepted_at is not None

    def test_quest_complete(self):
        """Test completing a quest."""
        quest = Quest(
            id="quest_1",
            title="Test Quest",
            description="Test",
            giver_npc_id="npc_1",
            difficulty=QuestDifficulty.NORMAL,
            objectives=[
                QuestObjective(id="obj_1", description="Test", completed=True)
            ],
        )

        quest.accept()
        quest.complete()

        assert quest.status == QuestStatus.COMPLETED
        assert quest.completed_at is not None

    def test_quest_complete_fails_without_objectives(self):
        """Test that completing quest with incomplete objectives fails."""
        quest = Quest(
            id="quest_1",
            title="Test Quest",
            description="Test",
            giver_npc_id="npc_1",
            difficulty=QuestDifficulty.NORMAL,
            objectives=[
                QuestObjective(id="obj_1", description="Test", completed=False)
            ],
        )

        quest.accept()

        with pytest.raises(ValueError, match="not complete"):
            quest.complete()

    def test_quest_fail(self):
        """Test failing a quest."""
        quest = Quest(
            id="quest_1",
            title="Test Quest",
            description="Test",
            giver_npc_id="npc_1",
            difficulty=QuestDifficulty.NORMAL,
        )

        quest.accept()
        quest.fail()

        assert quest.status == QuestStatus.FAILED

    def test_quest_time_limit(self):
        """Test quest time limit expiration."""
        quest = Quest(
            id="quest_1",
            title="Timed Quest",
            description="Hurry!",
            giver_npc_id="npc_1",
            difficulty=QuestDifficulty.NORMAL,
            time_limit_seconds=0.1,  # 100ms
        )

        quest.accept()

        # Wait for expiration
        time.sleep(0.2)

        expired = quest.check_time_limit()

        assert expired
        assert quest.status == QuestStatus.EXPIRED

    def test_quest_completion_percentage(self):
        """Test quest completion percentage calculation."""
        quest = Quest(
            id="quest_1",
            title="Multi-step Quest",
            description="Test",
            giver_npc_id="npc_1",
            difficulty=QuestDifficulty.NORMAL,
            objectives=[
                QuestObjective(id="obj_1", description="Step 1", completed=True),
                QuestObjective(id="obj_2", description="Step 2", completed=False),
                QuestObjective(id="obj_3", description="Step 3", completed=False),
            ],
        )

        percentage = quest.get_completion_percentage()
        assert percentage == pytest.approx(1/3, rel=0.01)

    def test_quest_to_dict(self):
        """Test quest serialization."""
        quest = Quest(
            id="quest_1",
            title="Test Quest",
            description="Test description",
            giver_npc_id="npc_1",
            difficulty=QuestDifficulty.NORMAL,
            objectives=[
                QuestObjective(id="obj_1", description="Step 1")
            ],
        )

        data = quest.to_dict()

        assert data["id"] == "quest_1"
        assert data["title"] == "Test Quest"
        assert data["difficulty"] == "normal"
        assert data["status"] == "available"
        assert len(data["objectives"]) == 1


class TestQuestGenerator:
    """Tests for QuestGenerator class."""

    @pytest.mark.asyncio
    async def test_generator_initialization(self):
        """Test quest generator initialization."""
        llm_client = DummyLLMClient()
        generator = QuestGenerator(llm_client)

        assert generator.llm_client is not None
        assert generator.system_prompt is not None

    @pytest.mark.asyncio
    async def test_generate_quest_basic(self):
        """Test basic quest generation."""
        # Create dummy LLM client that returns valid quest JSON
        llm_client = DummyLLMClient()

        # Override complete method to return valid quest JSON
        async def mock_complete(prompt, **kwargs):
            return """{
                "title": "Test Quest",
                "description": "A test quest",
                "difficulty": "normal",
                "objectives": [
                    {"description": "Test objective", "optional": false, "target": 1}
                ],
                "reward": {
                    "description": "Test reward",
                    "xp": 100,
                    "gold": 50
                },
                "time_limit_seconds": null,
                "emotional_rationale": "Test rationale"
            }"""

        llm_client.complete = mock_complete

        generator = QuestGenerator(llm_client)

        emotional_state = EmotionalState()
        quest = await generator.generate_quest(
            npc_id="npc_1",
            npc_name="Test NPC",
            npc_role="merchant",
            emotional_state=emotional_state,
        )

        assert quest is not None
        assert quest.title == "Test Quest"
        assert quest.description == "A test quest"
        assert quest.difficulty == QuestDifficulty.NORMAL
        assert len(quest.objectives) == 1
        assert quest.reward is not None

    @pytest.mark.asyncio
    async def test_generate_quest_with_emotion(self):
        """Test quest generation with emotional context."""
        llm_client = DummyLLMClient()

        async def mock_complete(prompt, **kwargs):
            # Verify emotion context is in prompt
            assert "angry" in prompt or "happy" in prompt or "neutral" in prompt

            return """{
                "title": "Emotion-Based Quest",
                "description": "Quest based on emotion",
                "difficulty": "hard",
                "objectives": [
                    {"description": "Difficult objective", "optional": false, "target": 1}
                ],
                "reward": {
                    "description": "Reward",
                    "xp": 200,
                    "gold": 100
                }
            }"""

        llm_client.complete = mock_complete

        generator = QuestGenerator(llm_client)

        # Angry NPC should generate harder quest
        angry_state = EmotionalState(valence=-0.6, arousal=0.8, trust=0.2)
        quest = await generator.generate_quest(
            npc_id="npc_1",
            npc_name="Angry Guard",
            npc_role="guard",
            emotional_state=angry_state,
        )

        assert quest.difficulty == QuestDifficulty.HARD

    @pytest.mark.asyncio
    async def test_generate_quest_invalid_json(self):
        """Test handling of invalid JSON response."""
        llm_client = DummyLLMClient()

        async def mock_complete(prompt, **kwargs):
            return "This is not valid JSON"

        llm_client.complete = mock_complete

        generator = QuestGenerator(llm_client)
        emotional_state = EmotionalState()

        with pytest.raises(QuestGenerationError, match="Invalid JSON"):
            await generator.generate_quest(
                npc_id="npc_1",
                npc_name="Test",
                npc_role="merchant",
                emotional_state=emotional_state,
            )

    @pytest.mark.asyncio
    async def test_suggest_difficulty_based_on_trust(self):
        """Test difficulty suggestion based on trust."""
        llm_client = DummyLLMClient()
        generator = QuestGenerator(llm_client)

        # Low trust should suggest harder difficulty
        low_trust_state = EmotionalState(trust=0.2)
        difficulty = generator._suggest_difficulty(
            low_trust_state,
            player_level=5,
            world_tension="calm",
        )

        assert difficulty in ["normal", "hard"]

    @pytest.mark.asyncio
    async def test_suggest_difficulty_based_on_tension(self):
        """Test difficulty suggestion based on world tension."""
        llm_client = DummyLLMClient()
        generator = QuestGenerator(llm_client)

        # High tension should suggest harder difficulty
        neutral_state = EmotionalState()
        difficulty = generator._suggest_difficulty(
            neutral_state,
            player_level=3,
            world_tension="critical",
        )

        assert difficulty in ["normal", "hard"]


@pytest.mark.asyncio
class TestQuestIntegration:
    """Integration tests for quest system."""

    async def test_quest_lifecycle(self):
        """Test complete quest lifecycle."""
        # Create quest
        quest = Quest(
            id="quest_1",
            title="Test Quest",
            description="Complete test",
            giver_npc_id="npc_1",
            difficulty=QuestDifficulty.NORMAL,
            objectives=[
                QuestObjective(id="obj_1", description="Step 1", target=1),
                QuestObjective(id="obj_2", description="Step 2", target=1),
            ],
            reward=QuestReward(
                description="Reward",
                xp=100,
                gold=50,
            ),
        )

        # Quest starts as available
        assert quest.status == QuestStatus.AVAILABLE

        # Accept quest
        quest.accept()
        assert quest.status == QuestStatus.ACTIVE

        # Complete objectives
        quest.objectives[0].completed = True
        assert quest.get_completion_percentage() == 0.5

        quest.objectives[1].completed = True
        assert quest.get_completion_percentage() == 1.0

        # Complete quest
        quest.complete()
        assert quest.status == QuestStatus.COMPLETED

    async def test_emotion_affects_quest_generation(self):
        """Test that NPC emotion affects quest difficulty."""
        llm_client = DummyLLMClient()

        async def mock_complete(prompt, **kwargs):
            # Check if emotion context is present
            has_angry = "angry" in prompt.lower()
            has_low_trust = "distrusts" in prompt.lower()

            difficulty = "hard" if (has_angry or has_low_trust) else "normal"

            return f"""{{
                "title": "Emotion Test Quest",
                "description": "Quest based on emotion",
                "difficulty": "{difficulty}",
                "objectives": [
                    {{"description": "Test", "optional": false, "target": 1}}
                ],
                "reward": {{
                    "description": "Reward",
                    "xp": 100,
                    "gold": 50
                }}
            }}"""

        llm_client.complete = mock_complete

        generator = QuestGenerator(llm_client)

        # Angry NPC
        angry_state = EmotionalState(valence=-0.6, arousal=0.8, trust=0.2)
        quest = await generator.generate_quest(
            npc_id="npc_1",
            npc_name="Angry NPC",
            npc_role="guard",
            emotional_state=angry_state,
        )

        # Should generate harder quest
        assert quest.difficulty == QuestDifficulty.HARD
