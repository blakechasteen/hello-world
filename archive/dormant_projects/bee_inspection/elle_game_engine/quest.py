"""
Dynamic quest generation system for Elle Game Engine.

Uses LLM to generate contextual, emotionally-aware quests based on:
- World state (time, weather, tension)
- Player state (progress, reputation, traits)
- NPC emotional state (valence, trust)
- Recent actions and narrative history

Created: 2025-11-16
"""

import json
import uuid
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from enum import Enum

from .emotion import EmotionalState
from .llm_client import BaseLLMClient


# ============================================================================
# Quest Models
# ============================================================================

class QuestStatus(str, Enum):
    """Quest completion status."""
    AVAILABLE = "available"  # Quest can be accepted
    ACTIVE = "active"  # Quest is in progress
    COMPLETED = "completed"  # Quest completed successfully
    FAILED = "failed"  # Quest failed
    EXPIRED = "expired"  # Quest time limit exceeded


class QuestDifficulty(str, Enum):
    """Quest difficulty levels."""
    TRIVIAL = "trivial"  # Very easy
    EASY = "easy"  # Below player level
    NORMAL = "normal"  # At player level
    HARD = "hard"  # Above player level
    EPIC = "epic"  # Significantly challenging


@dataclass
class QuestObjective:
    """Individual quest objective/step."""

    id: str
    description: str
    completed: bool = False
    optional: bool = False  # Can be skipped
    progress: int = 0  # Current progress (e.g., 3/5 items collected)
    target: int = 1  # Target progress (e.g., collect 5 items)

    def is_complete(self) -> bool:
        """Check if objective is complete."""
        return self.completed or (self.progress >= self.target)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "completed": self.completed,
            "optional": self.optional,
            "progress": self.progress,
            "target": self.target,
        }


@dataclass
class QuestReward:
    """Quest completion rewards."""

    description: str  # Human-readable description
    xp: int = 0  # Experience points
    gold: int = 0  # Currency
    items: List[str] = field(default_factory=list)  # Item IDs
    reputation_changes: Dict[str, int] = field(default_factory=dict)  # Faction reputation
    flags: Dict[str, bool] = field(default_factory=dict)  # World flags to set

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "description": self.description,
            "xp": self.xp,
            "gold": self.gold,
            "items": self.items,
            "reputation_changes": self.reputation_changes,
            "flags": self.flags,
        }


@dataclass
class Quest:
    """
    Complete quest definition.

    Quests are multi-objective tasks that:
    - Adapt to player level and progress
    - Reflect NPC emotional state
    - Have time limits and prerequisites
    - Provide meaningful rewards
    """

    id: str
    title: str
    description: str

    # Quest metadata
    giver_npc_id: str  # NPC who gave the quest
    difficulty: QuestDifficulty
    status: QuestStatus = QuestStatus.AVAILABLE

    # Objectives
    objectives: List[QuestObjective] = field(default_factory=list)

    # Rewards
    reward: Optional[QuestReward] = None

    # Prerequisites
    required_flags: Dict[str, bool] = field(default_factory=dict)  # Must have these flags
    required_level: int = 1  # Minimum player level

    # Time limits
    time_limit_seconds: Optional[float] = None  # None = no limit
    created_at: float = field(default_factory=time.time)
    accepted_at: Optional[float] = None
    completed_at: Optional[float] = None

    # Context
    emotion_at_creation: Optional[str] = None  # NPC emotion when quest created
    world_context: Dict[str, Any] = field(default_factory=dict)  # Weather, tension, etc.

    def __post_init__(self):
        """Validate quest."""
        if not self.id:
            raise ValueError("Quest ID cannot be empty")
        if not self.title:
            raise ValueError("Quest title cannot be empty")
        if not self.giver_npc_id:
            raise ValueError("Quest giver NPC ID cannot be empty")

    def accept(self):
        """Mark quest as accepted and active."""
        if self.status != QuestStatus.AVAILABLE:
            raise ValueError(f"Cannot accept quest in status {self.status}")

        self.status = QuestStatus.ACTIVE
        self.accepted_at = time.time()

    def complete(self):
        """Mark quest as completed."""
        if self.status != QuestStatus.ACTIVE:
            raise ValueError(f"Cannot complete quest in status {self.status}")

        # Check if all required objectives are complete
        for obj in self.objectives:
            if not obj.optional and not obj.is_complete():
                raise ValueError(f"Objective '{obj.description}' not complete")

        self.status = QuestStatus.COMPLETED
        self.completed_at = time.time()

    def fail(self):
        """Mark quest as failed."""
        if self.status not in [QuestStatus.AVAILABLE, QuestStatus.ACTIVE]:
            raise ValueError(f"Cannot fail quest in status {self.status}")

        self.status = QuestStatus.FAILED
        self.completed_at = time.time()

    def check_time_limit(self) -> bool:
        """
        Check if quest has expired.

        Returns:
            True if quest expired, False otherwise
        """
        if self.time_limit_seconds is None:
            return False

        if self.accepted_at is None:
            return False

        elapsed = time.time() - self.accepted_at

        if elapsed > self.time_limit_seconds:
            self.status = QuestStatus.EXPIRED
            return True

        return False

    def get_completion_percentage(self) -> float:
        """
        Get quest completion percentage.

        Returns:
            Completion percentage (0.0 to 1.0)
        """
        if not self.objectives:
            return 0.0

        completed = sum(1 for obj in self.objectives if obj.is_complete())
        total = len([obj for obj in self.objectives if not obj.optional])

        return completed / total if total > 0 else 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "giver_npc_id": self.giver_npc_id,
            "difficulty": self.difficulty.value,
            "status": self.status.value,
            "objectives": [obj.to_dict() for obj in self.objectives],
            "reward": self.reward.to_dict() if self.reward else None,
            "required_flags": self.required_flags,
            "required_level": self.required_level,
            "time_limit_seconds": self.time_limit_seconds,
            "created_at": self.created_at,
            "accepted_at": self.accepted_at,
            "completed_at": self.completed_at,
            "emotion_at_creation": self.emotion_at_creation,
            "world_context": self.world_context,
            "completion_percentage": self.get_completion_percentage(),
        }


# ============================================================================
# Quest Generator
# ============================================================================

class QuestGenerator:
    """
    LLM-powered quest generation engine.

    Generates contextual quests based on:
    - NPC emotional state (affects quest tone and difficulty)
    - World state (time, weather, tension)
    - Player state (level, reputation, traits)
    - Recent narrative events
    """

    def __init__(self, llm_client: BaseLLMClient):
        """
        Initialize quest generator.

        Args:
            llm_client: LLM client for generation
        """
        self.llm_client = llm_client

        # Quest generation prompt
        self.system_prompt = """You are a quest generation engine for an RPG game.

Your task is to generate engaging, contextual quests based on game state and NPC emotions.

Rules:
1. Quests should match NPC emotional state:
   - Happy/grateful NPCs give helpful, rewarding quests
   - Angry/suspicious NPCs give difficult, risky quests
   - Sad NPCs give emotionally-driven quests (rescue, recovery)
   - Fearful NPCs give defensive quests (protection, investigation)

2. Difficulty scales with:
   - Player level/progress
   - NPC trust (low trust = harder quests)
   - World tension (high tension = more challenging)
   - NPC emotional valence (negative = harder)

3. Objectives should be:
   - Clear and actionable
   - 2-5 steps for normal quests
   - Varied (collect, defeat, explore, talk, deliver)

4. Rewards should be:
   - Proportional to difficulty
   - Contextual to quest giver (merchants give gold, guards give reputation)

5. Time limits (if any):
   - Urgent quests: 5-15 minutes
   - Normal quests: no limit or 30-60 minutes
   - Long quests: 2-4 hours

IMPORTANT: Return ONLY valid JSON in the exact format below. No markdown, no explanations.

{
  "title": "Quest title (short, 3-7 words)",
  "description": "Quest description (2-3 sentences, NPC's voice)",
  "difficulty": "trivial|easy|normal|hard|epic",
  "objectives": [
    {
      "description": "First objective",
      "optional": false,
      "target": 1
    }
  ],
  "reward": {
    "description": "Reward description",
    "xp": 100,
    "gold": 50,
    "items": ["item_id"],
    "reputation_changes": {"faction": 10}
  },
  "time_limit_seconds": 600.0,
  "emotional_rationale": "Why the NPC's emotion led to this quest"
}
"""

    async def generate_quest(
        self,
        npc_id: str,
        npc_name: str,
        npc_role: str,
        emotional_state: EmotionalState,
        player_level: int = 1,
        player_reputation: Optional[str] = None,
        world_tension: str = "calm",
        world_time: str = "day",
        recent_events: Optional[str] = None,
    ) -> Quest:
        """
        Generate a contextual quest.

        Args:
            npc_id: NPC ID
            npc_name: NPC name
            npc_role: NPC role (merchant, guard, etc.)
            emotional_state: NPC's current emotional state
            player_level: Player level (affects difficulty)
            player_reputation: Player reputation
            world_tension: World tension level
            world_time: Time of day
            recent_events: Recent narrative events (optional)

        Returns:
            Generated quest

        Raises:
            QuestGenerationError: If generation fails
        """
        try:
            # Build context prompt
            user_prompt = self._build_generation_prompt(
                npc_id=npc_id,
                npc_name=npc_name,
                npc_role=npc_role,
                emotional_state=emotional_state,
                player_level=player_level,
                player_reputation=player_reputation,
                world_tension=world_tension,
                world_time=world_time,
                recent_events=recent_events,
            )

            # Call LLM
            response_text = await self.llm_client.complete(
                prompt=user_prompt,
                system_prompt=self.system_prompt,
                max_tokens=800,
                temperature=0.8,  # Higher creativity for quest generation
            )

            # Parse response
            quest_data = self._parse_quest_response(response_text)

            # Build Quest object
            quest = self._build_quest(
                quest_data=quest_data,
                npc_id=npc_id,
                emotional_state=emotional_state,
                world_tension=world_tension,
                world_time=world_time,
            )

            return quest

        except Exception as e:
            raise QuestGenerationError(f"Failed to generate quest: {e}") from e

    def _build_generation_prompt(
        self,
        npc_id: str,
        npc_name: str,
        npc_role: str,
        emotional_state: EmotionalState,
        player_level: int,
        player_reputation: Optional[str],
        world_tension: str,
        world_time: str,
        recent_events: Optional[str],
    ) -> str:
        """Build user prompt for quest generation."""

        emotion_label = emotional_state.get_emotion_label()
        trust = emotional_state.trust
        valence = emotional_state.valence

        # Trust descriptor
        if trust < 0.3:
            trust_desc = "distrusts the player"
        elif trust < 0.6:
            trust_desc = "is neutral toward the player"
        else:
            trust_desc = "trusts the player"

        # Suggested difficulty based on emotion
        suggested_difficulty = self._suggest_difficulty(
            emotional_state, player_level, world_tension
        )

        prompt = f"""Generate a quest for the following context:

NPC: {npc_name} ({npc_role})
Emotion: {emotion_label} (valence: {valence:.2f}, trust: {trust:.2f})
The NPC {trust_desc}.

Player Level: {player_level}
Player Reputation: {player_reputation or 'neutral'}

World State:
- Time: {world_time}
- Tension: {world_tension}

Suggested Difficulty: {suggested_difficulty}

{f'Recent Events: {recent_events}' if recent_events else ''}

Generate a quest that:
1. Reflects the NPC's {emotion_label} emotional state
2. Matches their {trust_desc} attitude
3. Is appropriate for player level {player_level}
4. Fits the {world_tension} world tension

Return ONLY the JSON quest definition.
"""
        return prompt

    def _suggest_difficulty(
        self,
        emotional_state: EmotionalState,
        player_level: int,
        world_tension: str,
    ) -> str:
        """Suggest quest difficulty based on context."""

        # Base difficulty from player level
        if player_level <= 3:
            base_difficulty = "easy"
        elif player_level <= 7:
            base_difficulty = "normal"
        else:
            base_difficulty = "hard"

        # Modifiers
        trust = emotional_state.trust
        valence = emotional_state.valence

        # Low trust = harder
        if trust < 0.3:
            if base_difficulty == "easy":
                base_difficulty = "normal"
            elif base_difficulty == "normal":
                base_difficulty = "hard"

        # Negative emotion = harder
        if valence < -0.5:
            if base_difficulty == "easy":
                base_difficulty = "normal"

        # High tension = harder
        if world_tension in ["tense", "high", "critical"]:
            if base_difficulty == "easy":
                base_difficulty = "normal"

        return base_difficulty

    def _parse_quest_response(self, response_text: str) -> Dict:
        """Parse LLM response into quest data."""

        # Clean response (remove markdown if present)
        cleaned = response_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise QuestGenerationError(
                f"Invalid JSON response: {e}\nResponse: {response_text}"
            ) from e

        # Validate required fields
        required = ["title", "description", "difficulty", "objectives"]
        for field in required:
            if field not in data:
                raise QuestGenerationError(f"Missing required field: {field}")

        return data

    def _build_quest(
        self,
        quest_data: Dict,
        npc_id: str,
        emotional_state: EmotionalState,
        world_tension: str,
        world_time: str,
    ) -> Quest:
        """Build Quest object from parsed data."""

        # Generate quest ID
        quest_id = str(uuid.uuid4())

        # Parse difficulty
        try:
            difficulty = QuestDifficulty(quest_data["difficulty"])
        except ValueError:
            difficulty = QuestDifficulty.NORMAL

        # Parse objectives
        objectives = []
        for i, obj_data in enumerate(quest_data.get("objectives", [])):
            objectives.append(QuestObjective(
                id=f"{quest_id}_obj_{i}",
                description=obj_data["description"],
                optional=obj_data.get("optional", False),
                target=obj_data.get("target", 1),
            ))

        # Parse reward
        reward = None
        if "reward" in quest_data:
            reward_data = quest_data["reward"]
            reward = QuestReward(
                description=reward_data.get("description", ""),
                xp=reward_data.get("xp", 0),
                gold=reward_data.get("gold", 0),
                items=reward_data.get("items", []),
                reputation_changes=reward_data.get("reputation_changes", {}),
                flags=reward_data.get("flags", {}),
            )

        # Build quest
        quest = Quest(
            id=quest_id,
            title=quest_data["title"],
            description=quest_data["description"],
            giver_npc_id=npc_id,
            difficulty=difficulty,
            objectives=objectives,
            reward=reward,
            time_limit_seconds=quest_data.get("time_limit_seconds"),
            emotion_at_creation=emotional_state.get_emotion_label(),
            world_context={
                "tension": world_tension,
                "time": world_time,
            },
        )

        return quest


class QuestGenerationError(Exception):
    """Error during quest generation."""
    pass
