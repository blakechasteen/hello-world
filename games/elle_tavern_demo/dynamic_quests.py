"""
Dynamic quest generation integration for Elle Tavern Demo.

Demonstrates how to use Elle's QuestGenerator to create emotion-aware quests
that adapt to NPC emotional state, player progress, and world conditions.

Features:
- Dynamic quest generation using LLM
- Emotion-based difficulty scaling
- Quest template system for consistency
- Hybrid approach (handcrafted + procedural)

Created: 2025-11-16
"""

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Import Elle Game Engine components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../apps'))

from elle_game_engine.quest import QuestGenerator, Quest, QuestDifficulty
from elle_game_engine.emotion import EmotionalState
from elle_game_engine.llm_client import create_llm_client


# ============================================================================
# Quest Templates
# ============================================================================

QUEST_TEMPLATES = {
    "fetch": {
        "structure": "Talk to {giver} → Find {item} → Return to {giver}",
        "difficulty_modifiers": {
            "trivial": {"item_distance": "nearby", "obstacles": "none"},
            "easy": {"item_distance": "medium", "obstacles": "minor"},
            "normal": {"item_distance": "far", "obstacles": "moderate"},
            "hard": {"item_distance": "very_far", "obstacles": "significant"}
        }
    },
    "combat": {
        "structure": "Talk to {giver} → Find enemies → Defeat {count} {enemy_type} → Return",
        "difficulty_modifiers": {
            "trivial": {"count": 3, "enemy_strength": "weak"},
            "easy": {"count": 5, "enemy_strength": "normal"},
            "normal": {"count": 8, "enemy_strength": "strong"},
            "hard": {"count": 12, "enemy_strength": "elite"}
        }
    },
    "investigation": {
        "structure": "Talk to {giver} → Gather clues ({clue_count}) → Solve mystery → Report back",
        "difficulty_modifiers": {
            "trivial": {"clue_count": 2, "complexity": "obvious"},
            "easy": {"clue_count": 3, "complexity": "simple"},
            "normal": {"clue_count": 4, "complexity": "moderate"},
            "hard": {"clue_count": 5, "complexity": "complex"}
        }
    },
    "escort": {
        "structure": "Talk to {giver} → Escort {target} to {destination} → Protect from {threat}",
        "difficulty_modifiers": {
            "trivial": {"distance": "short", "threat": "none"},
            "easy": {"distance": "medium", "threat": "minor"},
            "normal": {"distance": "long", "threat": "moderate"},
            "hard": {"distance": "very_long", "threat": "major"}
        }
    }
}


# ============================================================================
# Emotion-Based Quest Scaling
# ============================================================================

class EmotionQuestScaler:
    """
    Scales quest difficulty and tone based on NPC emotional state.

    Rules:
    - High valence (happy) → Easier quests, better rewards
    - Low valence (sad/angry) → Harder quests, desperation
    - High trust → More rewarding, less risky
    - Low trust → Test quests, dangerous tasks
    - High arousal → Urgent quests, time limits
    - Low arousal → Calm quests, no rush
    """

    @staticmethod
    def scale_difficulty(
        base_difficulty: QuestDifficulty,
        emotional_state: EmotionalState,
        player_level: int
    ) -> QuestDifficulty:
        """
        Adjust quest difficulty based on emotion and player level.

        Args:
            base_difficulty: Base difficulty level
            emotional_state: NPC emotional state
            player_level: Player level

        Returns:
            Adjusted difficulty
        """
        # Convert difficulty to numeric scale
        difficulty_scale = {
            QuestDifficulty.TRIVIAL: 1,
            QuestDifficulty.EASY: 2,
            QuestDifficulty.NORMAL: 3,
            QuestDifficulty.HARD: 4,
            QuestDifficulty.EPIC: 5
        }

        current_level = difficulty_scale[base_difficulty]

        # Modifiers based on emotion
        valence = emotional_state.valence
        trust = emotional_state.trust
        arousal = emotional_state.arousal

        # Low trust = harder quests (test the player)
        if trust < 0.3:
            current_level += 1
        elif trust > 0.7:
            current_level -= 1

        # Negative emotion = harder quests (desperation)
        if valence < -0.5:
            current_level += 1
        elif valence > 0.5:
            current_level -= 1

        # High arousal = more challenging (urgent situations)
        if arousal > 0.7:
            current_level += 0.5

        # Clamp to valid range
        current_level = max(1, min(5, round(current_level)))

        # Convert back to difficulty
        level_to_difficulty = {
            1: QuestDifficulty.TRIVIAL,
            2: QuestDifficulty.EASY,
            3: QuestDifficulty.NORMAL,
            4: QuestDifficulty.HARD,
            5: QuestDifficulty.EPIC
        }

        return level_to_difficulty[current_level]

    @staticmethod
    def get_emotional_tone(emotional_state: EmotionalState) -> Dict[str, Any]:
        """
        Get quest tone parameters based on emotion.

        Returns:
            Dictionary with tone guidance for quest generation
        """
        valence = emotional_state.valence
        arousal = emotional_state.arousal
        trust = emotional_state.trust

        # Determine tone
        if valence > 0.5:
            tone = "friendly"
            urgency = "low"
        elif valence < -0.5 and arousal > 0.6:
            tone = "desperate"
            urgency = "high"
        elif valence < -0.5:
            tone = "sad"
            urgency = "moderate"
        else:
            tone = "neutral"
            urgency = "moderate"

        # Trust affects how much the NPC asks
        if trust < 0.3:
            request_type = "test_quest"  # Prove yourself
        elif trust < 0.6:
            request_type = "standard_quest"
        else:
            request_type = "important_quest"  # Trusts you with critical tasks

        return {
            "tone": tone,
            "urgency": urgency,
            "request_type": request_type,
            "time_pressure": arousal > 0.7,
            "high_reward": trust > 0.7 and valence > 0.3
        }


# ============================================================================
# Dynamic Quest Generator Integration
# ============================================================================

class DynamicQuestManager:
    """
    Manages dynamic quest generation using Elle's QuestGenerator.

    Combines:
    - Handcrafted quest templates
    - LLM-generated quests
    - Emotion-based scaling
    """

    def __init__(self, llm_provider: str = "dummy", llm_model: Optional[str] = None):
        """
        Initialize dynamic quest manager.

        Args:
            llm_provider: LLM provider (dummy, ollama, anthropic, openai)
            llm_model: Model name (optional, uses provider default)
        """
        self.llm_client = create_llm_client(provider=llm_provider, model_name=llm_model)
        self.quest_generator = QuestGenerator(self.llm_client)
        self.emotion_scaler = EmotionQuestScaler()

    async def generate_quest_for_npc(
        self,
        npc_id: str,
        npc_name: str,
        npc_role: str,
        emotional_state: EmotionalState,
        player_level: int = 1,
        player_reputation: Optional[str] = None,
        world_state: Optional[Dict[str, Any]] = None
    ) -> Quest:
        """
        Generate a dynamic quest for an NPC based on their emotional state.

        Args:
            npc_id: NPC identifier
            npc_name: NPC display name
            npc_role: NPC role (merchant, guard, etc.)
            emotional_state: NPC's current emotional state
            player_level: Player level
            player_reputation: Player reputation (optional)
            world_state: Current world state (optional)

        Returns:
            Generated quest
        """
        world_state = world_state or {}

        # Get emotional tone
        tone_params = self.emotion_scaler.get_emotional_tone(emotional_state)

        # Generate quest using Elle's QuestGenerator
        quest = await self.quest_generator.generate_quest(
            npc_id=npc_id,
            npc_name=npc_name,
            npc_role=npc_role,
            emotional_state=emotional_state,
            player_level=player_level,
            player_reputation=player_reputation,
            world_tension=world_state.get("tension", "calm"),
            world_time=world_state.get("time", "day"),
            recent_events=world_state.get("recent_events")
        )

        return quest

    async def generate_quest_from_template(
        self,
        template_type: str,
        npc_id: str,
        npc_name: str,
        emotional_state: EmotionalState,
        player_level: int = 1,
        **template_params
    ) -> Optional[Quest]:
        """
        Generate quest from template with emotion-based scaling.

        Args:
            template_type: Template type (fetch, combat, investigation, escort)
            npc_id: NPC identifier
            npc_name: NPC name
            emotional_state: NPC emotional state
            player_level: Player level
            **template_params: Template-specific parameters

        Returns:
            Generated quest or None if template not found
        """
        if template_type not in QUEST_TEMPLATES:
            return None

        template = QUEST_TEMPLATES[template_type]

        # Scale difficulty based on emotion
        base_difficulty = QuestDifficulty.NORMAL
        scaled_difficulty = self.emotion_scaler.scale_difficulty(
            base_difficulty,
            emotional_state,
            player_level
        )

        # Get difficulty modifiers
        difficulty_key = scaled_difficulty.value
        modifiers = template["difficulty_modifiers"].get(difficulty_key, {})

        # This is a simplified template-based approach
        # In production, you'd expand this with full quest generation
        print(f"Template: {template_type}")
        print(f"Difficulty: {scaled_difficulty.value}")
        print(f"Modifiers: {modifiers}")

        # For full implementation, generate quest objectives based on template
        # and call quest_generator with structured parameters

        return None  # Return actual quest in production


# ============================================================================
# Example Usage
# ============================================================================

async def demo_dynamic_quests():
    """Demonstrate dynamic quest generation with different emotions."""
    print("=== Dynamic Quest Generation Demo ===\n")

    # Create quest manager
    manager = DynamicQuestManager(llm_provider="dummy")

    # Example 1: Happy merchant (easy, rewarding quest)
    print("1. Happy Merchant Quest:")
    happy_emotion = EmotionalState(
        valence=0.7,
        arousal=0.4,
        trust=0.8
    )

    quest1 = await manager.generate_quest_for_npc(
        npc_id="merchant",
        npc_name="Marcus the Merchant",
        npc_role="merchant",
        emotional_state=happy_emotion,
        player_level=2,
        player_reputation="good",
        world_state={"tension": "calm", "time": "day"}
    )

    print(f"  Title: {quest1.title}")
    print(f"  Difficulty: {quest1.difficulty.value}")
    print(f"  Description: {quest1.description}")
    print(f"  Objectives: {len(quest1.objectives)}")
    if quest1.reward:
        print(f"  Reward XP: {quest1.reward.xp}, Gold: {quest1.reward.gold}")
    print()

    # Example 2: Desperate guard (hard, urgent quest)
    print("2. Desperate Guard Quest:")
    desperate_emotion = EmotionalState(
        valence=-0.6,
        arousal=0.8,
        trust=0.4
    )

    quest2 = await manager.generate_quest_for_npc(
        npc_id="guard",
        npc_name="Captain Sarah",
        npc_role="guard",
        emotional_state=desperate_emotion,
        player_level=3,
        player_reputation="neutral",
        world_state={"tension": "high", "time": "night", "recent_events": "Bandit attacks increasing"}
    )

    print(f"  Title: {quest2.title}")
    print(f"  Difficulty: {quest2.difficulty.value}")
    print(f"  Description: {quest2.description}")
    print(f"  Time Limit: {quest2.time_limit_seconds}s" if quest2.time_limit_seconds else "  Time Limit: None")
    print(f"  Objectives: {len(quest2.objectives)}")
    print()

    # Example 3: Sad child (easy, emotional quest)
    print("3. Sad Child Quest:")
    sad_emotion = EmotionalState(
        valence=-0.5,
        arousal=0.3,
        trust=0.6
    )

    quest3 = await manager.generate_quest_for_npc(
        npc_id="child",
        npc_name="Lily",
        npc_role="child",
        emotional_state=sad_emotion,
        player_level=1,
        player_reputation="friendly",
        world_state={"tension": "calm", "time": "day"}
    )

    print(f"  Title: {quest3.title}")
    print(f"  Difficulty: {quest3.difficulty.value}")
    print(f"  Description: {quest3.description}")
    print(f"  Emotion at creation: {quest3.emotion_at_creation}")
    print()

    # Example 4: Emotion-based difficulty scaling
    print("4. Difficulty Scaling Examples:")
    scaler = EmotionQuestScaler()

    emotions = [
        ("Happy (high trust)", EmotionalState(valence=0.8, arousal=0.4, trust=0.9)),
        ("Angry (low trust)", EmotionalState(valence=-0.6, arousal=0.8, trust=0.2)),
        ("Neutral", EmotionalState(valence=0.0, arousal=0.5, trust=0.5)),
    ]

    for label, emotion in emotions:
        scaled = scaler.scale_difficulty(QuestDifficulty.NORMAL, emotion, player_level=2)
        tone = scaler.get_emotional_tone(emotion)
        print(f"  {label}:")
        print(f"    Difficulty: NORMAL → {scaled.value}")
        print(f"    Tone: {tone['tone']}, Urgency: {tone['urgency']}")
        print()


# ============================================================================
# Integration Example with Handcrafted Quests
# ============================================================================

def integrate_dynamic_with_static():
    """
    Example of how to integrate dynamic quests with handcrafted quests.

    Strategy:
    1. Use handcrafted quests for main storyline (rat_problem, lost_cat, etc.)
    2. Use dynamic quests for:
       - Side quests
       - Repeatable daily quests
       - Random encounters
       - NPC-specific reactions to player actions
    """
    print("=== Dynamic + Static Quest Integration ===\n")

    print("Handcrafted Quests (Main Story):")
    print("  - Rat Problem (always the same, tested, balanced)")
    print("  - Lost Cat (scripted, predictable)")
    print("  - Lost Shipment (key story beat)")
    print("  - Bandit Trouble (climax)")
    print("  - Hidden Path (finale)")
    print()

    print("Dynamic Quests (Side Content):")
    print("  - Daily bounties from guard (combat quests)")
    print("  - Merchant deliveries (fetch quests)")
    print("  - Innkeeper requests (investigation quests)")
    print("  - Random traveler tasks (escort quests)")
    print()

    print("Benefits:")
    print("  ✓ Main story is consistent and well-crafted")
    print("  ✓ Side content adapts to NPC emotions")
    print("  ✓ Infinite replayability through procedural quests")
    print("  ✓ NPCs feel alive (different requests based on mood)")
    print()


if __name__ == "__main__":
    # Run demo
    print("Starting dynamic quest generation demo...\n")

    # Run async demo
    asyncio.run(demo_dynamic_quests())

    print("\n" + "=" * 60 + "\n")

    # Show integration strategy
    integrate_dynamic_with_static()
