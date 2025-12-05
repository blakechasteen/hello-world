"""
Quest system for Elle Tavern Demo.

Implements 5 interconnected quests with emotion-based progression:
1. Rat Problem (Easy) - Innkeeper
2. Lost Cat (Easy) - Child
3. Lost Shipment (Medium) - Merchant
4. Bandit Trouble (Hard) - Guard
5. Hidden Path (Epic) - Mysterious Stranger

Each quest requires specific emotional states and unlocks others.

Created: 2025-11-16
"""

import time
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from copy import deepcopy

# Import Elle Game Engine quest models
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../apps'))

from elle_game_engine.quest import (
    Quest,
    QuestObjective,
    QuestReward,
    QuestStatus,
    QuestDifficulty
)
from elle_game_engine.emotion import EmotionalState


# ============================================================================
# Quest Definitions
# ============================================================================

QUESTS = {
    "rat_problem": {
        "id": "rat_problem",
        "title": "The Cellar Rat Problem",
        "giver": "innkeeper",
        "description": "Bob the innkeeper needs help clearing rats from the tavern cellar. They're eating all the food stores!",
        "difficulty": QuestDifficulty.TRIVIAL,
        "objectives": [
            {
                "id": "talk_to_bob",
                "description": "Talk to Bob about the rats",
                "completed": False,
                "optional": False,
                "target": 1
            },
            {
                "id": "clear_rats",
                "description": "Clear 5 rats from cellar",
                "completed": False,
                "optional": False,
                "progress": 0,
                "target": 5
            },
            {
                "id": "report_to_bob",
                "description": "Report back to Bob",
                "completed": False,
                "optional": False,
                "target": 1
            }
        ],
        "reward": {
            "description": "Bob's gratitude and a small reward",
            "xp": 50,
            "gold": 10,
            "items": [],
            "reputation_changes": {"innkeeper": 20},
            "flags": {"rats_cleared": True}
        },
        "unlocks": ["lost_shipment"],  # Merchant trust increases
        "emotional_requirements": {
            "innkeeper": {"trust": 0.3, "valence": -0.2}  # Innkeeper is worried
        },
        "emotion_changes": {
            "innkeeper": {"valence": +0.3, "trust": +0.2}
        },
        "required_level": 1,
        "time_limit": None,
        "world_context": {
            "location": "rusty_tankard_inn",
            "urgency": "moderate"
        }
    },

    "lost_cat": {
        "id": "lost_cat",
        "title": "Lily's Lost Cat",
        "giver": "child",
        "description": "Little Lily's cat Whiskers has gone missing! She's very upset and needs help finding her beloved pet.",
        "difficulty": QuestDifficulty.EASY,
        "objectives": [
            {
                "id": "talk_to_lily",
                "description": "Talk to Lily about Whiskers",
                "completed": False,
                "optional": False,
                "target": 1
            },
            {
                "id": "search_tavern",
                "description": "Search the tavern",
                "completed": False,
                "optional": False,
                "target": 1
            },
            {
                "id": "search_market",
                "description": "Search the market square",
                "completed": False,
                "optional": False,
                "target": 1
            },
            {
                "id": "find_cat",
                "description": "Find Whiskers (hint: check behind the barrels)",
                "completed": False,
                "optional": False,
                "target": 1
            },
            {
                "id": "return_cat",
                "description": "Return Whiskers to Lily",
                "completed": False,
                "optional": False,
                "target": 1
            }
        ],
        "reward": {
            "description": "Lily's happiness and a lucky charm",
            "xp": 75,
            "gold": 5,
            "items": ["lucky_charm"],
            "reputation_changes": {"child": 50},
            "flags": {"cat_found": True}
        },
        "unlocks": ["bandit_trouble"],  # Child tells you about guard's problem
        "emotional_requirements": {
            "child": {"trust": 0.2, "valence": -0.4}  # Child is sad
        },
        "emotion_changes": {
            "child": {"valence": +0.5, "trust": +0.3}
        },
        "required_level": 1,
        "time_limit": None,
        "world_context": {
            "location": "town_square",
            "urgency": "low"
        }
    },

    "lost_shipment": {
        "id": "lost_shipment",
        "title": "Marcus's Lost Shipment",
        "giver": "merchant",
        "description": "Marcus the merchant had a valuable shipment stolen by bandits. He needs someone brave to recover it.",
        "difficulty": QuestDifficulty.NORMAL,
        "objectives": [
            {
                "id": "talk_to_marcus",
                "description": "Talk to Marcus about the stolen goods",
                "completed": False,
                "optional": False,
                "target": 1
            },
            {
                "id": "talk_to_guard",
                "description": "Ask Captain Sarah about the bandits",
                "completed": False,
                "optional": False,
                "target": 1
            },
            {
                "id": "find_bandits",
                "description": "Find the bandit camp (follow forest path from town square)",
                "completed": False,
                "optional": False,
                "target": 1
            },
            {
                "id": "recover_goods",
                "description": "Recover the stolen goods",
                "completed": False,
                "optional": False,
                "target": 1
            },
            {
                "id": "return_goods",
                "description": "Return goods to Marcus",
                "completed": False,
                "optional": False,
                "target": 1
            }
        ],
        "reward": {
            "description": "Marcus's gratitude and valuable items",
            "xp": 150,
            "gold": 50,
            "items": ["silver_ring"],
            "reputation_changes": {"merchant": 30},
            "flags": {"shipment_recovered": True}
        },
        "prerequisites": ["rat_problem"],  # Must help innkeeper first
        "unlocks": ["bandit_trouble"],
        "emotional_requirements": {
            "merchant": {"trust": 0.4, "valence": -0.3},  # Merchant is worried
            "innkeeper": {"trust": 0.5}  # Innkeeper must trust you
        },
        "emotion_changes": {
            "merchant": {"valence": +0.4, "trust": +0.4}
        },
        "required_level": 2,
        "time_limit": None,
        "world_context": {
            "location": "market_square",
            "urgency": "moderate"
        }
    },

    "bandit_trouble": {
        "id": "bandit_trouble",
        "title": "Clear the Bandit Camp",
        "giver": "guard",
        "description": "Captain Sarah needs help permanently clearing the bandit camp that's been terrorizing travelers.",
        "difficulty": QuestDifficulty.HARD,
        "objectives": [
            {
                "id": "talk_to_sarah",
                "description": "Talk to Captain Sarah",
                "completed": False,
                "optional": False,
                "target": 1
            },
            {
                "id": "scout_camp",
                "description": "Scout the bandit camp for weaknesses",
                "completed": False,
                "optional": False,
                "target": 1
            },
            {
                "id": "defeat_bandits",
                "description": "Defeat 8 bandits",
                "completed": False,
                "optional": False,
                "progress": 0,
                "target": 8
            },
            {
                "id": "defeat_leader",
                "description": "Defeat the bandit leader",
                "completed": False,
                "optional": False,
                "target": 1
            },
            {
                "id": "report_to_sarah",
                "description": "Report back to Captain Sarah",
                "completed": False,
                "optional": False,
                "target": 1
            }
        ],
        "reward": {
            "description": "Captain Sarah's respect and official recognition",
            "xp": 300,
            "gold": 100,
            "items": ["guard_badge", "iron_sword"],
            "reputation_changes": {"guard": 50},
            "flags": {"bandits_cleared": True, "hero_status": True}
        },
        "prerequisites": ["lost_shipment", "lost_cat"],  # Multiple paths lead here
        "unlocks": ["hidden_path"],
        "emotional_requirements": {
            "guard": {"trust": 0.5, "valence": -0.2},  # Guard needs to trust you
            "merchant": {"trust": 0.6},  # Merchant vouches for you
            "child": {"trust": 0.4}  # Child mentions you to guard
        },
        "emotion_changes": {
            "guard": {"valence": +0.5, "trust": +0.5}
        },
        "required_level": 3,
        "time_limit": None,
        "world_context": {
            "location": "town_square",
            "urgency": "high"
        }
    },

    "hidden_path": {
        "id": "hidden_path",
        "title": "Secrets of the Forest",
        "giver": "mysterious_stranger",
        "description": "A mysterious stranger reveals there are ancient secrets hidden in the forest, accessible only to those who have proven themselves.",
        "difficulty": QuestDifficulty.EPIC,
        "objectives": [
            {
                "id": "talk_to_stranger",
                "description": "Speak with the mysterious stranger",
                "completed": False,
                "optional": False,
                "target": 1
            },
            {
                "id": "gather_trust",
                "description": "Gather testimonials from townspeople",
                "completed": False,
                "optional": False,
                "progress": 0,
                "target": 3  # Need 3 NPCs to vouch for you
            },
            {
                "id": "unlock_forest",
                "description": "Unlock the hidden forest path",
                "completed": False,
                "optional": False,
                "target": 1
            },
            {
                "id": "explore_forest",
                "description": "Explore the ancient forest area",
                "completed": False,
                "optional": False,
                "target": 1
            },
            {
                "id": "solve_puzzle",
                "description": "Solve the ancient puzzle",
                "completed": False,
                "optional": False,
                "target": 1
            },
            {
                "id": "discover_secret",
                "description": "Discover the ancient secret",
                "completed": False,
                "optional": False,
                "target": 1
            }
        ],
        "reward": {
            "description": "Ancient knowledge and powerful artifacts",
            "xp": 500,
            "gold": 200,
            "items": ["ancient_amulet", "mystery_scroll"],
            "reputation_changes": {"mysterious_stranger": 100},
            "flags": {"forest_unlocked": True, "ancient_secret_found": True}
        },
        "prerequisites": ["bandit_trouble"],
        "unlocks": [],  # Final quest
        "emotional_requirements": {
            "guard": {"trust": 0.7},
            "innkeeper": {"trust": 0.6},
            "merchant": {"trust": 0.5},
            "mysterious_stranger": {"trust": 0.3}
        },
        "emotion_changes": {
            "mysterious_stranger": {"valence": +0.3, "trust": +0.6}
        },
        "required_level": 5,
        "time_limit": 3600,  # 1 hour time limit for tension
        "world_context": {
            "location": "forest_path",
            "urgency": "critical"
        }
    }
}


# ============================================================================
# Quest Manager
# ============================================================================

@dataclass
class GameState:
    """Simplified game state for quest management."""
    player_level: int = 1
    completed_quests: Set[str] = field(default_factory=set)
    active_quests: Set[str] = field(default_factory=set)
    world_flags: Dict[str, bool] = field(default_factory=dict)
    npc_emotions: Dict[str, Dict[str, float]] = field(default_factory=dict)
    npc_reputation: Dict[str, int] = field(default_factory=dict)
    player_inventory: List[str] = field(default_factory=list)


class QuestManager:
    """
    Manages quest state, availability, and progression.

    Handles:
    - Quest availability based on prerequisites and emotions
    - Quest progression and objective tracking
    - Reward distribution
    - Quest unlocking and dependencies
    """

    def __init__(self, game_state: Optional[GameState] = None):
        """
        Initialize quest manager.

        Args:
            game_state: Current game state (creates new if None)
        """
        self.game_state = game_state or GameState()
        self.quest_instances: Dict[str, Quest] = {}

    def get_available_quests(self, npc_id: Optional[str] = None) -> List[Dict]:
        """
        Get quests available to the player.

        Args:
            npc_id: If specified, only return quests from this NPC

        Returns:
            List of available quest data
        """
        available = []

        for quest_id, quest_data in QUESTS.items():
            # Skip if already active or completed
            if quest_id in self.game_state.active_quests:
                continue
            if quest_id in self.game_state.completed_quests:
                continue

            # Filter by NPC if specified
            if npc_id and quest_data["giver"] != npc_id:
                continue

            # Check if quest is available
            if self._is_quest_available(quest_id, quest_data):
                available.append(quest_data)

        return available

    def _is_quest_available(self, quest_id: str, quest_data: Dict) -> bool:
        """
        Check if quest is available based on prerequisites and emotions.

        Args:
            quest_id: Quest ID
            quest_data: Quest data dictionary

        Returns:
            True if quest is available
        """
        # Check level requirement
        if self.game_state.player_level < quest_data.get("required_level", 1):
            return False

        # Check prerequisites (completed quests)
        prerequisites = quest_data.get("prerequisites", [])
        for prereq in prerequisites:
            if prereq not in self.game_state.completed_quests:
                return False

        # Check emotional requirements
        emotional_reqs = quest_data.get("emotional_requirements", {})
        for npc_id, requirements in emotional_reqs.items():
            npc_emotion = self.game_state.npc_emotions.get(npc_id, {})

            # Check trust requirement
            if "trust" in requirements:
                npc_trust = npc_emotion.get("trust", 0.5)
                if npc_trust < requirements["trust"]:
                    return False

            # Check valence requirement
            if "valence" in requirements:
                npc_valence = npc_emotion.get("valence", 0.0)
                if npc_valence < requirements["valence"]:
                    return False

        # Check world flags
        required_flags = quest_data.get("required_flags", {})
        for flag, value in required_flags.items():
            if self.game_state.world_flags.get(flag) != value:
                return False

        return True

    def start_quest(self, quest_id: str) -> Optional[Quest]:
        """
        Start a quest (accept it).

        Args:
            quest_id: ID of quest to start

        Returns:
            Quest instance if successfully started, None otherwise
        """
        if quest_id not in QUESTS:
            return None

        quest_data = QUESTS[quest_id]

        # Check if available
        if not self._is_quest_available(quest_id, quest_data):
            return None

        # Create Quest instance
        quest = self._create_quest_instance(quest_id, quest_data)

        # Accept the quest
        quest.accept()

        # Track quest
        self.quest_instances[quest_id] = quest
        self.game_state.active_quests.add(quest_id)

        return quest

    def _create_quest_instance(self, quest_id: str, quest_data: Dict) -> Quest:
        """Create Quest instance from quest data."""
        # Create objectives
        objectives = []
        for obj_data in quest_data["objectives"]:
            objectives.append(QuestObjective(
                id=obj_data["id"],
                description=obj_data["description"],
                completed=obj_data.get("completed", False),
                optional=obj_data.get("optional", False),
                progress=obj_data.get("progress", 0),
                target=obj_data.get("target", 1)
            ))

        # Create reward
        reward_data = quest_data.get("reward", {})
        reward = QuestReward(
            description=reward_data.get("description", ""),
            xp=reward_data.get("xp", 0),
            gold=reward_data.get("gold", 0),
            items=reward_data.get("items", []),
            reputation_changes=reward_data.get("reputation_changes", {}),
            flags=reward_data.get("flags", {})
        )

        # Create quest
        quest = Quest(
            id=quest_id,
            title=quest_data["title"],
            description=quest_data["description"],
            giver_npc_id=quest_data["giver"],
            difficulty=quest_data["difficulty"],
            objectives=objectives,
            reward=reward,
            required_level=quest_data.get("required_level", 1),
            time_limit_seconds=quest_data.get("time_limit"),
            world_context=quest_data.get("world_context", {})
        )

        return quest

    def update_quest_objective(
        self,
        quest_id: str,
        objective_id: str,
        progress: Optional[int] = None,
        completed: bool = False
    ) -> bool:
        """
        Update quest objective progress.

        Args:
            quest_id: Quest ID
            objective_id: Objective ID
            progress: New progress value (optional)
            completed: Mark as completed (optional)

        Returns:
            True if successfully updated
        """
        if quest_id not in self.quest_instances:
            return False

        quest = self.quest_instances[quest_id]

        # Find objective
        for obj in quest.objectives:
            if obj.id == objective_id:
                if progress is not None:
                    obj.progress = min(progress, obj.target)
                if completed:
                    obj.completed = True

                # Auto-complete if target reached
                if obj.progress >= obj.target:
                    obj.completed = True

                return True

        return False

    def complete_quest(self, quest_id: str) -> bool:
        """
        Complete a quest and apply rewards.

        Args:
            quest_id: Quest ID to complete

        Returns:
            True if successfully completed
        """
        if quest_id not in self.quest_instances:
            return False

        quest = self.quest_instances[quest_id]
        quest_data = QUESTS[quest_id]

        try:
            # Complete quest (validates all objectives)
            quest.complete()

            # Apply rewards
            if quest.reward:
                # Apply reputation changes
                for npc_id, rep_change in quest.reward.reputation_changes.items():
                    current_rep = self.game_state.npc_reputation.get(npc_id, 0)
                    self.game_state.npc_reputation[npc_id] = current_rep + rep_change

                # Apply world flags
                for flag, value in quest.reward.flags.items():
                    self.game_state.world_flags[flag] = value

                # Add items to inventory
                self.game_state.player_inventory.extend(quest.reward.items)

            # Apply emotion changes
            emotion_changes = quest_data.get("emotion_changes", {})
            for npc_id, changes in emotion_changes.items():
                if npc_id not in self.game_state.npc_emotions:
                    self.game_state.npc_emotions[npc_id] = {
                        "valence": 0.0,
                        "arousal": 0.4,
                        "dominance": 0.5,
                        "trust": 0.5
                    }

                for dim, delta in changes.items():
                    current = self.game_state.npc_emotions[npc_id].get(dim, 0.0)
                    self.game_state.npc_emotions[npc_id][dim] = max(-1.0, min(1.0, current + delta))

            # Update quest tracking
            self.game_state.active_quests.discard(quest_id)
            self.game_state.completed_quests.add(quest_id)

            # Unlock dependent quests (mark as available)
            unlocks = quest_data.get("unlocks", [])
            # (Unlocked quests will show up in get_available_quests automatically)

            return True

        except ValueError as e:
            print(f"Failed to complete quest {quest_id}: {e}")
            return False

    def get_active_quests(self) -> List[Quest]:
        """Get all active quest instances."""
        return [
            self.quest_instances[quest_id]
            for quest_id in self.game_state.active_quests
            if quest_id in self.quest_instances
        ]

    def get_quest_status(self, quest_id: str) -> Optional[str]:
        """
        Get quest status.

        Returns:
            "available", "active", "completed", or None if quest doesn't exist
        """
        if quest_id in self.game_state.completed_quests:
            return "completed"
        if quest_id in self.game_state.active_quests:
            return "active"
        if quest_id in QUESTS:
            quest_data = QUESTS[quest_id]
            if self._is_quest_available(quest_id, quest_data):
                return "available"
        return None

    def get_quest_chain_progress(self) -> Dict[str, Any]:
        """
        Get overall quest chain progress.

        Returns:
            Dictionary with progress statistics
        """
        total_quests = len(QUESTS)
        completed = len(self.game_state.completed_quests)
        active = len(self.game_state.active_quests)
        available = len(self.get_available_quests())

        return {
            "total_quests": total_quests,
            "completed": completed,
            "active": active,
            "available": available,
            "completion_percentage": (completed / total_quests) * 100 if total_quests > 0 else 0,
            "locked": total_quests - completed - active - available
        }
