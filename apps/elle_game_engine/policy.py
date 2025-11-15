"""
Game policy engine for Elle.

Responsibilities:
1. Build prompts from GameStateSnapshot + PlayerIntent
2. Call LLM client
3. Parse JSON response into ElleGameAction
4. Handle errors gracefully
"""

import json
import os
from pathlib import Path
from typing import Optional

from .models import (
    GameStateSnapshot,
    PlayerIntent,
    ElleGameAction,
    ActionMode,
    DialogueLine,
    WorldChange,
)
from .llm_client import BaseLLMClient


class GamePolicy:
    """
    Policy engine for game narrative decisions.

    Takes game state + player intent, returns structured action.
    Stateless - all decisions are pure functions of inputs.
    """

    def __init__(self, llm_client: BaseLLMClient, *, core_prompt_path: Optional[str] = None):
        """
        Initialize policy engine.

        Args:
            llm_client: LLM client to use for completions
            core_prompt_path: Optional path to core prompt file (defaults to core_prompt.txt)
        """
        self.llm_client = llm_client

        # Load core prompt
        if core_prompt_path is None:
            # Default: load from same directory
            module_dir = Path(__file__).parent
            core_prompt_path = module_dir / "core_prompt.txt"

        with open(core_prompt_path, "r") as f:
            self.core_prompt = f.read()

    async def decide(
        self,
        game_state: GameStateSnapshot,
        player_intent: PlayerIntent,
    ) -> ElleGameAction:
        """
        Make a narrative decision based on game state and player intent.

        Args:
            game_state: Current game state snapshot
            player_intent: What the player is doing

        Returns:
            Structured action for the game to execute

        Raises:
            PolicyError: If decision-making fails
        """
        try:
            # 1. Build prompt
            user_prompt = self._build_user_prompt(game_state, player_intent)

            # 2. Call LLM
            response_text = await self.llm_client.complete(
                prompt=user_prompt,
                system_prompt=self.core_prompt,
                max_tokens=500,
                temperature=0.7,
            )

            # 3. Parse response
            action = self._parse_response(response_text)

            return action

        except Exception as e:
            raise PolicyError(f"Failed to make decision: {e}") from e

    def _build_user_prompt(
        self,
        game_state: GameStateSnapshot,
        player_intent: PlayerIntent,
    ) -> str:
        """
        Build the user prompt from game state and intent.

        This serializes the state into a clear, structured format
        that the LLM can easily parse.
        """
        # Serialize NPCs
        npcs_text = ""
        for npc in game_state.npcs:
            npcs_text += f"  - {npc.name} ({npc.id})\n"
            npcs_text += f"    Role: {npc.role}\n"
            npcs_text += f"    Mood: {npc.mood or 'neutral'}\n"
            npcs_text += f"    Location: {npc.location}\n"
            if npc.flags:
                npcs_text += f"    Flags: {npc.flags}\n"

        # Serialize player
        player_text = f"""  Name: {game_state.player.name}
  Location: {game_state.player.location}
  Quest Stage: {game_state.player.quest_stage or 'none'}
  Reputation: {game_state.player.reputation or 'neutral'}"""

        if game_state.player.traits:
            player_text += f"\n  Traits: {game_state.player.traits}"
        if game_state.player.inventory_tags:
            player_text += f"\n  Inventory: {', '.join(game_state.player.inventory_tags)}"

        # Serialize world
        world_text = f"""  Time of Day: {game_state.world.time_of_day}
  Weather: {game_state.world.weather or 'clear'}
  Tension Level: {game_state.world.tension_level or 'calm'}"""

        # Serialize intent
        intent_text = f"""  Type: {player_intent.type.value}"""
        if player_intent.target_npc_id:
            intent_text += f"\n  Target NPC: {player_intent.target_npc_id}"
        if player_intent.raw_input:
            intent_text += f"\n  Player Input: {player_intent.raw_input}"

        # Build complete prompt
        prompt = f"""GAME STATE:

Scene: {game_state.scene_id}

NPCs Present:
{npcs_text if npcs_text else "  (none)"}

Player:
{player_text}

World:
{world_text}

Tags: {', '.join(game_state.tags) if game_state.tags else '(none)'}

PLAYER INTENT:
{intent_text}

Please respond with a JSON action following the format specified in your instructions.
"""
        return prompt

    def _parse_response(self, response_text: str) -> ElleGameAction:
        """
        Parse LLM response into structured ElleGameAction.

        Handles:
        - JSON extraction (in case LLM wraps in markdown)
        - Validation
        - Error handling
        """
        # Clean response (remove markdown code fences if present)
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
            raise PolicyError(f"Invalid JSON response: {e}\nResponse: {response_text}") from e

        # Parse mode
        try:
            mode = ActionMode(data["mode"])
        except (KeyError, ValueError) as e:
            raise PolicyError(f"Invalid or missing mode: {e}") from e

        # Parse priority
        priority = data.get("priority", "medium")
        if priority not in ["low", "medium", "high"]:
            raise PolicyError(f"Invalid priority: {priority}")

        # Parse dialogue
        dialogue = []
        for d in data.get("dialogue", []):
            dialogue.append(DialogueLine(
                npc_id=d["npc_id"],
                text=d["text"],
                tone=d.get("tone"),
            ))

        # Parse world reaction
        world_reaction = None
        if data.get("world_reaction"):
            wr = data["world_reaction"]
            world_reaction = WorldChange(
                description=wr["description"],
                flag_changes=wr.get("flag_changes", {}),
            )

        # Build action
        action = ElleGameAction(
            mode=mode,
            priority=priority,
            dialogue=dialogue,
            hint_text=data.get("hint_text"),
            world_reaction=world_reaction,
            debug_notes=data.get("debug_notes"),
        )

        return action


class PolicyError(Exception):
    """Error during policy execution."""
    pass
