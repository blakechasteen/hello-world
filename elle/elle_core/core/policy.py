"""
Elle Core - Policy

The core decision-making engine that determines Elle's actions.
"""
from typing import Optional
import logging

from .llm_client import LLMClient
from .prompt import compose_prompt
from ..models import (
    SceneSnapshot,
    UserState,
    UserIntent,
    MemorySnapshot,
    ElleAction,
    ActionMode,
    Priority,
    Visual,
    Followup,
    VisualDirective,
)


logger = logging.getLogger(__name__)


class EllePolicy:
    """
    Elle's decision-making policy.

    Takes a situation (scene + user state + intent + memory) and decides
    what action Elle should take (or not take).
    """

    def __init__(
        self,
        llm_client: LLMClient,
        temperature: float = 0.7,
        max_tokens: int = 1500,
    ):
        """
        Initialize the policy.

        Args:
            llm_client: The LLM client to use for decisions
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens for LLM response
        """
        self.llm_client = llm_client
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def decide(
        self,
        scene: SceneSnapshot,
        user_state: UserState,
        user_intent: Optional[UserIntent] = None,
        memory: Optional[MemorySnapshot] = None,
    ) -> ElleAction:
        """
        Decide what action Elle should take.

        Args:
            scene: Current scene snapshot
            user_state: Current user state
            user_intent: Optional user intent
            memory: Optional memory context

        Returns:
            ElleAction describing what Elle should do

        Raises:
            ValueError: If LLM returns invalid response
        """
        # Compose the prompt
        system_prompt, user_prompt = compose_prompt(
            scene=scene,
            user_state=user_state,
            user_intent=user_intent,
            memory=memory,
        )

        logger.debug(f"Calling LLM with prompt (system: {len(system_prompt)} chars, user: {len(user_prompt)} chars)")

        # Get LLM decision
        try:
            response_json = await self.llm_client.complete_json(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            # Return safe default (idle mode, no intervention)
            return ElleAction(
                mode=ActionMode.IDLE,
                priority=Priority.LOW,
                reasoning=f"LLM call failed: {e}. Defaulting to idle.",
            )

        # Parse JSON into ElleAction
        try:
            action = self._parse_action(response_json)
            logger.info(f"Decision: {action.mode.value} - {action.reasoning}")
            return action
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}\n\nResponse: {response_json}")
            # Return safe default
            return ElleAction(
                mode=ActionMode.IDLE,
                priority=Priority.LOW,
                reasoning=f"Failed to parse response: {e}. Defaulting to idle.",
            )

    def _parse_action(self, response: dict) -> ElleAction:
        """
        Parse JSON response into ElleAction.

        Args:
            response: JSON dict from LLM

        Returns:
            ElleAction object

        Raises:
            ValueError: If response is invalid
        """
        # Parse mode
        mode_str = response.get("mode", "idle").lower()
        try:
            mode = ActionMode(mode_str)
        except ValueError:
            logger.warning(f"Invalid mode '{mode_str}', defaulting to idle")
            mode = ActionMode.IDLE

        # Parse priority
        priority_str = response.get("priority", "low").lower()
        try:
            priority = Priority(priority_str)
        except ValueError:
            logger.warning(f"Invalid priority '{priority_str}', defaulting to low")
            priority = Priority.LOW

        # Parse utterance (can be null)
        utterance = response.get("utterance")

        # Parse visuals
        visuals = []
        for visual_data in response.get("visuals", []):
            try:
                visual_type = VisualDirective(visual_data.get("type", "indicator"))
                targets = visual_data.get("targets", [])
                metadata = visual_data.get("metadata", {})
                visuals.append(Visual(type=visual_type, targets=targets, metadata=metadata))
            except (ValueError, KeyError) as e:
                logger.warning(f"Skipping invalid visual: {e}")

        # Parse followups
        followups = []
        for followup_data in response.get("followups", []):
            try:
                action_name = followup_data.get("action")
                params = followup_data.get("params", {})
                if action_name:
                    followups.append(Followup(action=action_name, params=params))
            except (KeyError, TypeError) as e:
                logger.warning(f"Skipping invalid followup: {e}")

        # Get reasoning
        reasoning = response.get("reasoning", "No reasoning provided")

        return ElleAction(
            mode=mode,
            priority=priority,
            utterance=utterance,
            visuals=visuals,
            followups=followups,
            reasoning=reasoning,
        )
