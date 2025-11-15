"""
Elle Core - Engine

The main orchestrator that handles requests and coordinates all components.
"""
import logging
from typing import Optional
from dataclasses import dataclass

from .models import (
    SceneSnapshot,
    UserState,
    UserIntent,
    ElleAction,
    MemoryEntry,
    MemoryType,
)
from .core import EllePolicy, LLMClient
from .services.memory import MemoryService


logger = logging.getLogger(__name__)


@dataclass
class ElleRequest:
    """
    A request for Elle to process.

    This is the normalized input format from any interface (AR, Matrix, CLI, etc.).
    """
    scene: SceneSnapshot
    user_state: UserState
    user_intent: Optional[UserIntent] = None


class ElleEngine:
    """
    The main Elle orchestrator.

    Entry point: handle(request) -> action
    - Loads memory snapshot
    - Calls policy to decide
    - Dispatches followups
    - Logs interaction
    """

    def __init__(
        self,
        policy: EllePolicy,
        memory_service: MemoryService,
    ):
        """
        Initialize the engine.

        Args:
            policy: The decision-making policy
            memory_service: Memory storage service
        """
        self.policy = policy
        self.memory = memory_service

    async def handle(self, request: ElleRequest) -> ElleAction:
        """
        Handle a request and return Elle's action.

        This is the main entry point for all interactions.

        Args:
            request: The normalized request

        Returns:
            ElleAction describing what Elle should do
        """
        logger.info(f"Handling request: {request.scene.location}")

        # Step 1: Load memory snapshot
        memory_snapshot = await self.memory.get_snapshot(
            location=request.scene.location,
            related_objects=[obj.id for obj in request.scene.objects],
        )

        # Step 2: Call policy to decide
        action = await self.policy.decide(
            scene=request.scene,
            user_state=request.user_state,
            user_intent=request.user_intent,
            memory=memory_snapshot,
        )

        # Step 3: Log this interaction
        await self._log_interaction(request, action)

        # Step 4: Dispatch followups
        await self._dispatch_followups(action, request)

        return action

    async def _log_interaction(
        self,
        request: ElleRequest,
        action: ElleAction,
    ) -> None:
        """Log this interaction to memory."""
        # Create a summary of the interaction
        utterance_part = f'Said: "{action.utterance}"' if action.utterance else "Silent"
        summary = (
            f"{request.scene.location} - "
            f"{request.user_state.focus.value} focus, "
            f"{request.user_state.mood.value} mood. "
            f"Elle: {action.mode.value}. {utterance_part}"
        )

        await self.memory.log_interaction(
            content=summary,
            location=request.scene.location,
            importance=0.5 if action.mode.value == "idle" else 0.7,
        )

    async def _dispatch_followups(
        self,
        action: ElleAction,
        request: ElleRequest,
    ) -> None:
        """
        Dispatch followup actions to appropriate services.

        This is where task creation, scheduling, logging, etc. happens.
        """
        for followup in action.followups:
            logger.info(f"Dispatching followup: {followup.action}")

            try:
                if followup.action == "log_observation":
                    await self._handle_log_observation(followup.params, request)
                elif followup.action == "schedule_reminder":
                    await self._handle_schedule_reminder(followup.params)
                elif followup.action == "create_task":
                    await self._handle_create_task(followup.params)
                else:
                    logger.warning(f"Unknown followup action: {followup.action}")
            except Exception as e:
                logger.error(f"Followup failed: {followup.action} - {e}")

    async def _handle_log_observation(
        self,
        params: dict,
        request: ElleRequest,
    ) -> None:
        """Handle log_observation followup."""
        content = params.get("activity", "observation")
        location = params.get("location", request.scene.location)

        await self.memory.log_observation(
            content=f"{content} observed",
            location=location,
            importance=0.5,
        )

    async def _handle_schedule_reminder(self, params: dict) -> None:
        """
        Handle schedule_reminder followup.

        In a real implementation, this would integrate with a task scheduler.
        For now, we just log it.
        """
        target = params.get("target", "unknown")
        when = params.get("when", "later")
        message = params.get("message", "reminder")

        logger.info(f"Reminder scheduled: {message} about {target} {when}")

        # Log as a task memory
        await self.memory.store.store(
            MemoryEntry(
                id=f"reminder_{target}_{when}",
                type=MemoryType.TASK,
                content=f"Reminder: {message} about {target}",
                metadata={"when": when, "target": target},
                importance=0.7,
            )
        )

    async def _handle_create_task(self, params: dict) -> None:
        """
        Handle create_task followup.

        In a real implementation, this would create a Task object.
        For now, we just log it.
        """
        description = params.get("description", "task")
        location = params.get("location")

        logger.info(f"Task created: {description}")

        await self.memory.store.store(
            MemoryEntry(
                id=f"task_{description[:20]}",
                type=MemoryType.TASK,
                content=f"Task: {description}",
                location=location,
                importance=0.8,
            )
        )


def create_engine(
    llm_client: LLMClient,
    memory_service: Optional[MemoryService] = None,
) -> ElleEngine:
    """
    Factory function to create a fully configured ElleEngine.

    Args:
        llm_client: The LLM client to use
        memory_service: Optional memory service (creates InMemory if not provided)

    Returns:
        Configured ElleEngine
    """
    from .services.memory import InMemoryStore

    # Create policy
    policy = EllePolicy(llm_client=llm_client)

    # Create memory service if not provided
    if memory_service is None:
        store = InMemoryStore()
        memory_service = MemoryService(store=store)

    # Create engine
    return ElleEngine(
        policy=policy,
        memory_service=memory_service,
    )
