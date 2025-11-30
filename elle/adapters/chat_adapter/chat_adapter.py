"""Chat adapter for text-based Elle interaction.

Translates simple text messages into ElleRequests and
ElleActions back into text responses.

Supports seamless transitions to AR mode when visual context is needed.
"""

from typing import Optional, Dict, Any, Callable, Awaitable
from datetime import datetime
import uuid
import re
import logging

from elle.domain.scene import SceneSnapshot
from elle.domain.intent import UserIntent, UserState, InteractionMode, ScanType
from elle.domain.action import ElleAction, ElleMode
from elle.engine.request import create_request

from .conversation import Conversation, ConversationManager
from .chat_scene import create_virtual_scene, detect_scene_from_text

# AR bridge for mode transitions
from elle.adapters.chat_ar_bridge import (
    ChatARBridge,
    create_chat_ar_bridge,
    ARTrigger,
    ElleMode as BridgeMode,
)


logger = logging.getLogger(__name__)


class ChatAdapter:
    """
    Adapter for text-based chat with Elle.

    Converts:
    - User text → ElleRequest
    - ElleAction → Text response

    Maintains conversation context for multi-turn interactions.
    Supports seamless AR mode transitions via ChatARBridge.
    """

    def __init__(
        self,
        user_name: str = "User",
        user_id: Optional[str] = None,
        enable_ar_bridge: bool = True,
        on_ar_activate: Optional[Callable[..., Awaitable[None]]] = None,
        on_ar_deactivate: Optional[Callable[..., Awaitable[None]]] = None,
    ):
        self.user_name = user_name
        self.user_id = user_id or f"chat_user_{uuid.uuid4().hex[:8]}"
        self.conversation_manager = ConversationManager()
        self.current_scene: Optional[SceneSnapshot] = None

        # AR bridge for mode transitions
        self.ar_bridge: Optional[ChatARBridge] = None
        if enable_ar_bridge:
            self.ar_bridge = create_chat_ar_bridge(
                on_ar_activate=on_ar_activate,
                on_ar_deactivate=on_ar_deactivate,
                auto_transition=True,
            )

        # Last AR trigger detected (for response generation)
        self._pending_ar_trigger: Optional[ARTrigger] = None

    def start_session(self) -> Conversation:
        """Start a new chat session."""
        conv = self.conversation_manager.start_conversation()
        self.current_scene = create_virtual_scene()
        logger.info(f"Started chat session: {conv.session_id}")
        return conv

    def get_conversation(self) -> Conversation:
        """Get current conversation, creating if needed."""
        return self.conversation_manager.get_or_create()

    def text_to_request(self, text: str) -> 'ElleRequest':
        """
        Convert user text to ElleRequest.

        Args:
            text: User's message

        Returns:
            ElleRequest ready for ElleEngine.handle()
        """
        # Update conversation
        conv = self.get_conversation()
        conv.add_user_message(text)

        # Detect or update scene from text
        self.current_scene = detect_scene_from_text(text)

        # Infer intent from text
        intent = self._infer_intent(text)

        # Build user state
        user_state = UserState(
            user_id=self.user_id,
            name=self.user_name,
            current_energy_level="medium",
            preferred_pace="calm",
            current_projects=[],
        )

        # Build request
        return create_request(
            scene=self.current_scene,
            intent=intent,
            user=user_state,
            source_adapter="chat",
            request_id=f"chat_{uuid.uuid4().hex[:8]}",
        )

    def _infer_intent(self, text: str) -> UserIntent:
        """
        Infer user intent from text.

        Simple heuristics for now - could be enhanced with NLP.
        """
        text_lower = text.lower()

        # Check for explicit requests/commands
        explicit_request = None
        is_question = "?" in text
        is_command = any(text_lower.startswith(cmd) for cmd in [
            "help", "show", "tell", "what", "how", "where", "why", "can you"
        ])

        # Determine interaction mode
        if is_question or is_command:
            mode = InteractionMode.SEEKING_GUIDANCE
        elif any(kw in text_lower for kw in ["remind", "schedule", "later", "tomorrow"]):
            mode = InteractionMode.REQUESTING_INFO  # Closest match for task management
        elif any(kw in text_lower for kw in ["start", "begin", "do", "let's"]):
            mode = InteractionMode.TASK_EXECUTION
        else:
            mode = InteractionMode.PASSIVE

        # Check for urgency indicators
        is_rushed = any(kw in text_lower for kw in ["urgent", "asap", "quick", "hurry", "now"])

        # Check for tiredness indicators
        is_tired = any(kw in text_lower for kw in [
            "tired", "exhausted", "overwhelmed", "can't think", "brain fog"
        ])

        # Check for exploration mode
        is_exploring = any(kw in text_lower for kw in [
            "wonder", "curious", "explore", "what if", "maybe", "thinking about"
        ])

        return UserIntent(
            mode=mode,
            scan_type=ScanType.FOCUSED,  # Chat is always focused
            is_tired=is_tired,
            is_rushed=is_rushed,
            is_exploring=is_exploring,
            explicit_request=text if is_command or is_question else None,
        )

    def action_to_text(self, action: ElleAction) -> str:
        """
        Convert ElleAction to text response.

        Args:
            action: Elle's decision

        Returns:
            Text to display to user
        """
        # Update conversation with Elle's response
        conv = self.get_conversation()

        # Get the utterance or generate one from mode
        response = self._extract_response(action)

        conv.add_elle_response(
            content=response,
            reasoning=action.reasoning,
            mode=action.mode.value if action.mode else None,
        )

        return response

    def _extract_response(self, action: ElleAction) -> str:
        """Extract text response from action."""
        # If Elle has something to say, use it
        if action.utterance:
            return action.utterance

        # Mode-based fallbacks
        if action.mode == ElleMode.SILENT:
            return "(Elle remains quiet, observing)"
        elif action.mode == ElleMode.IDLE:
            return "(Elle is present but chooses not to intervene)"
        elif action.mode == ElleMode.DEFERRED:
            return "(Elle notes this for later)"

        # If there's a suggested task
        if action.suggested_task_summary:
            return f"Consider: {action.suggested_task_summary}"

        # Default
        return "(Elle listens)"

    def get_context_summary(self) -> str:
        """Get summary of current conversation context."""
        conv = self.get_conversation()
        return conv.get_context_for_prompt()

    def end_session(self) -> Optional[Conversation]:
        """End current chat session."""
        ended = self.conversation_manager.end_conversation()
        if ended:
            logger.info(f"Ended chat session: {ended.session_id}")
        return ended

    # ========================================================================
    # AR Bridge Methods
    # ========================================================================

    async def check_ar_trigger(self, text: str) -> Dict[str, Any]:
        """
        Check if message should trigger AR mode transition.

        Args:
            text: User's message

        Returns:
            Dict with transition info:
                - should_transition: bool
                - ar_trigger: ARTrigger or None
                - response_prefix: Text to prepend to response
        """
        if not self.ar_bridge:
            return {"should_transition": False, "ar_trigger": None, "response_prefix": None}

        # Get chat context for AR
        conv = self.get_conversation()
        chat_context = {
            "recent_messages": conv.get_context_for_prompt(),
            "user_name": self.user_name,
        }

        result = await self.ar_bridge.process_chat_message(text, chat_context)

        # Store pending trigger for response generation
        if result.get("ar_trigger"):
            self._pending_ar_trigger = result["ar_trigger"]

        return result

    def get_ar_mode(self) -> str:
        """Get current AR bridge mode."""
        if not self.ar_bridge:
            return "chat"
        return self.ar_bridge.current_mode.value

    def is_ar_mode(self) -> bool:
        """Check if currently in AR mode."""
        return self.ar_bridge and self.ar_bridge.current_mode == BridgeMode.AR

    def get_ar_context_for_prompt(self) -> str:
        """Get AR-relevant context for prompt building."""
        if not self.ar_bridge:
            return ""
        return self.ar_bridge.generate_ar_prompt_context()

    def get_pending_ar_trigger(self) -> Optional[ARTrigger]:
        """Get and clear pending AR trigger."""
        trigger = self._pending_ar_trigger
        self._pending_ar_trigger = None
        return trigger


def create_chat_adapter(
    user_name: str = "User",
    user_id: Optional[str] = None,
    enable_ar_bridge: bool = True,
    on_ar_activate: Optional[Callable[..., Awaitable[None]]] = None,
    on_ar_deactivate: Optional[Callable[..., Awaitable[None]]] = None,
) -> ChatAdapter:
    """Factory function for creating chat adapter."""
    return ChatAdapter(
        user_name=user_name,
        user_id=user_id,
        enable_ar_bridge=enable_ar_bridge,
        on_ar_activate=on_ar_activate,
        on_ar_deactivate=on_ar_deactivate,
    )
