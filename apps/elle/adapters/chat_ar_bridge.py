"""Chat-AR Bridge for seamless mode transitions.

Enables Elle to switch between chat and AR modes based on user intent.

Example flow:
    [Chat Mode]
    User: Show me where to start organizing

    [Elle triggers AR Mode]
    Elle: Looking at your space now...
    *Signals AR client to activate camera*
    *AR client connects and detects objects*
    *Elle highlights workbench*

Created: 2025-11-29
"""

import re
import logging
from typing import Optional, Dict, Any, Callable, Awaitable, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


logger = logging.getLogger(__name__)


class ElleMode(Enum):
    """Current Elle interaction mode."""
    CHAT = "chat"
    AR = "ar"
    TRANSITIONING = "transitioning"


@dataclass
class ARTrigger:
    """Detected AR trigger from chat."""
    trigger_type: str  # "visual", "spatial", "demonstration"
    trigger_phrase: str
    confidence: float
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModeTransition:
    """Represents a mode transition request."""
    from_mode: ElleMode
    to_mode: ElleMode
    trigger: Optional[ARTrigger]
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)


# AR Trigger patterns
AR_TRIGGER_PATTERNS = {
    # Visual requests - user wants to SEE something
    "visual": [
        r"\bshow me\b",
        r"\blook at\b",
        r"\blet me see\b",
        r"\bwhat does.*look like\b",
        r"\bcan you see\b",
        r"\bsee (the|this|my|that)\b",
        r"\bvisual(ly|ize)?\b",
        r"\bhighlight\b",
        r"\bpoint (to|at|out)\b",
        r"\bwhere (is|are|should)\b",
    ],
    # Spatial requests - user wants location/positioning
    "spatial": [
        r"\bwhere should i (put|place|start|go)\b",
        r"\bfind (the|a|my)\b",
        r"\blocate\b",
        r"\bguide me (to|through)\b",
        r"\bnavigat",
        r"\bposition\b",
        r"\barrange\b",
        r"\blayout\b",
    ],
    # Demonstration requests - user wants step-by-step visual guidance
    "demonstration": [
        r"\bshow me how\b",
        r"\bdemonstrat",
        r"\bwalk me through\b",
        r"\bstep by step\b",
        r"\bguide me\b",
        r"\bhelp me (do|start|with)\b",
        r"\bteach me\b",
    ],
}

# Phrases that indicate returning to chat mode
CHAT_RETURN_PATTERNS = [
    r"\bstop (looking|camera|ar|visual)\b",
    r"\bgo back to (chat|text)\b",
    r"\bjust (talk|text|chat)\b",
    r"\bdone (with|looking)\b",
    r"\bend (ar|visual|camera)\b",
    r"\bclose (camera|ar)\b",
    r"\bnever ?mind\b",
    r"\bcancel\b",
]


class ChatARBridge:
    """
    Bridge enabling seamless transitions between chat and AR modes.

    Responsibilities:
    - Detect AR triggers in chat messages
    - Coordinate mode transitions
    - Pass context between modes
    - Manage transition state
    """

    def __init__(
        self,
        on_ar_activate: Optional[Callable[..., Awaitable[None]]] = None,
        on_ar_deactivate: Optional[Callable[..., Awaitable[None]]] = None,
        auto_transition: bool = True,
    ):
        """
        Initialize Chat-AR Bridge.

        Args:
            on_ar_activate: Callback when AR mode should activate
            on_ar_deactivate: Callback when AR mode should deactivate
            auto_transition: Whether to automatically transition modes
        """
        self.current_mode = ElleMode.CHAT
        self.on_ar_activate = on_ar_activate
        self.on_ar_deactivate = on_ar_deactivate
        self.auto_transition = auto_transition

        # Context preserved across mode transitions
        self.shared_context: Dict[str, Any] = {}

        # Transition history
        self.transitions: List[ModeTransition] = []

        # Compile patterns for efficiency
        self._ar_patterns = {
            trigger_type: [re.compile(p, re.IGNORECASE) for p in patterns]
            for trigger_type, patterns in AR_TRIGGER_PATTERNS.items()
        }
        self._chat_return_patterns = [
            re.compile(p, re.IGNORECASE) for p in CHAT_RETURN_PATTERNS
        ]

    def detect_ar_trigger(self, text: str) -> Optional[ARTrigger]:
        """
        Detect if chat message contains an AR trigger.

        Args:
            text: User's chat message

        Returns:
            ARTrigger if detected, None otherwise
        """
        text_lower = text.lower()

        best_match: Optional[ARTrigger] = None
        best_confidence = 0.0

        for trigger_type, patterns in self._ar_patterns.items():
            for pattern in patterns:
                match = pattern.search(text_lower)
                if match:
                    # Calculate confidence based on match specificity
                    confidence = self._calculate_trigger_confidence(
                        match, text_lower, trigger_type
                    )

                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = ARTrigger(
                            trigger_type=trigger_type,
                            trigger_phrase=match.group(),
                            confidence=confidence,
                            context={"original_text": text},
                        )

        return best_match

    def detect_chat_return(self, text: str) -> bool:
        """
        Detect if message indicates user wants to return to chat mode.

        Args:
            text: User's message

        Returns:
            True if chat return detected
        """
        text_lower = text.lower()

        for pattern in self._chat_return_patterns:
            if pattern.search(text_lower):
                return True

        return False

    def _calculate_trigger_confidence(
        self,
        match: re.Match,
        text: str,
        trigger_type: str,
    ) -> float:
        """Calculate confidence score for AR trigger detection."""
        confidence = 0.5  # Base confidence for any match

        # Longer matches are more confident
        match_len = len(match.group())
        if match_len > 10:
            confidence += 0.2
        elif match_len > 5:
            confidence += 0.1

        # Multiple trigger words boost confidence
        total_triggers = sum(
            1 for patterns in self._ar_patterns.values()
            for p in patterns
            if p.search(text)
        )
        if total_triggers > 1:
            confidence += 0.1 * min(total_triggers - 1, 3)

        # Demonstration triggers are high-confidence
        if trigger_type == "demonstration":
            confidence += 0.1

        return min(confidence, 1.0)

    async def process_chat_message(
        self,
        text: str,
        chat_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process chat message and handle mode transitions.

        Args:
            text: User's chat message
            chat_context: Current chat context (conversation history, etc.)

        Returns:
            Dict with:
                - should_transition: bool
                - transition: ModeTransition or None
                - ar_trigger: ARTrigger or None
                - response_prefix: Optional text to prepend to response
        """
        result = {
            "should_transition": False,
            "transition": None,
            "ar_trigger": None,
            "response_prefix": None,
        }

        # Check for mode return first
        if self.current_mode == ElleMode.AR and self.detect_chat_return(text):
            transition = await self._transition_to_chat()
            result["should_transition"] = True
            result["transition"] = transition
            result["response_prefix"] = "Returning to chat. "
            return result

        # Check for AR trigger
        trigger = self.detect_ar_trigger(text)

        if trigger and self.current_mode == ElleMode.CHAT:
            result["ar_trigger"] = trigger

            if self.auto_transition and trigger.confidence >= 0.6:
                # High confidence - transition automatically
                transition = await self._transition_to_ar(trigger, chat_context)
                result["should_transition"] = True
                result["transition"] = transition
                result["response_prefix"] = "Looking at your space now... "
            elif trigger.confidence >= 0.4:
                # Medium confidence - suggest AR mode
                result["response_prefix"] = (
                    "Would you like me to look at your space? "
                    "I can help better if I can see what you're working with. "
                )

        return result

    async def _transition_to_ar(
        self,
        trigger: ARTrigger,
        chat_context: Optional[Dict[str, Any]] = None,
    ) -> ModeTransition:
        """Execute transition from chat to AR mode."""
        transition = ModeTransition(
            from_mode=ElleMode.CHAT,
            to_mode=ElleMode.AR,
            trigger=trigger,
            context={
                "chat_context": chat_context or {},
                "trigger_phrase": trigger.trigger_phrase,
            },
        )

        self.current_mode = ElleMode.TRANSITIONING

        # Preserve chat context for AR
        self.shared_context.update({
            "last_chat_message": trigger.context.get("original_text"),
            "transition_reason": trigger.trigger_type,
            "chat_history": chat_context,
        })

        # Call AR activation callback
        if self.on_ar_activate:
            try:
                await self.on_ar_activate(self.shared_context)
            except Exception as e:
                logger.error(f"AR activation failed: {e}")
                self.current_mode = ElleMode.CHAT
                raise

        self.current_mode = ElleMode.AR
        self.transitions.append(transition)

        logger.info(f"Transitioned to AR mode: {trigger.trigger_type}")
        return transition

    async def _transition_to_chat(self) -> ModeTransition:
        """Execute transition from AR back to chat mode."""
        transition = ModeTransition(
            from_mode=ElleMode.AR,
            to_mode=ElleMode.CHAT,
            trigger=None,
            context={"ar_context": self.shared_context.copy()},
        )

        self.current_mode = ElleMode.TRANSITIONING

        # Call AR deactivation callback
        if self.on_ar_deactivate:
            try:
                await self.on_ar_deactivate()
            except Exception as e:
                logger.error(f"AR deactivation failed: {e}")

        self.current_mode = ElleMode.CHAT
        self.transitions.append(transition)

        logger.info("Transitioned back to chat mode")
        return transition

    def get_mode_context(self) -> Dict[str, Any]:
        """Get current mode context for request building."""
        return {
            "current_mode": self.current_mode.value,
            "shared_context": self.shared_context,
            "transition_count": len(self.transitions),
            "last_transition": (
                self.transitions[-1].timestamp.isoformat()
                if self.transitions else None
            ),
        }

    def generate_ar_prompt_context(self) -> str:
        """
        Generate context string for Elle's prompt in AR mode.

        Includes relevant chat history and transition reason.
        """
        if not self.shared_context:
            return ""

        parts = []

        if self.shared_context.get("last_chat_message"):
            parts.append(
                f"User just asked: \"{self.shared_context['last_chat_message']}\""
            )

        if self.shared_context.get("transition_reason"):
            reason = self.shared_context["transition_reason"]
            if reason == "visual":
                parts.append("User wants to see something specific.")
            elif reason == "spatial":
                parts.append("User needs help with location or positioning.")
            elif reason == "demonstration":
                parts.append("User wants step-by-step visual guidance.")

        return " ".join(parts)


# Convenience factory
def create_chat_ar_bridge(
    on_ar_activate: Optional[Callable[..., Awaitable[None]]] = None,
    on_ar_deactivate: Optional[Callable[..., Awaitable[None]]] = None,
    auto_transition: bool = True,
) -> ChatARBridge:
    """Create a Chat-AR bridge instance."""
    return ChatARBridge(
        on_ar_activate=on_ar_activate,
        on_ar_deactivate=on_ar_deactivate,
        auto_transition=auto_transition,
    )
