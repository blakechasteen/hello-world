"""
Voice Mode State Machine

Defines the three voice interaction modes and their transitions:
- CONVERSATIONAL: Natural dialogue with HoloLoom's neural processing
- COMMAND: Structured commands (threads, navigation, app orchestration)
- STREAMING: Continuous cognition with live weaving (Milestone 3)

State Machine:
    CONVERSATIONAL ⟷ COMMAND
         ↓              ↓
        STREAMING ←-----┘

Transitions triggered by:
- Explicit mode switches ("switch to command mode")
- Pattern detection (command keywords detected)
- User preferences (default mode)
"""

import logging
from dataclasses import dataclass
from enum import Enum, auto

logger = logging.getLogger(__name__)


class VoiceMode(Enum):
    """Voice interaction modes"""

    CONVERSATIONAL = auto()  # Natural dialogue, neural processing
    COMMAND = auto()          # Structured commands, thread management
    STREAMING = auto()        # Continuous cognition (Milestone 3)

    def __str__(self) -> str:
        return self.name.lower()


@dataclass
class VoiceModeTransition:
    """
    Represents a mode transition with reason and context.

    Attributes:
        from_mode: Previous mode (None if initial)
        to_mode: Target mode
        trigger: What triggered the transition
        confidence: Confidence in classification (0.0-1.0)
    """
    from_mode: VoiceMode | None
    to_mode: VoiceMode
    trigger: str  # "explicit", "pattern_detected", "preference"
    confidence: float = 1.0

    def __str__(self) -> str:
        if self.from_mode is None:
            return f"Initial mode: {self.to_mode}"
        return f"{self.from_mode} → {self.to_mode} ({self.trigger}, confidence: {self.confidence:.2f})"


class VoiceModeStateMachine:
    """
    State machine for voice mode transitions.

    Features:
    - Validates transitions (ensures legal state changes)
    - Tracks transition history
    - Provides mode-specific defaults
    """

    # Legal transitions (undirected graph)
    VALID_TRANSITIONS = {
        VoiceMode.CONVERSATIONAL: {VoiceMode.COMMAND, VoiceMode.STREAMING},
        VoiceMode.COMMAND: {VoiceMode.CONVERSATIONAL, VoiceMode.STREAMING},
        VoiceMode.STREAMING: {VoiceMode.CONVERSATIONAL, VoiceMode.COMMAND}
    }

    def __init__(self, initial_mode: VoiceMode = VoiceMode.CONVERSATIONAL):
        """
        Initialize state machine.

        Args:
            initial_mode: Starting mode (default: CONVERSATIONAL)
        """
        self.current_mode = initial_mode
        self.transition_history: list[VoiceModeTransition] = []

        # Log initial state
        initial_transition = VoiceModeTransition(
            from_mode=None,
            to_mode=initial_mode,
            trigger="initialization"
        )
        self.transition_history.append(initial_transition)
        logger.info(f"Initialized voice mode: {initial_mode}")

    def can_transition_to(self, target_mode: VoiceMode) -> bool:
        """Check if transition to target mode is valid."""
        if self.current_mode == target_mode:
            return True  # Already in target mode
        return target_mode in self.VALID_TRANSITIONS.get(self.current_mode, set())

    def transition(
        self,
        target_mode: VoiceMode,
        trigger: str = "explicit",
        confidence: float = 1.0
    ) -> VoiceModeTransition:
        """
        Transition to target mode.

        Args:
            target_mode: Mode to transition to
            trigger: What triggered the transition
            confidence: Confidence in transition (0.0-1.0)

        Returns:
            VoiceModeTransition object

        Raises:
            ValueError: If transition is invalid
        """
        if not self.can_transition_to(target_mode):
            raise ValueError(
                f"Invalid transition: {self.current_mode} → {target_mode}. "
                f"Valid targets: {self.VALID_TRANSITIONS.get(self.current_mode, set())}"
            )

        # Create transition record
        transition = VoiceModeTransition(
            from_mode=self.current_mode,
            to_mode=target_mode,
            trigger=trigger,
            confidence=confidence
        )

        # Update state
        self.current_mode = target_mode
        self.transition_history.append(transition)

        logger.info(f"Mode transition: {transition}")
        return transition

    def get_recent_transitions(self, n: int = 5) -> list[VoiceModeTransition]:
        """Get last n transitions."""
        return self.transition_history[-n:]

    def get_mode_defaults(self) -> dict[str, any]:
        """
        Get default settings for current mode.

        Returns:
            Dict with mode-specific defaults
        """
        defaults = {
            VoiceMode.CONVERSATIONAL: {
                'use_neural_processing': True,
                'enable_thread_context': True,
                'response_style': 'natural',
                'latency_target_ms': 500
            },
            VoiceMode.COMMAND: {
                'use_neural_processing': False,
                'enable_thread_context': True,
                'response_style': 'structured',
                'latency_target_ms': 200
            },
            VoiceMode.STREAMING: {
                'use_neural_processing': True,
                'enable_thread_context': True,
                'response_style': 'ambient',
                'latency_target_ms': 150
            }
        }
        return defaults.get(self.current_mode, {})


# Mode-specific response templates
MODE_RESPONSES = {
    VoiceMode.CONVERSATIONAL: {
        'greeting': "I'm listening. What would you like to explore?",
        'acknowledgment': "Got it.",
        'clarification': "Could you clarify what you mean by that?"
    },
    VoiceMode.COMMAND: {
        'greeting': "Command mode active. What would you like to do?",
        'acknowledgment': "Command executed.",
        'clarification': "Command not recognized. Try 'help' for available commands."
    },
    VoiceMode.STREAMING: {
        'greeting': "Streaming mode active. I'm listening continuously.",
        'acknowledgment': "Noted.",
        'clarification': "Continue..."
    }
}


def get_response_template(mode: VoiceMode, template_key: str) -> str:
    """Get mode-specific response template."""
    return MODE_RESPONSES.get(mode, {}).get(
        template_key,
        "I'm not sure how to respond to that."
    )
