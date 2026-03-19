"""
Voice Router

Routes voice queries to appropriate handlers based on intent classification.

Architecture:
    Voice Input
        ↓
    VoiceGrammar.classify()
        ↓
    ┌─────────────────┐
    │  VoiceRouter    │
    └─────────────────┘
         ↓        ↓
    Thread Cmd  Conversational
         ↓        ↓
      Elle    HoloLoom

Routing Logic:
- Thread commands (create, switch, list) → Elle.ThreadManager
- Conversational queries → HoloLoom.VoiceAgent + WeavingOrchestrator
- Mode switches → Update VoiceModeStateMachine
- Ambiguous → Ask for clarification

Example:
    >>> router = VoiceRouter(hololoom_handler, elle_handler)
    >>> response = await router.route("start a new thread for planning")
    >>> # Routes to Elle's thread manager
"""

import logging
from collections.abc import Awaitable, Callable

from ..grammar.voice_grammar import CommandIntent, CommandType, VoiceGrammar
from .voice_modes import VoiceMode, VoiceModeStateMachine, get_response_template

logger = logging.getLogger(__name__)


# Type aliases
VoiceHandler = Callable[[str, dict], Awaitable[str]]


class VoiceRouter:
    """
    Routes voice queries to appropriate handlers.

    The router maintains a state machine for voice modes and uses grammar
    classification to determine which handler should process each query.
    """

    def __init__(
        self,
        hololoom_handler: VoiceHandler | None = None,
        elle_handler: VoiceHandler | None = None,
        initial_mode: VoiceMode = VoiceMode.CONVERSATIONAL
    ):
        """
        Initialize voice router.

        Args:
            hololoom_handler: Handler for conversational queries (async callable)
            elle_handler: Handler for thread/command queries (async callable)
            initial_mode: Starting voice mode
        """
        self.hololoom_handler = hololoom_handler
        self.elle_handler = elle_handler

        # Initialize components
        self.grammar = VoiceGrammar()
        self.state_machine = VoiceModeStateMachine(initial_mode)

        logger.info(f"VoiceRouter initialized in {initial_mode} mode")

    @property
    def current_mode(self) -> VoiceMode:
        """Get current voice mode."""
        return self.state_machine.current_mode

    def classify(self, text: str) -> CommandIntent:
        """
        Classify voice input.

        Args:
            text: Voice input text

        Returns:
            CommandIntent with classification and parameters
        """
        return self.grammar.classify(text)

    async def route(self, text: str, context: dict | None = None) -> str:
        """
        Route voice query to appropriate handler.

        Args:
            text: Voice input text
            context: Optional context (thread info, session data, etc.)

        Returns:
            Response string

        Raises:
            ValueError: If no appropriate handler is available
        """
        context = context or {}

        # Classify input
        intent = self.classify(text)

        logger.info(f"Routing: {intent.command_type.name} (confidence: {intent.confidence:.2f})")

        # Route based on command type
        if intent.command_type == CommandType.MODE_SWITCH:
            return await self._handle_mode_switch(intent)

        elif intent.command_type in {
            CommandType.THREAD_CREATE,
            CommandType.THREAD_SWITCH,
            CommandType.THREAD_LIST,
            CommandType.THREAD_SUMMARIZE,
            CommandType.THREAD_BRANCH,
            CommandType.THREAD_MERGE
        }:
            return await self._handle_thread_command(intent, context)

        elif intent.command_type == CommandType.HELP:
            return self._handle_help(intent)

        elif intent.command_type == CommandType.STATUS:
            return self._handle_status(context)

        elif intent.command_type == CommandType.CONVERSATIONAL:
            return await self._handle_conversational(text, context)

        else:
            logger.warning(f"Unknown command type: {intent.command_type}")
            return "I'm not sure how to handle that. Try 'help' for available commands."

    async def _handle_mode_switch(self, intent: CommandIntent) -> str:
        """Handle mode switching commands."""
        mode_name = intent.params.get('mode', '').lower()

        try:
            # Map mode name to VoiceMode enum
            target_mode = {
                'conversational': VoiceMode.CONVERSATIONAL,
                'command': VoiceMode.COMMAND,
                'streaming': VoiceMode.STREAMING
            }.get(mode_name)

            if target_mode is None:
                return f"Unknown mode: {mode_name}. Available: conversational, command, streaming."

            # Attempt transition
            transition = self.state_machine.transition(target_mode, trigger="explicit")

            # Return mode-specific greeting
            greeting = get_response_template(target_mode, 'greeting')
            return f"Switched to {target_mode}. {greeting}"

        except ValueError as e:
            logger.error(f"Mode switch failed: {e}")
            return f"Cannot switch to {mode_name} mode from {self.current_mode}."

    async def _handle_thread_command(self, intent: CommandIntent, context: dict) -> str:
        """Handle thread management commands."""
        if self.elle_handler is None:
            logger.error("Elle handler not available for thread commands")
            return "Thread management is not available. Elle handler not initialized."

        # Prepare command for Elle
        command_data = {
            'command_type': intent.command_type.name,
            'params': intent.params,
            'original_text': intent.original_text
        }

        try:
            response = await self.elle_handler(intent.original_text, command_data)
            return response

        except Exception as e:
            logger.error(f"Elle handler error: {e}")
            return f"Error processing thread command: {str(e)}"

    async def _handle_conversational(self, text: str, context: dict) -> str:
        """Handle conversational queries via HoloLoom."""
        if self.hololoom_handler is None:
            logger.error("HoloLoom handler not available for conversational queries")
            return "Conversational mode is not available. HoloLoom handler not initialized."

        try:
            # Add thread context if available
            query_context = {
                'mode': self.current_mode.name,
                **context
            }

            response = await self.hololoom_handler(text, query_context)
            return response

        except Exception as e:
            logger.error(f"HoloLoom handler error: {e}")
            return f"Error processing query: {str(e)}"

    def _handle_help(self, intent: CommandIntent) -> str:
        """Handle help requests."""
        topic = intent.params.get('topic')

        if topic:
            # Topic-specific help
            return f"Help for '{topic}':\n\nTry asking specific questions or use 'help' for general commands."

        # General help
        examples = self.grammar.get_all_examples()

        help_text = ["Available voice commands:\n"]

        # Thread operations
        help_text.append("Thread Management:")
        for example in examples.get('THREAD_CREATE', []):
            help_text.append(f"  - {example}")
        for example in examples.get('THREAD_SWITCH', [])[:2]:
            help_text.append(f"  - {example}")

        # Mode switching
        help_text.append("\nMode Switching:")
        for example in examples.get('MODE_SWITCH', []):
            help_text.append(f"  - {example}")

        # Meta commands
        help_text.append("\nMeta Commands:")
        help_text.append("  - status (show current mode and thread)")
        help_text.append("  - list threads (show all active threads)")

        help_text.append("\nOr just ask a question naturally!")

        return "\n".join(help_text)

    def _handle_status(self, context: dict) -> str:
        """Handle status requests."""
        active_thread = context.get('active_thread', 'None')
        thread_count = context.get('thread_count', 0)

        status_text = [
            f"Current mode: {self.current_mode}",
            f"Active thread: {active_thread}",
            f"Total threads: {thread_count}",
        ]

        # Add recent transitions
        recent_transitions = self.state_machine.get_recent_transitions(n=3)
        if len(recent_transitions) > 1:  # Exclude initial transition
            status_text.append("\nRecent mode changes:")
            for t in recent_transitions[-3:]:
                if t.from_mode is not None:  # Skip initial
                    status_text.append(f"  - {t}")

        return "\n".join(status_text)

    def get_stats(self) -> dict:
        """Get router statistics."""
        return {
            'current_mode': self.current_mode.name,
            'transition_count': len(self.state_machine.transition_history),
            'recent_transitions': [
                str(t) for t in self.state_machine.get_recent_transitions(n=5)
            ]
        }
