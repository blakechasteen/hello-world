"""
Unified Voice Agent

Combines HoloLoom's neural voice processing with Elle's thread management
into a single, unified voice interface.

Architecture:
    UnifiedVoiceAgent
        ↓
    VoiceRouter
        ↓     ↓
    HoloLoom Elle
      (neural) (threads)

Features:
- Automatic routing based on intent classification
- Thread-aware conversational queries
- Seamless mode switching
- Graceful degradation if components unavailable

Example:
    >>> from HoloLoom.voice.threads import UnifiedVoiceAgent
    >>> from HoloLoom.config import Config
    >>>
    >>> agent = UnifiedVoiceAgent(config=Config.fast())
    >>> await agent.initialize()
    >>>
    >>> # Thread management
    >>> response = await agent.process("start a new thread for orchard planning")
    >>> print(response)  # "Created new thread 'orchard planning'"
    >>>
    >>> # Conversational query (with thread context)
    >>> response = await agent.process("how should I space apple trees?")
    >>> print(response)  # [HoloLoom response]
"""

import asyncio
from typing import Optional, Dict
import logging

from .voice_router import VoiceRouter
from .voice_modes import VoiceMode

# HoloLoom imports (graceful degradation)
try:
    from HoloLoom.voice.voice_agent import VoiceAgent
    from HoloLoom.weaving_orchestrator import WeavingOrchestrator
    from HoloLoom.config import Config
    from HoloLoom.protocols.types import Query
    HOLOLOOM_AVAILABLE = True
except ImportError:
    HOLOLOOM_AVAILABLE = False
    VoiceAgent = None
    WeavingOrchestrator = None
    Config = None
    Query = None

# Elle imports (graceful degradation)
try:
    from elle.voice.assistant import VoiceAssistant
    ELLE_AVAILABLE = True
except ImportError:
    ELLE_AVAILABLE = False
    VoiceAssistant = None

logger = logging.getLogger(__name__)


class UnifiedVoiceAgent:
    """
    Unified voice interface combining HoloLoom + Elle.

    This is the main entry point for voice-first interactions.
    """

    def __init__(
        self,
        config: Optional[any] = None,
        initial_mode: VoiceMode = VoiceMode.CONVERSATIONAL,
        enable_hololoom: bool = True,
        enable_elle: bool = True
    ):
        """
        Initialize unified voice agent.

        Args:
            config: HoloLoom Config (if None, uses Config.fast())
            initial_mode: Starting voice mode
            enable_hololoom: Enable HoloLoom integration (neural processing)
            enable_elle: Enable Elle integration (thread management)

        Raises:
            RuntimeError: If neither HoloLoom nor Elle is available
        """
        self.config = config

        # Initialize flags
        self.hololoom_enabled = enable_hololoom and HOLOLOOM_AVAILABLE
        self.elle_enabled = enable_elle and ELLE_AVAILABLE

        if not self.hololoom_enabled and not self.elle_enabled:
            raise RuntimeError(
                "Neither HoloLoom nor Elle is available. "
                "At least one voice system must be available."
            )

        # Log availability
        logger.info(f"UnifiedVoiceAgent initializing: "
                   f"HoloLoom={'✓' if self.hololoom_enabled else '✗'}, "
                   f"Elle={'✓' if self.elle_enabled else '✗'}")

        # Initialize components (deferred to initialize())
        self.hololoom_agent: Optional[VoiceAgent] = None
        self.elle_agent: Optional[VoiceAssistant] = None
        self.orchestrator: Optional[WeavingOrchestrator] = None

        # Create router
        self.router = VoiceRouter(
            hololoom_handler=self._hololoom_handler if self.hololoom_enabled else None,
            elle_handler=self._elle_handler if self.elle_enabled else None,
            initial_mode=initial_mode
        )

        logger.info(f"UnifiedVoiceAgent created in {initial_mode} mode")

    async def initialize(self):
        """
        Initialize voice systems (async setup).

        Call this before processing any queries.
        """
        logger.info("Initializing UnifiedVoiceAgent...")

        # Initialize HoloLoom
        if self.hololoom_enabled:
            await self._initialize_hololoom()

        # Initialize Elle
        if self.elle_enabled:
            await self._initialize_elle()

        logger.info("UnifiedVoiceAgent initialization complete")

    async def _initialize_hololoom(self):
        """Initialize HoloLoom voice agent."""
        if not HOLOLOOM_AVAILABLE:
            logger.warning("HoloLoom not available")
            return

        # Get or create config
        if self.config is None:
            self.config = Config.fast()

        # Create orchestrator
        # Note: This requires memory shards - in production, load from persistent storage
        try:
            from HoloLoom.memory.graph import KG
            kg = KG()  # Empty graph for demo - would load from storage in production

            self.orchestrator = WeavingOrchestrator(
                cfg=self.config,
                shards=[]  # Would load from storage in production
            )

            # Create voice agent
            self.hololoom_agent = VoiceAgent(
                orchestrator=self.orchestrator,
                agent_name="HoloLoom",
                voice="nova",
                turn_mode="hybrid"
            )

            logger.info("HoloLoom voice agent initialized")

        except Exception as e:
            logger.error(f"Failed to initialize HoloLoom: {e}")
            self.hololoom_enabled = False

    async def _initialize_elle(self):
        """Initialize Elle voice assistant."""
        if not ELLE_AVAILABLE:
            logger.warning("Elle not available")
            return

        try:
            self.elle_agent = VoiceAssistant(
                sop_dir="elle/sops",
                whisper_model="tiny",
                use_llm_parser=False,  # Use pattern matching for MVP
                verbose=False
            )

            # Initialize Elle's HoloLoom integration
            await self.elle_agent.initialize()

            logger.info("Elle voice assistant initialized")

        except Exception as e:
            logger.error(f"Failed to initialize Elle: {e}")
            self.elle_enabled = False

    async def _hololoom_handler(self, text: str, context: Dict) -> str:
        """
        Handler for conversational queries via HoloLoom.

        Args:
            text: User query text
            context: Context dict (thread info, etc.)

        Returns:
            Response text
        """
        if self.hololoom_agent is None:
            return "HoloLoom is not initialized."

        # Add thread context if available
        thread_context = context.get('thread_context', '')

        # Process with VoiceAgent
        try:
            response = await self.hololoom_agent.process_voice_input(text)
            return response

        except Exception as e:
            logger.error(f"HoloLoom handler error: {e}")
            return f"Error processing query: {str(e)}"

    async def _elle_handler(self, text: str, command_data: Dict) -> str:
        """
        Handler for thread commands via Elle.

        Args:
            text: Original voice input
            command_data: Parsed command data from VoiceRouter

        Returns:
            Response text
        """
        if self.elle_agent is None:
            return "Elle is not initialized."

        # Process with Elle
        try:
            response = await self.elle_agent.process_voice_input(text)
            return response

        except Exception as e:
            logger.error(f"Elle handler error: {e}")
            return f"Error processing command: {str(e)}"

    async def process(self, text: str) -> str:
        """
        Process voice input (main entry point).

        Args:
            text: Voice input text (from STT)

        Returns:
            Response text (to TTS)

        Example:
            >>> agent = UnifiedVoiceAgent()
            >>> await agent.initialize()
            >>>
            >>> response = await agent.process("start a new thread for planning")
            >>> print(response)  # "Created new thread 'planning'"
        """
        # Get thread context if Elle is available
        context = {}
        if self.elle_agent is not None:
            active_thread = self.elle_agent.thread_manager.get_active_thread()
            if active_thread:
                context['active_thread'] = active_thread.name
                context['thread_context'] = active_thread.format_as_context()
            context['thread_count'] = len(self.elle_agent.thread_manager.threads)

        # Route query
        try:
            response = await self.router.route(text, context=context)
            return response

        except Exception as e:
            logger.error(f"Processing error: {e}")
            return f"I encountered an error: {str(e)}"

    @property
    def current_mode(self) -> VoiceMode:
        """Get current voice mode."""
        return self.router.current_mode

    def get_status(self) -> Dict:
        """
        Get agent status.

        Returns:
            Dict with status information
        """
        status = {
            'mode': self.current_mode.name,
            'hololoom_available': self.hololoom_enabled,
            'elle_available': self.elle_enabled,
        }

        # Add thread info if Elle available
        if self.elle_agent is not None:
            active_thread = self.elle_agent.thread_manager.get_active_thread()
            status['active_thread'] = active_thread.name if active_thread else None
            status['thread_count'] = len(self.elle_agent.thread_manager.threads)

        # Add router stats
        status['router'] = self.router.get_stats()

        return status

    async def close(self):
        """Cleanup resources."""
        logger.info("Closing UnifiedVoiceAgent...")

        # Cleanup HoloLoom
        if self.orchestrator is not None:
            try:
                await self.orchestrator.close()
            except Exception:
                pass

        logger.info("UnifiedVoiceAgent closed")
