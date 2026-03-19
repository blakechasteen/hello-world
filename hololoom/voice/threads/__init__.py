"""
Voice-First UX Layer (November 2025)

Unified voice interface combining HoloLoom's neural processing with Elle's
thread management and command parsing.

Features:
- 3 voice modes: Conversational, Command, Streaming
- Thread management by voice (create, switch, list)
- Automatic routing to appropriate handler
- Natural language and structured command support

Quick Start:
    >>> from hololoom.voice.threads import UnifiedVoiceAgent
    >>> from hololoom.config import Config
    >>>
    >>> agent = UnifiedVoiceAgent(config=Config.fast())
    >>> response = await agent.process("start a new thread for planning")
    >>> print(response)

See demos/demo_voice_first.py for complete examples.
"""

from .core.unified_agent import UnifiedVoiceAgent
from .core.voice_modes import VoiceMode, VoiceModeTransition
from .core.voice_router import VoiceRouter
from .grammar.voice_grammar import VoiceGrammar

__all__ = [
    'VoiceMode',
    'VoiceModeTransition',
    'VoiceRouter',
    'UnifiedVoiceAgent',
    'VoiceGrammar'
]

__version__ = "0.1.0"
