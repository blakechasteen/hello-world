"""
HoloLoom Voice Module
=====================

Bidirectional voice interaction for HoloLoom agents.

Components:
- VoiceAgent: Main voice interface with orchestrator integration
- TTSProvider: Text-to-speech synthesis (OpenAI, etc.)
- VoiceActivityDetector: Speech detection (WebRTC VAD)
- TurnTakingManager: Conversation turn management
- ConversationMemory: Short and long-term conversation storage
- PersonalityManager: Multi-persona system with voice customization (Phase 3)

Date: November 15, 2025
Updated: November 16, 2025 (Phase 3 - Personality Framework)
"""

# Personality module (always available)
from .personality import (
    PersonalityTraits,
    Personality,
    PersonalityManager,
    PersonalityType
)

# Voice agent module (optional dependencies)
try:
    from .voice_agent import (
        VoiceAgent,
        TTSProvider,
        OpenAITTS,
        TTSManager,
        VoiceActivityDetector,
        TurnTakingManager,
        ConversationMemory,
        ConversationTurn,
        TurnState
    )
    VOICE_AGENT_AVAILABLE = True
except ImportError:
    VOICE_AGENT_AVAILABLE = False

__all__ = [
    # Personality (Phase 3)
    'PersonalityTraits',
    'Personality',
    'PersonalityManager',
    'PersonalityType',
]

if VOICE_AGENT_AVAILABLE:
    __all__.extend([
        'VoiceAgent',
        'TTSProvider',
        'OpenAITTS',
        'TTSManager',
        'VoiceActivityDetector',
        'TurnTakingManager',
        'ConversationMemory',
        'ConversationTurn',
        'TurnState'
    ])

__version__ = '1.0.0'
