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
- LanguageManager: Multi-language support with auto-detection (Phase 4)

Date: November 15, 2025
Updated: November 16, 2025 (Phase 3 - Personality Framework)
Updated: November 16, 2025 (Phase 4 - Multi-Language Support)
"""

# Personality module (always available)
from .personality import (
    PersonalityTraits,
    Personality,
    PersonalityManager,
    PersonalityType
)

# Language module (always available)
from .language import (
    LanguageCode,
    LanguageProfile,
    LanguageVariant,
    ConversationLanguageState,
    LanguageDetector,
    LanguageManager,
    create_language_manager,
    detect_language
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

    # Language (Phase 4)
    'LanguageCode',
    'LanguageProfile',
    'LanguageVariant',
    'ConversationLanguageState',
    'LanguageDetector',
    'LanguageManager',
    'create_language_manager',
    'detect_language',
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
