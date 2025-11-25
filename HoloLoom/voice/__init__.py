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
- EmotionBridge: Python ↔ Node.js bridge for 110/100 emotional intelligence (Phase 5)

Date: November 15, 2025
Updated: November 16, 2025 (Phase 3 - Personality Framework)
Updated: November 16, 2025 (Phase 4 - Multi-Language Support)
Updated: November 22, 2025 (Phase 5 - Emotion Bridge Integration)
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

# Emotion bridge module (Phase 5 - optional Node.js dependency)
try:
    from .emotion_bridge import (
        EmotionBridge,
        EmotionBridgeConfig,
        EmotionResult,
        EmotionalInput,
        FusionStrategy,
        MetaMode,
        PlanningStrategy,
        enhance_voice_agent_with_emotions
    )
    EMOTION_BRIDGE_AVAILABLE = True
except ImportError:
    EMOTION_BRIDGE_AVAILABLE = False

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

if EMOTION_BRIDGE_AVAILABLE:
    __all__.extend([
        # Emotion Bridge (Phase 5)
        'EmotionBridge',
        'EmotionBridgeConfig',
        'EmotionResult',
        'EmotionalInput',
        'FusionStrategy',
        'MetaMode',
        'PlanningStrategy',
        'enhance_voice_agent_with_emotions'
    ])

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
