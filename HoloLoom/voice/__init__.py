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

Date: November 15, 2025
"""

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

__all__ = [
    'VoiceAgent',
    'TTSProvider',
    'OpenAITTS',
    'TTSManager',
    'VoiceActivityDetector',
    'TurnTakingManager',
    'ConversationMemory',
    'ConversationTurn',
    'TurnState'
]

__version__ = '1.0.0'
