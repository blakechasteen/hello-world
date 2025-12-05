"""
Elle Voice Module

Complete voice interface for Elle Core with:
- Speech-to-text (Whisper)
- Natural language parsing (LLM + patterns)
- Text-to-speech (pyttsx3)
- Wake word detection
- Full voice assistant

Created: 2025-11-15
"""

from elle.voice.whisper_stt import WhisperSTT
from elle.voice.llm_parser import LLMParser, ParsedCommand, CommandVariationGenerator
from elle.voice.tts import TextToSpeech, VoiceGender
from elle.voice.wake_word import WakeWordDetector
from elle.voice.assistant import VoiceAssistant

__version__ = "0.1.0-alpha"
__all__ = [
    "WhisperSTT",
    "LLMParser",
    "ParsedCommand",
    "CommandVariationGenerator",
    "TextToSpeech",
    "VoiceGender",
    "WakeWordDetector",
    "VoiceAssistant",
]
