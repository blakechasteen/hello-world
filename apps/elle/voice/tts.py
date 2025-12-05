"""
Text-to-Speech for Elle Core

Provides voice responses using pyttsx3 (cross-platform, offline).

Created: 2025-11-15
"""

import asyncio
import queue
from typing import Optional, List
from enum import Enum


class VoiceGender(Enum):
    """Voice gender preference"""
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


class TextToSpeech:
    """
    Text-to-Speech using pyttsx3

    Supports:
    - Multiple voice options
    - Rate and volume control
    - Offline operation
    - Queue management for long responses
    """

    def __init__(
        self,
        rate: int = 150,
        volume: float = 0.9,
        voice_gender: VoiceGender = VoiceGender.FEMALE
    ):
        """
        Initialize TTS

        Args:
            rate: Speech rate (words per minute, 30-300)
            volume: Volume (0.0-1.0)
            voice_gender: Preferred voice gender
        """
        self.rate = rate
        self.volume = volume
        self.voice_gender = voice_gender

        self.engine = None
        self.pyttsx3_available = self._initialize_engine()

        if not self.pyttsx3_available:
            print("⚠ pyttsx3 not available. Install with: pip install pyttsx3")
            print("  Voice responses will be printed to console instead.")

    def _initialize_engine(self) -> bool:
        """Initialize pyttsx3 engine"""
        try:
            import pyttsx3

            self.engine = pyttsx3.init()

            # Set rate
            self.engine.setProperty("rate", self.rate)

            # Set volume
            self.engine.setProperty("volume", self.volume)

            # Set voice
            voices = self.engine.getProperty("voices")
            if voices:
                # Try to find voice matching preference
                selected_voice = None

                for voice in voices:
                    voice_gender = voice.name.lower()
                    if self.voice_gender.value in voice_gender:
                        selected_voice = voice.id
                        break

                # Fallback to first available voice
                if not selected_voice:
                    selected_voice = voices[0].id

                self.engine.setProperty("voice", selected_voice)

            return True

        except ImportError:
            return False
        except Exception as e:
            print(f"✗ Failed to initialize TTS: {e}")
            return False

    async def speak(self, text: str, wait: bool = True) -> bool:
        """
        Speak text

        Args:
            text: Text to speak
            wait: Whether to wait for completion

        Returns:
            True if successful, False otherwise
        """
        if not self.pyttsx3_available or not self.engine:
            # Fallback: print to console
            print(f"[VOICE]: {text}")
            return True

        try:
            # Clear any previous content
            self.engine.endLoopIteration()

            # Queue text
            self.engine.say(text)

            # Wait for completion if requested
            if wait:
                self.engine.runAndWait()

            return True

        except Exception as e:
            print(f"✗ TTS failed: {e}")
            return False

    async def speak_async(
        self,
        text: str,
        callback: Optional[callable] = None
    ) -> asyncio.Task:
        """
        Speak asynchronously

        Args:
            text: Text to speak
            callback: Optional callback when speech completes

        Returns:
            Asyncio task that can be awaited or cancelled
        """
        async def _speak_task():
            result = await self.speak(text, wait=True)
            if callback:
                callback(result)
            return result

        return asyncio.create_task(_speak_task())

    async def speak_queue(self, texts: List[str]) -> bool:
        """
        Speak multiple texts in sequence

        Args:
            texts: List of texts to speak

        Returns:
            True if all successful
        """
        for text in texts:
            if not await self.speak(text):
                return False
        return True

    def set_rate(self, rate: int):
        """Set speech rate (words per minute)"""
        self.rate = max(30, min(300, rate))
        if self.engine:
            self.engine.setProperty("rate", self.rate)

    def set_volume(self, volume: float):
        """Set volume (0.0-1.0)"""
        self.volume = max(0.0, min(1.0, volume))
        if self.engine:
            self.engine.setProperty("volume", self.volume)

    def get_voices(self) -> List[dict]:
        """Get available voices"""
        if not self.engine:
            return []

        try:
            voices = self.engine.getProperty("voices")
            return [
                {
                    "name": voice.name,
                    "id": voice.id,
                    "languages": getattr(voice, "languages", [])
                }
                for voice in voices
            ]
        except Exception:
            return []

    async def demo(self):
        """Demo TTS capabilities"""
        print("\n" + "="*60)
        print("TEXT-TO-SPEECH DEMO")
        print("="*60)

        # Show available voices
        voices = self.get_voices()
        print("\nAvailable voices:")
        for i, voice in enumerate(voices):
            print(f"  {i+1}. {voice['name']}")

        # Speak sample text
        print("\nSpeaking sample text...")
        await self.speak("Hello! I am Elle, your farm and kitchen assistant. Ready to help with your operations.")

        # Variable rate demo
        print("\nDemonstrating different speech rates:")
        rates = [80, 150, 200]
        for rate in rates:
            self.set_rate(rate)
            print(f"  Rate: {rate} words per minute")
            await self.speak(f"This is {rate} words per minute.", wait=True)

        # Reset to default
        self.set_rate(150)


async def demo_tts():
    """Demo: Text-to-Speech"""
    tts = TextToSpeech(rate=150, volume=0.9, voice_gender=VoiceGender.FEMALE)
    await tts.demo()


if __name__ == "__main__":
    asyncio.run(demo_tts())
