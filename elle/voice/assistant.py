"""
Elle Voice Assistant

Integrates STT, NLP, TTS, and wake word detection into a complete voice interface.

Created: 2025-11-15
"""

import asyncio
from typing import Optional, Callable
from pathlib import Path

from elle.voice.whisper_stt import WhisperSTT
from elle.voice.llm_parser import LLMParser, ParsedCommand
from elle.voice.tts import TextToSpeech, VoiceGender
from elle.voice.wake_word import WakeWordDetector
from elle.voice_interface import VoiceSOPEditor


class VoiceAssistant:
    """
    Complete voice assistant for Elle Core

    Combines:
    - Wake word detection ("Hey Elle")
    - Speech-to-text (Whisper)
    - Natural language parsing (LLM + patterns)
    - Command execution (VoiceSOPEditor)
    - Text-to-speech responses
    """

    def __init__(
        self,
        sop_dir: str = "elle/sops",
        whisper_model: str = "tiny",
        tts_rate: int = 150,
        use_llm_parser: bool = True,
        verbose: bool = True
    ):
        """
        Initialize voice assistant

        Args:
            sop_dir: Directory for SOP files
            whisper_model: Whisper model size (tiny, base, small)
            tts_rate: TTS speech rate (words per minute)
            use_llm_parser: Use LLM for enhanced parsing
            verbose: Print debug information
        """
        self.sop_dir = sop_dir
        self.verbose = verbose

        # Initialize components
        self.stt = WhisperSTT(model=whisper_model)
        self.parser = LLMParser(use_llm=use_llm_parser)
        self.tts = TextToSpeech(rate=tts_rate, voice_gender=VoiceGender.FEMALE)
        self.wake_detector = WakeWordDetector()
        self.editor = VoiceSOPEditor(sop_dir=sop_dir)

        # Session state
        self.is_listening = False
        self.context = {}

    async def initialize(self):
        """Initialize assistant (loads SOPs, initializes RAG, etc.)"""
        if self.verbose:
            print("Initializing Elle Voice Assistant...")

        await self.editor.initialize_hololoom()

    async def process_voice_input(self, text: str) -> str:
        """
        Process voice input: parse and execute command

        Args:
            text: Transcribed voice text

        Returns:
            Response text to speak
        """
        if not text:
            return "I didn't hear that. Please try again."

        # Parse command
        command = self.parser.parse(text)

        if self.verbose:
            print(f"Parsed command: {command.command_type}")
            print(f"  Confidence: {command.confidence:.1%}")
            print(f"  Entity: {command.entity}")
            print(f"  Intent: {command.intent}")

        # Handle unknown commands
        if command.command_type == "unknown":
            return "I didn't understand that command. Try 'show bread SOP' or 'start baking bread'."

        # Process command through voice SOP editor
        try:
            response = await self.editor.process_voice_command(text)
            return response
        except Exception as e:
            return f"Error processing command: {str(e)}"

    async def listen_and_respond(self) -> bool:
        """
        Listen for voice input, process it, and respond

        Returns:
            True if command was processed, False if error
        """
        try:
            # Record audio
            audio_path = await self.stt.record_audio(duration=5.0)
            if not audio_path:
                await self.tts.speak("I couldn't hear you. Please try again.")
                return False

            # Transcribe
            text, metadata = await self.stt.transcribe_with_fallback(audio_path)
            if not text:
                await self.tts.speak("Transcription failed. Please try again.")
                return False

            if self.verbose:
                print(f"Transcribed: {text}")

            # Process command
            response = await self.process_voice_input(text)

            # Speak response
            await self.tts.speak(response)

            return True

        except Exception as e:
            print(f"✗ Error: {e}")
            await self.tts.speak("An error occurred. Please try again.")
            return False

    async def wake_word_loop(self):
        """
        Continuous listening for wake word

        Listens for "Hey Elle", then listens for command
        """
        print("\n" + "="*60)
        print("ELLE VOICE ASSISTANT")
        print("="*60)
        print("\n👂 Listening for wake word...")
        await self.tts.speak("Elle is ready. Say hey Elle to begin.")

        while True:
            try:
                # Listen for wake word
                async def record_and_transcribe():
                    audio_path = await self.stt.record_audio(duration=5.0)
                    if audio_path:
                        text, _ = await self.stt.transcribe_with_fallback(audio_path)
                        return text
                    return None

                # Wait for wake word
                detected = await self.wake_detector.listen_for_wake_word(
                    record_and_transcribe,
                    on_detected=self._on_wake_word
                )

                if detected:
                    # Listen for command
                    await self.tts.speak("I'm listening.")
                    success = await self.listen_and_respond()

                    if success:
                        await asyncio.sleep(1)  # Pause before next wake word
                        print("\n👂 Listening for wake word...")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                await self.tts.speak("Goodbye.")
                break
            except Exception as e:
                print(f"✗ Error: {e}")
                await asyncio.sleep(1)

    async def _on_wake_word(self, text: str):
        """Callback when wake word is detected"""
        # Extract command from wake word phrase
        command = self.wake_detector.extract_command(text)
        if command and len(command) > 2:
            # Full command was spoken with wake word
            if self.verbose:
                print(f"Full command heard: {command}")
            response = await self.process_voice_input(command)
            await self.tts.speak(response)

    async def interactive_mode(self):
        """
        Interactive mode: text input instead of voice

        Useful for testing without audio hardware
        """
        print("\n" + "="*60)
        print("ELLE VOICE ASSISTANT - INTERACTIVE MODE")
        print("="*60)
        print("\nType commands like:")
        print("  - 'show bread SOP'")
        print("  - 'update bread SOP: increase proofing to 50 minutes'")
        print("  - 'what's the biochar inoculation ratio?'")
        print("  - 'start baking bread'")
        print("  - 'Type 'quit' to exit'\n")

        await self.tts.speak("Elle is ready in interactive mode.")

        while True:
            try:
                text = input("You> ").strip()

                if not text:
                    continue

                if text.lower() in ["quit", "exit", "bye"]:
                    print("Goodbye!")
                    await self.tts.speak("Goodbye.")
                    break

                response = await self.process_voice_input(text)
                print(f"Elle> {response}")
                await self.tts.speak(response)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                await self.tts.speak("Goodbye.")
                break
            except Exception as e:
                print(f"✗ Error: {e}")

    async def run(self, mode: str = "voice"):
        """
        Run the assistant

        Args:
            mode: "voice" for wake word listening, "interactive" for text input
        """
        await self.initialize()

        if mode == "voice":
            await self.wake_word_loop()
        elif mode == "interactive":
            await self.interactive_mode()
        else:
            raise ValueError(f"Unknown mode: {mode}")


async def main():
    """Main entry point"""
    import sys

    # Determine mode from command line
    mode = "interactive"  # Default to interactive for easier testing

    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

    if mode not in ["voice", "interactive"]:
        print(f"Usage: python assistant.py [voice|interactive]")
        print(f"  voice: Listen for 'Hey Elle' wake word (requires microphone)")
        print(f"  interactive: Type commands (for testing)")
        sys.exit(1)

    # Create and run assistant
    assistant = VoiceAssistant(
        whisper_model="tiny",  # Use tiny for speed
        tts_rate=150,
        use_llm_parser=False,  # Pattern matching only for now
        verbose=True
    )

    await assistant.run(mode=mode)


if __name__ == "__main__":
    asyncio.run(main())
