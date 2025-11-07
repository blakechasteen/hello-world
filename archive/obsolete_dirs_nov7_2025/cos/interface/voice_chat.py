"""
HoloLoom COS - Voice Chat with Emotional TTS

Uses BARK (or ElevenLabs) for natural, emotional voice responses.

Features:
- Natural pauses, breaths, laughs
- Emotional tone adaptation
- Conversational clarifications
- HITL verification via voice
- **DUAL-PROMPTING SYSTEM**: Separate script from vocal instructions

BARK advantages:
- Open source, runs locally
- Natural speech with emotions
- [laughs], [sighs], CAPITALIZATION for emphasis
- Multiple speakers

ElevenLabs advantages:
- Cloud-based (easier setup)
- Very high quality
- Faster generation
- Emotional controls

DUAL-PROMPTING ARCHITECTURE:
1. Content Script: What to say (business logic, facts, data)
2. Vocal Instructions: How to say it (pacing, emotion, pauses, emphasis)

Example:
    Script: "Your weekly revenue is $450, which is $50 below target."
    Vocal: "concerned, slow pace, sigh before delivery, emphasize numbers"

    Combined: "[sighs slowly] Your weekly revenue is FOUR FIFTY,
               which is... [pause] fifty dollars below target."
"""

import asyncio
from typing import Optional, Literal, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import sys
import io

# Fix Windows encoding for Unicode characters
if sys.platform == 'win32':
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cos.core.types import VerificationRequest, Event


@dataclass
class VocalInstructions:
    """
    Vocal delivery instructions separate from content.

    This enables:
    - Content generation (LLM/template)
    - Vocal direction (emotion engine)
    - Independent tuning of each
    """
    emotion: str = "neutral"  # happy, concerned, encouraging, patient, excited, sad
    pace: str = "normal"  # slow, normal, fast
    emphasis_words: list = None  # Words to CAPITALIZE
    pauses_after: list = None  # Words to add "..." after
    sounds_before: list = None  # [laughs], [sighs], [gasps] before script
    sounds_after: list = None  # Sounds after script
    volume: str = "normal"  # quiet, normal, loud
    pitch: str = "normal"  # low, normal, high (backend-dependent)

    def __post_init__(self):
        if self.emphasis_words is None:
            self.emphasis_words = []
        if self.pauses_after is None:
            self.pauses_after = []
        if self.sounds_before is None:
            self.sounds_before = []
        if self.sounds_after is None:
            self.sounds_after = []


@dataclass
class DualPrompt:
    """
    Dual-prompting system: Script + Vocal Instructions

    Workflow:
    1. Generate content script (business logic)
    2. Determine vocal instructions (emotion engine)
    3. Combine into final TTS input
    4. Generate audio
    """
    script: str  # What to say
    vocal: VocalInstructions  # How to say it

    def to_bark_text(self) -> str:
        """Combine script + vocal instructions for BARK"""
        text = self.script

        # Add emphasis (CAPITALIZATION)
        for word in self.vocal.emphasis_words:
            # Case-insensitive replacement
            import re
            text = re.sub(
                rf'\b{re.escape(word)}\b',
                word.upper(),
                text,
                flags=re.IGNORECASE
            )

        # Add pauses (ellipses)
        for word in self.vocal.pauses_after:
            import re
            text = re.sub(
                rf'\b{re.escape(word)}\b',
                f'{word}...',
                text,
                flags=re.IGNORECASE
            )

        # Add sounds before
        sounds_prefix = ' '.join(f'[{sound}]' for sound in self.vocal.sounds_before)
        if sounds_prefix:
            text = f"{sounds_prefix} {text}"

        # Add sounds after
        sounds_suffix = ' '.join(f'[{sound}]' for sound in self.vocal.sounds_after)
        if sounds_suffix:
            text = f"{text} {sounds_suffix}"

        # Adjust pacing with punctuation
        if self.vocal.pace == "slow":
            # Add commas for natural pauses
            text = text.replace('. ', '... ')
        elif self.vocal.pace == "fast":
            # Remove some commas
            text = text.replace(', ', ' ')

        return text

    def to_elevenlabs_params(self) -> Dict[str, Any]:
        """Convert vocal instructions to ElevenLabs parameters"""
        # Map emotion to stability/similarity
        emotion_map = {
            'neutral': {'stability': 0.5, 'similarity_boost': 0.75},
            'happy': {'stability': 0.3, 'similarity_boost': 0.85},
            'concerned': {'stability': 0.7, 'similarity_boost': 0.6},
            'encouraging': {'stability': 0.4, 'similarity_boost': 0.8},
            'patient': {'stability': 0.6, 'similarity_boost': 0.7},
            'excited': {'stability': 0.2, 'similarity_boost': 0.9},
            'sad': {'stability': 0.8, 'similarity_boost': 0.5},
        }

        params = emotion_map.get(self.vocal.emotion, emotion_map['neutral'])

        # Apply script with minimal formatting (ElevenLabs interprets naturally)
        text = self.script

        # Emphasis: repeat words slightly
        for word in self.vocal.emphasis_words:
            import re
            text = re.sub(
                rf'\b{re.escape(word)}\b',
                f'{word}... {word}',
                text,
                flags=re.IGNORECASE
            )

        params['text'] = text
        return params

# TTS Backend selection
TTS_BACKEND = "bark"  # or "elevenlabs" or "pyttsx3" (fallback)

# Try importing BARK
try:
    from bark import SAMPLE_RATE, generate_audio, preload_models
    import scipy.io.wavfile as wavfile
    BARK_AVAILABLE = True
except ImportError:
    BARK_AVAILABLE = False
    print("ℹ️  BARK not available. Install with: pip install git+https://github.com/suno-ai/bark.git")

# Try importing ElevenLabs
try:
    from elevenlabs import generate, play, set_api_key
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False
    print("ℹ️  ElevenLabs not available. Install with: pip install elevenlabs")

# Fallback: pyttsx3 (basic TTS, no emotions)
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False


class VoiceChat:
    """
    Natural voice conversation for clarifications.

    Emotional awareness:
    - Friendly for confirmations
    - Concerned for warnings
    - Encouraging for successes
    - Patient for clarifications
    """

    def __init__(
        self,
        backend: Literal["bark", "elevenlabs", "pyttsx3"] = "bark",
        voice: str = "v2/en_speaker_6"  # BARK voice ID
    ):
        self.backend = backend
        self.voice = voice

        # Initialize backend
        if backend == "bark" and BARK_AVAILABLE:
            # Preload BARK models
            preload_models()
            self.tts_func = self._bark_speak
        elif backend == "elevenlabs" and ELEVENLABS_AVAILABLE:
            # Set API key (from env)
            # set_api_key(os.getenv("ELEVENLABS_API_KEY"))
            self.tts_func = self._elevenlabs_speak
        elif PYTTSX3_AVAILABLE:
            # Fallback to basic TTS
            self.engine = pyttsx3.init()
            self.tts_func = self._pyttsx3_speak
        else:
            raise RuntimeError("No TTS backend available")

    async def _bark_speak(self, text: str, emotion: str = "neutral") -> bytes:
        """
        Generate speech with BARK using dual-prompting.

        BARK supports natural speech markers:
        - [laughs], [sighs], [gasps], [clears throat]
        - CAPITALIZATION for emphasis
        - ... for pauses
        - ♪ for singing

        Now uses DualPrompt for separation of content and delivery.
        """
        # Legacy support: if text is just a string, convert to DualPrompt
        if isinstance(text, str):
            text = self._legacy_to_dual_prompt(text, emotion)

        # If already a DualPrompt, use directly
        if isinstance(text, DualPrompt):
            bark_text = text.to_bark_text()
        else:
            bark_text = text

        # Generate audio
        audio_array = generate_audio(bark_text, history_prompt=self.voice)

        # Convert to bytes
        import io
        import scipy.io.wavfile as wavfile
        buffer = io.BytesIO()
        wavfile.write(buffer, SAMPLE_RATE, audio_array)
        return buffer.getvalue()

    def _legacy_to_dual_prompt(self, text: str, emotion: str) -> DualPrompt:
        """
        Convert legacy emotion-string approach to DualPrompt.

        Backwards compatibility helper.
        """
        # Map emotions to vocal instructions
        vocal_map = {
            'happy': VocalInstructions(
                emotion='happy',
                sounds_before=['laughs'],
                pace='normal'
            ),
            'concerned': VocalInstructions(
                emotion='concerned',
                sounds_before=['sighs'],
                pace='slow'
            ),
            'encouraging': VocalInstructions(
                emotion='encouraging',
                emphasis_words=['great', 'amazing', 'excellent'],
                pace='normal'
            ),
            'patient': VocalInstructions(
                emotion='patient',
                pace='slow',
                pauses_after=['clarify', 'understand']
            ),
            'excited': VocalInstructions(
                emotion='excited',
                sounds_before=['gasps'],
                emphasis_words=['wow', 'incredible', 'amazing'],
                pace='fast'
            ),
        }

        vocal = vocal_map.get(emotion, VocalInstructions(emotion='neutral'))
        return DualPrompt(script=text, vocal=vocal)

    async def _elevenlabs_speak(self, text: str, emotion: str = "neutral") -> bytes:
        """
        Generate speech with ElevenLabs.

        Emotions via voice settings.
        """
        # Map emotions to voice settings
        stability = 0.5
        similarity_boost = 0.75

        if emotion == "happy":
            stability = 0.3
            similarity_boost = 0.85
        elif emotion == "concerned":
            stability = 0.7
            similarity_boost = 0.6

        audio = generate(
            text=text,
            voice="Bella",  # Or configurable voice
            model="eleven_monolingual_v1",
            stability=stability,
            similarity_boost=similarity_boost
        )

        return audio

    async def _pyttsx3_speak(self, text: str, emotion: str = "neutral") -> None:
        """
        Fallback TTS (no audio bytes, just speak).

        No emotion support.
        """
        # Handle DualPrompt objects (convert to plain text)
        if isinstance(text, DualPrompt):
            # pyttsx3 doesn't support BARK markers, just use the script
            text_to_speak = text.script
        elif isinstance(text, str):
            text_to_speak = text
        else:
            # Fallback: convert to string
            text_to_speak = str(text)

        self.engine.say(text_to_speak)
        self.engine.runAndWait()
        return None

    async def speak(
        self,
        text,  # str or DualPrompt
        emotion: str = "neutral",
        save_path: Optional[str] = None
    ) -> Optional[bytes]:
        """
        Speak text with emotion.

        Args:
            text: What to say (str for legacy, DualPrompt for dual-prompting)
            emotion: 'neutral', 'happy', 'concerned', 'encouraging', 'patient' (legacy)
            save_path: Optional path to save audio file

        Returns:
            Audio bytes (if backend supports it)

        Examples:
            # Legacy approach (still works)
            await chat.speak("Hello!", emotion="happy")

            # Dual-prompting approach (recommended)
            prompt = DualPrompt(
                script="Your weekly revenue is $450, which is $50 below target.",
                vocal=VocalInstructions(
                    emotion="concerned",
                    pace="slow",
                    emphasis_words=["450", "50", "below"],
                    sounds_before=["sighs"],
                    pauses_after=["revenue", "target"]
                )
            )
            await chat.speak(prompt)
        """
        audio_bytes = await self.tts_func(text, emotion)

        if audio_bytes and save_path:
            Path(save_path).write_bytes(audio_bytes)

        return audio_bytes

    def create_business_prompt(
        self,
        script: str,
        metric_type: Literal["positive", "negative", "neutral", "warning"] = "neutral"
    ) -> DualPrompt:
        """
        Helper to create business-appropriate vocal instructions.

        Automatically determines vocal delivery based on business metric type.

        Args:
            script: The content to speak
            metric_type: Type of business metric
                - positive: Revenue up, goals met, high performance
                - negative: Revenue down, losses, failures
                - neutral: Reporting facts, no emotion needed
                - warning: Burnout risk, inventory low, attention needed

        Returns:
            DualPrompt with appropriate vocal instructions
        """
        import re

        # Extract numbers from script for emphasis
        numbers = re.findall(r'\$?\d+(?:,\d{3})*(?:\.\d+)?%?', script)

        # Extract key business terms
        positive_terms = ['profit', 'revenue', 'growth', 'success', 'target', 'achieved']
        negative_terms = ['loss', 'deficit', 'below', 'failed', 'shortage', 'missed']
        warning_terms = ['burnout', 'risk', 'critical', 'urgent', 'attention', 'warning']

        emphasis_words = numbers.copy()

        if metric_type == "positive":
            vocal = VocalInstructions(
                emotion='happy',
                sounds_before=['bright tone'],
                emphasis_words=emphasis_words + [w for w in positive_terms if w in script.lower()],
                pace='normal'
            )
        elif metric_type == "negative":
            vocal = VocalInstructions(
                emotion='concerned',
                sounds_before=['sighs'],
                emphasis_words=emphasis_words + [w for w in negative_terms if w in script.lower()],
                pace='slow',
                pauses_after=['however', 'unfortunately', 'below']
            )
        elif metric_type == "warning":
            vocal = VocalInstructions(
                emotion='concerned',
                sounds_before=['clears throat'],
                emphasis_words=emphasis_words + [w for w in warning_terms if w in script.lower()],
                pace='slow',
                pauses_after=['attention', 'risk', 'critical']
            )
        else:  # neutral
            vocal = VocalInstructions(
                emotion='neutral',
                emphasis_words=numbers,
                pace='normal'
            )

        return DualPrompt(script=script, vocal=vocal)

    async def clarify_verification(
        self,
        verification: VerificationRequest,
        voice_callback: Optional[callable] = None
    ) -> bool:
        """
        Voice conversation to clarify uncertain input.

        Workflow:
        1. Explain what we understood
        2. Ask for confirmation
        3. Provide options if needed
        4. Get voice response
        5. Confirm final understanding
        """
        event = verification.event

        # 1. Explain with concern
        explanation = self._build_explanation(event, verification.reason)
        await self.speak(explanation, emotion="patient")

        # 2. Ask for confirmation
        question = "Is that correct?"
        await self.speak(question, emotion="neutral")

        # 3. Get voice response (from callback)
        if voice_callback:
            response_text = await voice_callback()
        else:
            # Fallback to text input
            response_text = input("Your response: ").strip()

        # 4. Parse response
        is_confirmed = self._parse_confirmation(response_text)

        # 5. Acknowledge
        if is_confirmed:
            await self.speak("Perfect! Logging that now.", emotion="happy")
            return True
        else:
            await self.speak(
                "No problem. [clears throat] Let's try again.",
                emotion="patient"
            )
            return False

    def _build_explanation(self, event: Event, reason: str) -> str:
        """Build natural explanation of what we understood"""
        if event.type.value == "task":
            hours = event.quantity or 0
            task = event.item or "work"
            return (
                f"I heard that you worked on {task} for {hours} hours. "
                f"However, I'm only {event.confidence:.0%} confident about that. "
                f"{reason}"
            )
        elif event.type.value == "sale":
            amount = event.amount or 0
            item = event.item or "items"
            return (
                f"It sounds like you sold {item} for ${amount:.2f}. "
                f"But I'm {event.confidence:.0%} sure. "
                f"{reason}"
            )
        elif event.type.value == "purchase":
            amount = abs(event.amount) if event.amount else 0
            item = event.item or "something"
            return (
                f"I think you bought {item} for ${amount:.2f}. "
                f"My confidence is {event.confidence:.0%}. "
                f"{reason}"
            )
        else:
            return (
                f"I heard: '{event.raw_input}'. "
                f"But I'm only {event.confidence:.0%} confident. "
                f"{reason}"
            )

    def _parse_confirmation(self, response: str) -> bool:
        """Parse yes/no from natural language"""
        response_lower = response.lower()

        # Positive indicators
        if any(word in response_lower for word in ['yes', 'yeah', 'yep', 'correct', 'right', 'exactly']):
            return True

        # Negative indicators
        if any(word in response_lower for word in ['no', 'nope', 'wrong', 'incorrect', 'not']):
            return False

        # Ambiguous - ask again
        return False


async def demo():
    """Demo voice chat with dual-prompting system"""
    import sys
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')

    print("\n" + "="*60)
    print("HoloLoom COS - Dual-Prompting TTS Demo")
    print("="*60)

    # Determine available backend
    if BARK_AVAILABLE:
        backend = "bark"
        print("\n✓ Using BARK (natural, emotional speech)")
    elif ELEVENLABS_AVAILABLE:
        backend = "elevenlabs"
        print("\n✓ Using ElevenLabs (high quality)")
    elif PYTTSX3_AVAILABLE:
        backend = "pyttsx3"
        print("\n⚠️  Using pyttsx3 (basic fallback, no emotions)")
    else:
        print("\n❌ No TTS backend available!")
        print("Install one of:")
        print("  - BARK: pip install git+https://github.com/suno-ai/bark.git")
        print("  - ElevenLabs: pip install elevenlabs")
        print("  - pyttsx3: pip install pyttsx3")
        return

    chat = VoiceChat(backend=backend)

    # TEST 1: Legacy approach (still works)
    print("\n\nTEST 1: Legacy Emotion-String Approach")
    print("-"*60)
    print("This shows backwards compatibility with the old API.")

    emotions = [
        ("neutral", "Hello! I'm your COS voice assistant."),
        ("happy", "Great job on hitting your revenue target!"),
        ("concerned", "Your hours this week are quite high."),
    ]

    for emotion, text in emotions:
        print(f"\n[{emotion.upper()}]")
        print(f"  Script: {text}")
        audio = await chat.speak(text, emotion=emotion)
        if audio:
            print(f"  Audio: {len(audio)} bytes generated")

    # TEST 2: Dual-prompting approach
    print("\n\nTEST 2: Dual-Prompting System (Script + Vocal Instructions)")
    print("-"*60)
    print("This shows the new separation of WHAT to say from HOW to say it.")

    # Example 1: Revenue report (negative metric)
    script1 = "Your weekly revenue is $450, which is $50 below target."
    prompt1 = DualPrompt(
        script=script1,
        vocal=VocalInstructions(
            emotion="concerned",
            pace="slow",
            emphasis_words=["450", "50", "below", "target"],
            sounds_before=["sighs"],
            pauses_after=["revenue"]
        )
    )

    print(f"\nExample 1: Revenue Report")
    print(f"  Script: {script1}")
    print(f"  Vocal: concerned, slow, sighs before, emphasize numbers")
    if backend == "bark":
        print(f"  BARK text: {prompt1.to_bark_text()}")
    audio = await chat.speak(prompt1)
    if audio:
        print(f"  Audio: {len(audio)} bytes generated")

    # Example 2: Positive achievement
    script2 = "Amazing! You've achieved your Week 4 goal with revenue of $520."
    prompt2 = DualPrompt(
        script=script2,
        vocal=VocalInstructions(
            emotion="happy",
            pace="normal",
            emphasis_words=["Amazing", "520", "achieved"],
            sounds_before=["laughs"],
            sounds_after=["applause"]
        )
    )

    print(f"\nExample 2: Positive Achievement")
    print(f"  Script: {script2}")
    print(f"  Vocal: happy, normal pace, laughs before, applause after")
    if backend == "bark":
        print(f"  BARK text: {prompt2.to_bark_text()}")
    audio = await chat.speak(prompt2)
    if audio:
        print(f"  Audio: {len(audio)} bytes generated")

    # Example 3: Burnout warning
    script3 = "Warning: You've worked 52 hours this week. This is unsustainable and risks burnout."
    prompt3 = DualPrompt(
        script=script3,
        vocal=VocalInstructions(
            emotion="concerned",
            pace="slow",
            emphasis_words=["Warning", "52", "unsustainable", "burnout"],
            sounds_before=["clears throat"],
            pauses_after=["week", "unsustainable"]
        )
    )

    print(f"\nExample 3: Burnout Warning")
    print(f"  Script: {script3}")
    print(f"  Vocal: concerned, slow, clears throat, emphasize key warnings")
    if backend == "bark":
        print(f"  BARK text: {prompt3.to_bark_text()}")
    audio = await chat.speak(prompt3)
    if audio:
        print(f"  Audio: {len(audio)} bytes generated")

    # TEST 3: Business prompt helper
    print("\n\nTEST 3: Business Prompt Helper (Auto-Vocal Instructions)")
    print("-"*60)
    print("The create_business_prompt() method automatically determines vocal delivery.")

    business_metrics = [
        ("Your profit margin improved to 78%.", "positive"),
        ("Revenue dropped 15% compared to last week.", "negative"),
        ("Inventory critical: Only 2 days of flour remaining.", "warning"),
        ("Today's summary: 8 hours worked, $180 revenue.", "neutral"),
    ]

    for script, metric_type in business_metrics:
        prompt = chat.create_business_prompt(script, metric_type)
        print(f"\n[{metric_type.upper()}]")
        print(f"  Script: {script}")
        print(f"  Auto-detected vocal: {prompt.vocal.emotion}, pace={prompt.vocal.pace}")
        print(f"  Emphasis: {prompt.vocal.emphasis_words}")
        if backend == "bark":
            print(f"  BARK text: {prompt.to_bark_text()}")
        audio = await chat.speak(prompt)
        if audio:
            print(f"  Audio: {len(audio)} bytes generated")

    # TEST 4: Complex multi-instruction example
    print("\n\nTEST 4: Complex Multi-Instruction Example")
    print("-"*60)
    print("Shows full control over pacing, emphasis, pauses, and sounds.")

    script4 = ("Let's review your day. You worked 7 hours on bread production, "
               "sold 12 loaves for $72, and spent $18 on materials. "
               "Your profit was $54, giving you an excellent hourly rate of $7.71.")

    prompt4 = DualPrompt(
        script=script4,
        vocal=VocalInstructions(
            emotion="patient",
            pace="slow",
            emphasis_words=["7", "12", "72", "18", "54", "7.71", "excellent"],
            pauses_after=["day", "production", "loaves", "materials", "profit"],
            sounds_before=["clears throat"],
        )
    )

    print(f"  Script: {script4}")
    print(f"  Vocal: patient tone, slow pace, pause after each metric")
    print(f"  Emphasis: all numbers + 'excellent'")
    if backend == "bark":
        print(f"  BARK text: {prompt4.to_bark_text()}")
    audio = await chat.speak(prompt4)
    if audio:
        print(f"  Audio: {len(audio)} bytes generated")

    print("\n\n" + "="*60)
    print("✅ Dual-Prompting Demo Complete")
    print("="*60)

    print("\nKey Takeaways:")
    print("1. Legacy API still works (backwards compatible)")
    print("2. Dual-prompting separates WHAT (script) from HOW (vocal)")
    print("3. VocalInstructions provides fine-grained control:")
    print("   - emotion, pace, volume, pitch")
    print("   - emphasis_words, pauses_after")
    print("   - sounds_before, sounds_after")
    print("4. Business helper auto-generates appropriate delivery")
    print("5. Same DualPrompt works across BARK and ElevenLabs backends")

    if backend == "bark":
        print("\nNext Steps:")
        print("- Save audio files with save_path parameter")
        print("- Integrate with voice_input.py for HITL verification")
        print("- Use in daily_review.py for morning/evening workflows")


if __name__ == '__main__':
    asyncio.run(demo())
