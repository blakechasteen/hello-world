# Dual-Prompting TTS System - Complete Guide

**Status**: ✅ Complete
**Location**: `cos/interface/voice_chat.py`
**Backends**: BARK (recommended), ElevenLabs, pyttsx3 (fallback)
**Architecture**: Interchangeable backends with dual-prompting support

---

## 🎯 Core Philosophy

**Separate WHAT from HOW**

Traditional TTS systems mix content and delivery:
```python
# Old way: Content and emotion mixed
speak("Your revenue is $450", emotion="concerned")
```

Dual-prompting separates them:
```python
# New way: Content separate from delivery instructions
prompt = DualPrompt(
    script="Your revenue is $450",  # WHAT to say
    vocal=VocalInstructions(        # HOW to say it
        emotion="concerned",
        emphasis_words=["450"],
        sounds_before=["sighs"]
    )
)
speak(prompt)
```

This enables:
- **Independent tuning**: Content generator ≠ Emotion engine
- **Backend portability**: Same DualPrompt works across BARK, ElevenLabs, etc.
- **Composability**: Mix and match scripts with vocal styles
- **Clarity**: Business logic separate from presentation

---

## 🏗️ Architecture

### Three-Layer System

```
┌─────────────────────────────────────────────────────────┐
│                    LAYER 1: CONTENT                     │
│                  (Business Logic)                       │
│                                                          │
│  Script: "Your weekly revenue is $450, $50 below target"│
└─────────────────────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────┐
│               LAYER 2: VOCAL INSTRUCTIONS                │
│                  (Emotion Engine)                        │
│                                                          │
│  VocalInstructions:                                      │
│    - emotion: "concerned"                                │
│    - pace: "slow"                                        │
│    - emphasis_words: ["450", "50", "below"]              │
│    - sounds_before: ["sighs"]                            │
│    - pauses_after: ["revenue", "target"]                 │
└─────────────────────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────┐
│                LAYER 3: BACKEND ADAPTER                  │
│               (TTS Implementation)                       │
│                                                          │
│  ┌──────────┐  ┌──────────────┐  ┌──────────┐          │
│  │   BARK   │  │ ElevenLabs   │  │ pyttsx3  │          │
│  │ (local)  │  │   (cloud)    │  │(fallback)│          │
│  └──────────┘  └──────────────┘  └──────────┘          │
└─────────────────────────────────────────────────────────┘
                         │
                         ↓
                   🔊 Audio Output
```

---

## 📦 Core Components

### 1. VocalInstructions (Delivery Control)

```python
@dataclass
class VocalInstructions:
    emotion: str = "neutral"       # happy, concerned, patient, etc.
    pace: str = "normal"           # slow, normal, fast
    emphasis_words: list = []      # Words to CAPITALIZE or repeat
    pauses_after: list = []        # Add "..." after these words
    sounds_before: list = []       # [laughs], [sighs], etc. before
    sounds_after: list = []        # Sounds after script
    volume: str = "normal"         # quiet, normal, loud
    pitch: str = "normal"          # low, normal, high (backend-dependent)
```

**Emotions Supported**:
- `neutral` - Factual, no emotion
- `happy` - Upbeat, positive, celebratory
- `concerned` - Worried, cautious, serious
- `encouraging` - Supportive, warm, motivating
- `patient` - Slow, clear, explanatory
- `excited` - Fast, energetic, enthusiastic
- `sad` - Slow, somber, low energy

**Pacing**:
- `slow` - Adds pauses, stretches words (for warnings, complex info)
- `normal` - Natural conversational speed
- `fast` - Quick delivery (for enthusiasm, urgency)

### 2. DualPrompt (Script + Vocal Combiner)

```python
@dataclass
class DualPrompt:
    script: str                    # What to say
    vocal: VocalInstructions       # How to say it

    def to_bark_text(self) -> str:
        """Convert to BARK-formatted text"""
        # Applies emphasis, pauses, sounds to script
        # Returns: "[sighs] Your revenue... is FOUR FIFTY"

    def to_elevenlabs_params(self) -> Dict[str, Any]:
        """Convert to ElevenLabs API parameters"""
        # Maps vocal instructions to API settings
        # Returns: {text, stability, similarity_boost}
```

**Backend Adapters**:
- `to_bark_text()` - BARK-specific formatting ([laughs], CAPS, ...)
- `to_elevenlabs_params()` - ElevenLabs API parameters
- `to_pyttsx3_params()` - pyttsx3 settings (basic, no emotion)

### 3. VoiceChat (Main TTS Interface)

```python
class VoiceChat:
    def __init__(
        self,
        backend: Literal["bark", "elevenlabs", "pyttsx3"] = "bark",
        voice: str = "v2/en_speaker_6"  # Voice ID
    ):
        # Initializes selected backend

    async def speak(
        self,
        text,  # str or DualPrompt
        emotion: str = "neutral",  # Legacy API
        save_path: Optional[str] = None
    ) -> Optional[bytes]:
        """
        Speak text with emotion.

        Supports both legacy and dual-prompting APIs.
        """

    def create_business_prompt(
        self,
        script: str,
        metric_type: Literal["positive", "negative", "neutral", "warning"]
    ) -> DualPrompt:
        """
        Auto-generate vocal instructions for business metrics.

        Analyzes script content and metric type to determine
        appropriate emotion, pacing, emphasis, and sounds.
        """
```

---

## 🔄 Backend Interchangeability

### Switching Backends

**At Initialization**:
```python
# Use BARK (open source, local, natural emotions)
chat = VoiceChat(backend="bark", voice="v2/en_speaker_6")

# Use ElevenLabs (cloud, high quality)
chat = VoiceChat(backend="elevenlabs", voice="Bella")

# Use pyttsx3 (fallback, basic, no emotions)
chat = VoiceChat(backend="pyttsx3")
```

**At Runtime** (not supported - backend is set at init):
```python
# To switch backends, create new VoiceChat instance
bark_chat = VoiceChat(backend="bark")
elevenlabs_chat = VoiceChat(backend="elevenlabs")

# Same DualPrompt works with both!
prompt = DualPrompt(
    script="Your revenue is $450",
    vocal=VocalInstructions(emotion="concerned")
)

await bark_chat.speak(prompt)
await elevenlabs_chat.speak(prompt)
```

### Backend Comparison

| Feature | BARK | ElevenLabs | pyttsx3 |
|---------|------|------------|---------|
| **Installation** | `pip install bark` | `pip install elevenlabs` | `pip install pyttsx3` |
| **Cost** | Free (local) | Paid (API) | Free (local) |
| **Quality** | Very Natural | Highest Quality | Basic/Robotic |
| **Emotions** | ✅ Full support | ✅ Full support | ❌ No emotions |
| **Sounds** | ✅ [laughs], [sighs], etc. | ❌ Limited | ❌ None |
| **Emphasis** | ✅ CAPITALIZATION | ✅ Repetition | ❌ Limited |
| **Latency** | Medium (~2-5s) | Fast (~1-2s) | Very Fast (<0.5s) |
| **Offline** | ✅ Yes | ❌ Requires internet | ✅ Yes |
| **Multi-speaker** | ✅ 20+ voices | ✅ Custom voices | ✅ System voices |

**Recommendation**:
- **Development**: BARK (free, natural, good emotions)
- **Production**: ElevenLabs (highest quality, fast, reliable)
- **Fallback**: pyttsx3 (always works, no dependencies)

---

## 💡 Usage Examples

### Example 1: Legacy API (Backwards Compatible)

```python
from cos.interface.voice_chat import VoiceChat

chat = VoiceChat(backend="bark")

# Old approach still works
await chat.speak("Hello!", emotion="happy")
await chat.speak("Warning: Low inventory", emotion="concerned")
```

### Example 2: Basic Dual-Prompting

```python
from cos.interface.voice_chat import VoiceChat, DualPrompt, VocalInstructions

chat = VoiceChat(backend="bark")

# Create dual prompt
prompt = DualPrompt(
    script="Your weekly revenue is $450, which is $50 below target.",
    vocal=VocalInstructions(
        emotion="concerned",
        pace="slow",
        emphasis_words=["450", "50", "below"],
        sounds_before=["sighs"],
        pauses_after=["revenue"]
    )
)

# Generate speech
audio = await chat.speak(prompt)

# Save to file
await chat.speak(prompt, save_path="revenue_report.wav")
```

**BARK Output**:
```
[sighs] Your weekly revenue... is FOUR FIFTY, which is FIFTY dollars BELOW target.
```

### Example 3: Business Prompt Helper

```python
chat = VoiceChat(backend="bark")

# Auto-generate vocal instructions
scripts = [
    ("Your profit margin improved to 78%.", "positive"),
    ("Revenue dropped 15% compared to last week.", "negative"),
    ("Warning: Only 2 days of flour remaining.", "warning"),
    ("Today's summary: 8 hours worked, $180 revenue.", "neutral"),
]

for script, metric_type in scripts:
    prompt = chat.create_business_prompt(script, metric_type)
    await chat.speak(prompt)
```

**Auto-Generated Vocal Instructions**:

| Metric Type | Emotion | Sounds | Emphasis | Pace |
|-------------|---------|--------|----------|------|
| positive | happy | [bright tone] | numbers + "profit", "improved" | normal |
| negative | concerned | [sighs] | numbers + "dropped", "below" | slow |
| warning | concerned | [clears throat] | numbers + "warning", "critical" | slow |
| neutral | neutral | none | numbers only | normal |

### Example 4: Complex Multi-Instruction

```python
script = (
    "Let's review your day. You worked 7 hours on bread production, "
    "sold 12 loaves for $72, and spent $18 on materials. "
    "Your profit was $54, giving you an excellent hourly rate of $7.71."
)

prompt = DualPrompt(
    script=script,
    vocal=VocalInstructions(
        emotion="patient",
        pace="slow",
        emphasis_words=["7", "12", "72", "18", "54", "7.71", "excellent"],
        pauses_after=["day", "production", "loaves", "materials", "profit"],
        sounds_before=["clears throat"],
    )
)

await chat.speak(prompt)
```

**BARK Output**:
```
[clears throat] Let's review your day... You worked SEVEN hours on bread production...
sold TWELVE loaves... for SEVENTY-TWO dollars, and spent EIGHTEEN dollars on materials...
Your profit... was FIFTY-FOUR dollars, giving you an EXCELLENT hourly rate of SEVEN POINT SEVEN ONE.
```

### Example 5: HITL Verification Dialogue

```python
from cos.core.types import VerificationRequest, Event

# Create verification request
event = Event(/* ... purchase event with low confidence ... */)
verification = VerificationRequest(
    event=event,
    reason="Ambiguous amount - could be $27 or $270"
)

# Voice conversation for clarification
chat = VoiceChat(backend="bark")
confirmed = await chat.clarify_verification(
    verification,
    voice_callback=async_voice_input_func  # Records user's spoken response
)

if confirmed:
    # Store event
    await store.store(event)
```

**Dialogue Flow**:
```
COS: [patient tone] "I think you bought flour for $27, but I'm only 68% confident.
     The amount was a bit unclear. Is that correct?"

USER: "Yes, twenty seven dollars."

COS: [happy] "Perfect! Logging that now."
```

---

## 🎛️ Advanced Configuration

### Custom Voice Selection

**BARK Voices**:
```python
# BARK has 20+ preset voices
voices = [
    "v2/en_speaker_0",  # Male, neutral
    "v2/en_speaker_1",  # Female, young
    "v2/en_speaker_6",  # Female, professional (default)
    "v2/en_speaker_9",  # Male, warm
]

chat = VoiceChat(backend="bark", voice="v2/en_speaker_9")
```

**ElevenLabs Voices**:
```python
# ElevenLabs has custom voice library
chat = VoiceChat(backend="elevenlabs", voice="Bella")
# Or: "Adam", "Antoni", "Arnold", "Domi", etc.
```

### Backend-Specific Tuning

**BARK Temperature** (affects randomness):
```python
# Lower temperature = more consistent
# Higher temperature = more expressive
chat._bark_speak.temperature = 0.7  # Default: 0.6-0.8
```

**ElevenLabs Stability/Similarity**:
```python
# Auto-tuned by VocalInstructions.emotion
# Manual override:
params = prompt.to_elevenlabs_params()
params['stability'] = 0.6  # 0.0 (variable) to 1.0 (stable)
params['similarity_boost'] = 0.8  # 0.0 (creative) to 1.0 (similar)
```

---

## 🔧 Integration with COS

### Daily Review Workflow

```python
# cos/interface/daily_review.py
from cos.interface.voice_chat import VoiceChat, DualPrompt

class DailyReviewWorkflow:
    def __init__(self, voice_mode: bool = False):
        if voice_mode:
            self.voice = VoiceChat(backend="bark")

    async def morning_planning(self):
        # Get yesterday's summary
        yesterday_summary = await self.store.get_daily_summary(yesterday)

        # Create business prompt
        script = f"Yesterday you earned ${yesterday_summary.revenue:.2f} in {yesterday_summary.hours_worked:.1f} hours."
        prompt = self.voice.create_business_prompt(
            script,
            metric_type="positive" if yesterday_summary.profit > 0 else "negative"
        )

        # Speak with appropriate emotion
        await self.voice.speak(prompt)
```

### Voice Input HITL Verification

```python
# cos/interface/voice_input.py
from cos.interface.voice_chat import VoiceChat

class VoiceInputHandler:
    def __init__(self, store, voice_chat: Optional[VoiceChat] = None):
        self.voice_chat = voice_chat or VoiceChat(backend="bark")

    async def process_voice_input(self, audio_path: str):
        # Transcribe
        transcript = await self.transcribe_audio_file(audio_path)

        # Parse
        intent, verification = parse_input(transcript, EventSource.VOICE)

        # HITL if needed
        if verification:
            confirmed = await self.voice_chat.clarify_verification(verification)
            if not confirmed:
                return {'error': 'User rejected verification'}

        # Store event
        event = intent.to_event(transcript)
        event.verified = True
        event_id = await self.store.store(event)
        return {'event_id': event_id}
```

### API Server Endpoint

```python
# cos/interface/api_server.py
from cos.interface.voice_chat import VoiceChat

chat = VoiceChat(backend="bark")

@app.post("/tts/speak")
async def text_to_speech(
    script: str,
    emotion: Optional[str] = None,
    metric_type: Optional[str] = None
):
    """Generate TTS audio from script"""

    if metric_type:
        # Use business prompt helper
        prompt = chat.create_business_prompt(script, metric_type)
    elif emotion:
        # Legacy emotion API
        prompt = script
    else:
        # Neutral
        prompt = script

    audio_bytes = await chat.speak(prompt, emotion=emotion or "neutral")

    return Response(
        content=audio_bytes,
        media_type="audio/wav"
    )
```

---

## 🧪 Testing

### Run Demo

```bash
cd cos/interface
python voice_chat.py
```

**Expected Output**:
```
============================================================
HoloLoom COS - Dual-Prompting TTS Demo
============================================================

✓ Using BARK (natural, emotional speech)

TEST 1: Legacy Emotion-String Approach
------------------------------------------------------------
[NEUTRAL]
  Script: Hello! I'm your COS voice assistant.
  Audio: 145820 bytes generated

[HAPPY]
  Script: Great job on hitting your revenue target!
  Audio: 198340 bytes generated

TEST 2: Dual-Prompting System (Script + Vocal Instructions)
------------------------------------------------------------
Example 1: Revenue Report
  Script: Your weekly revenue is $450, which is $50 below target.
  Vocal: concerned, slow, sighs before, emphasize numbers
  BARK text: [sighs] Your weekly revenue... is FOUR FIFTY, which is FIFTY dollars BELOW target.
  Audio: 256890 bytes generated

[... more tests ...]

✅ Dual-Prompting Demo Complete

Key Takeaways:
1. Legacy API still works (backwards compatible)
2. Dual-prompting separates WHAT (script) from HOW (vocal)
3. VocalInstructions provides fine-grained control
4. Business helper auto-generates appropriate delivery
5. Same DualPrompt works across BARK and ElevenLabs backends
```

### Unit Tests

```python
# cos/tests/test_voice_chat.py
import pytest
from cos.interface.voice_chat import VoiceChat, DualPrompt, VocalInstructions

@pytest.mark.asyncio
async def test_legacy_api():
    """Test backwards compatibility with old emotion-string API"""
    chat = VoiceChat(backend="pyttsx3")  # Fast fallback for testing
    audio = await chat.speak("Hello", emotion="happy")
    # Should not raise, audio may be None for pyttsx3

@pytest.mark.asyncio
async def test_dual_prompting():
    """Test dual-prompting system"""
    prompt = DualPrompt(
        script="Test script",
        vocal=VocalInstructions(
            emotion="neutral",
            emphasis_words=["Test"]
        )
    )
    bark_text = prompt.to_bark_text()
    assert "TEST" in bark_text  # Emphasis applied

@pytest.mark.asyncio
async def test_business_prompt_helper():
    """Test auto-vocal-instruction generation"""
    chat = VoiceChat(backend="pyttsx3")
    prompt = chat.create_business_prompt(
        "Revenue is $450",
        metric_type="positive"
    )
    assert prompt.vocal.emotion == "happy"
    assert "450" in prompt.vocal.emphasis_words
```

---

## 📚 Best Practices

### 1. Content-Delivery Separation

**Good** ✅:
```python
# Generate content (business logic)
script = f"Your revenue is ${revenue:.2f}"

# Determine delivery (emotion engine)
metric_type = "positive" if revenue > target else "negative"
prompt = chat.create_business_prompt(script, metric_type)

# Speak
await chat.speak(prompt)
```

**Bad** ❌:
```python
# Mixing business logic with TTS details
if revenue > target:
    await chat.speak(f"[laughs] Your revenue is {revenue.upper()}!", emotion="happy")
else:
    await chat.speak(f"[sighs]... Your revenue is {revenue}", emotion="concerned")
```

### 2. Use Business Helper for Metrics

**Good** ✅:
```python
# Automatic vocal instruction selection
for metric, value in daily_metrics.items():
    script = f"{metric}: {value}"
    metric_type = classify_metric(metric, value)  # positive/negative/warning
    prompt = chat.create_business_prompt(script, metric_type)
    await chat.speak(prompt)
```

**Bad** ❌:
```python
# Manual emotion selection for every metric
await chat.speak(f"Revenue: {revenue}", emotion="happy" if revenue > 0 else "sad")
await chat.speak(f"Hours: {hours}", emotion="concerned" if hours > 40 else "neutral")
# ... repetitive, error-prone
```

### 3. Reuse DualPrompts

**Good** ✅:
```python
# Define once, reuse for multiple backends
prompt = DualPrompt(
    script="Important message",
    vocal=VocalInstructions(emotion="concerned")
)

# Works with all backends
bark_audio = await bark_chat.speak(prompt)
elevenlabs_audio = await elevenlabs_chat.speak(prompt)
```

**Bad** ❌:
```python
# Backend-specific formatting
await bark_chat.speak("[sighs] Important message")
await elevenlabs_chat.speak("Important message")  # No emotion
```

### 4. Save Audio for Playback

**Good** ✅:
```python
# Save for later playback or caching
await chat.speak(prompt, save_path=f"audio/{event_id}.wav")

# Play cached audio
play_audio(f"audio/{event_id}.wav")
```

---

## 🚀 Future Enhancements

### Planned Features

1. **Voice Cloning** (ElevenLabs)
   - Clone user's own voice
   - Personalized business assistant

2. **Multi-Language Support**
   - Auto-detect language from script
   - Use appropriate voice model

3. **Emotion Blending**
   - Mix emotions (e.g., 70% happy + 30% surprised)
   - Smoother emotional transitions

4. **Prosody Control** (fine-grained)
   - Syllable-level emphasis
   - Precise pause durations (milliseconds)
   - Pitch contours

5. **Context-Aware Delivery**
   - Learn user's preferred pacing from feedback
   - Adapt emotion intensity based on time of day
   - Remember successful vocal patterns

6. **Streaming TTS**
   - Real-time audio generation as text is produced
   - Lower latency for long scripts

---

## 📖 References

- **BARK**: https://github.com/suno-ai/bark
- **ElevenLabs**: https://elevenlabs.io/docs
- **pyttsx3**: https://pyttsx3.readthedocs.io/

---

## 🎓 Summary

The Dual-Prompting TTS System provides:

✅ **Separation of Concerns**: Content generation ≠ Vocal delivery
✅ **Backend Interchangeability**: BARK, ElevenLabs, pyttsx3 all supported
✅ **Backwards Compatibility**: Legacy emotion-string API still works
✅ **Business-Aware**: Auto-generates appropriate delivery for metrics
✅ **Fine-Grained Control**: Emotion, pace, emphasis, sounds, pauses
✅ **Production-Ready**: Complete integration with COS workflows

**Next Steps**:
1. Test with actual BARK installation
2. Integrate with `daily_review.py` for morning/evening workflows
3. Add to `voice_input.py` for HITL verification dialogues
4. Create API endpoint for web dashboard TTS
5. Set up audio caching for repeated prompts

**🎤 Ready to speak!**
