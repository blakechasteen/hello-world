# TTS Backend Quick Reference Card

**File**: `cos/interface/voice_chat.py`
**Architecture**: Fully interchangeable backends with dual-prompting support

---

## 🔄 Switching Backends

### Method 1: At Initialization

```python
from cos.interface.voice_chat import VoiceChat

# BARK (recommended for development)
chat_bark = VoiceChat(backend="bark", voice="v2/en_speaker_6")

# ElevenLabs (recommended for production)
chat_eleven = VoiceChat(backend="elevenlabs", voice="Bella")

# pyttsx3 (fallback, always works)
chat_fallback = VoiceChat(backend="pyttsx3")
```

### Method 2: Auto-Detect Best Available

```python
def create_voice_chat() -> VoiceChat:
    """Auto-select best available backend"""
    from cos.interface.voice_chat import BARK_AVAILABLE, ELEVENLABS_AVAILABLE

    if BARK_AVAILABLE:
        return VoiceChat(backend="bark")
    elif ELEVENLABS_AVAILABLE:
        return VoiceChat(backend="elevenlabs")
    else:
        return VoiceChat(backend="pyttsx3")

chat = create_voice_chat()
```

### Method 3: Runtime Configuration

```python
import os

# Set backend via environment variable
TTS_BACKEND = os.getenv("TTS_BACKEND", "bark")
TTS_VOICE = os.getenv("TTS_VOICE", "v2/en_speaker_6")

chat = VoiceChat(backend=TTS_BACKEND, voice=TTS_VOICE)
```

---

## 📊 Backend Comparison Matrix

| Feature | BARK | ElevenLabs | pyttsx3 |
|---------|------|------------|---------|
| **Installation** | `pip install bark` | `pip install elevenlabs` | `pip install pyttsx3` |
| **Dependencies** | PyTorch, numpy | requests | pywin32 (Windows) |
| **Cost** | Free | ~$0.30/1000 chars | Free |
| **Latency** | 2-5 seconds | 1-2 seconds | <0.5 seconds |
| **Quality** | ★★★★☆ Natural | ★★★★★ Highest | ★★☆☆☆ Robotic |
| **Emotions** | ✅ Full | ✅ Full | ❌ None |
| **Sounds** | ✅ [laughs], [sighs] | ❌ Limited | ❌ None |
| **Offline** | ✅ Yes | ❌ Requires internet | ✅ Yes |
| **Voices** | 20+ presets | 100+ custom | System voices |

---

## 🎯 Recommended Setup

### Development
```python
# Local, free, good emotions
chat = VoiceChat(backend="bark", voice="v2/en_speaker_6")
```

### Production
```python
# High quality, reliable, cloud-based
chat = VoiceChat(backend="elevenlabs", voice="Bella")
```

### CI/CD Testing
```python
# Fast, no dependencies, always works
chat = VoiceChat(backend="pyttsx3")
```

---

## 💡 Usage Examples

### Same DualPrompt, Different Backends

```python
from cos.interface.voice_chat import VoiceChat, DualPrompt, VocalInstructions

# Create prompt once
prompt = DualPrompt(
    script="Your weekly revenue is $450, which is $50 below target.",
    vocal=VocalInstructions(
        emotion="concerned",
        pace="slow",
        emphasis_words=["450", "50", "below"],
        sounds_before=["sighs"]
    )
)

# Works with BARK
bark_chat = VoiceChat(backend="bark")
bark_audio = await bark_chat.speak(prompt)

# Works with ElevenLabs
eleven_chat = VoiceChat(backend="elevenlabs")
eleven_audio = await eleven_chat.speak(prompt)

# Works with pyttsx3 (degrades gracefully - no emotions)
fallback_chat = VoiceChat(backend="pyttsx3")
await fallback_chat.speak(prompt)  # Speaks without emotional markers
```

### Backend-Specific Output

**BARK**:
```
[sighs] Your weekly revenue... is FOUR FIFTY, which is FIFTY dollars BELOW target.
```

**ElevenLabs**:
```
Your weekly revenue is $450... $450... which is $50... $50 dollars below target.
(Delivered with concerned tone via API parameters)
```

**pyttsx3**:
```
Your weekly revenue is $450, which is $50 below target.
(Plain delivery, no emotions)
```

---

## 🔧 Environment Setup

### Install All Backends

```bash
# BARK (recommended)
pip install git+https://github.com/suno-ai/bark.git
pip install scipy  # For audio file I/O

# ElevenLabs (optional, requires API key)
pip install elevenlabs
export ELEVENLABS_API_KEY="your-api-key"

# pyttsx3 (fallback, should always install)
pip install pyttsx3
```

### Minimal Install (pyttsx3 only)

```bash
pip install pyttsx3
```

---

## 📝 API Compatibility

All backends support:

```python
# Legacy emotion-string API
await chat.speak("Hello!", emotion="happy")

# Dual-prompting API
prompt = DualPrompt(script="Hello!", vocal=VocalInstructions(emotion="happy"))
await chat.speak(prompt)

# Business helper API
prompt = chat.create_business_prompt("Revenue: $450", metric_type="positive")
await chat.speak(prompt)
```

---

## 🚨 Error Handling

### Graceful Degradation

```python
from cos.interface.voice_chat import VoiceChat, BARK_AVAILABLE, ELEVENLABS_AVAILABLE

def get_voice_chat(preferred: str = "bark") -> VoiceChat:
    """Get voice chat with automatic fallback"""
    try:
        if preferred == "bark" and BARK_AVAILABLE:
            return VoiceChat(backend="bark")
        elif preferred == "elevenlabs" and ELEVENLABS_AVAILABLE:
            return VoiceChat(backend="elevenlabs")
        else:
            print("⚠️ Preferred backend unavailable, using pyttsx3")
            return VoiceChat(backend="pyttsx3")
    except Exception as e:
        print(f"❌ Error initializing TTS: {e}")
        print("Using pyttsx3 fallback")
        return VoiceChat(backend="pyttsx3")

chat = get_voice_chat(preferred="bark")
```

---

## 📚 Quick Integration Patterns

### Pattern 1: Daily Review with Auto-Backend

```python
class DailyReviewWorkflow:
    def __init__(self, voice_mode: bool = False, tts_backend: str = "bark"):
        self.voice_mode = voice_mode
        if voice_mode:
            self.voice = VoiceChat(backend=tts_backend)

    async def speak_summary(self, summary: DailySummary):
        if not self.voice_mode:
            return

        script = f"Your daily profit is ${summary.profit:.2f}"
        metric_type = "positive" if summary.profit > 0 else "negative"
        prompt = self.voice.create_business_prompt(script, metric_type)
        await self.voice.speak(prompt)
```

### Pattern 2: API Server with Backend Selection

```python
from fastapi import FastAPI, Query

app = FastAPI()

@app.post("/tts/speak")
async def text_to_speech(
    script: str,
    backend: str = Query("bark", regex="^(bark|elevenlabs|pyttsx3)$")
):
    """TTS endpoint with backend selection"""
    chat = VoiceChat(backend=backend)
    prompt = chat.create_business_prompt(script, metric_type="neutral")
    audio = await chat.speak(prompt)

    return Response(content=audio, media_type="audio/wav")
```

### Pattern 3: Fallback Chain

```python
async def speak_with_fallback(script: str, emotion: str = "neutral"):
    """Try backends in order: BARK → ElevenLabs → pyttsx3"""
    backends = ["bark", "elevenlabs", "pyttsx3"]

    for backend in backends:
        try:
            chat = VoiceChat(backend=backend)
            audio = await chat.speak(script, emotion=emotion)
            return audio
        except Exception as e:
            print(f"Backend {backend} failed: {e}, trying next...")
            continue

    raise RuntimeError("All TTS backends failed")
```

---

## 🎛️ Configuration File Pattern

### config.json
```json
{
  "tts": {
    "backend": "bark",
    "voice": "v2/en_speaker_6",
    "fallback": "pyttsx3",
    "defaults": {
      "emotion": "neutral",
      "pace": "normal",
      "save_audio": true,
      "audio_dir": "./audio_cache"
    }
  }
}
```

### Loading Config
```python
import json

def load_voice_config(config_path: str = "config.json") -> VoiceChat:
    with open(config_path) as f:
        config = json.load(f)

    tts_config = config.get("tts", {})

    return VoiceChat(
        backend=tts_config.get("backend", "bark"),
        voice=tts_config.get("voice", "v2/en_speaker_6")
    )

chat = load_voice_config()
```

---

## 🔍 Debugging Tips

### Check Backend Availability

```python
from cos.interface.voice_chat import BARK_AVAILABLE, ELEVENLABS_AVAILABLE, PYTTSX3_AVAILABLE

print(f"BARK: {'✓' if BARK_AVAILABLE else '✗'}")
print(f"ElevenLabs: {'✓' if ELEVENLABS_AVAILABLE else '✗'}")
print(f"pyttsx3: {'✓' if PYTTSX3_AVAILABLE else '✗'}")
```

### Test Audio Output

```python
chat = VoiceChat(backend="bark")
audio = await chat.speak("Test", save_path="test.wav")

# Check file size
import os
print(f"Audio file size: {os.path.getsize('test.wav')} bytes")

# Play audio (requires playsound or similar)
from playsound import playsound
playsound("test.wav")
```

---

## 📦 Summary

✅ **3 backends**: BARK, ElevenLabs, pyttsx3
✅ **Fully interchangeable**: Same API across all backends
✅ **Automatic fallback**: Degrades gracefully if backend unavailable
✅ **Dual-prompting**: Same DualPrompt works everywhere
✅ **Production-ready**: Error handling, configuration, testing

**Switch backends with one line**: `VoiceChat(backend="...")`
