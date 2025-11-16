# Voice Synthesis - Elle Game Engine

**Multi-backend text-to-speech with emotion-aware voices**

Date: 2025-11-16

---

## Table of Contents

- [Overview](#overview)
- [Backend Comparison](#backend-comparison)
- [Configuration](#configuration)
- [Voice Profiles](#voice-profiles)
- [API Usage](#api-usage)
- [Caching System](#caching-system)
- [Integration Examples](#integration-examples)
- [Backend Setup Guides](#backend-setup-guides)
- [Troubleshooting](#troubleshooting)

---

## Overview

Elle's voice synthesis system provides text-to-speech capabilities with multiple backend options, allowing you to choose the best balance of quality, latency, and cost for your game.

### Key Features

✅ **Multi-Backend Support**: ElevenLabs, OpenAI, Google Cloud, Piper (local), Dummy
✅ **Emotion-Aware**: Voice adapts to character mood/tone
✅ **Voice Profiles**: Per-NPC voice configuration (pitch, speed, stability)
✅ **Smart Caching**: Reuse common phrases (100MB default cache)
✅ **Format Support**: WAV, MP3, OGG, OPUS
✅ **Streaming**: Audio generation with low latency
✅ **Graceful Degradation**: Falls back to text if TTS unavailable

---

## Backend Comparison

| Backend | Quality | Latency | Cost/1K chars | Local | Emotions |
|---------|---------|---------|---------------|-------|----------|
| **ElevenLabs** | ⭐⭐⭐⭐⭐ (Best) | ~2-3s | $0.30 | ❌ | ✅ Full |
| **OpenAI TTS** | ⭐⭐⭐⭐ (Great) | ~1-2s | $0.015 | ❌ | 🟡 Basic |
| **Google Cloud** | ⭐⭐⭐⭐ (Great) | ~1-2s | $0.016 | ❌ | ✅ Good |
| **Piper** | ⭐⭐⭐ (Good) | <500ms | FREE | ✅ | 🟡 Basic |
| **Dummy** | ⭐ (Silent) | <1ms | FREE | ✅ | ❌ |

### Recommendation Matrix

| Use Case | Recommended Backend | Reason |
|----------|---------------------|--------|
| **AAA Production** | ElevenLabs | Best quality, full emotion control |
| **Indie Production** | OpenAI TTS | Great quality, affordable |
| **Offline/Mobile** | Piper | Local, fast, free |
| **Development** | Dummy | Fast iteration, no API costs |
| **Budget-Conscious** | Piper or OpenAI | Free (local) or cheap (cloud) |

---

## Configuration

### Environment Variables

```bash
# Backend selection
ELLE_VOICE_BACKEND="openai"  # elevenlabs, openai, piper, dummy

# Backend-specific API keys
ELEVENLABS_API_KEY="your-key-here"
OPENAI_API_KEY="your-key-here"
GOOGLE_CLOUD_API_KEY="your-key-here"

# Model selection (backend-specific)
ELLE_VOICE_MODEL="tts-1"  # OpenAI: tts-1, tts-1-hd
```

### Programmatic Configuration

```python
from apps.elle_game_engine.voice import create_voice_engine, TTSBackend

# Create with specific backend
voice_engine = create_voice_engine(
    backend="openai",
    api_key="your-key",
    model="tts-1-hd",
    enable_cache=True
)
```

---

## Voice Profiles

Voice profiles define how a character's voice should sound.

### Creating Voice Profiles

```python
from apps.elle_game_engine.voice import VoiceProfile, Emotion

# Create NPC voice profile
bob_profile = VoiceProfile(
    voice_id="alloy",        # Backend-specific voice ID
    pitch=1.0,               # 0.5 (low) to 2.0 (high)
    speed=1.0,               # 0.5 (slow) to 2.0 (fast)
    emotion=Emotion.WARM,    # Base emotional tone
    stability=0.5,           # Voice consistency (ElevenLabs)
    similarity_boost=0.75    # Voice clarity (ElevenLabs)
)

# Register profile
voice_engine.register_voice_profile("innkeeper", bob_profile)
```

### Voice Parameters

| Parameter | Range | Description | Backend Support |
|-----------|-------|-------------|----------------|
| `voice_id` | string | Backend-specific voice identifier | All |
| `pitch` | 0.5-2.0 | Voice pitch (1.0 = normal) | All |
| `speed` | 0.5-2.0 | Speech speed (1.0 = normal) | All |
| `emotion` | Emotion enum | Base emotional tone | All |
| `stability` | 0.0-1.0 | Voice consistency | ElevenLabs |
| `similarity_boost` | 0.0-1.0 | Voice clarity vs expressiveness | ElevenLabs |

### Available Emotions

```python
class Emotion(Enum):
    NEUTRAL = "neutral"
    WARM = "warm"
    STERN = "stern"
    EXCITED = "excited"
    SAD = "sad"
    CRYPTIC = "cryptic"
    CURIOUS = "curious"
    GRATEFUL = "grateful"
    ANNOYED = "annoyed"
    HOSTILE = "hostile"
```

---

## API Usage

### Basic Synthesis

```python
from apps.elle_game_engine.voice import create_voice_engine, VoiceProfile

# Initialize engine
voice_engine = create_voice_engine(backend="openai")

# Create voice profile
profile = VoiceProfile(voice_id="alloy")
voice_engine.register_voice_profile("bob", profile)

# Synthesize speech
result = voice_engine.synthesize(
    text="Hello, traveler! Welcome to my inn.",
    npc_id="bob"
)

# Get audio
audio_bytes = result.audio_data
duration = result.duration_seconds
was_cached = result.cached
```

### Synthesis with Emotion Override

```python
from apps.elle_game_engine.voice import Emotion

# Synthesize with specific emotion
result = voice_engine.synthesize(
    text="Get out of my shop!",
    npc_id="bob",
    emotion=Emotion.HOSTILE  # Override base emotion
)
```

### Synthesis with Custom Format

```python
from apps.elle_game_engine.voice import AudioFormat

# Generate OGG audio (smaller file size)
result = voice_engine.synthesize(
    text="The treasure lies within...",
    npc_id="wizard",
    format=AudioFormat.OGG
)
```

### Direct Voice Profile Synthesis

```python
# Synthesize without registration
profile = VoiceProfile(voice_id="nova", pitch=0.8, speed=1.2)

result = voice_engine.synthesize(
    text="I speak directly from the void.",
    voice_profile=profile
)
```

---

## Caching System

The voice cache stores synthesized audio to avoid re-generating common phrases.

### Cache Configuration

```python
from apps.elle_game_engine.voice import VoiceEngine

engine = VoiceEngine(
    backend="openai",
    enable_cache=True,
    cache_dir="./.voice_cache",  # Cache directory
    cache_size_mb=100             # Max cache size
)
```

### Cache Behavior

The cache key is based on: `(text, voice_id, pitch, speed, emotion)`

```python
# First call: Cold cache (~1-2s)
result1 = voice_engine.synthesize("Hello!", npc_id="bob")
assert result1.cached == False

# Second call: Cache hit (<1ms)
result2 = voice_engine.synthesize("Hello!", npc_id="bob")
assert result2.cached == True
```

### Cache Management

```python
# Get cache statistics
stats = voice_engine.get_cache_stats()
print(f"Cached entries: {stats['num_entries']}")
print(f"Total size: {stats['total_size_mb']:.1f}MB")

# Clear cache
voice_engine.clear_cache()
```

### Cache Eviction

LRU eviction kicks in when cache exceeds max size:

```python
# Cache automatically evicts oldest entries
# when total size > cache_size_mb
```

---

## Integration Examples

### Unity Integration

```csharp
using System.Net.Http;
using System.Text;
using UnityEngine;

public class ElleVoiceClient : MonoBehaviour
{
    private const string BASE_URL = "http://localhost:8000";

    public async Task<byte[]> SynthesizeVoice(string text, string voiceId)
    {
        var request = new {
            text = text,
            voice_profile = new {
                voice_id = voiceId,
                pitch = 1.0f,
                speed = 1.0f,
                emotion = "neutral"
            },
            format = "mp3"
        };

        var json = JsonUtility.ToJson(request);
        var content = new StringContent(json, Encoding.UTF8, "application/json");

        using var client = new HttpClient();
        var response = await client.PostAsync($"{BASE_URL}/elle/game/voice/synthesize", content);

        var resultJson = await response.Content.ReadAsStringAsync();
        var result = JsonUtility.FromJson<VoiceSynthesisResponse>(resultJson);

        // Decode base64 audio
        return Convert.FromBase64String(result.audio_data);
    }
}
```

### Godot Integration

```gdscript
extends Node

@onready var elle = Elle
@onready var audio_player = $AudioStreamPlayer

func get_voice_dialogue(npc_id: String, text: String):
    # ElleClient handles voice synthesis automatically
    var profile = VoiceProfile.new("alloy", 1.0, 1.0, "warm")

    # Elle will include audio in response if voice enabled
    await elle.get_npc_dialogue(npc_id, "scene", text)


func _on_action_received(action: ElleModels.ElleGameAction):
    if action.has_audio():
        # Audio is included in response
        var stream = AudioStreamOggVorbis.new()
        stream.data = action.audio_data
        audio_player.stream = stream
        audio_player.play()
```

### Python Integration

```python
import asyncio
from apps.elle_game_engine.voice import create_voice_engine, VoiceProfile, Emotion

async def main():
    # Initialize engine
    engine = create_voice_engine(backend="openai")

    # Create voice profiles for NPCs
    bob = VoiceProfile(voice_id="alloy", emotion=Emotion.WARM)
    guard = VoiceProfile(voice_id="onyx", pitch=0.8, emotion=Emotion.STERN)

    engine.register_voice_profile("innkeeper", bob)
    engine.register_voice_profile("guard", guard)

    # Synthesize dialogue
    innkeeper_audio = engine.synthesize(
        "Welcome to my inn!",
        npc_id="innkeeper"
    )

    guard_audio = engine.synthesize(
        "Halt! State your business.",
        npc_id="guard"
    )

    # Save audio files
    with open("innkeeper.mp3", "wb") as f:
        f.write(innkeeper_audio.audio_data)

    with open("guard.mp3", "wb") as f:
        f.write(guard_audio.audio_data)

asyncio.run(main())
```

---

## Backend Setup Guides

### ElevenLabs Setup

1. **Get API Key**: https://elevenlabs.io/
2. **Configure**:
   ```bash
   export ELLE_VOICE_BACKEND="elevenlabs"
   export ELEVENLABS_API_KEY="your-key"
   export ELLE_VOICE_MODEL="eleven_monolingual_v1"
   ```
3. **Voice IDs**: Get from ElevenLabs dashboard (e.g., `21m00Tcm4TlvDq8ikWAM`)

### OpenAI TTS Setup

1. **Get API Key**: https://platform.openai.com/api-keys
2. **Configure**:
   ```bash
   export ELLE_VOICE_BACKEND="openai"
   export OPENAI_API_KEY="sk-..."
   export ELLE_VOICE_MODEL="tts-1-hd"  # or tts-1
   ```
3. **Voice IDs**: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`

### Google Cloud TTS Setup

1. **Enable API**: https://console.cloud.google.com/apis/library/texttospeech.googleapis.com
2. **Get Credentials**: Create service account JSON
3. **Configure**:
   ```bash
   export ELLE_VOICE_BACKEND="google_cloud"
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
   ```

### Piper (Local) Setup

1. **Install Piper**:
   ```bash
   pip install piper-tts
   ```
2. **Download Models**: https://github.com/rhasspy/piper/releases
3. **Configure**:
   ```bash
   export ELLE_VOICE_BACKEND="piper"
   export PIPER_MODEL_PATH="/path/to/model.onnx"
   ```

---

## Troubleshooting

### Problem: API Key Errors

**Symptoms**: `ValueError: API key required`

**Solutions**:
1. Set environment variable: `export OPENAI_API_KEY="..."`
2. Pass API key directly: `create_voice_engine(api_key="...")`

### Problem: Slow Synthesis

**Symptoms**: >3s latency per request

**Solutions**:
1. Use faster model: `tts-1` instead of `tts-1-hd`
2. Switch to local backend: Piper
3. Enable caching to avoid repeated synthesis

### Problem: Poor Voice Quality

**Symptoms**: Robotic, unnatural voices

**Solutions**:
1. Use higher quality backend: ElevenLabs > OpenAI > Piper
2. Adjust `stability` and `similarity_boost` (ElevenLabs)
3. Use higher quality model: `tts-1-hd` (OpenAI)

### Problem: Cache Not Working

**Symptoms**: Every request takes full synthesis time

**Solutions**:
1. Verify cache enabled: `enable_cache=True`
2. Check cache directory writable
3. Different parameters create different cache keys

---

## Performance Tips

### Reduce Latency

1. **Use Local Backend**: Piper (<500ms vs 1-3s cloud)
2. **Pre-generate Common Phrases**: Cache on startup
3. **Smaller Chunks**: Break long text into sentences
4. **Streaming**: Request audio in chunks (future feature)

### Reduce Costs

1. **Use Cheaper Backend**: Piper (free) > OpenAI ($0.015) > ElevenLabs ($0.30)
2. **Aggressive Caching**: 100% hit rate for repeated phrases = $0 cost
3. **Shorter Text**: Costs scale with character count

### Example: Pre-warming Cache

```python
# Pre-generate common greetings on startup
COMMON_PHRASES = [
    "Hello, traveler!",
    "Welcome to my shop.",
    "I have nothing for you.",
    "Come back later."
]

async def prewarm_cache():
    for phrase in COMMON_PHRASES:
        await voice_engine.synthesize(phrase, npc_id="generic")

# Run on game startup
asyncio.run(prewarm_cache())
```

---

## Advanced Usage

### Custom Voice Backend

Implement your own TTS backend:

```python
from apps.elle_game_engine.voice import TTSBackendBase, VoiceSynthesisRequest, VoiceSynthesisResult

class MyCustomTTS(TTSBackendBase):
    def synthesize(self, request: VoiceSynthesisRequest) -> VoiceSynthesisResult:
        # Your custom TTS logic
        audio_bytes = my_tts_api.generate(request.text)

        return VoiceSynthesisResult(
            audio_data=audio_bytes,
            format=AudioFormat.MP3,
            duration_seconds=len(request.text) / 15.0,
            text=request.text,
            cached=False
        )
```

---

**Happy Voice Synthesis!** 🔊✨
