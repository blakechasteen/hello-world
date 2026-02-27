# Voice Integration Guide - Agentic Dashboard

**Status**: Ready to integrate
**Components**: Voice transcription (Whisper) + Dual-prompting TTS (BARK/ElevenLabs)
**Moonshot**: Talk to your dashboard tonight!

---

## 🎯 What You Get

- **Voice Input**: Speak your queries instead of typing
- **Voice Output**: Dashboard speaks responses with business-appropriate delivery
- **Dual-Prompting**: Auto-generates emotional delivery based on confidence and metric type
- **Backend Interchangeable**: BARK (local, free) or ElevenLabs (cloud, premium)

---

## 📦 Files Created

1. **voice_integration.py** (300 lines)
   - `VoiceIntegration` class combining Whisper + TTS
   - Business-aware speech generation
   - Auto-vocal-delivery based on response metadata

2. **voice_endpoints.py** (150 lines)
   - REST endpoints for voice capabilities
   - `/api/voice/transcribe` - Audio → Text
   - `/api/voice/speak` - Text → Audio
   - `/api/voice/speak_response` - Smart delivery
   - `/api/voice/status` - Check availability

3. **VOICE_INTEGRATION_GUIDE.md** (This file)
   - Step-by-step integration instructions

---

## 🚀 Quick Integration (3 Steps)

### Step 1: Add Voice to Server Startup

Edit `hololoom/web_dashboard/agentic_server.py`:

```python
# Add import at top (around line 70)
from hololoom.apps.workflow_builder.voice_integration import create_voice_integration
from hololoom.apps.workflow_builder.voice_endpoints import add_voice_endpoints

# Add global variable (around line 106)
voice_integration = None

# In lifespan() function, after spinners initialization (around line 260):
# Initialize Voice Integration
logger.info("Initializing Voice Integration...")
try:
    from hololoom.apps.workflow_builder.voice_integration import create_voice_integration
    from hololoom.apps.workflow_builder.voice_endpoints import add_voice_endpoints

    voice_integration = await create_voice_integration(
        tts_backend="bark",  # or "elevenlabs" or "pyttsx3"
        whisper_model="base",  # Fast model for dashboard
        auto_speak=False  # Don't auto-speak every response
    )

    # Add voice endpoints to app
    add_voice_endpoints(app, voice_integration)

    logger.info("Voice Integration enabled")
    logger.info(f"  - TTS: {voice_integration.tts_backend}")
    logger.info(f"  - Whisper: {voice_integration.whisper_model}")

except Exception as e:
    logger.warning(f"Voice Integration failed: {e}")
    logger.warning("  - Continuing without voice capabilities")
    voice_integration = None
```

### Step 2: Test Voice Endpoints

Start the server:
```bash
cd c:\Users\blake\OneDrive\Documents\mythRL
python hololoom/web_dashboard/agentic_server.py
```

Test transcription endpoint:
```bash
curl -X POST http://localhost:8002/api/voice/status
```

Expected output:
```json
{
  "tts_available": true,
  "tts_backend": "bark",
  "transcription_available": true,
  "whisper_model": "base",
  "auto_speak": false
}
```

### Step 3: Add Voice UI (Next)

See `VOICE_UI_GUIDE.md` for adding mic button and audio playback to dashboard HTML.

---

## 🎤 Using Voice Endpoints

### Transcribe Audio

```javascript
// Record audio (browser API)
const mediaRecorder = new MediaRecorder(stream);
let audioChunks = [];

mediaRecorder.ondataavailable = (event) => {
    audioChunks.push(event.data);
};

mediaRecorder.onstop = async () => {
    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });

    // Upload for transcription
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.wav');

    const response = await fetch('http://localhost:8002/api/voice/transcribe', {
        method: 'POST',
        body: formData
    });

    const data = await response.json();
    console.log('Transcript:', data.transcript);
};
```

### Generate Speech

```javascript
// Simple TTS
const formData = new FormData();
formData.append('text', 'Your weekly revenue is $450');
formData.append('metric_type', 'neutral');  // positive, negative, neutral, warning

const response = await fetch('http://localhost:8002/api/voice/speak', {
    method: 'POST',
    body: formData
});

const audioBlob = await response.blob();
const audioUrl = URL.createObjectURL(audioBlob);
const audio = new Audio(audioUrl);
audio.play();
```

### Smart Response Delivery

```javascript
// Auto-determines vocal delivery from confidence/mode
const formData = new FormData();
formData.append('response', 'Thompson Sampling balances exploration...');
formData.append('confidence', '0.92');  // High confidence
formData.append('mode', 'verify');

const response = await fetch('http://localhost:8002/api/voice/speak_response', {
    method: 'POST',
    body: formData
});

const audioBlob = await response.blob();
const audio = new Audio(URL.createObjectURL(audioBlob));
audio.play();
```

---

## 🔧 Backend Configuration

### Using BARK (Recommended for Development)

```python
voice_integration = await create_voice_integration(
    tts_backend="bark",  # Local, free, natural
    whisper_model="base"  # Good balance speed/quality
)
```

**Install**:
```bash
pip install git+https://github.com/suno-ai/bark.git scipy
pip install openai-whisper
```

### Using ElevenLabs (Production)

```python
voice_integration = await create_voice_integration(
    tts_backend="elevenlabs",  # Cloud, highest quality
    whisper_model="base"
)
```

**Install**:
```bash
pip install elevenlabs
pip install openai-whisper
export ELEVENLABS_API_KEY="your-api-key"
```

### Using pyttsx3 (Fallback)

```python
voice_integration = await create_voice_integration(
    tts_backend="pyttsx3",  # Always works, basic
    whisper_model="base"
)
```

**Install**:
```bash
pip install pyttsx3
pip install openai-whisper
```

---

## 🎨 Vocal Delivery Examples

The system automatically generates appropriate vocal delivery based on metric type:

| Metric Type | Emotion | Pace | Sounds | Use Case |
|-------------|---------|------|--------|----------|
| **positive** | happy | normal | [bright tone] | Revenue up, goals met |
| **negative** | concerned | slow | [sighs] | Revenue down, failures |
| **warning** | concerned | slow | [clears throat] | Low inventory, burnout risk |
| **neutral** | neutral | normal | none | Reporting facts |

**Example**:

```javascript
// Dashboard speaks with concerned tone, slow pace, sighs before delivery
formData.append('text', 'Your hours this week are quite high');
formData.append('metric_type', 'warning');
```

**Output**: "[clears throat] Your HOURS this week are quite HIGH."

---

## 📊 Auto-Vocal-Delivery Algorithm

The `speak_response()` method analyzes response metadata:

```python
if confidence < 0.5:
    metric_type = "warning"  # Cautious delivery
elif confidence > 0.85:
    metric_type = "positive"  # Confident delivery
elif mode == "verify":
    metric_type = "neutral"  # Factual delivery
else:
    metric_type = "neutral"  # Default
```

This means responses automatically get appropriate vocal delivery without manual configuration!

---

## 🛠️ Troubleshooting

### "TTS not available"

**Problem**: BARK/ElevenLabs not installed

**Solution**:
```bash
# For BARK
pip install git+https://github.com/suno-ai/bark.git scipy

# For ElevenLabs
pip install elevenlabs
export ELEVENLABS_API_KEY="your-api-key"

# Fallback
pip install pyttsx3
```

### "Whisper not available"

**Problem**: openai-whisper not installed

**Solution**:
```bash
pip install openai-whisper
```

### "ffmpeg not found"

**Problem**: Whisper requires ffmpeg

**Solution** (Windows):
```bash
winget install Gyan.FFmpeg
```

**Solution** (Mac):
```bash
brew install ffmpeg
```

**Solution** (Linux):
```bash
sudo apt-get install ffmpeg
```

### Port Already in Use

**Problem**: Port 8002 already in use

**Solution**:
```python
# Change port in agentic_server.py line 2618
uvicorn.run(..., port=8003)
```

---

## 🎯 Next Steps

1. **Test Voice Endpoints**: Start server and test `/api/voice/status`
2. **Add Voice UI**: See `VOICE_UI_GUIDE.md` for frontend integration
3. **Customize Delivery**: Tune vocal delivery for your business metrics
4. **Production**: Switch to ElevenLabs for highest quality

---

## 📚 Complete Integration Example

Here's the complete code to add to `agentic_server.py`:

```python
# ============================================================================
# Voice Integration (Add to agentic_server.py)
# ============================================================================

# 1. Add imports at top (around line 70)
from hololoom.apps.workflow_builder.voice_integration import create_voice_integration
from hololoom.apps.workflow_builder.voice_endpoints import add_voice_endpoints

# 2. Add global variable (around line 106)
voice_integration = None

# 3. Add to lifespan() function (around line 260, after URL spinner init)
    # Initialize Voice Integration
    logger.info("Initializing Voice Integration...")
    try:
        global voice_integration
        voice_integration = await create_voice_integration(
            tts_backend="bark",  # or "elevenlabs" or "pyttsx3"
            whisper_model="base",  # Fast model for dashboard
            auto_speak=False  # Don't auto-speak every response
        )

        # Add voice endpoints to app
        add_voice_endpoints(app, voice_integration)

        logger.info("Voice Integration enabled")
        logger.info(f"  - TTS: {voice_integration.tts_backend}")
        logger.info(f"  - Whisper: {voice_integration.whisper_model}")

    except Exception as e:
        logger.warning(f"Voice Integration failed: {e}")
        logger.warning("  - Continuing without voice capabilities")
        voice_integration = None

# 4. That's it! Voice endpoints are now available at:
#    - POST /api/voice/transcribe
#    - POST /api/voice/speak
#    - POST /api/voice/speak_response
#    - GET /api/voice/status
```

---

## 🎤 Ready to Talk!

Once integrated, you can:

- ✅ Upload audio files for transcription
- ✅ Generate speech from text
- ✅ Auto-vocal-delivery for responses
- ✅ Switch TTS backends on the fly
- ✅ Business-aware emotional delivery

**Next**: Add voice controls to dashboard HTML for complete voice experience!

---

**Total Integration Time**: ~5 minutes
**Lines Added**: ~30 lines to existing server
**Dependencies**: bark (or elevenlabs), openai-whisper, ffmpeg
**Moonshot Status**: 🚀 Ready to launch!
