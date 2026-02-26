# Voice Dashboard - Quick Start (Tonight!)

**Goal**: Talk to your dashboard in < 30 minutes

---

## ⚡ 3-Step Quick Start

### Step 1: Install Dependencies (2 minutes)

```bash
# Install BARK TTS (recommended, free)
pip install git+https://github.com/suno-ai/bark.git scipy

# Install Whisper for transcription
pip install openai-whisper

# Install ffmpeg (required for Whisper)
winget install Gyan.FFmpeg  # Windows
# OR: brew install ffmpeg (Mac)
# OR: sudo apt-get install ffmpeg (Linux)
```

**Verify installation**:
```bash
python -c "import bark; print('BARK OK')"
python -c "import whisper; print('Whisper OK')"
ffmpeg -version
```

---

### Step 2: Integrate Backend (10 minutes)

Open `HoloLoom/web_dashboard/agentic_server.py` and make these changes:

#### 2.1 Add Imports (around line 70)
```python
from HoloLoom.apps.workflow_builder.voice_integration import create_voice_integration
from HoloLoom.apps.workflow_builder.voice_endpoints import add_voice_endpoints
```

#### 2.2 Add Global Variable (around line 106)
```python
voice_integration = None
```

#### 2.3 Add to lifespan() Function (around line 260, after URL spinner)

```python
    # Initialize Voice Integration
    logger.info("Initializing Voice Integration...")
    try:
        global voice_integration
        voice_integration = await create_voice_integration(
            tts_backend="bark",  # Free, works offline
            whisper_model="base",  # Fast, good quality
            auto_speak=False  # Manual speak button (not auto)
        )

        add_voice_endpoints(app, voice_integration)

        logger.info("✓ Voice Integration enabled")
        logger.info(f"  - TTS: {voice_integration.tts_backend}")
        logger.info(f"  - Whisper: {voice_integration.whisper_model}")

    except Exception as e:
        logger.warning(f"Voice Integration failed: {e}")
        logger.warning("Dashboard will work without voice")
        voice_integration = None
```

**That's it for the backend!** Only ~30 lines added.

---

### Step 3: Integrate Frontend (15 minutes)

Open `HoloLoom/web_dashboard/agentic_dashboard.html`:

#### 3.1 Add CSS (in `<style>` section, around line 2156)

Copy the entire CSS section from `voice_ui_snippet.html` (lines 24-149):
- `.voice-button` styles
- `.recording-indicator` styles
- `.speak-button` styles
- `@keyframes pulse` animation

#### 3.2 Add HTML (in `<body>`, around line 2212, before `</body>`)

```html
<!-- Voice Button (Floating Action Button) -->
<button class="voice-button" id="voiceButton" title="Click to record voice">
    🎤
</button>

<!-- Recording Indicator -->
<div class="recording-indicator" id="recordingIndicator">
    <div class="recording-dot"></div>
    <span>Recording...</span>
</div>

<!-- Hidden Audio Player -->
<audio id="audioPlayer"></audio>
```

#### 3.3 Add JavaScript (in `<script>` section, around line 2603, before `</script>`)

Copy the entire JavaScript section from `voice_ui_snippet.html` (lines 173-413):
- Voice status checking
- Recording functionality
- Transcription workflow
- Speech generation
- Speak button integration

#### 3.4 Update addAssistantMessage() Function

Find the `addAssistantMessage()` function and add this at the end (before the closing `}`):

```javascript
    // Add speak button
    addSpeakButton(
        messageDiv,
        data.response || 'No response',
        data.confidence || 0.5,
        data.mode || 'direct'
    );
```

**Done!** Frontend integration complete.

---

## 🧪 Test It (1 minute)

### Start the Server

```bash
cd c:\Users\blake\OneDrive\Documents\mythRL
PYTHONPATH=. python HoloLoom/web_dashboard/agentic_server.py
```

Look for this in the logs:
```
✓ Voice Integration enabled
  - TTS: bark
  - Whisper: base
```

### Open Dashboard

```
http://localhost:8002
```

### Test Voice Input

1. Click the floating 🎤 button (bottom right)
2. Speak: "What is Thompson Sampling?"
3. Click 🎤 again to stop recording
4. Watch it transcribe and submit automatically
5. See the response appear

### Test Voice Output

1. Click the "🔊 Speak" button on any response
2. Hear the dashboard speak with appropriate delivery
3. High-confidence responses sound confident
4. Low-confidence responses sound cautious

---

## 🎯 You're Done!

You can now **talk to your dashboard**!

**Voice Input**: 🎤 → Speak → Auto-transcribe → Query submitted
**Voice Output**: 🔊 → Confidence-aware delivery → BARK TTS

---

## 🚧 Troubleshooting

### "Voice Integration failed: No module named 'bark'"
**Fix**: Install BARK
```bash
pip install git+https://github.com/suno-ai/bark.git scipy
```

### "Voice Integration failed: No module named 'whisper'"
**Fix**: Install Whisper
```bash
pip install openai-whisper
```

### "ffmpeg not found"
**Fix**: Install ffmpeg
```bash
winget install Gyan.FFmpeg
```

### Mic button shows "Voice not available"
**Fix**: Check browser console for errors, verify BARK + Whisper installed

### No audio output when clicking speak button
**Fix**:
1. Check speaker volume
2. Check browser console for errors
3. Verify BARK installed correctly

### Browser says "Microphone access denied"
**Fix**:
1. Click lock icon in address bar
2. Allow microphone access
3. Refresh page

---

## 📚 More Info

- **Complete Guide**: See `VOICE_INTEGRATION_GUIDE.md`
- **Architecture**: See `VOICE_MOONSHOT_COMPLETE.md`
- **Frontend Code**: See `voice_ui_snippet.html`

---

## ⏱️ Timeline

- **Step 1**: Install dependencies (2 min)
- **Step 2**: Backend integration (10 min)
- **Step 3**: Frontend integration (15 min)
- **Testing**: Verify it works (1 min)

**Total**: ~30 minutes to talking dashboard! 🚀

---

🎤 **Ready to ship!** Start with Step 1 and you'll be talking to your dashboard tonight.
