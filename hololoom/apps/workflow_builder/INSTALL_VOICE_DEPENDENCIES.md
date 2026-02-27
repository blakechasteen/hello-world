# Voice Dependencies Installation Guide

**Goal**: Install TTS and transcription dependencies for voice dashboard

---

## Current Status

Run this to check what's installed:
```bash
python hololoom/web_dashboard/test_voice_integration.py
```

---

## Option 1: Quick Start (pyttsx3) - Works Immediately ⚡

**Best for**: Testing tonight, want something working NOW

```bash
# Install pyttsx3 (basic TTS, works offline, no setup)
pip install pyttsx3

# Whisper already installed ✓
```

**Integration**: Change backend in agentic_server.py:
```python
voice_integration = await create_voice_integration(
    tts_backend="pyttsx3",  # ← Change to pyttsx3
    whisper_model="base",
    auto_speak=False
)
```

**Pros**:
- ✓ Installs in 5 seconds
- ✓ Works offline
- ✓ No GPU needed
- ✓ Zero configuration
- ✓ Cross-platform (Windows/Mac/Linux)

**Cons**:
- Basic voice quality (robotic)
- No emotional delivery
- No vocal instructions support

**Verdict**: **Use this to test the integration tonight!** You can always upgrade to BARK later.

---

## Option 2: High Quality (BARK) - Better Voice 🎤

**Best for**: Production use, want natural-sounding speech

```bash
# Install BARK (takes 5-10 minutes, downloads models)
pip install git+https://github.com/suno-ai/bark.git
pip install scipy

# SciPy already installed ✓
# Whisper already installed ✓
```

**Integration**: Default backend (already set):
```python
voice_integration = await create_voice_integration(
    tts_backend="bark",  # ← Default
    whisper_model="base",
    auto_speak=False
)
```

**Pros**:
- ✓ Very natural voice
- ✓ Supports emotional delivery
- ✓ Supports vocal instructions (sighs, pauses, etc.)
- ✓ Free and open source
- ✓ Works offline

**Cons**:
- Slower install (~5-10 min on first use, downloads models)
- Slower generation (~2-5 seconds per response)
- Requires more memory

**Verdict**: **Upgrade to this once you've verified pyttsx3 works**

---

## Option 3: Premium (ElevenLabs) - Best Quality 🌟

**Best for**: Production with budget, want highest quality

```bash
# Install ElevenLabs
pip install elevenlabs

# Set API key (get from https://elevenlabs.io)
export ELEVENLABS_API_KEY="your-api-key"
```

**Integration**: Change backend:
```python
voice_integration = await create_voice_integration(
    tts_backend="elevenlabs",  # ← Change to elevenlabs
    whisper_model="base",
    auto_speak=False
)
```

**Pros**:
- ✓ Highest quality voice
- ✓ Very fast (~1-2 seconds)
- ✓ Multiple voice options
- ✓ Professional quality

**Cons**:
- Requires API key
- Costs money (~$0.30 per 1000 characters)
- Requires internet connection

**Verdict**: **Use for production if you have budget**

---

## Recommended Installation Path

### Tonight (Testing)

```bash
# Step 1: Install pyttsx3 for immediate testing
pip install pyttsx3

# Step 2: Verify it works
python hololoom/web_dashboard/test_voice_integration.py

# Expected output:
#   TTS Available: True
#   TTS Backend: pyttsx3
#   Transcription Available: True
#   Whisper Model: base
```

### This Week (Quality)

```bash
# Upgrade to BARK for better quality
pip install git+https://github.com/suno-ai/bark.git scipy

# Change backend in agentic_server.py:
# tts_backend="bark"

# Test again
python hololoom/web_dashboard/test_voice_integration.py
```

### Production (If Needed)

```bash
# Only if you need highest quality and have budget
pip install elevenlabs
export ELEVENLABS_API_KEY="your-key"

# Change backend in agentic_server.py:
# tts_backend="elevenlabs"
```

---

## Current Dependencies Status

Based on test output:

| Dependency | Status | Action |
|------------|--------|--------|
| Whisper | ✓ Installed | None needed |
| PyTorch | ✓ Installed | None needed |
| SciPy | ✓ Installed | None needed |
| BARK | ✗ Not installed | `pip install git+https://github.com/suno-ai/bark.git` (optional) |
| pyttsx3 | ✗ Not installed | `pip install pyttsx3` (recommended for tonight) |
| ElevenLabs | ✗ Not installed | `pip install elevenlabs` (optional, premium) |

---

## Installation Commands (Quick Copy-Paste)

### For Tonight (pyttsx3):
```bash
pip install pyttsx3
python hololoom/web_dashboard/test_voice_integration.py
```

### For Quality (BARK):
```bash
pip install git+https://github.com/suno-ai/bark.git scipy
python hololoom/web_dashboard/test_voice_integration.py
```

### For Premium (ElevenLabs):
```bash
pip install elevenlabs
set ELEVENLABS_API_KEY=your-api-key
python hololoom/web_dashboard/test_voice_integration.py
```

---

## Troubleshooting

### "No module named 'pyttsx3'"
```bash
pip install pyttsx3
```

### "No module named 'bark'"
```bash
pip install git+https://github.com/suno-ai/bark.git scipy
```

### "BARK is slow on first use"
BARK downloads models (~500MB) on first use. This is normal. Subsequent uses are faster.

### "pyttsx3 voice sounds robotic"
This is expected. Upgrade to BARK for natural voice:
```bash
pip install git+https://github.com/suno-ai/bark.git scipy
```

Then change `tts_backend="bark"` in agentic_server.py.

### "ElevenLabs says 'unauthorized'"
Set your API key:
```bash
# Windows PowerShell
$env:ELEVENLABS_API_KEY="your-key"

# Windows CMD
set ELEVENLABS_API_KEY=your-key

# Mac/Linux
export ELEVENLABS_API_KEY="your-key"
```

---

## Comparison Table

| Feature | pyttsx3 | BARK | ElevenLabs |
|---------|---------|------|------------|
| **Quality** | Basic (robotic) | Very Natural | Highest |
| **Speed** | Very Fast (<0.5s) | Medium (2-5s) | Fast (1-2s) |
| **Install Time** | 5 seconds | 5-10 min | 10 seconds |
| **Cost** | Free | Free | ~$0.30/1K chars |
| **Offline** | ✓ Yes | ✓ Yes | ✗ No |
| **Emotions** | ✗ No | ✓ Yes | ✓ Yes |
| **Vocal Instructions** | ✗ No | ✓ Yes | ✓ Partial |
| **Setup Complexity** | None | Medium | Easy (need API key) |
| **Recommended For** | Testing | Production (free) | Production (premium) |

---

## Recommendation

**For Tonight**: Install pyttsx3
```bash
pip install pyttsx3
```

**For This Week**: Upgrade to BARK
```bash
pip install git+https://github.com/suno-ai/bark.git scipy
```

**For Production** (optional): Consider ElevenLabs if quality is critical
```bash
pip install elevenlabs
```

---

## Next Steps

1. Choose your backend (pyttsx3 recommended for tonight)
2. Install dependencies
3. Run verification: `python hololoom/web_dashboard/test_voice_integration.py`
4. Follow VOICE_QUICK_START.md for integration
5. Test voice in dashboard!

🎤 You're ready to add voice to your dashboard!
