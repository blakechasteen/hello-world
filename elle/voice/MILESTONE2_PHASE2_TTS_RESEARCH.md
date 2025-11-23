# Voice UX Milestone 2 - Phase 2: TTS Research

**Date**: November 22, 2025
**Phase**: Natural Voice Synthesis Integration
**Status**: Research Complete

## Research Summary

### LightSpindle vs LightSpeech

**Finding**: "LightSpindle" does not exist as a TTS system. Most likely reference is **LightSpeech**.

**LightSpeech** (Microsoft Research, 2021):
- Uses Neural Architecture Search (NAS) to optimize FastSpeech
- **15x model compression** vs baseline
- **6.5x inference speedup** on CPU
- Maintains voice quality parity with FastSpeech 2
- **Limitation**: Research project, no production-ready package available

---

## Top Open-Source TTS Options (2025)

### 1. Coqui TTS ⭐ **RECOMMENDED**

**Pros**:
- ✅ Production-ready Python package
- ✅ Wide library of pre-trained voices
- ✅ Multilingual support
- ✅ Fine-tuning capabilities
- ✅ Good documentation and community
- ✅ Can run offline (no API keys)
- ✅ Simple Python API

**Cons**:
- ⚠ Requires GPU for best performance (CPU possible but slower)
- ⚠ Larger model size (~200MB+)

**Installation**:
```bash
pip install TTS
```

**Usage**:
```python
from TTS.api import TTS

tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
tts.tts_to_file(text="Hello from Elle", file_path="output.wav")
```

**Voice Quality**: ⭐⭐⭐⭐⭐ (Excellent)
**Latency**: ~200-500ms (GPU), ~1-2s (CPU)
**Ease of Use**: ⭐⭐⭐⭐⭐ (Very Easy)

---

### 2. MeloTTS ⭐ **BEST FOR FAST RESPONSES**

**Pros**:
- ✅ Optimized for speed and efficiency
- ✅ Runs well on limited hardware
- ✅ Low latency (<300ms typical)
- ✅ Smaller model size
- ✅ Good for voice assistant use case

**Cons**:
- ⚠ Fewer voice options than Coqui
- ⚠ Newer project (less mature)

**Voice Quality**: ⭐⭐⭐⭐ (Very Good)
**Latency**: ~200-300ms (optimized for speed)
**Ease of Use**: ⭐⭐⭐⭐ (Easy)

**Best for**: Elle's **<500ms feedback** goal (Milestone 2 requirement)

---

### 3. XTTS-v2 (Voice Cloning)

**Pros**:
- ✅ High-quality multilingual synthesis
- ✅ Voice cloning from short samples
- ✅ Natural, adaptable voices

**Cons**:
- ⚠ Higher latency (~500ms-1s)
- ⚠ Requires GPU for reasonable performance
- ⚠ Larger model size

**Voice Quality**: ⭐⭐⭐⭐⭐ (Excellent, most natural)
**Latency**: ~500ms-1s (slower)
**Ease of Use**: ⭐⭐⭐ (Moderate)

**Best for**: High-quality voice cloning, not optimized for real-time

---

### 4. Mimic 3 (Mycroft AI)

**Pros**:
- ✅ Very small and efficient
- ✅ Runs locally without cloud
- ✅ Privacy-focused
- ✅ Good for embedded systems

**Cons**:
- ⚠ Voice quality lower than Coqui/XTTS
- ⚠ Limited voice options

**Voice Quality**: ⭐⭐⭐ (Good)
**Latency**: ~100-200ms (very fast)
**Ease of Use**: ⭐⭐⭐⭐ (Easy)

**Best for**: Edge devices, resource-constrained environments

---

### 5. Bark (Suno AI)

**Pros**:
- ✅ Generates expressive speech with intonation
- ✅ Can produce non-speech sounds (laughter, music)
- ✅ Very creative and experimental

**Cons**:
- ⚠ Unpredictable output
- ⚠ High computational requirements
- ⚠ Higher latency (~2-5s)

**Voice Quality**: ⭐⭐⭐⭐ (Very expressive, but unpredictable)
**Latency**: ~2-5s (very slow)
**Ease of Use**: ⭐⭐ (Complex)

**Best for**: Creative audio projects, not voice assistants

---

### 6. Current System (pyttsx3)

**Status**: Basic offline TTS, currently used in Elle

**Pros**:
- ✅ Very simple, zero dependencies
- ✅ Cross-platform
- ✅ Instant response (<50ms)

**Cons**:
- ❌ Robotic voice quality
- ❌ Limited voice options
- ❌ No neural synthesis
- ❌ Poor naturalness

**Voice Quality**: ⭐⭐ (Robotic)
**Latency**: <50ms (instant)
**Ease of Use**: ⭐⭐⭐⭐⭐ (Very Easy)

---

## Recommendation

### Primary Recommendation: **Coqui TTS**

**Why**:
1. ✅ Production-ready with excellent documentation
2. ✅ Meets <500ms latency goal (with GPU or good CPU)
3. ✅ Natural voice quality (neural synthesis)
4. ✅ Offline capable (no API keys required)
5. ✅ Simple Python API (minimal code changes)
6. ✅ Wide range of voices and languages

**Implementation**:
- Maintain pyttsx3 as fallback for when Coqui unavailable
- Graceful degradation: Coqui → pyttsx3 → text-only
- Add configuration flag to enable/disable neural TTS

### Alternative: **MeloTTS** (if speed is critical)

If <300ms latency is absolutely critical, consider MeloTTS:
- Optimized specifically for fast inference
- Still neural-quality voices
- Smaller model size

---

## Integration Plan

### Phase 2.1: Coqui TTS Integration (Week 7-8)

**Step 1: Install Dependencies**
```bash
pip install TTS torch
```

**Step 2: Create NeuralTTS Class**
- Wrapper around Coqui TTS API
- Asynchronous synthesis
- Audio queue management
- Fallback to pyttsx3

**Step 3: Update VoiceAssistant**
- Add `use_neural_tts` flag to config
- Initialize NeuralTTS alongside pyttsx3
- Graceful fallback on error

**Step 4: Testing**
- Voice quality comparison
- Latency benchmarking
- Memory usage profiling

### Phase 2.2: Voice Personality Configuration (Week 9-10)

**Features**:
- Multiple voice presets (professional, friendly, casual)
- Rate and pitch control
- Emotion parameters (if supported)
- Voice caching for common responses

**Configuration**:
```python
class VoicePersonality(Enum):
    PROFESSIONAL = "tts_models/en/ljspeech/tacotron2-DDC"
    FRIENDLY = "tts_models/en/vctk/vits"
    CASUAL = "tts_models/en/ljspeech/fast_pitch"

assistant = VoiceAssistant(
    neural_tts=True,
    voice_personality=VoicePersonality.FRIENDLY,
    tts_rate=150,
    enable_voice_cache=True
)
```

---

## Performance Targets

**Milestone 2 Goals**:
- ✅ Neural voice quality (vs robotic pyttsx3)
- ✅ <500ms feedback for brief responses
- ✅ Offline operation (no API dependencies)
- ✅ Graceful fallback to basic TTS

**Coqui TTS Expected Performance**:
- Voice quality: ⭐⭐⭐⭐⭐ (vs current ⭐⭐)
- Latency: ~200-500ms (vs current <50ms)
- Trade-off: +150-450ms latency for natural voice **acceptable**

**Brief Response Optimization**:
```
Command: "t3"
Response: "Thread 3"  (2 words)
Synthesis: ~100-200ms
Total: ~100-200ms ✅ Meets <500ms goal
```

---

## Code Changes Summary

**Files to Modify**:
1. `elle/voice/tts.py` - Add NeuralTTS class
2. `elle/voice/assistant.py` - Add neural_tts flag
3. `requirements.txt` - Add TTS package

**Files to Create**:
1. `elle/voice/neural_tts.py` - Coqui TTS wrapper (new file)
2. `elle/voice/voice_personality.py` - Voice configuration (new file)

**Estimated Lines of Code**: ~300 lines total

---

## Risks & Mitigation

**Risk 1**: Coqui TTS installation fails
- **Mitigation**: Graceful fallback to pyttsx3

**Risk 2**: GPU not available, CPU too slow
- **Mitigation**: MeloTTS as faster alternative, or keep pyttsx3

**Risk 3**: High latency on low-end hardware
- **Mitigation**: Benchmark on target hardware, adjust model choice

**Risk 4**: Large model download (200MB+)
- **Mitigation**: On-demand download, user prompt for consent

---

## Next Steps

1. ✅ **Research complete** (this document)
2. ⬜ Install Coqui TTS for testing
3. ⬜ Create neural_tts.py wrapper
4. ⬜ Benchmark latency on development machine
5. ⬜ Integrate with VoiceAssistant
6. ⬜ Add configuration options
7. ⬜ Test end-to-end voice interaction
8. ⬜ Document voice personality system

---

## References

- **Coqui TTS**: https://github.com/coqui-ai/TTS
- **LightSpeech Paper**: https://arxiv.org/abs/2102.04040
- **MeloTTS**: https://github.com/myshell-ai/MeloTTS
- **XTTS-v2**: https://huggingface.co/coqui/XTTS-v2
- **Northflank TTS Guide**: https://northflank.com/blog/best-open-source-text-to-speech-models-and-how-to-run-them

---

**Research Status**: ✅ Complete
**Recommendation**: Proceed with Coqui TTS integration
**Timeline**: 2-3 weeks for full Phase 2 implementation
