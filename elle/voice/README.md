# Elle Voice Interface

Complete voice control system for Elle Core with hands-free operation.

Created: 2025-11-15

## Overview

The Elle Voice Interface provides a complete voice control system combining:

- **Speech-to-Text**: OpenAI Whisper for local speech recognition
- **Natural Language Parsing**: LLM-enhanced command parsing with semantic understanding
- **Text-to-Speech**: pyttsx3 for voice responses
- **Wake Word Detection**: "Hey Elle" or "Elle" activation
- **Command Execution**: Integration with Elle Core SOP editor and task tracker

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Voice Assistant                        │
└─────────────────────────────────────────────────────────┘
                         ↓
        ┌────────────────┼────────────────┐
        ↓                ↓                ↓
    ┌────────┐      ┌─────────┐     ┌──────────┐
    │  STT   │      │ Parser  │     │   TTS    │
    │Whisper │      │  LLM    │     │ pyttsx3  │
    └────────┘      └─────────┘     └──────────┘
        ↓                ↓                ↑
        └────────────────┼────────────────┘
                         ↓
            ┌─────────────────────────┐
            │  Wake Word Detector     │
            │  (Hey Elle / Elle)      │
            └─────────────────────────┘
                         ↓
            ┌─────────────────────────┐
            │  Voice SOP Editor       │
            │  (Command Execution)    │
            └─────────────────────────┘
```

## Quick Start

### Installation

Install dependencies:

```bash
# Speech-to-text
pip install openai-whisper

# Text-to-speech
pip install pyttsx3

# Natural language (optional, for LLM parsing)
pip install anthropic
```

### Interactive Mode (No Microphone Needed)

Test the voice interface with text input:

```bash
cd /home/user/hello-world
PYTHONPATH=. python elle/voice/assistant.py interactive
```

Then type commands like:
- "show bread SOP"
- "update bread SOP: increase proofing to 50 minutes"
- "what's the biochar inoculation ratio?"
- "start baking bread"

### Voice Mode (Requires Microphone)

```bash
PYTHONPATH=. python elle/voice/assistant.py voice
```

Listen for "Hey Elle" prompt, then say your command.

## Voice Commands

### SOP Management

**Show SOP**
```
"Show bread SOP"
"Display the biochar SOP"
"What's in the mushroom SOP?"
```

**Update SOP**
```
"Update bread SOP: increase proofing time to 50 minutes"
"Change mushroom temperature to 75 degrees"
"Elle, set the inoculation ratio to 5 percent"
"Make bread proofing longer, 45 minutes"
"Bump up biochar heating to 300 degrees"
```

**Add Step**
```
"Add step to bread SOP: cool loaves for 30 minutes"
"Add new step to mushroom: check moisture level"
```

**Remove Step**
```
"Remove step 3 from bread SOP"
"Delete step 2 from biochar SOP"
```

### Knowledge Queries

```
"What's the biochar inoculation ratio?"
"How long does bread proofing take?"
"Tell me about the mushroom humidification process"
"Explain the biochar temperature control"
```

### Task Tracking

**Start Task**
```
"Start baking bread"
"Begin mushroom spawning"
"Start biochar heating"
```

**End Task**
```
"Finish baking, made 24 loaves for $180"
"Complete mushroom spawning, used 50 jars"
"End task"
```

**Pause/Resume**
```
"Pause the timer"
"Resume"
"Continue"
```

## Component Reference

### 1. WhisperSTT (`whisper_stt.py`)

Speech-to-text using OpenAI Whisper.

```python
from elle.voice import WhisperSTT

stt = WhisperSTT(model="tiny")  # tiny, base, small, medium

# Record audio
audio_path = await stt.record_audio(duration=5.0)

# Transcribe
text, metadata = await stt.transcribe(audio_path)
print(f"Transcribed: {text}")
print(f"Language: {metadata['language']}")
```

**Features:**
- Local speech recognition (no API calls)
- Multiple model sizes (tiny to large)
- Fallback to command-line whisper if library unavailable
- Automatic retry with exponential backoff

**Configuration:**
- `model`: "tiny" (39M), "base" (74M), "small" (244M), "medium" (769M), "large" (2.9G)
- `device`: "cpu" or "cuda" for GPU acceleration
- `language`: Language code (en, es, fr, etc.)
- `timeout`: Recording timeout in seconds

### 2. LLMParser (`llm_parser.py`)

Natural language command parsing with semantic understanding.

```python
from elle.voice import LLMParser

parser = LLMParser(use_llm=False)  # Pattern matching only

# Parse command
command = parser.parse("update bread SOP: increase proofing to 50 minutes")

print(f"Command type: {command.command_type}")  # "sop_update"
print(f"Entity: {command.entity}")              # "bread"
print(f"Intent: {command.intent}")              # "increase"
print(f"Value: {command.parameters['value']}")  # 50
print(f"Unit: {command.parameters['unit']}")    # "minutes"
print(f"Confidence: {command.confidence:.1%}")  # 0.9 (90%)
```

**Features:**
- Pattern-based parsing (default, no dependencies)
- LLM-enhanced parsing (with Anthropic Claude)
- Semantic understanding for command variations:
  - "increase proofing" = "make it longer" = "bump it up"
  - "set temperature" = "change temp" = "heat it up"
- Confidence scoring
- Parameter extraction (numbers, units, etc.)

**Supported Intents:**
- Duration changes: increase, decrease, set
- Temperature changes: increase_temp, decrease_temp
- Process control: start, stop, pause, resume
- Queries: ask

### 3. TextToSpeech (`tts.py`)

Text-to-speech responses using pyttsx3.

```python
from elle.voice import TextToSpeech, VoiceGender

tts = TextToSpeech(
    rate=150,  # words per minute
    volume=0.9,
    voice_gender=VoiceGender.FEMALE
)

# Speak
await tts.speak("Hello! What can I help you with?")

# Multiple texts
await tts.speak_queue([
    "Step 1: Prepare ingredients.",
    "Step 2: Mix thoroughly.",
    "Step 3: Let rise for 2 hours."
])

# Adjust settings
tts.set_rate(200)    # Faster speech
tts.set_volume(0.8)  # Quieter
```

**Features:**
- Cross-platform (Windows, Mac, Linux)
- Offline operation (no network required)
- Multiple voices (system dependent)
- Rate and volume control
- Fallback to console output if unavailable

### 4. WakeWordDetector (`wake_word.py`)

Detect wake words to activate listening.

```python
from elle.voice import WakeWordDetector

detector = WakeWordDetector(
    wake_words=["hey elle", "ok elle", "ella"],
    confidence_threshold=0.7
)

# Detect wake word
detected, confidence = detector.detect("hey elle, start baking")
print(f"Detected: {detected}")      # True
print(f"Confidence: {confidence}")  # 0.95

# Extract command after wake word
command = detector.extract_command("hey elle, show bread SOP")
print(f"Command: {command}")        # "show bread SOP"
```

**Features:**
- Multiple wake word options
- Fuzzy matching (tolerates speech errors)
- Confidence scoring
- Levenshtein distance matching
- Command extraction

### 5. VoiceAssistant (`assistant.py`)

Complete voice assistant integrating all components.

```python
from elle.voice import VoiceAssistant

assistant = VoiceAssistant(
    sop_dir="elle/sops",
    whisper_model="tiny",
    tts_rate=150,
    use_llm_parser=False,  # Use pattern matching
    verbose=True
)

# Initialize
await assistant.initialize()

# Interactive mode (text input)
await assistant.interactive_mode()

# Or voice mode (requires microphone)
await assistant.wake_word_loop()
```

## Testing

### Unit Tests

Test individual components:

```bash
# Test STT
PYTHONPATH=. python elle/voice/whisper_stt.py

# Test Parser
PYTHONPATH=. python elle/voice/llm_parser.py

# Test TTS
PYTHONPATH=. python elle/voice/tts.py

# Test Wake Word
PYTHONPATH=. python elle/voice/wake_word.py
```

### Integration Tests

```bash
# Interactive mode
PYTHONPATH=. python elle/voice/assistant.py interactive

# Voice mode (requires audio hardware)
PYTHONPATH=. python elle/voice/assistant.py voice
```

### Test Variations

Test command variations automatically:

```python
from elle.voice import CommandVariationGenerator, LLMParser

parser = LLMParser()

# Test SOP update variations
variations = CommandVariationGenerator.get_sop_update_variations(
    "bread", 50, "minutes"
)

for text in variations:
    cmd = parser.parse(text)
    print(f"✓ {text}")
    assert cmd.command_type == "sop_update"
    assert cmd.entity == "bread"
    assert cmd.parameters["value"] == 50
```

## Performance

| Operation | Latency | Notes |
|-----------|---------|-------|
| Wake word detection | <50ms | Pattern matching only |
| Speech recording (5s) | 5.0s | Real-time audio capture |
| Transcription (Whisper tiny) | ~3-5s | Depends on audio length |
| Command parsing | <10ms | Pattern matching |
| TTS synthesis | ~2-5s | Depends on text length |
| **Total (wake→response)** | ~10-20s | Typical user interaction |

## Graceful Degradation

The voice system is designed to work without optional dependencies:

| Component | Without Library | Fallback |
|-----------|-----------------|----------|
| Whisper STT | No speech input | Use command-line `whisper` CLI |
| pyttsx3 TTS | No voice output | Print responses to console |
| LLM Parser | No LLM parsing | Use regex pattern matching |
| PyAudio | No microphone input | Accept pre-recorded audio files |

Example:
```python
# Works even without PyAudio
audio_path = "pre_recorded.wav"
text, metadata = await stt.transcribe(audio_path)
```

## Troubleshooting

### Whisper Not Found

```bash
pip install openai-whisper
```

If you get CUDA errors, force CPU:
```python
stt = WhisperSTT(device="cpu")
```

### PyAudio Installation Issues

```bash
# On Ubuntu/Debian
sudo apt-get install portaudio19-dev python3-dev

# Then install
pip install pyaudio

# On macOS
brew install portaudio
pip install pyaudio
```

### No Audio Input

1. Check system audio settings
2. Try recording test: `sox -d test.wav` (requires sox)
3. Verify microphone is connected and enabled
4. Use `interactive` mode for testing without audio

### Whisper Accuracy Issues

1. Use larger model: `WhisperSTT(model="base")` or `"small"`
2. Speak clearly and slowly
3. Reduce background noise
4. Use language parameter if not English

## Future Enhancements

1. **Custom Wake Words**: Train on custom wake phrases
2. **Multi-Language Support**: Support Spanish, French, etc.
3. **Voice Profiles**: Different voice preferences per user
4. **Context Persistence**: Remember conversation context
5. **Streaming Responses**: Start speaking response while still processing
6. **Intent Confidence Routing**: Ask for confirmation on low-confidence commands
7. **Audio Feedback**: Beep/tone to indicate listening/processing
8. **Wake Word Learning**: Improve detection over time

## Integration with Elle Core

The voice interface integrates with existing Elle components:

```
Voice Assistant
    ↓
VoiceSOPEditor (command execution)
    ↓
    ├─ SOP (system operation procedures)
    ├─ TaskTracker (real-time tracking)
    └─ DecisionEngine (via HoloLoom RAG)
```

Commands flow:
1. Voice → Transcription → Parsed Command
2. Command → VoiceSOPEditor → SOP Update/Query
3. SOP → TaskTracker → Progress Tracking
4. Response → TTS → Voice Output

## API Reference

See individual module docstrings:
- `WhisperSTT`: Speech-to-text
- `LLMParser`: Natural language parsing
- `TextToSpeech`: Voice responses
- `WakeWordDetector`: Wake word detection
- `VoiceAssistant`: Complete integration

## Examples

### Example 1: Basic Voice Command

```python
import asyncio
from elle.voice import VoiceAssistant

async def main():
    assistant = VoiceAssistant(
        whisper_model="tiny",
        use_llm_parser=False,
        verbose=True
    )
    await assistant.initialize()

    # Test command
    response = await assistant.process_voice_input(
        "update bread SOP: increase proofing to 50 minutes"
    )
    print(f"Response: {response}")

asyncio.run(main())
```

### Example 2: Parse Command Variations

```python
from elle.voice import LLMParser, CommandVariationGenerator

parser = LLMParser(use_llm=False)

# Get variations
variations = CommandVariationGenerator.get_sop_update_variations(
    "bread", 45, "minutes"
)

# Parse all
for text in variations:
    cmd = parser.parse(text)
    print(f"✓ Parsed: {cmd.command_type} - {cmd.entity}")
    print(f"  Value: {cmd.parameters.get('value')}")
```

### Example 3: Wake Word Detection

```python
import asyncio
from elle.voice import WakeWordDetector

async def main():
    detector = WakeWordDetector()

    # Test various inputs
    inputs = [
        "hey elle, show bread SOP",
        "ella, what time is it?",
        "hello there",
    ]

    for text in inputs:
        detected, conf = detector.detect(text)
        print(f"'{text}' → Detected: {detected} ({conf:.1%})")

asyncio.run(main())
```

## License

Part of Elle Core • Comprehensive Operational Intelligence for Coz

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review demo scripts: `demos/demo_voice_assistant.py`
3. Check existing voice commands in Elle Core
4. Review component docstrings for API details
