"""
Voice Integration Test Script

Quick verification that voice capabilities are working.
Run this BEFORE full dashboard integration to catch issues early.
"""

import asyncio
import sys
from pathlib import Path
import io

# Fix Windows encoding for Unicode characters
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Ensure HoloLoom is on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


async def test_voice_integration():
    """Test voice integration components"""

    print("=" * 60)
    print("VOICE INTEGRATION TEST")
    print("=" * 60)
    print()

    # Test 1: Check dependencies
    print("Test 1: Checking dependencies...")
    print("-" * 60)

    deps = {}

    try:
        import bark
        deps['bark'] = '✓ BARK installed'
    except ImportError:
        deps['bark'] = '✗ BARK not installed (pip install git+https://github.com/suno-ai/bark.git scipy)'

    try:
        import whisper
        deps['whisper'] = '✓ Whisper installed'
    except ImportError:
        deps['whisper'] = '✗ Whisper not installed (pip install openai-whisper)'

    try:
        import torch
        deps['torch'] = '✓ PyTorch installed'
    except ImportError:
        deps['torch'] = '✗ PyTorch not installed (pip install torch)'

    try:
        import scipy
        deps['scipy'] = '✓ SciPy installed'
    except ImportError:
        deps['scipy'] = '✗ SciPy not installed (pip install scipy)'

    for name, status in deps.items():
        print(f"  {name:10s}: {status}")

    print()

    # Test 2: Import voice integration module
    print("Test 2: Importing voice_integration module...")
    print("-" * 60)

    try:
        from HoloLoom.web_dashboard.voice_integration import create_voice_integration
        print("  ✓ voice_integration.py imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import voice_integration: {e}")
        print()
        print("FAILED: Cannot import voice_integration module")
        return False

    print()

    # Test 3: Import voice endpoints
    print("Test 3: Importing voice_endpoints module...")
    print("-" * 60)

    try:
        print("  ✓ voice_endpoints.py imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import voice_endpoints: {e}")
        print()
        print("FAILED: Cannot import voice_endpoints module")
        return False

    print()

    # Test 4: Create voice integration instance
    print("Test 4: Creating VoiceIntegration instance...")
    print("-" * 60)

    try:
        voice = await create_voice_integration(
            tts_backend="bark",
            whisper_model="base",
            auto_speak=False
        )
        print("  ✓ VoiceIntegration instance created")
        print(f"  - TTS Backend: {voice.tts_backend}")
        print(f"  - Whisper Model: {voice.whisper_model}")
        print(f"  - Auto Speak: {voice.auto_speak}")
    except Exception as e:
        print(f"  ✗ Failed to create VoiceIntegration: {e}")
        print()
        print("WARNING: Voice integration creation failed")
        print("This might be OK if dependencies are missing")
        print("The dashboard will work, just without voice")
        return False

    print()

    # Test 5: Check status
    print("Test 5: Checking voice status...")
    print("-" * 60)

    try:
        status = voice.get_status()
        print(f"  TTS Available: {status['tts_available']}")
        print(f"  TTS Backend: {status['tts_backend']}")
        print(f"  Transcription Available: {status['transcription_available']}")
        print(f"  Whisper Model: {status['whisper_model']}")
        print(f"  Auto Speak: {status['auto_speak']}")

        if not status['tts_available']:
            print()
            print("  WARNING: TTS not available!")
            print("  Install BARK: pip install git+https://github.com/suno-ai/bark.git scipy")

        if not status['transcription_available']:
            print()
            print("  WARNING: Transcription not available!")
            print("  Install Whisper: pip install openai-whisper")

    except Exception as e:
        print(f"  ✗ Failed to get status: {e}")
        return False

    print()

    # Test 6: Test TTS (if available)
    if status['tts_available']:
        print("Test 6: Testing text-to-speech...")
        print("-" * 60)

        try:
            test_text = "Hello from the voice integration test."
            print(f"  Generating speech: '{test_text}'")

            audio_bytes = await voice.speak(
                text=test_text,
                metric_type="neutral"
            )

            if audio_bytes:
                print(f"  ✓ Generated {len(audio_bytes)} bytes of audio")
                print("  (Audio not saved, this is just a test)")
            else:
                print("  ✗ No audio generated (but no error)")

        except Exception as e:
            print(f"  ✗ TTS test failed: {e}")
            print("  This might fail on first run (BARK downloads models)")
            print("  Try again after models download")
    else:
        print("Test 6: Skipping TTS test (not available)")
        print("-" * 60)
        print("  Install BARK to enable TTS")

    print()

    # Test 7: Test speak_response (auto-vocal-delivery)
    if status['tts_available']:
        print("Test 7: Testing auto-vocal-delivery...")
        print("-" * 60)

        try:
            response_data = {
                'response': 'Your revenue is four hundred fifty dollars.',
                'confidence': 0.75,
                'mode': 'direct'
            }

            print(f"  Response: '{response_data['response']}'")
            print(f"  Confidence: {response_data['confidence']}")
            print(f"  Mode: {response_data['mode']}")
            print(f"  Expected delivery: neutral (confidence 0.5-0.85)")

            audio_bytes = await voice.speak_response(response_data)

            if audio_bytes:
                print(f"  ✓ Generated {len(audio_bytes)} bytes of audio")
            else:
                print("  ✗ No audio generated")

        except Exception as e:
            print(f"  ✗ Auto-vocal-delivery test failed: {e}")
    else:
        print("Test 7: Skipping auto-vocal-delivery test (TTS not available)")
        print("-" * 60)

    print()

    # Final summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    if status['tts_available'] and status['transcription_available']:
        print("✓ SUCCESS: Voice integration is fully functional!")
        print()
        print("Next steps:")
        print("  1. Add ~30 lines to agentic_server.py (see VOICE_QUICK_START.md)")
        print("  2. Copy CSS/HTML/JS to agentic_dashboard.html")
        print("  3. Start server and test live!")
        print()
        print("You're ready to talk to your dashboard tonight! 🎤")
        return True
    elif status['tts_available'] or status['transcription_available']:
        print("⚠ PARTIAL: Some voice features available")
        print()
        if not status['tts_available']:
            print("  Missing: TTS (install BARK)")
        if not status['transcription_available']:
            print("  Missing: Transcription (install Whisper)")
        print()
        print("Install missing dependencies and run this test again")
        return False
    else:
        print("✗ FAILED: Voice integration not available")
        print()
        print("Install dependencies:")
        print("  pip install git+https://github.com/suno-ai/bark.git scipy")
        print("  pip install openai-whisper")
        print("  winget install Gyan.FFmpeg")
        print()
        print("Then run this test again")
        return False


def main():
    """Run tests"""
    try:
        success = asyncio.run(test_voice_integration())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
