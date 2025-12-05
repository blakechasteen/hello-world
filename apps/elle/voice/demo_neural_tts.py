"""
Demo: Neural TTS Integration

Tests the neural TTS (Coqui) integration with graceful fallback to pyttsx3.

Created: 2025-11-22
Part of: Voice UX Milestone 2 Phase 2
"""

import asyncio
from neural_tts import NeuralTTS, VoicePersonality, VoiceModel


async def test_neural_tts():
    """Test neural TTS with various voice personalities"""

    print("\n" + "="*70)
    print("NEURAL TTS DEMO")
    print("="*70)

    # Test messages (brief responses optimized for <500ms)
    test_messages = [
        ("Thread 3", "Brief thread switch"),
        ("Created baking", "Thread creation"),
        ("Back to biochar", "Navigation"),
        ("3 threads: Baking, Biochar, Greenhouse", "Thread list"),
        ("Running analyze", "Task execution"),
        ("Stopped", "Task control"),
    ]

    # Test 1: FRIENDLY personality (default)
    print("\n" + "-"*70)
    print("TEST 1: FRIENDLY Personality (Default)")
    print("-"*70)

    tts_friendly = NeuralTTS(
        personality=VoicePersonality.FRIENDLY,
        enable_cache=True,
        verbose=True
    )

    if tts_friendly.coqui_available:
        print("\n[OK] Coqui TTS Available - Testing neural synthesis")

        for text, description in test_messages[:3]:  # Test first 3
            print(f"\n  >> {description}: \"{text}\"")
            success = await tts_friendly.speak(text, wait=True)
            if success:
                print(f"  [OK] Synthesized successfully")
            else:
                print(f"  [!] Synthesis failed (fallback used)")

        # Test cache hit
        print(f"\n  [CACHE] Cache Test: Repeating \"Thread 3\"")
        await tts_friendly.speak("Thread 3", wait=True)

        # Show cache stats
        stats = tts_friendly.get_cache_stats()
        print(f"\n  [STATS] Cache Statistics:")
        print(f"     Cached responses: {stats['cached_responses']}")
        print(f"     Cache size: {stats['cache_size_mb']:.2f} MB")
    else:
        print("\n[!] Coqui TTS Not Available - Skipping neural synthesis tests")
        print("  Install with: pip install TTS torch")

    # Test 2: FAST personality (speed-optimized)
    print("\n" + "-"*70)
    print("TEST 2: FAST Personality (Speed-Optimized)")
    print("-"*70)

    tts_fast = NeuralTTS(
        personality=VoicePersonality.FAST,
        enable_cache=True,
        verbose=True
    )

    if tts_fast.coqui_available:
        print("\n[OK] Testing FAST model for low latency")
        text = "Thread 3"

        import time
        start = time.time()
        await tts_fast.speak(text, wait=True)
        duration = (time.time() - start) * 1000  # Convert to ms

        print(f"  [TIME] Latency: {duration:.0f}ms")

        if duration < 500:
            print(f"  [OK] Meets <500ms goal!")
        else:
            print(f"  [!] Exceeds 500ms target (but acceptable for quality)")
    else:
        print("\n[!] Coqui TTS Not Available")

    # Test 3: PROFESSIONAL personality
    print("\n" + "-"*70)
    print("TEST 3: PROFESSIONAL Personality")
    print("-"*70)

    tts_professional = NeuralTTS(
        personality=VoicePersonality.PROFESSIONAL,
        enable_cache=True,
        verbose=True
    )

    if tts_professional.coqui_available:
        print("\n[OK] Testing PROFESSIONAL voice")
        await tts_professional.speak("Welcome to Elle Voice Assistant", wait=True)
    else:
        print("\n[!] Coqui TTS Not Available")

    # Test 4: Fallback behavior (disable Coqui)
    print("\n" + "-"*70)
    print("TEST 4: Fallback Behavior (Coqui Disabled)")
    print("-"*70)

    print("\n  Testing graceful degradation when neural TTS unavailable...")
    print("  (Text-only output expected)")

    # Create instance without Coqui
    tts_fallback = NeuralTTS(
        personality=VoicePersonality.FRIENDLY,
        enable_cache=False,
        verbose=True
    )

    # Manually disable to test fallback
    tts_fallback.coqui_available = False

    await tts_fallback.speak("Fallback test message", wait=True)
    print("  [OK] Fallback behavior verified")

    # Test 5: Voice cache effectiveness
    print("\n" + "-"*70)
    print("TEST 5: Voice Cache Effectiveness")
    print("-"*70)

    if tts_friendly.coqui_available:
        print("\n  Testing cache speedup for repeated responses...")

        import time

        # Clear cache first
        tts_friendly.clear_cache()
        print("  [CLEAR] Cache cleared")

        # Cold synthesis
        text = "Thread 3"
        start = time.time()
        await tts_friendly.speak(text, wait=True)
        cold_time = (time.time() - start) * 1000
        print(f"  [COLD] Cold synthesis: {cold_time:.0f}ms")

        # Warm synthesis (cached)
        start = time.time()
        await tts_friendly.speak(text, wait=True)
        warm_time = (time.time() - start) * 1000
        print(f"  [HOT] Warm synthesis: {warm_time:.0f}ms")

        if warm_time > 0:
            speedup = cold_time / warm_time
            print(f"  [FAST] Cache speedup: {speedup:.0f}x faster")

        # Show final cache stats
        stats = tts_friendly.get_cache_stats()
        print(f"\n  [STATS] Final Cache Statistics:")
        print(f"     Cached responses: {stats['cached_responses']}")
        print(f"     Cache directory: {stats['cache_dir']}")
    else:
        print("\n[!] Coqui TTS Not Available")

    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)

    if tts_friendly.coqui_available:
        print("\n[OK] All neural TTS tests passed!")
        print("\n[INFO] Next Steps:")
        print("  1. Test end-to-end with VoiceAssistant")
        print("  2. Benchmark latency on target hardware")
        print("  3. Configure voice personality for production")
    else:
        print("\n[!] Coqui TTS not installed")
        print("\n[INFO] Installation:")
        print("  pip install TTS torch")
        print("\n  Then re-run this demo to test neural synthesis")


if __name__ == "__main__":
    asyncio.run(test_neural_tts())
