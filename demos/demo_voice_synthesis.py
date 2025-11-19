"""
Demo: Voice Synthesis with Elle Game Engine

This demo shows how to use Elle's voice synthesis system to generate
NPC dialogue audio with different voices, emotions, and backends.

Usage:
    PYTHONPATH=. python demos/demo_voice_synthesis.py

Author: Elle Game Engine Team
Date: 2025-11-16
"""

import asyncio
from pathlib import Path

from apps.elle_game_engine.voice import (
    create_voice_engine,
    VoiceProfile,
    Emotion,
    AudioFormat,
    TTSBackend,
)


async def main():
    print("=" * 60)
    print("Elle Game Engine - Voice Synthesis Demo")
    print("=" * 60)
    print()

    # Create output directory
    output_dir = Path("demos/output/voice_samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    print()

    # Initialize voice engine (using dummy backend for demo)
    print("🔊 Initializing voice engine (dummy backend)...")
    voice_engine = create_voice_engine(
        backend="dummy",  # Change to "openai", "elevenlabs", etc. for real synthesis
        enable_cache=True
    )
    print("✅ Voice engine ready!")
    print()

    # ========================================================================
    # Example 1: Create Voice Profiles for NPCs
    # ========================================================================

    print("=" * 60)
    print("Example 1: Creating Voice Profiles")
    print("=" * 60)
    print()

    # Innkeeper - warm and friendly
    innkeeper_profile = VoiceProfile(
        voice_id="alloy",  # OpenAI voice
        pitch=1.0,
        speed=1.0,
        emotion=Emotion.WARM,
        stability=0.6,
        similarity_boost=0.75
    )
    voice_engine.register_voice_profile("innkeeper", innkeeper_profile)
    print("✅ Created voice profile for innkeeper (warm, normal pitch)")

    # Guard - stern and authoritative
    guard_profile = VoiceProfile(
        voice_id="onyx",  # OpenAI voice (deeper)
        pitch=0.8,  # Lower pitch
        speed=0.9,  # Slightly slower
        emotion=Emotion.STERN,
        stability=0.7,
        similarity_boost=0.8
    )
    voice_engine.register_voice_profile("guard", guard_profile)
    print("✅ Created voice profile for guard (stern, low pitch)")

    # Merchant - excited and fast-talking
    merchant_profile = VoiceProfile(
        voice_id="nova",  # OpenAI voice (energetic)
        pitch=1.2,  # Higher pitch
        speed=1.3,  # Faster
        emotion=Emotion.EXCITED,
        stability=0.4,
        similarity_boost=0.7
    )
    voice_engine.register_voice_profile("merchant", merchant_profile)
    print("✅ Created voice profile for merchant (excited, high pitch, fast)")

    # Wizard - cryptic and slow
    wizard_profile = VoiceProfile(
        voice_id="echo",  # OpenAI voice (mysterious)
        pitch=0.9,
        speed=0.8,  # Slower for mystery
        emotion=Emotion.CRYPTIC,
        stability=0.8,
        similarity_boost=0.9
    )
    voice_engine.register_voice_profile("wizard", wizard_profile)
    print("✅ Created voice profile for wizard (cryptic, slow)")
    print()

    # ========================================================================
    # Example 2: Synthesize Dialogue
    # ========================================================================

    print("=" * 60)
    print("Example 2: Synthesizing NPC Dialogue")
    print("=" * 60)
    print()

    dialogues = [
        ("innkeeper", "Welcome to my inn, traveler! Make yourself at home."),
        ("guard", "Halt! State your business in this town."),
        ("merchant", "Oh wonderful! I have the BEST deals today! Come, come!"),
        ("wizard", "The ancient secrets... they whisper in the darkness..."),
    ]

    for npc_id, text in dialogues:
        print(f"🎙️  Synthesizing: {npc_id}")
        print(f"   Text: \"{text}\"")

        # Synthesize audio
        result = voice_engine.synthesize(
            text=text,
            npc_id=npc_id,
            format=AudioFormat.WAV
        )

        # Save audio file
        filename = output_dir / f"{npc_id}_greeting.wav"
        with open(filename, "wb") as f:
            f.write(result.audio_data)

        print(f"   ✅ Saved: {filename}")
        print(f"   Duration: {result.duration_seconds:.1f}s")
        print(f"   Cached: {result.cached}")
        print()

    # ========================================================================
    # Example 3: Emotion Overrides
    # ========================================================================

    print("=" * 60)
    print("Example 3: Emotion Overrides")
    print("=" * 60)
    print()

    # Same NPC, different emotions
    innkeeper_dialogues = [
        (Emotion.WARM, "Welcome back, friend!"),
        (Emotion.ANNOYED, "Can't you see I'm busy?"),
        (Emotion.SAD, "Times are hard these days..."),
        (Emotion.EXCITED, "You won't believe what just happened!"),
    ]

    for emotion, text in innkeeper_dialogues:
        print(f"🎙️  Innkeeper ({emotion.value})")
        print(f"   Text: \"{text}\"")

        result = voice_engine.synthesize(
            text=text,
            npc_id="innkeeper",
            emotion=emotion,  # Override base emotion
            format=AudioFormat.WAV
        )

        filename = output_dir / f"innkeeper_{emotion.value}.wav"
        with open(filename, "wb") as f:
            f.write(result.audio_data)

        print(f"   ✅ Saved: {filename}")
        print()

    # ========================================================================
    # Example 4: Voice Caching
    # ========================================================================

    print("=" * 60)
    print("Example 4: Voice Caching")
    print("=" * 60)
    print()

    # Synthesize same phrase multiple times
    phrase = "Hello, traveler!"

    print(f"📝 Phrase: \"{phrase}\"")
    print()

    # First synthesis (cold cache)
    print("🔹 First synthesis (cold cache)...")
    result1 = voice_engine.synthesize(
        text=phrase,
        npc_id="innkeeper",
        format=AudioFormat.MP3
    )
    print(f"   Cached: {result1.cached}")
    print(f"   Size: {len(result1.audio_data)} bytes")
    print()

    # Second synthesis (should hit cache)
    print("🔹 Second synthesis (should hit cache)...")
    result2 = voice_engine.synthesize(
        text=phrase,
        npc_id="innkeeper",
        format=AudioFormat.MP3
    )
    print(f"   Cached: {result2.cached}")
    print(f"   Size: {len(result2.audio_data)} bytes")
    print()

    # Verify same audio
    assert result1.audio_data == result2.audio_data
    print("✅ Cache working! Same audio returned.")
    print()

    # ========================================================================
    # Example 5: Cache Statistics
    # ========================================================================

    print("=" * 60)
    print("Example 5: Cache Statistics")
    print("=" * 60)
    print()

    stats = voice_engine.get_cache_stats()

    print(f"📊 Voice Cache Stats:")
    print(f"   Entries: {stats['num_entries']}")
    print(f"   Size: {stats['total_size_mb']:.2f}MB / {stats['max_size_mb']:.0f}MB")
    print(f"   Cache Dir: {stats['cache_dir']}")
    print()

    # ========================================================================
    # Example 6: Different Audio Formats
    # ========================================================================

    print("=" * 60)
    print("Example 6: Audio Format Conversion")
    print("=" * 60)
    print()

    formats = [
        (AudioFormat.WAV, "lossless, large file"),
        (AudioFormat.MP3, "compressed, smaller"),
        (AudioFormat.OGG, "compressed, smallest"),
    ]

    text = "Greetings, adventurer!"

    for audio_format, description in formats:
        result = voice_engine.synthesize(
            text=text,
            npc_id="guard",
            format=audio_format
        )

        filename = output_dir / f"guard_greeting.{audio_format.value}"
        with open(filename, "wb") as f:
            f.write(result.audio_data)

        print(f"🔹 {audio_format.value.upper()}: {len(result.audio_data):,} bytes ({description})")

    print()

    # ========================================================================
    # Summary
    # ========================================================================

    print("=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print()
    print(f"✅ Generated {stats['num_entries']} voice samples")
    print(f"✅ Output directory: {output_dir}")
    print()
    print("💡 Tips:")
    print("   - Change ELLE_VOICE_BACKEND to 'openai', 'elevenlabs', etc. for real TTS")
    print("   - Set API keys via environment variables")
    print("   - Cache saves repeated synthesis (100x faster)")
    print("   - Use emotion overrides for dynamic dialogue")
    print()


if __name__ == "__main__":
    asyncio.run(main())
