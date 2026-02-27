"""
Demo: Whisper Spinner - Audio Transcription
==========================================

Demonstrates transcribing audio files with the WhisperSpinner.
"""

import asyncio
from pathlib import Path
from hololoom.spinningWheel.whisper_spinner import (
    WhisperSpinner,
    transcribe_audio,
    transcribe_with_timecodes
)


async def demo_basic_transcription(audio_path: Path):
    """Basic transcription example"""
    print("=== Basic Transcription ===\n")

    # Create spinner
    spinner = WhisperSpinner(
        model_size="base",  # Options: tiny, base, small, medium, large
        language=None,       # Auto-detect language
        enable_word_timestamps=True
    )

    # Check if available
    if not spinner.is_available():
        print("❌ Whisper not available. Install with: pip install openai-whisper")
        return

    # Transcribe
    print(f"Transcribing: {audio_path.name}")
    result = await spinner.spin(audio_path)

    if not result.success:
        print(f"❌ Transcription failed: {result.error}")
        return

    print(f"✅ Transcription successful!")
    print(f"   Shards created: {len(result.shards)}")
    print(f"   Duration: {result.duration:.2f}s\n")

    # Show full transcription
    full_shard = result.shards[0]  # First shard is always full transcription
    print("Full Text:")
    print("-" * 60)
    print(full_shard.text)
    print("-" * 60)
    print(f"\nMetadata:")
    print(f"  Language: {full_shard.metadata['language']}")
    print(f"  Duration: {full_shard.metadata['duration']:.2f}s")
    print(f"  Segments: {full_shard.metadata['segment_count']}")
    print(f"  Importance: {full_shard.metadata['importance_score']:.2f}")

    # Show first few segments with timecodes
    print("\n\nFirst 5 Segments with Timecodes:")
    print("=" * 60)
    for i, shard in enumerate(result.shards[1:6], start=1):  # Skip full, show 5 segments
        timecode = shard.metadata['timecode']
        text = shard.text[:80] + "..." if len(shard.text) > 80 else shard.text
        print(f"[{timecode}] {text}")

    return result


async def demo_with_srt_export(audio_path: Path):
    """Transcribe and export to SRT subtitle format"""
    print("\n\n=== Transcription with SRT Export ===\n")

    result = await transcribe_with_timecodes(
        audio_path,
        export_srt=True
    )

    print(f"✅ Transcription complete!")
    print(f"   Text: {len(result['text'])} characters")
    print(f"   Language: {result['language']}")
    print(f"   Duration: {result['duration']:.2f}s")
    print(f"   Segments: {len(result['segments'])}")

    if result['srt_path']:
        print(f"   SRT file: {result['srt_path']}")

    # Show first few segments
    print("\nFirst 3 segments:")
    for seg in result['segments'][:3]:
        print(f"  [{seg['timecode']}] {seg['text'][:60]}...")


async def demo_convenience_function(audio_path: Path):
    """Quick transcription with convenience function"""
    print("\n\n=== Quick Transcription (Convenience Function) ===\n")

    # One-liner transcription
    result = await transcribe_audio(
        audio_path,
        model_size="tiny",  # Fastest model
        language="en"       # Specify language for speed
    )

    if result.success:
        print(f"✅ Quick transcription successful!")
        print(f"   Shards: {len(result.shards)}")
        print(f"   Duration: {result.duration:.2f}s")
        print(f"\n   First 200 chars: {result.shards[0].text[:200]}...")


async def main():
    """Main demo"""
    print("Whisper Spinner Demo")
    print("=" * 60)

    # Check if user provided audio file
    audio_path = Path(input("\nEnter path to audio file (or 'demo' to see example): ").strip())

    if str(audio_path).lower() == 'demo':
        print("\n📝 Example Usage:\n")
        print("python demos/demo_whisper_spinner.py")
        print("\nThen provide path to audio file when prompted.")
        print("\nSupported formats: wav, mp3, m4a, flac, ogg, opus")
        print("\nExample paths:")
        print("  - C:\\Users\\blake\\Downloads\\audio.mp3")
        print("  - ./recordings/meeting_2025.wav")
        return

    if not audio_path.exists():
        print(f"❌ File not found: {audio_path}")
        return

    # Run demos
    try:
        # Basic transcription
        result = await demo_basic_transcription(audio_path)

        if result and result.success:
            # SRT export
            await demo_with_srt_export(audio_path)

            # Convenience function
            await demo_convenience_function(audio_path)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
