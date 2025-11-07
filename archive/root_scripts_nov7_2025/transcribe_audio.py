"""
Quick Audio Transcription with Whisper (Large Model)
===================================================
"""

import asyncio
import sys
from pathlib import Path
from HoloLoom.spinningWheel.whisper_spinner import WhisperSpinner


async def transcribe(audio_path: Path):
    """Transcribe audio file with large model"""

    print(f"Audio file: {audio_path.name}")
    print(f"Model: large (most accurate)")
    print(f"Loading Whisper model (this may take a minute)...\n")

    # Create spinner with large model
    spinner = WhisperSpinner(
        model_size="large",
        language=None,  # Auto-detect
        enable_word_timestamps=True,
        importance_threshold=0.2
    )

    # Check availability
    if not spinner.is_available():
        print("ERROR: Whisper not available!")
        print("\nInstall with:")
        print("  pip install openai-whisper")
        print("\nOptional (for GPU acceleration):")
        print("  pip install torch")
        return

    print("Model loaded! Starting transcription...\n")

    # Transcribe
    result = await spinner.spin(audio_path)

    if not result.success:
        error_msg = getattr(result, 'error_message', 'Unknown error')
        print(f"Transcription failed: {error_msg}")
        return

    # Results
    print("=" * 70)
    print("TRANSCRIPTION COMPLETE")
    print("=" * 70)

    full_shard = result.shards[0]

    print(f"\nStatistics:")
    print(f"   Language: {full_shard.metadata['language']}")
    print(f"   Duration: {full_shard.metadata['duration']:.1f} seconds")
    print(f"   Segments: {full_shard.metadata['segment_count']}")
    print(f"   Total shards: {len(result.shards)}")
    print(f"   Processing time: {result.duration:.1f}s")
    print(f"   Importance score: {full_shard.metadata['importance_score']:.2f}")

    print(f"\nFull Transcription:")
    print("-" * 70)
    print(full_shard.text)
    print("-" * 70)

    # Show segments with timecodes
    print(f"\nSegments with Timecodes (first 10):")
    print("=" * 70)
    for i, shard in enumerate(result.shards[1:11], start=1):  # Skip full, show 10
        timecode = shard.metadata['timecode']
        confidence = shard.metadata.get('confidence', 0.0)
        text = shard.text
        print(f"\n[{timecode}] (confidence: {confidence:.2f})")
        print(f"  {text}")

    if len(result.shards) > 11:
        print(f"\n... and {len(result.shards) - 11} more segments")

    # Save to file
    output_path = audio_path.with_suffix('.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"Transcription of: {audio_path.name}\n")
        f.write(f"Language: {full_shard.metadata['language']}\n")
        f.write(f"Duration: {full_shard.metadata['duration']:.1f}s\n")
        f.write("\n" + "=" * 70 + "\n\n")
        f.write(full_shard.text)
        f.write("\n\n" + "=" * 70 + "\n")
        f.write("\nSegments with Timecodes:\n")
        f.write("=" * 70 + "\n\n")
        for shard in result.shards[1:]:  # All segments
            timecode = shard.metadata['timecode']
            f.write(f"[{timecode}] {shard.text}\n\n")

    print(f"\nSaved transcription to: {output_path}")

    # Export SRT
    srt_path = audio_path.with_suffix('.srt')
    spinner.export_srt(
        [seg for seg in [
            type('Seg', (), {
                'start': s.metadata['start_time'],
                'end': s.metadata['end_time'],
                'text': s.text,
                'to_srt': lambda self, i: f"{i}\n{s.metadata['timecode']} --> {s.metadata['timecode']}\n{s.text}\n"
            })()
            for s in result.shards[1:]  # Skip full transcription
        ]],
        srt_path
    )

    print(f"Saved SRT subtitles to: {srt_path}")

    return result


async def main():
    """Main entry point"""

    if len(sys.argv) < 2:
        print("Whisper Audio Transcription (Large Model)")
        print("=" * 70)
        audio_path = input("\nEnter path to audio file: ").strip().strip('"')
    else:
        audio_path = sys.argv[1]

    audio_path = Path(audio_path)

    if not audio_path.exists():
        print(f"\nERROR: File not found: {audio_path}")
        print("\nSupported formats: wav, mp3, m4a, flac, ogg, opus")
        return

    try:
        await transcribe(audio_path)
    except KeyboardInterrupt:
        print("\n\nTranscription cancelled by user")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
