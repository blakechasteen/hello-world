#!/usr/bin/env python3
"""
Quick YouTube Transcript Fetcher
=================================
Pulls transcript from YouTube video and saves it to a file.

Usage:
    python fetch_youtube_transcript.py
"""

import asyncio
from pathlib import Path

# Try to use HoloLoom's YouTube spinner
try:
    from HoloLoom.spinningWheel.youtube_spinner import (
        YouTubeSpinner,
        YouTubeSpinnerConfig
    )
    HOLOLOOM_SPINNER = True
except ImportError:
    HOLOLOOM_SPINNER = False
    print("⚠️  HoloLoom spinner not available, trying direct approach...")


async def fetch_transcript_hololoom(url: str, output_file: str = "transcript.txt"):
    """Fetch transcript using HoloLoom spinner"""
    config = YouTubeSpinnerConfig(
        chunk_duration=None,  # Single shard
        include_timestamps=True,
        extract_entities=True
    )

    spinner = YouTubeSpinner(config)

    print(f"🎥 Fetching transcript from: {url}")
    print("⏳ Please wait...")

    try:
        result = await spinner.spin({'url': url, 'languages': ['en']})

        if not result.shards:
            print("❌ No transcript found!")
            return

        print(f"✅ Found {len(result.shards)} shard(s)")

        # Save to file
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"YouTube Transcript\n")
            f.write(f"{'=' * 50}\n")
            f.write(f"URL: {url}\n\n")

            for i, shard in enumerate(result.shards, 1):
                f.write(f"\n--- Shard {i} ---\n")
                f.write(f"{shard.text}\n")

                if shard.metadata:
                    f.write(f"\nMetadata:\n")
                    for key, value in shard.metadata.items():
                        f.write(f"  {key}: {value}\n")

                if shard.entities:
                    f.write(f"\nEntities: {', '.join(shard.entities)}\n")

        print(f"💾 Saved to: {output_path}")

        # Also print to console
        print(f"\n{'=' * 50}")
        print("TRANSCRIPT:")
        print('=' * 50)
        for shard in result.shards:
            print(shard.text)
            print()

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def fetch_transcript_direct(url: str, output_file: str = "transcript.txt"):
    """Fallback: fetch transcript using youtube-transcript-api directly"""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        import re
        from urllib.parse import urlparse, parse_qs
    except ImportError:
        print("❌ youtube-transcript-api not installed!")
        print("Install with: pip install youtube-transcript-api")
        return

    # Extract video ID
    if '/shorts/' in url:
        video_id = url.split('/shorts/')[-1].split('?')[0]
    elif 'watch?v=' in url:
        parsed = urlparse(url)
        video_id = parse_qs(parsed.query)['v'][0]
    elif 'youtu.be/' in url:
        video_id = url.split('youtu.be/')[-1].split('?')[0]
    else:
        video_id = url

    print(f"🎥 Video ID: {video_id}")
    print("⏳ Fetching transcript...")

    try:
        # Use the correct API: create instance and fetch
        api = YouTubeTranscriptApi()
        transcript_obj = api.fetch(video_id, languages=['en'])

        # Convert to raw data (list of dicts with 'text', 'start', 'duration')
        transcript = transcript_obj.to_raw_data()

        # Save to file
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"YouTube Transcript\n")
            f.write(f"{'=' * 50}\n")
            f.write(f"Video ID: {video_id}\n")
            f.write(f"URL: {url}\n\n")

            full_text = []
            for segment in transcript:
                timestamp = f"[{int(segment['start']//60):02d}:{int(segment['start']%60):02d}]"
                f.write(f"{timestamp} {segment['text']}\n")
                full_text.append(segment['text'])

            f.write(f"\n{'=' * 50}\n")
            f.write("FULL TRANSCRIPT (no timestamps):\n")
            f.write('=' * 50 + '\n')
            f.write(' '.join(full_text))

        print(f"✅ Transcript saved to: {output_path}")

        # Print to console
        print(f"\n{'=' * 50}")
        print("TRANSCRIPT:")
        print('=' * 50)
        print(' '.join(full_text))

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def main():
    url = "https://youtube.com/shorts/Flqljv8clcY?si=dY_rwyIFjhlrXOxM"
    output_file = "youtube_transcript.txt"

    if HOLOLOOM_SPINNER:
        await fetch_transcript_hololoom(url, output_file)
    else:
        await fetch_transcript_direct(url, output_file)


if __name__ == "__main__":
    asyncio.run(main())
