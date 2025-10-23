#!/usr/bin/env python3
"""
Simple YouTube Video Transcriber
Usage: python transcribe_simple.py [VIDEO_URL]
"""

import asyncio
import sys

# Direct import to avoid loading full HoloLoom dependencies
sys.path.insert(0, '/home/user/hello-world')
from HoloLoom.spinningWheel.youtube import transcribe_youtube


async def main():
    # Get video URL from command line or use default
    if len(sys.argv) > 1:
        video_url = sys.argv[1]
    else:
        # Example video - replace with your own
        video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        print(f"No URL provided, using example: {video_url}")
        print("Usage: python transcribe_simple.py YOUR_VIDEO_URL\n")

    print(f"Transcribing: {video_url}")
    print("-" * 80)

    try:
        # Transcribe the video
        shards = await transcribe_youtube(
            url=video_url,
            languages=['en'],
            chunk_duration=None  # Change to 60.0 for 60-second chunks
        )

        print(f"\n✓ Success! Generated {len(shards)} shard(s)\n")

        # Display information about the transcript
        for i, shard in enumerate(shards):
            print(f"{'='*80}")
            print(f"Shard {i+1}/{len(shards)}")
            print(f"{'='*80}")
            print(f"ID: {shard.id}")
            print(f"Language: {shard.metadata.get('language', 'unknown')}")
            print(f"Duration: {shard.metadata.get('duration', 0):.1f} seconds")
            print(f"Video ID: {shard.metadata.get('video_id', 'N/A')}")

            entities = shard.entities[:10]
            print(f"\nExtracted Entities ({len(shard.entities)} total):")
            print(f"  {', '.join(entities)}")
            if len(shard.entities) > 10:
                print(f"  ... and {len(shard.entities) - 10} more")

            print(f"\nTranscript Preview (first 300 chars):")
            print(f"  {shard.text[:300]}...")
            print()

        # Show full transcript for single shard
        if len(shards) == 1:
            print(f"\n{'='*80}")
            print("FULL TRANSCRIPT")
            print(f"{'='*80}\n")
            print(shards[0].text)
            print(f"\n{'='*80}")
            print(f"Total length: {len(shards[0].text)} characters")
            print(f"{'='*80}\n")

        # Save option
        response = input("\nSave transcript to file? (y/n): ").lower().strip()
        if response == 'y':
            video_id = shards[0].metadata.get('video_id', 'video')
            filename = f"transcript_{video_id}.txt"

            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"YouTube Video Transcript\n")
                f.write(f"{'='*80}\n\n")
                f.write(f"Video URL: {video_url}\n")
                f.write(f"Video ID: {video_id}\n")
                f.write(f"Language: {shards[0].metadata.get('language')}\n")
                f.write(f"Duration: {shards[0].metadata.get('duration'):.1f} seconds\n")
                f.write(f"Is Auto-Generated: {shards[0].metadata.get('is_generated')}\n\n")
                f.write(f"{'='*80}\n")
                f.write("TRANSCRIPT\n")
                f.write(f"{'='*80}\n\n")

                for shard in shards:
                    f.write(shard.text)
                    f.write("\n\n")

            print(f"✓ Saved to: {filename}")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print("  - Make sure the video has captions/transcripts enabled")
        print("  - Try a different video to test")
        print("  - Check your internet connection")
        print("  - Some networks/servers may block YouTube requests (403 errors)")
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
