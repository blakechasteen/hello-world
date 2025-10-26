#!/usr/bin/env python3
"""
Simple YouTube Video Transcriber
Usage: python transcribe_video.py [VIDEO_URL]
"""

import asyncio
import sys
from HoloLoom.spinningWheel import transcribe_youtube


async def main():
    # Get video URL from command line or use default
    if len(sys.argv) > 1:
        video_url = sys.argv[1]
    else:
        # Example video - replace with your own
        video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        print(f"No URL provided, using example: {video_url}")
        print("Usage: python transcribe_video.py YOUR_VIDEO_URL\n")

    print(f"Transcribing: {video_url}")
    print("-" * 80)

    try:
        # Transcribe the video
        # chunk_duration=None means single shard for entire video
        # Set to 60.0 to split into 60-second chunks
        shards = await transcribe_youtube(
            url=video_url,
            languages=['en'],
            chunk_duration=None  # Change to 60.0 for chunking
        )

        print(f"\n✓ Success! Generated {len(shards)} shard(s)\n")

        # Display information about each shard
        for i, shard in enumerate(shards):
            print(f"{'='*80}")
            print(f"Shard {i+1}/{len(shards)}")
            print(f"{'='*80}")
            print(f"ID: {shard.id}")
            print(f"Episode: {shard.episode}")
            print(f"Language: {shard.metadata.get('language', 'unknown')}")
            print(f"Duration: {shard.metadata.get('duration', 0):.1f} seconds")
            print(f"Video URL: {shard.metadata.get('url', 'N/A')}")
            print(f"\nExtracted Entities ({len(shard.entities)}):")
            print(f"  {', '.join(shard.entities[:10])}")
            if len(shard.entities) > 10:
                print(f"  ... and {len(shard.entities) - 10} more")

            print(f"\nTranscript Preview:")
            print(f"  {shard.text[:300]}...")

            if len(shards) == 1:
                # For single shard, show full text
                print(f"\nFull Transcript:")
                print(f"{'-'*80}")
                print(shard.text)
                print(f"{'-'*80}")

            print()

        # Optional: Save to file
        save = input("Save transcript to file? (y/n): ").lower().strip()
        if save == 'y':
            filename = f"transcript_{shards[0].metadata.get('video_id', 'video')}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Video: {video_url}\n")
                f.write(f"Language: {shards[0].metadata.get('language')}\n")
                f.write(f"Duration: {shards[0].metadata.get('duration'):.1f}s\n\n")
                f.write("="*80 + "\n")
                f.write("TRANSCRIPT\n")
                f.write("="*80 + "\n\n")

                for shard in shards:
                    f.write(shard.text + "\n\n")

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
