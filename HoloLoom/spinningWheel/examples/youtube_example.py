#!/usr/bin/env python3
"""
YouTube Spinner Examples
========================
Demonstrates how to use the YouTube spinner to ingest video transcripts
into HoloLoom as MemoryShards.

Usage:
    PYTHONPATH=. python HoloLoom/spinningWheel/examples/youtube_example.py
"""

import asyncio
from HoloLoom.spinningWheel import YouTubeSpinner, YouTubeSpinnerConfig, transcribe_youtube


async def example_1_basic():
    """Example 1: Basic YouTube transcription"""
    print("="*80)
    print("Example 1: Basic YouTube Transcription")
    print("="*80)

    # Create spinner with default config (single shard per video)
    spinner = YouTubeSpinner()

    # Transcribe a video
    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    try:
        shards = await spinner.spin({
            'url': video_url,
            'languages': ['en']
        })

        print(f"\nVideo: {video_url}")
        print(f"Generated {len(shards)} shard(s)")

        for shard in shards:
            print(f"\nShard ID: {shard.id}")
            print(f"Episode: {shard.episode}")
            print(f"Language: {shard.metadata.get('language')}")
            print(f"Duration: {shard.metadata.get('duration'):.1f}s")
            print(f"Entities: {len(shard.entities)}")
            print(f"Text preview: {shard.text[:200]}...")

    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: This example may fail in restricted environments.")
        print("Try with a different video or network environment.")


async def example_2_chunked():
    """Example 2: Chunked transcription for long videos"""
    print("\n\n" + "="*80)
    print("Example 2: Chunked Transcription")
    print("="*80)

    # Create spinner that splits into 60-second chunks
    config = YouTubeSpinnerConfig(
        chunk_duration=60.0,  # 60 seconds per chunk
        include_timestamps=True
    )
    spinner = YouTubeSpinner(config)

    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    try:
        shards = await spinner.spin({
            'url': video_url,
            'languages': ['en'],
            'metadata': {
                'source_type': 'educational',
                'ingested_at': '2025-10-23'
            }
        })

        print(f"\nVideo: {video_url}")
        print(f"Generated {len(shards)} chunks")

        for i, shard in enumerate(shards[:3]):  # Show first 3 chunks
            print(f"\n--- Chunk {i+1} ---")
            print(f"Shard ID: {shard.id}")
            print(f"Time range: {shard.metadata['chunk_start']:.1f}s - {shard.metadata['chunk_end']:.1f}s")
            print(f"URL with timestamp: {shard.metadata['url']}")
            print(f"Entities: {shard.entities[:5]}")
            print(f"Text: {shard.text[:150]}...")

    except Exception as e:
        print(f"Error: {e}")


async def example_3_convenience_function():
    """Example 3: Using the convenience function"""
    print("\n\n" + "="*80)
    print("Example 3: Convenience Function")
    print("="*80)

    # Quick transcription without creating spinner instance
    video_url = "dQw4w9WgXcQ"  # Can use just the video ID

    try:
        shards = await transcribe_youtube(
            url=video_url,
            languages=['en'],
            chunk_duration=30.0  # 30-second chunks
        )

        print(f"\nVideo ID: {video_url}")
        print(f"Generated {len(shards)} shards")
        print(f"\nFirst shard metadata:")
        for key, value in list(shards[0].metadata.items())[:5]:
            print(f"  {key}: {value}")

    except Exception as e:
        print(f"Error: {e}")


async def example_4_with_enrichment():
    """Example 4: With Ollama enrichment (requires Ollama running locally)"""
    print("\n\n" + "="*80)
    print("Example 4: With Enrichment (Optional)")
    print("="*80)

    config = YouTubeSpinnerConfig(
        enable_enrichment=True,  # Enable Ollama enrichment
        ollama_model="llama3.2:3b",
        chunk_duration=120.0  # 2-minute chunks
    )
    spinner = YouTubeSpinner(config)

    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    try:
        shards = await spinner.spin({
            'url': video_url,
            'languages': ['en']
        })

        print(f"\nEnriched {len(shards)} shards")
        print("\nFirst shard enrichment:")
        if shards[0].metadata.get('enrichment'):
            print(f"  Enrichment data: {list(shards[0].metadata['enrichment'].keys())}")
            print(f"  Enhanced entities: {shards[0].entities[:10]}")
            print(f"  Motifs: {shards[0].motifs[:5]}")
        else:
            print("  (No enrichment data - Ollama may not be running)")

    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: Enrichment requires Ollama to be running locally.")


async def example_5_integration_with_orchestrator():
    """Example 5: Integration with HoloLoom Orchestrator"""
    print("\n\n" + "="*80)
    print("Example 5: Integration with Orchestrator")
    print("="*80)

    # This example shows how YouTube shards feed into the orchestrator
    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    try:
        # Step 1: Ingest YouTube video
        shards = await transcribe_youtube(url=video_url, chunk_duration=60.0)

        print(f"Step 1: Ingested {len(shards)} shards from YouTube")

        # Step 2: These shards can be added to the orchestrator's memory
        print("\nStep 2: Add to Orchestrator memory")
        print("  Example code:")
        print("    from HoloLoom.orchestrator import Orchestrator")
        print("    orchestrator = Orchestrator(config)")
        print("    for shard in shards:")
        print("        orchestrator.memory.add_shard(shard)")

        # Step 3: Query the orchestrator with questions about the video
        print("\nStep 3: Query the video content")
        print("  Example query: 'What are the main topics discussed in this video?'")

        # Show what the shards look like
        print(f"\nShard structure (first shard):")
        shard = shards[0]
        print(f"  id: {shard.id}")
        print(f"  episode: {shard.episode}")
        print(f"  text: {shard.text[:100]}...")
        print(f"  entities: {shard.entities[:5]}")
        print(f"  metadata keys: {list(shard.metadata.keys())}")

    except Exception as e:
        print(f"Error: {e}")


async def example_6_multiple_languages():
    """Example 6: Multi-language support"""
    print("\n\n" + "="*80)
    print("Example 6: Multi-Language Support")
    print("="*80)

    spinner = YouTubeSpinner()

    # Try different language preferences
    test_cases = [
        ("dQw4w9WgXcQ", ['es', 'en'], "Spanish preferred, fallback to English"),
        ("dQw4w9WgXcQ", ['fr', 'en'], "French preferred, fallback to English"),
        ("dQw4w9WgXcQ", ['en'], "English only"),
    ]

    for video_id, languages, description in test_cases:
        print(f"\nTest: {description}")
        print(f"  Languages: {languages}")

        try:
            shards = await spinner.spin({
                'url': video_id,
                'languages': languages
            })

            print(f"  ✓ Success: Got transcript in '{shards[0].metadata['language']}'")

        except Exception as e:
            print(f"  ✗ Error: {e}")


async def main():
    """Run all examples"""
    print("\nYouTube Spinner Examples for HoloLoom")
    print("Note: Replace video URLs with actual videos you want to transcribe")
    print("Examples may fail if transcripts are disabled or in restricted environments\n")

    # Run examples
    await example_1_basic()
    await example_2_chunked()
    await example_3_convenience_function()
    await example_4_with_enrichment()
    await example_5_integration_with_orchestrator()
    await example_6_multiple_languages()

    print("\n" + "="*80)
    print("Examples Complete!")
    print("="*80)


if __name__ == '__main__':
    asyncio.run(main())
