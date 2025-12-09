"""
Bee Inspection Demo
===================

Demonstrates the complete bee inspection audio pipeline:
1. Creates a sample audio file (or uses existing)
2. Processes with Whisper transcription
3. Extracts structured schema
4. Generates vector embeddings
5. Stores in database
6. Performs semantic search

Usage:
    python bee_inspection_demo.py
    
    # Or with your own audio:
    python bee_inspection_ingest.py path/to/audio.wav --date 2024-10-18
"""

import asyncio
from pathlib import Path
from bee_inspection_ingest import BeeInspectionPipeline
import warnings


async def demo_with_sample_transcript():
    """Demo using a sample transcript (no audio needed)"""
    
    print("="*70)
    print("BEE INSPECTION AUDIO INGESTION DEMO")
    print("="*70)
    print()
    
    # Sample transcript (from your bee inspection guide examples)
    sample_transcript = """
    Bee inspection on October 18th, 2024. Starting with Hive Jodi, the Dennis line hive.
    
    Weather is clear, about 68 degrees Fahrenheit. Light breeze from the south.
    I'm wearing my veil today since I want to be cautious with the treatments.
    
    Opening up Hive Jodi now. Good entrance activity - I'd say moderate to good.
    About 15-20 bees coming and going per minute.
    
    Removing the inner cover... lots of bees on top, clustered nicely.
    I count about 8 frames of bees in the top box. Population looks strong.
    
    They're pretty calm today. Minimal smoke needed - just a few puffs at the entrance
    and they settled right down. Very gentle bees, as expected from the Dennis line.
    
    Checking the thymol treatment cards. Round 2, placed 14 days ago.
    Cards still have some residue but mostly consumed. Replacing with fresh cards today.
    Standard dosage - 2 cards in the top box.
    
    The hive is sealed up tight - propolis everywhere. They really like to seal.
    Compared to Hive Karen next to it, which barely seals at all. Interesting difference.
    
    Frames look good - plenty of capped brood, some pollen stores visible.
    No signs of disease. Queen is doing well.
    
    Moving to the bottom box... still good population down here, maybe 5-6 frames of bees.
    Plenty of honey stores for winter prep.
    
    Closing up Hive Jodi. Total inspection time: about 12 minutes.
    
    Overall assessment: Strong colony, good temperament, treatment on schedule.
    Should be in excellent shape for winter.
    
    Task for next inspection: Check bottom box more thoroughly for stores.
    Consider if they need supplemental feeding before cold weather.
    
    That's it for Hive Jodi. On to the next one.
    """
    
    # Create sample transcript file
    transcript_path = Path("sample_inspection.txt")
    transcript_path.write_text(sample_transcript)
    print(f"✓ Created sample transcript: {transcript_path}")
    
    # Initialize pipeline (no Neo4j required for demo)
    print("\nInitializing pipeline...")
    pipeline = BeeInspectionPipeline(
        whisper_model="base",  # Not used since we have transcript
        use_llm=False,         # Using rule-based extraction
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password"
    )
    
    # Process the transcript
    print("\n" + "="*70)
    result = await pipeline.process_audio(
        Path("sample_inspection.txt"),  # Will use .txt instead of audio
        date="2024-10-18",
        inspector="Demo User"
    )
    
    # Show results
    print("\n" + "="*70)
    print("EXTRACTION RESULTS")
    print("="*70)
    print(f"\nInspection Event:")
    print(f"  ID: {result['inspection'].eventId}")
    print(f"  Date: {result['inspection'].date}")
    print(f"  Inspector: {result['inspection'].inspector}")
    print(f"\nTranscript Preview:")
    print(f"  Length: {len(result['transcript'])} characters")
    print(f"  First 200 chars: {result['transcript'][:200]}...")
    
    print(f"\nEmbeddings Generated:")
    print(f"  Full transcript: {result['embeddings']['full_transcript'].shape}")
    print(f"  Inspection event: {result['embeddings']['inspection'].shape}")
    
    # Demo semantic search
    print("\n" + "="*70)
    print("SEMANTIC SEARCH DEMO")
    print("="*70)
    
    queries = [
        "How did the bees behave during inspection?",
        "What treatments were applied?",
        "Population strength and frames of bees",
        "Propolis and sealing behavior",
        "Winter preparation tasks"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = pipeline.search(query, top_k=3)
        
        if results:
            print(f"  Top result:")
            print(f"    Type: {results[0]['entity_type']}")
            print(f"    ID: {results[0]['entity_id']}")
            print(f"    Similarity: {results[0]['similarity']:.3f}")
            print(f"    Text: {results[0]['text'][:150]}...")
        else:
            print("  No results found")
    
    # Show storage summary
    print("\n" + "="*70)
    print("STORAGE SUMMARY")
    print("="*70)
    print(f"In-memory storage (Neo4j not required for demo):")
    print(f"  Inspections: {len(pipeline.database.memory_store['inspections'])}")
    print(f"  Hives: {len(pipeline.database.memory_store['hives'])}")
    print(f"  Embeddings: {len(pipeline.database.memory_store['embeddings'])}")
    
    # Cleanup
    pipeline.close()
    
    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. To use with real audio files:")
    print("   python bee_inspection_ingest.py audio.wav --date 2024-10-18")
    print()
    print("2. To enable Neo4j storage:")
    print("   - Start Neo4j (docker or local)")
    print("   - Update connection credentials in the script")
    print()
    print("3. To use LLM extraction:")
    print("   - Add OpenAI/Claude API key")
    print("   - Use --use-llm flag")
    print()
    print("4. To integrate with HoloLoom memory:")
    print("   - Use embeddings with WeavingOrchestrator")
    print("   - Store as MemoryShards")
    print("   - Enable recursive learning")


async def demo_with_audio(audio_path: Path):
    """Demo with actual audio file"""
    
    print(f"Processing audio file: {audio_path}")
    
    pipeline = BeeInspectionPipeline(
        whisper_model="base",
        use_llm=False
    )
    
    result = await pipeline.process_audio(
        audio_path,
        date="2024-10-18",
        inspector="Demo User"
    )
    
    print(f"\n✓ Processed successfully")
    print(f"  Transcript: {len(result['transcript'])} chars")
    print(f"  Segments: {len(result['segments'])}")
    
    pipeline.close()


async def main():
    """Main demo entry point"""
    
    # Check if sample audio exists
    sample_audio = Path("sample_inspection.wav")
    
    if sample_audio.exists():
        print("Found sample audio file - using real audio processing")
        await demo_with_audio(sample_audio)
    else:
        print("No audio file found - using text transcript demo")
        await demo_with_sample_transcript()


if __name__ == "__main__":
    asyncio.run(main())
