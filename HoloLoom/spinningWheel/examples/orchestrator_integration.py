#!/usr/bin/env python3
"""
SpinningWheel → Orchestrator Integration Example
================================================

Demonstrates the complete pipeline:
1. Ingest data via SpinningWheel spinners
2. Store in unified memory system
3. Query via WeavingOrchestrator

This example shows the standard integration pattern for all spinners.
"""

import asyncio
import sys
from pathlib import Path

# Add repository root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from HoloLoom.spinningWheel import (
    AudioSpinner, TextSpinner, CodeSpinner, WebsiteSpinner,
    spin_text, spin_webpage
)
from HoloLoom.memory.protocol import create_unified_memory, shards_to_memories


async def example_1_text_ingestion():
    """Example 1: Ingest text document into memory."""
    print("=" * 70)
    print("Example 1: Text Document Ingestion")
    print("=" * 70)

    # Step 1: Spin raw text into memory shards
    text_content = """
    Beekeeping Notes - October 2025

    Hive Inspection Results:
    - Hive Alpha: Strong colony, good honey stores
    - Hive Beta: Moderate strength, needs supplemental feeding
    - Hive Gamma: Weak colony, consider merging

    Weather: Cool and overcast, bees less active
    Next steps: Prepare for winter, add insulation
    """

    shards = await spin_text(
        text=text_content,
        source='beekeeping_notes_oct2025.txt',
        chunk_by='paragraph'
    )

    print(f"\n✓ Created {len(shards)} memory shards from text")
    for i, shard in enumerate(shards):
        print(f"  Shard {i+1}: {len(shard.text)} chars, {len(shard.entities)} entities")

    # Step 2: Convert shards → Memory objects
    memories = shards_to_memories(shards)
    print(f"\n✓ Converted to {len(memories)} Memory objects")

    # Step 3: Store in unified memory backend
    memory_backend = await create_unified_memory(user_id="blake")
    memory_ids = await memory_backend.store_many(memories)

    print(f"\n✓ Stored {len(memory_ids)} memories in backend")
    print(f"  Memory IDs: {memory_ids[:3]}..." if len(memory_ids) > 3 else f"  Memory IDs: {memory_ids}")

    # Step 4: Query memories (demonstrates retrieval)
    results = await memory_backend.recall(
        query="What is the status of Hive Beta?",
        strategy="fused",
        limit=5
    )

    print(f"\n✓ Retrieved {len(results)} relevant memories")
    if results:
        print(f"\n  Top result:")
        print(f"    Text: {results[0].text[:100]}...")
        print(f"    Score: {results[0].score:.3f}")
        print(f"    Context: {results[0].context}")

    print("\n" + "=" * 70)


async def example_2_website_ingestion():
    """Example 2: Ingest webpage into memory."""
    print("\n" + "=" * 70)
    print("Example 2: Webpage Ingestion")
    print("=" * 70)

    # Simulate webpage content (in production, this would be scraped)
    # For testing, we provide pre-fetched content
    webpage_content = """
    Winter Beekeeping Guide

    Preparing Your Hives for Winter

    As temperatures drop, bees cluster together to maintain warmth.
    Proper preparation is essential for colony survival through winter months.

    Key Preparations:
    1. Ensure adequate honey stores (40-60 lbs minimum)
    2. Install insulation or wraps for cold climates
    3. Reduce hive entrance to prevent drafts and pests
    4. Check for and seal any cracks or gaps
    5. Ensure proper ventilation to prevent moisture buildup

    Monitoring During Winter:
    - Perform brief inspections on warm days (above 50°F)
    - Listen for colony hum
    - Check entrance for dead bees (small numbers are normal)
    - Provide emergency feeding if honey stores are depleted
    """

    shards = await spin_webpage(
        url='https://example.com/winter-beekeeping-guide',
        title='Winter Beekeeping Guide',
        content=webpage_content,
        tags=['beekeeping', 'winter', 'guide']
    )

    print(f"\n✓ Created {len(shards)} shards from webpage")

    # Convert and store
    memories = shards_to_memories(shards)
    memory_backend = await create_unified_memory(user_id="blake")
    memory_ids = await memory_backend.store_many(memories)

    print(f"✓ Stored {len(memory_ids)} webpage chunks")

    # Check metadata
    first_memory = memories[0]
    print(f"\n  Metadata:")
    print(f"    URL: {first_memory.context.get('url')}")
    print(f"    Domain: {first_memory.context.get('domain')}")
    print(f"    Tags: {first_memory.context.get('tags')}")

    print("\n" + "=" * 70)


async def example_3_multimodal_ingestion():
    """Example 3: Ingest multiple modalities and query across them."""
    print("\n" + "=" * 70)
    print("Example 3: Multi-Modal Ingestion & Cross-Modal Querying")
    print("=" * 70)

    # Initialize memory backend
    memory_backend = await create_unified_memory(user_id="blake")

    # Ingest 1: Audio transcript
    audio_spinner = AudioSpinner()
    audio_shards = await audio_spinner.spin({
        'transcript': 'Today I inspected all three hives. Alpha is thriving, Beta needs feeding, Gamma is weak.',
        'episode': 'inspection_oct26'
    })
    audio_memories = shards_to_memories(audio_shards)
    await memory_backend.store_many(audio_memories)
    print(f"✓ Ingested {len(audio_memories)} audio transcript memories")

    # Ingest 2: Code snippet
    code_spinner = CodeSpinner()
    code_content = '''
def calculate_honey_harvest(hive_weight_lbs, tare_weight_lbs, reserve_lbs=40):
    """Calculate harvestable honey amount."""
    total_honey = hive_weight_lbs - tare_weight_lbs
    harvestable = max(0, total_honey - reserve_lbs)
    return harvestable
'''
    code_shards = await code_spinner.spin({
        'type': 'file',
        'path': 'beekeeping_utils.py',
        'content': code_content
    })
    code_memories = shards_to_memories(code_shards)
    await memory_backend.store_many(code_memories)
    print(f"✓ Ingested {len(code_memories)} code snippet memories")

    # Ingest 3: Text notes
    text_spinner = TextSpinner()
    text_shards = await text_spinner.spin({
        'text': 'Hive Beta weight: 75 lbs. Tare weight: 25 lbs. May need supplemental feeding.',
        'source': 'hive_measurements.txt'
    })
    text_memories = shards_to_memories(text_shards)
    await memory_backend.store_many(text_memories)
    print(f"✓ Ingested {len(text_memories)} text note memories")

    # Cross-modal query
    print("\n  Querying across all modalities...")
    results = await memory_backend.recall(
        query="How much honey can I harvest from Hive Beta?",
        strategy="fused",
        limit=5
    )

    print(f"\n✓ Found {len(results)} relevant memories across modalities:")
    for i, result in enumerate(results[:3]):
        content_type = result.context.get('type') or result.context.get('format') or 'unknown'
        print(f"\n  Result {i+1} ({content_type}):")
        print(f"    Text: {result.text[:80]}...")
        print(f"    Score: {result.score:.3f}")

    print("\n" + "=" * 70)


async def example_4_orchestrator_ready():
    """Example 4: Memories ready for orchestrator consumption."""
    print("\n" + "=" * 70)
    print("Example 4: Orchestrator-Ready Memory Integration")
    print("=" * 70)

    print("""
The memories ingested via SpinningWheel are now ready for the WeavingOrchestrator:

Integration Pattern:
--------------------

1. **Ingestion Layer (SpinningWheel)**
   - Spinners convert raw data → MemoryShards
   - Shards contain entities, motifs, metadata
   - Lightweight processing, no heavy ML

2. **Storage Layer (Unified Memory)**
   - Shards → Memory objects (via shards_to_memories)
   - Store in backend (Mem0, Neo4j, Qdrant)
   - Multi-strategy indexing (semantic, temporal, graph)

3. **Retrieval Layer (Memory Protocol)**
   - Recall memories via strategy (fused, semantic, graph, etc.)
   - Returns ranked results with scores
   - Provides context for orchestrator

4. **Processing Layer (WeavingOrchestrator)**
   - Consumes memories as context
   - Performs heavy processing (embeddings, spectral features)
   - Generates responses via policy engine

Example Orchestrator Usage:
---------------------------

```python
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.config import Config
from HoloLoom.Documentation.types import Query

# Initialize orchestrator with memory access
weaver = WeavingOrchestrator(config=Config.fused())

# Memories automatically retrieved from backend
query = Query(text="What is the status of my hives?")
result = await weaver.weave(query)

# Result contains:
# - Spacetime fabric with full computational trace
# - Tool selections and executions
# - Retrieved memories as context
# - Decision provenance
```

Standard Pipeline:
------------------

Raw Data
  ↓
SpinningWheel Spinner (parse, normalize, extract)
  ↓
MemoryShards (standardized format)
  ↓
Memory Backend (store with multi-strategy indexing)
  ↓
Query → Memory Recall (semantic, temporal, graph)
  ↓
WeavingOrchestrator (heavy processing, decisions)
  ↓
Response (with full computational provenance)

Key Benefits:
-------------
- **Separation of Concerns**: Spinners stay lightweight
- **Flexible Storage**: Swap backends without changing spinners
- **Multi-Strategy Retrieval**: Leverage different search strategies
- **Full Provenance**: Complete lineage from raw data → response
- **Graceful Degradation**: Works with partial backend availability
    """)

    print("\n" + "=" * 70)


async def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 10 + "SpinningWheel → Orchestrator Integration" + " " * 18 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    # Run examples
    await example_1_text_ingestion()
    await example_2_website_ingestion()
    await example_3_multimodal_ingestion()
    await example_4_orchestrator_ready()

    print("\n✅ All examples completed successfully!\n")
    print("Next Steps:")
    print("  1. Integrate spinners into your data ingestion pipeline")
    print("  2. Configure unified memory backend (mem0, neo4j, qdrant)")
    print("  3. Use WeavingOrchestrator for query processing")
    print("  4. Explore multi-modal queries across different data types")
    print()


if __name__ == '__main__':
    asyncio.run(main())
