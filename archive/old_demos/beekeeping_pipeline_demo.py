"""
Beekeeping Notes → Memory Pipeline Demo
========================================
Complete data flow: Text notes → TextSpinner → MemoryShards → Memory → Neo4j

This shows how to pipe your beekeeping inspection notes directly into
the Neo4j graph memory system using the protocol-based architecture.

Usage:
    python beekeeping_pipeline_demo.py
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import asyncio
from datetime import datetime
from pathlib import Path
import importlib.util


# ============================================================================
# Load Modules (bypassing import issues)
# ============================================================================

def load_module(name, path):
    """Load a module from file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


root = Path(__file__).parent

# Load protocol
protocol = load_module(
    "hololoom_protocol",
    root / "HoloLoom" / "memory" / "protocol.py"
)

# Load Neo4j store
neo4j_store = load_module(
    "neo4j_store",
    root / "HoloLoom" / "memory" / "stores" / "neo4j_store.py"
)

# Load text spinner
text_spinner = load_module(
    "text_spinner",
    root / "HoloLoom" / "spinningWheel" / "text.py"
)

# Load base spinner for MemoryShard type
base_spinner = load_module(
    "base_spinner",
    root / "HoloLoom" / "spinningWheel" / "base.py"
)

# Extract what we need
Memory = protocol.Memory
UnifiedMemoryInterface = protocol.UnifiedMemoryInterface
Strategy = protocol.Strategy
shards_to_memories = protocol.shards_to_memories
Neo4jMemoryStore = neo4j_store.Neo4jMemoryStore
spin_text = text_spinner.spin_text
TextSpinnerConfig = text_spinner.TextSpinnerConfig


# ============================================================================
# Sample Beekeeping Notes
# ============================================================================

INSPECTION_NOTES = """
Hive Inspection Report - October 24, 2025
==========================================

Weather: Clear, 62°F, light wind from west
Time: 2:30 PM - 4:15 PM
Beekeeper: Blake

Hive Jodi (Primary Colony)
---------------------------
Population: Very strong, 15 frames of brood
Queen: Spotted on frame 3, excellent laying pattern
Brood: Solid pattern, minimal gaps
Stores: 8 frames of capped honey, adequate for winter
Health: No signs of disease, mite drop within limits
Notes: Dennis's genetics proving excellent. This colony is our strongest.
Action: Continue monitoring, prepare for winter wrapping next week.

Hive 5 (Small Colony)
---------------------
Population: Critically weak, only 10 bees on inner cover
Queen: Not spotted
Brood: 2 frames, spotty pattern
Stores: Minimal, only 1 frame of honey
Health: Colony appears queenless or failing
Notes: This is our weakest hive. Decision needed immediately.
Action: URGENT - Combine with stronger colony before first frost or colony will die.

Dennis's Double Stack
---------------------
Population: Good, 8.5 frames of brood
Queen: Not spotted but laying pattern visible
Brood: Good solid pattern
Stores: 6 frames of honey
Health: Thymol treatment round 2 applied (35 units, reduced for safety)
Treatment: Monitoring mite levels closely
Notes: Conservative dosing approach working well.
Action: Check mite drop in 3 days, prepare winter configuration.

Jodi Split Colony
-----------------
Population: Good recovery after fall feeding, 8 frames
Queen: Young queen from Jodi's genetics, spotted on frame 2
Brood: Excellent pattern for a young colony
Stores: 5 frames of honey, building well
Health: Clean, no issues observed
Notes: This split is exceeding expectations. Jodi's line is very productive.
Action: Standard winter prep, consider this for breeding stock next spring.

Overall Apiary Status
---------------------
Total Hives: 4
Strong: 1 (Jodi Primary)
Good: 2 (Dennis Double, Jodi Split)
Weak/Critical: 1 (Hive 5 - action required)

Winter Preparation Tasks:
- Wrap strong hives next week
- Combine Hive 5 with Jodi Split (URGENT)
- Final mite treatments complete by Nov 1
- Reduce entrances before first frost
- Mouse guards installed
- Emergency feeding stations prepared

Weather forecast shows frost possible Nov 2-3.
Timeline is tight. Prioritize weak hive combination.
"""


TREATMENT_LOG = """
Treatment Log - Thymol Application
===================================
Date: October 23, 2025
Treatment: Thymol Round 2

Hive: Dennis's Double Stack
Dosage: 35 units (reduced from standard 50)
Reason: Conservative approach for strong but sensitive colony
Application: Top bars, evenly distributed
Temperature: 65°F (ideal for thymol vaporization)
Notes: Colony handled treatment well in Round 1, reducing dose for safety.
Next Check: October 27 (mite drop count)

Hive: Jodi Primary
Dosage: 45 units (standard)
Reason: Very strong colony, can handle full treatment
Application: Top bars, evenly distributed
Notes: Excellent mite drop after Round 1, minimal stress on colony.
Next Check: October 27

Hive: Jodi Split
Dosage: 30 units (reduced for young colony)
Reason: Young colony, lighter treatment
Application: Top bars
Notes: First treatment for this colony, monitoring closely.
Next Check: October 27

Hive 5: NOT TREATED
Reason: Colony too weak, treatment would kill them
Notes: Combination with stronger colony is only option.
"""


# ============================================================================
# Pipeline Functions
# ============================================================================

async def demo_basic_pipeline():
    """Show basic step-by-step pipeline."""
    print("=" * 80)
    print("DEMO 1: Basic Pipeline (Step-by-Step)")
    print("=" * 80)
    print()
    print("Pipeline: Inspection Notes → TextSpinner → MemoryShards → Memory → Neo4j")
    print()

    # Step 1: Spin text into shards
    print("STEP 1: Spinning inspection notes into shards...")
    print("-" * 80)

    config = TextSpinnerConfig(
        chunk_by='paragraph',  # Split by paragraph
        chunk_size=500,        # ~500 chars per chunk
        extract_entities=True  # Extract basic entities
    )

    # We need to manually create the spinner since spin_text is a convenience function
    from importlib import import_module

    # Import the spinner class
    spinner = text_spinner.TextSpinner(config)

    shards = await spinner.spin({
        'text': INSPECTION_NOTES,
        'source': 'hive_inspection_2025_10_24.txt',
        'episode': 'fall_inspection_2025_10_24',
        'metadata': {
            'author': 'Blake',
            'date': '2025-10-24',
            'type': 'hive_inspection',
            'season': 'fall'
        }
    })

    print(f"✓ Created {len(shards)} memory shards")
    print(f"\nExample shard:")
    print(f"  ID: {shards[0].id}")
    print(f"  Text: {shards[0].text[:100]}...")
    print(f"  Episode: {shards[0].episode}")
    print(f"  Entities: {shards[0].entities[:5]}")  # Show first 5
    print(f"  Metadata: {shards[0].metadata}")
    print()

    # Step 2: Convert shards to memories
    print("STEP 2: Converting shards → Memory objects...")
    print("-" * 80)

    memories = shards_to_memories(shards)
    print(f"✓ Converted {len(memories)} Memory objects")
    print(f"\nExample memory:")
    mem = memories[0]
    print(f"  ID: {mem.id}")
    print(f"  Text: {mem.text[:100]}...")
    print(f"  Timestamp: {mem.timestamp}")
    print(f"  Context (episode): {mem.context.get('episode')}")
    print(f"  Context (entities): {mem.context.get('entities', [])[:3]}")
    print(f"  Metadata: {mem.metadata}")
    print()

    # Step 3: Store in Neo4j
    print("STEP 3: Storing in Neo4j graph database...")
    print("-" * 80)

    try:
        # Connect to Neo4j
        store = Neo4jMemoryStore(
            uri="bolt://localhost:7688",
            username="neo4j",
            password="beekeeper123"
        )

        memory = UnifiedMemoryInterface(_store=store)

        # Store all memories
        ids = await memory.store_many(memories)

        print(f"✓ Stored {len(ids)} memories in Neo4j")
        print(f"\nStored memory IDs:")
        for idx, mem_id in enumerate(ids[:5]):  # Show first 5
            print(f"  {idx+1}. {mem_id}")
        if len(ids) > 5:
            print(f"  ... and {len(ids) - 5} more")
        print()

        # Step 4: Query the memories
        print("STEP 4: Querying stored memories...")
        print("-" * 80)

        queries = [
            ("What hives need attention?", Strategy.SEMANTIC),
            ("Winter preparation tasks", Strategy.FUSED),
            ("Recent inspections", Strategy.TEMPORAL)
        ]

        for query_text, strategy in queries:
            print(f"\nQuery: '{query_text}' (Strategy: {strategy.value})")
            results = await memory.recall(query_text, strategy=strategy, limit=3)
            print(f"Found {results.count} memories:")

            for idx, (mem, score) in enumerate(zip(results.memories, results.scores)):
                print(f"  [{score:.2f}] {mem.text[:70]}...")

        print()

        # Cleanup
        store.close()
        print("✓ Demo 1 complete!\n")

    except Exception as e:
        print(f"❌ Error connecting to Neo4j: {e}")
        print("   Make sure Neo4j is running on port 7688")
        print("   with username='neo4j' and password='beekeeper123'")
        print()


async def demo_treatment_log_pipeline():
    """Show piping treatment log with enriched context."""
    print("=" * 80)
    print("DEMO 2: Treatment Log Pipeline (with enriched context)")
    print("=" * 80)
    print()

    # Spin treatment log
    print("Spinning treatment log into shards...")

    config = TextSpinnerConfig(
        chunk_by='paragraph',
        chunk_size=400,
        extract_entities=True
    )

    spinner = text_spinner.TextSpinner(config)

    shards = await spinner.spin({
        'text': TREATMENT_LOG,
        'source': 'treatment_log_2025_10_23.txt',
        'episode': 'thymol_treatment_round_2',
        'metadata': {
            'author': 'Blake',
            'date': '2025-10-23',
            'type': 'treatment_log',
            'treatment_type': 'thymol',
            'season': 'fall',
            'priority': 'high'
        }
    })

    print(f"✓ Created {len(shards)} treatment shards\n")

    # Enrich each shard's metadata before conversion
    print("Enriching shards with beekeeping-specific context...")
    for shard in shards:
        # Extract hive mentions
        text_lower = shard.text.lower()
        if 'jodi' in text_lower:
            shard.metadata['hive'] = 'hive-jodi'
            shard.metadata['genetics'] = 'jodi-line'
        elif 'dennis' in text_lower:
            shard.metadata['hive'] = 'hive-dennis'
            shard.metadata['genetics'] = 'dennis-line'
        elif 'hive 5' in text_lower:
            shard.metadata['hive'] = 'hive-5'
            shard.metadata['colony_status'] = 'critical'

        # Extract dosage if mentioned
        if 'units' in text_lower:
            import re
            match = re.search(r'(\d+)\s*units', text_lower)
            if match:
                shard.metadata['dosage_units'] = int(match.group(1))

        # Tag urgent items
        if any(word in text_lower for word in ['urgent', 'critical', 'immediately']):
            shard.metadata['priority'] = 'urgent'

    print(f"✓ Enriched {len(shards)} shards\n")

    # Convert to memories
    memories = shards_to_memories(shards)

    print(f"Example enriched memory:")
    mem = memories[0]
    print(f"  Text: {mem.text[:80]}...")
    print(f"  Metadata: {mem.metadata}")
    print()

    # Store in Neo4j
    try:
        store = Neo4jMemoryStore(
            uri="bolt://localhost:7688",
            username="neo4j",
            password="beekeeper123"
        )

        memory = UnifiedMemoryInterface(_store=store)
        ids = await memory.store_many(memories)

        print(f"✓ Stored {len(ids)} treatment memories in Neo4j\n")

        # Query for critical items
        print("Querying for critical/urgent items...")
        results = await memory.recall("urgent critical weak", strategy=Strategy.SEMANTIC, limit=5)
        print(f"Found {results.count} critical items:")
        for idx, (mem, score) in enumerate(zip(results.memories, results.scores)):
            print(f"  [{score:.2f}] {mem.text[:70]}...")

        print()

        store.close()
        print("✓ Demo 2 complete!\n")

    except Exception as e:
        print(f"❌ Error: {e}\n")


async def demo_one_liner():
    """Show the one-liner convenience function."""
    print("=" * 80)
    print("DEMO 3: One-Liner Pipeline")
    print("=" * 80)
    print()
    print("Using pipe_text_to_memory() for instant storage")
    print()

    try:
        # Connect
        store = Neo4jMemoryStore(
            uri="bolt://localhost:7688",
            username="neo4j",
            password="beekeeper123"
        )
        memory = UnifiedMemoryInterface(_store=store)

        # One-liner!
        quick_note = """
        Quick observation: All hives showing good activity at entrance today.
        Temperature 68°F, perfect for late October. Noticed increased pollen
        bringing - likely last major flow before frost. Mouse guards installed
        on all entrances. Winter prep 80% complete.
        """

        print("Piping quick note directly into memory...")
        ids = await protocol.pipe_text_to_memory(
            text=quick_note,
            memory=memory,
            source='quick_notes.txt',
            chunk_by=None  # Don't chunk, store as single memory
        )

        print(f"✓ Stored {len(ids)} memory in one function call!")
        print(f"  ID: {ids[0]}")
        print()

        # Verify it's there
        results = await memory.recall("mouse guards pollen", strategy=Strategy.SEMANTIC, limit=1)
        if results.count > 0:
            print(f"✓ Retrieved: {results.memories[0].text[:80]}...")

        print()

        store.close()
        print("✓ Demo 3 complete!\n")

    except Exception as e:
        print(f"❌ Error: {e}\n")


async def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print(" BEEKEEPING NOTES → MEMORY PIPELINE")
    print(" Complete Data Flow Demonstration")
    print("=" * 80)
    print()
    print("This demo shows how to pipe your beekeeping notes directly into")
    print("the Neo4j graph memory system using the protocol-based architecture.")
    print()

    try:
        await demo_basic_pipeline()
        await demo_treatment_log_pipeline()
        await demo_one_liner()

        print("=" * 80)
        print(" ALL DEMOS COMPLETE ✓")
        print("=" * 80)
        print()
        print("📊 View your data in Neo4j Browser:")
        print("   http://localhost:7475")
        print()
        print("🔍 Useful Cypher queries:")
        print("   MATCH (m:Memory) RETURN m LIMIT 25")
        print("   MATCH (m:Memory) WHERE m.metadata.type = 'hive_inspection' RETURN m")
        print("   MATCH (m:Memory) WHERE m.metadata.priority = 'urgent' RETURN m")
        print()
        print("🎯 Key Takeaway:")
        print("   Text → Spinner → Shards → Memories → Neo4j")
        print("   The data pipeline is now obvious and elegant!")
        print()

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)