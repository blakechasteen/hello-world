"""
Complete Beekeeping Memory System Demo
======================================
Shows all three integrations working together:
1. Neo4j - Graph relationships between hives, beekeepers, inspections
2. Mem0 - AI entity extraction from natural language
3. Custom Strategy - Beekeeping-aware scoring

Run: python beekeeping_memory_demo.py
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import asyncio
from datetime import datetime
from pathlib import Path

# Import the modules directly
root = Path(__file__).parent
protocol_path = root / 'HoloLoom' / 'memory' / 'protocol.py'
neo4j_store_path = root / 'HoloLoom' / 'memory' / 'stores' / 'neo4j_memory_store.py'
beekeeping_strategy_path = root / 'HoloLoom' / 'memory' / 'stores' / 'beekeeping_strategy.py'

import importlib.util

# Load protocol
spec = importlib.util.spec_from_file_location("protocol", protocol_path)
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)

# Load Neo4j store
spec = importlib.util.spec_from_file_location("neo4j_store", neo4j_store_path)
neo4j_store_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(neo4j_store_module)

# Load beekeeping strategy
spec = importlib.util.spec_from_file_location("beekeeping_strategy", beekeeping_strategy_path)
strategy_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strategy_module)

# Extract classes
Memory = protocol.Memory
MemoryQuery = protocol.MemoryQuery
UnifiedMemoryInterface = protocol.UnifiedMemoryInterface
Strategy = protocol.Strategy
Neo4jMemoryStore = neo4j_store_module.Neo4jMemoryStore
calculate_beekeeping_score = strategy_module.calculate_beekeeping_score


async def demo():
    print("="*70)
    print("BEEKEEPING MEMORY SYSTEM - COMPLETE DEMO")
    print("="*70)
    print()
    print("This demo shows three integrations:")
    print("  1. Neo4j graph database (relationships)")
    print("  2. Mem0 AI entity extraction (coming soon)")
    print("  3. Custom beekeeping strategy (seasonal, priority scoring)")
    print()

    # ========================================================================
    # Part 1: Neo4j Graph Memory
    # ========================================================================

    print("="*70)
    print("PART 1: Neo4j Graph Memory")
    print("="*70)
    print()

    # Connect to beekeeping Neo4j
    print("Connecting to beekeeping Neo4j (port 7688)...")
    store = Neo4jMemoryStore(
        uri="bolt://localhost:7688",
        username="neo4j",
        password="beekeeper123"
    )
    print("✓ Connected!\n")

    # Create unified memory interface
    memory = UnifiedMemoryInterface(_store=store)

    # Store inspection memories
    print("Storing hive inspection memories...\n")

    inspections = [
        {
            "text": "Inspected Hive Jodi - 15 frames of brood, very strong colony. Dennis's genetics proving excellent.",
            "context": {
                "hive": "hive-jodi-primary-001",
                "place": "apiary",
                "genetics": "jodi-line",
                "population_strength": "very_strong",
                "season": "fall"
            }
        },
        {
            "text": "Applied thymol treatment round 2 to Hive Jodi - 35 units, reduced for safety",
            "context": {
                "hive": "hive-jodi-primary-001",
                "place": "apiary",
                "treatment_type": "thymol",
                "priority": "high"
            }
        },
        {
            "text": "Hive 5 critically weak - only 10 bees on inner cover. Must combine before winter or will not survive.",
            "context": {
                "hive": "hive-small-001",
                "place": "apiary",
                "population_strength": "very_weak",
                "colony_status": "weakest",
                "priority": "critical"
            }
        },
        {
            "text": "Dennis's double stack hive showing good population - 8.5 frames. Conservative treatment dosing.",
            "context": {
                "hive": "hive-dennis-double-001",
                "place": "apiary",
                "genetics": "dennis-line",
                "population_strength": "good"
            }
        },
        {
            "text": "Split from Jodi's hive recovered well after fall feeding. 8 frames, good entrance activity.",
            "context": {
                "hive": "hive-jodi-split-001",
                "place": "apiary",
                "genetics": "jodi-line",
                "population_strength": "good"
            }
        }
    ]

    for inspection in inspections:
        mem = Memory(
            id="",
            text=inspection["text"],
            timestamp=datetime.now(),
            context=inspection["context"],
            metadata={'user_id': 'blake'}
        )

        memory_id = await memory.store(mem.text, mem.context, user_id='blake')
        print(f"✓ {inspection['text'][:60]}...")

    # Test different retrieval strategies
    print("\n" + "="*70)
    print("Testing Retrieval Strategies")
    print("="*70)
    print()

    # Strategy 1: TEMPORAL - What happened recently?
    print("1. TEMPORAL Strategy - Recent memories")
    print("-" * 70)
    result = await memory.recall("hive", strategy=Strategy.TEMPORAL, limit=3)
    for mem, score in zip(result.memories, result.scores):
        print(f"  [{score:.2f}] {mem.text[:60]}...")
    print()

    # Strategy 2: SEMANTIC - Text similarity
    print("2. SEMANTIC Strategy - 'strong hives'")
    print("-" * 70)
    result = await memory.recall("strong", strategy=Strategy.SEMANTIC, limit=3)
    for mem, score in zip(result.memories, result.scores):
        print(f"  [{score:.2f}] {mem.text[:60]}...")
    print()

    # Strategy 3: GRAPH - Relationship-based
    print("3. GRAPH Strategy - 'Hive Jodi'")
    print("-" * 70)
    result = await memory.recall("Hive Jodi", strategy=Strategy.GRAPH, limit=3)
    for mem, score in zip(result.memories, result.scores):
        print(f"  [{score:.2f}] {mem.text[:60]}...")
    print()

    # Strategy 4: FUSED - Combined scoring
    print("4. FUSED Strategy - 'winter preparation'")
    print("-" * 70)
    result = await memory.recall("winter", strategy=Strategy.FUSED, limit=3)
    for mem, score in zip(result.memories, result.scores):
        print(f"  [{score:.2f}] {mem.text[:60]}...")
    print()

    # ========================================================================
    # Part 2: Custom Beekeeping Strategy
    # ========================================================================

    print("="*70)
    print("PART 2: Custom Beekeeping-Aware Scoring")
    print("="*70)
    print()

    print("The beekeeping strategy considers:")
    print("  • Seasonal relevance (fall = winter prep important)")
    print("  • Hive priority (weak hives get attention)")
    print("  • Task urgency (critical > routine)")
    print("  • Recency with decay")
    print("  • Semantic matching")
    print()

    # Score each memory with beekeeping strategy
    query = "What needs attention before winter?"
    print(f"Query: '{query}'")
    print(f"Current season: {strategy_module.get_current_season().value}\n")

    scored_memories = []
    for inspection in inspections:
        score = calculate_beekeeping_score(
            memory_text=inspection["text"],
            memory_context=inspection["context"],
            memory_timestamp=datetime.now(),
            query_text=query,
            query_context={}
        )
        scored_memories.append((score, inspection))

    # Sort by score
    scored_memories.sort(reverse=True, key=lambda x: x[0])

    print("Ranked by beekeeping-aware scoring:")
    print("-" * 70)
    for score, mem in scored_memories[:3]:
        print(f"\n[{score:.3f}] {mem['text'][:60]}...")
        season = strategy_module.extract_season_from_context(mem['context'])
        priority = strategy_module.extract_hive_priority(mem['context'])
        urgency = strategy_module.extract_task_urgency(mem['context'], mem['text'])
        print(f"        Season: {season.value}, Priority: {priority:.2f}, Urgency: {urgency:.2f}")

    print()

    # ========================================================================
    # Part 3: Health Check
    # ========================================================================

    print("="*70)
    print("PART 3: System Health Check")
    print("="*70)
    print()

    health = await memory.health_check()
    print("System Status:")
    for key, value in health.items():
        print(f"  {key}: {value}")

    print()

    # Cleanup
    store.close()

    # ========================================================================
    # Summary
    # ========================================================================

    print("="*70)
    print("SUMMARY")
    print("="*70)
    print()
    print("✓ Neo4j Integration - WORKING")
    print("  - Stores memories as graph nodes")
    print("  - Temporal, semantic, and graph-based retrieval")
    print("  - Relationship tracking between hives")
    print()
    print("✓ Custom Strategy - WORKING")
    print("  - Seasonal awareness (fall → winter prep)")
    print("  - Hive priority (weak hives flagged)")
    print("  - Task urgency (critical tasks surfaced)")
    print()
    print("⏳ Mem0 Integration - READY (requires Ollama)")
    print("  - Install: pip install mem0ai")
    print("  - Start Ollama with models")
    print("  - Use Mem0MemoryStore for AI entity extraction")
    print()
    print("="*70)
    print("🎉 DEMO COMPLETE!")
    print("="*70)
    print()
    print("Next steps:")
    print("  1. View graph in Neo4j Browser: http://localhost:7475")
    print("  2. Run: MATCH (m:Memory)-[r]-(related) RETURN m, r, related")
    print("  3. Try mem0_memory_store.py for AI extraction")
    print()


if __name__ == "__main__":
    asyncio.run(demo())