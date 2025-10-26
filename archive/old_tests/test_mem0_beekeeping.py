"""
Test Mem0 with Beekeeping Data
===============================
Uses local Ollama for AI-powered entity extraction from beekeeping notes.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from mem0 import Memory

print("="*70)
print("MEM0 BEEKEEPING DEMO - AI Entity Extraction")
print("="*70)
print()

# Configure Mem0 to use local Ollama
config = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "llama3.2:3b",
            "base_url": "http://localhost:11434",
            "ollama_base_url": "http://localhost:11434"
        }
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text:latest",
            "ollama_base_url": "http://localhost:11434"
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "beekeeping_memories",
            "path": "./beekeeping_qdrant_data"
        }
    }
}

print("Initializing Mem0 with Ollama...")
memory = Memory(config)
print("✓ Mem0 initialized!\n")

# Add beekeeping memories
print("="*70)
print("STORING BEEKEEPING MEMORIES")
print("="*70)
print()

beekeeping_notes = [
    "Inspected Hive Jodi today - absolutely incredible! 15 frames of brood, very strong population. Dennis's genetics are proving to be amazing. Definitely splitting this one in spring.",

    "Hive 5 is critically weak - only saw about 10 bees on the inner cover. Population is way too small for winter cluster. Need to combine this hive immediately or it won't survive.",

    "Applied thymol treatment round 2 to all five hives. Used conservative dosing - 20 units for smaller colonies, 35 units for Hive Jodi since it's so strong. Dennis's double stack got 20 units.",

    "Split from Jodi's hive has recovered really well after we started fall feeding. Now showing 8 frames and good entrance activity. Glad we didn't give up on this one.",

    "Need to order 2 more Hillco feeders before spring. Currently have 3, need 5 total for the full operation. Also running low on smoker fuel - cotton seed meal is almost out.",
]

user_id = "blake"

for i, note in enumerate(beekeeping_notes, 1):
    print(f"{i}. Storing: {note[:60]}...")

    messages = [{"role": "user", "content": note}]
    result = memory.add(messages=messages, user_id=user_id)

    # Show what Mem0 extracted
    if result and 'results' in result:
        extracted = result['results']
        if extracted:
            print(f"   ✓ Mem0 extracted {len(extracted)} facts:")
            for fact in extracted[:3]:  # Show first 3
                mem_text = fact.get('memory', '')
                print(f"     • {mem_text}")
    print()

# Search memories
print("="*70)
print("SEMANTIC SEARCH WITH MEM0")
print("="*70)
print()

queries = [
    "Which hives are strong?",
    "What needs attention before winter?",
    "Tell me about Dennis's genetics",
    "What equipment do I need to order?",
]

for query in queries:
    print(f"Query: '{query}'")
    print("-" * 70)

    results = memory.search(query=query, user_id=user_id, limit=3)

    if results and 'results' in results:
        for i, item in enumerate(results['results'], 1):
            mem_text = item.get('memory', 'N/A')
            score = item.get('score', 0)
            print(f"  [{score:.3f}] {mem_text}")
    else:
        print("  No results found")

    print()

# Get all memories for user
print("="*70)
print("ALL MEMORIES FOR USER")
print("="*70)
print()

all_memories = memory.get_all(user_id=user_id)
if all_memories and 'results' in all_memories:
    print(f"Total memories stored: {len(all_memories['results'])}\n")
    for i, mem in enumerate(all_memories['results'][:5], 1):
        mem_text = mem.get('memory', 'N/A')
        print(f"{i}. {mem_text}")

print()
print("="*70)
print("✓ MEM0 DEMO COMPLETE!")
print("="*70)
print()
print("What Mem0 does:")
print("  • Extracts key facts and entities automatically")
print("  • Understands context (strong vs weak hives)")
print("  • Semantic search (not just keywords)")
print("  • Links related memories")
print()
print("Data stored in: ./beekeeping_qdrant_data/")
print("Models used:")
print("  • LLM: llama3.2:3b (reasoning)")
print("  • Embedder: nomic-embed-text (semantic vectors)")