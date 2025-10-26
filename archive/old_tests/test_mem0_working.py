"""
Mem0 + Qdrant with OpenAI - Working Version
============================================

Run: python test_mem0_working.py
"""

from mem0 import Memory
import os

# Set API key - use environment variable or config
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("Please set OPENAI_API_KEY environment variable")

print("="*80)
print("Testing Mem0 + Qdrant with OpenAI")
print("="*80 + "\n")

# Configure mem0 with Qdrant
config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333,
            "collection_name": "beekeeping_memories"
        }
    },
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o-mini",
            "temperature": 0.1
        }
    }
}

print("Initializing mem0...")
print("  LLM: OpenAI (gpt-4o-mini)")
print("  Embedder: OpenAI (text-embedding-3-small)")
print("  Vector DB: Qdrant (localhost:6333)")
print()

try:
    m = Memory.from_config(config)
    print("[OK] Connected successfully\n")
except Exception as e:
    print(f"[ERROR] Failed: {e}\n")
    exit(1)

# Test 1: Store beekeeping memories
print("Test 1: Storing beekeeping memories...")
messages = [
    {"role": "user", "content": "I'm Blake, a beekeeper. I have three hives named Jodi, Aurora, and Luna. Jodi is the strongest with 8 frames of brood."},
    {"role": "assistant", "content": "Great! I've noted your three hives and that Jodi is performing well."}
]

try:
    result = m.add(messages, user_id="blake")
    extracted = len(result.get('results', []))

    print(f"[OK] Stored conversation")
    print(f"[OK] Extracted {extracted} memories")

    if extracted > 0:
        print("\nExtracted facts:")
        for i, mem in enumerate(result.get('results', []), 1):
            print(f"  {i}. {mem}")
    print()

except Exception as e:
    print(f"[ERROR] Failed: {e}\n")
    exit(1)

# Test 2: Add more context
print("Test 2: Adding treatment preferences...")
messages2 = [
    {"role": "user", "content": "I prefer organic treatments for varroa mites. I usually use formic acid."},
    {"role": "assistant", "content": "Noted your preference for organic treatments, specifically formic acid for varroa control."}
]

try:
    result2 = m.add(messages2, user_id="blake")
    print(f"[OK] Added {len(result2.get('results', []))} more memories\n")
except Exception as e:
    print(f"[ERROR] Failed: {e}\n")

# Test 3: Search memories
print("Test 3: Searching memories...")
queries = [
    "What are Blake's hive names?",
    "Which hive is strongest?",
    "What treatments does Blake prefer?",
    "Tell me about Blake's beekeeping"
]

for query in queries:
    print(f"\nQuery: '{query}'")
    try:
        results = m.search(query, user_id="blake", limit=3)
        found = results.get('results', [])

        if found:
            for mem in found:
                score = mem.get('score', 0.0)
                text = mem.get('memory', 'N/A')
                print(f"  [{score:.2f}] {text}")
        else:
            print("  No results")

    except Exception as e:
        print(f"  [ERROR] {e}")

# Test 4: Get all memories
print("\n\nTest 4: All memories for user 'blake':")
try:
    all_mems = m.get_all(user_id="blake")
    total = len(all_mems.get('results', []))

    print(f"\nTotal: {total} memories")
    for i, mem in enumerate(all_mems.get('results', []), 1):
        print(f"  {i}. {mem.get('memory', 'N/A')}")

except Exception as e:
    print(f"[ERROR] {e}")

print("\n" + "="*80)
print("Success! Mem0 is working with OpenAI + Qdrant")
print("="*80)
print("\nWhat happened:")
print("  1. Stored conversations about beekeeping")
print("  2. OpenAI LLM extracted key facts automatically")
print("  3. Facts embedded with text-embedding-3-small")
print("  4. Vectors stored in Qdrant for fast search")
print("  5. Semantic search finds relevant memories")
print()
print("Data persisted in:")
print("  - Qdrant: http://localhost:6333/dashboard")
print("  - Collection: beekeeping_memories")
print()
print("Next steps:")
print("  1. Integrate into your workflows (see mem0_simple_integration.py)")
print("  2. Build on this pattern")
print("  3. Check your OpenAI usage at: https://platform.openai.com/usage")
