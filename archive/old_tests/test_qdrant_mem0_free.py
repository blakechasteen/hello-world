"""
Mem0 + Qdrant with 100% FREE Ollama Backend
============================================

No OpenAI API key needed! Uses your local Ollama models.

Prerequisites:
1. Qdrant: docker ps | findstr qdrant (should be running)
2. Ollama: ollama serve (should be running)
3. Models: You already have llama3.2:3b and nomic-embed-text

Run: python test_qdrant_mem0_free.py
"""

from mem0 import Memory
import logging

logging.basicConfig(level=logging.INFO)

print("="*80)
print("Testing Mem0 + Qdrant with FREE Ollama Backend")
print("="*80 + "\n")

# Configure mem0 to use Ollama (100% free, local)
config = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "llama3.2:3b",
            "temperature": 0.1,
            "max_tokens": 1500,
        }
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text"
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333,
            "collection_name": "free_memories"
        }
    },
    "version": "v1.1"
}

print("Initializing mem0...")
print("  LLM: Ollama (llama3.2:3b)")
print("  Embedder: Ollama (nomic-embed-text)")
print("  Vector DB: Qdrant (localhost:6333)")
print()

try:
    m = Memory.from_config(config)
    print("[OK] Connected successfully\n")
except Exception as e:
    print(f"[ERROR] Failed to initialize: {e}\n")
    print("Make sure:")
    print("  1. Qdrant is running: docker ps | findstr qdrant")
    print("  2. Ollama is running: ollama serve")
    exit(1)

# Test 1: Store a memory
print("Test 1: Storing a memory...")
try:
    messages = [
        {"role": "user", "content": "I'm Blake, a beekeeper with three hives: Jodi, Aurora, and Luna."}
    ]

    result = m.add(messages, user_id="blake")
    extracted = len(result.get('results', []))

    print(f"[OK] Stored successfully")
    print(f"[OK] Extracted {extracted} memories")

    if extracted > 0:
        print("\nExtracted facts:")
        for i, mem_data in enumerate(result.get('results', []), 1):
            # Handle both dict and string formats
            if isinstance(mem_data, dict):
                mem_text = mem_data.get('memory', str(mem_data))
            else:
                mem_text = str(mem_data)
            print(f"  {i}. {mem_text}")
    print()

except Exception as e:
    print(f"[ERROR] Failed to store: {e}")
    print("\nNote: There may be compatibility issues with Ollama embeddings.")
    print("If this fails, use the simple fallback pattern instead:")
    print("  from mem0_simple_integration import UserMemory")
    print("  memory = UserMemory(provider='openai')  # Falls back gracefully")
    exit(1)

# Test 2: Search memories
print("Test 2: Searching memories...")
try:
    query = "What are Blake's hives?"
    print(f"Query: '{query}'")

    results = m.search(query, user_id="blake", limit=3)
    found = results.get('results', [])

    print(f"Found {len(found)} results:")
    for mem in found:
        score = mem.get('score', 0.0)
        text = mem.get('memory', 'N/A')
        print(f"  - [{score:.2f}] {text}")
    print()

except Exception as e:
    print(f"[ERROR] Search failed: {e}\n")

# Test 3: Get all memories
print("Test 3: Getting all memories for user...")
try:
    all_mems = m.get_all(user_id="blake")
    total = len(all_mems.get('results', []))

    print(f"Total memories: {total}")
    if total > 0:
        print("\nAll memories:")
        for i, mem in enumerate(all_mems.get('results', []), 1):
            print(f"  {i}. {mem.get('memory', 'N/A')}")
    print()

except Exception as e:
    print(f"[ERROR] Failed: {e}\n")

print("="*80)
print("Success! No OpenAI API key needed.")
print("="*80)
print("\nYou're using:")
print("  - Ollama for LLM extraction (FREE)")
print("  - Ollama for embeddings (FREE)")
print("  - Qdrant for vector storage (FREE)")
print("\nData stored in Qdrant: http://localhost:6333/dashboard")
print("\nIf you encounter errors, see BETTER_MEM0_INTEGRATION.md for alternatives.")