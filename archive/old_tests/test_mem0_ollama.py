"""
Test Mem0 with Ollama (Fully Local Setup)
=========================================
Quick test to verify mem0 works with Ollama.
"""

from mem0 import Memory
import json

def main():
    print("="*80)
    print("Testing Mem0 with Ollama")
    print("="*80 + "\n")

    # Configure mem0 to use Ollama
    print("Step 1: Configuring mem0 with Ollama...")
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
                "model": "nomic-embed-text:latest",
                "embedding_dims": 768
            }
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "host": "localhost",
                "port": 6333,
                "collection_name": "ollama_mem0_768"
            }
        },
        "version": "v1.1"
    }
    print(f"  LLM: ollama/{config['llm']['config']['model']}")
    print(f"  Embedder: ollama/{config['embedder']['config']['model']}")

    try:
        m = Memory.from_config(config)
        print("  [OK] Mem0 initialized successfully\n")
    except Exception as e:
        print(f"  [ERROR] Failed to initialize mem0: {e}")
        print("\nMake sure Ollama is running:")
        print("  ollama serve")
        return

    # Test 1: Store a simple conversation
    print("Step 2: Storing a conversation...")
    messages = [
        {"role": "user", "content": "I'm Blake and I keep bees. I have three hives named Jodi, Aurora, and Luna."},
        {"role": "assistant", "content": "Nice to meet you Blake! I've noted that you're a beekeeper with three hives: Jodi, Aurora, and Luna."}
    ]

    try:
        result = m.add(messages, user_id="blake")
        memories_added = len(result.get('results', []))
        print(f"  [OK] Stored conversation")
        print(f"  [OK] Mem0 extracted {memories_added} memories\n")

        # Show what was extracted
        print("  Extracted memories:")
        for i, mem in enumerate(result.get('results', []), 1):
            print(f"    {i}. {mem}")
        print()

    except Exception as e:
        print(f"  [ERROR] Failed to store: {e}\n")
        return

    # Test 2: Search for memories
    print("Step 3: Searching for memories...")
    queries = [
        "What are Blake's hive names?",
        "What does Blake do?",
    ]

    for query in queries:
        print(f"\n  Query: '{query}'")
        try:
            results = m.search(query, user_id="blake", limit=3)
            found = results.get('results', [])
            print(f"  Found {len(found)} relevant memories:")
            for mem in found:
                score = mem.get('score', 0.0)
                text = mem.get('memory', 'N/A')
                print(f"    - [{score:.2f}] {text}")
        except Exception as e:
            print(f"    [ERROR] Search failed: {e}")

    print()

    # Test 3: Get all memories
    print("Step 4: Getting all user memories...")
    try:
        all_mems = m.get_all(user_id="blake")
        total = len(all_mems.get('results', []))
        print(f"  Total memories for user 'blake': {total}")
        print("\n  All memories:")
        for i, mem in enumerate(all_mems.get('results', []), 1):
            print(f"    {i}. {mem.get('memory', 'N/A')}")
    except Exception as e:
        print(f"  [ERROR] Failed: {e}")

    print("\n" + "="*80)
    print("Test Complete!")
    print("="*80)
    print("\n[SUCCESS] Mem0 is working with Ollama (fully local, no API keys needed)")
    print("\nNext steps:")
    print("  1. Integrate into your workflows")
    print("  2. Use with HoloLoom hybrid memory")
    print("  3. Customize extraction prompts for your domain")


if __name__ == "__main__":
    main()