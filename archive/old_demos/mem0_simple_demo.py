"""
Simple Mem0 Demo - No HoloLoom Dependencies
============================================
Demonstrates basic mem0 usage without requiring PyTorch or HoloLoom.

Run: python mem0_simple_demo.py
"""

from mem0 import Memory

def main():
    print("="*80)
    print("Simple Mem0 Demo")
    print("="*80 + "\n")

    # Initialize mem0 with Ollama (local mode - no API key needed)
    print("Step 1: Initializing mem0 with Ollama...")

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
                "model": "nomic-embed-text:latest"
            }
        },
        "version": "v1.1"
    }

    try:
        m = Memory.from_config(config)
        print("  ✓ Mem0 initialized with Ollama\n")
    except Exception as e:
        print(f"  ⚠ Could not initialize with Ollama: {e}")
        print("  Make sure Ollama is running: ollama serve")
        print("  And models are installed:")
        print("    ollama pull llama3.2:3b")
        print("    ollama pull nomic-embed-text")
        return

    # Store some memories as conversations
    print("Step 2: Storing memories...")

    # Conversation 1: User preferences
    messages_1 = [
        {"role": "user", "content": "I prefer organic treatments for my bees"},
        {"role": "assistant", "content": "Got it! I'll remember you prefer organic treatments."}
    ]
    result_1 = m.add(messages_1, user_id="blake")
    print(f"  Stored conversation 1: {len(result_1.get('results', []))} memories extracted")

    # Conversation 2: Hive names
    messages_2 = [
        {"role": "user", "content": "My three hives are named Jodi, Aurora, and Luna"},
        {"role": "assistant", "content": "Nice names! I've noted your three hives: Jodi, Aurora, and Luna."}
    ]
    result_2 = m.add(messages_2, user_id="blake")
    print(f"  Stored conversation 2: {len(result_2.get('results', []))} memories extracted")

    # Conversation 3: Specific hive status
    messages_3 = [
        {"role": "user", "content": "Hive Jodi has 8 frames of brood and is very strong"},
        {"role": "assistant", "content": "That's great! Jodi sounds like a healthy hive with lots of brood."}
    ]
    result_3 = m.add(messages_3, user_id="blake")
    print(f"  Stored conversation 3: {len(result_3.get('results', []))} memories extracted\n")

    # Retrieve all memories for the user
    print("Step 3: Retrieving all user memories...")
    all_memories = m.get_all(user_id="blake")
    print(f"  Total memories: {len(all_memories.get('results', []))}")
    print("\n  Memories:")
    for i, mem in enumerate(all_memories.get('results', []), 1):
        print(f"    {i}. {mem.get('memory', 'N/A')}")
    print()

    # Search for relevant memories
    print("Step 4: Searching for relevant memories...")

    queries = [
        "What are my hive names?",
        "What treatments do I prefer?",
        "How is Jodi doing?"
    ]

    for query in queries:
        print(f"\n  Query: '{query}'")
        results = m.search(query, user_id="blake", limit=3)
        print(f"  Found {len(results.get('results', []))} relevant memories:")
        for mem in results.get('results', []):
            score = mem.get('score', 0.0)
            text = mem.get('memory', 'N/A')
            print(f"    - [{score:.2f}] {text}")

    print("\n" + "="*80)
    print("Demo Complete!")
    print("="*80 + "\n")

    print("What mem0 did:")
    print("  ✓ Automatically extracted important facts from conversations")
    print("  ✓ Associated memories with user 'blake'")
    print("  ✓ Made memories searchable by semantic similarity")
    print("  ✓ Ranked results by relevance")
    print()

    print("Next steps:")
    print("  1. Try the full HoloLoom integration (hybrid_memory_example.py)")
    print("  2. Use different user_ids for multi-user tracking")
    print("  3. Add memory filtering and decay")
    print("  4. Integrate with your agent workflows")


if __name__ == "__main__":
    main()
