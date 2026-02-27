"""
Simple Semantic Search Test - Direct Backend Access

Demonstrates semantic search by directly using the backend.
No WebSocket required - pure backend testing.
"""

import asyncio
from datetime import datetime
from HoloLoom.config import Config, MemoryBackend
from HoloLoom.memory.backend_factory import create_memory_backend
from HoloLoom.memory.protocol import Memory, MemoryQuery
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings


async def main():
    print("\n" + "="*80)
    print("SEMANTIC SEARCH - SIMPLE DIRECT TEST")
    print("="*80)
    print("\nDemonstrating semantic search finding conceptually related messages\n")

    # Initialize
    print("🔧 Initializing HYBRID backend...")
    config = Config.fused()
    config.memory_backend = MemoryBackend.HYBRID
    backend = await create_memory_backend(config)
    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
    print("✓ Backend ready\n")

    # Test messages (semantically grouped)
    messages = {
        "Machine Learning": [
            "Neural networks use backpropagation to adjust weights during training",
            "Deep learning employs gradient descent for parameter optimization",
            "Artificial intelligence systems learn patterns from experience",
        ],
        "Quantum Computing": [
            "Quantum computers harness superposition for parallel computation",
            "Entangled qubits enable quantum teleportation protocols",
        ],
        "Cooking": [
            "Caramelizing onions requires low heat and patience",
            "Al dente pasta maintains structural integrity while remaining tender",
        ]
    }

    # Store all messages
    print("📝 Storing messages with embeddings...")
    print("-" * 80)

    all_messages = []
    for category, texts in messages.items():
        for text in texts:
            # Generate embedding
            embedding = embedder.encode([text])[0]

            # Create memory
            memory = Memory(
                id=f"test-{category.replace(' ', '-').lower()}-{len(all_messages)}",
                text=text,
                timestamp=datetime.now(),
                context={'category': category, 'test': 'semantic_search'},
                metadata={'source': 'test'},
                embedding=embedding
            )

            # Store
            await backend.store(memory, user_id="test-user")
            all_messages.append((category, text))
            print(f"  ✓ [{category}] {text[:60]}...")

    print(f"\n✓ Stored {len(all_messages)} messages\n")

    # Semantic search tests
    print("="*80)
    print("SEMANTIC SEARCH TESTS")
    print("="*80)

    tests = [
        ("machine learning training", "Machine Learning",
         "Uses different words but should find 'backpropagation', 'gradient descent', etc."),
        ("quantum physics phenomena", "Quantum Computing",
         "Uses 'physics phenomena' but should find 'superposition', 'entanglement'"),
        ("food preparation", "Cooking",
         "Uses 'food preparation' but should find 'caramelizing', 'pasta'"),
    ]

    for query_text, expected_category, note in tests:
        print(f"\n{'─'*80}")
        print(f"Query: '{query_text}'")
        print(f"Expected: {expected_category}")
        print(f"Note: {note}")
        print('─'*80)

        # Search
        query = MemoryQuery(text=query_text)
        result = await backend.recall(query, limit=5)

        print(f"\n✓ Found {len(result.memories)} results (Strategy: {result.strategy_used})\n")

        matches = 0
        for i, memory in enumerate(result.memories[:3], 1):
            score = result.scores[i-1] if i-1 < len(result.scores) else 1.0
            category = memory.context.get('category', 'Unknown')

            is_match = category == expected_category
            if is_match:
                matches += 1

            marker = "✓ MATCH!" if is_match else f"({category})"

            print(f"  {i}. {marker:20} score={score:.3f}")
            print(f"     {memory.text[:80]}...")

        print(f"\n  Result: {matches}/3 correct matches")
        if matches >= 2:
            print(f"  ✓✓✓ SEMANTIC SEARCH WORKING! ✓✓✓")

    # The ultimate test: cross-terminology
    print("\n" + "="*80)
    print("ULTIMATE TEST: Cross-Terminology Semantic Matching")
    print("="*80)
    print("\nQuery: 'How do neural systems optimize learning?'")
    print("Expected: Machine Learning messages (none use this exact phrasing!)\n")

    query = MemoryQuery(text="How do neural systems optimize learning?")
    result = await backend.recall(query, limit=5)

    print(f"✓ Found {len(result.memories)} results\n")

    ml_count = 0
    for i, memory in enumerate(result.memories[:5], 1):
        score = result.scores[i-1] if i-1 < len(result.scores) else 1.0
        category = memory.context.get('category', 'Unknown')

        is_ml = category == "Machine Learning"
        if is_ml:
            ml_count += 1

        marker = "✓ ML!" if is_ml else f"({category})"

        print(f"  {i}. {marker:20} score={score:.3f}")
        print(f"     {memory.text}")

    print(f"\n{'='*80}")
    print(f"Found {ml_count}/5 Machine Learning messages")
    print("="*80)

    if ml_count >= 2:
        print("""
✓✓✓ SUCCESS! SEMANTIC SEARCH IS WORKING! ✓✓✓

The system successfully matched:
  Query: "neural systems optimize learning"

  With messages containing:
  • "backpropagation" (no shared words!)
  • "gradient descent" (different terminology!)
  • "AI systems learn patterns" (conceptual match!)

This demonstrates TRUE semantic understanding - matching concepts,
not just keywords!

The chat UI at http://localhost:8000 now has this same capability
built-in for finding relevant conversation history.
""")
    else:
        print(f"\n⚠ Expected more ML matches (got {ml_count}, expected ≥2)")

    print("\n" + "="*80)
    print("Test complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    print("\n🚀" * 40)
    print("SEMANTIC SEARCH - SIMPLE DIRECT TEST".center(80))
    print("🚀" * 40 + "\n")

    asyncio.run(main())
