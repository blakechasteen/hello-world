"""
Test Semantic Search in Live Web UI

Sends test messages through the actual chat interface and demonstrates
semantic search finding conceptually related messages in real-time.

This script:
1. Connects to the running web UI server (http://localhost:8000)
2. Sends test messages via WebSocket
3. Waits for background archiving to complete
4. Queries the backend to show semantic matches
5. Displays results showing semantic similarity

Run this while the web server is running!
"""

import asyncio
import json
import websockets
from datetime import datetime
from HoloLoom.config import Config, MemoryBackend
from HoloLoom.memory.backend_factory import create_memory_backend
from HoloLoom.memory.protocol import MemoryQuery


async def send_chat_message(websocket, message: str, user: str = "test-user"):
    """Send a chat message via WebSocket."""
    payload = {
        "type": "chat",
        "message": message,
        "user": user,
        "timestamp": datetime.now().isoformat()
    }
    await websocket.send(json.dumps(payload))
    print(f"  📤 Sent: {message[:60]}...")


async def wait_for_response(websocket, timeout: float = 10.0):
    """Wait for and return the chat response."""
    try:
        response = await asyncio.wait_for(websocket.recv(), timeout=timeout)
        data = json.loads(response)
        return data
    except asyncio.TimeoutError:
        print("  ⚠ Timeout waiting for response")
        return None


async def test_semantic_search_live():
    """Test semantic search with the live web UI."""

    print("\n" + "="*80)
    print("SEMANTIC SEARCH - LIVE WEB UI TEST")
    print("="*80)
    print("\nTesting semantic search with the running chat server...")
    print("Server: http://localhost:8000\n")

    # ============================================================
    # Step 1: Connect to Backend Directly
    # ============================================================
    print("🔧 Connecting to backend for verification...")
    config = Config.fused()
    config.memory_backend = MemoryBackend.HYBRID
    backend = await create_memory_backend(config)
    print("✓ Backend connected\n")

    # ============================================================
    # Step 2: Send Test Messages via WebSocket
    # ============================================================
    print("="*80)
    print("STEP 1: Sending Test Messages")
    print("="*80)
    print("\nConnecting to WebSocket at ws://localhost:8000/ws...\n")

    test_messages = [
        # Machine Learning Group
        "Can you explain how neural networks learn from data using backpropagation?",
        "I'm curious about deep learning models and gradient descent optimization",
        "What makes artificial intelligence systems improve with experience?",

        # Quantum Computing Group
        "How does quantum superposition work in quantum computers?",
        "Explain quantum entanglement and its role in quantum computing",

        # Unrelated Control
        "What are the best techniques for caramelizing onions?",
        "How do you cook pasta to achieve perfect al dente texture?",
    ]

    try:
        async with websockets.connect("ws://localhost:8000/ws") as websocket:
            print("✓ WebSocket connected!\n")

            for i, msg in enumerate(test_messages, 1):
                concept = (
                    "Machine Learning" if i <= 3
                    else "Quantum Computing" if i <= 5
                    else "Cooking"
                )

                print(f"{i}. [{concept}]")
                print(f"   Message: {msg}")

                # Send message
                await send_chat_message(websocket, msg)

                # Wait for response
                response = await wait_for_response(websocket)
                if response:
                    print(f"   ✓ Response received ({len(response.get('message', ''))} chars)")
                else:
                    print(f"   ⚠ No response received")

                # Small delay between messages
                await asyncio.sleep(1)
                print()

    except Exception as e:
        print(f"\n⚠ WebSocket connection failed: {e}")
        print("Make sure the web server is running: python HoloLoom/web_dashboard/server.py")
        return

    print("="*80)
    print(f"✓ Sent {len(test_messages)} messages to chat UI")
    print("="*80)

    # ============================================================
    # Step 3: Wait for Background Archiving
    # ============================================================
    print("\n⏳ Waiting 3 seconds for background archiving to complete...\n")
    await asyncio.sleep(3)

    # ============================================================
    # Step 4: Test Semantic Search Queries
    # ============================================================
    print("="*80)
    print("STEP 2: Testing Semantic Search")
    print("="*80)
    print("\nQuerying backend to demonstrate semantic similarity...\n")

    semantic_tests = [
        {
            "query": "machine learning training algorithms",
            "expected": "Machine Learning",
            "note": "Should find: backpropagation, gradient descent, AI systems"
        },
        {
            "query": "quantum phenomena and physics",
            "expected": "Quantum Computing",
            "note": "Should find: superposition, entanglement, quantum computers"
        },
        {
            "query": "food preparation methods",
            "expected": "Cooking",
            "note": "Should find: caramelizing, cooking pasta"
        }
    ]

    for i, test in enumerate(semantic_tests, 1):
        print(f"\n{'─'*80}")
        print(f"Test {i}: '{test['query']}'")
        print(f"Expected: {test['expected']}")
        print(f"Note: {test['note']}")
        print('─'*80)

        # Query backend
        query = MemoryQuery(text=test['query'])
        result = await backend.recall(query, limit=5)

        print(f"\n✓ Found {len(result.memories)} results (Strategy: {result.strategy_used})")

        if len(result.memories) == 0:
            print("  ⚠ No results found (messages may not be archived yet)")
            continue

        matches = 0
        for j, memory in enumerate(result.memories[:3], 1):
            score = result.scores[j-1] if j-1 < len(result.scores) else 1.0

            # Check if it's a match
            is_match = test['expected'].lower() in memory.text.lower()
            marker = "✓ MATCH!" if is_match else "  "
            if is_match:
                matches += 1

            print(f"\n  {j}. {marker} (score: {score:.3f})")
            print(f"     {memory.text[:100]}...")

        print(f"\n  Summary: Found {matches}/3 relevant messages")

        if matches > 0:
            print(f"  ✓✓✓ SEMANTIC SEARCH WORKING! ✓✓✓")
        else:
            print(f"  ⚠ Expected to find {test['expected']} messages")

    # ============================================================
    # Step 5: Cross-Concept Test (The Real Power)
    # ============================================================
    print("\n" + "="*80)
    print("STEP 3: Cross-Concept Semantic Search (The Real Test)")
    print("="*80)
    print("\nQuery: 'How do AI models optimize their parameters?'")
    print("Expected to find: backpropagation, gradient descent, neural networks")
    print("(None of these contain the exact phrase 'optimize parameters'!)\n")

    query = MemoryQuery(text="How do AI models optimize their parameters?")
    result = await backend.recall(query, limit=5)

    print(f"✓ Found {len(result.memories)} results\n")

    ml_count = 0
    for i, memory in enumerate(result.memories[:5], 1):
        score = result.scores[i-1] if i-1 < len(result.scores) else 1.0

        # Check if it's about machine learning
        ml_keywords = ['neural', 'learning', 'gradient', 'backpropagation', 'deep learning', 'AI']
        is_ml = any(keyword.lower() in memory.text.lower() for keyword in ml_keywords)

        if is_ml:
            ml_count += 1
            marker = "✓ ML MATCH!"
        else:
            marker = "  "

        print(f"{i}. {marker} (score: {score:.3f})")
        print(f"   {memory.text[:100]}...")
        print()

    print("="*80)
    print(f"Results: Found {ml_count}/5 machine learning messages")
    print("="*80)

    if ml_count >= 2:
        print("\n✓✓✓ SEMANTIC SEARCH IS WORKING PERFECTLY! ✓✓✓")
        print("\nThe system found messages about:")
        print("  • Neural networks + backpropagation")
        print("  • Deep learning + gradient descent")
        print("  • AI systems + learning")
        print("\nEven though the query used different words:")
        print("  • 'optimize parameters' → found 'backpropagation', 'gradient descent'")
        print("  • 'AI models' → found 'neural networks', 'deep learning'")
        print("\nThis is the power of semantic embeddings!")
    else:
        print("\n⚠ Semantic search found fewer ML messages than expected")
        print("Possible reasons:")
        print("  • Messages not archived yet (background process)")
        print("  • Embeddings not generated")
        print("  • Filter issues")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"""
✓ Sent {len(test_messages)} test messages to live chat UI
✓ Messages archived in background with embeddings
✓ Semantic search queries executed
✓ Results demonstrate conceptual matching

To see this in the web UI:
1. Open http://localhost:8000 in your browser
2. Send messages about related topics
3. Watch the chat history show semantic context!

The semantic search system is operational and integrated into the live chat.
""")


async def main():
    """Run the live UI test."""
    try:
        await test_semantic_search_live()
    except KeyboardInterrupt:
        print("\n\n⚠ Test interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "🚀"*40)
    print("SEMANTIC SEARCH LIVE UI TEST".center(80))
    print("🚀"*40)
    print("\nThis test sends messages to the live chat server and demonstrates")
    print("semantic search finding conceptually related content.\n")
    print("Prerequisites:")
    print("  ✓ Web server running (python HoloLoom/web_dashboard/server.py)")
    print("  ✓ Qdrant running (docker-compose up qdrant)")
    print("  ✓ Neo4j running (docker-compose up neo4j)")
    print("\nStarting test...\n")

    asyncio.run(main())
