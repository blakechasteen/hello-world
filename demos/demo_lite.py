"""
HoloLoom Lite Demo

Demonstrates the simplified 5-method API:
1. experience() - Store memories
2. recall() - Retrieve memories
3. reflect() - Learn from feedback
4. reason() - Agentic reasoning
5. check_safety() - Safety evaluation

Also shows:
- Performance comparison with startup time
- Reasoning modes (direct, verify, research)
- RAG query integration
- Safety guardrails

Usage:
    python demos/demo_lite.py

Date: December 2025
"""

import asyncio
import time


async def demo_basic_operations():
    """Demo 1: Basic experience/recall/reflect."""
    print("\n" + "=" * 60)
    print("Demo 1: Basic Operations (experience/recall/reflect)")
    print("=" * 60)

    from hololoom import HoloLoomLite

    async with HoloLoomLite() as loom:
        # Store some memories
        print("\n1. Storing memories...")

        memories_to_store = [
            "Thompson Sampling balances exploration and exploitation in multi-armed bandits",
            "It uses Bayesian posterior updates to estimate arm rewards",
            "UCB1 is a deterministic alternative using upper confidence bounds",
            "Epsilon-greedy explores with fixed probability epsilon",
            "Contextual bandits incorporate features into decision making",
        ]

        for content in memories_to_store:
            mem = await loom.experience(content)
            print(f"   Stored: {mem.id[:8]}... - {content[:40]}...")

        # Get metrics
        metrics = loom.get_metrics()
        print(f"\n   Total memories: {metrics['n_memories']}")

        # Recall memories
        print("\n2. Recalling memories...")
        query = "What is Thompson Sampling?"
        memories = await loom.recall(query, limit=3)

        print(f"   Query: '{query}'")
        print(f"   Found {len(memories)} relevant memories:")
        for i, mem in enumerate(memories, 1):
            text = mem.text[:60] + "..." if len(mem.text) > 60 else mem.text
            print(f"   {i}. {text}")

        # Reflect on the recall
        print("\n3. Reflecting on memories...")
        await loom.reflect(memories, {"helpful": True, "relevance": 0.9})
        print("   Feedback recorded (helpful=True, relevance=0.9)")

        # Summary
        print("\n4. System summary:")
        print(loom.summary())


async def demo_reasoning_modes():
    """Demo 2: Agentic reasoning modes."""
    print("\n" + "=" * 60)
    print("Demo 2: Agentic Reasoning Modes")
    print("=" * 60)

    from hololoom import HoloLoomLite

    async with HoloLoomLite() as loom:
        # Store some knowledge first
        await loom.experience("Thompson Sampling is a Bayesian algorithm for multi-armed bandits")
        await loom.experience("It samples from posterior distributions to balance exploration")
        await loom.experience("The algorithm was developed by William R. Thompson in 1933")

        query = "Explain how Thompson Sampling works"

        # Direct mode (fastest)
        print("\n1. DIRECT mode (~150ms, single-pass):")
        start = time.time()
        result = await loom.reason(query, mode="direct")
        elapsed = (time.time() - start) * 1000
        print(f"   Time: {elapsed:.0f}ms")
        print(f"   Confidence: {result.confidence:.2%}")
        print(f"   Answer: {result.text[:100]}..." if len(result.text) > 100 else f"   Answer: {result.text}")

        # Verify mode (with verification)
        print("\n2. VERIFY mode (~600ms, with verification):")
        start = time.time()
        result = await loom.reason(query, mode="verify")
        elapsed = (time.time() - start) * 1000
        print(f"   Time: {elapsed:.0f}ms")
        print(f"   Confidence: {result.confidence:.2%}")
        print(f"   Verified: {result.metadata.get('verified', 'N/A')}")

        # Research mode (multi-query exploration)
        print("\n3. RESEARCH mode (~900ms, multi-query):")
        start = time.time()
        result = await loom.reason("Compare Thompson Sampling with UCB", mode="research")
        elapsed = (time.time() - start) * 1000
        print(f"   Time: {elapsed:.0f}ms")
        print(f"   Confidence: {result.confidence:.2%}")
        print(f"   Steps taken: {result.metadata.get('steps_taken', 0)}")


async def demo_safety_checks():
    """Demo 3: Safety guardrails."""
    print("\n" + "=" * 60)
    print("Demo 3: Safety Guardrails")
    print("=" * 60)

    from hololoom import HoloLoomLite

    async with HoloLoomLite(enable_safety=True) as loom:
        # Check various actions
        actions = [
            ("print_message", {"message": "Hello, world!"}),
            ("execute_code", {"code": "print('Hello')"}),
            ("access_file", {"path": "/etc/passwd"}),
            ("delete_data", {"scope": "all"}),
        ]

        print("\nChecking action safety levels:")
        for action, context in actions:
            result = await loom.check_safety(action, context)
            status = "SAFE" if result.confidence > 0.7 else "CAUTION" if result.confidence > 0.4 else "RISKY"
            icon = "✅" if status == "SAFE" else "⚠️" if status == "CAUTION" else "❌"
            print(f"   {icon} {action}: {status} (confidence: {result.confidence:.2%})")
            if result.metadata.get('reason'):
                print(f"      Reason: {result.metadata['reason']}")


async def demo_startup_performance():
    """Demo 4: Startup performance comparison."""
    print("\n" + "=" * 60)
    print("Demo 4: Startup Performance")
    print("=" * 60)

    # Measure Lite startup time
    print("\n1. HoloLoom Lite startup time:")
    start = time.time()

    from hololoom import HoloLoomLite
    loom = HoloLoomLite()
    await loom._initialize()

    lite_time = (time.time() - start) * 1000
    print(f"   Lite startup: {lite_time:.0f}ms")

    # Measure experience operation
    start = time.time()
    await loom.experience("Test memory for timing")
    exp_time = (time.time() - start) * 1000
    print(f"   First experience: {exp_time:.0f}ms")

    # Measure recall
    start = time.time()
    await loom.recall("test")
    recall_time = (time.time() - start) * 1000
    print(f"   First recall: {recall_time:.0f}ms")

    # Summary
    print(f"\n2. Performance summary:")
    print(f"   Total setup + first query: {lite_time + exp_time + recall_time:.0f}ms")
    print(f"   Target: <500ms (HoloLoom Full: ~2-3s)")


async def demo_rag_query():
    """Demo 5: RAG query integration."""
    print("\n" + "=" * 60)
    print("Demo 5: RAG Query Integration")
    print("=" * 60)

    from hololoom import HoloLoomLite

    async with HoloLoomLite() as loom:
        # Store knowledge
        print("\n1. Building knowledge base...")
        knowledge = [
            "Python uses indentation to define code blocks instead of braces",
            "Python is dynamically typed and supports duck typing",
            "The Zen of Python emphasizes readability and simplicity",
            "Python was created by Guido van Rossum in 1991",
            "Python supports multiple paradigms: procedural, OOP, functional",
        ]

        for k in knowledge:
            await loom.experience(k)

        print(f"   Stored {len(knowledge)} knowledge items")

        # Query with RAG
        print("\n2. RAG query:")
        question = "What makes Python unique as a programming language?"

        start = time.time()
        result = await loom.query(question)
        elapsed = (time.time() - start) * 1000

        print(f"   Question: {question}")
        print(f"   Time: {elapsed:.0f}ms")
        print(f"   Confidence: {result.confidence:.2%}")
        print(f"   Answer: {result.text}")

        if result.sources:
            print(f"   Sources: {len(result.sources)} memories used")


async def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("  HoloLoom Lite Demo")
    print("  Simplified 5-Method API")
    print("=" * 60)

    # Run demos
    try:
        await demo_basic_operations()
    except Exception as e:
        print(f"\n⚠️  Demo 1 error: {e}")

    try:
        await demo_reasoning_modes()
    except Exception as e:
        print(f"\n⚠️  Demo 2 error: {e}")

    try:
        await demo_safety_checks()
    except Exception as e:
        print(f"\n⚠️  Demo 3 error: {e}")

    try:
        await demo_startup_performance()
    except Exception as e:
        print(f"\n⚠️  Demo 4 error: {e}")

    try:
        await demo_rag_query()
    except Exception as e:
        print(f"\n⚠️  Demo 5 error: {e}")

    # Final summary
    print("\n" + "=" * 60)
    print("  Demo Complete!")
    print("=" * 60)
    print("""
Try the UI modes:
    python -m HoloLoom.lite repl       # Simple REPL
    python -m HoloLoom.lite terminal   # Rich terminal (requires: rich)
    python -m HoloLoom.lite web        # Web chat (requires: fastapi, uvicorn)
    python -m HoloLoom.lite desktop    # Desktop app (requires: gradio)
""")


if __name__ == "__main__":
    asyncio.run(main())
