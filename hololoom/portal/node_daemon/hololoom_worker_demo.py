"""
HoloLoom Worker Demo

Shows how to use HoloLoomWorker for local distributed intelligence.

Run from repo root:
    PYTHONPATH=. python hololoom/portal/node_daemon/hololoom_worker_demo.py
"""

import asyncio
import json
from hololoom_worker import HoloLoomWorker


async def demo_basic_operations():
    """Demonstrate basic recall, experience, and weave operations."""
    print("=" * 70)
    print("HoloLoom Worker Demo")
    print("=" * 70)

    # Create worker in "fast" mode
    worker = HoloLoomWorker(mode="fast")

    try:
        # 1. Experience (store memories)
        print("\n[1] Storing memories with experience()...")
        experiences = [
            "Thompson Sampling balances exploration and exploitation",
            "Bayesian methods update beliefs based on evidence",
            "Reinforcement learning learns from interaction feedback"
        ]

        for content in experiences:
            result = await worker.experience(content)
            print(f"  • Stored: {result.get('memory_id')}")
            print(f"    Time: {result.get('execution_time_ms', 0):.1f}ms")

        # 2. Recall (search memories)
        print("\n[2] Searching memories with recall()...")
        recall_result = await worker.recall("Thompson Sampling", k=3)

        if recall_result.get("status") == "success":
            print(f"  Query: '{recall_result['query']}'")
            print(f"  Found: {recall_result['count']} memories")
            for mem in recall_result.get("memories", []):
                print(f"    • {mem['text'][:60]}...")
            print(f"  Time: {recall_result.get('execution_time_ms', 0):.1f}ms")
        else:
            print(f"  Error: {recall_result.get('error')}")

        # 3. Weave (reasoning)
        print("\n[3] Reasoning with weave()...")
        weave_result = await worker.weave(
            "How does Thompson Sampling work?",
            mode="fast"
        )

        if weave_result.get("status") == "success":
            print(f"  Query: '{weave_result['query']}'")
            print(f"  Mode: {weave_result['mode']}")
            print(f"  Response: {weave_result['response'][:100]}...")
            print(f"  Confidence: {weave_result.get('confidence', 0):.2f}")
            print(f"  Time: {weave_result.get('execution_time_ms', 0):.1f}ms")
        else:
            print(f"  Error: {weave_result.get('error')}")

    finally:
        # Cleanup
        await worker.close()
        print("\n[✓] Worker closed successfully")


async def demo_batch_operations():
    """Demonstrate batch execution of multiple operations."""
    print("\n" + "=" * 70)
    print("Batch Operations Demo")
    print("=" * 70)

    worker = HoloLoomWorker(mode="fast")

    try:
        # Store multiple experiences
        print("\n[1] Storing batch of experiences...")
        experiences = [
            "Python uses indentation for code blocks",
            "JavaScript is dynamically typed",
            "Rust requires explicit memory management"
        ]

        for content in experiences:
            result = await worker.experience(content)
            if result.get("status") == "success":
                print(f"  ✓ {result.get('memory_id')}")

        # Batch recall
        print("\n[2] Searching for multiple queries...")
        queries = ["Python", "JavaScript", "types"]

        for query in queries:
            result = await worker.recall(query, k=2)
            if result.get("status") == "success":
                print(f"  Query '{query}': {result.get('count')} results")

    finally:
        await worker.close()


async def demo_operation_dispatch():
    """Demonstrate generic operation dispatch."""
    print("\n" + "=" * 70)
    print("Operation Dispatch Demo")
    print("=" * 70)

    worker = HoloLoomWorker(mode="fast")

    try:
        print("\nUsing execute() for generic dispatch...")

        # Experience via execute()
        result = await worker.execute("experience", {
            "content": "The sky is blue"
        })
        print(f"  Experience: {result.get('status')}")
        print(f"    Memory ID: {result.get('memory_id')}")
        print(f"    Time: {result.get('execution_time_ms', 0):.1f}ms")

        # Recall via execute()
        result = await worker.execute("recall", {
            "query": "blue sky",
            "k": 5
        })
        print(f"  Recall: {result.get('status')}")
        print(f"    Found: {result.get('count')} memories")
        print(f"    Time: {result.get('execution_time_ms', 0):.1f}ms")

        # Weave via execute()
        result = await worker.execute("weave", {
            "query": "Describe the sky",
            "mode": "fast"
        })
        print(f"  Weave: {result.get('status')}")
        print(f"    Confidence: {result.get('confidence', 0):.2f}")
        print(f"    Time: {result.get('execution_time_ms', 0):.1f}ms")

    finally:
        await worker.close()


async def main():
    """Run all demos."""
    try:
        await demo_basic_operations()
        await demo_batch_operations()
        await demo_operation_dispatch()

        print("\n" + "=" * 70)
        print("All demos completed!")
        print("=" * 70)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
