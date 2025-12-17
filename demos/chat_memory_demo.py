"""
Simple Chat with Memory Demo
============================
Non-interactive demo showing HoloLoom's memory system working.
Processes a few queries to demonstrate the weaving cycle.

Usage:
    PYTHONPATH=. python demos/chat_memory_demo.py
"""

import asyncio
import sys
import os

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from HoloLoom.config import Config, MemoryBackend
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.protocols.types import Query


async def main():
    print("=" * 60)
    print("HoloLoom Chat with Memory Demo")
    print("=" * 60)
    print()

    # Initialize
    print("[1] Initializing HoloLoom...")
    config = Config.fast()
    # Force in-memory backend to avoid slow connection retries
    config.memory_backend = MemoryBackend.INMEMORY
    print("    Using in-memory backend (fast startup)")

    # Create orchestrator
    print("[2] Creating WeavingOrchestrator...")

    # Create some test shards for memory
    from HoloLoom.protocols.types import MemoryShard
    import uuid
    test_shards = [
        MemoryShard(
            id=str(uuid.uuid4()),
            text="Thompson Sampling is a Bayesian algorithm for the multi-armed bandit problem. It balances exploration and exploitation by sampling from posterior distributions.",
            entities=["Thompson Sampling", "Bayesian", "multi-armed bandit", "exploration", "exploitation"],
            motifs=["algorithm", "probability"],
            timestamp=0.0,
            metadata={"source": "knowledge_base", "confidence": 0.95}
        ),
        MemoryShard(
            id=str(uuid.uuid4()),
            text="HoloLoom uses a 9-step weaving cycle: Loom Command, Chrono Trigger, Yarn Graph, Resonance Shed, Warp Space, Memory Crawl, Convergence, Tool Execution, Spacetime.",
            entities=["HoloLoom", "Loom Command", "Chrono Trigger", "Yarn Graph", "Resonance Shed", "Warp Space"],
            motifs=["weaving", "pipeline"],
            timestamp=0.0,
            metadata={"source": "documentation", "confidence": 0.9}
        ),
        MemoryShard(
            id=str(uuid.uuid4()),
            text="The memory system supports multiple backends: Neo4j for graphs, Qdrant for vectors, and NetworkX for in-memory fallback.",
            entities=["Neo4j", "Qdrant", "NetworkX", "memory system"],
            motifs=["database", "storage"],
            timestamp=0.0,
            metadata={"source": "documentation", "confidence": 0.9}
        ),
    ]

    try:
        orchestrator = WeavingOrchestrator(
            cfg=config,
            shards=test_shards,
            enable_complexity_auto_detect=True
        )
        print("    Orchestrator created successfully!")
    except Exception as e:
        print(f"    Error creating orchestrator: {e}")
        return 1

    # Process queries
    queries = [
        "What is Thompson Sampling?",
        "How does HoloLoom work?",
        "Tell me about the memory system",
    ]

    print()
    print("[3] Processing queries through weaving cycle...")
    print("-" * 60)

    for i, query_text in enumerate(queries, 1):
        print(f"\nQuery {i}: {query_text}")
        print("-" * 40)

        try:
            query = Query(text=query_text)
            spacetime = await orchestrator.weave(query)

            print(f"Response: {spacetime.response[:200]}..." if len(spacetime.response) > 200 else f"Response: {spacetime.response}")
            print(f"Confidence: {spacetime.confidence:.2f}")
            print(f"Tool used: {spacetime.metadata.get('tool_used', 'unknown')}")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 60)
    print("Demo complete!")
    print()
    print("To run the FULL INTERACTIVE session, open a terminal and run:")
    print("  cd", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print("  set PYTHONPATH=.")
    print("  python demos/terminal_ui_demo.py")
    print("  Then select option 6 for Interactive session")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
