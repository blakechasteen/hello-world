
import asyncio
import os
import sys
from pathlib import Path

# Ensure project root is in path
# Ensure project root is in path (mythRL directory)
# examples/demo_maker.py -> examples/ -> HoloLoom/ -> mythRL/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from hololoom.fabric.materializer import Materializer
from hololoom.memory.stores.in_memory_store import InMemoryStore
from hololoom.memory.unified import UnifiedMemory
from hololoom.protocols.types import Query

from hololoom.config import Config
from hololoom.core.orchestrator.weaving_orchestrator import WeavingOrchestrator


async def main():
    print("Initializing HoloLoom Maker Demo...")

    # 1. Setup
    config = Config()
    # Ensure we use a safe output directory for the demo
    demo_output_dir = Path("c:/Users/blake/OneDrive/Documents/mythRL/HoloLoom/demo_output")
    demo_output_dir.mkdir(exist_ok=True)

    demo_output_dir.mkdir(exist_ok=True)

    # Initialize basic memory (required for orchestrator)
    # Use InMemoryStore for the demo to avoid dependency issues
    try:
        backend = InMemoryStore()
        memory = UnifiedMemory(
            user_id="demo_user",
            backend=backend
        )
        print("Initialized In-Memory Memory Store")
    except Exception as e:
        print(f"Warning: Memory init failed ({e}), using None")
        memory = None

    orchestrator = WeavingOrchestrator(config, memory=memory)
    materializer = Materializer(root_dir=demo_output_dir)

    # 2. Define Query
    # We explicitly ask to use the software-engineer skill format if the LLM isn't smart enough to pick it
    # But hopefully the skill system handles it.
    # For this demo, we'll try a direct prompt that encourages the format.
    query_text = (
        "Write a Python script called 'fib.py' that calculates the first 10 Fibonacci numbers. "
        "Also write a 'README.md' explaining how to run it. "
        "IMPORTANT: Use the '# filename: ...' code block format for both files so I can save them."
    )

    print(f"\nSending Query: {query_text}")
    print("Weaving spacetime fabric (this involves the LLM)...")

    try:
        # 3. Weave
        spacetime = await orchestrator.weave(Query(text=query_text))

        print("\n--- Spacetime Woven ---")
        print(f"Response:\n{spacetime.response[:500]}...") # Print first 500 chars
        print(f"\nArtifacts Detected: {len(spacetime.artifacts)}")

        for art in spacetime.artifacts:
            print(f" - {art.name} ({art.type.value})")

        # 4. Materialize
        if spacetime.artifacts:
            print(f"\nMaterializing to: {demo_output_dir}")
            results = materializer.materialize(spacetime)

            for type_, paths in results.items():
                if paths:
                    print(f"Created {type_} files:")
                    for p in paths:
                        print(f"  -> {p}")
        else:
            print("\nNo artifacts found to materialize. (Did the LLM follow the format?)")

    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
