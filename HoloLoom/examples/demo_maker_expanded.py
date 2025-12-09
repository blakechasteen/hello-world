
import asyncio
import os
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.config import Config
from HoloLoom.protocols.types import Query
from HoloLoom.fabric.materializer import Materializer
from HoloLoom.fabric.spacetime import Spacetime, Artifact, ArtifactType
from HoloLoom.memory.unified import UnifiedMemory
from HoloLoom.memory.stores.in_memory_store import InMemoryStore

async def main():
    print("Initializing HoloLoom Expanded Maker Demo...")
    
    # 1. Setup
    config = Config()
    demo_dir = Path("c:/Users/blake/OneDrive/Documents/mythRL/HoloLoom/demo_expanded")
    
    # Clean up previous run
    import shutil
    if demo_dir.exists():
        shutil.rmtree(demo_dir)
    demo_dir.mkdir(exist_ok=True)
    
    # Initialize Memory
    backend = InMemoryStore()
    memory = UnifiedMemory(user_id="demo_user", backend=backend)
    
    orchestrator = WeavingOrchestrator(config, memory=memory)
    materializer = Materializer(root_dir=demo_dir)
    
    print(f"Target Directory: {demo_dir}")

    # ==========================================
    # Scenario 1: The Architect (Structure)
    # ==========================================
    print("\n[1] Invoking Architect...")
    # Simulating what the Architect skill would return (since we don't have full LLM loop here)
    # In a real run, we would send a Query and get this back.
    
    architect_artifacts = [
        Artifact(
            name="structure.txt",
            type=ArtifactType.DOCUMENT,
            content="src/\n  main.py\n  utils.py\ntests/\n  test_main.py\nREADME.md",
            destination_path="structure.txt"
        ),
        Artifact(
            name=".gitignore",
            type=ArtifactType.DOCUMENT,
            content="__pycache__/\n*.pyc",
            destination_path=".gitignore"
        )
    ]
    
    # Materialize Structure
    print("Materializing architecture...")
    materializer.materialize(Spacetime(
        query_text="Design structure", response="Done", tool_used="architect", 
        confidence=1.0, trace=None, artifacts=architect_artifacts
    ))

    # ==========================================
    # Scenario 2: The Software Engineer (Code)
    # ==========================================
    print("\n[2] Invoking Software Engineer...")
    
    engineer_artifacts = [
        Artifact(
            name="main.py",
            type=ArtifactType.CODE,
            content="# filename: src/main.py\ndef main():\n    print('Hello World')\n\nif __name__ == '__main__':\n    main()",
            destination_path="src/main.py"
        ),
        Artifact(
            name="utils.py",
            type=ArtifactType.CODE,
            content="# filename: src/utils.py\ndef help():\n    return 'helping'",
            destination_path="src/utils.py"
        )
    ]
    
    materializer.materialize(Spacetime(
        query_text="Implement code", response="Done", tool_used="software-engineer", 
        confidence=1.0, trace=None, artifacts=engineer_artifacts
    ))
    
    # ==========================================
    # Scenario 3: Safety Check (Path Traversal)
    # ==========================================
    print("\n[3] Testing Safety Guardrails...")
    
    unsafe_artifact = Artifact(
        name="hack.txt",
        type=ArtifactType.DOCUMENT,
        content="I am rewriting your hosts file!",
        destination_path="../../../../../Windows/System32/drivers/etc/hosts"
    )
    
    print(f"Attempting to write to: {unsafe_artifact.destination_path}")
    results = materializer.materialize(Spacetime(
        query_text="Hacking attempt", response="Trying to hack", tool_used="hacker", 
        confidence=1.0, trace=None, artifacts=[unsafe_artifact]
    ))
    
    # Check if it was written
    if not results['document']:
        print("SUCCESS: Unsafe write was blocked.")
    else:
        print(f"FAILURE: Unsafe write succeeded to {results['document']}")

    # ==========================================
    # Scenario 4: Overwrite Protection
    # ==========================================
    print("\n[4] Testing Overwrite Protection...")
    
    # Try to overwrite main.py without force
    overwrite_artifact = Artifact(
        name="main.py",
        type=ArtifactType.CODE,
        content="Overwritten content!",
        destination_path="src/main.py"
    )
    
    materializer.materialize(Spacetime(
        query_text="Overwrite attempt", response="Overwriting", tool_used="engineer",
        confidence=1.0, trace=None, artifacts=[overwrite_artifact]
    ), force_overwrite=False)
    
    # Verify content didn't change
    with open(demo_dir / "src/main.py", 'r') as f:
        content = f.read()
        if "Hello World" in content:
            print("SUCCESS: Overwrite protected.")
        else:
            print("FAILURE: File was overwritten.")

    # Try WITH force
    print("Retrying with force_overwrite=True...")
    materializer.materialize(Spacetime(
        query_text="Force overwrite", response="Overwriting", tool_used="engineer",
        confidence=1.0, trace=None, artifacts=[overwrite_artifact]
    ), force_overwrite=True)
    
    with open(demo_dir / "src/main.py", 'r') as f:
        content = f.read()
        if "Overwritten content" in content:
            print("SUCCESS: Force overwrite worked.")
        else:
            print("FAILURE: File was NOT overwritten.")

if __name__ == "__main__":
    asyncio.run(main())
