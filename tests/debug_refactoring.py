
import sys
import asyncio
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

async def main():
    try:
        print("Importing WeavingOrchestratorRefactored...")
        from HoloLoom.weaving_orchestrator_refactored import WeavingOrchestratorRefactored
        from HoloLoom.config import Config
        from HoloLoom.protocols.types import MemoryShard

        print("Creating dependencies...")
        config = Config.fast()
        shards = [MemoryShard(id="1", text="test", metadata={})]

        print("Initializing Orchestrator...")
        orchestrator = WeavingOrchestratorRefactored(
            cfg=config,
            shards=shards,
            enable_complexity_auto_detect=True
        )
        print("Success!")
        await orchestrator.close()

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
