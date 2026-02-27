import asyncio
import sys
import os

# Ensure we can import hololoom
sys.path.append(os.getcwd())

from hololoom.eggroll.integration import EggrollIntegration

async def main():
    print("🌍 Distributed Evolution Simulation (Mini-Cluster)")
    
    # We use 'local' backend for the demo to run without specific Ray cluster setup
    # But the code path exercised is the same abstract interface.
    num_workers = 4 # Local CPU cores
    
    print(f"Creating Integration node with {num_workers} workers...")
    
    # We use the TRM model as it is lightweight
    integration = EggrollIntegration(
        num_workers=num_workers,
        model_type="trm", 
        backend_type="local",
        d_model=64, recur_depth=2 # Tiny model for speed
    )
    
    print("\n⚡ Starting Evolution Loop...")
    await integration.run_evolution_loop(num_epochs=5)
    
    print("\n✅ Distributed Simulation Complete.")
    
    # Shutdown backend (clean up processes)
    if hasattr(integration, 'backend'):
        integration.backend.shutdown()

if __name__ == "__main__":
    # Windows SelectorEventLoop fix
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(main())
