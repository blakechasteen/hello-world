#!/usr/bin/env python3
"""Quick check of HoloLoom memory store."""
import asyncio
import sys
import io
from pathlib import Path
import importlib.util

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent))

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

async def main():
    print("Checking HoloLoom hybrid memory store...\n")

    # Load store
    hybrid_module = load_module("hybrid", "HoloLoom/memory/stores/hybrid_neo4j_qdrant.py")
    store = hybrid_module.HybridNeo4jQdrant(
        neo4j_uri="bolt://localhost:7687",
        neo4j_password="hololoom123",
        qdrant_url="http://localhost:6333"
    )

    # Health check
    health = await store.health_check()

    print(f"Status: {health['status']}")
    print(f"Neo4j memories: {health['neo4j']['memories']}")
    print(f"Qdrant memories: {health['qdrant']['memories']}")

    store.close()

if __name__ == '__main__':
    asyncio.run(main())
