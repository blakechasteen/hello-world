#!/usr/bin/env python3
"""
Direct Qdrant Connection Test

Tests Qdrant initialization to diagnose the issue.
"""

import sys
sys.path.insert(0, 'c:/Users/blake/Documents/mythRL')

from hololoom.memory.stores.qdrant_store import QdrantMemoryStore

print("\n" + "="*70)
print("  QDRANT DIRECT CONNECTION TEST")
print("="*70 + "\n")

try:
    print("[1/3] Initializing QdrantMemoryStore...")
    qdrant = QdrantMemoryStore(
        url="http://localhost:6333",
        collection_prefix="test_memories"
    )
    print("✓ QdrantMemoryStore initialized successfully!\n")

    print("[2/3] Collections created:")
    import json
    print(f"  Scales: {qdrant.scales}")
    print(f"  Embedding dim: {qdrant.embedding_dim}\n")

    print("[3/3] Qdrant is working!")
    print("="*70 + "\n")

except Exception as e:
    print(f"✗ Failed to initialize Qdrant: {e}")
    print(f"  Type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    print("="*70 + "\n")
