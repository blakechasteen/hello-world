"""
Working Example: Free Local Embeddings with Qdrant
===================================================

This uses sentence-transformers for FREE local embeddings.
No OpenAI API key needed, real semantic embeddings!

Run: python test_embeddings_free.py
"""

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np
from typing import List, Dict
import uuid

print("="*80)
print("Free Local Embeddings Demo (sentence-transformers + Qdrant)")
print("="*80 + "\n")

# Step 1: Initialize embedding model (free, local)
print("Step 1: Loading embedding model...")
print("  Model: all-MiniLM-L6-v2 (384 dimensions)")
print("  (This will download ~90MB on first run)")

model = SentenceTransformer('all-MiniLM-L6-v2')
print("[OK] Model loaded\n")

# Step 2: Connect to Qdrant
print("Step 2: Connecting to Qdrant...")
client = QdrantClient(host="localhost", port=6333)

collection_name = "beekeeping_free"
vector_size = 384  # all-MiniLM-L6-v2 produces 384-dim vectors

# Create collection if it doesn't exist
try:
    client.get_collection(collection_name)
    print(f"[OK] Collection '{collection_name}' exists")
except:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )
    print(f"[OK] Created collection '{collection_name}'")

print()

# Step 3: Store some beekeeping memories
print("Step 3: Storing beekeeping memories...")

memories = [
    {"user": "blake", "text": "Blake is a beekeeper"},
    {"user": "blake", "text": "Blake has three hives named Jodi, Aurora, and Luna"},
    {"user": "blake", "text": "Jodi is the strongest hive with 8 frames of brood"},
    {"user": "blake", "text": "Blake prefers organic treatments for varroa mites"},
    {"user": "blake", "text": "Blake uses formic acid for mite treatment"},
    {"user": "blake", "text": "Aurora is a newer hive from a spring split"},
]

points = []
for i, mem in enumerate(memories):
    # Generate embedding (free, local!)
    embedding = model.encode(mem["text"])

    # Create point for Qdrant
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=embedding.tolist(),
        payload={
            "text": mem["text"],
            "user": mem["user"]
        }
    )
    points.append(point)
    print(f"  [OK] Embedded: {mem['text'][:50]}...")

# Upload to Qdrant
client.upsert(collection_name=collection_name, points=points)
print(f"\n[OK] Stored {len(points)} memories in Qdrant\n")

# Step 4: Search with semantic similarity
print("Step 4: Testing semantic search...")

queries = [
    "What are Blake's hive names?",
    "Which hive is strongest?",
    "What treatments does Blake use?",
    "Tell me about organic beekeeping",
]

for query in queries:
    print(f"\nQuery: '{query}'")

    # Embed the query (free, local!)
    query_vector = model.encode(query)

    # Search Qdrant
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector.tolist(),
        limit=3,
        query_filter={"must": [{"key": "user", "match": {"value": "blake"}}]}
    )

    print("  Results:")
    for hit in results.points:
        score = hit.score
        text = hit.payload.get("text", "N/A")
        print(f"    [{score:.3f}] {text}")

print("\n" + "="*80)
print("Success! You now have FREE semantic search with embeddings!")
print("="*80)

print("\nWhat you got:")
print("  ✓ Real embeddings (384-dimensional vectors)")
print("  ✓ Semantic search (understands meaning, not just keywords)")
print("  ✓ Stored in Qdrant (persistent, fast)")
print("  ✓ 100% FREE (no API costs)")
print("  ✓ 100% LOCAL (private)")
print()

print("Components:")
print("  - Embeddings: sentence-transformers (HuggingFace)")
print("  - Vector DB: Qdrant")
print("  - Model: all-MiniLM-L6-v2 (90MB, runs on CPU)")
print()

print("View in Qdrant dashboard:")
print("  http://localhost:6333/dashboard")
print()

print("Performance:")
print("  - Embedding speed: ~100-1000 texts/second (CPU)")
print("  - Search speed: Milliseconds")
print("  - Memory usage: ~200MB")
print()

print("Next steps:")
print("  1. Integrate into your workflows")
print("  2. Try larger models for better quality:")
print("     - all-mpnet-base-v2 (768 dim, better quality)")
print("     - multi-qa-MiniLM-L6-cos-v1 (384 dim, optimized for Q&A)")
print("  3. Add your own data")