"""
Debug semantic positions in HoloLoom memory system.

Tests whether the 228D semantic projections actually capture content meaning.
"""
import asyncio
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hololoom.hololoom import hololoom  # Memory-focused API


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute Euclidean distance between two vectors."""
    return float(np.linalg.norm(v1 - v2))


async def main():
    print("\n" + "="*70)
    print("  SEMANTIC POSITION DEBUG - Are positions meaningful?")
    print("="*70)

    # Test memories (ordered by relevance to "Bayesian bandit")
    # 1. Highly relevant
    # 2. Moderately relevant
    # 3. Irrelevant
    memories = [
        ("thompson", "Thompson Sampling is a Bayesian algorithm for multi-armed bandits"),
        ("ucb", "UCB provides an upper confidence bound for exploration"),
        ("weather", "The weather is sunny today"),
    ]

    query = "Bayesian bandit algorithms"

    print(f"\nQuery: \"{query}\"")
    print("\nMemories (expected relevance order):")
    for i, (name, text) in enumerate(memories, 1):
        print(f"  {i}. {name}: \"{text[:50]}...\"")

    # Create HoloLoom instance
    print("\n" + "-"*70)
    print("Creating HoloLoom instance...")

    async with HoloLoom() as loom:
        # Store memories
        print("\nStoring memories and capturing semantic positions...")
        stored_ids = []
        for name, text in memories:
            mem = await loom.experience(text)
            stored_ids.append(mem.id)
            print(f"  Stored: {name} -> {mem.id[:20]}...")

        # Get awareness graph to access semantic positions
        awareness = loom._awareness

        print(f"\nSemantic positions stored: {len(awareness.semantic_positions)}")

        # Perceive the query to get its semantic position
        print("\nPerceiving query...")
        perception = await awareness.perceive(query)
        query_pos = perception.position

        print(f"Query position shape: {query_pos.shape}")
        print(f"Query position norm: {np.linalg.norm(query_pos):.4f}")
        print(f"Query position first 10 dims: {query_pos[:10].round(3)}")

        # Examine each memory's position
        print("\n" + "-"*70)
        print("SEMANTIC POSITION ANALYSIS:")
        print("-"*70)

        results = []
        for i, (name, text) in enumerate(memories):
            mem_id = stored_ids[i]

            if mem_id in awareness.semantic_positions:
                mem_pos = awareness.semantic_positions[mem_id]

                # Compute metrics
                eucl_dist = euclidean_distance(query_pos, mem_pos)
                cos_sim = cosine_similarity(query_pos, mem_pos)

                results.append((name, eucl_dist, cos_sim))

                print(f"\n{name}:")
                print(f"  Position norm: {np.linalg.norm(mem_pos):.4f}")
                print(f"  First 10 dims: {mem_pos[:10].round(3)}")
                print(f"  Euclidean distance to query: {eucl_dist:.4f}")
                print(f"  Cosine similarity to query:  {cos_sim:.4f}")
            else:
                print(f"\n{name}: NO POSITION FOUND!")

        # Sort by distance (smaller = more similar)
        print("\n" + "-"*70)
        print("RANKING BY EUCLIDEAN DISTANCE (smaller = more relevant):")
        print("-"*70)
        sorted_by_dist = sorted(results, key=lambda x: x[1])
        for rank, (name, dist, sim) in enumerate(sorted_by_dist, 1):
            print(f"  {rank}. {name}: distance={dist:.4f}, cosine={sim:.4f}")

        # Sort by cosine similarity (larger = more similar)
        print("\n" + "-"*70)
        print("RANKING BY COSINE SIMILARITY (larger = more relevant):")
        print("-"*70)
        sorted_by_sim = sorted(results, key=lambda x: x[2], reverse=True)
        for rank, (name, dist, sim) in enumerate(sorted_by_sim, 1):
            print(f"  {rank}. {name}: cosine={sim:.4f}, distance={dist:.4f}")

        # Now test recall()
        print("\n" + "-"*70)
        print("ACTUAL RECALL RESULTS:")
        print("-"*70)
        recalled = await loom.recall(query, limit=10)

        print(f"Recalled {len(recalled)} memories:")
        for i, mem in enumerate(recalled, 1):
            # Match to original name
            matched_name = "?"
            for name, text in memories:
                if text in mem.text or mem.text in text:
                    matched_name = name
                    break
            print(f"  {i}. {matched_name}: \"{mem.text[:50]}...\"")

        # Check if positions are all similar (indicating mock/random data)
        print("\n" + "-"*70)
        print("POSITION VARIANCE CHECK:")
        print("-"*70)
        all_positions = list(awareness.semantic_positions.values())
        if all_positions:
            stacked = np.stack(all_positions)
            variance = np.var(stacked, axis=0)
            mean_variance = np.mean(variance)
            print(f"Mean variance across dimensions: {mean_variance:.6f}")
            print(f"Min variance: {np.min(variance):.6f}")
            print(f"Max variance: {np.max(variance):.6f}")

            # Check pairwise similarities
            print("\nPairwise cosine similarities between memories:")
            for i, (name_i, _) in enumerate(memories):
                for j, (name_j, _) in enumerate(memories):
                    if j > i:
                        pos_i = awareness.semantic_positions.get(stored_ids[i])
                        pos_j = awareness.semantic_positions.get(stored_ids[j])
                        if pos_i is not None and pos_j is not None:
                            sim = cosine_similarity(pos_i, pos_j)
                            print(f"  {name_i} <-> {name_j}: {sim:.4f}")

        print("\n" + "="*70)
        print("DEBUG COMPLETE")
        print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
