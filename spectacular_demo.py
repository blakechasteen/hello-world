#!/usr/bin/env python3
"""
🎭 HOLOLOOM SPECTACULAR LIVE DEMONSTRATION 🎭
============================================

A theatrical demonstration of HoloLoom's knowledge graph and memory system!
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import asyncio
from HoloLoom.memory.graph import KG, KGEdge
from datetime import datetime


def print_banner(text: str, char: str = "="):
    """Print a fancy banner"""
    print()
    print(char * 60)
    print(f"  {text}")
    print(char * 60)
    print()


async def main():
    print_banner("HOLOLOOM KNOWLEDGE GRAPH DEMONSTRATION", "=")

    print("Watch as we build a living knowledge graph in real-time!")
    print("This is the foundation of HoloLoom's memory system...")
    print()

    # Create knowledge graph
    print(">>> Initializing Yarn Graph (Knowledge Graph)...")
    kg = KG()
    print("    Status: ONLINE")
    print()

    await asyncio.sleep(0.5)

    # Add AI/ML concepts
    print_banner("PHASE 1: Building AI Knowledge Network", "-")

    concepts = [
        ("neural_network", "deep_learning", "IS_A", "Neural networks are a type of deep learning"),
        ("transformer", "neural_network", "IS_A", "Transformers are neural networks"),
        ("BERT", "transformer", "IS_A", "BERT is a transformer model"),
        ("GPT", "transformer", "IS_A", "GPT is a transformer model"),
        ("attention", "transformer", "USES", "Transformers use attention mechanisms"),
        ("embedding", "transformer", "USES", "Transformers use embeddings"),
    ]

    for source, target, rel, desc in concepts:
        print(f"  + Adding: {source} --[{rel}]--> {target}")
        kg.add_edge(KGEdge(
            src=source,
            dst=target,
            type=rel,
            weight=1.0,
            metadata={"description": desc, "timestamp": datetime.now().isoformat()}
        ))
        await asyncio.sleep(0.3)

    print()
    stats = kg.stats()
    print(f"  >> Created {stats['num_nodes']} entities")
    print(f"  >> Created {stats['num_edges']} relationships")
    print()

    await asyncio.sleep(0.5)

    # Add HoloLoom-specific concepts
    print_banner("PHASE 2: HoloLoom Architecture", "-")

    hololoom_concepts = [
        ("hololoom", "knowledge_graph", "USES", "HoloLoom uses knowledge graphs"),
        ("knowledge_graph", "neural_network", "USES", "Knowledge graphs integrate with neural networks"),
        ("thompson_sampling", "hololoom", "USES", "HoloLoom uses Thompson Sampling"),
        ("matryoshka", "embedding", "IS_A", "Matryoshka embeddings are multi-scale"),
        ("hololoom", "matryoshka", "USES", "HoloLoom uses Matryoshka embeddings"),
    ]

    for source, target, rel, desc in hololoom_concepts:
        print(f"  + Adding: {source} --[{rel}]--> {target}")
        kg.add_edge(KGEdge(
            src=source,
            dst=target,
            type=rel,
            weight=1.0,
            metadata={"description": desc, "timestamp": datetime.now().isoformat()}
        ))
        await asyncio.sleep(0.3)

    print()
    stats = kg.stats()
    print(f"  >> Total entities: {stats['num_nodes']}")
    print(f"  >> Total relationships: {stats['num_edges']}")
    print()

    await asyncio.sleep(0.5)

    # Query the graph
    print_banner("PHASE 3: Knowledge Retrieval", "-")

    print("Let's query the knowledge graph...")
    print()

    # Find what HoloLoom uses
    print("QUERY: What does HoloLoom use?")
    hololoom_uses = kg.get_related_by_type("hololoom", "USES", direction="out")
    for entity in hololoom_uses:
        print(f"  -> {entity}")
    print()

    await asyncio.sleep(0.5)

    # Find transformer-based models
    print("QUERY: What are types of transformers?")
    transformers = kg.get_related_by_type("transformer", "IS_A", direction="in")
    for entity in transformers:
        print(f"  -> {entity} (a transformer model)")
    print()

    await asyncio.sleep(0.5)

    # Find what transformers use
    print("QUERY: What do transformers use?")
    transformer_uses = kg.get_related_by_type("transformer", "USES", direction="out")
    for entity in transformer_uses:
        print(f"  -> {entity}")
    print()

    await asyncio.sleep(0.5)

    # Statistics
    print_banner("FINAL STATISTICS", "-")
    stats = kg.stats()
    print(f"  Total Entities: {stats['num_nodes']}")
    print(f"  Total Relationships: {stats['num_edges']}")
    print(f"  Relationship Types: IS_A, USES")
    print()
    print("  Knowledge Graph Features:")
    print("    - Multi-hop traversal")
    print("    - Temporal tracking")
    print("    - Metadata storage")
    print("    - Subgraph extraction")
    print("    - Spectral analysis ready")
    print()

    # Grand finale
    print_banner("THE GRAND FINALE", "=")
    print("This knowledge graph is now ready to:")
    print("  1. Power semantic search")
    print("  2. Enable multi-hop reasoning")
    print("  3. Support Thompson Sampling decisions")
    print("  4. Integrate with neural embeddings")
    print("  5. Drive the full HoloLoom weaving cycle")
    print()
    print("And that's just the BEGINNING! This is one of 11 memory systems")
    print("working together in HoloLoom's architecture.")
    print()
    print_banner("THANK YOU!", "=")
    print("Questions? Try asking HoloLoom itself! ;)")
    print()


if __name__ == "__main__":
    asyncio.run(main())
