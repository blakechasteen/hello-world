"""
Multi-Hop Reasoning Demo
========================

Visual demonstration of multi-hop graph traversal for complex queries.

This demo shows:
1. Simple 2-hop query: "How does X relate to Y?"
2. Complex 3-hop query: "What connects A to B through C?"
3. Multiple reasoning paths for same query
4. Explanation of each reasoning step
5. Comparison: direct retrieval vs multi-hop reasoning
6. Performance metrics (paths explored, latency)
"""

import asyncio
import sys
from pathlib import Path

# Add HoloLoom to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hololoom.memory.graph import KG, KGEdge
from hololoom.rag.multihop_reasoning import MultiHopRAGMixin
from hololoom.config import Config


# ============================================================================
# Mock RAG for Demo
# ============================================================================

class MockHoloLoom:
    """Mock HoloLoom with knowledge graph."""
    def __init__(self, kg):
        self.awareness_graph = type('obj', (object,), {'kg': kg})()

    async def recall(self, query, limit=5):
        """Mock recall."""
        from hololoom.memory.protocol import Memory
        from datetime import datetime

        return [
            Memory(
                id='mem_1',
                text=f"Retrieved memory about: {query}",
                timestamp=datetime.now(),
                context={},
                metadata={}
            )
        ]


class DemoRAG(MultiHopRAGMixin):
    """Demo RAG with multi-hop reasoning."""
    def __init__(self, kg, *args, **kwargs):
        self.loom = MockHoloLoom(kg)
        self.orchestrator = None
        super().__init__(*args, **kwargs)

    async def query(self, question, mode="verify"):
        """Mock standard query."""
        from hololoom.rag.simple_rag import RAGResult
        return RAGResult(
            response=f"Standard RAG answer: {question}",
            sources=["source1"],
            confidence=0.7,
            reasoning_mode=mode
        )


# ============================================================================
# Demo Functions
# ============================================================================

def build_demo_kg() -> KG:
    """
    Build demonstration knowledge graph.

    Graph structure:
        Neural Networks Domain:
        - attention → transformer (USES)
        - transformer → neural_network (IS_A)
        - BERT → transformer (IS_A)
        - GPT → transformer (IS_A)
        - encoder → transformer (PART_OF)
        - decoder → transformer (PART_OF)
        - multi-head attention → attention (IS_A)
        - self-attention → attention (IS_A)

        Machine Learning Domain:
        - supervised_learning → machine_learning (IS_A)
        - unsupervised_learning → machine_learning (IS_A)
        - neural_network → machine_learning (USES)
        - deep_learning → neural_network (IS_A)
    """
    kg = KG()

    # Neural networks domain
    neural_edges = [
        KGEdge("attention", "transformer", "USES", 1.0),
        KGEdge("transformer", "neural_network", "IS_A", 1.0),
        KGEdge("BERT", "transformer", "IS_A", 1.0),
        KGEdge("GPT", "transformer", "IS_A", 1.0),
        KGEdge("encoder", "transformer", "PART_OF", 0.8),
        KGEdge("decoder", "transformer", "PART_OF", 0.8),
        KGEdge("multi-head attention", "attention", "IS_A", 0.9),
        KGEdge("self-attention", "attention", "IS_A", 0.9),
    ]

    # Machine learning domain
    ml_edges = [
        KGEdge("supervised_learning", "machine_learning", "IS_A", 1.0),
        KGEdge("unsupervised_learning", "machine_learning", "IS_A", 1.0),
        KGEdge("neural_network", "machine_learning", "USES", 0.9),
        KGEdge("deep_learning", "neural_network", "IS_A", 0.95),
    ]

    kg.add_edges(neural_edges + ml_edges)
    return kg


def print_header(title: str):
    """Print formatted header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def print_path(path, index: int):
    """Print formatted reasoning path."""
    print(f"\nPath {index} (confidence: {path.confidence:.3f}, hops: {path.hop_count}):")

    # Visual path (use ASCII arrows for Windows compatibility)
    path_visual = path.entities[0]
    for i, rel in enumerate(path.relationships):
        if i + 1 < len(path.entities):
            path_visual += f" -[{rel}]-> {path.entities[i+1]}"
    print(f"  {path_visual}")

    # Explanation
    if path.explanation:
        print(f"  Explanation: {path.explanation}")

    # Edge weights
    if path.edge_weights:
        weights_str = ", ".join(f"{w:.2f}" for w in path.edge_weights)
        print(f"  Edge weights: [{weights_str}]")


async def demo_simple_2hop():
    """Demo 1: Simple 2-hop query."""
    print_header("Demo 1: Simple 2-Hop Query")
    print("Query: 'How does attention relate to BERT?'\n")

    kg = build_demo_kg()
    rag = DemoRAG(kg, max_hops=2, beam_width=5)

    result = await rag.query_multihop(
        "How does attention relate to BERT?",
        max_hops=2,
        return_paths=True
    )

    print(f"Response: {result.response}\n")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Paths explored: {result.paths_explored}")
    print(f"Latency: {result.metadata.get('latency_ms', 0):.1f}ms")

    print("\nReasoning Paths Found:")
    for i, path in enumerate(result.reasoning_paths[:3], 1):
        print_path(path, i)

    if result.best_path:
        print(f"\n[OK] Best path: {' -> '.join(result.best_path.entities)}")


async def demo_complex_3hop():
    """Demo 2: Complex 3-hop query."""
    print_header("Demo 2: Complex 3-Hop Query")
    print("Query: 'What connects attention to machine_learning?'\n")

    kg = build_demo_kg()
    rag = DemoRAG(kg, max_hops=3, beam_width=10)

    result = await rag.query_multihop(
        "What connects attention to machine_learning?",
        max_hops=3,
        return_paths=True
    )

    print(f"Response: {result.response}\n")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Paths explored: {result.paths_explored}")
    print(f"Latency: {result.metadata.get('latency_ms', 0):.1f}ms")

    print("\nTop 5 Reasoning Paths:")
    for i, path in enumerate(result.reasoning_paths[:5], 1):
        print_path(path, i)


async def demo_multiple_paths():
    """Demo 3: Multiple reasoning paths for same query."""
    print_header("Demo 3: Multiple Reasoning Paths")
    print("Query: 'Tell me about neural_network'\n")

    kg = build_demo_kg()
    rag = DemoRAG(kg, max_hops=2, beam_width=10)

    result = await rag.query_multihop(
        "Tell me about neural_network",
        max_hops=2,
        return_paths=True
    )

    print(f"Found {len(result.reasoning_paths)} different reasoning paths:\n")

    # Group paths by hop count
    paths_by_hop = {}
    for path in result.reasoning_paths:
        hop = path.hop_count
        if hop not in paths_by_hop:
            paths_by_hop[hop] = []
        paths_by_hop[hop].append(path)

    for hop_count in sorted(paths_by_hop.keys()):
        paths = paths_by_hop[hop_count]
        print(f"\n{hop_count}-hop paths ({len(paths)} found):")
        for i, path in enumerate(paths[:3], 1):
            print_path(path, i)


async def demo_comparison():
    """Demo 4: Direct retrieval vs multi-hop reasoning."""
    print_header("Demo 4: Direct Retrieval vs Multi-Hop Reasoning")

    kg = build_demo_kg()
    rag = DemoRAG(kg, max_hops=2, beam_width=5)

    query = "How does BERT use attention?"

    # Standard query (direct retrieval)
    print("A) Standard RAG (direct retrieval):")
    standard_result = await rag.query(query)
    print(f"   Response: {standard_result.response}")
    print(f"   Confidence: {standard_result.confidence:.3f}")
    print(f"   Sources: {len(standard_result.sources)}")

    # Multi-hop query
    print("\nB) Multi-Hop RAG (graph traversal):")
    multihop_result = await rag.query_multihop(query, max_hops=2)
    print(f"   Response: {multihop_result.response}")
    print(f"   Confidence: {multihop_result.confidence:.3f}")
    print(f"   Reasoning paths: {len(multihop_result.reasoning_paths)}")
    print(f"   Paths explored: {multihop_result.paths_explored}")

    if multihop_result.best_path:
        print(f"\n   Best reasoning chain:")
        print(f"   {multihop_result.best_path.explanation}")


async def demo_performance():
    """Demo 5: Performance analysis."""
    print_header("Demo 5: Performance Analysis")

    kg = build_demo_kg()

    print("Testing different configurations:\n")

    configs = [
        (1, 5, "1-hop, beam=5"),
        (2, 5, "2-hop, beam=5"),
        (3, 5, "3-hop, beam=5"),
        (2, 10, "2-hop, beam=10"),
        (2, 20, "2-hop, beam=20"),
    ]

    results = []
    for max_hops, beam_width, label in configs:
        rag = DemoRAG(kg, max_hops=max_hops, beam_width=beam_width)

        result = await rag.query_multihop(
            "What is BERT?",
            max_hops=max_hops,
            beam_width=beam_width
        )

        latency = result.metadata.get('latency_ms', 0)
        results.append((label, result.paths_explored, latency))

        print(f"{label:20s} | Paths: {result.paths_explored:4d} | Latency: {latency:6.1f}ms")

    print("\nObservations:")
    print("- More hops -> more paths explored (exponential growth)")
    print("- Larger beam width -> more paths kept (linear growth)")
    print("- Beam search prevents combinatorial explosion")


async def demo_visualization():
    """Demo 6: Knowledge graph visualization."""
    print_header("Demo 6: Knowledge Graph Visualization")

    kg = build_demo_kg()

    print("Knowledge Graph Statistics:")
    stats = kg.stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\nSample Subgraph (attention -> transformer -> BERT):")
    subgraph = kg.subgraph_for_entities(["attention", "BERT"], expand=True)
    print(f"  Nodes: {list(subgraph.nodes())}")
    print(f"  Edges: {subgraph.number_of_edges()}")

    print("\nRelationship Types:")
    edge_types = set()
    for u, v, data in kg.G.edges(data=True):
        edge_types.add(data.get('type', 'unknown'))
    for edge_type in sorted(edge_types):
        print(f"  - {edge_type}")


# ============================================================================
# Main Demo
# ============================================================================

async def main():
    """Run all demos."""
    print("\n" + "="*80)
    print("  HoloLoom Multi-Hop Reasoning Demo")
    print("="*80)

    try:
        await demo_simple_2hop()
        await demo_complex_3hop()
        await demo_multiple_paths()
        await demo_comparison()
        await demo_performance()
        await demo_visualization()

        print("\n" + "="*80)
        print("  [OK] All demos complete!")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
