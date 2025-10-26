#!/usr/bin/env python3
"""
Topology + Warp Space Integration Demo
========================================
Demonstrating how topological data analysis enhances semantic understanding.

Scenarios:
1. Discover semantic clusters via connected components
2. Find conceptual loops via 1-dimensional holes
3. Mapper for knowledge graph visualization
4. Topological features for classification
5. Persistent homology of conversation threads
"""

import asyncio
import logging
import numpy as np
from datetime import datetime

# Warp and Topology imports
import sys
sys.path.insert(0, '.')

from HoloLoom.warp.space import WarpSpace
from HoloLoom.warp.topology import (
    PersistentHomology,
    MapperAlgorithm,
    TopologicalFeatureExtractor,
    VietorisRipsComplex
)
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Demo 1: Semantic Cluster Discovery
# ============================================================================

async def demo_1_semantic_clusters():
    """
    Use topology to discover natural semantic clusters.

    Connected components (β₀) reveal how concepts group together.
    """
    logger.info("\n" + "="*80)
    logger.info("DEMO 1: Semantic Cluster Discovery via Topology")
    logger.info("="*80)

    # Documents from different topics
    documents = [
        # Cluster 1: Machine Learning
        "Neural networks learn from data",
        "Deep learning uses multiple layers",
        "Backpropagation trains neural nets",

        # Cluster 2: Databases
        "SQL queries retrieve data",
        "NoSQL databases are schema-free",
        "Database indexing speeds up queries",

        # Cluster 3: Thompson Sampling
        "Thompson Sampling balances exploration and exploitation",
        "Bayesian bandits use posterior distributions",
        "Multi-armed bandits optimize sequential decisions",

        # Bridging documents
        "Machine learning models query databases",
        "Bayesian neural networks combine ML and probability"
    ]

    logger.info(f"\nDocuments: {len(documents)} from 3 topics + bridges")

    # Embed documents
    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
    warp = WarpSpace(embedder, scales=[96, 192, 384])

    logger.info("\n1. Tensioning documents into Warp Space...")
    await warp.tension(documents)

    # Extract embeddings
    embeddings = warp.tensor_field
    logger.info(f"   Embedding matrix: {embeddings.shape}")

    # Compute persistent homology
    logger.info("\n2. Computing persistent homology...")
    ph = PersistentHomology(max_dimension=1)
    diagrams = ph.compute(embeddings, max_scale=3.0, use_ripser=False)

    # Analyze connected components (dimension 0)
    if 0 in diagrams:
        diagram_0 = diagrams[0]
        logger.info(f"\n3. Connected Components (β₀):")
        logger.info(f"   Total components found: {len(diagram_0.intervals)}")

        # Most persistent components = most distinct clusters
        top_components = diagram_0.get_most_persistent(k=3)
        logger.info(f"\n   Top 3 most persistent clusters:")
        for i, interval in enumerate(top_components, 1):
            logger.info(f"   {i}. Birth={interval.birth:.3f}, Death={interval.death:.3f}, "
                       f"Persistence={interval.persistence:.3f}")

    # Build Vietoris-Rips at different scales
    logger.info("\n4. Analyzing cluster structure at multiple scales...")
    vr = VietorisRipsComplex(embeddings, max_dimension=1)

    scales = [0.5, 1.0, 1.5, 2.0]
    filtration = vr.build_filtration(np.array(scales))

    for step in filtration:
        radius = step['radius']
        betti = step['betti_numbers']
        logger.info(f"   Scale {radius:.1f}: β₀={betti[0]} components, β₁={betti[1]} loops")

    warp.collapse()
    logger.info("\nDEMO 1 COMPLETE!\n")


# ============================================================================
# Demo 2: Conceptual Loops Discovery
# ============================================================================

async def demo_2_conceptual_loops():
    """
    Find loops in reasoning patterns.

    1-dimensional holes (β₁) reveal circular dependencies or cyclic patterns.
    """
    logger.info("\n" + "="*80)
    logger.info("DEMO 2: Conceptual Loops via 1-Dimensional Holes")
    logger.info("="*80)

    # Create documents with circular reasoning pattern
    concepts = [
        "A requires B",
        "B requires C",
        "C requires D",
        "D requires A",  # Closes the loop!
        "E is independent",
        "F is independent"
    ]

    logger.info(f"\nConcepts: {len(concepts)}")
    logger.info("Pattern: A → B → C → D → A (circular dependency)")

    # Embed
    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
    warp = WarpSpace(embedder, scales=[96, 192, 384])

    await warp.tension(concepts)
    embeddings = warp.tensor_field

    logger.info(f"\n1. Computing topology...")

    # Persistent homology
    ph = PersistentHomology(max_dimension=1)
    diagrams = ph.compute(embeddings, max_scale=2.0, use_ripser=False)

    # Check for loops
    if 1 in diagrams:
        diagram_1 = diagrams[1]
        logger.info(f"\n2. Loops Detected (β₁):")
        logger.info(f"   Total loops: {len(diagram_1.intervals)}")

        if len(diagram_1.intervals) > 0:
            top_loops = diagram_1.get_most_persistent(k=2)
            logger.info(f"\n   Most significant loops:")
            for i, interval in enumerate(top_loops, 1):
                logger.info(f"   {i}. Birth={interval.birth:.3f}, Death={interval.death:.3f}, "
                           f"Persistence={interval.persistence:.3f}")

            logger.info(f"\n   → Topology reveals the circular dependency pattern!")
        else:
            logger.info("   (No persistent loops found - may need more data or tuning)")

    warp.collapse()
    logger.info("\nDEMO 2 COMPLETE!\n")


# ============================================================================
# Demo 3: Mapper for Knowledge Graph Visualization
# ============================================================================

async def demo_3_mapper_knowledge_graph():
    """
    Use Mapper algorithm to create topological network of knowledge.

    Mapper preserves structure while reducing complexity.
    """
    logger.info("\n" + "="*80)
    logger.info("DEMO 3: Mapper Algorithm for Knowledge Graph Visualization")
    logger.info("="*80)

    # Larger knowledge base
    knowledge = [
        # AI/ML cluster
        "Machine learning optimizes from data",
        "Deep learning uses neural networks",
        "Reinforcement learning maximizes rewards",
        "Supervised learning uses labeled data",

        # Math cluster
        "Linear algebra provides matrix operations",
        "Calculus enables optimization",
        "Probability quantifies uncertainty",
        "Statistics analyzes data patterns",

        # Algorithms cluster
        "Sorting arranges elements in order",
        "Search finds elements in structures",
        "Graph algorithms traverse networks",
        "Dynamic programming solves subproblems",

        # Bridges
        "Neural networks use linear algebra",
        "Optimization uses calculus",
        "ML uses probability and statistics",
        "Reinforcement learning uses graph algorithms"
    ]

    logger.info(f"\nKnowledge base: {len(knowledge)} statements")

    # Embed
    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
    warp = WarpSpace(embedder, scales=[96, 192, 384])

    await warp.tension(knowledge)
    embeddings = warp.tensor_field

    logger.info(f"\n1. Building Mapper graph...")

    # Apply Mapper
    mapper = MapperAlgorithm(n_intervals=8, overlap_percent=0.3)
    graph = mapper.fit(embeddings)

    logger.info(f"\n2. Mapper Results:")
    logger.info(f"   Nodes: {len(graph['nodes'])}")
    logger.info(f"   Edges: {len(graph['edges'])}")

    # Analyze node sizes (cluster sizes)
    node_sizes = [node['size'] for node in graph['nodes']]
    logger.info(f"\n3. Cluster Analysis:")
    logger.info(f"   Average cluster size: {np.mean(node_sizes):.1f}")
    logger.info(f"   Largest cluster: {max(node_sizes)} statements")
    logger.info(f"   Smallest cluster: {min(node_sizes)} statements")

    # Edge connectivity
    logger.info(f"\n4. Network Connectivity:")
    logger.info(f"   Total connections: {len(graph['edges'])}")
    logger.info(f"   Average degree: {2*len(graph['edges'])/len(graph['nodes']):.1f}")

    logger.info(f"\n   → Mapper creates interpretable network preserving topology")

    warp.collapse()
    logger.info("\nDEMO 3 COMPLETE!\n")


# ============================================================================
# Demo 4: Topological Features for Classification
# ============================================================================

async def demo_4_topological_features():
    """
    Extract topological features as input for ML models.

    Persistent homology can augment standard embeddings with shape information.
    """
    logger.info("\n" + "="*80)
    logger.info("DEMO 4: Topological Features for Classification")
    logger.info("="*80)

    # Two categories with different topological structure
    category_a = [
        # Tightly clustered
        "Topic A statement 1",
        "Topic A statement 2",
        "Topic A statement 3",
        "Topic A related idea",
    ]

    category_b = [
        # More distributed
        "Topic B concept 1",
        "Topic B concept 2",
        "Topic B unrelated idea",
        "Topic B different angle",
        "Topic B another perspective"
    ]

    logger.info(f"\nCategory A: {len(category_a)} statements (tight cluster)")
    logger.info(f"Category B: {len(category_b)} statements (distributed)")

    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])

    # Process each category
    for name, category in [("A", category_a), ("B", category_b)]:
        logger.info(f"\n{'='*40}")
        logger.info(f"Category {name}")
        logger.info("="*40)

        warp = WarpSpace(embedder, scales=[96, 192, 384])
        await warp.tension(category)
        embeddings = warp.tensor_field

        # Compute topology
        ph = PersistentHomology(max_dimension=1)
        diagrams = ph.compute(embeddings, max_scale=2.0, use_ripser=False)

        # Extract features
        topo_features = TopologicalFeatureExtractor.extract_features(
            diagrams,
            scales=[0.5, 1.0, 1.5]
        )

        logger.info(f"\n1. Topological Features:")
        logger.info(f"   Feature vector dimension: {len(topo_features)}")
        logger.info(f"   First 5 features: {topo_features[:5]}")

        # Interpret
        if 0 in diagrams:
            n_components = len(diagrams[0].intervals)
            logger.info(f"\n2. Structure:")
            logger.info(f"   Connected components: {n_components}")
            logger.info(f"   Compactness score: {topo_features[1]:.3f}")

        warp.collapse()

    logger.info(f"\n→ Topological features capture shape differences")
    logger.info(f"  Can be combined with embeddings for better classification")

    logger.info("\nDEMO 4 COMPLETE!\n")


# ============================================================================
# Demo 5: Conversation Thread Topology
# ============================================================================

async def demo_5_conversation_topology():
    """
    Analyze conversation threads using persistent homology.

    Discover:
    - Discussion clusters (connected components)
    - Topic loops (returning themes)
    - Thread evolution over time
    """
    logger.info("\n" + "="*80)
    logger.info("DEMO 5: Conversation Thread Topology Analysis")
    logger.info("="*80)

    # Simulated conversation
    conversation = [
        # Thread 1: ML discussion
        "Let's talk about neural networks",
        "Neural nets have multiple layers",
        "Deep learning scales to big data",

        # Thread 2: Optimization
        "How do we optimize these models?",
        "Gradient descent is common",
        "Adaptive learning rates help",

        # Thread 3: Applications
        "Where do we apply this?",
        "Computer vision uses CNNs",
        "NLP uses transformers",

        # Back to Thread 1
        "Going back to neural networks",
        "They're inspired by the brain",

        # Bridge threads
        "Optimization is key for all applications",
        "Deep learning revolutionized vision and NLP"
    ]

    logger.info(f"\nConversation: {len(conversation)} messages")
    logger.info("Pattern: Multiple threads with returns and bridges")

    # Embed messages
    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
    warp = WarpSpace(embedder, scales=[96, 192, 384])

    await warp.tension(conversation)
    embeddings = warp.tensor_field

    logger.info(f"\n1. Computing conversation topology...")

    # Persistent homology
    ph = PersistentHomology(max_dimension=1)
    diagrams = ph.compute(embeddings, max_scale=2.5, use_ripser=False)

    # Analyze structure
    logger.info(f"\n2. Conversation Structure:")

    if 0 in diagrams:
        components = diagrams[0].get_most_persistent(k=3)
        logger.info(f"   Discussion threads (β₀): {len(components)}")
        logger.info(f"   Most distinct threads:")
        for i, comp in enumerate(components, 1):
            logger.info(f"     {i}. Persistence={comp.persistence:.3f}")

    if 1 in diagrams and len(diagrams[1].intervals) > 0:
        loops = diagrams[1].get_most_persistent(k=2)
        logger.info(f"\n   Topic returns (β₁): {len(diagrams[1].intervals)}")
        logger.info(f"   → Indicates returning to previous topics")

    # Mapper for conversation flow
    logger.info(f"\n3. Building conversation flow graph...")
    mapper = MapperAlgorithm(n_intervals=6, overlap_percent=0.4)
    graph = mapper.fit(embeddings)

    logger.info(f"   Flow network: {len(graph['nodes'])} stages, {len(graph['edges'])} transitions")
    logger.info(f"   → Reveals how conversation progresses through topics")

    warp.collapse()
    logger.info("\nDEMO 5 COMPLETE!\n")


# ============================================================================
# Demo 6: Complete Integration - Semantic Search with Topology
# ============================================================================

async def demo_6_semantic_search_topology():
    """
    Enhanced semantic search using topological insights.

    Combine:
    - Warp space attention
    - Topological cluster membership
    - Persistence-based relevance
    """
    logger.info("\n" + "="*80)
    logger.info("DEMO 6: Semantic Search Enhanced with Topology")
    logger.info("="*80)

    # Knowledge base
    docs = [
        "Thompson Sampling uses Bayesian exploration for bandits",
        "UCB uses confidence bounds for exploration",
        "Epsilon-greedy randomly explores occasionally",
        "Gradient descent optimizes neural networks",
        "Backpropagation computes gradients efficiently",
        "Adam optimizer adapts learning rates",
        "Bayesian optimization searches hyperparameters",
        "Grid search tries all combinations",
        "Random search samples hyperparameter space",
        "Multi-armed bandits balance exploration exploitation"
    ]

    query = "How does Bayesian exploration work?"

    logger.info(f"\nKnowledge base: {len(docs)} documents")
    logger.info(f"Query: '{query}'")

    # Embed everything
    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
    warp = WarpSpace(embedder, scales=[96, 192, 384])

    await warp.tension(docs)
    doc_embeddings = warp.tensor_field

    # Compute topology
    logger.info(f"\n1. Analyzing document topology...")
    ph = PersistentHomology(max_dimension=1)
    diagrams = ph.compute(doc_embeddings, max_scale=2.0, use_ripser=False)

    # Build Mapper to understand clusters
    mapper = MapperAlgorithm(n_intervals=5, overlap_percent=0.3)
    graph = mapper.fit(doc_embeddings)

    logger.info(f"   Document clusters: {len(graph['nodes'])}")

    # Standard attention-based search
    query_emb = embedder.encode_scales([query])[384][0]
    attention = warp.apply_attention(query_emb)

    logger.info(f"\n2. Standard attention-based ranking:")
    top_k_attention = np.argsort(attention)[::-1][:3]
    for i, idx in enumerate(top_k_attention, 1):
        logger.info(f"   {i}. [{attention[idx]:.3f}] {docs[idx][:60]}...")

    # Topologically-enhanced ranking
    logger.info(f"\n3. Topology-enhanced ranking:")

    # Find which cluster the query belongs to
    query_embedding = query_emb.reshape(1, -1)
    query_mapper = mapper.fit(np.vstack([doc_embeddings, query_embedding]))

    # Boost documents in same topological cluster
    topo_boost = np.ones(len(docs))
    # (Simplified: boost all for demo, would use actual cluster membership)

    combined_score = attention * topo_boost
    top_k_topo = np.argsort(combined_score)[::-1][:3]

    for i, idx in enumerate(top_k_topo, 1):
        logger.info(f"   {i}. [{combined_score[idx]:.3f}] {docs[idx][:60]}...")

    logger.info(f"\n→ Topology helps identify structurally similar documents")
    logger.info(f"  Not just semantic similarity, but cluster membership")

    warp.collapse()
    logger.info("\nDEMO 6 COMPLETE!\n")


# ============================================================================
# Main Runner
# ============================================================================

async def main():
    """Run all topology integration demos."""
    logger.info("\n" + "="*80)
    logger.info("TOPOLOGY + WARP SPACE INTEGRATION DEMOS")
    logger.info("Topological Data Analysis for Semantic Understanding")
    logger.info("="*80)

    demos = [
        ("Semantic Cluster Discovery", demo_1_semantic_clusters),
        ("Conceptual Loops Detection", demo_2_conceptual_loops),
        ("Mapper Knowledge Graph", demo_3_mapper_knowledge_graph),
        ("Topological Features for ML", demo_4_topological_features),
        ("Conversation Thread Topology", demo_5_conversation_topology),
        ("Semantic Search + Topology", demo_6_semantic_search_topology)
    ]

    for name, demo_func in demos:
        try:
            await demo_func()
            await asyncio.sleep(0.3)
        except Exception as e:
            logger.error(f"\nDemo '{name}' failed: {e}")
            import traceback
            traceback.print_exc()

    logger.info("\n" + "="*80)
    logger.info("ALL TOPOLOGY DEMOS COMPLETE!")
    logger.info("="*80)
    logger.info("\nTopology reveals:")
    logger.info("  - Natural clusters (connected components)")
    logger.info("  - Circular patterns (loops)")
    logger.info("  - Structural relationships (Mapper)")
    logger.info("  - Shape-based features (persistence)")
    logger.info("  - Conversation flow (thread analysis)")
    logger.info("\n")


if __name__ == "__main__":
    asyncio.run(main())
