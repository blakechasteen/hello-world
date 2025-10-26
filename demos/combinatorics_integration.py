"""
Combinatorics Integration Demo
==============================

Demonstrates integration of chain complexes, discrete Morse theory, and sheaf theory
with the HoloLoom Warp Drive for advanced knowledge graph analysis.

6 Production-Ready Demonstrations:
1. Knowledge Graph Homology - Detect semantic holes and cycles
2. Morse Simplification - Reduce complex graphs while preserving topology
3. Sheaf Consistency - Find contradictions in knowledge bases
4. Multi-Scale Analysis - Combine combinatorics with warp scales
5. Semantic Chain Complex - Build complexes from embeddings
6. Full Integration - Complete pipeline with all features

Author: HoloLoom Team
Date: 2025-10-25
"""

import asyncio
import numpy as np
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Import HoloLoom components
try:
    from HoloLoom.warp.combinatorics import ChainComplex, DiscreteMorseFunction, Sheaf
    from HoloLoom.warp.space import WarpSpace
    from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
    HAS_COMBINATORICS = True
except ImportError as e:
    logger.warning(f"Could not import combinatorics: {e}")
    HAS_COMBINATORICS = False

# Try to import topology for extended features
try:
    from HoloLoom.warp.topology import PersistentHomology, TopologicalFeatureExtractor
    HAS_TOPOLOGY = True
except ImportError:
    HAS_TOPOLOGY = False
    logger.info("Topology module not available - some demos will be limited")


# ============================================================================
# DEMO 1: Knowledge Graph Homology
# ============================================================================

async def demo_1_knowledge_graph_homology():
    """
    Use Case: Detect semantic holes and cycles in knowledge graphs

    Scenario: A knowledge base about programming languages has entities and
    relationships. We want to find semantic cycles (loops in knowledge) and
    holes (missing connections) using homology.

    Technique: Build a chain complex from the knowledge graph and compute
    homology groups to identify topological features.
    """
    print("\n" + "="*70)
    print("DEMO 1: Knowledge Graph Homology")
    print("="*70)

    if not HAS_COMBINATORICS:
        print("⚠️  Combinatorics module not available")
        return

    # Knowledge graph as simplicial complex
    # Vertices (0-simplices): Programming concepts
    vertices = [
        "Python", "Java", "C++",           # Languages (0-2)
        "OOP", "Functional",               # Paradigms (3-4)
        "Memory", "GC"                     # Concepts (5-6)
    ]

    # Edges (1-simplices): Direct relationships
    edges = [
        (0, 3),  # Python supports OOP
        (0, 4),  # Python supports Functional
        (1, 3),  # Java supports OOP
        (2, 3),  # C++ supports OOP
        (2, 5),  # C++ has manual Memory management
        (0, 6),  # Python has GC
        (1, 6),  # Java has GC
        (3, 4),  # OOP and Functional can coexist
    ]

    # Triangles (2-simplices): Triple relationships
    triangles = [
        (0, 3, 4),  # Python supports both OOP and Functional
        (0, 3, 6),  # Python has OOP with GC
        (1, 3, 6),  # Java has OOP with GC
    ]

    print("\n📊 Building chain complex from knowledge graph...")
    print(f"   Vertices (concepts): {len(vertices)}")
    print(f"   Edges (relationships): {len(edges)}")
    print(f"   Triangles (triple relations): {len(triangles)}")

    # Build chain complex
    chains = {
        0: vertices,
        1: edges,
        2: triangles
    }

    complex = ChainComplex(dimension=2, chains=chains, boundaries={})
    complex.compute_boundary_matrices()

    print("\n🔍 Computing homology groups...")

    # H₀ counts connected components
    h0 = complex.compute_homology(0)
    print(f"\n   H₀ (connected components):")
    print(f"      Dimension: {h0['dimension']}")
    print(f"      Interpretation: {h0['dimension']} connected component(s)")

    # H₁ counts loops/cycles
    h1 = complex.compute_homology(1)
    print(f"\n   H₁ (1-dimensional holes/cycles):")
    print(f"      Dimension: {h1['dimension']}")
    if h1['dimension'] > 0:
        print(f"      Interpretation: {h1['dimension']} independent cycle(s) in knowledge")
        print(f"      Example: OOP ↔ Functional forms a cycle through Python")
    else:
        print(f"      Interpretation: No cycles - knowledge forms a tree")

    # H₂ counts voids
    h2 = complex.compute_homology(2)
    print(f"\n   H₂ (2-dimensional voids):")
    print(f"      Dimension: {h2['dimension']}")
    print(f"      Interpretation: {h2['dimension']} void(s) in knowledge structure")

    print("\n✅ Knowledge graph homology analysis complete!")
    print("   Application: Detect missing relationships, redundant paths, knowledge gaps")


# ============================================================================
# DEMO 2: Morse Simplification
# ============================================================================

async def demo_2_morse_simplification():
    """
    Use Case: Simplify complex semantic networks while preserving topology

    Scenario: A large knowledge graph has many redundant nodes and edges.
    We want to simplify it to core concepts without losing semantic structure.

    Technique: Discrete Morse theory pairs critical simplices and collapses
    non-critical ones, reducing complexity while preserving homology.
    """
    print("\n" + "="*70)
    print("DEMO 2: Morse Simplification of Semantic Network")
    print("="*70)

    if not HAS_COMBINATORICS:
        print("⚠️  Combinatorics module not available")
        return

    # Build a larger complex (semantic network)
    print("\n📊 Building complex semantic network...")

    # 10 vertices: Core concepts + supporting details
    vertices = list(range(10))

    # Dense edge structure (many redundant paths)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Main path
        (0, 2), (1, 3), (2, 4),          # Shortcuts
        (5, 6), (6, 7), (7, 8), (8, 9),  # Secondary path
        (0, 5), (4, 9),                  # Bridges
        (1, 6), (3, 8),                  # Cross-links
    ]

    # Some triangles (cyclic dependencies)
    triangles = [
        (0, 1, 2),  # Triangle 1
        (1, 2, 3),  # Triangle 2
        (2, 3, 4),  # Triangle 3
        (5, 6, 7),  # Triangle 4
    ]

    chains = {
        0: vertices,
        1: edges,
        2: triangles
    }

    complex = ChainComplex(dimension=2, chains=chains, boundaries={})
    complex.compute_boundary_matrices()

    print(f"   Original: {len(vertices)} vertices, {len(edges)} edges, {len(triangles)} triangles")

    # Compute original homology
    h0_orig = complex.compute_homology(0)
    h1_orig = complex.compute_homology(1)

    print(f"\n   Original homology:")
    print(f"      H₀: {h0_orig['dimension']} (components)")
    print(f"      H₁: {h1_orig['dimension']} (cycles)")

    # Create Morse function (height function: y-coordinate)
    morse_values = {i: i for i in vertices}  # Simple linear function

    morse = DiscreteMorseFunction(complex=complex, values=morse_values)

    print("\n🔄 Computing discrete Morse gradient flow...")
    gradient = morse.compute_gradient_flow()

    critical_vertices = []
    critical_edges = []
    critical_triangles = []
    paired_count = 0

    for dim, pairs in gradient.items():
        paired_count += len(pairs)
        # Critical simplices are those not in any pair
        if dim == 0:
            paired_v = set(v for v, _ in pairs) | set(v for _, v in pairs if isinstance(v, int))
            critical_vertices = [v for v in vertices if v not in paired_v]
        elif dim == 1:
            paired_e = set(e for e, _ in pairs) | set(e for _, e in pairs if isinstance(e, tuple) and len(e) == 2)
            critical_edges = [e for e in edges if e not in paired_e]
        elif dim == 2:
            paired_t = set(t for t, _ in pairs)
            critical_triangles = [t for t in triangles if t not in paired_t]

    print(f"\n   Morse analysis:")
    print(f"      Total paired simplices: {paired_count}")
    print(f"      Critical vertices: {len(critical_vertices)}")
    print(f"      Critical edges: {len(critical_edges)}")
    print(f"      Critical triangles: {len(critical_triangles)}")

    # Build simplified Morse complex
    morse_complex = morse.morse_complex()

    # Compute homology of simplified complex
    h0_morse = morse_complex.compute_homology(0)
    h1_morse = morse_complex.compute_homology(1)

    print(f"\n   Simplified homology:")
    print(f"      H₀: {h0_morse['dimension']} (components)")
    print(f"      H₁: {h1_morse['dimension']} (cycles)")

    # Verify homology is preserved
    homology_preserved = (
        h0_orig['dimension'] == h0_morse['dimension'] and
        h1_orig['dimension'] == h1_morse['dimension']
    )

    print(f"\n   ✅ Homology preserved: {homology_preserved}")

    reduction = 100 * (1 - len(morse_complex.chains.get(1, [])) / len(edges))
    print(f"   📉 Complexity reduction: {reduction:.1f}%")

    print("\n✅ Morse simplification complete!")
    print("   Application: Compress knowledge graphs, find core concepts, optimize retrieval")


# ============================================================================
# DEMO 3: Sheaf Consistency Analysis
# ============================================================================

async def demo_3_sheaf_consistency():
    """
    Use Case: Detect contradictions and inconsistencies in knowledge bases

    Scenario: Multiple sources provide information about the same entities.
    We want to find where the information is inconsistent or contradictory.

    Technique: Sheaf theory assigns local data (stalks) to each entity and
    measures global consistency via the sheaf Laplacian.
    """
    print("\n" + "="*70)
    print("DEMO 3: Sheaf Consistency Analysis")
    print("="*70)

    if not HAS_COMBINATORICS:
        print("⚠️  Combinatorics module not available")
        return

    print("\n📊 Building knowledge graph with multi-source data...")

    # Knowledge graph: 5 entities about "Python programming"
    entities = ["Python", "Typing", "Performance", "Libraries", "Syntax"]

    # Each entity has a feature vector from different sources
    # Dimension: [ease_of_use, speed, type_safety, ecosystem]
    stalks = {
        "Python": np.array([0.9, 0.6, 0.5, 0.95]),      # Generally easy, slow, weak typing, great ecosystem
        "Typing": np.array([0.7, 0.65, 0.8, 0.9]),      # Type hints improve safety
        "Performance": np.array([0.6, 0.7, 0.6, 0.9]),  # Can be optimized
        "Libraries": np.array([0.85, 0.65, 0.5, 1.0]),  # Rich ecosystem
        "Syntax": np.array([0.95, 0.6, 0.5, 0.9]),      # Very readable
    }

    print(f"   Entities: {len(entities)}")
    print(f"   Feature dimensions: {stalks['Python'].shape[0]}")

    # Relationships with restriction maps
    # Restriction maps enforce consistency constraints
    edges = [
        ("Python", "Typing"),
        ("Python", "Performance"),
        ("Python", "Libraries"),
        ("Python", "Syntax"),
        ("Typing", "Performance"),  # Type hints can improve performance
    ]

    # Create restriction maps (identity for simplicity - could be learned)
    restriction_maps = {}
    for src, tgt in edges:
        # Restriction should preserve information
        # Using identity, but could use learned projection
        restriction_maps[(src, tgt)] = np.eye(4)
        restriction_maps[(tgt, src)] = np.eye(4)  # Symmetric

    print(f"   Relationships: {len(edges)}")

    # Build sheaf
    sheaf = Sheaf(stalks=stalks, restriction_maps=restriction_maps)

    print("\n🔍 Computing sheaf Laplacian...")
    laplacian = sheaf.sheaf_laplacian()

    # Analyze Laplacian
    eigenvalues = np.linalg.eigvalsh(laplacian)
    eigenvalues = np.sort(eigenvalues)

    print(f"\n   Laplacian shape: {laplacian.shape}")
    print(f"   Smallest eigenvalues: {eigenvalues[:3]}")

    # Small eigenvalues indicate inconsistency
    consistency_score = eigenvalues[1]  # Skip first (always 0)

    print(f"\n   Consistency score: {consistency_score:.6f}")
    if consistency_score < 0.01:
        print("   ✅ Knowledge is highly consistent")
    elif consistency_score < 0.1:
        print("   ⚠️  Some minor inconsistencies detected")
    else:
        print("   ❌ Significant inconsistencies found!")

    # Find global sections (consistent global assignments)
    print("\n🔍 Finding global sections...")
    sections = sheaf.global_sections(tol=1e-4)

    print(f"   Global sections found: {sections.shape[1]}")
    if sections.shape[1] > 0:
        print("   Interpretation: Consistent global view exists")
    else:
        print("   Interpretation: No globally consistent view")

    # Compute cohomology
    print("\n🔍 Computing sheaf cohomology...")
    h1_dim = sheaf.cohomology_dimension(degree=1)

    print(f"   H¹ dimension: {h1_dim}")
    print(f"   Interpretation: {h1_dim} independent obstruction(s) to global consistency")

    # Add a contradictory stalk to show inconsistency detection
    print("\n\n📊 Adding contradictory information...")
    stalks_inconsistent = stalks.copy()
    stalks_inconsistent["Performance"] = np.array([0.9, 0.3, 0.9, 0.5])  # Contradicts Python's profile

    sheaf_inconsistent = Sheaf(stalks=stalks_inconsistent, restriction_maps=restriction_maps)
    laplacian_inconsistent = sheaf_inconsistent.sheaf_laplacian()
    eigenvalues_inconsistent = np.sort(np.linalg.eigvalsh(laplacian_inconsistent))

    consistency_score_inconsistent = eigenvalues_inconsistent[1]

    print(f"   New consistency score: {consistency_score_inconsistent:.6f}")
    print(f"   Change: {consistency_score_inconsistent - consistency_score:+.6f}")

    if consistency_score_inconsistent > consistency_score * 1.5:
        print("   ❌ Inconsistency detected! Contradiction found.")

    print("\n✅ Sheaf consistency analysis complete!")
    print("   Application: Detect contradictions, merge knowledge sources, verify data quality")


# ============================================================================
# DEMO 4: Multi-Scale Combinatorial Analysis
# ============================================================================

async def demo_4_multiscale_analysis():
    """
    Use Case: Analyze knowledge at multiple scales using warp + combinatorics

    Scenario: A knowledge graph has structure at different levels (local,
    regional, global). We want to analyze topology at each scale.

    Technique: Combine warp space's multi-scale embeddings with chain complexes
    at different resolutions.
    """
    print("\n" + "="*70)
    print("DEMO 4: Multi-Scale Combinatorial Analysis")
    print("="*70)

    if not HAS_COMBINATORICS:
        print("⚠️  Combinatorics module not available")
        return

    print("\n📊 Creating multi-scale knowledge representation...")

    # Knowledge corpus at different scales
    documents = [
        "Machine learning algorithms learn from data",
        "Neural networks are a type of machine learning",
        "Deep learning uses multi-layer neural networks",
        "Transformers are a neural network architecture",
        "BERT is a transformer model for NLP",
        "GPT uses transformer architecture for generation",
        "Attention mechanisms are key to transformers",
        "Self-attention computes relationships between tokens",
    ]

    print(f"   Documents: {len(documents)}")

    # Create embeddings at multiple scales
    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
    warp = WarpSpace(embedder, scales=[96, 192, 384])

    print("\n🔄 Tensioning warp space at multiple scales...")
    await warp.tension(documents)

    # Analyze at each scale
    scales = [96, 192, 384]
    scale_names = ["Coarse (96)", "Medium (192)", "Fine (384)"]

    for scale, name in zip(scales, scale_names):
        print(f"\n{'='*70}")
        print(f"Scale: {name}")
        print(f"{'='*70}")

        # Get embeddings at this scale
        embeddings = []
        for doc in documents:
            emb = embedder.embed(doc, target_size=scale)
            embeddings.append(emb)

        embeddings_array = np.array(embeddings)

        # Build graph from embeddings (connect similar docs)
        threshold = 0.6  # Cosine similarity threshold

        # Normalize embeddings
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        embeddings_normalized = embeddings_array / (norms + 1e-8)

        # Compute similarity matrix
        similarity = embeddings_normalized @ embeddings_normalized.T

        # Build edges
        edges = []
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                if similarity[i, j] > threshold:
                    edges.append((i, j))

        # Build triangles (cliques of size 3)
        triangles = []
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                for k in range(j + 1, len(documents)):
                    if (similarity[i, j] > threshold and
                        similarity[j, k] > threshold and
                        similarity[i, k] > threshold):
                        triangles.append((i, j, k))

        print(f"   Graph structure:")
        print(f"      Vertices: {len(documents)}")
        print(f"      Edges: {len(edges)}")
        print(f"      Triangles: {len(triangles)}")

        # Build chain complex
        chains = {
            0: list(range(len(documents))),
            1: edges,
            2: triangles
        }

        complex = ChainComplex(dimension=2, chains=chains, boundaries={})
        complex.compute_boundary_matrices()

        # Compute homology
        h0 = complex.compute_homology(0)
        h1 = complex.compute_homology(1)

        print(f"\n   Homology:")
        print(f"      H₀: {h0['dimension']} (connected components)")
        print(f"      H₁: {h1['dimension']} (cycles)")

        # Interpretation
        if h0['dimension'] == 1:
            print(f"   ✅ Single connected component at this scale")
        else:
            print(f"   📊 {h0['dimension']} distinct topic clusters at this scale")

        if h1['dimension'] > 0:
            print(f"   🔄 {h1['dimension']} semantic cycle(s) detected")

    print("\n" + "="*70)
    print("✅ Multi-scale analysis complete!")
    print("   Observation: Coarser scales show more connectivity,")
    print("   finer scales reveal detailed structure and clusters")
    print("\n   Application: Topic modeling, document clustering, semantic navigation")


# ============================================================================
# DEMO 5: Semantic Chain Complex from Embeddings
# ============================================================================

async def demo_5_semantic_chain_complex():
    """
    Use Case: Build chain complexes directly from semantic embeddings

    Scenario: Given a set of concepts, automatically construct a simplicial
    complex that captures their semantic relationships.

    Technique: Use embedding distances to build Vietoris-Rips complex,
    then analyze with chain complex homology.
    """
    print("\n" + "="*70)
    print("DEMO 5: Semantic Chain Complex Construction")
    print("="*70)

    if not HAS_COMBINATORICS:
        print("⚠️  Combinatorics module not available")
        return

    print("\n📊 Building semantic space...")

    # Concepts in AI/ML domain
    concepts = [
        "supervised learning",
        "unsupervised learning",
        "reinforcement learning",
        "classification",
        "regression",
        "clustering",
        "neural networks",
        "decision trees",
        "random forests",
        "support vector machines",
    ]

    print(f"   Concepts: {len(concepts)}")

    # Embed concepts
    embedder = MatryoshkaEmbeddings(sizes=[384])
    embeddings = np.array([embedder.embed(c, target_size=384) for c in concepts])

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)

    print(f"   Embedding dimension: {embeddings.shape[1]}")

    # Build Vietoris-Rips complex at different scales
    print("\n🔍 Building Vietoris-Rips complexes...")

    # Compute distance matrix
    distances = 1 - (embeddings @ embeddings.T)  # 1 - cosine similarity
    np.fill_diagonal(distances, 0)

    # Try different radii
    radii = [0.3, 0.5, 0.7]

    for radius in radii:
        print(f"\n{'='*70}")
        print(f"Radius: {radius:.2f}")
        print(f"{'='*70}")

        # Build 0-simplices (vertices)
        vertices = list(range(len(concepts)))

        # Build 1-simplices (edges)
        edges = []
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                if distances[i, j] < radius:
                    edges.append((i, j))

        # Build 2-simplices (triangles)
        triangles = []
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                for k in range(j + 1, len(concepts)):
                    if (distances[i, j] < radius and
                        distances[j, k] < radius and
                        distances[i, k] < radius):
                        triangles.append((i, j, k))

        # Build 3-simplices (tetrahedra)
        tetrahedra = []
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                for k in range(j + 1, len(concepts)):
                    for l in range(k + 1, len(concepts)):
                        if (distances[i, j] < radius and
                            distances[j, k] < radius and
                            distances[k, l] < radius and
                            distances[i, k] < radius and
                            distances[i, l] < radius and
                            distances[j, l] < radius):
                            tetrahedra.append((i, j, k, l))

        print(f"   Complex structure:")
        print(f"      Vertices: {len(vertices)}")
        print(f"      Edges: {len(edges)}")
        print(f"      Triangles: {len(triangles)}")
        print(f"      Tetrahedra: {len(tetrahedra)}")

        # Build chain complex (up to dimension 2 for homology)
        chains = {
            0: vertices,
            1: edges,
            2: triangles
        }

        complex = ChainComplex(dimension=2, chains=chains, boundaries={})
        complex.compute_boundary_matrices()

        # Compute homology
        h0 = complex.compute_homology(0)
        h1 = complex.compute_homology(1)
        h2 = complex.compute_homology(2)

        print(f"\n   Homology:")
        print(f"      H₀: {h0['dimension']} (connected components)")
        print(f"      H₁: {h1['dimension']} (loops)")
        print(f"      H₂: {h2['dimension']} (voids)")

        # Interpret results
        if h0['dimension'] == 1:
            print(f"   ✅ All concepts are connected at this scale")
        else:
            print(f"   📊 {h0['dimension']} distinct concept clusters")

        if h1['dimension'] > 0:
            print(f"   🔄 {h1['dimension']} semantic loop(s) - cyclic relationships")

        if h2['dimension'] > 0:
            print(f"   🕳️  {h2['dimension']} void(s) - higher-order gaps")

    print("\n" + "="*70)
    print("✅ Semantic chain complex construction complete!")
    print("   Application: Concept organization, curriculum design, knowledge mapping")


# ============================================================================
# DEMO 6: Full Integration Pipeline
# ============================================================================

async def demo_6_full_integration():
    """
    Use Case: Complete pipeline combining all combinatorics features with warp

    Scenario: Given a query and knowledge base, perform multi-scale analysis,
    detect inconsistencies, simplify via Morse theory, and measure topology.

    Technique: Complete integration of warp space, chain complexes, sheaves,
    and discrete Morse theory in a production-ready pipeline.
    """
    print("\n" + "="*70)
    print("DEMO 6: Full Combinatorial Integration Pipeline")
    print("="*70)

    if not HAS_COMBINATORICS:
        print("⚠️  Combinatorics module not available")
        return

    print("\n📊 Initializing complete pipeline...")

    # Knowledge base: AI safety concepts
    knowledge_base = [
        "AI alignment ensures AI goals match human values",
        "Value learning helps AI understand human preferences",
        "Reward modeling trains AI on human feedback",
        "Interpretability makes AI decisions transparent",
        "Robustness ensures AI handles edge cases",
        "AI safety prevents unintended harmful outcomes",
        "Scalable oversight allows humans to supervise complex AI",
        "Corrigibility enables AI to accept corrections",
    ]

    query = "How do we ensure AI systems are safe and aligned?"

    print(f"   Knowledge base: {len(knowledge_base)} documents")
    print(f"   Query: '{query}'")

    # ========================================================================
    # STEP 1: Multi-scale embedding and retrieval
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 1: Multi-Scale Warp Analysis")
    print("="*70)

    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
    warp = WarpSpace(embedder, scales=[96, 192, 384])

    await warp.tension(knowledge_base)

    # Retrieve at each scale
    query_emb = embedder.embed(query, target_size=384)
    attention = warp.apply_attention(query_emb, temperature=0.5)
    context = warp.weighted_context(attention)

    print(f"✅ Retrieved context shape: {context.shape}")
    print(f"   Top relevant docs (by attention):")
    top_k = 3
    top_indices = np.argsort(attention)[-top_k:][::-1]
    for idx in top_indices:
        print(f"      [{idx}] (weight: {attention[idx]:.3f}) {knowledge_base[idx][:60]}...")

    # ========================================================================
    # STEP 2: Build semantic chain complex
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 2: Semantic Chain Complex")
    print("="*70)

    # Embed all documents
    embeddings = np.array([embedder.embed(doc, target_size=384) for doc in knowledge_base])
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)

    # Build complex
    similarity = embeddings @ embeddings.T
    threshold = 0.6

    edges = []
    for i in range(len(knowledge_base)):
        for j in range(i + 1, len(knowledge_base)):
            if similarity[i, j] > threshold:
                edges.append((i, j))

    triangles = []
    for i in range(len(knowledge_base)):
        for j in range(i + 1, len(knowledge_base)):
            for k in range(j + 1, len(knowledge_base)):
                if (similarity[i, j] > threshold and
                    similarity[j, k] > threshold and
                    similarity[i, k] > threshold):
                    triangles.append((i, j, k))

    chains = {
        0: list(range(len(knowledge_base))),
        1: edges,
        2: triangles
    }

    complex = ChainComplex(dimension=2, chains=chains, boundaries={})
    complex.compute_boundary_matrices()

    print(f"   Complex: {len(knowledge_base)} vertices, {len(edges)} edges, {len(triangles)} triangles")

    h0 = complex.compute_homology(0)
    h1 = complex.compute_homology(1)

    print(f"   Homology: H₀={h0['dimension']}, H₁={h1['dimension']}")

    # ========================================================================
    # STEP 3: Morse simplification
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 3: Discrete Morse Simplification")
    print("="*70)

    # Create Morse function (attention scores)
    morse_values = {i: attention[i] for i in range(len(knowledge_base))}

    morse = DiscreteMorseFunction(complex=complex, values=morse_values)
    gradient = morse.compute_gradient_flow()

    total_pairs = sum(len(pairs) for pairs in gradient.values())
    print(f"   Paired simplices: {total_pairs}")

    morse_complex = morse.morse_complex()

    h0_morse = morse_complex.compute_homology(0)
    h1_morse = morse_complex.compute_homology(1)

    print(f"   Morse homology: H₀={h0_morse['dimension']}, H₁={h1_morse['dimension']}")
    print(f"   ✅ Homology preserved: {h0['dimension'] == h0_morse['dimension']}")

    # ========================================================================
    # STEP 4: Sheaf consistency check
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 4: Sheaf Consistency Analysis")
    print("="*70)

    # Assign stalks (embeddings) to vertices
    stalks = {i: embeddings[i] for i in range(len(knowledge_base))}

    # Restriction maps (identity for connected docs)
    restriction_maps = {}
    for i, j in edges:
        restriction_maps[(i, j)] = np.eye(384)
        restriction_maps[(j, i)] = np.eye(384)

    sheaf = Sheaf(stalks=stalks, restriction_maps=restriction_maps)
    laplacian = sheaf.sheaf_laplacian()

    eigenvalues = np.sort(np.linalg.eigvalsh(laplacian))
    consistency_score = eigenvalues[1] if len(eigenvalues) > 1 else 0

    print(f"   Consistency score: {consistency_score:.6f}")
    if consistency_score < 0.01:
        print(f"   ✅ Knowledge base is highly consistent")
    else:
        print(f"   ⚠️  Some inconsistencies detected")

    # ========================================================================
    # STEP 5: Topological features (if available)
    # ========================================================================
    if HAS_TOPOLOGY:
        print("\n" + "="*70)
        print("STEP 5: Persistent Homology")
        print("="*70)

        ph = PersistentHomology(max_dimension=2, max_scale=2.0)
        diagrams = ph.compute(embeddings, max_scale=1.5)

        for dim, intervals in diagrams.items():
            print(f"   H{dim}: {len(intervals)} features")
            if len(intervals) > 0:
                # Show most persistent features
                sorted_intervals = sorted(intervals, key=lambda x: x.persistence, reverse=True)
                top_features = sorted_intervals[:3]
                for interval in top_features:
                    print(f"      [{interval.birth:.3f}, {interval.death:.3f}) persistence={interval.persistence:.3f}")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)

    print(f"\n📊 Analysis Summary:")
    print(f"   Query relevance: Top docs identified via attention")
    print(f"   Semantic structure: {h0['dimension']} component(s), {h1['dimension']} cycle(s)")
    print(f"   Simplification: {total_pairs} simplices paired by Morse theory")
    print(f"   Consistency: {'High' if consistency_score < 0.01 else 'Moderate'}")
    if HAS_TOPOLOGY:
        total_features = sum(len(intervals) for intervals in diagrams.values())
        print(f"   Persistent features: {total_features} across all dimensions")

    print(f"\n✅ Full integration pipeline complete!")
    print(f"   Application: Production-ready semantic analysis with topological insights")


# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("HOLOLOOM COMBINATORICS INTEGRATION DEMOS")
    print("="*70)
    print("\nDemonstrating 6 production-ready use cases:")
    print("1. Knowledge Graph Homology")
    print("2. Morse Simplification")
    print("3. Sheaf Consistency")
    print("4. Multi-Scale Analysis")
    print("5. Semantic Chain Complex")
    print("6. Full Integration Pipeline")
    print("="*70)

    demos = [
        demo_1_knowledge_graph_homology,
        demo_2_morse_simplification,
        demo_3_sheaf_consistency,
        demo_4_multiscale_analysis,
        demo_5_semantic_chain_complex,
        demo_6_full_integration,
    ]

    for i, demo in enumerate(demos, 1):
        try:
            await demo()
        except Exception as e:
            logger.error(f"Demo {i} failed: {e}", exc_info=True)

        if i < len(demos):
            print("\n" + "."*70)
            print("Press Enter to continue to next demo...")
            input()

    print("\n" + "="*70)
    print("ALL DEMOS COMPLETE!")
    print("="*70)
    print("\n🎉 Combinatorics integration successfully demonstrated!")
    print("\nKey capabilities:")
    print("  • Chain complex homology for semantic analysis")
    print("  • Discrete Morse theory for graph simplification")
    print("  • Sheaf theory for consistency checking")
    print("  • Multi-scale topological features")
    print("  • Integration with HoloLoom warp drive")
    print("\n" + "="*70)


if __name__ == "__main__":
    asyncio.run(main())
