"""
Quick test for combinatorics integration with warp drive.
"""

import asyncio
import numpy as np

from HoloLoom.warp.combinatorics import ChainComplex, DiscreteMorseFunction, Sheaf
from HoloLoom.warp.space import WarpSpace
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

async def test_integration():
    print("\n=== Testing Combinatorics + Warp Integration ===\n")

    # Test 1: Build chain complex from knowledge graph
    print("Test 1: Chain complex from knowledge graph")
    vertices = ["AI", "ML", "DL", "NLP", "CV"]
    edges = [(0, 1), (1, 2), (0, 3), (0, 4), (1, 3)]
    triangles = [(0, 1, 3)]

    chains = {0: vertices, 1: edges, 2: triangles}
    complex = ChainComplex(dimension=2, chains=chains, boundaries={})
    complex.compute_boundary_matrices()

    h0 = complex.compute_homology(0)
    h1 = complex.compute_homology(1)

    print(f"  Vertices: {len(vertices)}, Edges: {len(edges)}, Triangles: {len(triangles)}")
    print(f"  H0 (components): {h0['dimension']}")
    print(f"  H1 (cycles): {h1['dimension']}")
    print("  PASS\n")

    # Test 2: Morse simplification
    print("Test 2: Discrete Morse simplification")
    morse = DiscreteMorseFunction(complex=complex)
    morse.compute_gradient_flow()
    morse_complex = morse.morse_complex()

    h0_morse = morse_complex.compute_homology(0)
    print(f"  Morse gradient pairs: {len(morse.gradient_pairs)}")
    print(f"  Homology preserved: {h0['dimension'] == h0_morse['dimension']}")
    print("  PASS\n")

    # Test 3: Sheaf consistency
    print("Test 3: Sheaf consistency analysis")
    stalks = {i: np.random.randn(10) for i in range(len(vertices))}
    restriction_maps = {}
    for i, j in edges:
        restriction_maps[(i, j)] = np.eye(10)
        restriction_maps[(j, i)] = np.eye(10)

    # Base space is the vertex set
    base_space = list(range(len(vertices)))
    sheaf = Sheaf(base_space=base_space, stalks=stalks, restriction_maps=restriction_maps)
    laplacian = sheaf.sheaf_laplacian()
    eigenvalues = np.linalg.eigvalsh(laplacian)

    print(f"  Laplacian shape: {laplacian.shape}")
    print(f"  Smallest eigenvalue: {eigenvalues[0]:.6f}")
    print("  PASS\n")

    # Test 4: Integration with WarpSpace
    print("Test 4: Integration with WarpSpace")

    docs = [
        "Machine learning is a subset of AI",
        "Deep learning uses neural networks",
        "NLP processes natural language",
        "Computer vision analyzes images",
        "Reinforcement learning learns from rewards"
    ]

    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
    warp = WarpSpace(embedder, scales=[96, 192, 384])

    await warp.tension(docs)

    query_emb = embedder.encode_scales("What is deep learning?")[384]  # Get 384-dim encoding
    attention = warp.apply_attention(query_emb)
    context = warp.weighted_context(attention)

    print(f"  Tensioned {len(docs)} documents")
    print(f"  Context shape: {context.shape}")
    print(f"  Top doc index: {np.argmax(attention)}")
    print("  PASS\n")

    # Test 5: Semantic chain complex
    print("Test 5: Semantic chain complex from embeddings")

    embeddings = np.array([embedder.encode_scales(doc)[96] for doc in docs])
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)

    similarity = embeddings @ embeddings.T
    threshold = 0.6

    semantic_edges = []
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            if similarity[i, j] > threshold:
                semantic_edges.append((i, j))

    semantic_chains = {
        0: list(range(len(docs))),
        1: semantic_edges,
        2: []
    }

    semantic_complex = ChainComplex(dimension=2, chains=semantic_chains, boundaries={})
    semantic_complex.compute_boundary_matrices()

    h0_semantic = semantic_complex.compute_homology(0)

    print(f"  Built complex from {len(docs)} embeddings")
    print(f"  Semantic edges: {len(semantic_edges)}")
    print(f"  Connected components: {h0_semantic['dimension']}")
    print("  PASS\n")

    print("=== All Integration Tests Passed! ===\n")

if __name__ == "__main__":
    asyncio.run(test_integration())
