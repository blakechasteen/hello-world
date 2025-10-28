"""
HoloLoom Embedding Module
=========================
Generates vector embeddings at multiple scales (Matryoshka) and spectral features.

This is a "warp thread" module - completely independent, no imports from other modules.

Architecture:
- Protocol-based design (Embedder)
- Multiple implementations (Matryoshka, Spectral)
- Zero dependencies on other HoloLoom modules
- Only imports from shared types layer

Philosophy:
Embeddings are the "Ψ" (psi) - the mathematical representation of meaning.
We generate multi-scale embeddings (like Russian nesting dolls) to enable
coarse-to-fine processing across the pipeline.
"""

import os
import hashlib
import warnings
from typing import List, Dict, Tuple, Optional, Protocol, Callable, Union
from dataclasses import dataclass, field
import numpy as np
import networkx as nx

# Import only from shared types layer
# Use the project's shared types module (avoid shadowing stdlib `types`)
from HoloLoom.documentation.types import Vector

# Optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    _HAVE_SENTENCE_TRANSFORMERS = True
except ImportError:
    SentenceTransformer = None
    _HAVE_SENTENCE_TRANSFORMERS = False
    warnings.warn(
        "sentence-transformers not available. "
        "Install with: pip install sentence-transformers"
    )

try:
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import eigsh
    _HAVE_SCIPY = True
except ImportError:
    coo_matrix = None
    eigsh = None
    _HAVE_SCIPY = False
    warnings.warn("scipy not available - spectral features will use dense solver")


# ============================================================================
# Protocol - Abstract Interface
# ============================================================================

class Embedder(Protocol):
    """
    Protocol for embedding implementations.
    
    All embedders must implement encode methods.
    This enables the orchestrator to swap implementations without code changes.
    """
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts into vectors.
        
        Args:
            texts: List of text strings
            
        Returns:
            Matrix of embeddings (n_texts × embedding_dim)
        """
        ...


# ============================================================================
# Matryoshka Embeddings - Multi-Scale Representation
# ============================================================================

@dataclass
class MatryoshkaEmbeddings:
    """
    Multi-scale embeddings using Matryoshka representation learning.
    
    Like Russian nesting dolls, we create embeddings at multiple dimensions:
    - Small (96d): Fast retrieval, coarse similarity
    - Medium (192d): Balanced performance
    - Large (384d): High-quality, fine-grained similarity
    
    This enables adaptive compute: use small embeddings for fast filtering,
    large embeddings for final ranking.
    
    Technical approach:
    - Generate full-dimensional base embedding (e.g., 384d)
    - Project to smaller dimensions using learned/random projections
    - Maintain semantic quality at each scale
    """
    
    sizes: List[int] = field(default_factory=lambda: [96, 192, 384])
    base_model_name: Optional[str] = None
    external_heads: Optional[Dict[int, Callable[[np.ndarray], np.ndarray]]] = None
    
    def __post_init__(self):
        assert sorted(self.sizes) == self.sizes, "Sizes must be in ascending order"

        self.external_heads = self.external_heads or {}
        self._last_hash = None

        # Lazy loading: Don't load model until first encode
        self._model = None
        self._model_loaded = False
        self.base_dim = max(self.sizes)  # Placeholder until model loads

        # Embedding cache (text -> embedding)
        from HoloLoom.performance.cache import QueryCache
        self._embedding_cache = QueryCache(max_size=500, ttl_seconds=3600)

        # Initialize projection matrices
        self._build_projection(seed=12345)

    def _ensure_model_loaded(self):
        """Lazy load the sentence transformer model."""
        if self._model_loaded:
            return

        if _HAVE_SENTENCE_TRANSFORMERS:
            model_name = (
                self.base_model_name or
                os.environ.get("HOLOLOOM_BASE_ENCODER", "all-MiniLM-L12-v2")  # Upgraded from L6 (2021) → L12 (2022) for better quality
            )
            try:
                self._model = SentenceTransformer(model_name)
                # Probe to get base dimension
                probe = self._model.encode(["test"], normalize_embeddings=True)[0]
                self.base_dim = len(probe)
                # Rebuild projections with correct base_dim
                self._build_projection(seed=12345)
            except Exception as e:
                warnings.warn(f"Failed to load {model_name}: {e}")
                self._model = None
                self.base_dim = max(self.sizes)
        else:
            self._model = None
            self.base_dim = max(self.sizes)

        self._model_loaded = True
    
    def _build_projection(self, seed: int = 12345):
        """
        Build orthogonal projection matrices for each scale.
        
        Uses QR decomposition to create orthonormal bases,
        ensuring minimal information loss during projection.
        """
        rng = np.random.default_rng(seed)
        # Create random matrix and orthogonalize
        Q, _ = np.linalg.qr(rng.normal(size=(self.base_dim, self.base_dim)))
        
        # Store projections for each target dimension
        self.proj = {d: Q[:, :d] for d in self.sizes}
    
    def refresh_runtime_qr(self, corpus_texts: List[str]):
        """
        Refresh projection matrices based on corpus.
        
        Optional: Adapt projections to the specific corpus being processed.
        Uses SHA-256 hash of corpus for deterministic but corpus-specific projections.
        
        Args:
            corpus_texts: Representative texts from the corpus
        """
        # Hash corpus for deterministic seed
        corpus_str = "\n".join(corpus_texts)
        h = hashlib.sha256(corpus_str.encode("utf-8")).hexdigest()
        
        if h != self._last_hash:
            seed = int(h[:8], 16)  # Use first 8 hex chars as seed
            self._build_projection(seed=seed)
            self._last_hash = h
    
    def encode_base(self, texts: List[str]) -> np.ndarray:
        """
        Generate base (full-dimensional) embeddings with caching.

        Args:
            texts: List of text strings

        Returns:
            Matrix of base embeddings (n_texts × base_dim)
        """
        if not texts:
            return np.zeros((0, self.base_dim))

        # Ensure model is loaded (lazy loading)
        self._ensure_model_loaded()

        # Check cache for each text
        vecs = []
        texts_to_encode = []
        indices_to_encode = []

        for i, text in enumerate(texts):
            cached = self._embedding_cache.get(text)
            if cached is not None:
                vecs.append((i, cached))
            else:
                texts_to_encode.append(text)
                indices_to_encode.append(i)

        # Encode uncached texts
        if texts_to_encode:
            if self._model is not None:
                # Use sentence-transformers
                new_vecs = self._model.encode(texts_to_encode, normalize_embeddings=True)
            else:
                # Fallback: deterministic random embeddings (for testing)
                new_vecs = []
                for text in texts_to_encode:
                    seed = abs(hash(text)) % (2**32)
                    rng = np.random.default_rng(seed)
                    vec = rng.normal(0, 1, self.base_dim)
                    vec = vec / (np.linalg.norm(vec) + 1e-9)
                    new_vecs.append(vec)
                new_vecs = np.vstack(new_vecs) if new_vecs else np.array([])

            # Cache new embeddings
            for idx, (text, vec) in enumerate(zip(texts_to_encode, new_vecs)):
                self._embedding_cache.put(text, vec)
                vecs.append((indices_to_encode[idx], vec))

        # Sort by original index and extract vectors
        vecs.sort(key=lambda x: x[0])
        return np.vstack([v for _, v in vecs])
    
    def encode_scales(
        self, 
        texts: List[str], 
        size: Optional[int] = None
    ) -> Union[Dict[int, np.ndarray], np.ndarray]:
        """
        Encode texts at one or all scales.
        
        Args:
            texts: List of text strings
            size: Optional specific size to return. If None, returns all scales.
            
        Returns:
            If size specified: Matrix of embeddings (n_texts × size)
            If size is None: Dict mapping size → embedding matrix
        """
        if not texts:
            # Handle empty input gracefully
            if size is not None:
                return np.zeros((0, size))
            return {d: np.zeros((0, d)) for d in self.sizes}
        
        # Generate base embeddings
        base = self.encode_base(texts)
        
        # Single size requested
        if size is not None:
            if size in self.external_heads:
                return self.external_heads[size](base)
            return base @ self.proj[size]
        
        # All sizes requested
        out: Dict[int, np.ndarray] = {}
        for d in self.sizes:
            if d in self.external_heads:
                out[d] = self.external_heads[d](base)
            else:
                out[d] = base @ self.proj[d]
        return out
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts at largest scale (for Protocol compatibility).
        
        Args:
            texts: List of text strings
            
        Returns:
            Matrix of embeddings at max dimension
        """
        return self.encode_scales(texts, size=max(self.sizes))


# ============================================================================
# Spectral Features - Graph-Based Embeddings
# ============================================================================

@dataclass
class SpectralFusion:
    """
    Generates spectral features from knowledge graphs and text embeddings.
    
    Spectral features capture:
    1. Graph structure (via Laplacian eigenvalues)
       - Fiedler value (2nd smallest eigenvalue) measures connectivity
       - Spectrum reveals community structure
    
    2. Semantic coherence (via SVD of embeddings)
       - Topic variance from singular values
       - Semantic diversity in retrieved content
    
    Combined, these form the Ψ (psi) vector that feeds into policy decisions.
    
    Output: 6-dimensional feature vector
    - [0:4]: Spectral features from graph Laplacian
    - [4:6]: Topic features from embedding SVD
    """
    
    k_eigen: int = 4  # Number of eigenvalues to compute
    svd_components: int = 2  # Number of SVD components for topic features
    
    async def features(
        self,
        kg_sub: nx.MultiDiGraph,
        shard_texts: List[str],
        emb: MatryoshkaEmbeddings
    ) -> Tuple[Vector, Dict[str, float]]:
        """
        Extract spectral features from graph and text.
        
        Args:
            kg_sub: Knowledge graph subgraph
            shard_texts: Retrieved text shards
            emb: Embeddings instance for encoding texts
            
        Returns:
            Tuple of (psi_vector, metrics_dict)
            - psi_vector: 6-dimensional feature vector
            - metrics_dict: Interpretable metrics (fiedler, topic_var, coherence)
        """
        # Part 1: Graph spectral features
        spec, fiedler = self._graph_spectrum(kg_sub)
        
        # Part 2: Topic features from embeddings
        topic, topic_var = self._topic_features(shard_texts, emb)
        
        # Combine into Ψ vector
        psi = np.concatenate([spec, topic])
        
        # Compute coherence score
        coherence = (1.0 if fiedler > 1e-6 else 0.0) + topic_var
        
        metrics = {
            "fiedler": fiedler,
            "topic_var": topic_var,
            "coherence": coherence
        }
        
        return psi, metrics
    
    def _graph_spectrum(self, kg_sub: nx.MultiDiGraph) -> Tuple[np.ndarray, float]:
        """
        Compute graph Laplacian spectrum.
        
        The Laplacian eigenvalues reveal graph structure:
        - λ₀ = 0 (always)
        - λ₁ (Fiedler value): measures connectivity (0 = disconnected)
        - Higher eigenvalues: reveal community structure
        
        Returns:
            Tuple of (spectrum, fiedler_value)
            - spectrum: First k eigenvalues (padded to k)
            - fiedler_value: Second smallest eigenvalue
        """
        n = kg_sub.number_of_nodes()
        
        if n <= 1:
            # Trivial graph
            return np.zeros(self.k_eigen), 0.0
        
        # Build adjacency matrix
        nodes = list(kg_sub.nodes())
        idx = {u: i for i, u in enumerate(nodes)}
        
        rows, cols, data = [], [], []
        for u, v, d in kg_sub.edges(data=True):
            w = float(d.get("weight", 1.0))
            # Symmetric (undirected)
            rows.extend([idx[u], idx[v]])
            cols.extend([idx[v], idx[u]])
            data.extend([w, w])
        
        # Compute Laplacian: L = D - A
        if _HAVE_SCIPY:
            # Sparse computation (efficient for large graphs)
            A = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
            D = np.asarray(A.sum(axis=1)).ravel()
            L = np.diag(D) - A.toarray()
            
            # Compute smallest k eigenvalues
            k = min(self.k_eigen, n - 1)
            try:
                vals, _ = eigsh(L.astype(float), k=k, which='SM')
                vals = np.sort(vals)
            except Exception:
                # Fallback to dense
                vals = np.linalg.eigvalsh(L)
                vals = np.sort(vals)
        else:
            # Dense computation (fallback)
            A = np.zeros((n, n))
            for r, c, w in zip(rows, cols, data):
                A[r, c] += w
            
            D = np.diag(A.sum(axis=1))
            L = D - A
            
            vals = np.linalg.eigvalsh(L)
            vals = np.sort(vals)
        
        # Pad to k eigenvalues
        spectrum = np.pad(vals[:self.k_eigen], (0, max(0, self.k_eigen - len(vals))), constant_values=0.0)
        fiedler = float(spectrum[1]) if len(spectrum) > 1 else 0.0
        
        return spectrum, fiedler
    
    def _topic_features(
        self,
        shard_texts: List[str],
        emb: MatryoshkaEmbeddings
    ) -> Tuple[np.ndarray, float]:
        """
        Extract topic features via SVD of embeddings.
        
        Captures semantic diversity in the retrieved content:
        - Large singular values: concentrated topic
        - Small singular values: diverse topics
        
        Returns:
            Tuple of (topic_vector, topic_variance)
            - topic_vector: Top SVD singular values
            - topic_variance: Normalized variance captured
        """
        if not shard_texts:
            return np.zeros(self.svd_components), 0.0
        
        # Encode at maximum scale for quality
        V = emb.encode_scales(shard_texts, size=max(emb.sizes))
        
        try:
            # Compute SVD (use subset for efficiency)
            m = min(len(V), 64)  # Cap at 64 texts
            if m == 0:
                return np.zeros(self.svd_components), 0.0
            
            _, s, _ = np.linalg.svd(V[:max(16, m), :], full_matrices=False)
            
            # Extract top components
            topic = s[:self.svd_components]
            # Pad if necessary
            if len(topic) < self.svd_components:
                topic = np.pad(topic, (0, self.svd_components - len(topic)))
            
            # Topic variance: how much is captured by top components
            topic_var = float(topic.sum() / (s.sum() + 1e-9))
            
        except Exception as e:
            warnings.warn(f"SVD failed: {e}")
            topic = np.zeros(self.svd_components)
            topic_var = 0.0
        
        return topic, topic_var


# ============================================================================
# Factory Function
# ============================================================================

def create_embedder(
    mode: str = "matryoshka",
    sizes: List[int] = [96, 192, 384],
    base_model_name: Optional[str] = None
) -> Embedder:
    """
    Factory function to create embedder.
    
    Args:
        mode: Type of embedder ("matryoshka")
        sizes: Dimensions for Matryoshka embeddings
        base_model_name: Optional model name for sentence-transformers
        
    Returns:
        Embedder implementation
    """
    if mode == "matryoshka":
        return MatryoshkaEmbeddings(
            sizes=sizes,
            base_model_name=base_model_name
        )
    else:
        raise ValueError(f"Unknown embedder mode: {mode}")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def demo():
        print("=== Matryoshka Embeddings Demo ===\n")
        
        # Create embedder
        emb = MatryoshkaEmbeddings(sizes=[96, 192, 384])
        
        # Test texts
        texts = [
            "Multi-head attention processes multiple representation subspaces",
            "Transformers use self-attention mechanisms",
            "Neural networks learn hierarchical features"
        ]
        
        # Encode at all scales
        print("Encoding at all scales:")
        multi_scale = emb.encode_scales(texts)
        for size, vecs in multi_scale.items():
            print(f"  {size}d: shape={vecs.shape}, norm={np.linalg.norm(vecs[0]):.3f}")
        
        # Encode at single scale
        print("\nEncoding at 192d:")
        vecs_192 = emb.encode_scales(texts, size=192)
        print(f"  Shape: {vecs_192.shape}")
        
        # Test spectral features
        print("\n=== Spectral Features Demo ===\n")
        
        # Create sample graph
        G = nx.MultiDiGraph()
        G.add_edge("attention", "transformer", type="USES")
        G.add_edge("transformer", "neural_network", type="IS_A")
        G.add_edge("attention", "neural_network", type="PART_OF")
        
        spectral = SpectralFusion()
        psi, metrics = await spectral.features(G, texts, emb)
        
        print(f"Ψ vector: {psi}")
        print(f"Metrics: {metrics}")
        print(f"  Fiedler value: {metrics['fiedler']:.3f} (connectivity)")
        print(f"  Topic variance: {metrics['topic_var']:.3f} (coherence)")
        print(f"  Overall coherence: {metrics['coherence']:.3f}")
    
    asyncio.run(demo())