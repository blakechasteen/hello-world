"""
Warp Space - Tensioned Tensor Field
====================================
The temporary computational manifold where activated threads undergo
continuous mathematical operations.

Philosophy:
When the Chrono Trigger fires, specific threads from the Yarn Graph are
selected and "put under tensor tension" - pulled taut into Warp Space.
This creates a continuous mathematical region where embeddings, spectral
operations, and neural computations occur.

After the Convergence Engine collapses to a decision, Warp Space detensions
and the threads return to the discrete Yarn Graph, enriched with new weights
and connections.

Warp Space is TEMPORARY - it only exists during active computation.
"""

import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Warp Space
# ============================================================================

@dataclass
class TensionedThread:
    """
    A thread from Yarn Graph under active tension.

    Tracks both the discrete (symbolic) and continuous (tensor) representations.
    """
    thread_id: str
    entity: str
    embedding: np.ndarray  # Continuous representation
    tension: float = 1.0  # Activation strength (0-1)
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class WarpSpace:
    """
    Tensioned computational manifold.

    WarpSpace manages the activated subset of the Yarn Graph during query processing.
    It maintains both discrete (thread IDs, entities) and continuous (embeddings,
    tensor fields) representations.

    Lifecycle:
    1. tension() - Pull threads taut from Yarn Graph
    2. compute() - Perform continuous math operations
    3. collapse() - Return threads to Yarn Graph with updates

    Usage:
        warp = WarpSpace(embedder, scales=[96, 192, 384])
        await warp.tension(thread_ids, yarn_graph)
        # ... continuous computations ...
        updates = warp.collapse()
    """

    def __init__(
        self,
        embedder,
        scales: List[int] = [96, 192, 384],
        spectral_fusion = None
    ):
        """
        Initialize Warp Space.

        Args:
            embedder: MatryoshkaEmbeddings instance
            scales: Embedding scales for multi-scale representation
            spectral_fusion: Optional SpectralFusion for spectral features
        """
        self.embedder = embedder
        self.scales = scales
        self.spectral_fusion = spectral_fusion

        # Tensioned threads
        self.threads: List[TensionedThread] = []
        self.thread_index: Dict[str, TensionedThread] = {}

        # Tensor field state
        self.is_tensioned = False
        self.tensor_field = None  # Continuous representation

        # Computational history
        self.operations = []

        logger.info(f"WarpSpace initialized (scales={scales})")

    async def tension(
        self,
        thread_texts: List[str],
        thread_ids: Optional[List[str]] = None,
        tension_weights: Optional[List[float]] = None
    ) -> None:
        """
        Pull threads taut into Warp Space.

        Converts discrete thread data into continuous tensor representations.

        Args:
            thread_texts: Text content of threads to tension
            thread_ids: Optional IDs for threads
            tension_weights: Optional activation strengths (0-1)
        """
        if not thread_texts:
            logger.warning("No threads to tension")
            return

        logger.info(f"Tensioning {len(thread_texts)} threads into Warp Space")

        # Generate IDs if not provided
        if thread_ids is None:
            thread_ids = [f"thread_{i}" for i in range(len(thread_texts))]

        # Default uniform tension
        if tension_weights is None:
            tension_weights = [1.0] * len(thread_texts)

        # Encode threads at all scales
        embeddings_dict = self.embedder.encode_scales(thread_texts)

        # Use largest scale for primary embedding
        max_scale = max(self.scales)
        embeddings = embeddings_dict[max_scale]

        # Create tensioned threads
        self.threads = []
        self.thread_index = {}

        for idx, (text, thread_id, tension) in enumerate(zip(thread_texts, thread_ids, tension_weights)):
            thread = TensionedThread(
                thread_id=thread_id,
                entity=text[:50],  # First 50 chars as entity name
                embedding=embeddings[idx],
                tension=tension,
                metadata={
                    'text': text,
                    'index': idx,
                    'multi_scale_embeddings': {scale: embeddings_dict[scale][idx] for scale in self.scales}
                }
            )
            self.threads.append(thread)
            self.thread_index[thread_id] = thread

        # Create tensor field (stacked embeddings)
        self.tensor_field = np.stack([t.embedding for t in self.threads])
        self.is_tensioned = True

        logger.info(f"Warp Space tensioned: {len(self.threads)} threads, field shape={self.tensor_field.shape}")

    def get_field(self, scale: Optional[int] = None) -> np.ndarray:
        """
        Get tensor field at specified scale.

        Args:
            scale: Embedding dimension (None = largest scale)

        Returns:
            Tensor field matrix (n_threads × scale_dim)
        """
        if not self.is_tensioned:
            raise RuntimeError("Warp Space not tensioned")

        if scale is None:
            return self.tensor_field

        # Extract multi-scale embeddings
        embeddings = []
        for thread in self.threads:
            multi_scale = thread.metadata.get('multi_scale_embeddings', {})
            if scale in multi_scale:
                embeddings.append(multi_scale[scale])
            else:
                # Fallback: project from largest scale
                embeddings.append(thread.embedding[:scale])

        return np.stack(embeddings)

    def compute_spectral_features(self) -> Dict[str, Any]:
        """
        Compute spectral features from tensor field.

        Uses spectral fusion if available, otherwise basic statistics.

        Returns:
            Dict with spectral feature data
        """
        if not self.is_tensioned:
            logger.warning("Cannot compute spectral features: Warp Space not tensioned")
            return {}

        features = {}

        # Basic tensor statistics
        features['field_norm'] = np.linalg.norm(self.tensor_field)
        features['mean_activation'] = float(np.mean([t.tension for t in self.threads]))
        features['dimensionality'] = self.tensor_field.shape

        # SVD-based features
        try:
            _, s, _ = np.linalg.svd(self.tensor_field, full_matrices=False)
            features['singular_values'] = s[:min(6, len(s))].tolist()
            features['spectral_entropy'] = float(-np.sum(s * np.log(s + 1e-10)))
        except Exception as e:
            logger.warning(f"SVD computation failed: {e}")

        self.operations.append(('spectral_features', features))

        return features

    def apply_attention(self, query_embedding: np.ndarray) -> np.ndarray:
        """
        Apply attention from query to tensioned threads.

        Args:
            query_embedding: Query embedding vector

        Returns:
            Attention weights over threads
        """
        if not self.is_tensioned:
            raise RuntimeError("Warp Space not tensioned")

        # Compute attention scores (dot product similarity)
        scores = self.tensor_field @ query_embedding

        # Apply softmax
        exp_scores = np.exp(scores - np.max(scores))
        attention = exp_scores / np.sum(exp_scores)

        self.operations.append(('attention', {'scores': scores, 'attention': attention}))

        return attention

    def weighted_context(self, attention: np.ndarray) -> np.ndarray:
        """
        Compute weighted context vector from attention.

        Args:
            attention: Attention weights over threads

        Returns:
            Weighted sum of thread embeddings
        """
        if not self.is_tensioned:
            raise RuntimeError("Warp Space not tensioned")

        context = attention @ self.tensor_field

        self.operations.append(('weighted_context', {'attention_entropy': float(-np.sum(attention * np.log(attention + 1e-10)))}))

        return context

    def collapse(self) -> Dict[str, Any]:
        """
        Collapse Warp Space back to discrete representation.

        Detensions threads and returns updates for Yarn Graph.

        Returns:
            Dict with thread updates and computational trace
        """
        if not self.is_tensioned:
            logger.warning("Warp Space already collapsed")
            return {}

        logger.info(f"Collapsing Warp Space: {len(self.threads)} threads, {len(self.operations)} operations")

        # Prepare updates
        updates = {
            'threads': [
                {
                    'thread_id': t.thread_id,
                    'tension': t.tension,
                    'embedding_norm': float(np.linalg.norm(t.embedding)),
                    'metadata': t.metadata
                }
                for t in self.threads
            ],
            'operations': self.operations,
            'field_stats': {
                'shape': self.tensor_field.shape,
                'norm': float(np.linalg.norm(self.tensor_field)),
                'mean': float(np.mean(self.tensor_field)),
                'std': float(np.std(self.tensor_field))
            }
        }

        # Reset state
        self.threads = []
        self.thread_index = {}
        self.tensor_field = None
        self.is_tensioned = False
        self.operations = []

        logger.info("Warp Space collapsed")

        return updates

    def get_trace(self) -> Dict[str, Any]:
        """
        Get computational trace without collapsing.

        Returns:
            Dict with current state and operations
        """
        return {
            'is_tensioned': self.is_tensioned,
            'num_threads': len(self.threads),
            'operations': self.operations.copy(),
            'field_shape': self.tensor_field.shape if self.tensor_field is not None else None
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import asyncio
    from holoLoom.embedding.spectral import MatryoshkaEmbeddings

    async def demo():
        print("="*80)
        print("Warp Space Demo")
        print("="*80 + "\n")

        # Create embedder and warp space
        embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
        warp = WarpSpace(embedder, scales=[96, 192, 384])

        # Sample threads from Yarn Graph
        thread_texts = [
            "Thompson Sampling balances exploration and exploitation",
            "Neural networks learn hierarchical representations",
            "Attention mechanisms enable context-aware processing"
        ]

        # Tension threads into Warp Space
        print("1. Tensioning threads...")
        await warp.tension(thread_texts)
        print(f"   Field shape: {warp.tensor_field.shape}\n")

        # Compute spectral features
        print("2. Computing spectral features...")
        spectral = warp.compute_spectral_features()
        print(f"   Field norm: {spectral['field_norm']:.3f}")
        print(f"   Spectral entropy: {spectral.get('spectral_entropy', 0):.3f}\n")

        # Apply attention
        print("3. Applying attention from query...")
        query_text = ["What is Thompson Sampling?"]
        query_emb = embedder.encode_scales(query_text, size=384)
        attention = warp.apply_attention(query_emb[0])
        print(f"   Attention: {attention}")
        print(f"   Max attention on thread: {np.argmax(attention)}\n")

        # Get weighted context
        print("4. Computing weighted context...")
        context = warp.weighted_context(attention)
        print(f"   Context shape: {context.shape}\n")

        # Collapse
        print("5. Collapsing Warp Space...")
        updates = warp.collapse()
        print(f"   Operations performed: {len(updates['operations'])}")
        print(f"   Threads processed: {len(updates['threads'])}\n")

        print("✓ Demo complete!")

    asyncio.run(demo())
