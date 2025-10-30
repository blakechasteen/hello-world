"""
Multi-Modal Memory System - Elegant Cross-Modal Storage & Retrieval
====================================================================
Everything is a memory operation. Store, retrieve, connect.

Philosophy:
- Modality is just metadata - all memories are equal
- Cross-modal search is natural - embeddings bridge modalities
- Knowledge graphs connect everything - text links to images links to audio
- Elegant interfaces hide complexity - users think about intent, not mechanism

Architecture:
    Input → MultiModalSpinner → MemoryShard → MultiModalMemory
    Query → CrossModalSearch → Fused Results → Knowledge Graph

Operations:
- store(): Store any modality transparently
- retrieve(): Cross-modal semantic search
- connect(): Build relationships across modalities
- explore(): Navigate multi-modal knowledge graph
"""

from typing import List, Dict, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import asyncio
import numpy as np
from pathlib import Path
import warnings

try:
    from HoloLoom.documentation.types import MemoryShard, Query
    from HoloLoom.memory.protocol import Memory, MemoryQuery, RetrievalResult
    from HoloLoom.memory.neo4j_graph import Neo4jKG, Neo4jConfig
except ImportError:
    warnings.warn("HoloLoom types not available - using fallbacks")
    MemoryShard = None
    Query = None


# ============================================================================
# Multi-Modal Types
# ============================================================================

class ModalityType(Enum):
    """Supported modality types."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    STRUCTURED = "structured"
    MULTIMODAL = "multimodal"
    UNKNOWN = "unknown"


class FusionStrategy(Enum):
    """Cross-modal fusion strategies."""
    ATTENTION = "attention"  # Confidence-weighted
    AVERAGE = "average"      # Simple average
    MAX = "max"              # Element-wise max
    LEARNED = "learned"      # Learned fusion weights


@dataclass
class ModalityMetadata:
    """Metadata for modality-specific features."""
    modality_type: ModalityType
    confidence: float
    embedding: Optional[np.ndarray] = None
    features: Optional[Dict[str, Any]] = None
    source: Optional[str] = None
    timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for storage."""
        return {
            'modality_type': self.modality_type.value,
            'confidence': float(self.confidence),
            'embedding': self.embedding.tolist() if self.embedding is not None else None,
            'features': self.features,
            'source': self.source,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModalityMetadata':
        """Create from dict."""
        data = data.copy()
        data['modality_type'] = ModalityType(data['modality_type'])
        if data.get('embedding'):
            data['embedding'] = np.array(data['embedding'])
        if data.get('timestamp'):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class CrossModalResult:
    """Result from cross-modal search."""
    memories: List[Memory]
    scores: List[float]
    modalities: List[ModalityType]
    fusion_strategy: FusionStrategy
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def group_by_modality(self) -> Dict[ModalityType, List[Tuple[Memory, float]]]:
        """Group results by modality."""
        groups = {}
        for mem, score, mod in zip(self.memories, self.scores, self.modalities):
            if mod not in groups:
                groups[mod] = []
            groups[mod].append((mem, score))
        return groups


# ============================================================================
# Multi-Modal Memory Store
# ============================================================================

class MultiModalMemory:
    """
    Elegant multi-modal memory system.
    
    Core Operations:
    - store(shard): Store any modality transparently
    - retrieve(query, modality_filter): Cross-modal semantic search
    - connect(id1, id2, relationship): Link across modalities
    - explore(start_id, hops): Navigate knowledge graph
    
    Everything is a memory operation. Stay elegant.
    
    Usage:
        memory = MultiModalMemory()
        
        # Store text
        await memory.store(text_shard)
        
        # Store image
        await memory.store(image_shard)
        
        # Cross-modal search: "Show me text and images about quantum computing"
        results = await memory.retrieve(
            query="quantum computing",
            modality_filter=[ModalityType.TEXT, ModalityType.IMAGE],
            k=10
        )
        
        # Navigate knowledge graph
        related = await memory.explore(
            start_id="shard_123",
            hops=2,
            modality_filter=[ModalityType.IMAGE]
        )
    """
    
    def __init__(
        self,
        neo4j_config: Optional[Neo4jConfig] = None,
        enable_neo4j: bool = True,
        enable_qdrant: bool = True,
        default_fusion: FusionStrategy = FusionStrategy.ATTENTION
    ):
        """
        Initialize multi-modal memory.
        
        Args:
            neo4j_config: Neo4j connection config
            enable_neo4j: Enable graph storage
            enable_qdrant: Enable vector storage
            default_fusion: Default fusion strategy
        """
        self.default_fusion = default_fusion
        self.enable_neo4j = enable_neo4j
        self.enable_qdrant = enable_qdrant
        
        # Storage backends
        self.graph_store = None
        self.vector_store = None
        
        # In-memory cache for fast access
        self.memory_cache: Dict[str, Memory] = {}
        self.modality_index: Dict[ModalityType, Set[str]] = {
            mod: set() for mod in ModalityType
        }
        
        # Initialize backends
        if enable_neo4j:
            try:
                from HoloLoom.memory.neo4j_graph import Neo4jKG
                self.graph_store = Neo4jKG(neo4j_config or Neo4jConfig.from_env())
            except Exception as e:
                warnings.warn(f"Neo4j unavailable: {e}. Using in-memory only.")
                self.enable_neo4j = False
        
        if enable_qdrant:
            # Qdrant integration coming soon
            warnings.warn("Qdrant integration pending - using in-memory vectors")
            self.enable_qdrant = False
    
    # ========================================================================
    # Core Operations (Elegant API)
    # ========================================================================
    
    async def store(
        self,
        shard: Any,
        user_id: str = "default"
    ) -> str:
        """
        Store a memory shard (any modality).
        
        Elegantly handles:
        - Modality detection from metadata
        - Embedding extraction
        - Graph node creation
        - Vector indexing
        - Cross-modal linking
        
        Args:
            shard: MemoryShard with modality metadata
            user_id: User identifier
            
        Returns:
            memory_id: Unique identifier
        """
        # Extract modality metadata
        modality_meta = self._extract_modality(shard)
        
        # Convert to Memory
        memory = self._shard_to_memory(shard, modality_meta)
        
        # Cache in memory
        self.memory_cache[memory.id] = memory
        self.modality_index[modality_meta.modality_type].add(memory.id)
        
        # Store in graph (if enabled)
        if self.enable_neo4j and self.graph_store:
            try:
                await self._store_in_graph(memory, modality_meta)
            except Exception as e:
                warnings.warn(f"Graph storage failed: {e}")
        
        # Store in vector index (if enabled)
        if self.enable_qdrant and self.vector_store:
            try:
                await self._store_in_vectors(memory, modality_meta)
            except Exception as e:
                warnings.warn(f"Vector storage failed: {e}")
        
        return memory.id
    
    async def store_batch(
        self,
        shards: List[Any],
        user_id: str = "default"
    ) -> List[str]:
        """
        Store multiple shards efficiently.
        
        Args:
            shards: List of MemoryShards
            user_id: User identifier
            
        Returns:
            memory_ids: List of unique identifiers
        """
        # Store all in parallel
        tasks = [self.store(shard, user_id) for shard in shards]
        return await asyncio.gather(*tasks)
    
    async def retrieve(
        self,
        query: str,
        modality_filter: Optional[List[ModalityType]] = None,
        k: int = 10,
        fusion_strategy: Optional[FusionStrategy] = None,
        threshold: float = 0.0
    ) -> CrossModalResult:
        """
        Cross-modal semantic search.
        
        Elegantly handles:
        - Query embedding
        - Modality filtering
        - Cross-modal similarity
        - Result fusion
        - Relevance ranking
        
        Args:
            query: Natural language query
            modality_filter: Filter by modalities (None = all)
            k: Number of results
            fusion_strategy: How to fuse cross-modal results
            threshold: Minimum similarity threshold
            
        Returns:
            CrossModalResult with memories, scores, modalities
        """
        fusion = fusion_strategy or self.default_fusion
        
        # Get query embedding (placeholder - needs actual embedding)
        query_embedding = self._embed_query(query)
        
        # Filter by modality
        candidate_ids = self._get_candidates(modality_filter)
        
        # Compute similarities
        results = []
        for mem_id in candidate_ids:
            memory = self.memory_cache.get(mem_id)
            if not memory:
                continue
            
            # Extract embedding from metadata
            embedding = memory.metadata.get('embedding')
            if embedding is None:
                continue
            
            # Convert to numpy if needed
            if isinstance(embedding, list):
                embedding = np.array(embedding)
            
            # Compute similarity
            similarity = self._compute_similarity(query_embedding, embedding)
            
            if similarity >= threshold:
                # Get modality safely
                modality_str = memory.metadata.get('modality_type', 'unknown')
                if isinstance(modality_str, str):
                    modality_map = {
                        'text': ModalityType.TEXT,
                        'image': ModalityType.IMAGE,
                        'audio': ModalityType.AUDIO,
                        'video': ModalityType.VIDEO,
                        'structured': ModalityType.STRUCTURED,
                        'multimodal': ModalityType.MULTIMODAL,
                        'unknown': ModalityType.UNKNOWN
                    }
                    modality = modality_map.get(modality_str.lower(), ModalityType.UNKNOWN)
                else:
                    modality = ModalityType.UNKNOWN
                results.append((memory, similarity, modality))
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Take top k
        results = results[:k]
        
        if not results:
            return CrossModalResult([], [], [], fusion, {})
        
        # Unpack
        memories, scores, modalities = zip(*results)
        
        return CrossModalResult(
            memories=list(memories),
            scores=list(scores),
            modalities=list(modalities),
            fusion_strategy=fusion,
            metadata={
                'query': query,
                'total_candidates': len(candidate_ids),
                'filtered_modalities': [m.value for m in modality_filter] if modality_filter else 'all'
            }
        )
    
    async def connect(
        self,
        id1: str,
        id2: str,
        relationship: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Create cross-modal relationship.
        
        Examples:
        - Text mentions Image: "describes"
        - Audio describes Document: "narrates"
        - Image part of Video: "frame_of"
        
        Args:
            id1: Source memory ID
            id2: Target memory ID
            relationship: Relationship type
            metadata: Additional metadata
            
        Returns:
            success: True if connected
        """
        if self.enable_neo4j and self.graph_store:
            try:
                # Create edge in graph
                await self.graph_store.add_edge(
                    from_node=id1,
                    to_node=id2,
                    edge_type=relationship,
                    metadata=metadata or {}
                )
                return True
            except Exception as e:
                warnings.warn(f"Failed to create connection: {e}")
                return False
        else:
            # In-memory connection tracking (simple)
            warnings.warn("Graph storage not enabled - connection not persisted")
            return False
    
    async def explore(
        self,
        start_id: str,
        hops: int = 1,
        modality_filter: Optional[List[ModalityType]] = None,
        relationship_filter: Optional[List[str]] = None
    ) -> List[Tuple[Memory, int, str]]:
        """
        Navigate multi-modal knowledge graph.
        
        Args:
            start_id: Starting memory ID
            hops: Number of hops to traverse
            modality_filter: Filter by modalities
            relationship_filter: Filter by relationship types
            
        Returns:
            List of (Memory, distance, relationship) tuples
        """
        if not self.enable_neo4j or not self.graph_store:
            warnings.warn("Graph exploration requires Neo4j")
            return []
        
        try:
            # Use graph traversal
            nodes = await self.graph_store.traverse(
                start_node=start_id,
                max_depth=hops,
                relationship_types=relationship_filter
            )
            
            # Filter by modality and convert to memories
            results = []
            for node, distance, rel in nodes:
                memory = self.memory_cache.get(node['id'])
                if not memory:
                    continue
                
                # Check modality filter
                if modality_filter:
                    modality = ModalityType(memory.metadata.get('modality_type', 'unknown'))
                    if modality not in modality_filter:
                        continue
                
                results.append((memory, distance, rel))
            
            return results
        except Exception as e:
            warnings.warn(f"Graph exploration failed: {e}")
            return []
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _extract_modality(self, shard: Any) -> ModalityMetadata:
        """Extract modality metadata from shard."""
        metadata = getattr(shard, 'metadata', {}) or {}
        
        # Try to get modality from metadata or modality field
        modality_str = metadata.get('modality_type', 'unknown')
        if hasattr(shard, 'modality') and shard.modality:
            modality_str = str(shard.modality).lower()
        
        # Map common modality strings
        modality_map = {
            'text': ModalityType.TEXT,
            'image': ModalityType.IMAGE,
            'audio': ModalityType.AUDIO,
            'video': ModalityType.VIDEO,
            'structured': ModalityType.STRUCTURED,
            'multimodal': ModalityType.MULTIMODAL,
            'unknown': ModalityType.UNKNOWN
        }
        
        modality_type = modality_map.get(modality_str.lower(), ModalityType.UNKNOWN)
        
        confidence = metadata.get('confidence', 1.0)
        embedding = metadata.get('embedding')
        if embedding and isinstance(embedding, list):
            embedding = np.array(embedding)
        
        return ModalityMetadata(
            modality_type=modality_type,
            confidence=confidence,
            embedding=embedding,
            features=metadata.get('features'),
            source=metadata.get('source'),
            timestamp=datetime.now()
        )
    
    def _shard_to_memory(self, shard: Any, modality_meta: ModalityMetadata) -> Memory:
        """Convert MemoryShard to Memory."""
        from HoloLoom.memory.protocol import Memory
        
        return Memory(
            id=shard.id,
            text=shard.text,
            timestamp=modality_meta.timestamp or datetime.now(),
            context={
                'episode': getattr(shard, 'episode', None),
                'entities': getattr(shard, 'entities', []),
                'motifs': getattr(shard, 'motifs', []),
            },
            metadata={
                **modality_meta.to_dict(),
                **(getattr(shard, 'metadata', {}) or {})
            }
        )
    
    async def _store_in_graph(self, memory: Memory, modality_meta: ModalityMetadata):
        """Store memory in Neo4j graph."""
        # Add node with modality properties
        await self.graph_store.add_node(
            node_id=memory.id,
            node_type=modality_meta.modality_type.value,
            properties={
                'text': memory.text[:1000],  # Truncate for storage
                'modality': modality_meta.modality_type.value,
                'confidence': modality_meta.confidence,
                'timestamp': memory.timestamp.isoformat(),
                'entities': memory.context.get('entities', []),
                'motifs': memory.context.get('motifs', [])
            }
        )
        
        # Connect to entities
        for entity in memory.context.get('entities', []):
            entity_name = entity if isinstance(entity, str) else entity.get('text', '')
            if entity_name:
                try:
                    await self.connect(
                        memory.id,
                        f"entity_{entity_name}",
                        "mentions",
                        {'entity': entity_name}
                    )
                except:
                    pass  # Entity might not exist yet
    
    async def _store_in_vectors(self, memory: Memory, modality_meta: ModalityMetadata):
        """Store memory embedding in Qdrant."""
        # Qdrant integration coming soon
        pass
    
    def _embed_query(self, query: str) -> np.ndarray:
        """Embed query for search (uses same dimension as stored embeddings)."""
        # Check what embedding dimensions we have in cache
        embedding_dims = set()
        for memory in self.memory_cache.values():
            emb = memory.metadata.get('embedding')
            if emb is not None:
                if isinstance(emb, list):
                    emb = np.array(emb)
                embedding_dims.add(len(emb))
        
        # Use most common dimension, or default to 128
        target_dim = max(embedding_dims) if embedding_dims else 128
        
        # Simple hash-based embedding for now (replace with actual embedder)
        np.random.seed(hash(query) % (2**31))
        return np.random.randn(target_dim)
    
    def _get_candidates(
        self,
        modality_filter: Optional[List[ModalityType]]
    ) -> Set[str]:
        """Get candidate memory IDs filtered by modality."""
        if modality_filter is None:
            # Return all memory IDs
            return set(self.memory_cache.keys())
        
        # Union of modality indices
        candidates = set()
        for modality in modality_filter:
            candidates.update(self.modality_index[modality])
        return candidates
    
    def _compute_similarity(
        self,
        query_embedding: np.ndarray,
        memory_embedding: np.ndarray
    ) -> float:
        """Compute cosine similarity (handles dimension mismatch gracefully)."""
        # Handle dimension mismatch
        if query_embedding.shape != memory_embedding.shape:
            # Use smaller dimension
            min_dim = min(len(query_embedding), len(memory_embedding))
            query_embedding = query_embedding[:min_dim]
            memory_embedding = memory_embedding[:min_dim]
        
        # Normalize
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        memory_norm = memory_embedding / (np.linalg.norm(memory_embedding) + 1e-8)
        
        # Cosine similarity
        similarity = np.dot(query_norm, memory_norm)
        return float(similarity)
    
    # ========================================================================
    # Statistics & Introspection
    # ========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        stats = {
            'total_memories': len(self.memory_cache),
            'by_modality': {
                mod.value: len(ids) 
                for mod, ids in self.modality_index.items()
                if len(ids) > 0
            },
            'backends': {
                'neo4j': self.enable_neo4j,
                'qdrant': self.enable_qdrant
            }
        }
        return stats
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"MultiModalMemory("
            f"memories={stats['total_memories']}, "
            f"modalities={list(stats['by_modality'].keys())}, "
            f"backends={stats['backends']})"
        )


# ============================================================================
# Convenience Functions
# ============================================================================

async def create_multimodal_memory(
    neo4j_config: Optional[Neo4jConfig] = None,
    **kwargs
) -> MultiModalMemory:
    """
    Create and initialize multi-modal memory system.
    
    Usage:
        memory = await create_multimodal_memory()
        await memory.store(shard)
    """
    return MultiModalMemory(neo4j_config=neo4j_config, **kwargs)


# ============================================================================
# Example Usage
# ============================================================================

async def example_usage():
    """Example of elegant multi-modal memory operations."""
    from HoloLoom.spinningWheel.multimodal_spinner import MultiModalSpinner, CrossModalSpinner
    
    # Create memory system
    memory = await create_multimodal_memory()
    
    # Create spinners
    text_spinner = MultiModalSpinner()
    cross_spinner = CrossModalSpinner()
    
    # Store text
    text_shards = await text_spinner.spin("Quantum computing uses qubits.")
    for shard in text_shards:
        await memory.store(shard)
    
    # Store structured data
    data_shards = await text_spinner.spin({"topic": "quantum", "year": 2025})
    for shard in data_shards:
        await memory.store(shard)
    
    # Cross-modal query
    results = await memory.retrieve(
        query="quantum computing",
        modality_filter=[ModalityType.TEXT, ModalityType.STRUCTURED],
        k=5
    )
    
    print(f"Found {len(results.memories)} memories")
    print(f"Modalities: {[m.value for m in results.modalities]}")
    print(f"Scores: {results.scores}")
    
    # Statistics
    print("\nMemory Stats:")
    print(memory.get_stats())


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
