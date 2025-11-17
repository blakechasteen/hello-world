"""
HoloLoom Multi-Modal Search
============================
Cross-modal retrieval and hybrid search across modalities.

This module enables searching across different modalities:
- Text → Image: "find images of cats"
- Image → Text: "find documents about this image"
- Audio → Video: "find videos with similar audio"
- Cross-modal: "find content about machine learning" (any modality)

Features:
- Cross-modal similarity search
- Hybrid ranking with modal fusion
- Integration with Qdrant vector store
- Relevance scoring with modal weights
- Filter by modality or content type

Architecture:
- Unified 768D embedding space enables cross-modal search
- Query in any modality, retrieve from any modality
- Fusion strategies for multi-modal queries
- Optional re-ranking with cross-modal attention
"""

import logging
import warnings
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np

# Import from base module
from HoloLoom.multimodal.base import (
    Modality,
    ModalEmbedding,
    MultiModalEmbedding
)

logger = logging.getLogger(__name__)

# Import encoders
try:
    from HoloLoom.multimodal.image_encoder import ImageEncoder
    from HoloLoom.multimodal.audio_encoder import AudioEncoder
    from HoloLoom.multimodal.video_encoder import VideoEncoder
    from HoloLoom.multimodal.fusion import MultiModalFuser
    _HAVE_ENCODERS = True
except ImportError:
    ImageEncoder = None
    AudioEncoder = None
    VideoEncoder = None
    MultiModalFuser = None
    _HAVE_ENCODERS = False
    warnings.warn("Encoders not available")

# Import text embedder
try:
    from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
    _HAVE_TEXT_EMBEDDER = True
except ImportError:
    MatryoshkaEmbeddings = None
    _HAVE_TEXT_EMBEDDER = False
    warnings.warn("MatryoshkaEmbeddings not available")

# Import Qdrant store
try:
    from HoloLoom.memory.qdrant_store import QdrantStore
    _HAVE_QDRANT = True
except ImportError:
    QdrantStore = None
    _HAVE_QDRANT = False
    warnings.warn("QdrantStore not available")


# ============================================================================
# Search Result
# ============================================================================

@dataclass
class SearchResult:
    """
    Single search result with relevance score.

    Attributes:
        id: Document/content ID
        modality: Source modality
        embedding: 768D embedding vector
        metadata: Content metadata (file path, transcription, etc.)
        score: Relevance score (0.0 to 1.0)
        cross_modal: Whether this is a cross-modal match
    """
    id: str
    modality: Modality
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    cross_modal: bool = False


@dataclass
class SearchResults:
    """
    Collection of search results with metadata.

    Attributes:
        results: List of SearchResult objects
        query_modality: Modality of the query
        target_modalities: Modalities searched
        fusion_method: Fusion method used (if multi-modal query)
        total_results: Total number of results found
    """
    results: List[SearchResult] = field(default_factory=list)
    query_modality: Optional[Modality] = None
    target_modalities: List[Modality] = field(default_factory=list)
    fusion_method: Optional[str] = None
    total_results: int = 0


# ============================================================================
# Multi-Modal Search Engine
# ============================================================================

class MultiModalSearch:
    """
    Multi-modal search engine with cross-modal retrieval.

    Enables searching across modalities using a unified embedding space.
    Integrates with Qdrant for efficient vector similarity search.

    Features:
    - Encode queries in any modality
    - Search across single or multiple modalities
    - Re-rank results with fusion strategies
    - Filter by modality, content type, metadata

    Example:
        search = MultiModalSearch()
        results = await search.search(
            query="cats",
            query_modality=Modality.TEXT,
            target_modalities=[Modality.IMAGE, Modality.VIDEO]
        )
    """

    def __init__(
        self,
        collection_name: str = "hololoom_multimodal",
        qdrant_url: str = "http://localhost:6333",
        fusion_method: str = "late"
    ):
        """
        Initialize multi-modal search engine.

        Args:
            collection_name: Qdrant collection name
            qdrant_url: Qdrant server URL
            fusion_method: Fusion method for multi-modal queries
        """
        self.collection_name = collection_name
        self.qdrant_url = qdrant_url
        self.fusion_method = fusion_method

        # Initialize encoders
        self._text_encoder = None
        self._image_encoder = None
        self._audio_encoder = None
        self._video_encoder = None
        self._fuser = None
        self._qdrant = None

        # Load encoders
        if _HAVE_TEXT_EMBEDDER:
            try:
                self._text_encoder = MatryoshkaEmbeddings(sizes=[768])
            except Exception as e:
                logger.error(f"Failed to load text encoder: {e}")

        if _HAVE_ENCODERS:
            try:
                self._image_encoder = ImageEncoder()
                self._audio_encoder = AudioEncoder()
                self._video_encoder = VideoEncoder()
                self._fuser = MultiModalFuser(method=fusion_method)
            except Exception as e:
                logger.error(f"Failed to load encoders: {e}")

        # Initialize Qdrant connection
        if _HAVE_QDRANT:
            try:
                self._qdrant = QdrantStore(
                    collection_name=collection_name,
                    url=qdrant_url,
                    embedding_dim=768
                )
            except Exception as e:
                logger.error(f"Failed to connect to Qdrant: {e}")
                self._qdrant = None

    async def encode_query(
        self,
        query: Union[str, Path, np.ndarray],
        modality: Modality,
        **kwargs
    ) -> ModalEmbedding:
        """
        Encode query in specified modality.

        Args:
            query: Query content (text, file path, array)
            modality: Query modality
            **kwargs: Encoder-specific options

        Returns:
            ModalEmbedding with 768D vector

        Raises:
            RuntimeError: If encoder for modality unavailable
        """
        if modality == Modality.TEXT:
            if self._text_encoder is None:
                raise RuntimeError("Text encoder unavailable")

            # Encode text
            emb_matrix = self._text_encoder.encode_scales(
                [str(query)],
                size=768
            )
            embedding = emb_matrix[0].tolist()

            return ModalEmbedding(
                embedding=embedding,
                modality=Modality.TEXT,
                metadata={"query": str(query)},
                confidence=1.0
            )

        elif modality == Modality.IMAGE:
            if self._image_encoder is None:
                raise RuntimeError("Image encoder unavailable")
            return await self._image_encoder.encode(query, **kwargs)

        elif modality == Modality.AUDIO:
            if self._audio_encoder is None:
                raise RuntimeError("Audio encoder unavailable")
            return await self._audio_encoder.encode(query, **kwargs)

        elif modality == Modality.VIDEO:
            if self._video_encoder is None:
                raise RuntimeError("Video encoder unavailable")
            return await self._video_encoder.encode(query, **kwargs)

        else:
            raise ValueError(f"Unknown modality: {modality}")

    async def search(
        self,
        query: Union[str, Path, np.ndarray, List[float]],
        query_modality: Optional[Modality] = None,
        target_modalities: Optional[List[Modality]] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> SearchResults:
        """
        Search for similar content across modalities.

        Args:
            query: Query content (text, file path, array, or embedding)
            query_modality: Modality of query (required if query not embedding)
            target_modalities: Modalities to search (None = all)
            top_k: Number of results to return
            filters: Optional metadata filters
            **kwargs: Additional search options

        Returns:
            SearchResults with ranked matches

        Example:
            # Text → Image search
            results = await search.search(
                query="cats playing",
                query_modality=Modality.TEXT,
                target_modalities=[Modality.IMAGE]
            )

            # Image → similar content search
            results = await search.search(
                query="cat.jpg",
                query_modality=Modality.IMAGE,
                target_modalities=None  # Search all modalities
            )
        """
        # Encode query if not already an embedding
        if isinstance(query, list) and len(query) == 768:
            # Already an embedding
            query_embedding = query
            query_mod = query_modality or Modality.TEXT
        else:
            if query_modality is None:
                raise ValueError("query_modality required when query is not embedding")

            # Encode query
            query_emb = await self.encode_query(query, query_modality, **kwargs)
            query_embedding = query_emb.embedding
            query_mod = query_modality

        # Search in Qdrant
        if self._qdrant is None:
            logger.warning("Qdrant unavailable - returning empty results")
            return SearchResults(
                query_modality=query_mod,
                target_modalities=target_modalities or []
            )

        # Build filter for target modalities
        search_filter = filters or {}
        if target_modalities:
            search_filter["modality"] = [m.value for m in target_modalities]

        # Perform search
        try:
            hits = await self._qdrant.search(
                query_vector=query_embedding,
                top_k=top_k,
                filter_dict=search_filter
            )

            # Convert to SearchResults
            results = []
            for hit in hits:
                # Extract modality from payload
                modality_str = hit.payload.get("modality", "text")
                modality = Modality(modality_str)

                # Check if cross-modal
                cross_modal = (modality != query_mod)

                result = SearchResult(
                    id=str(hit.id),
                    modality=modality,
                    embedding=hit.vector if hasattr(hit, 'vector') else [],
                    metadata=hit.payload,
                    score=float(hit.score),
                    cross_modal=cross_modal
                )
                results.append(result)

            return SearchResults(
                results=results,
                query_modality=query_mod,
                target_modalities=target_modalities or [],
                total_results=len(results)
            )

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return SearchResults(
                query_modality=query_mod,
                target_modalities=target_modalities or []
            )

    async def index_content(
        self,
        content: Union[str, Path],
        modality: Modality,
        content_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> bool:
        """
        Index content in the multi-modal store.

        Args:
            content: Content to index (text, file path)
            modality: Content modality
            content_id: Optional ID (auto-generated if None)
            metadata: Optional metadata to store
            **kwargs: Encoder options

        Returns:
            True if successful

        Example:
            # Index image
            await search.index_content(
                content="cat.jpg",
                modality=Modality.IMAGE,
                metadata={"tags": ["animal", "pet"]}
            )

            # Index text
            await search.index_content(
                content="Machine learning is...",
                modality=Modality.TEXT,
                metadata={"source": "textbook"}
            )
        """
        if self._qdrant is None:
            logger.error("Qdrant unavailable - cannot index")
            return False

        try:
            # Encode content
            embedding = await self.encode_query(content, modality, **kwargs)

            # Generate ID if not provided
            if content_id is None:
                import uuid
                content_id = str(uuid.uuid4())

            # Prepare payload
            payload = metadata or {}
            payload.update({
                "modality": modality.value,
                "content": str(content) if isinstance(content, (str, Path)) else "binary",
                **embedding.metadata
            })

            # Store in Qdrant
            await self._qdrant.upsert(
                points=[{
                    "id": content_id,
                    "vector": embedding.embedding,
                    "payload": payload
                }]
            )

            logger.info(f"Indexed {modality.value} content: {content_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to index content: {e}")
            return False

    async def hybrid_search(
        self,
        text_query: str,
        image_query: Optional[Union[str, Path]] = None,
        audio_query: Optional[Union[str, Path]] = None,
        weights: Optional[Dict[Modality, float]] = None,
        top_k: int = 10,
        **kwargs
    ) -> SearchResults:
        """
        Hybrid search with multiple query modalities.

        Encodes multiple query modalities, fuses them, and searches.

        Args:
            text_query: Text query (required)
            image_query: Optional image query
            audio_query: Optional audio query
            weights: Optional modal weights for fusion
            top_k: Number of results
            **kwargs: Additional options

        Returns:
            SearchResults with fused query

        Example:
            results = await search.hybrid_search(
                text_query="cats playing",
                image_query="cat.jpg",
                weights={Modality.TEXT: 0.6, Modality.IMAGE: 0.4}
            )
        """
        # Encode all query modalities
        query_embeddings = {}

        # Text (required)
        if self._text_encoder:
            text_emb = await self.encode_query(text_query, Modality.TEXT)
            query_embeddings[Modality.TEXT] = text_emb.embedding

        # Image (optional)
        if image_query and self._image_encoder:
            image_emb = await self.encode_query(image_query, Modality.IMAGE)
            query_embeddings[Modality.IMAGE] = image_emb.embedding

        # Audio (optional)
        if audio_query and self._audio_encoder:
            audio_emb = await self.encode_query(audio_query, Modality.AUDIO)
            query_embeddings[Modality.AUDIO] = audio_emb.embedding

        # Fuse query embeddings
        if self._fuser is None:
            # Fallback: simple average
            fused = np.mean(
                [np.array(emb) for emb in query_embeddings.values()],
                axis=0
            ).tolist()
        else:
            fused_result = self._fuser.fuse_embeddings(
                query_embeddings,
                weights=weights
            )
            fused = fused_result.fused_embedding

        # Search with fused query
        results = await self.search(
            query=fused,
            query_modality=None,  # Multi-modal query
            top_k=top_k,
            **kwargs
        )

        # Add fusion metadata
        results.fusion_method = self.fusion_method
        results.query_modality = None  # Indicate multi-modal query

        return results


# ============================================================================
# Factory Function
# ============================================================================

def create_search_engine(
    collection_name: str = "hololoom_multimodal",
    qdrant_url: str = "http://localhost:6333",
    fusion_method: str = "late",
    **kwargs
) -> MultiModalSearch:
    """
    Factory function to create multi-modal search engine.

    Args:
        collection_name: Qdrant collection name
        qdrant_url: Qdrant server URL
        fusion_method: Fusion method for multi-modal queries
        **kwargs: Additional options

    Returns:
        MultiModalSearch instance
    """
    return MultiModalSearch(
        collection_name=collection_name,
        qdrant_url=qdrant_url,
        fusion_method=fusion_method
    )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import asyncio

    async def demo():
        print("=" * 80)
        print("Multi-Modal Search Demo")
        print("=" * 80 + "\n")

        # Create search engine
        search = create_search_engine(
            collection_name="demo_collection",
            fusion_method="late"
        )

        print(f"Collection: {search.collection_name}")
        print(f"Fusion method: {search.fusion_method}")
        print(f"Qdrant available: {_HAVE_QDRANT}")
        print(f"Encoders available: {_HAVE_ENCODERS}")

        print("\n✓ Multi-modal search ready!")
        print("\nExample usage:")
        print("  # Text → Image search")
        print("  results = await search.search(")
        print("      query='cats playing',")
        print("      query_modality=Modality.TEXT,")
        print("      target_modalities=[Modality.IMAGE]")
        print("  )")
        print("\n  # Hybrid search")
        print("  results = await search.hybrid_search(")
        print("      text_query='machine learning',")
        print("      image_query='diagram.jpg'")
        print("  )")

    asyncio.run(demo())
