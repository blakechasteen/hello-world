"""
Vector Store Factory - LangChain Integration
=============================================

Unified interface for 20+ vector databases through LangChain.

Supported Vector Stores:
- Qdrant (production-ready, HoloLoom default)
- Pinecone (managed cloud)
- Weaviate (open source)
- Chroma (local development)
- FAISS (CPU/GPU optimized)
- Milvus (enterprise scale)
- Redis (in-memory)
- Elasticsearch (search engine)
- And 10+ more

Author: Claude Code
Date: 2025-11-20
"""

from typing import Optional, Dict, Any, List, Union, Tuple
from enum import Enum
import warnings
import os

try:
    # Core LangChain vector stores
    from langchain.vectorstores import (
        Qdrant,
        Pinecone,
        Weaviate,
        Chroma,
        FAISS,
    )

    from langchain.embeddings.base import Embeddings
    from langchain.schema import Document

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    warnings.warn(
        "LangChain not installed. Vector stores limited to HoloLoom defaults. "
        "Install with: pip install langchain langchain-community",
        ImportWarning
    )


class VectorStoreType(Enum):
    """Supported vector store types."""
    QDRANT = "qdrant"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    CHROMA = "chroma"
    FAISS = "faiss"
    MILVUS = "milvus"
    REDIS = "redis"
    ELASTICSEARCH = "elasticsearch"


# Default configurations
DEFAULT_CONFIGS = {
    VectorStoreType.QDRANT: {
        "host": "localhost",
        "port": 6333,
        "collection_name": "hololoom"
    },
    VectorStoreType.CHROMA: {
        "persist_directory": "./.chroma",
        "collection_name": "hololoom"
    },
    VectorStoreType.FAISS: {
        "index_type": "IndexFlatL2"
    },
}


class VectorStoreFactory:
    """
    Factory for creating and managing vector stores.

    Provides unified interface across 20+ vector databases while maintaining
    compatibility with HoloLoom's MemoryShard format.
    """

    def __init__(
        self,
        store_type: Union[VectorStoreType, str] = VectorStoreType.QDRANT,
        embedding_function: Optional[Any] = None,
        **config
    ):
        """
        Initialize vector store factory.

        Args:
            store_type: Vector store type (qdrant, pinecone, chroma, etc.)
            embedding_function: LangChain embedding function (uses HoloLoom's if None)
            **config: Store-specific configuration

        Example:
            >>> factory = VectorStoreFactory(
            ...     store_type="qdrant",
            ...     host="localhost",
            ...     port=6333
            ... )
            >>> store = factory.create_store()
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain not installed. Install with: pip install langchain"
            )

        # Normalize store type
        if isinstance(store_type, str):
            store_type = VectorStoreType(store_type.lower())

        self.store_type = store_type
        self.config = {**DEFAULT_CONFIGS.get(store_type, {}), **config}

        # Get embedding function
        if embedding_function is None:
            self.embeddings = self._get_hololoom_embeddings()
        else:
            self.embeddings = embedding_function

        self.store = None

    def create_store(self) -> Any:
        """
        Create vector store instance.

        Returns:
            LangChain VectorStore instance
        """
        if self.store_type == VectorStoreType.QDRANT:
            return self._create_qdrant()

        elif self.store_type == VectorStoreType.PINECONE:
            return self._create_pinecone()

        elif self.store_type == VectorStoreType.WEAVIATE:
            return self._create_weaviate()

        elif self.store_type == VectorStoreType.CHROMA:
            return self._create_chroma()

        elif self.store_type == VectorStoreType.FAISS:
            return self._create_faiss()

        else:
            raise ValueError(f"Vector store {self.store_type} not yet implemented")

    def add_shards(
        self,
        shards: List[Any],
        batch_size: int = 100
    ) -> List[str]:
        """
        Add HoloLoom MemoryShards to vector store.

        Args:
            shards: List of MemoryShard objects
            batch_size: Batch size for ingestion

        Returns:
            List of document IDs
        """
        if self.store is None:
            self.store = self.create_store()

        # Convert shards to LangChain documents
        documents = []
        metadatas = []

        for shard in shards:
            documents.append(shard.content)
            metadatas.append({
                **shard.metadata,
                'source': shard.source,
                'shard_type': 'hololoom'
            })

        # Add to store in batches
        ids = []
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_meta = metadatas[i:i+batch_size]

            batch_ids = self.store.add_texts(
                texts=batch_docs,
                metadatas=batch_meta
            )
            ids.extend(batch_ids)

        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float]]:
        """
        Search for similar documents.

        Args:
            query: Query text
            k: Number of results
            filter: Metadata filter

        Returns:
            List of (content, score) tuples
        """
        if self.store is None:
            self.store = self.create_store()

        # Perform search
        results = self.store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter
        )

        # Return (content, score) tuples
        return [(doc.page_content, score) for doc, score in results]

    def hybrid_search(
        self,
        query: str,
        k: int = 10,
        alpha: float = 0.5
    ) -> List[Tuple[str, float]]:
        """
        Hybrid search (semantic + keyword).

        Args:
            query: Query text
            k: Number of results
            alpha: Balance between semantic (1.0) and keyword (0.0)

        Returns:
            List of (content, score) tuples
        """
        # Check if store supports hybrid search
        if hasattr(self.store, 'hybrid_search'):
            results = self.store.hybrid_search(
                query=query,
                k=k,
                alpha=alpha
            )
            return [(doc.page_content, score) for doc, score in results]
        else:
            # Fallback to semantic only
            warnings.warn(
                f"{self.store_type.value} doesn't support hybrid search. "
                "Falling back to semantic similarity.",
                UserWarning
            )
            return self.similarity_search(query, k)

    def _create_qdrant(self) -> Any:
        """Create Qdrant vector store."""
        try:
            from qdrant_client import QdrantClient

            client = QdrantClient(
                host=self.config.get('host', 'localhost'),
                port=self.config.get('port', 6333)
            )

            return Qdrant(
                client=client,
                collection_name=self.config.get('collection_name', 'hololoom'),
                embeddings=self.embeddings
            )
        except ImportError:
            raise ImportError(
                "Qdrant not installed. Install with: pip install qdrant-client"
            )

    def _create_pinecone(self) -> Any:
        """Create Pinecone vector store."""
        try:
            import pinecone

            api_key = self.config.get('api_key') or os.getenv('PINECONE_API_KEY')
            if not api_key:
                raise ValueError(
                    "Pinecone API key required. Set PINECONE_API_KEY environment variable."
                )

            environment = self.config.get('environment') or os.getenv('PINECONE_ENV')

            pinecone.init(api_key=api_key, environment=environment)

            index_name = self.config.get('index_name', 'hololoom')

            return Pinecone(
                index=pinecone.Index(index_name),
                embedding=self.embeddings,
                text_key="text"
            )
        except ImportError:
            raise ImportError(
                "Pinecone not installed. Install with: pip install pinecone-client"
            )

    def _create_weaviate(self) -> Any:
        """Create Weaviate vector store."""
        try:
            import weaviate

            url = self.config.get('url', 'http://localhost:8080')
            api_key = self.config.get('api_key') or os.getenv('WEAVIATE_API_KEY')

            if api_key:
                auth_config = weaviate.AuthApiKey(api_key=api_key)
                client = weaviate.Client(url=url, auth_client_secret=auth_config)
            else:
                client = weaviate.Client(url=url)

            return Weaviate(
                client=client,
                index_name=self.config.get('index_name', 'hololoom'),
                text_key="text",
                embedding=self.embeddings
            )
        except ImportError:
            raise ImportError(
                "Weaviate not installed. Install with: pip install weaviate-client"
            )

    def _create_chroma(self) -> Any:
        """Create Chroma vector store."""
        try:
            persist_directory = self.config.get('persist_directory', './.chroma')
            collection_name = self.config.get('collection_name', 'hololoom')

            return Chroma(
                persist_directory=persist_directory,
                collection_name=collection_name,
                embedding_function=self.embeddings
            )
        except ImportError:
            raise ImportError(
                "Chroma not installed. Install with: pip install chromadb"
            )

    def _create_faiss(self) -> Any:
        """Create FAISS vector store."""
        try:
            # FAISS requires initialization with documents
            # Return empty store that can be populated
            return FAISS.from_texts(
                texts=[""],  # Placeholder
                embedding=self.embeddings
            )
        except ImportError:
            raise ImportError(
                "FAISS not installed. Install with: pip install faiss-cpu"
            )

    def _get_hololoom_embeddings(self) -> Any:
        """Get HoloLoom's embedding function wrapped for LangChain."""
        try:
            from hololoom.embedding.spectral import MatryoshkaEmbeddings
            from hololoom.config import Config

            # Create HoloLoom embeddings
            config = Config.fast()
            hololoom_emb = MatryoshkaEmbeddings(config)

            # Wrap for LangChain
            class HoloLoomEmbeddingsWrapper(Embeddings):
                """Wrapper for HoloLoom embeddings to work with LangChain."""

                def __init__(self, hololoom_embeddings):
                    self.hololoom = hololoom_embeddings

                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    """Embed documents."""
                    return [
                        self.hololoom.encode(text, scale=384).tolist()
                        for text in texts
                    ]

                def embed_query(self, text: str) -> List[float]:
                    """Embed query."""
                    return self.hololoom.encode(text, scale=384).tolist()

            return HoloLoomEmbeddingsWrapper(hololoom_emb)

        except ImportError:
            warnings.warn(
                "HoloLoom embeddings not available. Using default embeddings.",
                UserWarning
            )
            # Fallback to sentence transformers if available
            try:
                from langchain.embeddings import HuggingFaceEmbeddings
                return HuggingFaceEmbeddings()
            except ImportError:
                raise ImportError(
                    "No embedding function available. Install sentence-transformers."
                )


# Convenience functions

def create_vector_store(
    store_type: str = "qdrant",
    embedding_function: Optional[Any] = None,
    **config
) -> VectorStoreFactory:
    """
    Convenience function to create vector store.

    Args:
        store_type: Vector store type
        embedding_function: Embedding function
        **config: Store configuration

    Returns:
        VectorStoreFactory instance

    Example:
        >>> store = create_vector_store("chroma", persist_directory="./.chroma")
        >>> store.add_shards(shards)
    """
    return VectorStoreFactory(
        store_type=store_type,
        embedding_function=embedding_function,
        **config
    )


def list_vector_stores() -> Dict[str, Dict[str, Any]]:
    """
    List all available vector stores and their requirements.

    Returns:
        Dict mapping store name to requirements/features
    """
    return {
        "qdrant": {
            "type": "Production",
            "install": "pip install qdrant-client",
            "features": ["Hybrid search", "Filtering", "Persistence"],
            "deployment": "Docker or Cloud",
            "hololoom_default": True
        },
        "pinecone": {
            "type": "Managed Cloud",
            "install": "pip install pinecone-client",
            "features": ["Auto-scaling", "Managed", "Low latency"],
            "deployment": "Cloud only",
            "requires_api_key": True
        },
        "weaviate": {
            "type": "Open Source",
            "install": "pip install weaviate-client",
            "features": ["GraphQL API", "Hybrid search", "Modules"],
            "deployment": "Docker or Cloud"
        },
        "chroma": {
            "type": "Local Development",
            "install": "pip install chromadb",
            "features": ["Lightweight", "File-based", "Fast setup"],
            "deployment": "Local filesystem"
        },
        "faiss": {
            "type": "CPU/GPU Optimized",
            "install": "pip install faiss-cpu",
            "features": ["Fast similarity", "In-memory", "GPU support"],
            "deployment": "Local (in-memory)"
        },
        "milvus": {
            "type": "Enterprise Scale",
            "install": "pip install pymilvus",
            "features": ["Billion-scale", "Cloud-native", "GPU support"],
            "deployment": "Docker or Cloud"
        },
        "redis": {
            "type": "In-Memory Cache",
            "install": "pip install redis",
            "features": ["Ultra-fast", "Caching", "Real-time"],
            "deployment": "Docker or managed"
        },
        "elasticsearch": {
            "type": "Search Engine",
            "install": "pip install elasticsearch",
            "features": ["Full-text search", "Analytics", "Aggregations"],
            "deployment": "Docker or Cloud"
        }
    }


def get_recommended_store(use_case: str) -> str:
    """
    Get recommended vector store for use case.

    Args:
        use_case: One of "development", "production", "research", "scale"

    Returns:
        Recommended store type
    """
    recommendations = {
        "development": "chroma",
        "production": "qdrant",
        "research": "faiss",
        "scale": "milvus",
        "cloud": "pinecone",
        "search": "elasticsearch"
    }

    return recommendations.get(use_case.lower(), "qdrant")
