"""
Custom Embedding Plugins for HoloLoom RAG
=========================================

Plugin architecture enabling users to swap embedding models (HuggingFace, OpenAI, Cohere, custom).

Protocols:
    - EmbeddingProvider: Protocol for custom embedding models

Built-in Providers:
    - MatryoshkaEmbedding: Default (96/192/384 dims)
    - HuggingFaceEmbedding: Any Sentence Transformer model
    - OpenAIEmbedding: OpenAI's text-embedding-3-small/large
    - CohereEmbedding: Cohere's embed-english-v3.0

Key Features:
    - Protocol-based design (extensible)
    - Graceful degradation (fallback to Matryoshka)
    - Type validation on initialization
    - Zero-copy compatibility detection
"""

import logging
from typing import List, Protocol, Optional, runtime_checkable
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Protocol - Abstract Interface
# ============================================================================

@runtime_checkable
class EmbeddingProvider(Protocol):
    """
    Protocol for custom embedding models.

    Any embedding provider must implement:
    - dimension property: Embedding dimension
    - encode(): Encode texts to embeddings
    - encode_query(): Encode a single query (may differ from documents)
    """

    @property
    def dimension(self) -> int:
        """
        Embedding dimension.

        Returns:
            int: Number of dimensions in embedding vector
        """
        ...

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: List of strings to encode

        Returns:
            np.ndarray of shape (len(texts), dimension)
        """
        ...

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query.

        May differ from document encoding (e.g., different prompt for queries).

        Args:
            query: Query string

        Returns:
            np.ndarray of shape (dimension,)
        """
        ...


# ============================================================================
# Built-in Providers
# ============================================================================

class MatryoshkaEmbedding:
    """
    Default Matryoshka embeddings (96/192/384 dims).

    Wraps HoloLoom's existing spectral.py implementation.
    Matryoshka embeddings enable multi-scale retrieval with zero-copy optimization.

    Properties:
        - Zero-copy compatible (prefix property)
        - Fast (uses cached embeddings)
        - Multi-scale (can query at 96, 192, or 384 dims)
    """

    def __init__(self):
        """Initialize Matryoshka embeddings."""
        # Lazy import to avoid circular dependencies
        try:
            from hololoom.embedding.spectral import MatryoshkaEmbeddings
            self._embedder = MatryoshkaEmbeddings(
                scales=[96, 192, 384],
                enable_spectral=False  # Just use Matryoshka, not spectral features
            )
            self.dimension = 384  # Full dimension
            logger.info("✓ MatryoshkaEmbedding initialized (384 dims)")
        except Exception as e:
            logger.error(f"Failed to initialize MatryoshkaEmbedding: {e}")
            raise

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to Matryoshka embeddings.

        Args:
            texts: List of strings

        Returns:
            np.ndarray of shape (len(texts), 384)
        """
        embeddings = self._embedder.encode(texts)
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a query (same as document encoding for Matryoshka).

        Args:
            query: Query string

        Returns:
            np.ndarray of shape (384,)
        """
        embeddings = self.encode([query])
        return embeddings[0]


class HuggingFaceEmbedding:
    """
    HuggingFace Sentence Transformer embeddings.

    Allows using any Sentence Transformer model:
    - "all-MiniLM-L6-v2": 384 dims, fast
    - "all-mpnet-base-v2": 768 dims, higher quality
    - "multilingual-e5-large": 1024 dims, multilingual

    Args:
        model_name: HuggingFace model identifier (defaults to MiniLM)

    Example:
        embedding = HuggingFaceEmbedding("all-mpnet-base-v2")
        embeddings = embedding.encode(["hello", "world"])
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize HuggingFace Sentence Transformer.

        Args:
            model_name: Model identifier (defaults to lightweight MiniLM)

        Raises:
            ImportError: If sentence-transformers not installed
            Exception: If model download fails
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            self.model_name = model_name
            logger.info(f"✓ HuggingFaceEmbedding initialized ({model_name}, {self.dimension} dims)")
        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFaceEmbedding: {e}")
            raise

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: List of strings

        Returns:
            np.ndarray of shape (len(texts), dimension)
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return np.asarray(embeddings, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query.

        Args:
            query: Query string

        Returns:
            np.ndarray of shape (dimension,)
        """
        embeddings = self.encode([query])
        return embeddings[0]


class OpenAIEmbedding:
    """
    OpenAI text-embedding-3-small/large.

    Models:
    - text-embedding-3-small: 1536 dims, $0.02 per 1M tokens (fast, cheap)
    - text-embedding-3-large: 3072 dims, $0.13 per 1M tokens (higher quality)

    Args:
        model: Model name ("text-embedding-3-small" or "text-embedding-3-large")
        api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)

    Example:
        embedding = OpenAIEmbedding("text-embedding-3-small")
        embeddings = embedding.encode(["hello", "world"])

    Note:
        Requires OpenAI API key and internet connection.
        Consider costs: ~$0.02-$0.13 per 1M tokens.
    """

    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        """
        Initialize OpenAI Embedding.

        Args:
            model: Model name
            api_key: OpenAI API key (uses OPENAI_API_KEY env if not provided)

        Raises:
            ImportError: If openai not installed
            ValueError: If API key not found
        """
        try:
            import openai
            if api_key:
                openai.api_key = api_key
            self.client = openai.Client()
            self.model = model

            # Set dimension based on model
            if model == "text-embedding-3-small":
                self.dimension = 1536
            elif model == "text-embedding-3-large":
                self.dimension = 3072
            else:
                # Try to query model info, fallback to 1536
                logger.warning(f"Unknown model {model}, assuming 1536 dims")
                self.dimension = 1536

            logger.info(f"✓ OpenAIEmbedding initialized ({model}, {self.dimension} dims)")
        except ImportError:
            logger.error(
                "openai not installed. "
                "Install with: pip install openai"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize OpenAIEmbedding: {e}")
            raise

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to embeddings via OpenAI API.

        Args:
            texts: List of strings

        Returns:
            np.ndarray of shape (len(texts), dimension)

        Note:
            Makes API call - costs money and requires internet.
        """
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model
            )
            embeddings = np.array([e.embedding for e in response.data], dtype=np.float32)
            logger.debug(f"Encoded {len(texts)} texts via OpenAI API")
            return embeddings
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query.

        Args:
            query: Query string

        Returns:
            np.ndarray of shape (dimension,)
        """
        embeddings = self.encode([query])
        return embeddings[0]


class CohereEmbedding:
    """
    Cohere embeddings.

    Models:
    - embed-english-v3.0: 1024 dims, $0.10 per 1M tokens
    - embed-english-v2.0: 4096 dims, slightly lower cost

    Args:
        model: Model name (defaults to v3.0)
        api_key: Cohere API key (defaults to COHERE_API_KEY env var)

    Example:
        embedding = CohereEmbedding("embed-english-v3.0")
        embeddings = embedding.encode(["hello", "world"])

    Note:
        Requires Cohere API key and internet connection.
    """

    def __init__(self, model: str = "embed-english-v3.0", api_key: Optional[str] = None):
        """
        Initialize Cohere Embedding.

        Args:
            model: Model name
            api_key: Cohere API key (uses COHERE_API_KEY env if not provided)

        Raises:
            ImportError: If cohere not installed
        """
        try:
            import cohere
            self.client = cohere.Client(api_key=api_key) if api_key else cohere.Client()
            self.model = model

            # Set dimension based on model
            if "v3" in model:
                self.dimension = 1024
            elif "v2" in model:
                self.dimension = 4096
            else:
                logger.warning(f"Unknown Cohere model {model}, assuming 1024 dims")
                self.dimension = 1024

            logger.info(f"✓ CohereEmbedding initialized ({model}, {self.dimension} dims)")
        except ImportError:
            logger.error(
                "cohere not installed. "
                "Install with: pip install cohere"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize CohereEmbedding: {e}")
            raise

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to embeddings via Cohere API.

        Args:
            texts: List of strings

        Returns:
            np.ndarray of shape (len(texts), dimension)

        Note:
            Makes API call - costs money and requires internet.
        """
        try:
            response = self.client.embed(
                texts=texts,
                model=self.model,
                input_type="search_document"
            )
            embeddings = np.array(response.embeddings, dtype=np.float32)
            logger.debug(f"Encoded {len(texts)} texts via Cohere API")
            return embeddings
        except Exception as e:
            logger.error(f"Cohere API call failed: {e}")
            raise

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query with search_query input type.

        Cohere distinguishes between search_document and search_query.

        Args:
            query: Query string

        Returns:
            np.ndarray of shape (dimension,)
        """
        try:
            response = self.client.embed(
                texts=[query],
                model=self.model,
                input_type="search_query"  # Different from documents!
            )
            return np.array(response.embeddings[0], dtype=np.float32)
        except Exception as e:
            logger.error(f"Cohere query embedding failed: {e}")
            raise


# ============================================================================
# Validation & Factory
# ============================================================================

def validate_embedding_provider(provider: EmbeddingProvider) -> bool:
    """
    Validate that a provider implements the EmbeddingProvider protocol.

    Tests:
    1. Has dimension property (int)
    2. Has encode() method (returns correct shape)
    3. Has encode_query() method (returns correct shape)

    Args:
        provider: Object to validate

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        # Check dimension
        if not hasattr(provider, 'dimension'):
            logger.error(f"Provider missing 'dimension' attribute: {type(provider)}")
            return False

        if not isinstance(provider.dimension, int) or provider.dimension <= 0:
            logger.error(f"Provider dimension invalid: {provider.dimension}")
            return False

        # Check encode method
        if not hasattr(provider, 'encode') or not callable(provider.encode):
            logger.error(f"Provider missing 'encode' method: {type(provider)}")
            return False

        # Check encode_query method
        if not hasattr(provider, 'encode_query') or not callable(provider.encode_query):
            logger.error(f"Provider missing 'encode_query' method: {type(provider)}")
            return False

        # Test encode with sample texts
        test_texts = ["test1", "test2"]
        embeddings = provider.encode(test_texts)

        if not isinstance(embeddings, np.ndarray):
            logger.error(f"Provider encode() should return np.ndarray, got {type(embeddings)}")
            return False

        if embeddings.shape != (2, provider.dimension):
            logger.error(
                f"Provider encode() shape incorrect. "
                f"Expected (2, {provider.dimension}), got {embeddings.shape}"
            )
            return False

        # Test encode_query with sample query
        query_embedding = provider.encode_query("test query")

        if not isinstance(query_embedding, np.ndarray):
            logger.error(f"Provider encode_query() should return np.ndarray, got {type(query_embedding)}")
            return False

        if query_embedding.shape != (provider.dimension,):
            logger.error(
                f"Provider encode_query() shape incorrect. "
                f"Expected ({provider.dimension},), got {query_embedding.shape}"
            )
            return False

        logger.debug(f"✓ Provider validation passed: {type(provider).__name__}")
        return True

    except Exception as e:
        logger.error(f"Provider validation failed with exception: {e}")
        return False


def create_embedding_provider(
    provider_type: str = "matryoshka",
    **kwargs
) -> EmbeddingProvider:
    """
    Factory function to create embedding providers by type.

    Args:
        provider_type: Type of provider
            - "matryoshka": Default (Matryoshka, 384 dims)
            - "huggingface": HuggingFace Sentence Transformer
            - "openai": OpenAI embeddings
            - "cohere": Cohere embeddings
        **kwargs: Provider-specific arguments

    Returns:
        EmbeddingProvider instance

    Example:
        provider = create_embedding_provider("huggingface", model_name="all-mpnet-base-v2")
    """
    provider_type = provider_type.lower().strip()

    if provider_type == "matryoshka":
        return MatryoshkaEmbedding()
    elif provider_type == "huggingface":
        return HuggingFaceEmbedding(**kwargs)
    elif provider_type == "openai":
        return OpenAIEmbedding(**kwargs)
    elif provider_type == "cohere":
        return CohereEmbedding(**kwargs)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'EmbeddingProvider',
    'MatryoshkaEmbedding',
    'HuggingFaceEmbedding',
    'OpenAIEmbedding',
    'CohereEmbedding',
    'validate_embedding_provider',
    'create_embedding_provider',
]
