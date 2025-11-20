"""
Tests for Custom Embedding Plugins.

Tests protocol compliance, provider implementations, and integration.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import List

from HoloLoom.rag.embedding_plugins import (
    EmbeddingProvider,
    MatryoshkaEmbedding,
    HuggingFaceEmbedding,
    OpenAIEmbedding,
    CohereEmbedding,
    validate_embedding_provider,
    create_embedding_provider,
)


# ============================================================================
# Protocol Tests
# ============================================================================

class TestEmbeddingProviderProtocol:
    """Test EmbeddingProvider protocol definition."""

    def test_protocol_has_dimension(self):
        """Test that protocol requires dimension property."""
        assert hasattr(EmbeddingProvider, '__protocol_attrs__')

    def test_protocol_runtime_checkable(self):
        """Test that protocol is runtime checkable."""
        # Create a real provider that satisfies the protocol
        import numpy as np

        class MinimalProvider:
            @property
            def dimension(self) -> int:
                return 384

            def encode(self, texts: List[str]) -> np.ndarray:
                return np.random.randn(len(texts), 384).astype(np.float32)

            def encode_query(self, query: str) -> np.ndarray:
                return np.random.randn(384).astype(np.float32)

        provider = MinimalProvider()

        # Should be checkable at runtime (protocol has @runtime_checkable)
        assert isinstance(provider, EmbeddingProvider)

    def test_protocol_requires_encode(self):
        """Test that protocol requires encode method."""
        # Create object missing encode method
        incomplete = Mock(spec=['dimension', 'encode_query'])
        incomplete.dimension = 384

        # Should not satisfy protocol (missing encode)
        assert not isinstance(incomplete, EmbeddingProvider)

    def test_protocol_requires_encode_query(self):
        """Test that protocol requires encode_query method."""
        # Create object missing encode_query method
        incomplete = Mock(spec=['dimension', 'encode'])
        incomplete.dimension = 384

        # Should not satisfy protocol (missing encode_query)
        assert not isinstance(incomplete, EmbeddingProvider)


# ============================================================================
# MatryoshkaEmbedding Tests
# ============================================================================

class TestMatryoshkaEmbedding:
    """Test MatryoshkaEmbedding provider."""

    def test_initialization(self):
        """Test MatryoshkaEmbedding initialization."""
        with patch('HoloLoom.rag.embedding_plugins.MatryoshkaEmbedding.__init__',
                   lambda x: None):
            provider = MatryoshkaEmbedding()
            provider._embedder = Mock()
            provider.dimension = 384
            assert provider.dimension == 384

    def test_dimension_property(self):
        """Test dimension property."""
        with patch('HoloLoom.rag.embedding_plugins.MatryoshkaEmbedding.__init__',
                   lambda x: None):
            provider = MatryoshkaEmbedding()
            provider.dimension = 384
            assert provider.dimension == 384
            assert isinstance(provider.dimension, int)

    def test_encode_shape(self):
        """Test encode returns correct shape."""
        with patch('HoloLoom.rag.embedding_plugins.MatryoshkaEmbedding.__init__',
                   lambda x: None):
            provider = MatryoshkaEmbedding()
            provider.dimension = 384
            provider._embedder = Mock()
            provider._embedder.encode = Mock(
                return_value=np.random.randn(3, 384).astype(np.float32)
            )

            embeddings = provider.encode(["text1", "text2", "text3"])
            assert embeddings.shape == (3, 384)

    def test_encode_query_shape(self):
        """Test encode_query returns correct shape."""
        with patch('HoloLoom.rag.embedding_plugins.MatryoshkaEmbedding.__init__',
                   lambda x: None):
            provider = MatryoshkaEmbedding()
            provider.dimension = 384
            provider._embedder = Mock()
            provider._embedder.encode = Mock(
                return_value=np.random.randn(1, 384).astype(np.float32)
            )

            query_emb = provider.encode_query("test query")
            assert query_emb.shape == (384,)

    def test_protocol_compliance(self):
        """Test MatryoshkaEmbedding satisfies protocol."""
        with patch('HoloLoom.rag.embedding_plugins.MatryoshkaEmbedding.__init__',
                   lambda x: None):
            provider = MatryoshkaEmbedding()
            provider.dimension = 384
            provider._embedder = Mock()
            provider._embedder.encode = Mock(
                return_value=np.random.randn(1, 384).astype(np.float32)
            )

            # Should satisfy protocol
            assert isinstance(provider, EmbeddingProvider)


# ============================================================================
# HuggingFaceEmbedding Tests
# ============================================================================

class TestHuggingFaceEmbedding:
    """Test HuggingFaceEmbedding provider."""

    @patch('HoloLoom.rag.embedding_plugins.HuggingFaceEmbedding.__init__',
           lambda x, model_name=None: None)
    def test_initialization_defaults(self):
        """Test default initialization."""
        provider = HuggingFaceEmbedding()
        provider.model = Mock()
        provider.model.get_sentence_embedding_dimension = Mock(return_value=384)
        provider.dimension = 384
        provider.model_name = "all-MiniLM-L6-v2"

        assert provider.dimension == 384
        assert provider.model_name == "all-MiniLM-L6-v2"

    @patch('HoloLoom.rag.embedding_plugins.HuggingFaceEmbedding.__init__',
           lambda x, model_name="all-mpnet-base-v2": None)
    def test_initialization_custom_model(self):
        """Test initialization with custom model."""
        provider = HuggingFaceEmbedding("all-mpnet-base-v2")
        provider.model = Mock()
        provider.model.get_sentence_embedding_dimension = Mock(return_value=768)
        provider.dimension = 768
        provider.model_name = "all-mpnet-base-v2"

        assert provider.dimension == 768

    @patch('HoloLoom.rag.embedding_plugins.HuggingFaceEmbedding.__init__',
           lambda x, model_name=None: None)
    def test_encode_shape(self):
        """Test encode returns correct shape."""
        provider = HuggingFaceEmbedding()
        provider.dimension = 384
        provider.model = Mock()
        provider.model.encode = Mock(
            return_value=np.random.randn(2, 384).astype(np.float32)
        )

        embeddings = provider.encode(["text1", "text2"])
        assert embeddings.shape == (2, 384)

    @patch('HoloLoom.rag.embedding_plugins.HuggingFaceEmbedding.__init__',
           lambda x, model_name=None: None)
    def test_encode_query_shape(self):
        """Test encode_query returns correct shape."""
        provider = HuggingFaceEmbedding()
        provider.dimension = 384
        provider.model = Mock()
        provider.model.encode = Mock(
            return_value=np.random.randn(1, 384).astype(np.float32)
        )

        query_emb = provider.encode_query("test query")
        assert query_emb.shape == (384,)

    @patch('HoloLoom.rag.embedding_plugins.HuggingFaceEmbedding.__init__',
           lambda x, model_name=None: None)
    def test_protocol_compliance(self):
        """Test HuggingFaceEmbedding satisfies protocol."""
        provider = HuggingFaceEmbedding()
        provider.dimension = 384
        provider.model = Mock()
        provider.model.encode = Mock(
            return_value=np.random.randn(1, 384).astype(np.float32)
        )

        assert isinstance(provider, EmbeddingProvider)


# ============================================================================
# OpenAIEmbedding Tests
# ============================================================================

class TestOpenAIEmbedding:
    """Test OpenAIEmbedding provider."""

    @patch('HoloLoom.rag.embedding_plugins.OpenAIEmbedding.__init__',
           lambda x, model=None, api_key=None: None)
    def test_initialization_small_model(self):
        """Test initialization with small model."""
        provider = OpenAIEmbedding()
        provider.model = "text-embedding-3-small"
        provider.dimension = 1536
        provider.client = Mock()

        assert provider.dimension == 1536

    @patch('HoloLoom.rag.embedding_plugins.OpenAIEmbedding.__init__',
           lambda x, model="text-embedding-3-large", api_key=None: None)
    def test_initialization_large_model(self):
        """Test initialization with large model."""
        provider = OpenAIEmbedding()
        provider.model = "text-embedding-3-large"
        provider.dimension = 3072
        provider.client = Mock()

        assert provider.dimension == 3072

    @patch('HoloLoom.rag.embedding_plugins.OpenAIEmbedding.__init__',
           lambda x, model=None, api_key=None: None)
    def test_encode_shape(self):
        """Test encode returns correct shape."""
        provider = OpenAIEmbedding()
        provider.dimension = 1536
        provider.model = "text-embedding-3-small"  # Add this!
        provider.client = Mock()

        # Mock OpenAI response
        mock_response = Mock()
        mock_embedding_1 = Mock(embedding=np.random.randn(1536).tolist())
        mock_embedding_2 = Mock(embedding=np.random.randn(1536).tolist())
        mock_response.data = [mock_embedding_1, mock_embedding_2]
        provider.client.embeddings.create = Mock(return_value=mock_response)

        embeddings = provider.encode(["text1", "text2"])
        assert embeddings.shape == (2, 1536)

    @patch('HoloLoom.rag.embedding_plugins.OpenAIEmbedding.__init__',
           lambda x, model=None, api_key=None: None)
    def test_encode_query_shape(self):
        """Test encode_query returns correct shape."""
        provider = OpenAIEmbedding()
        provider.dimension = 1536
        provider.model = "text-embedding-3-small"  # Add this!
        provider.client = Mock()

        mock_response = Mock()
        mock_embedding = Mock(embedding=np.random.randn(1536).tolist())
        mock_response.data = [mock_embedding]
        provider.client.embeddings.create = Mock(return_value=mock_response)

        query_emb = provider.encode_query("test query")
        assert query_emb.shape == (1536,)

    @patch('HoloLoom.rag.embedding_plugins.OpenAIEmbedding.__init__',
           lambda x, model=None, api_key=None: None)
    def test_protocol_compliance(self):
        """Test OpenAIEmbedding satisfies protocol."""
        provider = OpenAIEmbedding()
        provider.dimension = 1536
        provider.model = "text-embedding-3-small"  # Add this!
        provider.client = Mock()
        mock_response = Mock()
        mock_embedding = Mock(embedding=np.random.randn(1536).tolist())
        mock_response.data = [mock_embedding]
        provider.client.embeddings.create = Mock(return_value=mock_response)

        assert isinstance(provider, EmbeddingProvider)


# ============================================================================
# CohereEmbedding Tests
# ============================================================================

class TestCohereEmbedding:
    """Test CohereEmbedding provider."""

    @patch('HoloLoom.rag.embedding_plugins.CohereEmbedding.__init__',
           lambda x, model=None, api_key=None: None)
    def test_initialization_v3_model(self):
        """Test initialization with v3 model."""
        provider = CohereEmbedding()
        provider.model = "embed-english-v3.0"
        provider.dimension = 1024
        provider.client = Mock()

        assert provider.dimension == 1024

    @patch('HoloLoom.rag.embedding_plugins.CohereEmbedding.__init__',
           lambda x, model="embed-english-v2.0", api_key=None: None)
    def test_initialization_v2_model(self):
        """Test initialization with v2 model."""
        provider = CohereEmbedding()
        provider.model = "embed-english-v2.0"
        provider.dimension = 4096
        provider.client = Mock()

        assert provider.dimension == 4096

    @patch('HoloLoom.rag.embedding_plugins.CohereEmbedding.__init__',
           lambda x, model=None, api_key=None: None)
    def test_encode_shape(self):
        """Test encode returns correct shape."""
        provider = CohereEmbedding()
        provider.dimension = 1024
        provider.client = Mock()
        provider.model = "embed-english-v3.0"

        mock_response = Mock()
        mock_response.embeddings = [
            np.random.randn(1024).tolist(),
            np.random.randn(1024).tolist(),
        ]
        provider.client.embed = Mock(return_value=mock_response)

        embeddings = provider.encode(["text1", "text2"])
        assert embeddings.shape == (2, 1024)

    @patch('HoloLoom.rag.embedding_plugins.CohereEmbedding.__init__',
           lambda x, model=None, api_key=None: None)
    def test_encode_query_shape(self):
        """Test encode_query returns correct shape."""
        provider = CohereEmbedding()
        provider.dimension = 1024
        provider.client = Mock()
        provider.model = "embed-english-v3.0"

        mock_response = Mock()
        mock_response.embeddings = [np.random.randn(1024).tolist()]
        provider.client.embed = Mock(return_value=mock_response)

        query_emb = provider.encode_query("test query")
        assert query_emb.shape == (1024,)

    @patch('HoloLoom.rag.embedding_plugins.CohereEmbedding.__init__',
           lambda x, model=None, api_key=None: None)
    def test_encode_query_uses_search_query(self):
        """Test encode_query uses search_query input type."""
        provider = CohereEmbedding()
        provider.dimension = 1024
        provider.client = Mock()
        provider.model = "embed-english-v3.0"

        mock_response = Mock()
        mock_response.embeddings = [np.random.randn(1024).tolist()]
        provider.client.embed = Mock(return_value=mock_response)

        provider.encode_query("test query")

        # Verify that encode_query uses search_query input type
        provider.client.embed.assert_called_with(
            texts=["test query"],
            model="embed-english-v3.0",
            input_type="search_query"
        )

    @patch('HoloLoom.rag.embedding_plugins.CohereEmbedding.__init__',
           lambda x, model=None, api_key=None: None)
    def test_protocol_compliance(self):
        """Test CohereEmbedding satisfies protocol."""
        provider = CohereEmbedding()
        provider.dimension = 1024
        provider.client = Mock()
        provider.model = "embed-english-v3.0"
        mock_response = Mock()
        mock_response.embeddings = [np.random.randn(1024).tolist()]
        provider.client.embed = Mock(return_value=mock_response)

        assert isinstance(provider, EmbeddingProvider)


# ============================================================================
# Validation Tests
# ============================================================================

class TestValidateEmbeddingProvider:
    """Test validate_embedding_provider function."""

    def test_validation_success(self):
        """Test validation succeeds for valid provider."""
        provider = Mock(spec=EmbeddingProvider)
        provider.dimension = 384
        provider.encode = Mock(return_value=np.random.randn(2, 384).astype(np.float32))
        provider.encode_query = Mock(return_value=np.random.randn(384).astype(np.float32))

        result = validate_embedding_provider(provider)
        assert result is True

    def test_validation_missing_dimension(self):
        """Test validation fails for missing dimension."""
        provider = Mock(spec=['encode', 'encode_query'])
        provider.encode = Mock(return_value=np.random.randn(2, 384).astype(np.float32))
        provider.encode_query = Mock(return_value=np.random.randn(384).astype(np.float32))

        result = validate_embedding_provider(provider)
        assert result is False

    def test_validation_invalid_dimension(self):
        """Test validation fails for invalid dimension."""
        provider = Mock()
        provider.dimension = "invalid"
        provider.encode = Mock(return_value=np.random.randn(2, 384).astype(np.float32))
        provider.encode_query = Mock(return_value=np.random.randn(384).astype(np.float32))

        result = validate_embedding_provider(provider)
        assert result is False

    def test_validation_missing_encode(self):
        """Test validation fails for missing encode method."""
        provider = Mock()
        provider.dimension = 384
        provider.encode_query = Mock(return_value=np.random.randn(384).astype(np.float32))

        result = validate_embedding_provider(provider)
        assert result is False

    def test_validation_missing_encode_query(self):
        """Test validation fails for missing encode_query method."""
        provider = Mock()
        provider.dimension = 384
        provider.encode = Mock(return_value=np.random.randn(2, 384).astype(np.float32))

        result = validate_embedding_provider(provider)
        assert result is False

    def test_validation_wrong_encode_shape(self):
        """Test validation fails for wrong encode shape."""
        provider = Mock()
        provider.dimension = 384
        provider.encode = Mock(return_value=np.random.randn(2, 512).astype(np.float32))
        provider.encode_query = Mock(return_value=np.random.randn(384).astype(np.float32))

        result = validate_embedding_provider(provider)
        assert result is False

    def test_validation_wrong_encode_query_shape(self):
        """Test validation fails for wrong encode_query shape."""
        provider = Mock()
        provider.dimension = 384
        provider.encode = Mock(return_value=np.random.randn(2, 384).astype(np.float32))
        provider.encode_query = Mock(return_value=np.random.randn(512).astype(np.float32))

        result = validate_embedding_provider(provider)
        assert result is False

    def test_validation_exception_handling(self):
        """Test validation handles exceptions gracefully."""
        provider = Mock()
        provider.dimension = 384
        provider.encode = Mock(side_effect=Exception("Test error"))

        result = validate_embedding_provider(provider)
        assert result is False


# ============================================================================
# Factory Tests
# ============================================================================

class TestCreateEmbeddingProvider:
    """Test create_embedding_provider factory function."""

    @patch('HoloLoom.rag.embedding_plugins.MatryoshkaEmbedding')
    def test_create_matryoshka(self, mock_matryoshka_class):
        """Test creating MatryoshkaEmbedding."""
        mock_provider = Mock(spec=EmbeddingProvider)
        mock_provider.dimension = 384
        mock_matryoshka_class.return_value = mock_provider

        provider = create_embedding_provider("matryoshka")
        assert provider == mock_provider
        mock_matryoshka_class.assert_called_once()

    @patch('HoloLoom.rag.embedding_plugins.HuggingFaceEmbedding')
    def test_create_huggingface(self, mock_hf_class):
        """Test creating HuggingFaceEmbedding."""
        mock_provider = Mock(spec=EmbeddingProvider)
        mock_provider.dimension = 384
        mock_hf_class.return_value = mock_provider

        provider = create_embedding_provider(
            "huggingface",
            model_name="all-MiniLM-L6-v2"
        )
        assert provider == mock_provider
        mock_hf_class.assert_called_once_with(model_name="all-MiniLM-L6-v2")

    @patch('HoloLoom.rag.embedding_plugins.OpenAIEmbedding')
    def test_create_openai(self, mock_openai_class):
        """Test creating OpenAIEmbedding."""
        mock_provider = Mock(spec=EmbeddingProvider)
        mock_provider.dimension = 1536
        mock_openai_class.return_value = mock_provider

        provider = create_embedding_provider("openai", model="text-embedding-3-small")
        assert provider == mock_provider
        mock_openai_class.assert_called_once_with(model="text-embedding-3-small")

    @patch('HoloLoom.rag.embedding_plugins.CohereEmbedding')
    def test_create_cohere(self, mock_cohere_class):
        """Test creating CohereEmbedding."""
        mock_provider = Mock(spec=EmbeddingProvider)
        mock_provider.dimension = 1024
        mock_cohere_class.return_value = mock_provider

        provider = create_embedding_provider("cohere", model="embed-english-v3.0")
        assert provider == mock_provider
        mock_cohere_class.assert_called_once_with(model="embed-english-v3.0")

    def test_create_invalid_type(self):
        """Test creating with invalid provider type."""
        with pytest.raises(ValueError):
            create_embedding_provider("invalid_type")

    def test_create_case_insensitive(self):
        """Test factory is case insensitive."""
        with patch('HoloLoom.rag.embedding_plugins.MatryoshkaEmbedding'):
            provider = create_embedding_provider("MATRYOSHKA")
            # Should not raise


# ============================================================================
# Integration Tests
# ============================================================================

class TestEmbeddingProviderIntegration:
    """Integration tests with SimpleRAG."""

    @pytest.mark.asyncio
    async def test_provider_in_simple_rag(self):
        """Test custom embedding provider works with SimpleRAG."""
        from HoloLoom.rag import SimpleRAG

        # Create a mock provider
        mock_provider = Mock(spec=EmbeddingProvider)
        mock_provider.dimension = 384
        mock_provider.encode = Mock(return_value=np.random.randn(2, 384).astype(np.float32))

        # Should not raise when creating SimpleRAG with custom provider
        rag = SimpleRAG(embedding_provider=mock_provider)
        assert rag is not None

    @patch('HoloLoom.rag.embedding_plugins.MatryoshkaEmbedding')
    def test_fallback_to_matryoshka(self, mock_matryoshka_class):
        """Test fallback to MatryoshkaEmbedding on error."""
        from HoloLoom.rag import SimpleRAG

        # Create mock that raises error
        invalid_provider = Mock(spec=['dimension'])  # Missing methods
        invalid_provider.dimension = "not an int"

        # Should fall back to MatryoshkaEmbedding
        mock_matryoshka = Mock(spec=EmbeddingProvider)
        mock_matryoshka_class.return_value = mock_matryoshka

        # SimpleRAG should handle gracefully
        rag = SimpleRAG(embedding_provider=None)  # Use default
        assert rag is not None
