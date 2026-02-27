"""
Tests for Memory-Augmented LLM Wrapper (Model Extension SDK Core)

Tests the core wrapper functionality:
- RetrievalStrategy and LearningMode enums
- LLMConfig configuration dataclass
- MemorySource and MemoryAugmentedResponse dataclasses
- MemoryAugmentedLLM main class
- InMemoryBackend fallback
- Convenience functions
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch
from hololoom.model_extension.wrapper import (
    RetrievalStrategy,
    LearningMode,
    LLMConfig,
    MemorySource,
    MemoryAugmentedResponse,
    MemoryAugmentedLLM,
    InMemoryBackend,
    create_memory_augmented_llm,
    create_config,
)
from hololoom.model_extension.uncertainty import (
    UncertaintyEnvelope,
    ConfidenceTier,
    UncertaintySource,
)
from hololoom.model_extension.verification import (
    VerificationStatus,
    VerificationTier,
    ClaimType,
)
from hololoom.model_extension.governance import (
    PolicyDecision,
    PolicyTier,
    Decision,
)
from hololoom.model_extension.providers import GenerationConfig


class TestRetrievalStrategy:
    """Tests for RetrievalStrategy enum."""

    def test_strategy_values(self):
        """Test all strategy values exist."""
        assert RetrievalStrategy.SEMANTIC.value == "semantic"
        assert RetrievalStrategy.GRAPH.value == "graph"
        assert RetrievalStrategy.HYBRID.value == "hybrid"
        assert RetrievalStrategy.ADAPTIVE.value == "adaptive"

    def test_strategy_count(self):
        """Test total number of strategies."""
        assert len(RetrievalStrategy) == 4

    def test_strategy_from_string(self):
        """Test creating strategy from string."""
        assert RetrievalStrategy("semantic") == RetrievalStrategy.SEMANTIC
        assert RetrievalStrategy("hybrid") == RetrievalStrategy.HYBRID


class TestLearningMode:
    """Tests for LearningMode enum."""

    def test_mode_values(self):
        """Test all mode values exist."""
        assert LearningMode.DISABLED.value == "disabled"
        assert LearningMode.PASSIVE.value == "passive"
        assert LearningMode.ACTIVE.value == "active"
        assert LearningMode.RESEARCH.value == "research"

    def test_mode_count(self):
        """Test total number of modes."""
        assert len(LearningMode) == 4

    def test_mode_from_string(self):
        """Test creating mode from string."""
        assert LearningMode("passive") == LearningMode.PASSIVE
        assert LearningMode("research") == LearningMode.RESEARCH


class TestLLMConfig:
    """Tests for LLMConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LLMConfig()

        # Provider defaults
        assert config.provider == "anthropic"
        assert config.model is None
        assert config.api_key is None
        assert config.base_url is None

        # Memory defaults
        assert config.retrieval_strategy == RetrievalStrategy.HYBRID
        assert config.max_context_tokens == 4000
        assert config.context_overlap == 100
        assert config.max_retrieved_memories == 10

        # Learning defaults
        assert config.learning_mode == LearningMode.PASSIVE
        assert config.learning_rate == 0.1
        assert config.exploration_bonus == 0.2

        # Quality defaults
        assert config.confidence_threshold == 0.6
        assert config.require_verification is True
        assert config.verification_tier == VerificationTier.TIER_2_EXTERNAL
        assert config.multi_sample_count == 1

        # Safety defaults
        assert config.enable_governance is True
        assert config.human_approval_threshold == 0.3

    def test_generation_config_auto_created(self):
        """Test that generation_config is auto-created if None."""
        config = LLMConfig()
        assert config.generation_config is not None
        assert isinstance(config.generation_config, GenerationConfig)

    def test_custom_values(self):
        """Test setting custom configuration values."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4",
            retrieval_strategy=RetrievalStrategy.SEMANTIC,
            learning_mode=LearningMode.RESEARCH,
            confidence_threshold=0.8,
        )

        assert config.provider == "openai"
        assert config.model == "gpt-4"
        assert config.retrieval_strategy == RetrievalStrategy.SEMANTIC
        assert config.learning_mode == LearningMode.RESEARCH
        assert config.confidence_threshold == 0.8

    def test_verification_tier_setting(self):
        """Test verification tier configuration."""
        config = LLMConfig(
            verification_tier=VerificationTier.TIER_3_AUTHORITATIVE
        )
        assert config.verification_tier == VerificationTier.TIER_3_AUTHORITATIVE

    def test_disabled_settings(self):
        """Test disabling features."""
        config = LLMConfig(
            require_verification=False,
            enable_governance=False,
            learning_mode=LearningMode.DISABLED,
        )

        assert config.require_verification is False
        assert config.enable_governance is False
        assert config.learning_mode == LearningMode.DISABLED


class TestMemorySource:
    """Tests for MemorySource dataclass."""

    def test_create_memory_source(self):
        """Test creating a memory source."""
        source = MemorySource(
            content="Thompson Sampling is a Bayesian approach",
            node_id="mem_001",
            relevance=0.92,
            source_type="semantic",
        )

        assert source.content == "Thompson Sampling is a Bayesian approach"
        assert source.node_id == "mem_001"
        assert source.relevance == 0.92
        assert source.source_type == "semantic"

    def test_default_metadata(self):
        """Test default metadata is empty dict."""
        source = MemorySource(
            content="Test content",
            node_id="test_001",
            relevance=0.5,
            source_type="graph",
        )
        assert source.metadata == {}

    def test_timestamp_auto_created(self):
        """Test timestamp is auto-created."""
        source = MemorySource(
            content="Test",
            node_id="test",
            relevance=0.5,
            source_type="test",
        )
        assert source.timestamp is not None
        assert isinstance(source.timestamp, datetime)

    def test_with_metadata(self):
        """Test creating source with metadata."""
        source = MemorySource(
            content="Content",
            node_id="node",
            relevance=0.8,
            source_type="semantic",
            metadata={"domain": "ml", "author": "test"},
        )
        assert source.metadata["domain"] == "ml"
        assert source.metadata["author"] == "test"


class TestMemoryAugmentedResponse:
    """Tests for MemoryAugmentedResponse dataclass."""

    def test_create_basic_response(self):
        """Test creating a basic response."""
        uncertainty = UncertaintyEnvelope.from_confidence(0.85)

        response = MemoryAugmentedResponse(
            answer="Thompson Sampling balances exploration and exploitation.",
            confidence=0.85,
            uncertainty=uncertainty,
        )

        assert "Thompson Sampling" in response.answer
        assert response.confidence == 0.85
        assert response.uncertainty.point_estimate == 0.85

    def test_default_values(self):
        """Test default values for optional fields."""
        uncertainty = UncertaintyEnvelope.from_confidence(0.7)

        response = MemoryAugmentedResponse(
            answer="Test answer",
            confidence=0.7,
            uncertainty=uncertainty,
        )

        assert response.sources == []
        assert response.verification is None
        assert response.governance is None
        assert response.raw_response is None
        assert response.metadata == {}

    def test_timestamp_auto_created(self):
        """Test timestamp is auto-created."""
        response = MemoryAugmentedResponse(
            answer="Test",
            confidence=0.5,
            uncertainty=UncertaintyEnvelope.from_confidence(0.5),
        )
        assert response.timestamp is not None
        assert isinstance(response.timestamp, datetime)

    def test_is_verified_without_verification(self):
        """Test is_verified returns False without verification."""
        response = MemoryAugmentedResponse(
            answer="Test",
            confidence=0.8,
            uncertainty=UncertaintyEnvelope.from_confidence(0.8),
        )
        assert response.is_verified is False

    def test_is_verified_with_passed_verification(self):
        """Test is_verified returns True when all tiers passed."""
        verification = VerificationStatus(
            tier_1_passed=True,
            tier_2_passed=True,
        )

        response = MemoryAugmentedResponse(
            answer="Verified answer",
            confidence=0.9,
            uncertainty=UncertaintyEnvelope.from_confidence(0.9),
            verification=verification,
        )
        assert response.is_verified is True

    def test_is_verified_with_failed_verification(self):
        """Test is_verified returns False when verification failed."""
        verification = VerificationStatus(
            tier_1_passed=True,
            tier_2_passed=False,
        )

        response = MemoryAugmentedResponse(
            answer="Unverified answer",
            confidence=0.9,
            uncertainty=UncertaintyEnvelope.from_confidence(0.9),
            verification=verification,
        )
        assert response.is_verified is False

    def test_requires_human_review_low_confidence(self):
        """Test requires_human_review for uncertain responses."""
        # Create envelope that requires verification
        uncertainty = UncertaintyEnvelope.from_confidence(0.35)

        response = MemoryAugmentedResponse(
            answer="Uncertain answer",
            confidence=0.35,
            uncertainty=uncertainty,
        )

        # Low confidence triggers verification requirement
        assert uncertainty.requires_verification is True
        assert response.requires_human_review is True

    def test_requires_human_review_with_escalation(self):
        """Test requires_human_review with governance escalation."""
        # Create governance decision with escalate
        governance = PolicyDecision(
            decision=Decision.ESCALATE,
            reason="High-risk action",
        )

        response = MemoryAugmentedResponse(
            answer="Escalated answer",
            confidence=0.9,
            uncertainty=UncertaintyEnvelope.from_confidence(0.9),
            governance=governance,
        )
        assert response.requires_human_review is True

    def test_to_dict(self):
        """Test JSON serialization."""
        sources = [
            MemorySource(
                content="Source content that is longer than 200 chars " * 5,
                node_id="src_001",
                relevance=0.8,
                source_type="semantic",
            )
        ]

        response = MemoryAugmentedResponse(
            answer="Test answer",
            confidence=0.75,
            uncertainty=UncertaintyEnvelope.from_confidence(0.75),
            sources=sources,
            metadata={"key": "value"},
        )

        d = response.to_dict()

        assert "answer" in d
        assert "confidence" in d
        assert "uncertainty" in d
        assert "sources" in d
        assert "is_verified" in d
        assert "requires_human_review" in d
        assert "timestamp" in d
        assert "metadata" in d
        assert d["confidence"] == 0.75
        assert d["metadata"]["key"] == "value"

    def test_to_dict_truncates_long_source_content(self):
        """Test that to_dict truncates long source content."""
        long_content = "X" * 300
        sources = [
            MemorySource(
                content=long_content,
                node_id="src",
                relevance=0.8,
                source_type="semantic",
            )
        ]

        response = MemoryAugmentedResponse(
            answer="Test",
            confidence=0.7,
            uncertainty=UncertaintyEnvelope.from_confidence(0.7),
            sources=sources,
        )

        d = response.to_dict()
        # Content should be truncated to ~203 chars (200 + "...")
        assert len(d["sources"][0]["content"]) < 250
        assert d["sources"][0]["content"].endswith("...")


class TestInMemoryBackend:
    """Tests for InMemoryBackend fallback."""

    def test_add_memory(self):
        """Test adding memory."""
        backend = InMemoryBackend()
        node_id = backend.add("Test content")

        assert node_id.startswith("mem_")
        assert "1" in node_id

    def test_add_multiple_memories(self):
        """Test adding multiple memories get unique IDs."""
        backend = InMemoryBackend()

        id1 = backend.add("Content 1")
        id2 = backend.add("Content 2")
        id3 = backend.add("Content 3")

        assert id1 != id2
        assert id2 != id3

    def test_add_with_metadata(self):
        """Test adding memory with metadata."""
        backend = InMemoryBackend()
        node_id = backend.add(
            "Test content",
            metadata={"domain": "ml", "source": "test"}
        )

        # Search should return the metadata
        results = backend.search("Test", k=1)
        assert len(results) == 1
        assert results[0]["metadata"]["domain"] == "ml"

    def test_search_exact_match(self):
        """Test searching with exact match."""
        backend = InMemoryBackend()
        backend.add("Thompson Sampling is a Bayesian approach")
        backend.add("UCB is a frequentist approach")

        results = backend.search("Thompson", k=5)

        assert len(results) >= 1
        assert "Thompson Sampling" in results[0]["content"]
        assert results[0]["score"] == 0.8  # Exact match score

    def test_search_fuzzy_match(self):
        """Test searching with word overlap."""
        backend = InMemoryBackend()
        backend.add("Machine learning models are powerful")
        backend.add("Deep learning uses neural networks")

        results = backend.search("learning", k=5)

        assert len(results) >= 2
        for r in results:
            assert r["score"] > 0

    def test_search_no_match(self):
        """Test searching with no match."""
        backend = InMemoryBackend()
        backend.add("Python is a programming language")

        results = backend.search("quantum physics", k=5)
        assert len(results) == 0

    def test_search_respects_k_limit(self):
        """Test search respects k limit."""
        backend = InMemoryBackend()
        for i in range(10):
            backend.add(f"Test content {i}")

        results = backend.search("Test", k=3)
        assert len(results) == 3

    def test_get_related_alias(self):
        """Test get_related is alias for search."""
        backend = InMemoryBackend()
        backend.add("Test content for relation")

        results1 = backend.search("Test", k=5)
        results2 = backend.get_related("Test", k=5)

        assert len(results1) == len(results2)

    @pytest.mark.asyncio
    async def test_async_recall(self):
        """Test async recall wrapper."""
        backend = InMemoryBackend()
        backend.add("Async test content")

        results = await backend.recall("Async", k=5)
        assert len(results) >= 1
        assert "Async" in results[0]["content"]

    @pytest.mark.asyncio
    async def test_async_store(self):
        """Test async store wrapper."""
        backend = InMemoryBackend()

        node_id = await backend.store(
            content="Stored content",
            metadata={"test": True},
            source="test_source",
        )

        assert node_id.startswith("mem_")

        # Verify metadata includes source
        results = backend.search("Stored", k=1)
        assert results[0]["metadata"]["source"] == "test_source"

    @pytest.mark.asyncio
    async def test_async_close(self):
        """Test async close does not raise."""
        backend = InMemoryBackend()
        await backend.close()  # Should not raise


class TestMemoryAugmentedLLMInit:
    """Tests for MemoryAugmentedLLM initialization."""

    def test_init_with_defaults(self):
        """Test initialization with defaults."""
        llm = MemoryAugmentedLLM()

        assert llm.config.provider == "anthropic"
        assert llm._initialized is False
        assert llm._provider is None

    def test_init_with_config(self):
        """Test initialization with full config."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4",
            learning_mode=LearningMode.RESEARCH,
        )

        llm = MemoryAugmentedLLM(config=config)

        assert llm.config.provider == "openai"
        assert llm.config.model == "gpt-4"
        assert llm.config.learning_mode == LearningMode.RESEARCH

    def test_init_with_kwargs(self):
        """Test initialization with kwargs shortcuts."""
        llm = MemoryAugmentedLLM(
            provider="ollama",
            model="llama2",
            api_key="test-key",
        )

        assert llm.config.provider == "ollama"
        assert llm.config.model == "llama2"
        assert llm.config.api_key == "test-key"

    def test_init_session_id_created(self):
        """Test session ID is created on init."""
        llm = MemoryAugmentedLLM()
        assert llm._session_id is not None
        assert len(llm._session_id) > 0


class TestMemoryAugmentedLLMContextManager:
    """Tests for MemoryAugmentedLLM async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_initializes(self):
        """Test context manager initializes LLM."""
        with patch("hololoom.model_extension.wrapper.create_provider") as mock_provider:
            mock_provider.return_value = MagicMock(is_available=lambda: False)

            async with MemoryAugmentedLLM(provider="test") as llm:
                assert llm._initialized is True

    @pytest.mark.asyncio
    async def test_context_manager_cleans_up(self):
        """Test context manager cleans up on exit."""
        with patch("hololoom.model_extension.wrapper.create_provider") as mock_provider:
            mock_provider.return_value = MagicMock(is_available=lambda: False)

            async with MemoryAugmentedLLM(provider="test") as llm:
                assert llm._initialized is True

            # After exit
            assert llm._initialized is False


class TestMemoryAugmentedLLMLearn:
    """Tests for MemoryAugmentedLLM.learn() method."""

    @pytest.mark.asyncio
    async def test_learn_single_content(self):
        """Test learning single content."""
        with patch("hololoom.model_extension.wrapper.create_provider") as mock_provider:
            mock_provider.return_value = MagicMock(is_available=lambda: False)

            async with MemoryAugmentedLLM() as llm:
                node_ids = await llm.learn("Thompson Sampling is Bayesian")

                assert len(node_ids) >= 1

    @pytest.mark.asyncio
    async def test_learn_multiple_contents(self):
        """Test learning multiple contents."""
        with patch("hololoom.model_extension.wrapper.create_provider") as mock_provider:
            mock_provider.return_value = MagicMock(is_available=lambda: False)

            async with MemoryAugmentedLLM() as llm:
                contents = [
                    "Content 1",
                    "Content 2",
                    "Content 3",
                ]
                node_ids = await llm.learn(contents)

                assert len(node_ids) == 3

    @pytest.mark.asyncio
    async def test_learn_updates_learning_buffer(self):
        """Test learning updates learning buffer."""
        with patch("hololoom.model_extension.wrapper.create_provider") as mock_provider:
            mock_provider.return_value = MagicMock(is_available=lambda: False)

            async with MemoryAugmentedLLM() as llm:
                # Mock _store_memory to avoid real memory backend initialization
                async def mock_store(*args, **kwargs):
                    return "mock-node-id"
                llm._store_memory = mock_store

                await llm.learn("Test content")

                assert len(llm._learning_buffer) >= 1
                assert llm._learning_buffer[0]["type"] == "learn"

    @pytest.mark.asyncio
    async def test_learn_disabled_mode_no_buffer_update(self):
        """Test learning doesn't update buffer in disabled mode."""
        with patch("hololoom.model_extension.wrapper.create_provider") as mock_provider:
            mock_provider.return_value = MagicMock(is_available=lambda: False)

            async with MemoryAugmentedLLM(learning_mode="disabled") as llm:
                # Mock _store_memory to avoid real memory backend initialization
                async def mock_store(*args, **kwargs):
                    return "mock-node-id"
                llm._store_memory = mock_store

                await llm.learn("Test content")

                # Should still store, but not buffer for learning
                learn_entries = [e for e in llm._learning_buffer if e["type"] == "learn"]
                assert len(learn_entries) == 0


class TestMemoryAugmentedLLMRecall:
    """Tests for MemoryAugmentedLLM.recall() method."""

    @pytest.mark.asyncio
    async def test_recall_uses_default_strategy(self):
        """Test recall uses config's default strategy."""
        with patch("hololoom.model_extension.wrapper.create_provider") as mock_provider:
            mock_provider.return_value = MagicMock(is_available=lambda: False)

            async with MemoryAugmentedLLM(
                retrieval_strategy=RetrievalStrategy.SEMANTIC
            ) as llm:
                # Mock _store_memory to avoid Neo4j/Qdrant connection
                async def mock_store(*args, **kwargs):
                    return "mock-node-id"
                llm._store_memory = mock_store

                # First learn something
                await llm.learn("Thompson Sampling test content")

                # Then recall
                sources = await llm.recall("Thompson", k=5)

                # Should return list of MemorySource
                assert isinstance(sources, list)

    @pytest.mark.asyncio
    async def test_recall_with_override_strategy(self):
        """Test recall with strategy override."""
        # Mock create_memory_backend BEFORE entering context manager to prevent hang
        # during __aenter__ -> _initialize() -> _initialize_memory()
        # Use AsyncMock because create_memory_backend is an async function
        with patch("hololoom.model_extension.wrapper.create_provider") as mock_provider, \
             patch("hololoom.memory.backend_factory.create_memory_backend", new_callable=AsyncMock) as mock_backend:
            mock_provider.return_value = MagicMock(is_available=lambda: False)
            # Return InMemoryBackend to avoid Neo4j/Qdrant connection
            mock_backend.return_value = InMemoryBackend()

            async with MemoryAugmentedLLM() as llm:
                await llm.learn("Graph retrieval test content")

                sources = await llm.recall(
                    "Graph",
                    k=5,
                    strategy=RetrievalStrategy.GRAPH,
                )

                assert isinstance(sources, list)


class TestMemoryAugmentedLLMReflect:
    """Tests for MemoryAugmentedLLM.reflect() method."""

    @pytest.mark.asyncio
    async def test_reflect_updates_pattern_weights(self):
        """Test reflect updates pattern weights on positive feedback."""
        with patch("hololoom.model_extension.wrapper.create_provider") as mock_provider, \
             patch("hololoom.memory.backend_factory.create_memory_backend", new_callable=AsyncMock) as mock_backend:
            mock_provider.return_value = MagicMock(is_available=lambda: False)
            mock_backend.return_value = InMemoryBackend()

            async with MemoryAugmentedLLM() as llm:
                # Create response with sources
                sources = [
                    MemorySource(
                        content="Test",
                        node_id="test",
                        relevance=0.8,
                        source_type="semantic",
                    )
                ]
                response = MemoryAugmentedResponse(
                    answer="Test answer",
                    confidence=0.8,
                    uncertainty=UncertaintyEnvelope.from_confidence(0.8),
                    sources=sources,
                )

                # Reflect with positive feedback
                await llm.reflect(response, feedback={"helpful": True})

                # Pattern weight should increase
                assert "source:semantic" in llm._pattern_weights
                assert llm._pattern_weights["source:semantic"] > 0.5

    @pytest.mark.asyncio
    async def test_reflect_updates_pattern_weights_negative(self):
        """Test reflect updates pattern weights on negative feedback."""
        with patch("hololoom.model_extension.wrapper.create_provider") as mock_provider, \
             patch("hololoom.memory.backend_factory.create_memory_backend", new_callable=AsyncMock) as mock_backend:
            mock_provider.return_value = MagicMock(is_available=lambda: False)
            mock_backend.return_value = InMemoryBackend()

            async with MemoryAugmentedLLM(learning_rate=0.2) as llm:
                # Initialize weight
                llm._pattern_weights["source:graph"] = 0.5

                sources = [
                    MemorySource(
                        content="Test",
                        node_id="test",
                        relevance=0.8,
                        source_type="graph",
                    )
                ]
                response = MemoryAugmentedResponse(
                    answer="Test answer",
                    confidence=0.8,
                    uncertainty=UncertaintyEnvelope.from_confidence(0.8),
                    sources=sources,
                )

                # Reflect with negative feedback
                await llm.reflect(response, feedback={"helpful": False})

                # Pattern weight should decrease
                assert llm._pattern_weights["source:graph"] < 0.5

    @pytest.mark.asyncio
    async def test_reflect_disabled_mode_does_nothing(self):
        """Test reflect does nothing in disabled mode."""
        with patch("hololoom.model_extension.wrapper.create_provider") as mock_provider, \
             patch("hololoom.memory.backend_factory.create_memory_backend", new_callable=AsyncMock) as mock_backend:
            mock_provider.return_value = MagicMock(is_available=lambda: False)
            mock_backend.return_value = InMemoryBackend()

            async with MemoryAugmentedLLM(learning_mode="disabled") as llm:
                response = MemoryAugmentedResponse(
                    answer="Test",
                    confidence=0.8,
                    uncertainty=UncertaintyEnvelope.from_confidence(0.8),
                )

                initial_buffer_len = len(llm._learning_buffer)
                await llm.reflect(response, feedback={"helpful": True})

                # Buffer should not grow
                assert len(llm._learning_buffer) == initial_buffer_len


class TestMemoryAugmentedLLMQuery:
    """Tests for MemoryAugmentedLLM.query() method."""

    @pytest.mark.asyncio
    async def test_query_memory_only_no_provider(self):
        """Test query falls back to memory-only when no provider."""
        with patch("hololoom.model_extension.wrapper.create_provider") as mock_provider, \
             patch("hololoom.memory.backend_factory.create_memory_backend", new_callable=AsyncMock) as mock_backend:
            mock_provider.return_value = MagicMock(is_available=lambda: False)
            mock_backend.return_value = InMemoryBackend()
            with patch("hololoom.model_extension.wrapper.get_best_available_provider") as mock_best:
                mock_best.return_value = None

                async with MemoryAugmentedLLM() as llm:
                    # Mock _store_memory to avoid Neo4j/Qdrant connection
                    async def mock_store(*args, **kwargs):
                        return "mock-node-id"
                    llm._store_memory = mock_store

                    # Learn something
                    await llm.learn("Thompson Sampling balances exploration")

                    # Query
                    response = await llm.query("What is Thompson Sampling?")

                    # Should get memory-only response
                    assert response.metadata.get("memory_only") is True

    @pytest.mark.asyncio
    async def test_query_with_context_override(self):
        """Test query with context override."""
        with patch("hololoom.model_extension.wrapper.create_provider") as mock_provider, \
             patch("hololoom.memory.backend_factory.create_memory_backend", new_callable=AsyncMock) as mock_backend:
            mock_instance = MagicMock()
            mock_instance.is_available.return_value = True
            mock_instance.model = "test-model"

            # Create async generate method
            async def mock_generate(*args, **kwargs):
                from hololoom.model_extension.providers import GenerationResult
                return GenerationResult(
                    text="Generated response",
                    model="test",
                    provider="test",
                    usage={},
                    finish_reason="stop",
                    metadata={},
                )

            mock_instance.generate = mock_generate
            mock_provider.return_value = mock_instance
            mock_backend.return_value = InMemoryBackend()

            async with MemoryAugmentedLLM() as llm:
                response = await llm.query(
                    "What is X?",
                    context_override="X is a test concept.",
                )

                # Sources should be empty since we overrode context
                assert response.sources == []


class TestMemoryAugmentedLLMHelpers:
    """Tests for helper methods."""

    def test_extract_confidence_no_markers(self):
        """Test confidence extraction with no uncertainty markers."""
        llm = MemoryAugmentedLLM()
        conf = llm._extract_confidence("Thompson Sampling is a method.")
        assert conf == 0.8  # Base confidence

    def test_extract_confidence_with_markers(self):
        """Test confidence extraction with uncertainty markers."""
        llm = MemoryAugmentedLLM()
        conf = llm._extract_confidence("I'm not sure, but it might be Thompson Sampling")
        assert conf < 0.8  # Reduced confidence

    def test_extract_confidence_many_markers(self):
        """Test confidence extraction with many uncertainty markers."""
        llm = MemoryAugmentedLLM()
        conf = llm._extract_confidence(
            "I'm not sure, I think it might be, probably, perhaps Thompson Sampling"
        )
        assert conf < 0.6  # Significantly reduced

    def test_extract_confidence_minimum(self):
        """Test confidence has minimum value."""
        llm = MemoryAugmentedLLM()
        # Many markers
        conf = llm._extract_confidence(
            "I'm not sure I'm uncertain I don't know might be could be possibly "
            "perhaps I think I believe probably"
        )
        assert conf >= 0.3  # Minimum bound

    def test_format_context_empty(self):
        """Test formatting empty context."""
        llm = MemoryAugmentedLLM()
        context = llm._format_context([])
        assert context == ""

    def test_format_context_multiple_sources(self):
        """Test formatting multiple sources."""
        llm = MemoryAugmentedLLM()
        sources = [
            MemorySource(content="First source", node_id="1", relevance=0.9, source_type="semantic"),
            MemorySource(content="Second source", node_id="2", relevance=0.8, source_type="graph"),
        ]

        context = llm._format_context(sources)

        assert "[1]" in context
        assert "First source" in context
        assert "[2]" in context
        assert "Second source" in context

    def test_build_query_prompt_with_context(self):
        """Test building prompt with context."""
        llm = MemoryAugmentedLLM()
        prompt = llm._build_query_prompt(
            "What is X?",
            "X is a test concept."
        )

        assert "Context" in prompt
        assert "X is a test concept" in prompt
        assert "Question: What is X?" in prompt

    def test_build_query_prompt_without_context(self):
        """Test building prompt without context."""
        llm = MemoryAugmentedLLM()
        prompt = llm._build_query_prompt("What is X?", "")

        assert "Context" not in prompt
        assert "Question: What is X?" in prompt

    def test_get_learning_statistics(self):
        """Test getting learning statistics."""
        llm = MemoryAugmentedLLM()
        llm._learning_buffer = [{"type": "test"}]
        llm._pattern_weights = {"test": 0.6}
        llm._query_history = [{"query": "test"}]

        stats = llm.get_learning_statistics()

        assert stats["buffer_size"] == 1
        assert stats["pattern_weights"]["test"] == 0.6
        assert stats["query_count"] == 1
        assert "session_id" in stats


class TestCreateMemoryAugmentedLLM:
    """Tests for create_memory_augmented_llm convenience function."""

    @pytest.mark.asyncio
    async def test_creates_initialized_llm(self):
        """Test convenience function creates initialized LLM."""
        with patch("hololoom.model_extension.wrapper.create_provider") as mock_provider, \
             patch("hololoom.memory.backend_factory.create_memory_backend", new_callable=AsyncMock) as mock_backend:
            mock_provider.return_value = MagicMock(is_available=lambda: False)
            mock_backend.return_value = InMemoryBackend()

            llm = await create_memory_augmented_llm(provider="test")

            try:
                assert llm._initialized is True
            finally:
                await llm._cleanup()

    @pytest.mark.asyncio
    async def test_accepts_kwargs(self):
        """Test convenience function accepts kwargs."""
        with patch("hololoom.model_extension.wrapper.create_provider") as mock_provider, \
             patch("hololoom.memory.backend_factory.create_memory_backend", new_callable=AsyncMock) as mock_backend:
            mock_provider.return_value = MagicMock(is_available=lambda: False)
            mock_backend.return_value = InMemoryBackend()

            llm = await create_memory_augmented_llm(
                provider="openai",
                model="gpt-4",
                learning_mode="research",
            )

            try:
                assert llm.config.provider == "openai"
                assert llm.config.model == "gpt-4"
                assert llm.config.learning_mode == LearningMode.RESEARCH
            finally:
                await llm._cleanup()


class TestCreateConfig:
    """Tests for create_config convenience function."""

    def test_creates_config_with_defaults(self):
        """Test creating config with defaults."""
        config = create_config()

        assert config.provider == "anthropic"
        assert config.learning_mode == LearningMode.PASSIVE
        assert config.require_verification is True

    def test_creates_config_with_overrides(self):
        """Test creating config with overrides."""
        config = create_config(
            provider="openai",
            learning_mode="research",
            enable_verification=False,
        )

        assert config.provider == "openai"
        assert config.learning_mode == LearningMode.RESEARCH
        assert config.require_verification is False

    def test_accepts_kwargs(self):
        """Test create_config accepts additional kwargs."""
        config = create_config(
            provider="ollama",
            confidence_threshold=0.9,
            max_context_tokens=8000,
        )

        assert config.provider == "ollama"
        assert config.confidence_threshold == 0.9
        assert config.max_context_tokens == 8000


class TestMergeSources:
    """Tests for source merging logic."""

    def test_merge_removes_duplicates(self):
        """Test merging removes duplicate node IDs."""
        llm = MemoryAugmentedLLM()

        sources1 = [
            MemorySource(content="A", node_id="1", relevance=0.9, source_type="semantic"),
            MemorySource(content="B", node_id="2", relevance=0.8, source_type="semantic"),
        ]
        sources2 = [
            MemorySource(content="C", node_id="2", relevance=0.7, source_type="graph"),
            MemorySource(content="D", node_id="3", relevance=0.6, source_type="graph"),
        ]

        merged = llm._merge_sources(sources1, sources2)

        # Should have 3 unique sources (node "2" deduplicated)
        assert len(merged) == 3
        node_ids = [s.node_id for s in merged]
        assert len(set(node_ids)) == 3

    def test_merge_sorts_by_relevance(self):
        """Test merged sources are sorted by relevance."""
        llm = MemoryAugmentedLLM()

        sources1 = [
            MemorySource(content="Low", node_id="1", relevance=0.3, source_type="semantic"),
        ]
        sources2 = [
            MemorySource(content="High", node_id="2", relevance=0.9, source_type="graph"),
        ]

        merged = llm._merge_sources(sources1, sources2)

        assert merged[0].relevance == 0.9
        assert merged[0].node_id == "2"


class TestPatternWeightsClamping:
    """Tests for pattern weights clamping."""

    @pytest.mark.asyncio
    async def test_weights_clamped_high(self):
        """Test weights are clamped at max 0.9."""
        with patch("hololoom.model_extension.wrapper.create_provider") as mock_provider:
            mock_provider.return_value = MagicMock(is_available=lambda: False)

            async with MemoryAugmentedLLM(learning_rate=0.5) as llm:
                llm._pattern_weights["source:semantic"] = 0.85

                sources = [
                    MemorySource(content="T", node_id="t", relevance=0.9, source_type="semantic")
                ]
                response = MemoryAugmentedResponse(
                    answer="T", confidence=0.9,
                    uncertainty=UncertaintyEnvelope.from_confidence(0.9),
                    sources=sources,
                )

                await llm.reflect(response, feedback={"helpful": True})

                # Should be clamped at 0.9
                assert llm._pattern_weights["source:semantic"] <= 0.9

    @pytest.mark.asyncio
    async def test_weights_clamped_low(self):
        """Test weights are clamped at min 0.1."""
        with patch("hololoom.model_extension.wrapper.create_provider") as mock_provider:
            mock_provider.return_value = MagicMock(is_available=lambda: False)

            async with MemoryAugmentedLLM(learning_rate=0.5) as llm:
                llm._pattern_weights["source:graph"] = 0.15

                sources = [
                    MemorySource(content="T", node_id="t", relevance=0.9, source_type="graph")
                ]
                response = MemoryAugmentedResponse(
                    answer="T", confidence=0.9,
                    uncertainty=UncertaintyEnvelope.from_confidence(0.9),
                    sources=sources,
                )

                # Negative feedback several times
                for _ in range(5):
                    await llm.reflect(response, feedback={"helpful": False})

                # Should be clamped at 0.1
                assert llm._pattern_weights["source:graph"] >= 0.1

