"""
Test Resonance Shed - Feature Interference Zone
================================================

Tests for HoloLoom/resonance/shed.py - Multi-modal feature extraction.

Coverage:
- Feature thread lifting (motif, embedding, spectral, semantic_flow)
- Interference patterns (weighted_sum, attention, concat)
- Complete weave cycle (lift → interfere → lower)
- Pressure relief mechanism (max_feature_density)
- Guardrail integration
- Thread weights and metadata
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Any

from HoloLoom.resonance.shed import (
    ResonanceShed,
    FeatureThread,
    create_resonance_shed
)
from HoloLoom.documentation.types import Motif


# ============================================================================
# Mock Extractors
# ============================================================================

class MockMotifDetector:
    """Mock motif detector."""
    async def detect(self, text: str) -> List[Motif]:
        return [
            Motif(pattern="ALGORITHM", span=(0, 9), score=0.9),
            Motif(pattern="OPTIMIZATION", span=(10, 22), score=0.8)
        ]


class MockEmbedder:
    """Mock embedder with matryoshka support."""
    def encode(self, texts: List[str]) -> np.ndarray:
        # Return 384-dim embeddings
        return np.random.randn(len(texts), 384)

    def encode_scales(self, texts: List[str], size: int) -> np.ndarray:
        # Return embeddings at specified scale
        return np.random.randn(len(texts), size)


class MockSpectralFusion:
    """Mock spectral feature extractor."""
    async def features(self, graph, texts, embedder):
        spectral_vector = np.random.randn(10)
        metrics = {"graph_size": 5, "avg_degree": 2.5}
        return spectral_vector, metrics


class MockSemanticCalculus:
    """Mock semantic flow calculus."""
    def extract_features(self, text: str):
        return {
            "trajectory": Mock(),
            "avg_velocity": 0.5,
            "avg_acceleration": 0.1,
            "curvature": [0.2, 0.3, 0.1],
            "total_distance": 2.5,
            "n_states": 5
        }


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def shed():
    """Create basic resonance shed with all extractors."""
    return ResonanceShed(
        motif_detector=MockMotifDetector(),
        embedder=MockEmbedder(),
        spectral_fusion=MockSpectralFusion(),
        semantic_calculus=MockSemanticCalculus(),
        interference_mode="weighted_sum",
        max_feature_density=1.0,
        guardrails=None  # Disable for basic tests
    )


@pytest.fixture
def minimal_shed():
    """Create minimal shed with only embedder."""
    return ResonanceShed(
        motif_detector=None,
        embedder=MockEmbedder(),
        spectral_fusion=None,
        semantic_calculus=None,
        guardrails=None
    )


@pytest.fixture
def test_text():
    """Standard test text."""
    return "Thompson Sampling balances exploration and exploitation"


# ============================================================================
# Test Initialization
# ============================================================================

def test_shed_initialization():
    """Test shed initializes correctly."""
    shed = ResonanceShed(
        motif_detector=MockMotifDetector(),
        embedder=MockEmbedder(),
        interference_mode="attention",
        max_feature_density=0.85
    )

    assert shed.interference_mode == "attention"
    assert shed.max_feature_density == 0.85
    assert shed.is_lifted is False
    assert len(shed.threads) == 0
    assert shed.current_density == 0.0


# ============================================================================
# Test Feature Thread Lifting
# ============================================================================

@pytest.mark.asyncio
async def test_lift_all_threads(shed, test_text):
    """Test lifting all available feature threads."""
    await shed.lift(test_text)

    assert shed.is_lifted is True
    assert len(shed.threads) == 4  # motif, embedding, spectral, semantic_flow

    # Check thread names
    thread_names = {t.name for t in shed.threads}
    assert thread_names == {"motif", "embedding", "spectral", "semantic_flow"}


@pytest.mark.asyncio
async def test_lift_motif_thread(shed, test_text):
    """Test motif thread extraction."""
    await shed.lift(test_text)

    motif_thread = next((t for t in shed.threads if t.name == "motif"), None)
    assert motif_thread is not None
    assert len(motif_thread.features) == 2  # 2 motifs
    assert motif_thread.features == ["ALGORITHM", "OPTIMIZATION"]
    assert motif_thread.metadata["count"] == 2


@pytest.mark.asyncio
async def test_lift_embedding_thread(shed, test_text):
    """Test embedding thread extraction."""
    await shed.lift(test_text)

    embedding_thread = next((t for t in shed.threads if t.name == "embedding"), None)
    assert embedding_thread is not None
    assert embedding_thread.metadata["dim"] == 384
    assert len(embedding_thread.features) == 384


@pytest.mark.asyncio
async def test_lift_spectral_thread(shed, test_text):
    """Test spectral thread extraction."""
    mock_graph = Mock()  # Mock knowledge graph

    await shed.lift(test_text, context_graph=mock_graph)

    spectral_thread = next((t for t in shed.threads if t.name == "spectral"), None)
    assert spectral_thread is not None
    assert "metrics" in spectral_thread.metadata
    assert spectral_thread.metadata["metrics"]["graph_size"] == 5


@pytest.mark.asyncio
async def test_lift_semantic_flow_thread(shed, test_text):
    """Test semantic flow thread extraction."""
    await shed.lift(test_text)

    flow_thread = next((t for t in shed.threads if t.name == "semantic_flow"), None)
    assert flow_thread is not None
    assert flow_thread.metadata["n_words"] == 5
    assert flow_thread.metadata["avg_speed"] == 0.5


@pytest.mark.asyncio
async def test_lift_with_custom_weights(shed, test_text):
    """Test lifting with custom thread weights."""
    custom_weights = {
        "motif": 0.5,
        "embedding": 1.0,
        "spectral": 0.3
    }

    await shed.lift(test_text, thread_weights=custom_weights)

    # Check weights applied
    motif_thread = next((t for t in shed.threads if t.name == "motif"), None)
    embedding_thread = next((t for t in shed.threads if t.name == "embedding"), None)

    assert motif_thread.weight == 0.5
    assert embedding_thread.weight == 1.0


@pytest.mark.asyncio
async def test_lift_minimal_shed(minimal_shed, test_text):
    """Test lifting with minimal extractors."""
    await minimal_shed.lift(test_text)

    # Should only have embedding thread
    assert len(minimal_shed.threads) == 1
    assert minimal_shed.threads[0].name == "embedding"


@pytest.mark.asyncio
async def test_lift_already_lifted_lowers_first(shed, test_text):
    """Test lifting when already lifted lowers first."""
    # First lift
    await shed.lift(test_text)
    assert shed.is_lifted is True
    first_thread_count = len(shed.threads)

    # Second lift (should lower first)
    await shed.lift(test_text)
    assert shed.is_lifted is True
    assert len(shed.threads) == first_thread_count


# ============================================================================
# Test Interference Patterns
# ============================================================================

@pytest.mark.asyncio
async def test_interfere_creates_dotplasma(shed, test_text):
    """Test interference creates DotPlasma."""
    await shed.lift(test_text)
    plasma = shed.interfere()

    assert "motifs" in plasma
    assert "psi" in plasma  # Embedding
    assert "spectral" in plasma
    assert "semantic_flow" in plasma
    assert "metadata" in plasma
    assert "threads" in plasma


@pytest.mark.asyncio
async def test_interfere_without_lift_returns_empty(shed):
    """Test interfering without lifting returns empty."""
    plasma = shed.interfere()

    assert plasma == {}


@pytest.mark.asyncio
async def test_interfere_includes_metadata(shed, test_text):
    """Test interference includes metadata."""
    await shed.lift(test_text)
    plasma = shed.interfere()

    metadata = plasma["metadata"]
    assert metadata["thread_count"] == 4
    assert metadata["interference_mode"] == "weighted_sum"
    assert "thread_weights" in metadata


@pytest.mark.asyncio
async def test_interfere_thread_details(shed, test_text):
    """Test interference includes thread details."""
    await shed.lift(test_text)
    plasma = shed.interfere()

    assert len(plasma["threads"]) == 4

    # Check thread structure
    for thread_info in plasma["threads"]:
        assert "name" in thread_info
        assert "weight" in thread_info
        assert "metadata" in thread_info


@pytest.mark.asyncio
async def test_interfere_modes():
    """Test different interference modes."""
    modes = ["weighted_sum", "attention", "concat"]

    for mode in modes:
        shed = ResonanceShed(
            embedder=MockEmbedder(),
            interference_mode=mode,
            guardrails=None
        )

        await shed.lift("test text")
        plasma = shed.interfere()

        assert plasma["metadata"]["interference_mode"] == mode


# ============================================================================
# Test Complete Weave Cycle
# ============================================================================

@pytest.mark.asyncio
async def test_weave_complete_cycle(shed, test_text):
    """Test complete weave cycle: lift → interfere → lower."""
    plasma = await shed.weave(test_text)

    # Should have all features
    assert "motifs" in plasma
    assert "psi" in plasma
    assert len(plasma["threads"]) == 4

    # Shed should be lowered after weave
    assert shed.is_lifted is False
    assert len(shed.threads) == 0


@pytest.mark.asyncio
async def test_weave_with_context_graph(shed, test_text):
    """Test weaving with context graph."""
    mock_graph = Mock()

    plasma = await shed.weave(test_text, context_graph=mock_graph)

    # Should include spectral features (requires context graph)
    assert plasma["spectral"] is not None


@pytest.mark.asyncio
async def test_weave_with_custom_weights(shed, test_text):
    """Test weaving with custom thread weights."""
    custom_weights = {"motif": 0.7, "embedding": 1.0}

    plasma = await shed.weave(test_text, thread_weights=custom_weights)

    # Check weights in metadata
    thread_weights = plasma["metadata"]["thread_weights"]
    assert thread_weights["motif"] == 0.7
    assert thread_weights["embedding"] == 1.0


# ============================================================================
# Test Pressure Relief
# ============================================================================

@pytest.mark.asyncio
async def test_pressure_relief_triggered():
    """Test pressure relief drops lowest-weight threads."""
    shed = ResonanceShed(
        motif_detector=MockMotifDetector(),
        embedder=MockEmbedder(),
        spectral_fusion=MockSpectralFusion(),
        semantic_calculus=MockSemanticCalculus(),
        max_feature_density=0.5,  # Only allow 50% density
        guardrails=None
    )

    await shed.lift("test text")

    # Should have shed some threads (50% of 4 = 2 threads)
    assert len(shed.threads) <= 2
    assert shed.pressure_relief_count == 1


@pytest.mark.asyncio
async def test_pressure_relief_keeps_highest_weight():
    """Test pressure relief keeps highest-weight threads."""
    shed = ResonanceShed(
        motif_detector=MockMotifDetector(),
        embedder=MockEmbedder(),
        spectral_fusion=MockSpectralFusion(),
        max_feature_density=0.5,
        guardrails=None
    )

    custom_weights = {
        "motif": 0.3,
        "embedding": 1.0,  # Highest weight
        "spectral": 0.5
    }

    await shed.lift("test text", thread_weights=custom_weights)

    # Embedding should be kept (highest weight)
    thread_names = {t.name for t in shed.threads}
    assert "embedding" in thread_names


# ============================================================================
# Test Lower (Shed Deactivation)
# ============================================================================

@pytest.mark.asyncio
async def test_lower_clears_threads(shed, test_text):
    """Test lowering clears threads."""
    await shed.lift(test_text)
    assert len(shed.threads) > 0

    shed.lower()

    assert shed.is_lifted is False
    assert len(shed.threads) == 0
    assert shed.current_density == 0.0


@pytest.mark.asyncio
async def test_lower_when_not_lifted_is_safe(shed):
    """Test lowering when not lifted is safe."""
    shed.lower()  # Should not crash

    assert shed.is_lifted is False


# ============================================================================
# Test Trace Export
# ============================================================================

@pytest.mark.asyncio
async def test_get_trace_while_lifted(shed, test_text):
    """Test trace export while lifted."""
    await shed.lift(test_text)

    trace = shed.get_trace()

    assert trace["is_lifted"] is True
    assert trace["thread_count"] == 4
    assert len(trace["threads"]) == 4
    assert trace["interference_mode"] == "weighted_sum"


@pytest.mark.asyncio
async def test_get_trace_after_lower(shed, test_text):
    """Test trace export after lowering."""
    await shed.lift(test_text)
    shed.lower()

    trace = shed.get_trace()

    assert trace["is_lifted"] is False
    assert trace["thread_count"] == 0


@pytest.mark.asyncio
async def test_get_trace_thread_details(shed, test_text):
    """Test trace includes thread details."""
    await shed.lift(test_text)

    trace = shed.get_trace()

    # Check thread structure
    for thread_info in trace["threads"]:
        assert "name" in thread_info
        assert "weight" in thread_info
        assert "has_features" in thread_info
        assert "metadata" in thread_info


# ============================================================================
# Test Matryoshka Scale-Aware Encoding
# ============================================================================

@pytest.mark.asyncio
async def test_matryoshka_scale_aware_encoding():
    """Test matryoshka scale-aware encoding (Phase 5)."""
    embedder = MockEmbedder()

    shed = ResonanceShed(
        embedder=embedder,
        target_scale=192,  # Request 192-dim encoding
        guardrails=None
    )

    await shed.lift("test text")

    embedding_thread = next((t for t in shed.threads if t.name == "embedding"), None)

    # Should use 192-dim encoding
    assert len(embedding_thread.features) == 192


@pytest.mark.asyncio
async def test_fallback_to_full_dimension():
    """Test fallback to full dimension if scale not supported."""
    # Create embedder without encode_scales method
    class LegacyEmbedder:
        def encode(self, texts):
            return np.random.randn(len(texts), 384)

    shed = ResonanceShed(
        embedder=LegacyEmbedder(),
        target_scale=192,  # Request scale, but not supported
        guardrails=None
    )

    await shed.lift("test text")

    embedding_thread = next((t for t in shed.threads if t.name == "embedding"), None)

    # Should fallback to full 384-dim
    assert len(embedding_thread.features) == 384


# ============================================================================
# Test Error Handling
# ============================================================================

@pytest.mark.asyncio
async def test_extractor_failure_does_not_crash():
    """Test extractor failures are handled gracefully."""
    # Create embedder that raises exception
    class FailingEmbedder:
        def encode(self, texts):
            raise RuntimeError("Embedding failed!")

    shed = ResonanceShed(
        embedder=FailingEmbedder(),
        motif_detector=MockMotifDetector(),
        guardrails=None
    )

    # Should not crash, just skip embedding thread
    await shed.lift("test text")

    # Should only have motif thread (embedding failed)
    assert len(shed.threads) == 1
    assert shed.threads[0].name == "motif"


@pytest.mark.asyncio
async def test_spectral_without_context_graph_skips():
    """Test spectral extraction skips without context graph."""
    shed = ResonanceShed(
        spectral_fusion=MockSpectralFusion(),
        guardrails=None
    )

    await shed.lift("test text", context_graph=None)

    # Should not have spectral thread (no context graph)
    thread_names = {t.name for t in shed.threads}
    assert "spectral" not in thread_names


# ============================================================================
# Test Factory Function
# ============================================================================

def test_create_resonance_shed():
    """Test factory function creates shed."""
    shed = create_resonance_shed(
        motif_detector=MockMotifDetector(),
        embedder=MockEmbedder(),
        mode="attention",
        guardrails=None
    )

    assert isinstance(shed, ResonanceShed)
    assert shed.interference_mode == "attention"


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
- Initialization: 1 test
- Feature Thread Lifting: 8 tests
- Interference Patterns: 5 tests
- Complete Weave Cycle: 3 tests
- Pressure Relief: 2 tests
- Lower/Deactivation: 2 tests
- Trace Export: 3 tests
- Matryoshka Encoding: 2 tests
- Error Handling: 2 tests
- Factory Function: 1 test

Total: 29 tests covering critical resonance shed functionality
"""
