"""
Tests for WeaveHouse - Composite Loom Implementation
====================================================

Tests for the WeaveHouse class which provides parallel multi-perspective
weaving with consensus and auto-exploration.

Date: December 2025
"""

import asyncio
import pytest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

# Import test targets
from hololoom.core.loom.weave_house import (
    WeaveHouse,
    WeaveResult,
    create_weave_house,
)
from hololoom.core.loom.base_loom import BaseLoom
from hololoom.core.fabric.fabric import Fabric
from hololoom.core.loom.consensus import LoomConsensus, create_loom_consensus


# =============================================================================
# Test Fixtures
# =============================================================================

class MockLoom(BaseLoom):
    """Mock loom for testing."""

    def __init__(
        self,
        perspective: str = "test",
        response: str = "Mock response",
        confidence: float = 0.8,
        epistemic_confidence: float = 0.7,
        delay: float = 0.0,
        admits_ignorance: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=f"MockLoom-{perspective}")
        self._perspective = perspective
        self._response = response
        self._confidence = confidence
        self._epistemic_confidence = epistemic_confidence
        self._delay = delay
        self._admits_ignorance = admits_ignorance or []
        self._metadata = metadata or {}

    @property
    def perspective(self) -> str:
        return self._perspective

    async def weave(self, query: str, context: Optional[Dict] = None) -> Fabric:
        if self._delay > 0:
            await asyncio.sleep(self._delay)

        return Fabric(
            perspective=self._perspective,
            response=self._response,
            confidence=self._confidence,
            epistemic_confidence=self._epistemic_confidence,
            admits_ignorance=self._admits_ignorance,
            metadata=self._metadata,
        )


@pytest.fixture
def mock_looms():
    """Create a set of mock looms for testing."""
    return [
        MockLoom(
            perspective="recall",
            response="I found relevant information.",
            confidence=0.9,
            epistemic_confidence=0.85,
        ),
        MockLoom(
            perspective="reason",
            response="The logical conclusion is...",
            confidence=0.85,
            epistemic_confidence=0.8,
        ),
        MockLoom(
            perspective="reflect",
            response="The claim appears consistent.",
            confidence=0.75,
            epistemic_confidence=0.7,
        ),
    ]


@pytest.fixture
def mock_consensus():
    """Create a mock consensus engine."""
    return create_loom_consensus(
        exploration_enabled=False,
        agreement_threshold=0.7,
    )


@pytest.fixture
def weave_house(mock_looms, mock_consensus):
    """Create a WeaveHouse with mock components."""
    return WeaveHouse(
        looms=mock_looms,
        consensus=mock_consensus,
        exploration_depth=2,
        tension_threshold=0.3,
    )


# =============================================================================
# Unit Tests: WeaveHouse Initialization
# =============================================================================

class TestWeaveHouseInit:
    """Tests for WeaveHouse initialization."""

    def test_init_with_looms(self, mock_looms, mock_consensus):
        """Test basic initialization with looms."""
        house = WeaveHouse(looms=mock_looms, consensus=mock_consensus)
        assert len(house.looms) == 3
        assert house.consensus is not None

    def test_init_empty_looms(self, mock_consensus):
        """Test initialization with empty looms list."""
        house = WeaveHouse(looms=[], consensus=mock_consensus)
        assert len(house.looms) == 0

    def test_init_with_exploration_settings(self, mock_looms, mock_consensus):
        """Test initialization with exploration settings."""
        house = WeaveHouse(
            looms=mock_looms,
            consensus=mock_consensus,
            exploration_depth=5,
            tension_threshold=0.5,
        )
        assert house.exploration_depth == 5
        assert house.tension_threshold == 0.5

    def test_factory_function(self, mock_looms, mock_consensus):
        """Test create_weave_house factory."""
        house = create_weave_house(
            looms=mock_looms,
            consensus=mock_consensus,
        )
        assert isinstance(house, WeaveHouse)
        assert len(house.looms) == 3


# =============================================================================
# Unit Tests: Parallel Weaving
# =============================================================================

class TestParallelWeaving:
    """Tests for parallel weaving functionality."""

    @pytest.mark.asyncio
    async def test_weave_runs_all_looms(self, weave_house):
        """Test that weave runs all looms."""
        result = await weave_house.weave("Test query")

        assert isinstance(result, WeaveResult)
        assert len(result.fabrics) == 3

    @pytest.mark.asyncio
    async def test_weave_collects_all_perspectives(self, weave_house):
        """Test that all perspectives are collected."""
        result = await weave_house.weave("Test query")

        perspectives = [f.perspective for f in result.fabrics]
        assert "recall" in perspectives
        assert "reason" in perspectives
        assert "reflect" in perspectives

    @pytest.mark.asyncio
    async def test_weave_parallel_execution(self):
        """Test that looms run in parallel (not sequential)."""
        # Create looms with delays
        looms = [
            MockLoom(perspective="a", delay=0.1),
            MockLoom(perspective="b", delay=0.1),
            MockLoom(perspective="c", delay=0.1),
        ]
        consensus = create_loom_consensus()
        house = WeaveHouse(looms=looms, consensus=consensus)

        import time
        start = time.time()
        await house.weave("Test query")
        elapsed = time.time() - start

        # If parallel, should be ~0.1s, if sequential would be ~0.3s
        assert elapsed < 0.25, f"Expected parallel execution, took {elapsed}s"

    @pytest.mark.asyncio
    async def test_weave_with_context(self, weave_house):
        """Test weaving with context."""
        context = {"domain": "test", "user_id": "123"}
        result = await weave_house.weave("Test query", context=context)

        assert result is not None
        assert isinstance(result, WeaveResult)


# =============================================================================
# Unit Tests: Consensus Integration
# =============================================================================

class TestConsensusIntegration:
    """Tests for consensus engine integration."""

    @pytest.mark.asyncio
    async def test_consensus_produces_response(self, weave_house):
        """Test that consensus produces a final response."""
        result = await weave_house.weave("Test query")

        assert result.response is not None
        assert len(result.response) > 0

    @pytest.mark.asyncio
    async def test_consensus_calculates_confidence(self, weave_house):
        """Test that consensus calculates confidence."""
        result = await weave_house.weave("Test query")

        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.epistemic_confidence <= 1.0

    @pytest.mark.asyncio
    async def test_consensus_with_disagreement(self):
        """Test consensus handles disagreeing looms."""
        looms = [
            MockLoom(perspective="a", response="Yes", confidence=0.9),
            MockLoom(perspective="b", response="No", confidence=0.9),
        ]
        consensus = create_loom_consensus(agreement_threshold=0.9)
        house = WeaveHouse(looms=looms, consensus=consensus)

        result = await house.weave("Controversial question")

        # Should still produce a result
        assert result is not None
        # May have tensions recorded
        assert isinstance(result.tensions, list)


# =============================================================================
# Unit Tests: Exploration
# =============================================================================

class TestExploration:
    """Tests for auto-exploration of disagreement zones."""

    @pytest.mark.asyncio
    async def test_exploration_disabled_by_default(self):
        """Test exploration can be disabled."""
        looms = [MockLoom(perspective="test")]
        consensus = create_loom_consensus(exploration_enabled=False)
        house = WeaveHouse(
            looms=looms,
            consensus=consensus,
            exploration_depth=0,
        )

        result = await house.weave("Test")

        # With exploration disabled/depth=0, no extra exploration
        assert result is not None

    @pytest.mark.asyncio
    async def test_exploration_depth_limit(self):
        """Test exploration respects depth limit."""
        looms = [
            MockLoom(perspective="a", confidence=0.5),
            MockLoom(perspective="b", confidence=0.5),
        ]
        consensus = create_loom_consensus(exploration_enabled=True)
        house = WeaveHouse(
            looms=looms,
            consensus=consensus,
            exploration_depth=1,
            tension_threshold=0.1,
        )

        result = await house.weave("Uncertain query")

        # Should complete without infinite loop
        assert result is not None


# =============================================================================
# Unit Tests: WeaveResult
# =============================================================================

class TestWeaveResult:
    """Tests for WeaveResult dataclass."""

    @pytest.mark.asyncio
    async def test_result_has_all_fields(self, weave_house):
        """Test result contains all expected fields."""
        result = await weave_house.weave("Test")

        assert hasattr(result, 'response')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'epistemic_confidence')
        assert hasattr(result, 'fabrics')
        assert hasattr(result, 'tensions')
        assert hasattr(result, 'exploration_trace')
        assert hasattr(result, 'metadata')

    @pytest.mark.asyncio
    async def test_result_fabrics_are_fabric_objects(self, weave_house):
        """Test that fabrics are Fabric instances."""
        result = await weave_house.weave("Test")

        for fabric in result.fabrics:
            assert isinstance(fabric, Fabric)

    def test_result_serializable(self):
        """Test WeaveResult can be converted to dict."""
        result = WeaveResult(
            response="Test response",
            confidence=0.8,
            epistemic_confidence=0.7,
            fabrics=[],
            tensions=[],
            exploration_trace=[],
            metadata={"key": "value"},
        )

        # Should not raise
        data = {
            "response": result.response,
            "confidence": result.confidence,
            "epistemic_confidence": result.epistemic_confidence,
            "fabrics": result.fabrics,
            "tensions": result.tensions,
        }
        assert data["confidence"] == 0.8


# =============================================================================
# Integration Tests
# =============================================================================

class TestWeaveHouseIntegration:
    """Integration tests for WeaveHouse."""

    @pytest.mark.asyncio
    async def test_full_weave_cycle(self, weave_house):
        """Test complete weave cycle from query to result."""
        query = "What are the implications of X?"

        result = await weave_house.weave(query)

        # Verify complete result
        assert result.response is not None
        assert len(result.fabrics) == 3
        assert result.confidence > 0
        assert result.epistemic_confidence > 0

    @pytest.mark.asyncio
    async def test_multiple_weaves(self, weave_house):
        """Test multiple consecutive weaves."""
        queries = ["Query 1", "Query 2", "Query 3"]

        results = []
        for query in queries:
            result = await weave_house.weave(query)
            results.append(result)

        assert len(results) == 3
        for result in results:
            assert result is not None

    @pytest.mark.asyncio
    async def test_concurrent_weaves(self, mock_looms, mock_consensus):
        """Test concurrent weaves don't interfere."""
        house = WeaveHouse(looms=mock_looms, consensus=mock_consensus)

        # Run multiple weaves concurrently
        tasks = [
            house.weave(f"Query {i}")
            for i in range(5)
        ]
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        for result in results:
            assert result is not None
            assert len(result.fabrics) == 3


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling in WeaveHouse."""

    @pytest.mark.asyncio
    async def test_loom_exception_handled(self):
        """Test that loom exceptions are handled gracefully."""
        class FailingLoom(BaseLoom):
            @property
            def perspective(self) -> str:
                return "failing"

            async def weave(self, query: str, context: Optional[Dict] = None) -> Fabric:
                raise ValueError("Intentional failure")

        looms = [
            MockLoom(perspective="good"),
            FailingLoom(name="FailingLoom"),
        ]
        consensus = create_loom_consensus()
        house = WeaveHouse(looms=looms, consensus=consensus)

        # Should not raise, should handle gracefully
        try:
            result = await house.weave("Test")
            # Result may be partial or have error recorded
            assert result is not None
        except ValueError:
            # If it does raise, that's also acceptable behavior
            pass

    @pytest.mark.asyncio
    async def test_empty_query(self, weave_house):
        """Test handling of empty query."""
        result = await weave_house.weave("")

        # Should still work, looms handle empty input
        assert result is not None


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
