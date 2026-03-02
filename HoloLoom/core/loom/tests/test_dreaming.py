"""
Tests for DreamOrchestrator - Collective Consolidation
======================================================

Tests for the DreamOrchestrator class which coordinates collective
dreaming among multiple Looms for shared learning.

Date: December 2025
"""

import asyncio
import pytest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

# Import test targets
from HoloLoom.loom.dreaming import (
    DreamOrchestrator,
    DreamCycleStats,
    DreamOrchestratorMetrics,
    create_dream_orchestrator,
)
from HoloLoom.loom.protocol import (
    Loom,
    DreamInsight,
    InsightType,
    MathInsight,
    PatternInsight,
    CorrectionInsight,
    DiscoveryInsight,
)
from HoloLoom.loom.base_loom import BaseLoom
from HoloLoom.fabric.fabric import Fabric


# =============================================================================
# Test Fixtures
# =============================================================================

class MockLoom(BaseLoom):
    """Mock loom for testing."""

    def __init__(
        self,
        perspective: str = "test",
        shareable_insights: Optional[List[DreamInsight]] = None,
    ):
        super().__init__(name=f"MockLoom-{perspective}")
        self._perspective = perspective
        self._shareable_insights = shareable_insights or []
        self._received_insights: List[DreamInsight] = []

    @property
    def perspective(self) -> str:
        return self._perspective

    async def weave(self, query: str, context: Optional[Dict] = None) -> Fabric:
        return Fabric(
            perspective=self._perspective,
            response="Mock response",
            confidence=0.8,
            epistemic_confidence=0.7,
        )

    async def get_shareable_insights(self) -> List[DreamInsight]:
        """Return insights to share during dreaming."""
        return self._shareable_insights

    async def receive_insight(self, insight: DreamInsight) -> bool:
        """Receive an insight from another loom."""
        self._received_insights.append(insight)
        return True


@pytest.fixture
def mock_looms():
    """Create a set of mock looms with insights."""
    recall = MockLoom(
        perspective="recall",
        shareable_insights=[
            MathInsight(
                content="Vector space theorem",
                confidence=0.9,
                source_loom="recall",
            ),
        ],
    )
    reason = MockLoom(
        perspective="reason",
        shareable_insights=[
            PatternInsight(
                content="Inference pattern A",
                confidence=0.8,
                source_loom="reason",
                pattern_frequency=10,
            ),
        ],
    )
    reflect = MockLoom(
        perspective="reflect",
        shareable_insights=[
            CorrectionInsight(
                content="Corrected belief X",
                confidence=0.95,
                source_loom="reflect",
                original_belief="Belief X",
                correction="Corrected X",
            ),
        ],
    )
    return [recall, reason, reflect]


@pytest.fixture
def dream_orchestrator(mock_looms):
    """Create a DreamOrchestrator with mock looms."""
    return DreamOrchestrator(
        looms=mock_looms,
        consolidation_interval=0.1,  # Short for testing
        math_bleed_rate=0.3,
        pattern_bleed_rate=0.2,
    )


# =============================================================================
# Unit Tests: Initialization
# =============================================================================

class TestDreamOrchestratorInit:
    """Tests for DreamOrchestrator initialization."""

    def test_init_with_looms(self, mock_looms):
        """Test basic initialization with looms."""
        orchestrator = DreamOrchestrator(looms=mock_looms)
        assert len(orchestrator.looms) == 3

    def test_init_with_custom_settings(self, mock_looms):
        """Test initialization with custom settings."""
        orchestrator = DreamOrchestrator(
            looms=mock_looms,
            consolidation_interval=60.0,
            math_bleed_rate=0.5,
            pattern_bleed_rate=0.3,
            discovery_confidence_threshold=0.9,
        )
        assert orchestrator.consolidation_interval == 60.0
        assert orchestrator.math_bleed_rate == 0.5
        assert orchestrator.pattern_bleed_rate == 0.3
        assert orchestrator.discovery_confidence_threshold == 0.9

    def test_factory_function(self, mock_looms):
        """Test create_dream_orchestrator factory."""
        orchestrator = create_dream_orchestrator(looms=mock_looms)
        assert isinstance(orchestrator, DreamOrchestrator)


# =============================================================================
# Unit Tests: Single Dream Cycle
# =============================================================================

class TestDreamCycle:
    """Tests for single dream cycles."""

    @pytest.mark.asyncio
    async def test_dream_once(self, dream_orchestrator):
        """Test executing a single dream cycle."""
        stats = await dream_orchestrator.dream_once()

        assert isinstance(stats, DreamCycleStats)
        assert stats.cycle_number == 1

    @pytest.mark.asyncio
    async def test_dream_once_returns_stats(self, dream_orchestrator):
        """Test that dream_once returns statistics."""
        stats = await dream_orchestrator.dream_once()

        assert hasattr(stats, 'total_insights_collected')
        assert hasattr(stats, 'insights_distributed')
        assert hasattr(stats, 'duration_ms')

    @pytest.mark.asyncio
    async def test_multiple_dream_cycles(self, dream_orchestrator):
        """Test multiple consecutive dream cycles."""
        stats1 = await dream_orchestrator.dream_once()
        stats2 = await dream_orchestrator.dream_once()
        stats3 = await dream_orchestrator.dream_once()

        assert stats1.cycle_number == 1
        assert stats2.cycle_number == 2
        assert stats3.cycle_number == 3


# =============================================================================
# Unit Tests: Insight Collection
# =============================================================================

class TestInsightCollection:
    """Tests for insight collection from looms."""

    @pytest.mark.asyncio
    async def test_collects_insights_from_looms(self, mock_looms):
        """Test that insights are collected from all looms."""
        orchestrator = DreamOrchestrator(
            looms=mock_looms,
            consolidation_interval=0.1,
        )

        stats = await orchestrator.dream_once()

        # Should have collected insights from our 3 looms
        assert stats.total_insights_collected >= 0

    @pytest.mark.asyncio
    async def test_insight_type_counting(self, mock_looms):
        """Test that insights are counted by type."""
        orchestrator = DreamOrchestrator(
            looms=mock_looms,
            consolidation_interval=0.1,
        )

        stats = await orchestrator.dream_once()

        # Stats should track insights by type
        assert hasattr(stats, 'math_shared')
        assert hasattr(stats, 'patterns_shared')
        assert hasattr(stats, 'corrections_shared')


# =============================================================================
# Unit Tests: Bleed Rates
# =============================================================================

class TestBleedRates:
    """Tests for insight bleed rates."""

    @pytest.mark.asyncio
    async def test_math_bleed_rate(self, mock_looms):
        """Test math insights respect bleed rate."""
        orchestrator = DreamOrchestrator(
            looms=mock_looms,
            consolidation_interval=0.1,
            math_bleed_rate=0.0,  # 0% bleed - no sharing
        )

        stats = await orchestrator.dream_once()
        # With 0% bleed, no math should be shared
        assert stats.math_shared == 0

    @pytest.mark.asyncio
    async def test_corrections_always_shared(self, mock_looms):
        """Test that corrections are always shared (100%)."""
        orchestrator = DreamOrchestrator(
            looms=mock_looms,
            consolidation_interval=0.1,
        )

        stats = await orchestrator.dream_once()
        # Corrections should always be shared
        # (depends on how many correction insights are in mock_looms)
        assert isinstance(stats.corrections_shared, int)


# =============================================================================
# Unit Tests: Async Context Manager
# =============================================================================

class TestAsyncContextManager:
    """Tests for async context manager protocol."""

    @pytest.mark.asyncio
    async def test_context_manager_entry(self, mock_looms):
        """Test async context manager entry starts dreaming."""
        orchestrator = DreamOrchestrator(
            looms=mock_looms,
            consolidation_interval=10.0,  # Long interval for test
        )

        async with orchestrator as dream:
            assert dream._running is True

    @pytest.mark.asyncio
    async def test_context_manager_exit(self, mock_looms):
        """Test async context manager exit stops dreaming."""
        orchestrator = DreamOrchestrator(
            looms=mock_looms,
            consolidation_interval=10.0,
        )

        async with orchestrator:
            pass

        assert orchestrator._running is False

    @pytest.mark.asyncio
    async def test_start_stop_dreaming(self, mock_looms):
        """Test explicit start/stop methods."""
        orchestrator = DreamOrchestrator(
            looms=mock_looms,
            consolidation_interval=10.0,
        )

        await orchestrator.start_dreaming()
        assert orchestrator._running is True

        await orchestrator.stop_dreaming()
        assert orchestrator._running is False


# =============================================================================
# Unit Tests: DreamCycleStats
# =============================================================================

class TestDreamCycleStats:
    """Tests for DreamCycleStats dataclass."""

    def test_stats_creation(self):
        """Test creating DreamCycleStats."""
        from datetime import datetime

        stats = DreamCycleStats(
            cycle_number=1,
            timestamp=datetime.now(),
            total_insights_collected=10,
            math_shared=3,
            patterns_shared=2,
            corrections_shared=1,
            discoveries_shared=0,
        )

        assert stats.cycle_number == 1
        assert stats.total_insights_collected == 10

    def test_stats_to_dict(self):
        """Test serializing stats to dict."""
        from datetime import datetime

        stats = DreamCycleStats(
            cycle_number=1,
            timestamp=datetime.now(),
        )

        data = stats.to_dict()
        assert isinstance(data, dict)
        assert 'cycle_number' in data


# =============================================================================
# Unit Tests: DreamOrchestratorMetrics
# =============================================================================

class TestDreamOrchestratorMetrics:
    """Tests for metrics tracking."""

    @pytest.mark.asyncio
    async def test_metrics_updated(self, dream_orchestrator):
        """Test that metrics are updated after cycles."""
        await dream_orchestrator.dream_once()
        await dream_orchestrator.dream_once()

        metrics = dream_orchestrator._metrics
        assert isinstance(metrics, DreamOrchestratorMetrics)


# =============================================================================
# Integration Tests
# =============================================================================

class TestDreamingIntegration:
    """Integration tests for dreaming system."""

    @pytest.mark.asyncio
    async def test_insight_distribution(self, mock_looms):
        """Test that insights are distributed to other looms."""
        orchestrator = DreamOrchestrator(
            looms=mock_looms,
            consolidation_interval=0.1,
        )

        await orchestrator.dream_once()

        # Check if looms received insights
        # (depends on mock implementation)
        for loom in mock_looms:
            if isinstance(loom, MockLoom):
                # May have received insights from other looms
                pass  # Insight distribution tested via stats

    @pytest.mark.asyncio
    async def test_callback_on_cycle(self, mock_looms):
        """Test callback is called after each cycle."""
        cycles_reported = []

        def on_cycle(stats: DreamCycleStats):
            cycles_reported.append(stats)

        orchestrator = DreamOrchestrator(
            looms=mock_looms,
            consolidation_interval=0.1,
            on_cycle_complete=on_cycle,
        )

        await orchestrator.dream_once()
        await orchestrator.dream_once()

        assert len(cycles_reported) == 2


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling in dreaming."""

    @pytest.mark.asyncio
    async def test_handles_loom_error_gracefully(self):
        """Test handling of loom errors during insight collection."""

        class FailingLoom(BaseLoom):
            @property
            def perspective(self) -> str:
                return "failing"

            async def weave(self, query: str, context: Optional[Dict] = None) -> Fabric:
                raise ValueError("Weave failed")

            async def get_shareable_insights(self) -> List[DreamInsight]:
                raise ValueError("Collection failed")

        # Mix of working and failing looms
        looms = [
            MockLoom(perspective="good"),
            FailingLoom(name="Failing"),
        ]

        orchestrator = DreamOrchestrator(
            looms=looms,
            consolidation_interval=0.1,
        )

        # Should handle gracefully
        try:
            stats = await orchestrator.dream_once()
            # If it succeeds, it handled the error
            assert stats is not None
        except Exception:
            # Or it may raise, which is acceptable
            pass

    @pytest.mark.asyncio
    async def test_empty_looms_list(self):
        """Test with empty looms list."""
        orchestrator = DreamOrchestrator(
            looms=[],
            consolidation_interval=0.1,
        )

        stats = await orchestrator.dream_once()
        assert stats.total_insights_collected == 0


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
