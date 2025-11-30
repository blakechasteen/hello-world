"""
Tests for HoloLoom Adapter - Phase 1 MVP

Tests the integration between NeuroHood dreams and HoloLoom memory.

Run with:
    cd NeuroHood/dreams
    PYTHONPATH=../.. python -m pytest tests/test_hololoom_adapter.py -v

Author: Phase 1 MVP Integration Tests
Date: November 2025
"""

import asyncio
import pytest
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional
import sys
from pathlib import Path

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Mock Classes (for testing without full NeuroHood)
# =============================================================================

class MockNarrativeArchetype(Enum):
    JOURNEY = "journey"
    CONFRONTATION = "confrontation"
    TRANSFORMATION = "transformation"


class MockDreamMood(Enum):
    NIGHTMARE = "nightmare"
    WONDER = "wonder"
    MYSTERIOUS = "mysterious"


@dataclass
class MockNarrativeAct:
    act_number: int
    act_type: str
    description: str
    symbols: List[Dict]
    emotional_intensity: float
    literary_echo: Optional[str] = None


@dataclass
class MockDreamNarrative:
    title: str
    archetype: MockNarrativeArchetype
    mood: MockDreamMood
    acts: List[MockNarrativeAct]
    protagonist_state: Dict[str, float]
    symbols_used: List[str]
    literary_inspirations: List[str]
    jungian_interpretation: str
    generation_timestamp: datetime = field(default_factory=datetime.now)

    def full_narrative(self) -> str:
        parts = [f"# {self.title}\n"]
        for act in self.acts:
            parts.append(f"Act {act.act_number}: {act.description}")
        return "\n".join(parts)


# =============================================================================
# Tests
# =============================================================================

class TestHoloLoomAdapter:
    """Test suite for HoloLoom adapter."""

    def create_mock_narrative(self) -> MockDreamNarrative:
        """Create a mock dream narrative for testing."""
        acts = [
            MockNarrativeAct(
                act_number=1,
                act_type="setup",
                description="You find yourself in a vast ocean, waves crashing around you.",
                symbols=[{"symbol_id": "ocean", "description": "vast sea"}],
                emotional_intensity=0.6,
                literary_echo="The Odyssey"
            ),
            MockNarrativeAct(
                act_number=2,
                act_type="climax",
                description="A great whale emerges from the depths, eyes wise and ancient.",
                symbols=[{"symbol_id": "whale", "description": "ancient creature"}],
                emotional_intensity=0.9
            ),
            MockNarrativeAct(
                act_number=3,
                act_type="resolution",
                description="You ride upon its back to a shore of golden light.",
                symbols=[{"symbol_id": "light", "description": "golden radiance"}],
                emotional_intensity=0.7
            )
        ]

        return MockDreamNarrative(
            title="Journey Through the Deep",
            archetype=MockNarrativeArchetype.JOURNEY,
            mood=MockDreamMood.WONDER,
            acts=acts,
            protagonist_state={"hope": 0.6, "fear": 0.3, "transformation": 0.5},
            symbols_used=["ocean", "whale", "light"],
            literary_inspirations=["The Odyssey", "Moby Dick"],
            jungian_interpretation="This journey represents the individuation process."
        )

    def test_import_adapter(self):
        """Test that adapter can be imported."""
        from hololoom_adapter import (
            store_dream_narrative,
            store_dream_scene,
            recall_dreams,
            recall_resident_dreams,
            reflect_on_dream,
            is_hololoom_available,
            DreamMemoryBridge,
        )
        # Just check imports work
        assert callable(store_dream_narrative)
        assert callable(recall_dreams)
        assert callable(is_hololoom_available)

    def test_graceful_degradation(self):
        """Test that adapter works gracefully without HoloLoom."""
        from hololoom_adapter import is_hololoom_available

        # This should return True or False without crashing
        available = is_hololoom_available()
        assert isinstance(available, bool)

    @pytest.mark.asyncio
    async def test_store_without_hololoom(self):
        """Test store returns None when HoloLoom unavailable."""
        from hololoom_adapter import store_dream_narrative

        narrative = self.create_mock_narrative()
        result = await store_dream_narrative(None, narrative, "resident_001")

        # Should return None, not crash
        assert result is None

    @pytest.mark.asyncio
    async def test_recall_without_hololoom(self):
        """Test recall returns empty list when HoloLoom unavailable."""
        from hololoom_adapter import recall_dreams

        result = await recall_dreams(None, "water dreams")

        # Should return empty list, not crash
        assert result == []

    @pytest.mark.asyncio
    async def test_reflect_without_hololoom(self):
        """Test reflect handles missing HoloLoom gracefully."""
        from hololoom_adapter import reflect_on_dream

        # Should not crash
        await reflect_on_dream(None, [], {"meaningful": True})

    def test_narrative_to_content(self):
        """Test narrative conversion to content string."""
        from hololoom_adapter import _narrative_to_content

        narrative = self.create_mock_narrative()
        content = _narrative_to_content(narrative)

        # Check key elements are present
        assert "Journey Through the Deep" in content
        assert "wonder" in content
        assert "journey" in content
        assert "ocean" in content
        assert "whale" in content
        assert "light" in content
        assert "individuation" in content

    @pytest.mark.asyncio
    async def test_bridge_context_manager(self):
        """Test DreamMemoryBridge context manager."""
        from hololoom_adapter import DreamMemoryBridge

        async with DreamMemoryBridge() as bridge:
            # Should work even if HoloLoom not available
            available = bridge.available

            # available should be bool
            assert isinstance(available, bool)

            # get_metrics should return dict
            metrics = bridge.get_metrics()
            assert isinstance(metrics, dict)
            assert "available" in metrics


class TestHoloLoomIntegration:
    """Integration tests (run only when HoloLoom is available)."""

    @pytest.fixture
    def check_hololoom(self):
        """Skip if HoloLoom not available."""
        from hololoom_adapter import is_hololoom_available
        if not is_hololoom_available():
            pytest.skip("HoloLoom not available")

    def create_mock_narrative(self) -> MockDreamNarrative:
        """Create a mock dream narrative."""
        acts = [
            MockNarrativeAct(
                act_number=1,
                act_type="setup",
                description="A forest of crystalline trees stretches before you.",
                symbols=[{"symbol_id": "crystal_forest", "description": "trees of light"}],
                emotional_intensity=0.7
            )
        ]

        return MockDreamNarrative(
            title="The Crystal Grove",
            archetype=MockNarrativeArchetype.TRANSFORMATION,
            mood=MockDreamMood.MYSTERIOUS,
            acts=acts,
            protagonist_state={"transformation": 0.8, "mystery": 0.6},
            symbols_used=["crystal_forest"],
            literary_inspirations=["Alice in Wonderland"],
            jungian_interpretation="Crystal represents clarity emerging from chaos."
        )

    @pytest.mark.asyncio
    async def test_full_integration(self, check_hololoom):
        """Test full store -> recall -> reflect cycle."""
        from hololoom_adapter import DreamMemoryBridge

        async with DreamMemoryBridge() as bridge:
            if not bridge.available:
                pytest.skip("HoloLoom not available")

            # Store a dream
            narrative = self.create_mock_narrative()
            memory = await bridge.store(narrative, "test_resident")

            assert memory is not None
            assert "Crystal Grove" in memory.text

            # Recall the dream
            dreams = await bridge.recall("crystal")

            assert len(dreams) >= 1
            assert any("crystal" in d.text.lower() for d in dreams)

            # Reflect on the dream
            await bridge.reflect(dreams, {"meaningful": True, "emotional_impact": 0.9})

            # Check metrics
            metrics = bridge.get_metrics()
            assert metrics.get("available") is True


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
