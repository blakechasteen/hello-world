"""
Tests for Phase 2 NeuroHood-HoloLoom Integration
================================================

Tests DreamSpinner, DreamDepartment, and HoloLoomBridge.

Run with:
    cd mythRL
    PYTHONPATH=. python -m pytest NeuroHood/dreams/tests/test_phase2_integration.py -v

Author: Phase 2 Integration Tests
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

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Mock Classes (for testing without full NeuroHood setup)
# =============================================================================

class MockNarrativeArchetype(Enum):
    JOURNEY = "journey"
    CONFRONTATION = "confrontation"
    TRANSFORMATION = "transformation"


class MockDreamMood(Enum):
    NIGHTMARE = "nightmare"
    WONDER = "wonder"
    MYSTERIOUS = "mysterious"
    TRANSFORMATIVE = "transformative"


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


def create_mock_narrative() -> MockDreamNarrative:
    """Create a valid mock dream narrative."""
    return MockDreamNarrative(
        title="Journey Through the Deep Ocean",
        archetype=MockNarrativeArchetype.JOURNEY,
        mood=MockDreamMood.WONDER,
        acts=[
            MockNarrativeAct(
                act_number=1,
                act_type="setup",
                description="You find yourself in a vast ocean, waves crashing around you. "
                           "The water is warm and welcoming despite its depth.",
                symbols=[{"symbol_id": "ocean", "description": "vast sea"}],
                emotional_intensity=0.6,
                literary_echo="The Odyssey"
            ),
            MockNarrativeAct(
                act_number=2,
                act_type="climax",
                description="A great whale emerges from the depths, eyes ancient and wise. "
                           "It speaks without words, communicating through pure emotion.",
                symbols=[{"symbol_id": "whale", "description": "ancient creature"}],
                emotional_intensity=0.9
            ),
            MockNarrativeAct(
                act_number=3,
                act_type="resolution",
                description="You ride upon its back to a shore of golden light. "
                           "The journey has transformed you.",
                symbols=[{"symbol_id": "light", "description": "golden radiance"}],
                emotional_intensity=0.7
            )
        ],
        protagonist_state={"hope": 0.7, "wonder": 0.8, "transformation": 0.5},
        symbols_used=["ocean", "whale", "light", "water"],
        literary_inspirations=["The Odyssey", "Moby Dick", "Life of Pi"],
        jungian_interpretation="This journey represents the individuation process - "
                               "descending into the unconscious (ocean) and emerging transformed."
    )


def create_invalid_narrative() -> MockDreamNarrative:
    """Create an invalid narrative for verification testing."""
    return MockDreamNarrative(
        title="",  # Invalid: empty title
        archetype=MockNarrativeArchetype.JOURNEY,
        mood=MockDreamMood.WONDER,
        acts=[],  # Invalid: no acts
        protagonist_state={},
        symbols_used=[],
        literary_inspirations=[],
        jungian_interpretation=""
    )


# =============================================================================
# Test DreamSpinner
# =============================================================================

class TestDreamSpinner:
    """Test suite for DreamSpinner."""

    def test_import_spinner(self):
        """Test that DreamSpinner can be imported."""
        try:
            from HoloLoom.spinningWheel.dream_spinner import (
                DreamSpinner,
                DreamSpinnerConfig
            )
            assert DreamSpinner is not None
            assert DreamSpinnerConfig is not None
        except ImportError as e:
            pytest.skip(f"DreamSpinner not available: {e}")

    def test_spinner_instantiation(self):
        """Test DreamSpinner instantiation."""
        try:
            from HoloLoom.spinningWheel.dream_spinner import (
                DreamSpinner,
                DreamSpinnerConfig
            )

            config = DreamSpinnerConfig()
            spinner = DreamSpinner(config)

            assert spinner.name == "dream"
            assert spinner.config is not None
        except ImportError:
            pytest.skip("DreamSpinner not available")

    def test_spinner_capabilities(self):
        """Test spinner capabilities."""
        try:
            from HoloLoom.spinningWheel.dream_spinner import DreamSpinner

            spinner = DreamSpinner()
            caps = spinner.get_capabilities()

            # Check supported formats/types are defined
            assert caps.basic_processing is True
            assert caps.importance_scoring is True
        except ImportError:
            pytest.skip("DreamSpinner not available")

    @pytest.mark.asyncio
    async def test_spin_narrative(self):
        """Test spinning a dream narrative into MemoryShards."""
        try:
            from HoloLoom.spinningWheel.dream_spinner import DreamSpinner

            spinner = DreamSpinner()
            narrative = create_mock_narrative()

            result = await spinner.spin(narrative, resident_id="test_resident")

            assert result is not None
            assert result.shards is not None
            assert len(result.shards) >= 1  # At least main shard

            # Check first shard has expected fields
            shard = result.shards[0]
            assert hasattr(shard, 'id')
            assert hasattr(shard, 'text')
            assert "Journey" in shard.text or "ocean" in shard.text.lower()

        except ImportError:
            pytest.skip("DreamSpinner not available")

    def test_importance_scoring(self):
        """Test dream importance scoring."""
        try:
            from HoloLoom.spinningWheel.dream_spinner import DreamSpinner

            spinner = DreamSpinner()
            narrative = create_mock_narrative()

            score = spinner.score_importance(narrative)

            assert score is not None
            assert hasattr(score, 'score')  # ImportanceScore uses 'score' attribute
            assert 0.0 <= score.score <= 1.0
            assert score.score > 0.5  # Valid narrative should score well

        except ImportError:
            pytest.skip("DreamSpinner not available")


# =============================================================================
# Test DS-STAR Verification
# =============================================================================

class TestDSStarVerification:
    """Test suite for DS-STAR verification."""

    def test_import_verifier(self):
        """Test that DreamDSStarVerifier can be imported."""
        try:
            from HoloLoom.departments.dream import (
                DreamDSStarVerifier,
                DSStarCheck,
                VerificationResult
            )
            assert DreamDSStarVerifier is not None
            assert DSStarCheck is not None
        except ImportError as e:
            pytest.skip(f"DreamDepartment not available: {e}")

    def test_verification_pass(self):
        """Test verification passes for valid narrative."""
        try:
            from HoloLoom.departments.dream import DreamDSStarVerifier

            verifier = DreamDSStarVerifier(strict=False)
            narrative = create_mock_narrative()

            result = verifier.verify(narrative)

            assert result.passed is True
            assert result.confidence > 0.5
            assert len(result.failed_checks) < 3  # Most checks should pass

        except ImportError:
            pytest.skip("DreamDepartment not available")

    def test_verification_fail(self):
        """Test verification fails for invalid narrative."""
        try:
            from HoloLoom.departments.dream import DreamDSStarVerifier

            verifier = DreamDSStarVerifier(strict=False)
            narrative = create_invalid_narrative()

            result = verifier.verify(narrative)

            # Should fail multiple checks
            assert result.confidence < 0.5
            assert len(result.failed_checks) >= 2

        except ImportError:
            pytest.skip("DreamDepartment not available")

    def test_strict_verification(self):
        """Test strict mode requires all checks."""
        try:
            from HoloLoom.departments.dream import DreamDSStarVerifier

            verifier = DreamDSStarVerifier(strict=True)
            narrative = create_mock_narrative()

            result = verifier.verify(narrative)

            # Strict mode: all checks must pass
            if result.passed:
                assert len(result.failed_checks) == 0

        except ImportError:
            pytest.skip("DreamDepartment not available")

    def test_ds_star_dimensions(self):
        """Test all 5 DS-STAR dimensions are checked."""
        try:
            from HoloLoom.departments.dream import DreamDSStarVerifier, DSStarCheck

            verifier = DreamDSStarVerifier()
            narrative = create_mock_narrative()

            result = verifier.verify(narrative)

            # All 5 dimensions should be present
            assert DSStarCheck.DOMAIN in result.checks
            assert DSStarCheck.SENSIBILITY in result.checks
            assert DSStarCheck.TEMPORAL in result.checks
            assert DSStarCheck.ARGUMENT in result.checks
            assert DSStarCheck.REFERENCE in result.checks

            # All should have scores
            assert len(result.scores) == 5

        except ImportError:
            pytest.skip("DreamDepartment not available")


# =============================================================================
# Test DreamDepartment
# =============================================================================

class TestDreamDepartment:
    """Test suite for DreamDepartment."""

    def test_import_department(self):
        """Test that DreamDepartment can be imported."""
        try:
            from HoloLoom.departments.dream import (
                DreamDepartment,
                DreamAction,
                create_dream_department
            )
            assert DreamDepartment is not None
            assert DreamAction is not None
        except ImportError as e:
            pytest.skip(f"DreamDepartment not available: {e}")

    def test_department_instantiation(self):
        """Test DreamDepartment instantiation."""
        try:
            from HoloLoom.departments.dream import create_dream_department

            dept = create_dream_department(
                verify_on_store=True,
                strict_verification=False
            )

            assert dept is not None
            assert dept.verify_on_store is True
            assert dept.strict_verification is False

        except ImportError:
            pytest.skip("DreamDepartment not available")

    @pytest.mark.asyncio
    async def test_department_lifecycle(self):
        """Test department start/stop lifecycle."""
        try:
            from HoloLoom.departments.dream import create_dream_department

            dept = create_dream_department()

            await dept.start()
            assert dept.started is True

            health = await dept.health_check()
            assert health is True

            await dept.stop()
            assert dept.started is False

        except ImportError:
            pytest.skip("DreamDepartment not available")

    @pytest.mark.asyncio
    async def test_verify_action(self):
        """Test verify action."""
        try:
            from HoloLoom.departments.dream import create_dream_department

            dept = create_dream_department()
            await dept.start()

            try:
                narrative = create_mock_narrative()
                result = await dept.process({
                    "action": "verify",
                    "narrative": narrative
                })

                assert result["success"] is True
                assert "verification" in result
                assert result["verification"]["passed"] is True

            finally:
                await dept.stop()

        except ImportError:
            pytest.skip("DreamDepartment not available")

    @pytest.mark.asyncio
    async def test_analyze_action(self):
        """Test analyze action."""
        try:
            from HoloLoom.departments.dream import create_dream_department

            dept = create_dream_department()
            await dept.start()

            try:
                narrative = create_mock_narrative()
                result = await dept.process({
                    "action": "analyze",
                    "narrative": narrative
                })

                assert result["success"] is True
                assert "result" in result
                assert result["result"]["title"] == "Journey Through the Deep Ocean"
                assert result["result"]["archetype"] == "journey"
                assert "emotional_profile" in result["result"]

            finally:
                await dept.stop()

        except ImportError:
            pytest.skip("DreamDepartment not available")

    @pytest.mark.asyncio
    async def test_invalid_action(self):
        """Test invalid action handling."""
        try:
            from HoloLoom.departments.dream import create_dream_department

            dept = create_dream_department()
            await dept.start()

            try:
                result = await dept.process({
                    "action": "invalid_action"
                })

                assert result["success"] is False
                assert "error" in result

            finally:
                await dept.stop()

        except ImportError:
            pytest.skip("DreamDepartment not available")


# =============================================================================
# Test HoloLoomBridge
# =============================================================================

class TestHoloLoomBridge:
    """Test suite for HoloLoomBridge."""

    def test_import_bridge(self):
        """Test that HoloLoomBridge can be imported."""
        from NeuroHood.dreams.hololoom_bridge import (
            HoloLoomBridge,
            StoreResult,
            QueryResult,
            AnalysisResult,
            is_phase2_available
        )
        assert HoloLoomBridge is not None
        assert is_phase2_available is not None

    def test_bridge_instantiation(self):
        """Test HoloLoomBridge instantiation."""
        from NeuroHood.dreams.hololoom_bridge import HoloLoomBridge

        bridge = HoloLoomBridge(
            enable_verification=True,
            strict_verification=False,
            enable_spinner=True
        )

        assert bridge is not None
        assert bridge.enable_verification is True
        assert bridge.strict_verification is False

    @pytest.mark.asyncio
    async def test_bridge_lifecycle(self):
        """Test bridge start/stop lifecycle."""
        from NeuroHood.dreams.hololoom_bridge import HoloLoomBridge

        async with HoloLoomBridge() as bridge:
            assert bridge._started is True

            stats = bridge.get_stats()
            assert "start_time" in stats
            assert stats["start_time"] is not None

        # After context manager, should be stopped
        # (bridge variable still exists but _started is False)

    @pytest.mark.asyncio
    async def test_verify_dream(self):
        """Test dream verification via bridge."""
        from NeuroHood.dreams.hololoom_bridge import HoloLoomBridge

        async with HoloLoomBridge() as bridge:
            narrative = create_mock_narrative()
            result = await bridge.verify_dream(narrative)

            # Should return verification result
            assert "passed" in result or "error" in result

    @pytest.mark.asyncio
    async def test_analyze_dream(self):
        """Test dream analysis via bridge."""
        from NeuroHood.dreams.hololoom_bridge import HoloLoomBridge

        async with HoloLoomBridge() as bridge:
            narrative = create_mock_narrative()
            result = await bridge.analyze_dream(narrative)

            # Should return analysis
            assert result is not None
            if result.error is None:
                assert result.title == "Journey Through the Deep Ocean"
                assert result.archetype == "journey"

    @pytest.mark.asyncio
    async def test_spin_dream(self):
        """Test dream spinning via bridge."""
        from NeuroHood.dreams.hololoom_bridge import HoloLoomBridge

        async with HoloLoomBridge(enable_spinner=True) as bridge:
            if not bridge.spinner_available:
                pytest.skip("DreamSpinner not available")

            narrative = create_mock_narrative()
            result = await bridge.spin_dream(narrative, resident_id="test")

            assert result is not None
            assert result.shard_count >= 1
            assert len(result.shards) >= 1

    @pytest.mark.asyncio
    async def test_graceful_degradation(self):
        """Test graceful degradation when HoloLoom unavailable."""
        from NeuroHood.dreams.hololoom_bridge import HoloLoomBridge

        async with HoloLoomBridge() as bridge:
            narrative = create_mock_narrative()

            # These should not crash even if HoloLoom unavailable
            store_result = await bridge.store_dream(narrative, "test_resident")
            query_result = await bridge.query_dreams("test query")

            # Should return results (even if indicating unavailable)
            assert store_result is not None
            assert query_result is not None

    @pytest.mark.asyncio
    async def test_stats_tracking(self):
        """Test statistics tracking."""
        from NeuroHood.dreams.hololoom_bridge import HoloLoomBridge

        async with HoloLoomBridge() as bridge:
            # Initial stats
            stats = bridge.get_stats()
            assert stats["dreams_analyzed"] == 0

            # Analyze a dream
            narrative = create_mock_narrative()
            await bridge.analyze_dream(narrative)

            # Stats should update
            stats = bridge.get_stats()
            assert stats["dreams_analyzed"] == 1

    @pytest.mark.asyncio
    async def test_metrics(self):
        """Test metrics retrieval."""
        from NeuroHood.dreams.hololoom_bridge import HoloLoomBridge

        async with HoloLoomBridge() as bridge:
            metrics = bridge.get_metrics()

            assert "bridge" in metrics
            assert "dreams_stored" in metrics["bridge"]


# =============================================================================
# Test Quick Functions
# =============================================================================

class TestQuickFunctions:
    """Test convenience quick_* functions."""

    @pytest.mark.asyncio
    async def test_quick_analyze(self):
        """Test quick_analyze function."""
        from NeuroHood.dreams.hololoom_bridge import quick_analyze

        narrative = create_mock_narrative()
        result = await quick_analyze(narrative)

        assert result is not None
        if result.error is None:
            assert result.title is not None

    @pytest.mark.asyncio
    async def test_quick_query(self):
        """Test quick_query function."""
        from NeuroHood.dreams.hololoom_bridge import quick_query

        result = await quick_query("ocean dreams", limit=5)

        assert result is not None
        assert result.query == "ocean dreams"


# =============================================================================
# Integration Tests
# =============================================================================

class TestPhase2Integration:
    """Full Phase 2 integration tests."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        """Test complete pipeline: verify → spin → store → query."""
        from NeuroHood.dreams.hololoom_bridge import HoloLoomBridge

        async with HoloLoomBridge(
            enable_verification=True,
            enable_spinner=True
        ) as bridge:
            narrative = create_mock_narrative()

            # Step 1: Verify
            verification = await bridge.verify_dream(narrative)
            if "error" not in verification:
                assert verification.get("passed") is True

            # Step 2: Spin (if available)
            if bridge.spinner_available:
                spin_result = await bridge.spin_dream(narrative)
                assert spin_result.shard_count >= 1

            # Step 3: Analyze
            analysis = await bridge.analyze_dream(narrative)
            if analysis.error is None:
                assert analysis.title == "Journey Through the Deep Ocean"
                assert analysis.archetype == "journey"
                assert analysis.mood == "wonder"

            # Step 4: Store (may not persist without HoloLoom)
            store_result = await bridge.store_dream(narrative, "integration_test")
            assert store_result is not None

            # Step 5: Query (may be empty without HoloLoom)
            query_result = await bridge.query_dreams("ocean whale journey")
            assert query_result is not None

    @pytest.mark.asyncio
    async def test_batch_operations(self):
        """Test batch dream operations."""
        from NeuroHood.dreams.hololoom_bridge import HoloLoomBridge

        # Create multiple narratives
        narratives = [create_mock_narrative() for _ in range(3)]

        async with HoloLoomBridge() as bridge:
            # Analyze all
            analyses = []
            for n in narratives:
                result = await bridge.analyze_dream(n)
                analyses.append(result)

            assert len(analyses) == 3

            # Check stats
            stats = bridge.get_stats()
            assert stats["dreams_analyzed"] == 3


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
