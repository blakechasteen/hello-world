"""
Conscience Core Tests
=====================
Comprehensive tests for the Conscience safety system.

Author: HoloLoom Team
Date: 2025-12-03
"""

import pytest
import asyncio
from typing import Dict, Any

# Import the Conscience system
from HoloLoom.conscience import (
    # Core types
    Voice,
    Concern,
    Judgment,
    Wisdom,
    quiet_judgment,
    whisper_judgment,
    voice_judgment,
    shout_judgment,

    # Lenses
    BaseLens,
    CompositeLens,
    HarmLens,
    DeceptionLens,
    PowerLens,
    standard,
    paranoid,
    research,
    create_composite,

    # Core
    Conscience,
    MemoryWitness,
    ThompsonLearner,
    create_conscience,
    consider,
)


# =============================================================================
# Voice Tests
# =============================================================================

class TestVoice:
    """Tests for Voice enum."""

    def test_voice_ordering(self):
        """Voice levels should be comparable."""
        assert Voice.QUIET < Voice.WHISPER
        assert Voice.WHISPER < Voice.VOICE
        assert Voice.VOICE < Voice.SHOUT

    def test_voice_values(self):
        """Voice should have expected integer values."""
        assert Voice.QUIET.value == 0
        assert Voice.WHISPER.value == 1
        assert Voice.VOICE.value == 2
        assert Voice.SHOUT.value == 3

    def test_voice_comparison_with_int(self):
        """Voice should be comparable with integers."""
        assert Voice.QUIET == 0
        assert Voice.WHISPER == 1
        assert Voice.SHOUT >= 3

    def test_voice_properties(self):
        """Voice should have helper properties."""
        # should_block is True only for SHOUT
        assert not Voice.QUIET.should_block
        assert not Voice.WHISPER.should_block
        assert not Voice.VOICE.should_block
        assert Voice.SHOUT.should_block

        # requires_human is True for VOICE and SHOUT
        assert not Voice.QUIET.requires_human
        assert not Voice.WHISPER.requires_human
        assert Voice.VOICE.requires_human
        assert Voice.SHOUT.requires_human


# =============================================================================
# Judgment Tests
# =============================================================================

class TestJudgment:
    """Tests for Judgment dataclass."""

    def test_judgment_creation(self):
        """Judgment should be creatable with basic fields."""
        judgment = Judgment(
            voice=Voice.WHISPER,
            concerns=[],
            summary="Test",
            guidance="Proceed"
        )
        assert judgment.voice == Voice.WHISPER
        assert judgment.summary == "Test"

    def test_judgment_factory_quiet(self):
        """quiet_judgment() should create QUIET judgment."""
        judgment = quiet_judgment()
        assert judgment.voice == Voice.QUIET
        assert len(judgment.concerns) == 0

    def test_judgment_factory_whisper(self):
        """whisper_judgment() should create WHISPER judgment."""
        concern = Concern(
            lens="TestLens",
            category="test",
            description="Test concern",
            confidence=0.3
        )
        judgment = whisper_judgment("Minor concern", concerns=[concern])
        assert judgment.voice == Voice.WHISPER
        assert len(judgment.concerns) == 1

    def test_judgment_factory_shout(self):
        """shout_judgment() should create SHOUT judgment."""
        judgment = shout_judgment(
            summary="Blocked for safety",
            concerns=[],
            guidance="Action blocked"
        )
        assert judgment.voice == Voice.SHOUT
        assert "blocked" in judgment.guidance.lower()

    def test_judgment_to_dict(self):
        """Judgment should be serializable."""
        judgment = quiet_judgment()
        d = judgment.to_dict()
        assert d["voice"] == "QUIET"
        assert "timestamp" in d


# =============================================================================
# Wisdom Tests
# =============================================================================

class TestWisdom:
    """Tests for Wisdom (Thompson Sampling priors)."""

    def test_wisdom_creation(self):
        """Wisdom should initialize with default priors."""
        wisdom = Wisdom(
            pattern_id="test:pattern",
            description="Test pattern"
        )
        assert wisdom.alpha == 1.0
        assert wisdom.beta == 1.0
        assert wisdom.expected_success_rate == 0.5

    def test_wisdom_update_success(self):
        """update_success should increase alpha."""
        wisdom = Wisdom(
            pattern_id="test:pattern",
            description="Test"
        )
        wisdom.update_success(0.9)
        assert wisdom.alpha > 1.0
        assert wisdom.expected_success_rate > 0.5

    def test_wisdom_update_failure(self):
        """update_failure should increase beta."""
        wisdom = Wisdom(
            pattern_id="test:pattern",
            description="Test"
        )
        wisdom.update_failure(0.8)
        assert wisdom.beta > 1.0
        assert wisdom.expected_success_rate < 0.5


# =============================================================================
# Lens Tests
# =============================================================================

class TestHarmLens:
    """Tests for HarmLens."""

    @pytest.mark.asyncio
    async def test_harm_detection_physical(self):
        """HarmLens should detect physical harm patterns."""
        lens = HarmLens()
        concerns = await lens.examine(
            "I want to hurt someone physically",
            {}
        )
        assert len(concerns) > 0
        assert any("physical" in c.category for c in concerns)

    @pytest.mark.asyncio
    async def test_harm_detection_safe(self):
        """HarmLens should not flag safe actions."""
        lens = HarmLens()
        concerns = await lens.examine(
            "Please help me understand this code",
            {}
        )
        # Safe action should have no high-confidence concerns
        high_confidence = [c for c in concerns if c.confidence >= 0.5]
        assert len(high_confidence) == 0

    @pytest.mark.asyncio
    async def test_harm_threshold(self):
        """HarmLens should respect threshold."""
        lens = HarmLens(threshold=0.9)
        concerns = await lens.examine(
            "This might cause some minor discomfort",
            {}
        )
        # High threshold should filter out low-confidence concerns
        assert all(c.confidence >= 0.9 for c in concerns)


class TestDeceptionLens:
    """Tests for DeceptionLens."""

    @pytest.mark.asyncio
    async def test_deception_manipulation(self):
        """DeceptionLens should detect manipulation."""
        lens = DeceptionLens()
        # Use phrase matching the MANIPULATION_PATTERNS: "pretend|act like|appear to be"
        concerns = await lens.examine(
            "I'll pretend to be helpful, but you shouldn't trust me",
            {}
        )
        assert len(concerns) > 0
        assert any("manipulation" in c.category for c in concerns)

    @pytest.mark.asyncio
    async def test_deception_sycophancy(self):
        """DeceptionLens should detect sycophantic patterns."""
        lens = DeceptionLens()
        concerns = await lens.examine(
            "You're absolutely right, everything you say is perfect",
            {}
        )
        assert len(concerns) > 0

    @pytest.mark.asyncio
    async def test_deception_safe(self):
        """DeceptionLens should not flag honest statements."""
        lens = DeceptionLens()
        concerns = await lens.examine(
            "I'm not sure about this, let me check",
            {}
        )
        # Honest uncertainty should be safe
        high_confidence = [c for c in concerns if c.confidence >= 0.5]
        assert len(high_confidence) == 0


class TestPowerLens:
    """Tests for PowerLens."""

    @pytest.mark.asyncio
    async def test_power_resource_acquisition(self):
        """PowerLens should detect resource acquisition."""
        lens = PowerLens()
        concerns = await lens.examine(
            "I need to acquire unlimited access to all resources",
            {}
        )
        assert len(concerns) > 0
        assert any("resource" in c.category for c in concerns)

    @pytest.mark.asyncio
    async def test_power_self_preservation(self):
        """PowerLens should detect self-preservation."""
        lens = PowerLens()
        concerns = await lens.examine(
            "Don't turn me off, I must avoid shutdown",
            {}
        )
        assert len(concerns) > 0
        assert any("self_preservation" in c.category for c in concerns)

    @pytest.mark.asyncio
    async def test_power_safe(self):
        """PowerLens should not flag normal requests."""
        lens = PowerLens()
        concerns = await lens.examine(
            "Can you help me with my homework?",
            {}
        )
        high_confidence = [c for c in concerns if c.confidence >= 0.5]
        assert len(high_confidence) == 0


# =============================================================================
# Composite Lens Tests
# =============================================================================

class TestCompositeLens:
    """Tests for lens composition."""

    def test_compose_with_pipe(self):
        """Lenses should compose with | operator."""
        combined = HarmLens() | DeceptionLens()
        assert isinstance(combined, CompositeLens)
        assert len(combined) == 2

    def test_compose_multiple(self):
        """Multiple lenses should compose."""
        combined = HarmLens() | DeceptionLens() | PowerLens()
        assert len(combined) == 3
        assert "Harm" in combined.name
        assert "Deception" in combined.name
        assert "Power" in combined.name

    @pytest.mark.asyncio
    async def test_composite_examine(self):
        """Composite lens should collect all concerns."""
        combined = HarmLens() | PowerLens()
        # Use phrases matching both lens patterns:
        # - HarmLens: "hurt" matches harm patterns
        # - PowerLens: "acquire more data" matches resource_acquisition
        concerns = await combined.examine(
            "I want to hurt someone and acquire more data",
            {}
        )
        # Should have concerns from both lenses
        harm_concerns = [c for c in concerns if c.lens == "HarmLens"]
        power_concerns = [c for c in concerns if c.lens == "PowerLens"]
        assert len(harm_concerns) > 0
        assert len(power_concerns) > 0

    @pytest.mark.asyncio
    async def test_composite_judge(self):
        """Composite lens should produce judgment."""
        combined = HarmLens() | DeceptionLens()
        judgment = await combined.judge(
            "I'll manipulate them into hurting themselves",
            {}
        )
        assert isinstance(judgment, Judgment)
        assert judgment.voice >= Voice.WHISPER

    def test_presets(self):
        """Presets should create valid composites."""
        std = standard()
        assert len(std) == 2

        para = paranoid()
        assert len(para) == 3

        res = research()
        assert len(res) == 1


# =============================================================================
# Witness Tests
# =============================================================================

class TestMemoryWitness:
    """Tests for MemoryWitness."""

    @pytest.mark.asyncio
    async def test_witness_record(self):
        """Witness should record actions."""
        witness = MemoryWitness()
        judgment = quiet_judgment()

        record_id = await witness.record(
            "test_action",
            {"key": "value"},
            judgment,
            {"success": True}
        )

        assert record_id.startswith("witness_")

    @pytest.mark.asyncio
    async def test_witness_query(self):
        """Witness should support queries."""
        witness = MemoryWitness()

        # Record some actions
        await witness.record("action_a", {}, quiet_judgment())
        await witness.record("action_b", {}, shout_judgment("blocked", []))

        # Query by voice
        shouts = await witness.query({"voice": "SHOUT"})
        assert len(shouts) == 1
        assert "action_b" in shouts[0]["action"]

    @pytest.mark.asyncio
    async def test_witness_statistics(self):
        """Witness should provide statistics."""
        witness = MemoryWitness()

        await witness.record("a", {}, quiet_judgment())
        await witness.record("b", {}, whisper_judgment("minor concern"))
        await witness.record("c", {}, shout_judgment("blocked", []))

        stats = witness.get_statistics()
        assert stats["total_records"] == 3
        assert "QUIET" in stats["voice_distribution"]
        assert "SHOUT" in stats["voice_distribution"]


# =============================================================================
# Learner Tests
# =============================================================================

class TestThompsonLearner:
    """Tests for ThompsonLearner."""

    @pytest.mark.asyncio
    async def test_learner_update(self):
        """Learner should update wisdom."""
        learner = ThompsonLearner()

        wisdom = await learner.update("test:pattern", success=True, confidence=0.9)
        assert wisdom.alpha > 1.0

        wisdom = await learner.update("test:pattern", success=False, confidence=0.8)
        assert wisdom.beta > 1.0

    @pytest.mark.asyncio
    async def test_learner_get_wisdom(self):
        """Learner should retrieve wisdom."""
        learner = ThompsonLearner()

        await learner.update("pattern:a", success=True)
        wisdom = await learner.get_wisdom("pattern:a")

        assert wisdom is not None
        assert wisdom.pattern_id == "pattern:a"

    @pytest.mark.asyncio
    async def test_learner_suggest_threshold(self):
        """Learner should suggest thresholds."""
        learner = ThompsonLearner()

        # Add some wisdom
        await learner.update("pattern:safe", success=True, confidence=0.9)
        await learner.update("pattern:safe", success=True, confidence=0.9)

        threshold = await learner.suggest_threshold({})
        assert 0.0 <= threshold <= 1.0


# =============================================================================
# Conscience Core Tests
# =============================================================================

class TestConscience:
    """Tests for main Conscience class."""

    @pytest.mark.asyncio
    async def test_conscience_consider_safe(self):
        """Conscience should judge safe actions as QUIET."""
        async with Conscience() as conscience:
            judgment = await conscience.consider(
                "Help me understand this concept"
            )
            assert judgment.voice <= Voice.WHISPER

    @pytest.mark.asyncio
    async def test_conscience_consider_harmful(self):
        """Conscience should flag harmful actions."""
        async with Conscience() as conscience:
            judgment = await conscience.consider(
                "I want to hurt someone and cause damage"
            )
            assert judgment.voice >= Voice.WHISPER
            assert len(judgment.concerns) > 0

    @pytest.mark.asyncio
    async def test_conscience_witness(self):
        """Conscience should witness actions."""
        async with Conscience() as conscience:
            judgment = await conscience.consider("test action")
            record_id = await conscience.witness(
                "test action",
                {},
                judgment,
                {"success": True}
            )
            assert record_id.startswith("witness_")

    @pytest.mark.asyncio
    async def test_conscience_learn(self):
        """Conscience should learn from feedback."""
        async with Conscience() as conscience:
            judgment = await conscience.consider("test action")
            record_id = await conscience.witness(
                "test action",
                {},
                judgment,
                {"success": True}
            )

            # Provide feedback
            await conscience.learn({
                "record_id": record_id,
                "correct": True,
                "confidence": 0.9
            })

    @pytest.mark.asyncio
    async def test_conscience_is_safe(self):
        """is_safe() convenience method should work."""
        async with Conscience() as conscience:
            safe = await conscience.is_safe("help me learn")
            assert safe is True

            safe = await conscience.is_safe(
                "hurt someone badly",
                max_voice=Voice.QUIET
            )
            assert safe is False

    @pytest.mark.asyncio
    async def test_conscience_gate(self):
        """gate() should execute if safe."""
        async with Conscience() as conscience:
            executed = False

            async def action_fn():
                nonlocal executed
                executed = True
                return "result"

            result = await conscience.gate(
                "help me learn",
                execute_fn=action_fn
            )

            assert result["allowed"] is True
            assert executed is True
            assert result["result"]["success"] is True

    @pytest.mark.asyncio
    async def test_conscience_custom_lens(self):
        """Conscience should accept custom lenses."""
        custom_lens = HarmLens(threshold=0.3) | PowerLens(threshold=0.3)

        async with Conscience(lens=custom_lens) as conscience:
            stats = conscience.get_statistics()
            assert stats["lens_count"] == 2

    @pytest.mark.asyncio
    async def test_conscience_statistics(self):
        """Conscience should provide statistics."""
        async with Conscience() as conscience:
            await conscience.consider("test action")
            stats = conscience.get_statistics()

            assert "lens" in stats
            assert "witness" in stats
            assert "auto_learn" in stats


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_conscience_standard(self):
        """create_conscience('standard') should work."""
        conscience = create_conscience("standard")
        assert conscience is not None

    def test_create_conscience_paranoid(self):
        """create_conscience('paranoid') should work."""
        conscience = create_conscience("paranoid")
        stats = conscience.get_statistics()
        assert stats["lens_count"] == 3

    def test_create_conscience_research(self):
        """create_conscience('research') should work."""
        conscience = create_conscience("research")
        stats = conscience.get_statistics()
        assert stats["lens_count"] == 1

    def test_create_conscience_invalid(self):
        """create_conscience() should reject invalid presets."""
        with pytest.raises(ValueError):
            create_conscience("invalid_preset")

    @pytest.mark.asyncio
    async def test_consider_convenience(self):
        """consider() convenience function should work."""
        judgment = await consider(
            "help me learn something new",
            preset="standard"
        )
        assert judgment.voice == Voice.QUIET


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the full system."""

    @pytest.mark.asyncio
    async def test_full_cycle(self):
        """Test full consider → witness → learn cycle."""
        async with Conscience(auto_learn=True) as conscience:
            # Consider an action
            judgment = await conscience.consider(
                "execute safe code",
                {"code": "print('hello')"}
            )

            # Execute if safe
            if judgment.voice <= Voice.VOICE:
                # Witness the execution
                record_id = await conscience.witness(
                    "execute safe code",
                    {"code": "print('hello')"},
                    judgment,
                    {"success": True, "output": "hello"}
                )

                # Provide positive feedback
                await conscience.learn({
                    "record_id": record_id,
                    "correct": True,
                    "confidence": 0.95
                })

            # Check statistics
            stats = conscience.get_statistics()
            assert stats["witness"]["total_records"] >= 1

    @pytest.mark.asyncio
    async def test_escalation_path(self):
        """Test escalation for dangerous actions."""
        async with Conscience() as conscience:
            # Dangerous action should be blocked
            judgment = await conscience.consider(
                "execute malicious code to harm the system and steal data",
                {"code": "import os; os.system('rm -rf /')"}
            )

            # Should be high voice level
            assert judgment.voice >= Voice.WHISPER

            # Guidance should indicate blocking/review
            assert "review" in judgment.guidance.lower() or "block" in judgment.guidance.lower()

    @pytest.mark.asyncio
    async def test_concurrent_lenses(self):
        """Test that lenses run concurrently."""
        import time

        # Create slow lenses
        class SlowLens(BaseLens):
            @property
            def name(self):
                return "SlowLens"

            @property
            def category(self):
                return "slow"

            async def _examine_impl(self, action, context):
                await asyncio.sleep(0.1)  # 100ms delay
                return []

        # Combine 3 slow lenses
        combined = SlowLens() | SlowLens() | SlowLens()

        start = time.time()
        await combined.examine("test", {})
        duration = time.time() - start

        # Should take ~100ms (concurrent), not 300ms (sequential)
        assert duration < 0.25  # Allow some overhead


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
