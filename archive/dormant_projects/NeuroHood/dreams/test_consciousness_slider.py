"""
Test Suite: Consciousness Slider (test_consciousness_slider.py)
================================================================

Tests for consciousness slider mechanics:
- Slider value → settings conversion
- Consciousness level determination (individual/shared/universal)
- Parameter interpolation and boundary conditions
- Dream parameter adjustment
- Ego reference handling
- Narrative voice transitions
"""

import pytest
import numpy as np
from consciousness_slider import (
    ConsciousnessSlider,
    ConsciousnessSettings,
    ConsciousnessLevel,
    PrivacyMode,
    DreamAdjustment,
    CONSCIOUSNESS_LEVEL_BOUNDARIES,
    PARAMETER_RANGES,
)


class TestConsciousnessSliderBasics:
    """Test basic slider functionality."""

    def setup_method(self):
        """Setup slider for each test."""
        self.slider = ConsciousnessSlider()

    def test_slider_init(self):
        """Test slider initialization."""
        assert self.slider is not None
        assert hasattr(self.slider, "get_settings")
        assert hasattr(self.slider, "adjust_dream_parameters")

    def test_slider_value_range_validation(self):
        """Test that slider rejects invalid values."""
        with pytest.raises(ValueError):
            self.slider.get_settings(-0.1)

        with pytest.raises(ValueError):
            self.slider.get_settings(1.1)

    def test_valid_slider_values(self):
        """Test that valid slider values work."""
        valid_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        for value in valid_values:
            settings = self.slider.get_settings(value)
            assert settings.slider_value == value


class TestConsciousnessLevels:
    """Test consciousness level determination."""

    def setup_method(self):
        """Setup slider for each test."""
        self.slider = ConsciousnessSlider()

    def test_individual_level(self):
        """Test individual consciousness level (0-0.33)."""
        for value in [0.0, 0.1, 0.2, 0.32]:
            settings = self.slider.get_settings(value)
            assert settings.level == ConsciousnessLevel.INDIVIDUAL
            assert settings.level.value == "individual"

    def test_shared_level(self):
        """Test shared consciousness level (0.33-0.66)."""
        for value in [0.33, 0.5, 0.65]:
            settings = self.slider.get_settings(value)
            assert settings.level == ConsciousnessLevel.SHARED
            assert settings.level.value == "shared"

    def test_universal_level(self):
        """Test universal consciousness level (0.66-1.0)."""
        for value in [0.66, 0.8, 1.0]:
            settings = self.slider.get_settings(value)
            assert settings.level == ConsciousnessLevel.UNIVERSAL
            assert settings.level.value == "universal"

    def test_boundary_conditions(self):
        """Test exact boundary values."""
        # At boundaries, should go to next level
        settings_low = self.slider.get_settings(0.33)
        assert settings_low.level == ConsciousnessLevel.SHARED

        settings_mid = self.slider.get_settings(0.66)
        assert settings_mid.level == ConsciousnessLevel.UNIVERSAL


class TestParameterInterpolation:
    """Test parameter interpolation across slider range."""

    def setup_method(self):
        """Setup slider for each test."""
        self.slider = ConsciousnessSlider()

    def test_max_participants_progression(self):
        """Test that max_participants increases with slider."""
        settings_individual = self.slider.get_settings(0.15)
        settings_shared = self.slider.get_settings(0.5)
        settings_universal = self.slider.get_settings(0.85)

        # Individual: 1 participant
        assert settings_individual.max_participants == 1

        # Shared: 2-5 participants
        assert 2 <= settings_shared.max_participants <= 5

        # Universal: 6+ participants
        assert settings_universal.max_participants >= 6

    def test_privacy_gradient_progression(self):
        """Test that privacy_gradient increases with slider."""
        settings_individual = self.slider.get_settings(0.1)
        settings_shared = self.slider.get_settings(0.5)
        settings_universal = self.slider.get_settings(0.9)

        # Privacy should increase: individual < shared < universal
        assert settings_individual.privacy_gradient < settings_shared.privacy_gradient
        assert settings_shared.privacy_gradient < settings_universal.privacy_gradient

    def test_archetypal_strength_progression(self):
        """Test that archetypal_strength increases with slider."""
        settings_individual = self.slider.get_settings(0.1)
        settings_shared = self.slider.get_settings(0.5)
        settings_universal = self.slider.get_settings(0.9)

        # Archetypal strength should increase
        assert (
            settings_individual.archetypal_strength
            < settings_shared.archetypal_strength
        )
        assert (
            settings_shared.archetypal_strength
            < settings_universal.archetypal_strength
        )

    def test_ego_dissolution_progression(self):
        """Test that ego_dissolution increases with slider."""
        settings_individual = self.slider.get_settings(0.1)
        settings_shared = self.slider.get_settings(0.5)
        settings_universal = self.slider.get_settings(0.9)

        # Ego dissolution should increase
        assert settings_individual.ego_dissolution < settings_shared.ego_dissolution
        assert settings_shared.ego_dissolution < settings_universal.ego_dissolution

    def test_symbol_universality_progression(self):
        """Test that symbol_universality increases with slider."""
        settings_individual = self.slider.get_settings(0.1)
        settings_shared = self.slider.get_settings(0.5)
        settings_universal = self.slider.get_settings(0.9)

        # Symbol universality should increase
        assert (
            settings_individual.symbol_universality
            < settings_shared.symbol_universality
        )
        assert (
            settings_shared.symbol_universality
            < settings_universal.symbol_universality
        )

    def test_parameter_ranges(self):
        """Test that all parameters stay within defined ranges."""
        for value in np.linspace(0.0, 1.0, 11):
            settings = self.slider.get_settings(value)

            assert 0.0 <= settings.privacy_gradient <= 1.0
            assert 0.0 <= settings.archetypal_strength <= 1.0
            assert 0.0 <= settings.ego_dissolution <= 1.0
            assert 0.0 <= settings.symbol_universality <= 1.0
            assert 1 <= settings.max_participants <= 999


class TestConsciousnessSettings:
    """Test ConsciousnessSettings dataclass."""

    def setup_method(self):
        """Setup slider for each test."""
        self.slider = ConsciousnessSlider()

    def test_settings_creation(self):
        """Test that settings are created correctly."""
        settings = self.slider.get_settings(0.5)
        assert isinstance(settings, ConsciousnessSettings)
        assert 0.0 <= settings.slider_value <= 1.0

    def test_settings_post_init(self):
        """Test that post_init computes derived parameters."""
        settings = self.slider.get_settings(0.1)

        # Should have all attributes
        assert hasattr(settings, "privacy_mode")
        assert hasattr(settings, "level_position")
        assert hasattr(settings, "narrative_voice")
        assert hasattr(settings, "symbol_filter_threshold")
        assert hasattr(settings, "detail_obscuration")
        assert hasattr(settings, "emotional_universalization")

    def test_privacy_mode_determination(self):
        """Test privacy mode is determined correctly."""
        settings_low = self.slider.get_settings(0.1)
        assert settings_low.privacy_mode == PrivacyMode.NONE

        settings_mid = self.slider.get_settings(0.5)
        assert settings_mid.privacy_mode == PrivacyMode.SYMBOLIC

        settings_high = self.slider.get_settings(0.9)
        assert settings_high.privacy_mode == PrivacyMode.ARCHETYPAL

    def test_narrative_voice_determination(self):
        """Test narrative voice is determined correctly."""
        settings_individual = self.slider.get_settings(0.1)
        assert settings_individual.narrative_voice == "first_person"

        settings_shared = self.slider.get_settings(0.5)
        assert settings_shared.narrative_voice == "third_person"

        settings_universal = self.slider.get_settings(0.85)
        assert settings_universal.narrative_voice == "collective"

    def test_settings_to_dict(self):
        """Test settings serialization to dict."""
        settings = self.slider.get_settings(0.5)
        settings_dict = settings.to_dict()

        assert isinstance(settings_dict, dict)
        assert "slider_value" in settings_dict
        assert "level" in settings_dict
        assert settings_dict["level"] == "shared"
        assert settings_dict["narrative_voice"] == "third_person"


class TestDreamAdjustment:
    """Test dream parameter adjustment."""

    def setup_method(self):
        """Setup slider for each test."""
        self.slider = ConsciousnessSlider()
        self.base_dream = {
            "content": "I am trapped in this situation. I feel helpless.",
            "participants": ["alice", "bob", "charlie", "diana", "eve"],
            "emotional_essence": {"primary_emotion": "trapped", "intensity": 0.8},
            "symbols": [
                {"name": "cage", "universality": 0.9},
                {"name": "lost_child", "universality": 0.8},
            ],
        }

    def test_adjust_dream_no_adjustment_needed(self):
        """Test adjustment when no change needed."""
        settings = self.slider.get_settings(0.15)
        adjusted = self.slider.adjust_dream_parameters(self.base_dream, settings)

        # Should have consciousness settings
        assert "consciousness_settings" in adjusted
        assert adjusted["consciousness_settings"]["level"] == "individual"

    def test_participant_filtering(self):
        """Test that participants are filtered based on consciousness level."""
        settings = self.slider.get_settings(0.15)  # Individual: max 1
        adjusted = self.slider.adjust_dream_parameters(self.base_dream, settings)

        # For individual, should only have 1 participant
        if "consciousness_filtering" in adjusted:
            assert adjusted["consciousness_filtering"]["selected_count"] == 1
            assert len(adjusted["participants"]) == 1

    def test_privacy_instructions_added(self):
        """Test that privacy instructions are added."""
        settings = self.slider.get_settings(0.5)
        adjusted = self.slider.adjust_dream_parameters(self.base_dream, settings)

        assert "privacy_instructions" in adjusted
        assert "privacy_gradient" in adjusted["privacy_instructions"]

    def test_symbol_requirements_added(self):
        """Test that symbol requirements are added."""
        settings = self.slider.get_settings(0.5)
        adjusted = self.slider.adjust_dream_parameters(self.base_dream, settings)

        assert "symbol_requirements" in adjusted
        assert "minimum_universality" in adjusted["symbol_requirements"]

    def test_narrative_adjustments_added(self):
        """Test that narrative adjustments are added."""
        settings = self.slider.get_settings(0.5)
        adjusted = self.slider.adjust_dream_parameters(self.base_dream, settings)

        assert "narrative_adjustments" in adjusted
        assert "narrative_voice" in adjusted["narrative_adjustments"]
        assert "perspective" in adjusted["narrative_adjustments"]

    def test_ego_reference_adjustment_individual(self):
        """Test ego reference handling for individual consciousness."""
        settings = self.slider.get_settings(0.15)
        adjusted = self.slider.adjust_dream_parameters(self.base_dream, settings)

        # Should preserve "I" pronouns for individual
        assert "I am" in adjusted["content"]
        assert "I feel" in adjusted["content"]

    def test_ego_reference_adjustment_shared(self):
        """Test ego reference handling for shared consciousness."""
        settings = self.slider.get_settings(0.5)
        adjusted = self.slider.adjust_dream_parameters(self.base_dream, settings)

        # Should replace with third person
        assert "Someone is" in adjusted["content"]
        assert "A presence feels" in adjusted["content"]
        assert "I am" not in adjusted["content"]

    def test_ego_reference_adjustment_universal(self):
        """Test ego reference handling for universal consciousness."""
        settings = self.slider.get_settings(0.85)
        adjusted = self.slider.adjust_dream_parameters(self.base_dream, settings)

        # Should replace with collective
        assert "We are" in adjusted["content"]
        assert "We feel" in adjusted["content"]
        assert "I am" not in adjusted["content"]

    def test_emotional_translation_added(self):
        """Test that emotional translation instructions are added."""
        settings = self.slider.get_settings(0.5)
        adjusted = self.slider.adjust_dream_parameters(self.base_dream, settings)

        if "emotional_essence" in self.base_dream:
            assert "emotional_translation" in adjusted
            assert "universalization_level" in adjusted["emotional_translation"]


class TestDreamAdjustmentObject:
    """Test DreamAdjustment object creation."""

    def setup_method(self):
        """Setup slider for each test."""
        self.slider = ConsciousnessSlider()
        self.base_dream = {
            "content": "Dream content",
            "participants": ["alice", "bob"],
        }

    def test_create_dream_adjustment(self):
        """Test creating DreamAdjustment object."""
        settings = self.slider.get_settings(0.5)
        adjustment = self.slider.create_dream_adjustment(settings, self.base_dream)

        assert isinstance(adjustment, DreamAdjustment)
        assert hasattr(adjustment, "participant_filter")
        assert hasattr(adjustment, "privacy_instructions")
        assert hasattr(adjustment, "symbol_requirements")

    def test_dream_adjustment_for_universal(self):
        """Test DreamAdjustment for universal consciousness."""
        settings = self.slider.get_settings(0.85)
        adjustment = self.slider.create_dream_adjustment(settings, self.base_dream)

        # For universal, participants should be "all"
        assert adjustment.participant_filter == "all"


class TestSliderDescriptions:
    """Test human-readable descriptions."""

    def setup_method(self):
        """Setup slider for each test."""
        self.slider = ConsciousnessSlider()

    def test_slider_description_individual(self):
        """Test description for individual consciousness."""
        desc = self.slider.get_slider_description(0.15)
        assert "Individual" in desc
        assert "private" in desc.lower()

    def test_slider_description_shared(self):
        """Test description for shared consciousness."""
        desc = self.slider.get_slider_description(0.5)
        assert "Shared" in desc
        assert "relational" in desc.lower()

    def test_slider_description_universal(self):
        """Test description for universal consciousness."""
        desc = self.slider.get_slider_description(0.85)
        assert "Universal" in desc
        assert "collective" in desc.lower()


class TestLevelPosition:
    """Test level_position calculations."""

    def setup_method(self):
        """Setup slider for each test."""
        self.slider = ConsciousnessSlider()

    def test_level_position_at_start(self):
        """Test level_position at start of level."""
        settings = self.slider.get_settings(0.0)
        assert settings.level_position == pytest.approx(0.0, abs=0.01)

    def test_level_position_at_mid(self):
        """Test level_position at middle of level."""
        settings = self.slider.get_settings(0.165)  # Mid of individual level
        assert settings.level_position == pytest.approx(0.5, abs=0.01)

    def test_level_position_at_end(self):
        """Test level_position at end of level."""
        # Use 0.329 (just before boundary) to stay in individual level
        settings = self.slider.get_settings(0.329)
        assert settings.level_position == pytest.approx(1.0, abs=0.02)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def setup_method(self):
        """Setup slider for each test."""
        self.slider = ConsciousnessSlider()

    def test_zero_slider_value(self):
        """Test with slider at 0."""
        settings = self.slider.get_settings(0.0)
        assert settings.level == ConsciousnessLevel.INDIVIDUAL
        assert settings.max_participants == 1
        assert settings.privacy_gradient < 0.2

    def test_full_slider_value(self):
        """Test with slider at 1.0."""
        settings = self.slider.get_settings(1.0)
        assert settings.level == ConsciousnessLevel.UNIVERSAL
        assert settings.max_participants > 5
        assert settings.privacy_gradient > 0.9
        assert settings.ego_dissolution > 0.9

    def test_empty_dream(self):
        """Test adjustment with minimal dream."""
        settings = self.slider.get_settings(0.5)
        minimal_dream = {"content": "A dream"}
        adjusted = self.slider.adjust_dream_parameters(minimal_dream, settings)

        # Should still add consciousness settings
        assert "consciousness_settings" in adjusted

    def test_dream_without_participants(self):
        """Test adjustment for dream without participants list."""
        settings = self.slider.get_settings(0.5)
        dream = {"content": "No participants"}
        adjusted = self.slider.adjust_dream_parameters(dream, settings)

        # Should not crash, should still work
        assert "consciousness_settings" in adjusted


class TestIntegration:
    """Integration tests combining multiple components."""

    def setup_method(self):
        """Setup slider for each test."""
        self.slider = ConsciousnessSlider()

    def test_full_workflow_individual(self):
        """Test complete workflow for individual consciousness."""
        slider_value = 0.2
        dream = {
            "content": "I am struggling with my identity.",
            "participants": ["myself"],
            "emotional_essence": {"primary_emotion": "confusion", "intensity": 0.7},
        }

        settings = self.slider.get_settings(slider_value)
        adjusted = self.slider.adjust_dream_parameters(dream, settings)

        assert adjusted["consciousness_settings"]["level"] == "individual"
        assert adjusted["consciousness_settings"]["max_participants"] == 1
        assert "I am" in adjusted["content"]

    def test_full_workflow_shared(self):
        """Test complete workflow for shared consciousness."""
        slider_value = 0.5
        dream = {
            "content": "I feel abandoned by my friend.",
            "participants": ["person_a", "person_b"],
            "emotional_essence": {"primary_emotion": "abandonment", "intensity": 0.8},
        }

        settings = self.slider.get_settings(slider_value)
        adjusted = self.slider.adjust_dream_parameters(dream, settings)

        assert adjusted["consciousness_settings"]["level"] == "shared"
        assert 2 <= adjusted["consciousness_settings"]["max_participants"] <= 5
        assert "I am" not in adjusted["content"]

    def test_full_workflow_universal(self):
        """Test complete workflow for universal consciousness."""
        slider_value = 0.9
        dream = {
            "content": "I am suffering from existential dread.",
            "participants": ["all"],
            "emotional_essence": {"primary_emotion": "existential_dread", "intensity": 0.9},
        }

        settings = self.slider.get_settings(slider_value)
        adjusted = self.slider.adjust_dream_parameters(dream, settings)

        assert adjusted["consciousness_settings"]["level"] == "universal"
        assert adjusted["consciousness_settings"]["max_participants"] >= 6
        assert "We are" in adjusted["content"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
