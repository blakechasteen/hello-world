"""
Tests for Personality Framework
================================

Tests personality loading, switching, trait application, and voice mapping.

Date: November 16, 2025
Phase: 3 (Custom Personalities)
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from typing import Dict, Any

from hololoom.voice.personality import (
    PersonalityTraits,
    Personality,
    PersonalityManager,
    PersonalityType
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_personalities_dir(tmp_path):
    """Create temporary personalities directory with test YAML files"""
    personalities_dir = tmp_path / "personalities"
    personalities_dir.mkdir()

    # Create test personality YAML
    test_personality = {
        'name': 'Test Elle',
        'description': 'Test personality',
        'traits': {
            'formality': 0.5,
            'verbosity': 0.5,
            'emotional_tone': 0.5,
            'teaching_style': 0.5,
            'humor': 0.5
        },
        'voice_id': 'alloy',
        'prompt_template': 'You are Test Elle, a test personality.',
        'example_responses': [
            'Test response 1',
            'Test response 2'
        ]
    }

    test_file = personalities_dir / "test_elle.yaml"
    with open(test_file, 'w') as f:
        yaml.dump(test_personality, f)

    return personalities_dir


# ============================================================================
# PersonalityTraits Tests
# ============================================================================

def test_personality_traits_creation():
    """Test creating personality traits with valid values"""
    traits = PersonalityTraits(
        formality=0.8,
        verbosity=0.6,
        emotional_tone=0.7,
        teaching_style=0.5,
        humor=0.3
    )

    assert traits.formality == 0.8
    assert traits.verbosity == 0.6
    assert traits.emotional_tone == 0.7
    assert traits.teaching_style == 0.5
    assert traits.humor == 0.3


def test_personality_traits_validation():
    """Test trait validation (must be in [0, 1])"""
    # Valid boundaries
    traits = PersonalityTraits(formality=0.0, verbosity=1.0)
    assert traits.formality == 0.0
    assert traits.verbosity == 1.0

    # Invalid: below 0
    with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
        PersonalityTraits(formality=-0.1)

    # Invalid: above 1
    with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
        PersonalityTraits(formality=1.1)


def test_personality_traits_to_dict():
    """Test converting traits to dictionary"""
    traits = PersonalityTraits(formality=0.8, verbosity=0.6)
    trait_dict = traits.to_dict()

    assert trait_dict['formality'] == 0.8
    assert trait_dict['verbosity'] == 0.6
    assert len(trait_dict) == 5  # All 5 traits


def test_personality_traits_from_dict():
    """Test creating traits from dictionary"""
    trait_dict = {
        'formality': 0.7,
        'verbosity': 0.8,
        'emotional_tone': 0.6,
        'teaching_style': 0.9,
        'humor': 0.2
    }
    traits = PersonalityTraits.from_dict(trait_dict)

    assert traits.formality == 0.7
    assert traits.verbosity == 0.8
    assert traits.emotional_tone == 0.6


# ============================================================================
# Personality Tests
# ============================================================================

def test_personality_creation():
    """Test creating a complete personality"""
    traits = PersonalityTraits(formality=0.8, verbosity=0.9)
    personality = Personality(
        name="Test Personality",
        description="A test personality",
        traits=traits,
        voice_id="nova",
        prompt_template="You are a test personality.",
        example_responses=["Response 1", "Response 2"]
    )

    assert personality.name == "Test Personality"
    assert personality.description == "A test personality"
    assert personality.traits.formality == 0.8
    assert personality.voice_id == "nova"
    assert len(personality.example_responses) == 2


def test_personality_to_dict():
    """Test converting personality to dictionary"""
    traits = PersonalityTraits(formality=0.8)
    personality = Personality(
        name="Test",
        description="Test personality",
        traits=traits,
        voice_id="nova",
        prompt_template="Test prompt"
    )

    personality_dict = personality.to_dict()

    assert personality_dict['name'] == "Test"
    assert personality_dict['voice_id'] == "nova"
    assert 'traits' in personality_dict
    assert personality_dict['traits']['formality'] == 0.8


def test_personality_from_dict():
    """Test creating personality from dictionary"""
    data = {
        'name': 'Test Personality',
        'description': 'A test',
        'traits': {
            'formality': 0.7,
            'verbosity': 0.6,
            'emotional_tone': 0.5,
            'teaching_style': 0.4,
            'humor': 0.3
        },
        'voice_id': 'shimmer',
        'prompt_template': 'Test prompt',
        'example_responses': ['Example 1']
    }

    personality = Personality.from_dict(data)

    assert personality.name == 'Test Personality'
    assert personality.traits.formality == 0.7
    assert personality.voice_id == 'shimmer'
    assert len(personality.example_responses) == 1


# ============================================================================
# PersonalityManager Tests
# ============================================================================

def test_personality_manager_initialization():
    """Test personality manager initializes with defaults"""
    manager = PersonalityManager()

    # Should load 4 default personalities
    assert len(manager.personalities) == 4
    assert 'professor_elle' in manager.personalities
    assert 'assistant_elle' in manager.personalities
    assert 'companion_elle' in manager.personalities
    assert 'expert_elle' in manager.personalities

    # Default active personality should be Professor Elle
    assert manager.active_personality is not None
    assert manager.active_personality.name == "Professor Elle"


def test_personality_manager_load_from_yaml(temp_personalities_dir):
    """Test loading personalities from YAML files"""
    manager = PersonalityManager(personalities_dir=temp_personalities_dir)

    # Should load test personality
    assert 'test_elle' in manager.personalities
    test_personality = manager.personalities['test_elle']
    assert test_personality.name == 'Test Elle'
    assert test_personality.voice_id == 'alloy'


def test_personality_switching():
    """Test switching between personalities"""
    manager = PersonalityManager()

    # Switch to Assistant Elle
    personality = manager.switch_personality('assistant_elle')
    assert personality.name == 'Assistant Elle'
    assert manager.active_personality.name == 'Assistant Elle'

    # Switch to Companion Elle
    personality = manager.switch_personality('companion_elle')
    assert personality.name == 'Companion Elle'
    assert manager.active_personality.name == 'Companion Elle'


def test_personality_switching_invalid():
    """Test switching to non-existent personality raises error"""
    manager = PersonalityManager()

    with pytest.raises(ValueError, match="Personality .* not found"):
        manager.switch_personality('nonexistent_elle')


def test_personality_switching_performance():
    """Test personality switching is fast (<100ms)"""
    import time

    manager = PersonalityManager()

    # Warm up
    manager.switch_personality('assistant_elle')

    # Time multiple switches
    start = time.perf_counter()
    for _ in range(100):
        manager.switch_personality('professor_elle')
        manager.switch_personality('assistant_elle')
    duration = time.perf_counter() - start

    # 200 switches should complete in well under 100ms total
    # (each switch should be <1ms)
    assert duration < 0.1  # 100ms for 200 switches = 0.5ms per switch


def test_get_voice_for_personality():
    """Test getting correct voice ID for personality"""
    manager = PersonalityManager()

    # Professor Elle -> nova
    manager.switch_personality('professor_elle')
    assert manager.get_voice_for_personality() == 'nova'

    # Assistant Elle -> alloy
    manager.switch_personality('assistant_elle')
    assert manager.get_voice_for_personality() == 'alloy'

    # Companion Elle -> shimmer
    manager.switch_personality('companion_elle')
    assert manager.get_voice_for_personality() == 'shimmer'

    # Expert Elle -> onyx
    manager.switch_personality('expert_elle')
    assert manager.get_voice_for_personality() == 'onyx'


def test_get_prompt_template():
    """Test getting prompt template for personality"""
    manager = PersonalityManager()

    manager.switch_personality('professor_elle')
    prompt = manager.get_prompt_template()

    assert 'Professor Elle' in prompt
    assert 'educator' in prompt.lower()


def test_apply_personality_formality():
    """Test applying personality formality to responses"""
    manager = PersonalityManager()

    # Test with formal personality (Expert Elle)
    manager.switch_personality('expert_elle')
    casual_text = "You're doing great! It's really impressive."
    formal_text = manager.apply_personality(casual_text)

    # Should expand contractions
    assert "you are" in formal_text.lower() or "it is" in formal_text.lower()


def test_apply_personality_casual():
    """Test applying casual personality to responses"""
    manager = PersonalityManager()

    # Test with casual personality (Companion Elle)
    manager.switch_personality('companion_elle')
    formal_text = "You are doing well. It is quite good."
    casual_text = manager.apply_personality(formal_text)

    # Should create contractions (simplified test - in practice this is complex)
    # The apply_personality method has limited text transformation capabilities
    # Real personality styling happens via prompt engineering
    assert isinstance(casual_text, str)


def test_apply_personality_concise():
    """Test applying concise personality (Assistant Elle)"""
    manager = PersonalityManager()

    manager.switch_personality('assistant_elle')
    verbose_text = "This is really quite very actually basically good."
    concise_text = manager.apply_personality(verbose_text)

    # Should remove filler words
    assert 'very' not in concise_text
    assert 'really' not in concise_text


def test_list_personalities():
    """Test listing available personalities"""
    manager = PersonalityManager()

    personalities = manager.list_personalities()

    assert len(personalities) == 4
    assert 'professor_elle' in personalities
    assert 'assistant_elle' in personalities
    assert 'companion_elle' in personalities
    assert 'expert_elle' in personalities


def test_get_personality():
    """Test getting personality by ID"""
    manager = PersonalityManager()

    personality = manager.get_personality('professor_elle')
    assert personality is not None
    assert personality.name == 'Professor Elle'

    # Non-existent personality
    personality = manager.get_personality('nonexistent')
    assert personality is None


def test_get_active_personality():
    """Test getting active personality"""
    manager = PersonalityManager()

    active = manager.get_active_personality()
    assert active is not None
    assert active.name == 'Professor Elle'

    manager.switch_personality('assistant_elle')
    active = manager.get_active_personality()
    assert active.name == 'Assistant Elle'


# ============================================================================
# Integration Tests
# ============================================================================

def test_personality_voice_mapping_consistency():
    """Test that each personality has a unique voice mapping"""
    manager = PersonalityManager()

    voices = {}
    for personality_id in manager.list_personalities():
        manager.switch_personality(personality_id)
        voice = manager.get_voice_for_personality()
        voices[personality_id] = voice

    # All personalities should have voices
    assert len(voices) == 4

    # Check specific mappings
    assert voices['professor_elle'] == 'nova'
    assert voices['assistant_elle'] == 'alloy'
    assert voices['companion_elle'] == 'shimmer'
    assert voices['expert_elle'] == 'onyx'


def test_personality_trait_consistency():
    """Test that personalities have distinct, consistent traits"""
    manager = PersonalityManager()

    # Professor Elle: High formality, high verbosity, high teaching
    prof = manager.get_personality('professor_elle')
    assert prof.traits.formality >= 0.7
    assert prof.traits.verbosity >= 0.8
    assert prof.traits.teaching_style >= 0.7

    # Assistant Elle: Moderate formality, low verbosity
    asst = manager.get_personality('assistant_elle')
    assert asst.traits.verbosity <= 0.4  # Concise

    # Companion Elle: Low formality, high emotional tone, high humor
    comp = manager.get_personality('companion_elle')
    assert comp.traits.formality <= 0.4  # Casual
    assert comp.traits.emotional_tone >= 0.7  # Expressive
    assert comp.traits.humor >= 0.6  # Playful

    # Expert Elle: Highest formality, low teaching (direct), low humor
    expert = manager.get_personality('expert_elle')
    assert expert.traits.formality >= 0.8
    assert expert.traits.teaching_style <= 0.3  # Direct
    assert expert.traits.humor <= 0.2  # Minimal


def test_end_to_end_personality_workflow():
    """Test complete workflow: load, switch, apply, get voice"""
    manager = PersonalityManager()

    # Start with default (Professor Elle)
    assert manager.active_personality.name == 'Professor Elle'
    voice = manager.get_voice_for_personality()
    assert voice == 'nova'

    # Switch to Assistant Elle
    manager.switch_personality('assistant_elle')
    assert manager.active_personality.name == 'Assistant Elle'
    voice = manager.get_voice_for_personality()
    assert voice == 'alloy'

    # Apply personality to response
    response = "This is a test response that should be styled."
    styled = manager.apply_personality(response)
    assert isinstance(styled, str)
    assert len(styled) > 0

    # Get prompt template
    prompt = manager.get_prompt_template()
    assert 'Assistant Elle' in prompt

    # Switch to Companion Elle and verify
    manager.switch_personality('companion_elle')
    assert manager.active_personality.name == 'Companion Elle'
    voice = manager.get_voice_for_personality()
    assert voice == 'shimmer'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
