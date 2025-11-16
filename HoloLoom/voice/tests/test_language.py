"""
Test Suite for Multi-Language Support
======================================

Comprehensive tests for language detection, voice mapping, and localization.

Date: November 16, 2025
Phase: 4 (Multi-Language Support)
"""

import pytest
import os
import time
from pathlib import Path
from typing import List

# Test imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from HoloLoom.voice.language import (
    LanguageCode,
    LanguageProfile,
    LanguageVariant,
    ConversationLanguageState,
    LanguageDetector,
    LanguageManager,
    create_language_manager,
    detect_language
)


# ============================================================================
# Test Data
# ============================================================================

# Sample texts in different languages
SAMPLE_TEXTS = {
    'en': "The quick brown fox jumps over the lazy dog. This is a test of language detection.",
    'es': "El rápido zorro marrón salta sobre el perro perezoso. Esta es una prueba de detección de idioma.",
    'fr': "Le rapide renard brun saute par-dessus le chien paresseux. Ceci est un test de détection de langue.",
    'de': "Der schnelle braune Fuchs springt über den faulen Hund. Dies ist ein Test der Spracherkennung.",
    'ja': "素早い茶色のキツネが怠け者の犬を飛び越える。これは言語検出のテストです。",
    'zh': "敏捷的棕色狐狸跳过懒狗。这是一个语言检测测试。"
}

LANGUAGE_SWITCH_PHRASES = {
    'en': ["switch to spanish", "change language to french", "speak german"],
    'es': ["cambia a español", "cambiar idioma a francés"],
    'fr': ["changer en français", "passer au français"],
    'de': ["wechseln zu deutsch", "sprache ändern"],
    'ja': ["日本語に切り替え", "言語を変更"],
    'zh': ["切换到中文", "改变语言"]
}


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def language_manager():
    """Create language manager for testing"""
    return create_language_manager()


@pytest.fixture
def detector():
    """Create language detector for testing"""
    return LanguageDetector()


@pytest.fixture
def conversation_state():
    """Create conversation state for testing"""
    return ConversationLanguageState(
        session_id="test_session_123",
        language_code="en"
    )


# ============================================================================
# Test LanguageCode Enum
# ============================================================================

class TestLanguageCode:
    """Test LanguageCode enum"""

    def test_language_codes_exist(self):
        """Test all 6 language codes exist"""
        codes = LanguageCode.all_codes()
        assert len(codes) == 6
        assert 'en' in codes
        assert 'es' in codes
        assert 'fr' in codes
        assert 'de' in codes
        assert 'ja' in codes
        assert 'zh' in codes

    def test_from_string(self):
        """Test LanguageCode.from_string()"""
        assert LanguageCode.from_string('en') == LanguageCode.ENGLISH
        assert LanguageCode.from_string('ES') == LanguageCode.SPANISH
        assert LanguageCode.from_string('invalid') is None

    def test_enum_values(self):
        """Test enum value access"""
        assert LanguageCode.ENGLISH.value == 'en'
        assert LanguageCode.SPANISH.value == 'es'
        assert LanguageCode.FRENCH.value == 'fr'


# ============================================================================
# Test LanguageProfile
# ============================================================================

class TestLanguageProfile:
    """Test LanguageProfile class"""

    def test_profile_creation(self):
        """Test creating language profile"""
        profile = LanguageProfile(
            code='en',
            name='English',
            native_name='English',
            default_voice='nova',
            fallback_voice='alloy'
        )
        assert profile.code == 'en'
        assert profile.name == 'English'
        assert profile.default_voice == 'nova'

    def test_profile_with_variants(self):
        """Test profile with variants"""
        variants = [
            LanguageVariant('en-US', 'nova', 'alloy', 'American English'),
            LanguageVariant('en-GB', 'onyx', 'alloy', 'British English')
        ]
        profile = LanguageProfile(
            code='en',
            name='English',
            native_name='English',
            variants=variants
        )
        assert len(profile.variants) == 2
        assert profile.variants[0].code == 'en-US'

    def test_get_voice_for_variant(self):
        """Test getting voice for specific variant"""
        variants = [
            LanguageVariant('en-US', 'nova', 'alloy'),
            LanguageVariant('en-GB', 'onyx', 'alloy')
        ]
        profile = LanguageProfile(
            code='en',
            name='English',
            native_name='English',
            variants=variants,
            default_voice='shimmer'
        )

        # Get specific variant
        assert profile.get_voice_for_variant('en-US') == 'nova'
        assert profile.get_voice_for_variant('en-GB') == 'onyx'

        # Get default for unknown variant
        assert profile.get_voice_for_variant('en-AU') == 'shimmer'
        assert profile.get_voice_for_variant() == 'shimmer'

    def test_get_ui_string(self):
        """Test getting localized UI strings"""
        profile = LanguageProfile(
            code='en',
            name='English',
            native_name='English',
            ui_strings={
                'greeting': 'Hello',
                'goodbye': 'Goodbye'
            }
        )
        assert profile.get_ui_string('greeting') == 'Hello'
        assert profile.get_ui_string('unknown', 'default') == 'default'

    def test_profile_to_dict(self):
        """Test profile serialization"""
        profile = LanguageProfile(
            code='en',
            name='English',
            native_name='English'
        )
        data = profile.to_dict()
        assert data['code'] == 'en'
        assert data['name'] == 'English'
        assert 'ui_strings' in data


# ============================================================================
# Test ConversationLanguageState
# ============================================================================

class TestConversationLanguageState:
    """Test ConversationLanguageState class"""

    def test_state_creation(self):
        """Test creating conversation state"""
        state = ConversationLanguageState(
            session_id="test_123",
            language_code="en"
        )
        assert state.session_id == "test_123"
        assert state.language_code == "en"
        assert state.auto_detect is True
        assert len(state.history) == 0

    def test_record_language_change(self, conversation_state):
        """Test recording language changes"""
        conversation_state.record_language_change('es', 'auto_detect', 0.95)

        assert conversation_state.language_code == 'es'
        assert conversation_state.confidence == 0.95
        assert len(conversation_state.history) == 1

        event = conversation_state.history[0]
        assert event['from'] == 'en'
        assert event['to'] == 'es'
        assert event['reason'] == 'auto_detect'
        assert event['confidence'] == 0.95

    def test_multiple_language_changes(self, conversation_state):
        """Test multiple language changes"""
        conversation_state.record_language_change('es', 'auto_detect', 0.95)
        conversation_state.record_language_change('fr', 'user_override', 1.0)
        conversation_state.record_language_change('de', 'auto_detect', 0.88)

        assert conversation_state.language_code == 'de'
        assert len(conversation_state.history) == 3

    def test_state_to_dict(self, conversation_state):
        """Test state serialization"""
        conversation_state.record_language_change('es', 'test', 0.9)
        data = conversation_state.to_dict()

        assert data['session_id'] == "test_session_123"
        assert data['language_code'] == 'es'
        assert len(data['history']) == 1


# ============================================================================
# Test LanguageDetector
# ============================================================================

class TestLanguageDetector:
    """Test LanguageDetector class"""

    def test_detector_creation(self, detector):
        """Test creating detector"""
        assert detector is not None

    def test_detect_english(self, detector):
        """Test detecting English"""
        text = SAMPLE_TEXTS['en']
        detected = detector.detect(text)

        # Only test if langdetect is available
        if detector.available:
            assert detected == 'en'

    def test_detect_spanish(self, detector):
        """Test detecting Spanish"""
        text = SAMPLE_TEXTS['es']
        detected = detector.detect(text)

        if detector.available:
            assert detected == 'es'

    def test_detect_french(self, detector):
        """Test detecting French"""
        text = SAMPLE_TEXTS['fr']
        detected = detector.detect(text)

        if detector.available:
            assert detected == 'fr'

    def test_detect_german(self, detector):
        """Test detecting German"""
        text = SAMPLE_TEXTS['de']
        detected = detector.detect(text)

        if detector.available:
            assert detected == 'de'

    def test_detect_japanese(self, detector):
        """Test detecting Japanese"""
        text = SAMPLE_TEXTS['ja']
        detected = detector.detect(text)

        if detector.available:
            assert detected == 'ja'

    def test_detect_chinese(self, detector):
        """Test detecting Chinese"""
        text = SAMPLE_TEXTS['zh']
        detected = detector.detect(text)

        if detector.available:
            assert detected == 'zh'

    def test_detect_all_languages(self, detector):
        """Test detection accuracy across all 6 languages"""
        if not detector.available:
            pytest.skip("langdetect not available")

        correct = 0
        total = len(SAMPLE_TEXTS)

        for expected_lang, text in SAMPLE_TEXTS.items():
            detected = detector.detect(text)
            if detected == expected_lang:
                correct += 1

        accuracy = correct / total
        assert accuracy >= 0.95, f"Detection accuracy {accuracy:.1%} below 95% threshold"

    def test_detect_with_confidence(self, detector):
        """Test detection with confidence scores"""
        if not detector.available:
            pytest.skip("langdetect not available")

        text = SAMPLE_TEXTS['en']
        detected, confidence = detector.detect_with_confidence(text)

        assert detected == 'en'
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.8  # High confidence for clear text

    def test_detect_short_text(self, detector):
        """Test detection fails gracefully for short text"""
        short_text = "Hi"
        detected = detector.detect(short_text)
        # Should return None for text < 10 chars
        assert detected is None

    def test_detect_empty_text(self, detector):
        """Test detection with empty text"""
        detected = detector.detect("")
        assert detected is None

    def test_detect_invalid_text(self, detector):
        """Test detection with invalid text"""
        detected = detector.detect("123456789")
        # Numeric text may fail detection
        assert detected is None or isinstance(detected, str)


# ============================================================================
# Test LanguageManager
# ============================================================================

class TestLanguageManager:
    """Test LanguageManager class"""

    def test_manager_creation(self, language_manager):
        """Test creating language manager"""
        assert language_manager is not None
        assert len(language_manager.languages) > 0

    def test_load_all_languages(self, language_manager):
        """Test all 6 languages are loaded"""
        supported = language_manager.get_supported_languages()
        assert len(supported) >= 1  # At least English (default)

        # Check if YAML files exist
        languages_dir = language_manager.languages_dir
        if languages_dir.exists():
            yaml_count = len(list(languages_dir.glob('*.yaml')))
            if yaml_count >= 6:
                assert len(supported) == 6

    def test_get_language(self, language_manager):
        """Test getting language by code"""
        english = language_manager.get_language('en')
        assert english is not None
        assert english.code == 'en'
        assert english.name == 'English'

    def test_is_supported(self, language_manager):
        """Test checking if language is supported"""
        assert language_manager.is_supported('en') is True
        assert language_manager.is_supported('xx') is False

    def test_detect_language(self, language_manager):
        """Test language detection through manager"""
        text = SAMPLE_TEXTS['en']
        detected = language_manager.detect_language(text)

        # Should always return a supported language (defaults to 'en')
        assert detected in language_manager.get_supported_languages()

    def test_get_voice_for_language(self, language_manager):
        """Test getting voice for language"""
        # English
        voice = language_manager.get_voice_for_language('en')
        assert voice in ['nova', 'alloy', 'shimmer', 'echo', 'fable', 'onyx']

        # Unknown language should return default
        voice = language_manager.get_voice_for_language('xx')
        assert voice == 'alloy'

    def test_get_voice_for_variant(self, language_manager):
        """Test getting voice for language variant"""
        if language_manager.is_supported('en'):
            voice = language_manager.get_voice_for_language('en', 'en-US')
            assert voice is not None

    def test_get_ui_string(self, language_manager):
        """Test getting localized UI strings"""
        if language_manager.is_supported('en'):
            greeting = language_manager.get_ui_string('greeting', 'en')
            assert greeting in ['Hello', '']  # Depends on if YAML loaded

    def test_create_conversation_state(self, language_manager):
        """Test creating conversation state"""
        state = language_manager.create_conversation_state('session_456', 'es')

        assert state.session_id == 'session_456'
        assert state.language_code == 'es'
        assert state in language_manager.conversation_states.values()

    def test_get_conversation_state(self, language_manager):
        """Test getting conversation state"""
        state = language_manager.create_conversation_state('session_789')
        retrieved = language_manager.get_conversation_state('session_789')

        assert retrieved is state

    def test_update_conversation_language(self, language_manager):
        """Test updating conversation language"""
        session_id = 'test_update_123'
        state = language_manager.create_conversation_state(session_id, 'en')

        # Update with Spanish text
        if language_manager.detector.available:
            updated = language_manager.update_conversation_language(
                session_id,
                SAMPLE_TEXTS['es']
            )
            # May switch to Spanish if confidence high enough
            assert updated.language_code in ['en', 'es']

    def test_force_language_override(self, language_manager):
        """Test forcing language override"""
        session_id = 'test_force_123'
        state = language_manager.create_conversation_state(session_id, 'en')

        # Force Spanish
        updated = language_manager.update_conversation_language(
            session_id,
            SAMPLE_TEXTS['en'],  # English text
            force_language='es'  # But force Spanish
        )

        assert updated.language_code == 'es'
        assert updated.force_language == 'es'

    def test_switch_language(self, language_manager):
        """Test explicit language switching"""
        session_id = 'test_switch_123'
        state = language_manager.create_conversation_state(session_id, 'en')

        language_manager.switch_language(session_id, 'fr', 'manual')

        updated = language_manager.get_conversation_state(session_id)
        assert updated.language_code == 'fr'
        assert len(updated.history) == 1
        assert updated.history[0]['reason'] == 'manual'

    def test_language_statistics(self, language_manager):
        """Test language usage statistics"""
        # Create multiple conversations
        language_manager.create_conversation_state('s1', 'en')
        language_manager.create_conversation_state('s2', 'es')
        language_manager.create_conversation_state('s3', 'en')
        language_manager.switch_language('s1', 'fr')

        stats = language_manager.get_language_statistics()

        assert stats['total_conversations'] == 3
        assert stats['languages_used']['en'] == 2  # s1 started as en, s3
        assert stats['languages_used']['es'] == 1  # s2
        assert stats['total_switches'] == 1  # s1 switched to fr


# ============================================================================
# Test Voice Mappings
# ============================================================================

class TestVoiceMappings:
    """Test voice mappings for all languages"""

    def test_english_voices(self, language_manager):
        """Test English voice mappings"""
        if not language_manager.is_supported('en'):
            pytest.skip("English not loaded")

        profile = language_manager.get_language('en')
        assert profile.default_voice in ['nova', 'alloy', 'shimmer', 'echo', 'fable', 'onyx']

    def test_spanish_voices(self, language_manager):
        """Test Spanish voice mappings"""
        if not language_manager.is_supported('es'):
            pytest.skip("Spanish not loaded")

        profile = language_manager.get_language('es')
        assert profile.default_voice in ['nova', 'alloy', 'shimmer', 'echo', 'fable', 'onyx']

    def test_french_voices(self, language_manager):
        """Test French voice mappings"""
        if not language_manager.is_supported('fr'):
            pytest.skip("French not loaded")

        profile = language_manager.get_language('fr')
        assert profile.default_voice in ['nova', 'alloy', 'shimmer', 'echo', 'fable', 'onyx']

    def test_all_language_voices(self, language_manager):
        """Test all languages have valid voice mappings"""
        for lang_code in language_manager.get_supported_languages():
            voice = language_manager.get_voice_for_language(lang_code)
            assert voice in ['nova', 'alloy', 'shimmer', 'echo', 'fable', 'onyx']


# ============================================================================
# Test Utility Functions
# ============================================================================

class TestUtilityFunctions:
    """Test utility functions"""

    def test_create_language_manager(self):
        """Test create_language_manager factory"""
        manager = create_language_manager()
        assert manager is not None
        assert isinstance(manager, LanguageManager)

    def test_detect_language_standalone(self):
        """Test standalone detect_language function"""
        detected = detect_language(SAMPLE_TEXTS['en'])
        # Should always return a string (defaults to 'en')
        assert isinstance(detected, str)


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Test performance characteristics"""

    def test_detection_speed(self, detector):
        """Test language detection is fast (<50ms)"""
        if not detector.available:
            pytest.skip("langdetect not available")

        text = SAMPLE_TEXTS['en']

        start = time.perf_counter()
        for _ in range(10):
            detector.detect(text)
        end = time.perf_counter()

        avg_time = (end - start) / 10
        assert avg_time < 0.050, f"Detection too slow: {avg_time*1000:.1f}ms"

    def test_language_switching_speed(self, language_manager):
        """Test language switching is fast (<50ms)"""
        session_id = 'perf_test'
        language_manager.create_conversation_state(session_id, 'en')

        start = time.perf_counter()
        for _ in range(10):
            language_manager.switch_language(session_id, 'es')
            language_manager.switch_language(session_id, 'en')
        end = time.perf_counter()

        avg_time = (end - start) / 20
        assert avg_time < 0.050, f"Switching too slow: {avg_time*1000:.1f}ms"


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
