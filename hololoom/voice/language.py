"""
Language Management System for HoloLoom VoiceAgent
===================================================

Multi-language support with automatic detection, voice mapping, and UI localization.

Features:
- 6+ language support (English, Spanish, French, German, Japanese, Mandarin)
- Automatic language detection using langdetect
- Per-language voice mapping (OpenAI TTS voices)
- Conversation language persistence
- Language switching support
- UI string localization
- Fallback logic for unsupported languages

Date: November 16, 2025
Phase: 4 (Multi-Language Support)
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
    logger = structlog.get_logger()
except ImportError:
    STRUCTLOG_AVAILABLE = False
    import logging
    logger = logging.getLogger(__name__)

# Language detection
try:
    from langdetect import DetectorFactory, LangDetectException, detect
    # Set seed for reproducibility
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logger.warning("langdetect not available - language detection disabled, defaulting to English")


# ============================================================================
# Data Models
# ============================================================================

class LanguageCode(Enum):
    """Supported language codes (ISO 639-1)"""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE = "zh"  # Mandarin

    @classmethod
    def from_string(cls, code: str) -> Optional['LanguageCode']:
        """Get LanguageCode from string"""
        code = code.lower()
        for lang in cls:
            if lang.value == code:
                return lang
        return None

    @classmethod
    def all_codes(cls) -> list[str]:
        """Get all supported language codes"""
        return [lang.value for lang in cls]


@dataclass
class LanguageVariant:
    """
    Language variant (e.g., en-US vs en-GB)

    Attributes:
        code: Variant code (e.g., "en-US")
        voice_id: OpenAI TTS voice for this variant
        fallback_voice: Fallback voice if primary unavailable
        name: Human-readable name
    """
    code: str
    voice_id: str
    fallback_voice: str
    name: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            'code': self.code,
            'voice_id': self.voice_id,
            'fallback_voice': self.fallback_voice,
            'name': self.name
        }


@dataclass
class LanguageProfile:
    """
    Complete language profile with voice mapping and UI strings

    Attributes:
        code: ISO 639-1 language code
        name: Language name in English
        native_name: Language name in native script
        variants: List of language variants (e.g., en-US, en-GB)
        default_voice: Default OpenAI TTS voice
        fallback_voice: Fallback voice if default unavailable
        rtl: Right-to-left script (Arabic, Hebrew)
        ui_strings: Localized UI strings
        metadata: Additional metadata
    """
    code: str
    name: str
    native_name: str
    variants: list[LanguageVariant] = field(default_factory=list)
    default_voice: str = "alloy"
    fallback_voice: str = "alloy"
    rtl: bool = False
    ui_strings: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_voice_for_variant(self, variant_code: str | None = None) -> str:
        """Get voice for specific variant or default"""
        if variant_code:
            for variant in self.variants:
                if variant.code == variant_code:
                    return variant.voice_id
        return self.default_voice

    def get_ui_string(self, key: str, default: str = "") -> str:
        """Get localized UI string"""
        return self.ui_strings.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            'code': self.code,
            'name': self.name,
            'native_name': self.native_name,
            'variants': [v.to_dict() for v in self.variants],
            'default_voice': self.default_voice,
            'fallback_voice': self.fallback_voice,
            'rtl': self.rtl,
            'ui_strings': self.ui_strings,
            'metadata': self.metadata
        }

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'LanguageProfile':
        """Load language profile from YAML file"""
        with open(yaml_path, encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # Parse variants
        variants = []
        for variant_data in data.get('variants', []):
            for variant_code, variant_info in variant_data.items():
                variants.append(LanguageVariant(
                    code=variant_code,
                    voice_id=variant_info['voice_id'],
                    fallback_voice=variant_info['fallback'],
                    name=variant_info.get('name', '')
                ))

        return cls(
            code=data['code'],
            name=data['name'],
            native_name=data['native_name'],
            variants=variants,
            default_voice=data.get('default_voice', 'alloy'),
            fallback_voice=data.get('fallback_voice', 'alloy'),
            rtl=data.get('rtl', False),
            ui_strings=data.get('ui_strings', {}),
            metadata=data.get('metadata', {})
        )


@dataclass
class ConversationLanguageState:
    """
    Tracks language state across conversation turns

    Attributes:
        session_id: Conversation session identifier
        language_code: Current language code (ISO 639-1)
        variant_code: Current variant code (e.g., en-US)
        confidence: Language detection confidence (0.0-1.0)
        auto_detect: Whether to auto-detect language changes
        force_language: User-forced language override
        history: Language change history
    """
    session_id: str
    language_code: str = "en"
    variant_code: str | None = None
    confidence: float = 1.0
    auto_detect: bool = True
    force_language: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def record_language_change(self, new_language: str, reason: str, confidence: float = 1.0):
        """Record language change event"""
        self.history.append({
            'from': self.language_code,
            'to': new_language,
            'reason': reason,
            'confidence': confidence,
            'timestamp': __import__('time').time()
        })
        self.language_code = new_language
        self.confidence = confidence

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            'session_id': self.session_id,
            'language_code': self.language_code,
            'variant_code': self.variant_code,
            'confidence': self.confidence,
            'auto_detect': self.auto_detect,
            'force_language': self.force_language,
            'history': self.history
        }


# ============================================================================
# Language Detection
# ============================================================================

class LanguageDetector:
    """
    Automatic language detection using langdetect

    Uses langdetect library for fast, accurate language detection.
    Supports 55+ languages including all 6 HoloLoom supported languages.
    """

    def __init__(self):
        self.available = LANGDETECT_AVAILABLE
        if not self.available:
            logger.warning("Language detection unavailable - install langdetect: pip install langdetect")

    def detect(self, text: str) -> str | None:
        """
        Detect language from text

        Args:
            text: Input text

        Returns:
            ISO 639-1 language code or None if detection fails
        """
        if not self.available:
            return None

        if not text or len(text.strip()) < 10:
            # Too short for reliable detection
            return None

        try:
            detected = detect(text)
            return detected
        except (LangDetectException, Exception) as e:
            logger.warning(f"Language detection failed: {e}")
            return None

    def detect_with_confidence(self, text: str) -> tuple[str | None, float]:
        """
        Detect language with confidence score

        Args:
            text: Input text

        Returns:
            Tuple of (language_code, confidence) or (None, 0.0)
        """
        if not self.available:
            return (None, 0.0)

        if not text or len(text.strip()) < 10:
            return (None, 0.0)

        try:
            from langdetect import detect_langs
            results = detect_langs(text)
            if results:
                # Get top result
                top = results[0]
                return (top.lang, top.prob)
            return (None, 0.0)
        except (LangDetectException, Exception) as e:
            logger.warning(f"Language detection failed: {e}")
            return (None, 0.0)


# ============================================================================
# Language Manager
# ============================================================================

class LanguageManager:
    """
    Central language management system

    Handles:
    - Language profile loading
    - Language detection
    - Voice mapping
    - UI string localization
    - Conversation language persistence
    """

    def __init__(self, languages_dir: Path | None = None):
        """
        Initialize language manager

        Args:
            languages_dir: Directory containing language YAML files
        """
        self.languages_dir = languages_dir or self._get_default_languages_dir()
        self.languages: dict[str, LanguageProfile] = {}
        self.detector = LanguageDetector()
        self.conversation_states: dict[str, ConversationLanguageState] = {}

        # Load all languages
        self._load_languages()

    def _get_default_languages_dir(self) -> Path:
        """Get default languages directory"""
        return Path(__file__).parent / 'languages'

    def _load_languages(self):
        """Load all language profiles from YAML files"""
        if not self.languages_dir.exists():
            logger.warning(f"Languages directory not found: {self.languages_dir}")
            # Create default English profile
            self._create_default_english_profile()
            return

        yaml_files = list(self.languages_dir.glob('*.yaml'))
        if not yaml_files:
            logger.warning(f"No language YAML files found in {self.languages_dir}")
            self._create_default_english_profile()
            return

        for yaml_file in yaml_files:
            try:
                profile = LanguageProfile.from_yaml(yaml_file)
                self.languages[profile.code] = profile
                logger.info(f"Loaded language profile: {profile.name} ({profile.code})")
            except Exception as e:
                logger.error(f"Failed to load language {yaml_file}: {e}")

        if not self.languages:
            self._create_default_english_profile()

    def _create_default_english_profile(self):
        """Create default English profile as fallback"""
        logger.info("Creating default English language profile")
        self.languages['en'] = LanguageProfile(
            code='en',
            name='English',
            native_name='English',
            default_voice='nova',
            fallback_voice='alloy',
            ui_strings={
                'greeting': 'Hello',
                'processing': 'Processing your request...',
                'error': "I'm sorry, there was an error.",
                'thinking': 'Let me think about that...',
                'complete': 'Done!',
                'goodbye': 'Goodbye!'
            }
        )

    def get_language(self, code: str) -> LanguageProfile | None:
        """Get language profile by code"""
        return self.languages.get(code)

    def get_supported_languages(self) -> list[str]:
        """Get list of supported language codes"""
        return list(self.languages.keys())

    def is_supported(self, code: str) -> bool:
        """Check if language is supported"""
        return code in self.languages

    def detect_language(self, text: str) -> str | None:
        """
        Detect language from text

        Args:
            text: Input text

        Returns:
            Language code or None if detection fails
        """
        detected = self.detector.detect(text)

        # Check if detected language is supported
        if detected and self.is_supported(detected):
            return detected

        # Fallback to English
        return 'en'

    def detect_language_with_confidence(self, text: str) -> tuple[str, float]:
        """
        Detect language with confidence score

        Args:
            text: Input text

        Returns:
            Tuple of (language_code, confidence)
        """
        detected, confidence = self.detector.detect_with_confidence(text)

        # Check if detected language is supported
        if detected and self.is_supported(detected):
            return (detected, confidence)

        # Fallback to English with low confidence
        return ('en', 0.5)

    def get_voice_for_language(self, language_code: str, variant_code: str | None = None) -> str:
        """
        Get appropriate voice for language/variant

        Args:
            language_code: ISO 639-1 language code
            variant_code: Optional variant code (e.g., en-US)

        Returns:
            OpenAI TTS voice ID
        """
        profile = self.get_language(language_code)
        if not profile:
            # Fallback to default voice
            return 'alloy'

        return profile.get_voice_for_variant(variant_code)

    def get_ui_string(self, key: str, language_code: str, default: str = "") -> str:
        """
        Get localized UI string

        Args:
            key: UI string key
            language_code: Language code
            default: Default value if not found

        Returns:
            Localized string
        """
        profile = self.get_language(language_code)
        if not profile:
            return default

        return profile.get_ui_string(key, default)

    def create_conversation_state(self, session_id: str, initial_language: str = "en") -> ConversationLanguageState:
        """
        Create conversation language state

        Args:
            session_id: Session identifier
            initial_language: Initial language code

        Returns:
            ConversationLanguageState instance
        """
        state = ConversationLanguageState(
            session_id=session_id,
            language_code=initial_language
        )
        self.conversation_states[session_id] = state
        return state

    def get_conversation_state(self, session_id: str) -> ConversationLanguageState | None:
        """Get conversation language state"""
        return self.conversation_states.get(session_id)

    def update_conversation_language(
        self,
        session_id: str,
        text: str,
        force_language: str | None = None
    ) -> ConversationLanguageState:
        """
        Update conversation language based on input text

        Args:
            session_id: Session identifier
            text: Input text for language detection
            force_language: Force specific language (user override)

        Returns:
            Updated ConversationLanguageState
        """
        # Get or create state
        state = self.conversation_states.get(session_id)
        if not state:
            state = self.create_conversation_state(session_id)

        # Force language if specified
        if force_language:
            if force_language != state.language_code:
                state.record_language_change(force_language, "user_override", 1.0)
                state.force_language = force_language
            return state

        # Skip auto-detection if disabled or forced
        if not state.auto_detect or state.force_language:
            return state

        # Detect language from text
        detected, confidence = self.detect_language_with_confidence(text)

        # Only change if confidence is high enough (>0.7) and different from current
        if confidence > 0.7 and detected != state.language_code:
            state.record_language_change(detected, "auto_detect", confidence)
            logger.info(f"Language auto-switched: {state.language_code} -> {detected} (confidence: {confidence:.2f})")

        return state

    def switch_language(self, session_id: str, new_language: str, reason: str = "manual"):
        """
        Explicitly switch conversation language

        Args:
            session_id: Session identifier
            new_language: New language code
            reason: Reason for switch
        """
        state = self.conversation_states.get(session_id)
        if not state:
            state = self.create_conversation_state(session_id, new_language)
            return

        if new_language != state.language_code:
            state.record_language_change(new_language, reason, 1.0)

    def get_language_statistics(self) -> dict[str, Any]:
        """Get language usage statistics across all conversations"""
        stats = {
            'total_conversations': len(self.conversation_states),
            'languages_used': {},
            'total_switches': 0
        }

        for state in self.conversation_states.values():
            lang = state.language_code
            stats['languages_used'][lang] = stats['languages_used'].get(lang, 0) + 1
            stats['total_switches'] += len(state.history)

        return stats


# ============================================================================
# Utility Functions
# ============================================================================

def create_language_manager(languages_dir: Path | None = None) -> LanguageManager:
    """
    Create language manager instance

    Args:
        languages_dir: Optional custom languages directory

    Returns:
        LanguageManager instance
    """
    return LanguageManager(languages_dir=languages_dir)


def detect_language(text: str) -> str:
    """
    Quick language detection (standalone)

    Args:
        text: Input text

    Returns:
        Language code (defaults to 'en')
    """
    detector = LanguageDetector()
    detected = detector.detect(text)
    return detected or 'en'
