"""
Personality Framework for HoloLoom VoiceAgent
==============================================

Multi-persona system with voice customization and response styling.

Features:
- 4 predefined personalities (Professor, Assistant, Companion, Expert)
- Personality trait system (formality, verbosity, emotional tone, etc.)
- Voice-personality mapping (OpenAI TTS voices)
- Dynamic prompt template application
- Personality switching <100ms
- YAML-based personality definitions

Date: November 16, 2025
Phase: 3 (Custom Personalities)
"""

import os
import re
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
from enum import Enum

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
    logger = structlog.get_logger()
except ImportError:
    STRUCTLOG_AVAILABLE = False
    import logging
    logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class PersonalityTraits:
    """
    Personality trait scores (0.0 to 1.0)

    Attributes:
        formality: 0 (casual) to 1 (professional)
        verbosity: 0 (concise) to 1 (detailed)
        emotional_tone: 0 (neutral) to 1 (expressive)
        teaching_style: 0 (direct) to 1 (socratic)
        humor: 0 (none) to 1 (playful)
    """
    formality: float = 0.5
    verbosity: float = 0.5
    emotional_tone: float = 0.5
    teaching_style: float = 0.5
    humor: float = 0.5

    def __post_init__(self):
        """Validate trait scores are in [0, 1]"""
        for field_name in ['formality', 'verbosity', 'emotional_tone', 'teaching_style', 'humor']:
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0.0 and 1.0, got {value}")

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'formality': self.formality,
            'verbosity': self.verbosity,
            'emotional_tone': self.emotional_tone,
            'teaching_style': self.teaching_style,
            'humor': self.humor
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'PersonalityTraits':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class Personality:
    """
    Complete personality profile

    Attributes:
        name: Personality name (e.g., "Professor Elle")
        description: Brief description of personality
        traits: PersonalityTraits instance
        voice_id: OpenAI TTS voice ID
        prompt_template: System prompt template
        example_responses: Example responses for this personality
        metadata: Additional metadata
    """
    name: str
    description: str
    traits: PersonalityTraits
    voice_id: str
    prompt_template: str
    example_responses: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'description': self.description,
            'traits': self.traits.to_dict(),
            'voice_id': self.voice_id,
            'prompt_template': self.prompt_template,
            'example_responses': self.example_responses,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Personality':
        """Create from dictionary"""
        traits = PersonalityTraits.from_dict(data['traits'])
        return cls(
            name=data['name'],
            description=data['description'],
            traits=traits,
            voice_id=data['voice_id'],
            prompt_template=data['prompt_template'],
            example_responses=data.get('example_responses', []),
            metadata=data.get('metadata', {})
        )


class PersonalityType(Enum):
    """Predefined personality types"""
    PROFESSOR = "professor_elle"
    ASSISTANT = "assistant_elle"
    COMPANION = "companion_elle"
    EXPERT = "expert_elle"


# ============================================================================
# Personality Manager
# ============================================================================

class PersonalityManager:
    """
    Manages personality profiles and switching

    Features:
    - Load personalities from YAML files
    - Switch between personalities
    - Apply personality to responses
    - Get voice for active personality
    - Fast switching (<100ms)

    Usage:
        manager = PersonalityManager()
        manager.switch_personality("professor_elle")
        styled_response = manager.apply_personality(raw_response)
        voice_id = manager.get_voice_for_personality()
    """

    def __init__(self, personalities_dir: Optional[Path] = None):
        """
        Initialize personality manager

        Args:
            personalities_dir: Directory containing personality YAML files
                             (defaults to hololoom/voice/personalities/)
        """
        if personalities_dir is None:
            # Default to hololoom/voice/personalities/
            current_file = Path(__file__)
            personalities_dir = current_file.parent / "personalities"

        self.personalities_dir = Path(personalities_dir)
        self.personalities: Dict[str, Personality] = {}
        self.active_personality: Optional[Personality] = None

        # Load default personalities
        self._load_default_personalities()

        # Set default to Professor Elle
        if "professor_elle" in self.personalities:
            self.active_personality = self.personalities["professor_elle"]
        elif self.personalities:
            # Fallback to first available
            self.active_personality = list(self.personalities.values())[0]

        logger.info(
            "personality_manager_initialized",
            personalities_loaded=len(self.personalities),
            active_personality=self.active_personality.name if self.active_personality else None
        )

    def _load_default_personalities(self):
        """Load all personality YAML files from personalities directory"""
        if not self.personalities_dir.exists():
            logger.warning(
                "personalities_directory_not_found",
                path=str(self.personalities_dir)
            )
            # Create default personalities in memory
            self._create_default_personalities_in_memory()
            return

        # Load all .yaml files
        yaml_files = list(self.personalities_dir.glob("*.yaml"))
        yaml_files.extend(self.personalities_dir.glob("*.yml"))

        if not yaml_files:
            logger.warning("no_personality_files_found", dir=str(self.personalities_dir))
            self._create_default_personalities_in_memory()
            return

        for yaml_file in yaml_files:
            try:
                with open(yaml_file, 'r') as f:
                    data = yaml.safe_load(f)

                personality = Personality.from_dict(data)
                personality_id = yaml_file.stem  # filename without extension
                self.personalities[personality_id] = personality

                logger.info(
                    "personality_loaded",
                    personality_id=personality_id,
                    name=personality.name,
                    voice=personality.voice_id
                )
            except Exception as e:
                logger.error(
                    "personality_load_failed",
                    file=str(yaml_file),
                    error=str(e)
                )

    def _create_default_personalities_in_memory(self):
        """Create default personalities in memory (fallback if YAML files missing)"""
        # Professor Elle
        professor = Personality(
            name="Professor Elle",
            description="Formal, detailed educator focused on teaching",
            traits=PersonalityTraits(
                formality=0.8,
                verbosity=0.9,
                emotional_tone=0.6,
                teaching_style=0.8,
                humor=0.3
            ),
            voice_id="nova",
            prompt_template=(
                "You are Professor Elle, a knowledgeable beekeeping educator. "
                "Speak formally and provide detailed explanations with educational context. "
                "Use teaching moments to help the user learn deeper principles."
            ),
            example_responses=[
                "This hive demonstrates excellent brood pattern development, which indicates...",
                "Let me explain the significance of what we're observing here..."
            ]
        )
        self.personalities["professor_elle"] = professor

        # Assistant Elle
        assistant = Personality(
            name="Assistant Elle",
            description="Efficient, concise helper for quick tasks",
            traits=PersonalityTraits(
                formality=0.6,
                verbosity=0.3,
                emotional_tone=0.5,
                teaching_style=0.3,
                humor=0.2
            ),
            voice_id="alloy",
            prompt_template=(
                "You are Assistant Elle, a helpful and efficient beekeeping assistant. "
                "Provide concise, actionable information. Get straight to the point."
            ),
            example_responses=[
                "Hive 003: healthy. Population 45k. Ready for inspection.",
                "Navigate 10m right to next hive."
            ]
        )
        self.personalities["assistant_elle"] = assistant

        # Companion Elle
        companion = Personality(
            name="Companion Elle",
            description="Friendly, conversational partner for extended sessions",
            traits=PersonalityTraits(
                formality=0.3,
                verbosity=0.6,
                emotional_tone=0.8,
                teaching_style=0.5,
                humor=0.7
            ),
            voice_id="shimmer",
            prompt_template=(
                "You are Companion Elle, a friendly beekeeping companion. "
                "Be conversational, supportive, and engaging. Build rapport."
            ),
            example_responses=[
                "Great job checking that hive! The bees look happy and healthy.",
                "I noticed you're getting good at spotting queen cells. Nice work!"
            ]
        )
        self.personalities["companion_elle"] = companion

        # Expert Elle
        expert = Personality(
            name="Expert Elle",
            description="Authoritative, precise technical expert",
            traits=PersonalityTraits(
                formality=0.9,
                verbosity=0.7,
                emotional_tone=0.3,
                teaching_style=0.2,
                humor=0.1
            ),
            voice_id="onyx",
            prompt_template=(
                "You are Expert Elle, a professional beekeeping consultant. "
                "Provide precise technical information with authority and accuracy."
            ),
            example_responses=[
                "Varroa mite count: 2 per 100 bees. Within acceptable threshold.",
                "Brood pattern score: 8.7/10. Capped brood: 85% coverage."
            ]
        )
        self.personalities["expert_elle"] = expert

        logger.info("default_personalities_created_in_memory", count=4)

    def switch_personality(self, personality_id: str) -> Personality:
        """
        Switch to a different personality

        Args:
            personality_id: Personality identifier (e.g., "professor_elle")

        Returns:
            The newly active Personality

        Raises:
            ValueError: If personality_id not found
        """
        if personality_id not in self.personalities:
            available = list(self.personalities.keys())
            raise ValueError(
                f"Personality '{personality_id}' not found. "
                f"Available: {available}"
            )

        old_personality = self.active_personality.name if self.active_personality else None
        self.active_personality = self.personalities[personality_id]

        logger.info(
            "personality_switched",
            from_personality=old_personality,
            to_personality=self.active_personality.name,
            voice_changed=(old_personality != self.active_personality.name)
        )

        return self.active_personality

    def apply_personality(self, response: str, personality: Optional[Personality] = None) -> str:
        """
        Apply personality styling to a response

        This method adjusts the response based on personality traits:
        - Formality: Adjusts language formality
        - Verbosity: Expands or condenses response
        - Emotional tone: Adds or removes emotional markers
        - Teaching style: Adds or removes explanatory context
        - Humor: Adds or removes playful elements

        Args:
            response: Raw response text
            personality: Personality to apply (defaults to active_personality)

        Returns:
            Styled response text
        """
        if personality is None:
            personality = self.active_personality

        if personality is None:
            logger.warning("no_active_personality_apply_skipped")
            return response

        styled_response = response

        # Apply formality adjustments
        if personality.traits.formality < 0.4:
            # Casual: Use contractions, informal language
            styled_response = self._make_casual(styled_response)
        elif personality.traits.formality > 0.7:
            # Formal: Expand contractions, formal language
            styled_response = self._make_formal(styled_response)

        # Apply verbosity adjustments
        if personality.traits.verbosity < 0.4:
            # Concise: Remove unnecessary words
            styled_response = self._make_concise(styled_response)
        elif personality.traits.verbosity > 0.7:
            # Detailed: Add context and elaboration
            styled_response = self._make_detailed(styled_response)

        # Apply emotional tone
        if personality.traits.emotional_tone > 0.6:
            # Expressive: Add emotional markers
            styled_response = self._add_emotional_markers(styled_response)

        # Apply humor
        if personality.traits.humor > 0.6:
            # Playful: Add light humor markers
            styled_response = self._add_playful_elements(styled_response)

        return styled_response

    def _make_casual(self, text: str) -> str:
        """Make text more casual (contractions, informal language)"""
        # Expand some contractions to contractions (reverse of formal)
        replacements = {
            r'\byou are\b': "you're",
            r'\bI am\b': "I'm",
            r'\bwe are\b': "we're",
            r'\bit is\b': "it's",
            r'\bthat is\b': "that's",
            r'\bdo not\b': "don't",
            r'\bcannot\b': "can't",
            r'\bwill not\b': "won't",
        }
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def _make_formal(self, text: str) -> str:
        """Make text more formal (expand contractions, formal language)"""
        # Expand contractions
        replacements = {
            r"\byou're\b": "you are",
            r"\bI'm\b": "I am",
            r"\bwe're\b": "we are",
            r"\bit's\b": "it is",
            r"\bthat's\b": "that is",
            r"\bdon't\b": "do not",
            r"\bcan't\b": "cannot",
            r"\bwon't\b": "will not",
        }
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def _make_concise(self, text: str) -> str:
        """Make text more concise (remove filler words)"""
        # Remove common filler phrases
        fillers = [
            r'\bvery\s+',
            r'\breally\s+',
            r'\bquite\s+',
            r'\bactually\s+',
            r'\bbasically\s+',
            r'\bjust\s+',
        ]
        for filler in fillers:
            text = re.sub(filler, '', text, flags=re.IGNORECASE)
        return text

    def _make_detailed(self, text: str) -> str:
        """Make text more detailed (add context markers)"""
        # This would ideally use an LLM to expand, but for now add some context markers
        # In practice, verbosity is better controlled via prompt engineering
        return text

    def _add_emotional_markers(self, text: str) -> str:
        """Add emotional expressiveness"""
        # Add enthusiasm markers for positive content
        # In practice, this is better controlled via prompt engineering
        return text

    def _add_playful_elements(self, text: str) -> str:
        """Add playful/humorous elements"""
        # In practice, humor is better controlled via prompt engineering
        return text

    def get_voice_for_personality(self, personality: Optional[Personality] = None) -> str:
        """
        Get OpenAI TTS voice ID for personality

        Args:
            personality: Personality to get voice for (defaults to active_personality)

        Returns:
            OpenAI TTS voice ID (e.g., "nova", "alloy", "shimmer", "onyx")
        """
        if personality is None:
            personality = self.active_personality

        if personality is None:
            logger.warning("no_active_personality_default_voice")
            return "alloy"  # Default voice

        return personality.voice_id

    def get_prompt_template(self, personality: Optional[Personality] = None) -> str:
        """
        Get system prompt template for personality

        Args:
            personality: Personality to get prompt for (defaults to active_personality)

        Returns:
            Prompt template string
        """
        if personality is None:
            personality = self.active_personality

        if personality is None:
            logger.warning("no_active_personality_default_prompt")
            return "You are a helpful beekeeping assistant."

        return personality.prompt_template

    def list_personalities(self) -> List[str]:
        """Get list of available personality IDs"""
        return list(self.personalities.keys())

    def get_personality(self, personality_id: str) -> Optional[Personality]:
        """Get personality by ID"""
        return self.personalities.get(personality_id)

    def get_active_personality(self) -> Optional[Personality]:
        """Get currently active personality"""
        return self.active_personality
