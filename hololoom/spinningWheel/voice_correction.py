"""
Voice Correction System - Natural Language Schema Improvements
==============================================================

Enables users to correct and improve schema transformations using
natural language voice commands.

Philosophy:
    "Wool becomes yarn through conversation, not configuration."

Features:
- Natural language intent parsing
- Field-level corrections
- Schema evolution (add fields dynamically)
- Pattern learning from corrections
- Automatic propagation to similar transformations

Usage:
    from hololoom.spinningWheel import VoiceCorrector

    corrector = VoiceCorrector()

    # Apply correction
    await corrector.apply_correction(
        transformation_id="tx_123",
        voice_command="the merchant is actually Whole Foods"
    )

    # System learns pattern and applies to future transformations!

Author: Claude Code
Date: January 2025
"""

from typing import Dict, List, Optional, Any, Protocol
from dataclasses import dataclass, field
from enum import Enum
import re
import time
from pathlib import Path
import json


# ============================================================================
# Intent Types
# ============================================================================

class IntentType(Enum):
    """Type of correction intent."""
    FIELD_CORRECTION = "field_correction"      # Fix specific field value
    FIELD_MAPPING = "field_mapping"            # Map source field to target
    SCHEMA_EVOLUTION = "schema_evolution"      # Add/modify schema fields
    CATEGORY_ADDITION = "category_addition"    # Add new category
    VALIDATION_OVERRIDE = "validation_override" # Override validation rule
    BATCH_CORRECTION = "batch_correction"      # Apply to multiple items
    UNKNOWN = "unknown"


@dataclass
class Intent:
    """Parsed intent from voice command."""
    type: IntentType
    confidence: float  # 0.0-1.0

    # Field correction
    field_name: Optional[str] = None
    field_value: Optional[Any] = None
    original_value: Optional[Any] = None

    # Field mapping
    source_field: Optional[str] = None
    target_field: Optional[str] = None

    # Schema evolution
    schema_name: Optional[str] = None
    field_definition: Optional[Dict] = None

    # Category
    category_name: Optional[str] = None

    # Metadata
    raw_command: str = ""
    parsed_tokens: List[str] = field(default_factory=list)


# ============================================================================
# Correction Pattern Learning
# ============================================================================

class PatternType(Enum):
    """Type of learned pattern."""
    FIELD_MAPPING = "field_mapping"           # "amt" → "total"
    VALUE_NORMALIZATION = "value_normalization"  # "WH FOODS" → "Whole Foods"
    SCHEMA_EXTENSION = "schema_extension"     # Add "tip" field
    EXTRACTION_RULE = "extraction_rule"       # "Total follows 'Total:'"


@dataclass
class CorrectionPattern:
    """Learned pattern from user corrections."""
    pattern_id: str
    pattern_type: PatternType

    # Pattern definition
    source_pattern: str  # What triggers this (regex or rule)
    target_action: str   # What to do
    context: Dict[str, Any] = field(default_factory=dict)

    # Learning metrics
    confidence: float = 0.5  # 0.0-1.0
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0

    # Metadata
    created_at: float = field(default_factory=time.time)
    last_used: Optional[float] = None
    learned_from: List[str] = field(default_factory=list)  # Correction IDs

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.success_count / total

    def update_confidence(self) -> None:
        """Update confidence based on success rate and usage."""
        # Confidence = success_rate * usage_weight
        usage_weight = min(1.0, self.usage_count / 10.0)  # Max at 10 uses
        self.confidence = self.success_rate * 0.7 + usage_weight * 0.3

    def record_success(self) -> None:
        """Record successful application."""
        self.success_count += 1
        self.usage_count += 1
        self.last_used = time.time()
        self.update_confidence()

    def record_failure(self) -> None:
        """Record failed application."""
        self.failure_count += 1
        self.last_used = time.time()
        self.update_confidence()


@dataclass
class Correction:
    """Single correction applied by user."""
    correction_id: str
    transformation_id: str
    timestamp: float

    # Intent
    intent: Intent
    voice_command: str

    # Before/after
    original_data: Dict[str, Any]
    corrected_data: Dict[str, Any]

    # Learning
    pattern_learned: Optional[CorrectionPattern] = None
    applied_automatically: bool = False  # Was this auto-applied from pattern?


# ============================================================================
# Intent Parser (LLM-Optional)
# ============================================================================

class IntentParser:
    """
    Parse natural language commands into structured intents.

    Uses rule-based parsing by default, can upgrade to LLM if available.
    """

    def __init__(self, use_llm: bool = False):
        """
        Initialize intent parser.

        Args:
            use_llm: Use LLM for parsing (requires API key)
        """
        self.use_llm = use_llm

        # Rule-based patterns (fallback)
        self.patterns = {
            # Field corrections
            r"(?:the\s+)?(\w+)\s+is\s+(?:actually\s+)?(.+)": IntentType.FIELD_CORRECTION,
            r"(?:change|fix|correct)\s+(\w+)\s+to\s+(.+)": IntentType.FIELD_CORRECTION,
            r"(\w+)\s+should\s+be\s+(.+)": IntentType.FIELD_CORRECTION,

            # Field mappings
            r"map\s+['\"]?(\w+)['\"]?\s+to\s+['\"]?(\w+)['\"]?": IntentType.FIELD_MAPPING,
            r"['\"]?(\w+)['\"]?\s+means\s+['\"]?(\w+)['\"]?": IntentType.FIELD_MAPPING,
            r"treat\s+['\"]?(\w+)['\"]?\s+as\s+['\"]?(\w+)['\"]?": IntentType.FIELD_MAPPING,

            # Schema evolution
            r"add\s+(?:a\s+)?['\"]?(\w+)['\"]?\s+field": IntentType.SCHEMA_EVOLUTION,
            r"create\s+(?:a\s+)?['\"]?(\w+)['\"]?\s+field": IntentType.SCHEMA_EVOLUTION,

            # Categories
            r"(?:add\s+)?category\s+['\"]?(\w+)['\"]?": IntentType.CATEGORY_ADDITION,
            r"this\s+is\s+category\s+['\"]?(\w+)['\"]?": IntentType.CATEGORY_ADDITION,
        }

    async def parse(self, voice_command: str) -> Intent:
        """
        Parse voice command into intent.

        Args:
            voice_command: Natural language command

        Returns:
            Parsed Intent
        """
        if self.use_llm:
            return await self._parse_with_llm(voice_command)
        else:
            return self._parse_with_rules(voice_command)

    def _parse_with_rules(self, voice_command: str) -> Intent:
        """Rule-based parsing (no LLM required)."""
        command_lower = voice_command.lower().strip()

        # Try each pattern
        for pattern, intent_type in self.patterns.items():
            match = re.search(pattern, command_lower)
            if match:
                return self._build_intent_from_match(
                    intent_type,
                    match,
                    voice_command
                )

        # Unknown intent
        return Intent(
            type=IntentType.UNKNOWN,
            confidence=0.0,
            raw_command=voice_command
        )

    def _build_intent_from_match(
        self,
        intent_type: IntentType,
        match: re.Match,
        voice_command: str
    ) -> Intent:
        """Build Intent from regex match."""
        groups = match.groups()

        if intent_type == IntentType.FIELD_CORRECTION:
            # "merchant is Whole Foods"
            return Intent(
                type=IntentType.FIELD_CORRECTION,
                confidence=0.9,
                field_name=groups[0],
                field_value=groups[1],
                raw_command=voice_command
            )

        elif intent_type == IntentType.FIELD_MAPPING:
            # "map 'amt' to 'total'"
            return Intent(
                type=IntentType.FIELD_MAPPING,
                confidence=0.9,
                source_field=groups[0],
                target_field=groups[1],
                raw_command=voice_command
            )

        elif intent_type == IntentType.SCHEMA_EVOLUTION:
            # "add tip field"
            return Intent(
                type=IntentType.SCHEMA_EVOLUTION,
                confidence=0.8,
                field_name=groups[0],
                raw_command=voice_command
            )

        elif intent_type == IntentType.CATEGORY_ADDITION:
            # "category medical"
            return Intent(
                type=IntentType.CATEGORY_ADDITION,
                confidence=0.9,
                category_name=groups[0],
                raw_command=voice_command
            )

        return Intent(
            type=IntentType.UNKNOWN,
            confidence=0.0,
            raw_command=voice_command
        )

    async def _parse_with_llm(self, voice_command: str) -> Intent:
        """LLM-based parsing (higher accuracy)."""
        # TODO: Implement LLM-based parsing
        # For now, fallback to rules
        return self._parse_with_rules(voice_command)


# ============================================================================
# Self-Tuning Engine
# ============================================================================

class SelfTuningEngine:
    """
    Learn patterns from user corrections and apply automatically.

    Core mechanism:
    1. User corrects field: "merchant is Whole Foods"
    2. System extracts pattern: "WH FOODS" → "Whole Foods Market"
    3. Future receipts with "WH FOODS" automatically corrected
    4. Confidence increases with successful applications
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        min_confidence: float = 0.7
    ):
        """
        Initialize self-tuning engine.

        Args:
            storage_path: Path to persist learned patterns
            min_confidence: Minimum confidence to auto-apply pattern
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.min_confidence = min_confidence

        # Learned patterns
        self.patterns: Dict[str, CorrectionPattern] = {}

        # Correction history
        self.corrections: Dict[str, Correction] = {}

        # Load patterns if storage exists
        if self.storage_path and self.storage_path.exists():
            self._load_patterns()

        # Track last applied patterns (for metadata)
        self.last_patterns_applied: List[CorrectionPattern] = []

    async def learn_from_correction(
        self,
        original_data: Dict[str, Any],
        corrected_data: Dict[str, Any],
        context: Dict[str, Any],
        voice_command: str
    ) -> Optional[CorrectionPattern]:
        """
        Learn pattern from user correction.

        Args:
            original_data: Data before correction
            corrected_data: Data after correction
            context: Context (schema, transformation ID, etc.)
            voice_command: Original voice command

        Returns:
            Learned CorrectionPattern (if pattern extracted)
        """
        # Find what changed
        changes = self._find_changes(original_data, corrected_data)

        if not changes:
            return None

        # Extract pattern based on change type
        learned_pattern = None

        for field_name, (old_value, new_value) in changes.items():
            # String value normalization
            if isinstance(old_value, str) and isinstance(new_value, str):
                # Check if it's a normalization (expansion/correction)
                # e.g., "WH FOODS" → "Whole Foods Market"
                # or any string correction where one contains the other or they're similar

                pattern = self._create_value_normalization_pattern(
                    field_name,
                    old_value,
                    new_value,
                    context
                )

                self._add_or_update_pattern(pattern)
                learned_pattern = pattern

            # Numeric value correction
            elif isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
                # Could be format error: 4599 → 45.99
                # Learn this as a normalization too
                pattern = self._create_value_normalization_pattern(
                    field_name,
                    str(old_value),
                    str(new_value),
                    context
                )

                self._add_or_update_pattern(pattern)
                learned_pattern = pattern

            # None → value (field addition)
            elif old_value is None and new_value is not None:
                # User added a field that was missing
                # This is a schema evolution signal
                pass

        return learned_pattern

    def _find_changes(
        self,
        original: Dict[str, Any],
        corrected: Dict[str, Any]
    ) -> Dict[str, tuple]:
        """Find what changed between original and corrected data."""
        changes = {}

        for key in set(original.keys()) | set(corrected.keys()):
            orig_val = original.get(key)
            corr_val = corrected.get(key)

            if orig_val != corr_val:
                changes[key] = (orig_val, corr_val)

        return changes

    def _create_value_normalization_pattern(
        self,
        field_name: str,
        old_value: str,
        new_value: str,
        context: Dict[str, Any]
    ) -> CorrectionPattern:
        """Create value normalization pattern."""
        pattern_id = f"norm_{field_name}_{hash(old_value)}"

        return CorrectionPattern(
            pattern_id=pattern_id,
            pattern_type=PatternType.VALUE_NORMALIZATION,
            source_pattern=old_value,
            target_action=new_value,
            context={
                'field_name': field_name,
                'schema': context.get('schema'),
                'match_type': 'exact'  # or 'fuzzy', 'regex'
            },
            confidence=0.6  # Initial confidence
        )

    def _add_or_update_pattern(self, pattern: CorrectionPattern) -> None:
        """Add new pattern or update existing."""
        if pattern.pattern_id in self.patterns:
            # Update existing pattern
            existing = self.patterns[pattern.pattern_id]
            existing.usage_count += 1
            existing.update_confidence()
        else:
            # Add new pattern
            self.patterns[pattern.pattern_id] = pattern

        # Persist
        if self.storage_path:
            self._save_patterns()

    async def apply_learned_patterns(
        self,
        data: Dict[str, Any],
        schema_name: str
    ) -> Dict[str, Any]:
        """
        Apply learned patterns to data.

        Args:
            data: Data to improve
            schema_name: Schema context

        Returns:
            Improved data
        """
        improved = data.copy()
        self.last_patterns_applied = []

        # Apply value normalization patterns
        for pattern in self.patterns.values():
            if pattern.confidence < self.min_confidence:
                continue

            if pattern.pattern_type == PatternType.VALUE_NORMALIZATION:
                improved = self._apply_normalization_pattern(
                    improved,
                    pattern,
                    schema_name
                )

                if improved != data:
                    self.last_patterns_applied.append(pattern)

        return improved

    def _apply_normalization_pattern(
        self,
        data: Dict[str, Any],
        pattern: CorrectionPattern,
        schema_name: str
    ) -> Dict[str, Any]:
        """Apply value normalization pattern."""
        # Check context matches
        if pattern.context.get('schema') and pattern.context['schema'] != schema_name:
            return data

        field_name = pattern.context.get('field_name')
        if not field_name or field_name not in data:
            return data

        # Check value matches
        current_value = data.get(field_name)
        source_pattern = pattern.source_pattern

        if current_value == source_pattern:
            # Exact match - apply correction
            data[field_name] = pattern.target_action
            pattern.record_success()

        return data

    def get_patterns_for_field(
        self,
        field_name: str,
        min_confidence: Optional[float] = None
    ) -> List[CorrectionPattern]:
        """Get learned patterns for specific field."""
        min_conf = min_confidence or self.min_confidence

        return [
            p for p in self.patterns.values()
            if p.context.get('field_name') == field_name
            and p.confidence >= min_conf
        ]

    def _save_patterns(self) -> None:
        """Persist patterns to disk."""
        if not self.storage_path:
            return

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        patterns_data = {
            pattern_id: {
                'pattern_type': pattern.pattern_type.value,
                'source_pattern': pattern.source_pattern,
                'target_action': pattern.target_action,
                'context': pattern.context,
                'confidence': pattern.confidence,
                'usage_count': pattern.usage_count,
                'success_count': pattern.success_count,
                'failure_count': pattern.failure_count,
                'created_at': pattern.created_at,
                'last_used': pattern.last_used
            }
            for pattern_id, pattern in self.patterns.items()
        }

        with open(self.storage_path, 'w') as f:
            json.dump(patterns_data, f, indent=2)

    def _load_patterns(self) -> None:
        """Load patterns from disk."""
        if not self.storage_path or not self.storage_path.exists():
            return

        with open(self.storage_path, 'r') as f:
            patterns_data = json.load(f)

        for pattern_id, data in patterns_data.items():
            pattern = CorrectionPattern(
                pattern_id=pattern_id,
                pattern_type=PatternType(data['pattern_type']),
                source_pattern=data['source_pattern'],
                target_action=data['target_action'],
                context=data['context'],
                confidence=data['confidence'],
                usage_count=data['usage_count'],
                success_count=data['success_count'],
                failure_count=data['failure_count'],
                created_at=data['created_at'],
                last_used=data.get('last_used')
            )

            self.patterns[pattern_id] = pattern


# ============================================================================
# Voice Corrector (Main Interface)
# ============================================================================

class VoiceCorrector:
    """
    Main interface for voice-based corrections.

    Combines intent parsing + self-tuning + correction application.
    """

    def __init__(
        self,
        tuning_engine: Optional[SelfTuningEngine] = None,
        intent_parser: Optional[IntentParser] = None
    ):
        """
        Initialize voice corrector.

        Args:
            tuning_engine: Self-tuning engine (creates default if None)
            intent_parser: Intent parser (creates default if None)
        """
        self.tuning_engine = tuning_engine or SelfTuningEngine()
        self.intent_parser = intent_parser or IntentParser()

        # Track corrections
        self.corrections: Dict[str, Correction] = {}

    async def apply_correction(
        self,
        transformation_id: str,
        voice_command: str,
        original_data: Dict[str, Any],
        schema_name: str
    ) -> Correction:
        """
        Apply correction from voice command.

        Args:
            transformation_id: ID of transformation to correct
            voice_command: Natural language command
            original_data: Original extracted data
            schema_name: Schema context

        Returns:
            Correction object
        """
        # Parse intent
        intent = await self.intent_parser.parse(voice_command)

        # Apply correction based on intent
        corrected_data = original_data.copy()

        if intent.type == IntentType.FIELD_CORRECTION:
            # Fix field value
            corrected_data[intent.field_name] = intent.field_value

        elif intent.type == IntentType.FIELD_MAPPING:
            # Map source field to target
            if intent.source_field in corrected_data:
                corrected_data[intent.target_field] = corrected_data[intent.source_field]
                del corrected_data[intent.source_field]

        # Learn pattern
        pattern = await self.tuning_engine.learn_from_correction(
            original_data,
            corrected_data,
            context={'schema': schema_name},
            voice_command=voice_command
        )

        # Create correction record
        correction = Correction(
            correction_id=f"corr_{int(time.time() * 1000)}",
            transformation_id=transformation_id,
            timestamp=time.time(),
            intent=intent,
            voice_command=voice_command,
            original_data=original_data,
            corrected_data=corrected_data,
            pattern_learned=pattern
        )

        self.corrections[correction.correction_id] = correction

        return correction

    def get_suggestions(
        self,
        data: Dict[str, Any],
        schema_name: str
    ) -> List[str]:
        """
        Get voice-friendly correction suggestions.

        Args:
            data: Current data
            schema_name: Schema context

        Returns:
            List of suggested voice commands
        """
        suggestions = []

        # Check for common patterns
        for field_name, value in data.items():
            patterns = self.tuning_engine.get_patterns_for_field(field_name)

            for pattern in patterns:
                if pattern.source_pattern in str(value):
                    suggestions.append(
                        f"Did you mean '{pattern.target_action}' instead of '{value}'?"
                    )

        return suggestions
