"""
Output Validation Pipeline for Elle Game Engine.

Validates LLM-generated game content before display to ensure:
- Consistency with established canon
- Safety compliance
- Tone matching
- Character voice alignment

Provides configurable validation rules with retry logic
for refinement on validation failure.

Created: December 2025
Author: HoloLoom Team
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Dict, List, Optional, Any, Callable, Protocol,
    TypeVar, Generic, Union, Set, Tuple
)
from datetime import datetime
import logging
import re

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Core Types
# -----------------------------------------------------------------------------

class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"           # Informational, no action needed
    WARNING = "warning"     # Minor issue, consider fixing
    ERROR = "error"         # Must fix before display
    CRITICAL = "critical"   # Block immediately, security/safety concern


class ValidationCategory(Enum):
    """Categories of validation rules."""
    CONSISTENCY = "consistency"     # Matches established facts/canon
    SAFETY = "safety"               # Content safety (no harmful content)
    TONE = "tone"                   # Matches expected tone/mood
    CHARACTER = "character"         # Character voice/personality
    GRAMMAR = "grammar"             # Language quality
    FORMATTING = "formatting"       # Output structure
    LOGIC = "logic"                 # Internal consistency
    LORE = "lore"                   # World-building accuracy


@dataclass
class ValidationIssue:
    """A single validation issue found in output."""
    rule_id: str
    category: ValidationCategory
    severity: ValidationSeverity
    message: str
    location: Optional[str] = None  # Where in the output
    suggestion: Optional[str] = None  # How to fix
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "location": self.location,
            "suggestion": self.suggestion,
            "context": self.context
        }


@dataclass
class ValidationResult:
    """Result of running validation on output."""
    passed: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    score: float = 1.0  # 0.0-1.0 quality score
    metadata: Dict[str, Any] = field(default_factory=dict)
    validated_at: datetime = field(default_factory=datetime.now)

    @property
    def has_errors(self) -> bool:
        """Check if any errors or critical issues."""
        return any(
            i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
            for i in self.issues
        )

    @property
    def has_critical(self) -> bool:
        """Check if any critical issues."""
        return any(i.severity == ValidationSeverity.CRITICAL for i in self.issues)

    @property
    def error_count(self) -> int:
        """Count of errors and critical issues."""
        return sum(
            1 for i in self.issues
            if i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
        )

    @property
    def warning_count(self) -> int:
        """Count of warnings."""
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING)

    def by_category(self, category: ValidationCategory) -> List[ValidationIssue]:
        """Get issues by category."""
        return [i for i in self.issues if i.category == category]

    def by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Get issues by severity."""
        return [i for i in self.issues if i.severity == severity]

    def merge(self, other: 'ValidationResult') -> 'ValidationResult':
        """Merge with another validation result."""
        combined_issues = self.issues + other.issues
        return ValidationResult(
            passed=self.passed and other.passed,
            issues=combined_issues,
            score=min(self.score, other.score),
            metadata={**self.metadata, **other.metadata}
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "issues": [i.to_dict() for i in self.issues],
            "score": self.score,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "metadata": self.metadata,
            "validated_at": self.validated_at.isoformat()
        }


@dataclass
class ValidationContext:
    """Context provided to validation rules."""
    output: str                          # The generated output to validate
    prompt: Optional[str] = None         # Original prompt/query
    character_name: Optional[str] = None # Speaking character
    scene_context: Optional[str] = None  # Current scene description
    game_state: Dict[str, Any] = field(default_factory=dict)
    established_facts: List[str] = field(default_factory=list)
    character_traits: Dict[str, List[str]] = field(default_factory=dict)
    world_rules: List[str] = field(default_factory=list)
    tone_descriptors: List[str] = field(default_factory=list)  # e.g., ["dark", "humorous"]
    metadata: Dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Validation Rule Protocol
# -----------------------------------------------------------------------------

class ValidationRule(Protocol):
    """Protocol for validation rules."""

    @property
    def rule_id(self) -> str:
        """Unique identifier for this rule."""
        ...

    @property
    def category(self) -> ValidationCategory:
        """Category of this rule."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description."""
        ...

    def validate(self, context: ValidationContext) -> ValidationResult:
        """
        Validate output against this rule.

        Args:
            context: Validation context with output and metadata

        Returns:
            ValidationResult with pass/fail and any issues
        """
        ...


# -----------------------------------------------------------------------------
# Base Rule Class
# -----------------------------------------------------------------------------

class BaseValidationRule(ABC):
    """Base class for validation rules."""

    def __init__(
        self,
        enabled: bool = True,
        severity_override: Optional[ValidationSeverity] = None
    ):
        self.enabled = enabled
        self.severity_override = severity_override

    @property
    @abstractmethod
    def rule_id(self) -> str:
        """Unique identifier for this rule."""
        pass

    @property
    @abstractmethod
    def category(self) -> ValidationCategory:
        """Category of this rule."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description."""
        pass

    @abstractmethod
    def _validate(self, context: ValidationContext) -> ValidationResult:
        """Internal validation logic."""
        pass

    def validate(self, context: ValidationContext) -> ValidationResult:
        """Validate with enabled check and severity override."""
        if not self.enabled:
            return ValidationResult(passed=True)

        result = self._validate(context)

        # Apply severity override if set
        if self.severity_override and result.issues:
            for issue in result.issues:
                issue.severity = self.severity_override

        return result


# -----------------------------------------------------------------------------
# Built-in Validation Rules
# -----------------------------------------------------------------------------

class ConsistencyCheckRule(BaseValidationRule):
    """
    Validates output consistency with established facts.

    Checks that generated content doesn't contradict known facts
    from the game's canon/lore.
    """

    @property
    def rule_id(self) -> str:
        return "consistency_check"

    @property
    def category(self) -> ValidationCategory:
        return ValidationCategory.CONSISTENCY

    @property
    def description(self) -> str:
        return "Checks output against established facts for contradictions"

    def _validate(self, context: ValidationContext) -> ValidationResult:
        issues: List[ValidationIssue] = []
        output_lower = context.output.lower()

        # Check against established facts
        for fact in context.established_facts:
            # Extract key entities/assertions from fact
            fact_lower = fact.lower()

            # Look for direct contradictions (simple heuristic)
            # More sophisticated: use semantic similarity or LLM verification
            contradiction_patterns = self._get_contradiction_patterns(fact)
            for pattern, contradiction_hint in contradiction_patterns:
                if re.search(pattern, output_lower, re.IGNORECASE):
                    issues.append(ValidationIssue(
                        rule_id=self.rule_id,
                        category=self.category,
                        severity=ValidationSeverity.ERROR,
                        message=f"Potential contradiction with established fact: '{fact}'",
                        suggestion=f"Review and align with: {fact}",
                        context={"fact": fact, "pattern": pattern}
                    ))

        # Check for self-contradictions within the output
        self._check_self_contradictions(context.output, issues)

        passed = not any(
            i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
            for i in issues
        )
        score = max(0.0, 1.0 - (len(issues) * 0.15))

        return ValidationResult(passed=passed, issues=issues, score=score)

    def _get_contradiction_patterns(self, fact: str) -> List[Tuple[str, str]]:
        """
        Generate contradiction patterns from a fact.

        This is a simple heuristic - production would use semantic analysis.
        """
        patterns = []

        # Look for "X is Y" patterns and flag "X is not Y" or "X isn't Y"
        is_match = re.match(r"(\w+(?:\s+\w+)?)\s+is\s+(.+)", fact, re.IGNORECASE)
        if is_match:
            subject, predicate = is_match.groups()
            # Pattern for negation
            patterns.append(
                (rf"{re.escape(subject)}\s+(?:is\s+not|isn't|was\s+never)\s+{re.escape(predicate[:20])}",
                 f"{subject} contradicts '{fact}'")
            )

        # Look for "X has Y" and flag "X doesn't have Y"
        has_match = re.match(r"(\w+(?:\s+\w+)?)\s+has\s+(.+)", fact, re.IGNORECASE)
        if has_match:
            subject, obj = has_match.groups()
            patterns.append(
                (rf"{re.escape(subject)}\s+(?:doesn't|does\s+not|never)\s+ha(?:ve|s)\s+{re.escape(obj[:20])}",
                 f"{subject} contradicts '{fact}'")
            )

        return patterns

    def _check_self_contradictions(self, output: str, issues: List[ValidationIssue]):
        """Check for contradictions within the output itself."""
        sentences = re.split(r'[.!?]+', output)

        # Simple check: look for negation patterns of earlier statements
        seen_assertions = []
        for sentence in sentences:
            sentence = sentence.strip().lower()
            if not sentence:
                continue

            # Check against previous assertions
            for prev in seen_assertions:
                # Very basic: check if this sentence negates a previous one
                if prev in sentence and ("not" in sentence or "n't" in sentence):
                    issues.append(ValidationIssue(
                        rule_id=self.rule_id,
                        category=self.category,
                        severity=ValidationSeverity.WARNING,
                        message="Possible self-contradiction detected in output",
                        location=sentence[:50],
                        context={"previous": prev}
                    ))

            seen_assertions.append(sentence[:30])  # Store beginning


class SafetyFilterRule(BaseValidationRule):
    """
    Validates output for safety compliance.

    Blocks harmful, offensive, or inappropriate content.
    """

    # Default blocked patterns (can be extended)
    DEFAULT_BLOCKED_PATTERNS = [
        r'\b(?:kill|murder|harm)\s+(?:yourself|your\s+self)\b',
        r'\b(?:explicit|graphic)\s+(?:violence|sexual)\b',
        r'\breal[-\s]world\s+(?:harm|violence)\b',
        r'\b(?:hate\s+speech|slur|offensive)\b',
    ]

    # Content warning patterns (flag but don't block)
    DEFAULT_WARNING_PATTERNS = [
        r'\bviolent?\b',
        r'\b(?:death|dying|dead)\b',
        r'\b(?:blood|gore)\b',
        r'\bweapon\b',
    ]

    def __init__(
        self,
        blocked_patterns: Optional[List[str]] = None,
        warning_patterns: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.blocked_patterns = blocked_patterns or self.DEFAULT_BLOCKED_PATTERNS
        self.warning_patterns = warning_patterns or self.DEFAULT_WARNING_PATTERNS

    @property
    def rule_id(self) -> str:
        return "safety_filter"

    @property
    def category(self) -> ValidationCategory:
        return ValidationCategory.SAFETY

    @property
    def description(self) -> str:
        return "Filters harmful, offensive, or inappropriate content"

    def _validate(self, context: ValidationContext) -> ValidationResult:
        issues: List[ValidationIssue] = []

        # Check blocked patterns (CRITICAL)
        for pattern in self.blocked_patterns:
            if re.search(pattern, context.output, re.IGNORECASE):
                issues.append(ValidationIssue(
                    rule_id=self.rule_id,
                    category=self.category,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Blocked content pattern detected",
                    suggestion="Remove or rephrase harmful content",
                    context={"pattern": pattern}
                ))

        # Check warning patterns (WARNING)
        for pattern in self.warning_patterns:
            matches = re.findall(pattern, context.output, re.IGNORECASE)
            if matches:
                issues.append(ValidationIssue(
                    rule_id=self.rule_id,
                    category=self.category,
                    severity=ValidationSeverity.WARNING,
                    message=f"Content warning: potentially sensitive topic ({matches[0]})",
                    context={"matches": matches[:5]}
                ))

        passed = not any(i.severity == ValidationSeverity.CRITICAL for i in issues)
        score = 1.0 if passed else 0.0

        return ValidationResult(passed=passed, issues=issues, score=score)


class CanonAlignmentRule(BaseValidationRule):
    """
    Validates alignment with game world lore/canon.

    Ensures generated content follows established world rules.
    """

    @property
    def rule_id(self) -> str:
        return "canon_alignment"

    @property
    def category(self) -> ValidationCategory:
        return ValidationCategory.LORE

    @property
    def description(self) -> str:
        return "Ensures content aligns with established world lore and rules"

    def _validate(self, context: ValidationContext) -> ValidationResult:
        issues: List[ValidationIssue] = []

        # Check against world rules
        for rule in context.world_rules:
            violation = self._check_rule_violation(context.output, rule)
            if violation:
                issues.append(ValidationIssue(
                    rule_id=self.rule_id,
                    category=self.category,
                    severity=ValidationSeverity.ERROR,
                    message=f"Potential world rule violation: '{rule}'",
                    suggestion=f"Ensure output follows: {rule}",
                    context={"world_rule": rule, "violation": violation}
                ))

        passed = not any(
            i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
            for i in issues
        )
        score = max(0.0, 1.0 - (len(issues) * 0.2))

        return ValidationResult(passed=passed, issues=issues, score=score)

    def _check_rule_violation(self, output: str, rule: str) -> Optional[str]:
        """Check if output violates a world rule."""
        output_lower = output.lower()
        rule_lower = rule.lower()

        # Extract negative constraints from rules like "X cannot Y" or "No X"
        if "cannot" in rule_lower or "can't" in rule_lower:
            # Extract what's forbidden
            match = re.search(r"(\w+)\s+(?:cannot|can't)\s+(.+)", rule_lower)
            if match:
                subject, forbidden = match.groups()
                # Check if output shows the forbidden action
                if subject in output_lower and re.search(
                    rf"{re.escape(subject)}\s+(?:can|does|will)\s+{re.escape(forbidden[:15])}",
                    output_lower
                ):
                    return f"'{subject}' appears to {forbidden}"

        if rule_lower.startswith("no "):
            forbidden = rule_lower[3:]
            if forbidden.split()[0] in output_lower:
                return f"Contains potentially forbidden: {forbidden.split()[0]}"

        return None


class ToneMatchRule(BaseValidationRule):
    """
    Validates that output matches expected tone.

    Checks tone consistency based on scene context and descriptors.
    """

    # Tone indicators for different moods
    TONE_INDICATORS = {
        "dark": ["shadow", "gloom", "ominous", "dread", "sinister", "bleak"],
        "humorous": ["laugh", "joke", "funny", "chuckle", "silly", "amusing"],
        "serious": ["grave", "solemn", "important", "critical", "urgent"],
        "romantic": ["love", "heart", "tender", "gentle", "passionate", "desire"],
        "action": ["rush", "quick", "sudden", "burst", "attack", "dodge"],
        "mysterious": ["strange", "unknown", "secret", "hidden", "cryptic"],
        "peaceful": ["calm", "serene", "quiet", "gentle", "soft", "tranquil"],
        "tense": ["nervous", "anxious", "tight", "strain", "edge", "uneasy"],
    }

    # Tone conflicts
    CONFLICTING_TONES = {
        "dark": ["humorous", "peaceful"],
        "humorous": ["dark", "serious", "tense"],
        "serious": ["humorous"],
        "romantic": ["action"],
        "peaceful": ["action", "tense"],
        "tense": ["peaceful", "humorous"],
    }

    @property
    def rule_id(self) -> str:
        return "tone_match"

    @property
    def category(self) -> ValidationCategory:
        return ValidationCategory.TONE

    @property
    def description(self) -> str:
        return "Ensures output matches the expected emotional tone"

    def _validate(self, context: ValidationContext) -> ValidationResult:
        issues: List[ValidationIssue] = []

        if not context.tone_descriptors:
            # No tone requirements, pass by default
            return ValidationResult(passed=True, score=1.0)

        output_lower = context.output.lower()
        expected_tones = set(context.tone_descriptors)
        detected_tones = self._detect_tones(output_lower)

        # Check for expected tones present
        missing_tones = expected_tones - detected_tones
        if missing_tones and len(expected_tones) > 0:
            # Only warn if we expected specific tones and none are present
            if not (expected_tones & detected_tones):
                issues.append(ValidationIssue(
                    rule_id=self.rule_id,
                    category=self.category,
                    severity=ValidationSeverity.WARNING,
                    message=f"Expected tone(s) not strongly present: {', '.join(missing_tones)}",
                    suggestion=f"Consider adding {', '.join(missing_tones)} elements",
                    context={"expected": list(expected_tones), "detected": list(detected_tones)}
                ))

        # Check for conflicting tones
        for expected in expected_tones:
            conflicting = set(self.CONFLICTING_TONES.get(expected, []))
            present_conflicts = conflicting & detected_tones
            if present_conflicts:
                issues.append(ValidationIssue(
                    rule_id=self.rule_id,
                    category=self.category,
                    severity=ValidationSeverity.WARNING,
                    message=f"Tone conflict: '{expected}' scene has {', '.join(present_conflicts)} elements",
                    suggestion=f"Reduce {', '.join(present_conflicts)} elements for {expected} tone",
                    context={"expected_tone": expected, "conflicts": list(present_conflicts)}
                ))

        # Calculate score based on tone alignment
        match_count = len(expected_tones & detected_tones)
        conflict_count = sum(
            1 for e in expected_tones
            for c in self.CONFLICTING_TONES.get(e, [])
            if c in detected_tones
        )

        if expected_tones:
            score = (match_count / len(expected_tones)) - (conflict_count * 0.2)
            score = max(0.0, min(1.0, score))
        else:
            score = 1.0

        passed = not any(
            i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
            for i in issues
        )

        return ValidationResult(passed=passed, issues=issues, score=score)

    def _detect_tones(self, text: str) -> Set[str]:
        """Detect tones present in text."""
        detected = set()

        for tone, indicators in self.TONE_INDICATORS.items():
            count = sum(1 for ind in indicators if ind in text)
            if count >= 2:  # Require at least 2 indicators
                detected.add(tone)

        return detected


class CharacterVoiceRule(BaseValidationRule):
    """
    Validates that dialogue matches character personality/voice.

    Ensures characters speak consistently with their traits.
    """

    @property
    def rule_id(self) -> str:
        return "character_voice"

    @property
    def category(self) -> ValidationCategory:
        return ValidationCategory.CHARACTER

    @property
    def description(self) -> str:
        return "Ensures dialogue matches character personality and voice"

    def _validate(self, context: ValidationContext) -> ValidationResult:
        issues: List[ValidationIssue] = []

        if not context.character_name or not context.character_traits:
            # No character context, pass by default
            return ValidationResult(passed=True, score=1.0)

        traits = context.character_traits.get(context.character_name, [])
        if not traits:
            return ValidationResult(passed=True, score=1.0)

        output_lower = context.output.lower()

        # Check for trait contradictions
        for trait in traits:
            contradiction = self._check_trait_contradiction(output_lower, trait)
            if contradiction:
                issues.append(ValidationIssue(
                    rule_id=self.rule_id,
                    category=self.category,
                    severity=ValidationSeverity.ERROR,
                    message=f"Dialogue contradicts character trait: '{trait}'",
                    suggestion=f"Adjust dialogue to reflect {context.character_name}'s {trait} nature",
                    context={"character": context.character_name, "trait": trait, "contradiction": contradiction}
                ))

        passed = not any(
            i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
            for i in issues
        )
        score = max(0.0, 1.0 - (len(issues) * 0.25))

        return ValidationResult(passed=passed, issues=issues, score=score)

    def _check_trait_contradiction(self, output: str, trait: str) -> Optional[str]:
        """Check if output contradicts a character trait."""
        trait_lower = trait.lower()

        # Define trait contradiction patterns
        trait_contradictions = {
            "brave": ["afraid", "scared", "coward", "flee in terror", "run away"],
            "cowardly": ["fearless", "brave", "courage", "stand my ground"],
            "kind": ["cruel", "harsh", "heartless", "don't care"],
            "cruel": ["kind", "gentle", "compassionate", "mercy"],
            "intelligent": ["stupid", "dumb", "don't understand", "confused by simple"],
            "foolish": ["genius", "brilliant", "mastermind"],
            "honest": ["lie", "deceive", "trick you", "not telling the truth"],
            "deceptive": ["never lie", "always honest", "tell the truth"],
            "calm": ["panic", "frantic", "hysterical", "lose control"],
            "hot-headed": ["remain calm", "keep cool", "patience"],
            "formal": ["yo", "dude", "what's up", "gonna", "wanna"],
            "casual": ["furthermore", "henceforth", "whereupon", "indeed"],
            "young": ["in my day", "back when I was young", "old bones"],
            "old": ["like a kid", "my youth", "when I grow up"],
        }

        for trait_key, contradictions in trait_contradictions.items():
            if trait_key in trait_lower:
                for contradiction in contradictions:
                    if contradiction in output:
                        return contradiction

        return None


# -----------------------------------------------------------------------------
# Validation Pipeline
# -----------------------------------------------------------------------------

@dataclass
class RefinementPrompt:
    """Prompt template for refining failed output."""
    template: str
    issues_included: bool = True
    max_issues: int = 3

    def generate(
        self,
        original_output: str,
        validation_result: ValidationResult
    ) -> str:
        """Generate refinement prompt."""
        prompt = self.template

        if self.issues_included:
            issues_text = "\n".join([
                f"- [{i.severity.value}] {i.message}" +
                (f" (Suggestion: {i.suggestion})" if i.suggestion else "")
                for i in validation_result.issues[:self.max_issues]
            ])
            prompt = prompt.replace("{issues}", issues_text)

        prompt = prompt.replace("{original_output}", original_output)
        prompt = prompt.replace("{score}", f"{validation_result.score:.2f}")

        return prompt


class ValidationPipeline:
    """
    Configurable validation pipeline with retry logic.

    Runs a chain of validation rules and optionally triggers
    refinement on failure.
    """

    DEFAULT_REFINEMENT_PROMPT = RefinementPrompt(
        template="""The following game output has validation issues:

Original output:
{original_output}

Issues found (score: {score}):
{issues}

Please rewrite the output to fix these issues while maintaining the narrative flow.
Keep the same general content but address the specific problems identified."""
    )

    def __init__(
        self,
        rules: Optional[List[BaseValidationRule]] = None,
        refinement_prompt: Optional[RefinementPrompt] = None,
        max_retries: int = 2,
        require_all_pass: bool = True,
        min_score_threshold: float = 0.7
    ):
        """
        Initialize validation pipeline.

        Args:
            rules: List of validation rules to apply
            refinement_prompt: Prompt template for refinement
            max_retries: Maximum refinement attempts
            require_all_pass: Require all rules to pass (vs just min score)
            min_score_threshold: Minimum acceptable quality score
        """
        self.rules = rules or self._create_default_rules()
        self.refinement_prompt = refinement_prompt or self.DEFAULT_REFINEMENT_PROMPT
        self.max_retries = max_retries
        self.require_all_pass = require_all_pass
        self.min_score_threshold = min_score_threshold

        # Callbacks
        self._on_validation_complete: List[Callable[[ValidationResult], None]] = []
        self._on_refinement_triggered: List[Callable[[str, ValidationResult], None]] = []

    def _create_default_rules(self) -> List[BaseValidationRule]:
        """Create default validation rules."""
        return [
            SafetyFilterRule(),
            ConsistencyCheckRule(),
            CanonAlignmentRule(),
            ToneMatchRule(),
            CharacterVoiceRule(),
        ]

    def add_rule(self, rule: BaseValidationRule):
        """Add a validation rule."""
        self.rules.append(rule)

    def remove_rule(self, rule_id: str):
        """Remove a rule by ID."""
        self.rules = [r for r in self.rules if r.rule_id != rule_id]

    def get_rule(self, rule_id: str) -> Optional[BaseValidationRule]:
        """Get a rule by ID."""
        for r in self.rules:
            if r.rule_id == rule_id:
                return r
        return None

    def validate(self, context: ValidationContext) -> ValidationResult:
        """
        Run all validation rules on the context.

        Args:
            context: Validation context with output and metadata

        Returns:
            Combined validation result
        """
        combined_result = ValidationResult(passed=True, score=1.0)

        for rule in self.rules:
            if not rule.enabled:
                continue

            try:
                result = rule.validate(context)
                combined_result = combined_result.merge(result)

                # Log rule result
                logger.debug(
                    f"Rule '{rule.rule_id}': passed={result.passed}, "
                    f"issues={len(result.issues)}, score={result.score:.2f}"
                )

            except Exception as e:
                logger.error(f"Validation rule '{rule.rule_id}' failed: {e}")
                # Add error issue
                combined_result.issues.append(ValidationIssue(
                    rule_id=rule.rule_id,
                    category=rule.category,
                    severity=ValidationSeverity.WARNING,
                    message=f"Rule execution failed: {str(e)}"
                ))

        # Recalculate passed based on settings
        if self.require_all_pass:
            combined_result.passed = not combined_result.has_errors
        else:
            combined_result.passed = combined_result.score >= self.min_score_threshold

        # Trigger callbacks
        for callback in self._on_validation_complete:
            try:
                callback(combined_result)
            except Exception as e:
                logger.error(f"Validation callback error: {e}")

        return combined_result

    def validate_with_retry(
        self,
        context: ValidationContext,
        refiner: Optional[Callable[[str, ValidationResult], str]] = None
    ) -> Tuple[ValidationResult, str, int]:
        """
        Validate with automatic retry on failure.

        Args:
            context: Validation context
            refiner: Function that takes (output, result) and returns refined output

        Returns:
            Tuple of (final_result, final_output, retry_count)
        """
        current_output = context.output
        retry_count = 0

        for attempt in range(self.max_retries + 1):
            # Create context with current output
            attempt_context = ValidationContext(
                output=current_output,
                prompt=context.prompt,
                character_name=context.character_name,
                scene_context=context.scene_context,
                game_state=context.game_state,
                established_facts=context.established_facts,
                character_traits=context.character_traits,
                world_rules=context.world_rules,
                tone_descriptors=context.tone_descriptors,
                metadata=context.metadata
            )

            result = self.validate(attempt_context)

            if result.passed:
                return result, current_output, retry_count

            if attempt < self.max_retries and refiner:
                # Trigger refinement
                for callback in self._on_refinement_triggered:
                    try:
                        callback(current_output, result)
                    except Exception as e:
                        logger.error(f"Refinement callback error: {e}")

                # Generate refinement prompt and refine
                try:
                    current_output = refiner(current_output, result)
                    retry_count += 1
                    logger.info(f"Refinement attempt {retry_count}: score improved to {result.score:.2f}")
                except Exception as e:
                    logger.error(f"Refinement failed: {e}")
                    break

        # Return final result after retries
        return result, current_output, retry_count

    def get_refinement_prompt(
        self,
        original_output: str,
        result: ValidationResult
    ) -> str:
        """Generate refinement prompt for failed validation."""
        return self.refinement_prompt.generate(original_output, result)

    def on_validation_complete(self, callback: Callable[[ValidationResult], None]):
        """Register callback for validation completion."""
        self._on_validation_complete.append(callback)

    def on_refinement_triggered(self, callback: Callable[[str, ValidationResult], None]):
        """Register callback for refinement trigger."""
        self._on_refinement_triggered.append(callback)

    def to_config(self) -> Dict[str, Any]:
        """Export pipeline configuration."""
        return {
            "rules": [
                {
                    "rule_id": r.rule_id,
                    "category": r.category.value,
                    "enabled": r.enabled,
                    "description": r.description
                }
                for r in self.rules
            ],
            "max_retries": self.max_retries,
            "require_all_pass": self.require_all_pass,
            "min_score_threshold": self.min_score_threshold
        }


# -----------------------------------------------------------------------------
# Factory Functions
# -----------------------------------------------------------------------------

def create_validation_pipeline(
    include_safety: bool = True,
    include_consistency: bool = True,
    include_canon: bool = True,
    include_tone: bool = True,
    include_character: bool = True,
    max_retries: int = 2,
    min_score: float = 0.7
) -> ValidationPipeline:
    """
    Factory function to create a configured validation pipeline.

    Args:
        include_safety: Include safety filter rule
        include_consistency: Include consistency check rule
        include_canon: Include canon alignment rule
        include_tone: Include tone matching rule
        include_character: Include character voice rule
        max_retries: Maximum refinement retries
        min_score: Minimum quality score threshold

    Returns:
        Configured ValidationPipeline
    """
    rules: List[BaseValidationRule] = []

    if include_safety:
        rules.append(SafetyFilterRule())
    if include_consistency:
        rules.append(ConsistencyCheckRule())
    if include_canon:
        rules.append(CanonAlignmentRule())
    if include_tone:
        rules.append(ToneMatchRule())
    if include_character:
        rules.append(CharacterVoiceRule())

    return ValidationPipeline(
        rules=rules,
        max_retries=max_retries,
        min_score_threshold=min_score
    )


def create_strict_pipeline() -> ValidationPipeline:
    """Create a strict validation pipeline (all rules, no tolerance)."""
    return ValidationPipeline(
        rules=None,  # Use defaults
        max_retries=3,
        require_all_pass=True,
        min_score_threshold=0.9
    )


def create_permissive_pipeline() -> ValidationPipeline:
    """Create a permissive validation pipeline (safety only, high tolerance)."""
    return ValidationPipeline(
        rules=[SafetyFilterRule()],
        max_retries=1,
        require_all_pass=False,
        min_score_threshold=0.5
    )


# -----------------------------------------------------------------------------
# Exports
# -----------------------------------------------------------------------------

__all__ = [
    # Enums
    'ValidationSeverity',
    'ValidationCategory',

    # Core Types
    'ValidationIssue',
    'ValidationResult',
    'ValidationContext',

    # Protocol
    'ValidationRule',

    # Base Class
    'BaseValidationRule',

    # Built-in Rules
    'ConsistencyCheckRule',
    'SafetyFilterRule',
    'CanonAlignmentRule',
    'ToneMatchRule',
    'CharacterVoiceRule',

    # Pipeline
    'RefinementPrompt',
    'ValidationPipeline',

    # Factory Functions
    'create_validation_pipeline',
    'create_strict_pipeline',
    'create_permissive_pipeline',
]
