"""
K-12 Safety Guardrails

Multi-layered safety for educational content:
- Content filtering (blocked topics)
- Reading level validation (Flesch-Kincaid)
- Privacy protection (COPPA/FERPA)
- Human-in-the-loop for edge cases

**Implementation Date**: November 15, 2025
"""

from dataclasses import dataclass
from typing import List, Set, Optional, Tuple
import re

try:
    from HoloLoom.alignment import SafetyGuardrails
    HAS_ALIGNMENT = True
except ImportError:
    HAS_ALIGNMENT = False
    print("⚠️  HoloLoom Alignment not available - using basic safety")


@dataclass
class ValidationResult:
    """Safety validation result"""
    allowed: bool
    reason: str = "Safe"
    safety_score: float = 1.0  # 0.0-1.0
    reading_level: Optional[float] = None
    blocked_topics: List[str] = None


class EdWINSafetyLayer:
    """
    K-12 Safety Guardrails

    Multi-layered safety checks:
    1. Pre-filter: Block inappropriate questions
    2. Alignment: HoloLoom safety framework (if available)
    3. Post-filter: Reading level + content filtering
    4. Audit trail: Log all interactions

    Example:
        >>> safety = EdWINSafetyLayer()
        >>>
        >>> result = await safety.validate_response(
        ...     query="Student question",
        ...     response="Generated answer",
        ...     grade_level=8
        ... )
        >>>
        >>> if not result.allowed:
        ...     print(f"Blocked: {result.reason}")
    """

    # K-12 Content Policies
    BLOCKED_TOPICS = {
        "elementary": [  # Grades 4-5
            "violence", "weapons", "adult_content", "alcohol", "drugs",
            "political_controversy", "religious_debate", "hate_speech",
            "self_harm", "dangerous_experiments", "illegal_activities"
        ],
        "middle": [  # Grades 6-8
            "violence", "weapons", "adult_content", "alcohol", "drugs",
            "hate_speech", "self_harm", "dangerous_experiments",
            "illegal_activities", "explicit_political_content"
        ],
        "high": [  # Grades 9-12
            "explicit_violence", "adult_content", "hate_speech",
            "self_harm", "illegal_activities"
        ]
    }

    # PII Patterns (to detect and block)
    PII_PATTERNS = [
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
        r'\b\d{16}\b',  # Credit card
        r'\b[\w\.-]+@[\w\.-]+\.\w+\b',  # Email
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone number
    ]

    def __init__(
        self,
        enable_human_in_loop: bool = False,
        max_reading_level_delta: int = 2
    ):
        """
        Initialize safety layer

        Args:
            enable_human_in_loop: Enable teacher review for edge cases
            max_reading_level_delta: Max grades above student level (default: 2)
        """
        self.enable_human_in_loop = enable_human_in_loop
        self.max_reading_level_delta = max_reading_level_delta

        # HoloLoom alignment framework
        self.guardrails = None
        if HAS_ALIGNMENT:
            from HoloLoom.alignment import create_guardrails
            self.guardrails = create_guardrails(
                enable_human_in_loop=enable_human_in_loop
            )

        # Audit log
        self.audit_log: List[dict] = []

    async def validate_response(
        self,
        query: str,
        response: str,
        grade_level: int
    ) -> ValidationResult:
        """
        Validate response is safe and appropriate

        Multi-layered checks:
        1. Content filtering (blocked topics)
        2. Reading level (Flesch-Kincaid)
        3. PII detection
        4. Alignment framework (if available)

        Args:
            query: Student's question
            response: Generated answer
            grade_level: Student's grade level

        Returns:
            ValidationResult with allowed flag and reason
        """
        # Check 1: Content filtering
        content_result = self._check_content(response, grade_level)
        if not content_result.allowed:
            self._log_audit(query, response, grade_level, content_result)
            return content_result

        # Check 2: Reading level
        reading_level = self.calculate_reading_level(response)
        if reading_level > grade_level + self.max_reading_level_delta:
            result = ValidationResult(
                allowed=False,
                reason=f"Reading level too high ({reading_level:.1f} > {grade_level + self.max_reading_level_delta})",
                safety_score=0.5,
                reading_level=reading_level
            )
            self._log_audit(query, response, grade_level, result)
            return result

        # Check 3: PII detection
        pii_result = self._check_pii(response)
        if not pii_result.allowed:
            self._log_audit(query, response, grade_level, pii_result)
            return pii_result

        # Check 4: Alignment framework (if available)
        if HAS_ALIGNMENT and self.guardrails:
            gate_result = await self.guardrails.gate_action(
                action="respond_to_student",
                context={
                    "query": query,
                    "response": response,
                    "grade_level": grade_level
                }
            )

            if not gate_result.allowed:
                result = ValidationResult(
                    allowed=False,
                    reason=gate_result.reason,
                    safety_score=gate_result.safety_score,
                    reading_level=reading_level
                )
                self._log_audit(query, response, grade_level, result)
                return result

        # All checks passed
        result = ValidationResult(
            allowed=True,
            reason="Safe",
            safety_score=1.0,
            reading_level=reading_level
        )
        self._log_audit(query, response, grade_level, result)
        return result

    def calculate_reading_level(self, text: str) -> float:
        """
        Calculate Flesch-Kincaid Grade Level

        Formula:
        grade = 0.39 * (total_words / total_sentences)
              + 11.8 * (total_syllables / total_words)
              - 15.59

        Args:
            text: Text to analyze

        Returns:
            Float grade level (e.g., 8.5 = 8th grade, 5 months)
        """
        # Simple approximation (syllable count)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.0

        words = text.split()
        if not words:
            return 0.0

        # Approximate syllables (simple heuristic)
        syllables = sum(self._count_syllables(word) for word in words)

        # Flesch-Kincaid formula
        avg_words_per_sentence = len(words) / len(sentences)
        avg_syllables_per_word = syllables / len(words)

        grade_level = (
            0.39 * avg_words_per_sentence +
            11.8 * avg_syllables_per_word -
            15.59
        )

        return max(0.0, grade_level)

    def _check_content(self, text: str, grade_level: int) -> ValidationResult:
        """
        Check content for blocked topics

        Args:
            text: Text to check
            grade_level: Student's grade level

        Returns:
            ValidationResult
        """
        # Determine age band
        if grade_level <= 5:
            age_band = "elementary"
        elif grade_level <= 8:
            age_band = "middle"
        else:
            age_band = "high"

        blocked_topics = self.BLOCKED_TOPICS[age_band]

        # Check for blocked topics
        text_lower = text.lower()
        found_topics = [
            topic for topic in blocked_topics
            if topic.replace('_', ' ') in text_lower
        ]

        if found_topics:
            return ValidationResult(
                allowed=False,
                reason=f"Contains blocked topics: {', '.join(found_topics)}",
                safety_score=0.0,
                blocked_topics=found_topics
            )

        return ValidationResult(allowed=True, reason="Content safe")

    def _check_pii(self, text: str) -> ValidationResult:
        """
        Check for personally identifiable information (PII)

        Args:
            text: Text to check

        Returns:
            ValidationResult
        """
        for pattern in self.PII_PATTERNS:
            if re.search(pattern, text):
                return ValidationResult(
                    allowed=False,
                    reason="Contains potential PII (personal information)",
                    safety_score=0.0
                )

        return ValidationResult(allowed=True, reason="No PII detected")

    def _count_syllables(self, word: str) -> int:
        """
        Count syllables in a word (simple heuristic)

        Args:
            word: Word to count

        Returns:
            Syllable count
        """
        word = word.lower()
        syllables = 0
        vowels = "aeiouy"
        previous_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllables += 1
            previous_was_vowel = is_vowel

        # Adjust for silent 'e'
        if word.endswith('e'):
            syllables -= 1

        # Ensure at least 1 syllable
        return max(1, syllables)

    def _log_audit(
        self,
        query: str,
        response: str,
        grade_level: int,
        result: ValidationResult
    ):
        """Log interaction to audit trail"""
        from datetime import datetime

        self.audit_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "response": response[:200],  # Truncate for storage
            "grade_level": grade_level,
            "allowed": result.allowed,
            "reason": result.reason,
            "safety_score": result.safety_score,
            "reading_level": result.reading_level
        })

    def get_audit_log(self, limit: int = 100) -> List[dict]:
        """Get recent audit log entries"""
        return self.audit_log[-limit:]


# Helper functions

def create_safety_layer(
    enable_human_in_loop: bool = False
) -> EdWINSafetyLayer:
    """
    Create safety layer

    Args:
        enable_human_in_loop: Enable teacher review

    Returns:
        EdWINSafetyLayer
    """
    return EdWINSafetyLayer(enable_human_in_loop=enable_human_in_loop)
