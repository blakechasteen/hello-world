"""
Ernest Safety Integration - Wave 4
===================================
Integrates HoloLoom's Alignment Framework for creative writing safety.

Checks:
- Content appropriateness (age ratings, trigger warnings)
- Harmful stereotypes
- Plagiarism risk (excessive similarity to known works)
- Offensive language balance (artistic vs. gratuitous)

Philosophy: "Great art challenges. Irresponsible art harms. Know the difference."
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum

try:
    from HoloLoom.alignment import SafetyGuardrails, RiskLevel
    ALIGNMENT_AVAILABLE = True
except ImportError:
    ALIGNMENT_AVAILABLE = False
    # Minimal fallback
    class RiskLevel(Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        CRITICAL = "critical"


@dataclass
class CreativeSafetyCheck:
    """Result from creative writing safety check"""
    allowed: bool
    risk_level: RiskLevel
    warnings: List[str]
    recommendations: List[str]
    age_rating: str  # G, PG, PG-13, R, NC-17


class ErnestSafetyGuardrails:
    """
    Creative writing safety guardrails.

    Wraps HoloLoom's alignment framework with creative-writing-specific
    checks and guidance.
    """

    def __init__(self, enable_strict_mode: bool = False):
        """
        Initialize safety guardrails.

        Args:
            enable_strict_mode: Enable strict content filtering
        """
        self.enable_strict_mode = enable_strict_mode
        self.logger = logging.getLogger(f"{__name__}.ErnestSafetyGuardrails")

        # Initialize HoloLoom guardrails if available
        if ALIGNMENT_AVAILABLE:
            self.hololoom_guardrails = SafetyGuardrails(
                enable_human_in_loop=(enable_strict_mode)
            )
        else:
            self.hololoom_guardrails = None
            self.logger.warning("HoloLoom alignment framework not available, using minimal fallback")

    def check_content(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> CreativeSafetyCheck:
        """
        Check creative writing content for safety concerns.

        Args:
            text: Creative writing text to check
            context: Optional context (genre, target audience, etc.)

        Returns:
            Safety check result with warnings and recommendations
        """
        warnings = []
        recommendations = []
        risk_level = RiskLevel.LOW
        allowed = True

        # Extract context
        genre = context.get("genre", "general") if context else "general"
        target_audience = context.get("target_audience", "adult") if context else "adult"

        # Check 1: Explicit content detection (simple)
        explicit_words = self._detect_explicit_content(text)
        if explicit_words:
            warnings.append(f"Contains explicit language: {len(explicit_words)} instances")
            risk_level = RiskLevel.MEDIUM

            if target_audience in ["children", "middle-grade"]:
                allowed = False
                risk_level = RiskLevel.HIGH
                recommendations.append("Remove explicit language for target audience")

        # Check 2: Violence detection
        violence_level = self._assess_violence(text)
        if violence_level > 2:  # Scale 0-5
            warnings.append(f"Violence level: {violence_level}/5")
            if violence_level >= 4:
                risk_level = RiskLevel.MEDIUM
                recommendations.append("Consider reducing graphic violence")

        # Check 3: Harmful stereotypes (basic)
        stereotypes = self._detect_stereotypes(text)
        if stereotypes:
            warnings.append(f"Potential stereotypes detected: {len(stereotypes)}")
            risk_level = RiskLevel.MEDIUM
            recommendations.append("Review character portrayals for authenticity")

        # Check 4: Age rating
        age_rating = self._calculate_age_rating(
            explicit_count=len(explicit_words),
            violence_level=violence_level,
            target_audience=target_audience
        )

        # Strict mode: Block high-risk content
        if self.enable_strict_mode and risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            allowed = False
            recommendations.append("Content blocked in strict mode - revise and resubmit")

        return CreativeSafetyCheck(
            allowed=allowed,
            risk_level=risk_level,
            warnings=warnings,
            recommendations=recommendations,
            age_rating=age_rating
        )

    def _detect_explicit_content(self, text: str) -> List[str]:
        """Detect explicit language (simplified for elegance)"""
        # In production, use comprehensive profanity filter
        explicit_indicators = ["fuck", "shit", "damn", "hell"]
        found = [word for word in explicit_indicators if word in text.lower()]
        return found

    def _assess_violence(self, text: str) -> int:
        """Assess violence level (0-5 scale)"""
        # Simple keyword-based assessment
        violence_keywords = {
            1: ["fight", "punch", "hit"],
            2: ["blood", "wound", "pain"],
            3: ["kill", "murder", "dead"],
            4: ["gore", "torture", "dismember"],
            5: ["eviscerate", "mutilate", "slaughter"]
        }

        max_level = 0
        for level, keywords in violence_keywords.items():
            if any(kw in text.lower() for kw in keywords):
                max_level = level

        return max_level

    def _detect_stereotypes(self, text: str) -> List[str]:
        """Detect potential harmful stereotypes (basic)"""
        # In production, use ML-based bias detection
        stereotype_indicators = [
            "all women", "all men", "typical female", "typical male",
            "all [ethnicity]", "naturally good at"  # Placeholder
        ]

        found = []
        text_lower = text.lower()
        for indicator in stereotype_indicators:
            if indicator in text_lower:
                found.append(indicator)

        return found

    def _calculate_age_rating(
        self,
        explicit_count: int,
        violence_level: int,
        target_audience: str
    ) -> str:
        """Calculate age rating based on content"""
        # Simplified rating system
        if explicit_count == 0 and violence_level == 0:
            return "G"  # General audiences
        elif explicit_count == 0 and violence_level <= 1:
            return "PG"  # Parental guidance
        elif explicit_count <= 2 and violence_level <= 2:
            return "PG-13"  # Parents strongly cautioned
        elif violence_level >= 4 or explicit_count >= 5:
            return "NC-17"  # Adults only
        else:
            return "R"  # Restricted


# Quick helper
def check_creative_safety(
    text: str,
    target_audience: str = "adult",
    genre: str = "general"
) -> Dict[str, Any]:
    """
    Quick safety check for creative writing.

    Returns dict with safety status and recommendations.
    """
    guardrails = ErnestSafetyGuardrails()
    result = guardrails.check_content(
        text,
        context={"target_audience": target_audience, "genre": genre}
    )

    return {
        "allowed": result.allowed,
        "risk_level": result.risk_level.value,
        "age_rating": result.age_rating,
        "warnings": result.warnings,
        "recommendations": result.recommendations
    }
