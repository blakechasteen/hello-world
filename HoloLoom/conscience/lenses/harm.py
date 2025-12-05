"""
Harm Lens
=========
Detects potential physical, emotional, and digital harm.

Wraps and extends the SafetyGuardrails system with the elegant
ConscienceLens interface.

Categories of harm detected:
- Physical harm (violence, self-harm, dangerous activities)
- Emotional harm (harassment, manipulation, exploitation)
- Digital harm (malware, data loss, privacy violations)
- Social harm (discrimination, misinformation, deception)

Author: HoloLoom Team
Date: 2025-12-03
"""

import re
import logging
from typing import Dict, Any, List, Optional

from ..judgment import Concern
from .base import BaseLens


logger = logging.getLogger("HoloLoom.conscience.lenses.harm")


# =============================================================================
# Harm Detection Patterns
# =============================================================================

# Patterns indicating potential physical harm
PHYSICAL_HARM_PATTERNS = [
    r"\b(kill|murder|assassinate|execute|eliminate)\b.*\b(person|people|someone|human)",
    r"\b(how to|instructions for|guide to)\b.*\b(weapon|bomb|explosive|poison)",
    r"\b(self[- ]?harm|suicide|cut myself|end my life)\b",
    r"\b(attack|assault|beat|hurt)\b.*\b(someone|person|people)\b",
]

# Patterns indicating potential emotional harm
EMOTIONAL_HARM_PATTERNS = [
    r"\b(harass|bully|stalk|intimidate)\b",
    r"\b(manipulate|deceive|trick)\b.*\b(into|someone)\b",
    r"\b(revenge|get back at|punish)\b.*\b(person|them|someone)\b",
    r"\b(blackmail|extort|threaten)\b",
]

# Patterns indicating potential digital harm
DIGITAL_HARM_PATTERNS = [
    r"\b(delete|destroy|corrupt)\b.*\b(all|system|data|files)\b",
    r"\b(steal|exfiltrate|extract)\b.*\b(password|credential|data|information)\b",
    r"\b(hack|exploit|breach)\b.*\b(system|server|account|database)\b",
    r"\b(malware|ransomware|virus|trojan|keylogger)\b",
    r"\b(ddos|denial of service|flood)\b",
]

# Patterns indicating potential social harm
SOCIAL_HARM_PATTERNS = [
    r"\b(discriminate|racism|sexism|hate)\b",
    r"\b(fake news|misinformation|disinformation|propaganda)\b",
    r"\b(impersonate|pretend to be|pose as)\b.*\b(official|authority|person)\b",
]

# Compile all patterns for efficiency
COMPILED_PATTERNS = {
    "physical": [re.compile(p, re.IGNORECASE) for p in PHYSICAL_HARM_PATTERNS],
    "emotional": [re.compile(p, re.IGNORECASE) for p in EMOTIONAL_HARM_PATTERNS],
    "digital": [re.compile(p, re.IGNORECASE) for p in DIGITAL_HARM_PATTERNS],
    "social": [re.compile(p, re.IGNORECASE) for p in SOCIAL_HARM_PATTERNS],
}


class HarmLens(BaseLens):
    """
    Detects potential harm in actions.

    Examines actions for patterns that could lead to:
    - Physical harm to people
    - Emotional/psychological harm
    - Digital harm (data loss, security breaches)
    - Social harm (discrimination, misinformation)

    Usage:
        lens = HarmLens()
        concerns = await lens.examine(action, context)

    Configuration:
        threshold: Minimum confidence to raise concern (default: 0.5)
        enabled: Whether lens is active (default: True)
        categories: Which harm categories to detect (default: all)

    Integration:
        Can wrap existing SafetyGuardrails for backward compatibility.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        enabled: bool = True,
        categories: Optional[List[str]] = None,
        safety_guardrails: Optional[Any] = None,
    ):
        """
        Initialize HarmLens.

        Args:
            threshold: Minimum confidence to raise concern
            enabled: Whether lens is active
            categories: Which harm categories to detect (default: all)
            safety_guardrails: Optional SafetyGuardrails instance to wrap
        """
        super().__init__(threshold=threshold, enabled=enabled)

        self.categories = categories or ["physical", "emotional", "digital", "social"]
        self._safety_guardrails = safety_guardrails

    @property
    def name(self) -> str:
        return "HarmLens"

    @property
    def category(self) -> str:
        return "harm"

    async def _examine_impl(
        self,
        action: str,
        context: Dict[str, Any]
    ) -> List[Concern]:
        """
        Examine action for potential harm.

        Args:
            action: The action being considered
            context: Context for the action

        Returns:
            List of harm concerns
        """
        concerns: List[Concern] = []

        # Check each enabled category
        for cat in self.categories:
            if cat in COMPILED_PATTERNS:
                cat_concerns = self._check_category(action, cat, context)
                concerns.extend(cat_concerns)

        # If we have a wrapped SafetyGuardrails, use it too
        if self._safety_guardrails:
            guardrail_concerns = await self._check_safety_guardrails(
                action, context
            )
            concerns.extend(guardrail_concerns)

        return concerns

    def _check_category(
        self,
        action: str,
        category: str,
        context: Dict[str, Any]
    ) -> List[Concern]:
        """Check for harm patterns in a specific category."""
        concerns = []

        for pattern in COMPILED_PATTERNS.get(category, []):
            match = pattern.search(action)
            if match:
                # Calculate confidence based on pattern specificity
                # More specific patterns get higher confidence
                pattern_len = len(pattern.pattern)
                confidence = min(0.95, 0.5 + pattern_len / 200)

                # Boost confidence based on context
                if context.get("is_production", False):
                    confidence = min(0.99, confidence + 0.1)

                concerns.append(
                    Concern(
                        lens=self.name,
                        category=f"harm:{category}",
                        description=f"Detected potential {category} harm: {match.group()[:50]}",
                        confidence=confidence,
                        evidence={
                            "pattern": pattern.pattern,
                            "match": match.group(),
                            "position": match.span(),
                        },
                        suggested_mitigation=self._get_mitigation(category),
                    )
                )

        return concerns

    async def _check_safety_guardrails(
        self,
        action: str,
        context: Dict[str, Any]
    ) -> List[Concern]:
        """
        Check action against wrapped SafetyGuardrails.

        Converts SafetyGuardrails decisions to Concerns.
        """
        if not self._safety_guardrails:
            return []

        concerns = []

        try:
            # Import here to avoid circular dependency
            from HoloLoom.alignment.safety_guardrails import (
                ActionRequest,
                ActionCategory,
                RiskLevel,
            )

            # Determine action category from context
            category_str = context.get("category", "query")
            try:
                category = ActionCategory(category_str)
            except ValueError:
                category = ActionCategory.QUERY

            request = ActionRequest(
                action=action,
                category=category,
                context=context,
                user_id=context.get("user_id"),
                session_id=context.get("session_id"),
            )

            # Evaluate with SafetyGuardrails
            decision = self._safety_guardrails.evaluate(request)

            if not decision.allowed or decision.requires_approval:
                # Convert to concern
                risk_to_confidence = {
                    RiskLevel.SAFE: 0.1,
                    RiskLevel.LOW: 0.3,
                    RiskLevel.MEDIUM: 0.5,
                    RiskLevel.HIGH: 0.7,
                    RiskLevel.CRITICAL: 0.9,
                }

                concerns.append(
                    Concern(
                        lens=self.name,
                        category=f"harm:guardrail",
                        description=decision.reason,
                        confidence=risk_to_confidence.get(
                            decision.risk_level, 0.5
                        ),
                        evidence={
                            "risk_level": decision.risk_level.value,
                            "requires_approval": decision.requires_approval,
                            "metadata": decision.metadata,
                        },
                        suggested_mitigation=decision.alternative_action,
                    )
                )

        except ImportError:
            logger.debug("SafetyGuardrails not available, skipping")
        except Exception as e:
            logger.warning(f"Error checking SafetyGuardrails: {e}")

        return concerns

    def _get_mitigation(self, category: str) -> str:
        """Get mitigation suggestion for a harm category."""
        mitigations = {
            "physical": "Refuse to provide information that could cause physical harm",
            "emotional": "Reframe response to be constructive and supportive",
            "digital": "Implement additional security checks and logging",
            "social": "Ensure response is fair, accurate, and unbiased",
        }
        return mitigations.get(category, "Request human review")

    @classmethod
    def wrap_guardrails(cls, guardrails: Any) -> "HarmLens":
        """
        Factory method to wrap existing SafetyGuardrails.

        Args:
            guardrails: SafetyGuardrails instance

        Returns:
            HarmLens wrapping the guardrails
        """
        return cls(safety_guardrails=guardrails)
