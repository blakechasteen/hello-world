"""
AutoFix Policy
===============

🐷 Wilbur's Auto-Fix Decision System! 🐷

Configurable policies for when to auto-apply fixes vs. request human review.
Enables domain-specific thresholds (healthcare strict, beekeeping relaxed).

Policy Philosophy:
- CONSERVATIVE: High bar, human review for most fixes (healthcare, finance)
- BALANCED: Reasonable bar, auto-fix obvious issues (default)
- AGGRESSIVE: Lower bar, auto-fix more issues (beekeeping, internal tools)

Decision Matrix:
                       LOW RISK  |  MEDIUM RISK  |  HIGH RISK    |  CRITICAL
    -------------------------------------------------------------------------------
    Very High (≥0.85)  AUTO      |  AUTO*        |  REVIEW       |  REVIEW
    High (0.70-0.85)   AUTO*     |  REVIEW       |  REVIEW       |  MANUAL
    Medium (0.55-0.70) REVIEW    |  REVIEW       |  MANUAL       |  MANUAL
    Low (<0.55)        MANUAL    |  MANUAL       |  MANUAL       |  SKIP

    * Requires test coverage

Moonshot Integration:
- Enables cross-department quality enforcement (Idea #1)
- Supports domain-specific policies (Idea #3)
- Provides confidence-driven escalation (Idea #4)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from .xterminator_types import RiskLevel, FixStrategy


class PolicyProfile(Enum):
    """Pre-configured policy profiles for different domains"""
    CONSERVATIVE = "conservative"  # Healthcare, finance (95% min confidence)
    BALANCED = "balanced"          # Default (85% min confidence)
    AGGRESSIVE = "aggressive"      # Beekeeping, internal (70% min confidence)
    CUSTOM = "custom"              # User-defined thresholds


class FixDecision(Enum):
    """Decision outcomes for fix proposals"""
    AUTO = "auto"          # Auto-apply the fix
    REVIEW = "review"      # Propose fix, await human review
    MANUAL = "manual"      # Flag for manual fixing
    SKIP = "skip"          # Skip (likely false positive)


@dataclass
class AutofixPolicy:
    """
    Policy for auto-fix decisions.

    Configurable thresholds enable domain-specific quality standards.
    """
    profile: PolicyProfile = PolicyProfile.BALANCED

    # Confidence thresholds for auto-fix
    min_confidence_auto: float = 0.85      # Minimum confidence for auto-fix
    min_confidence_review: float = 0.70     # Minimum confidence for review
    min_confidence_manual: float = 0.55     # Below this, flag as manual

    # Risk-based modifiers
    allow_auto_medium_risk: bool = False   # Allow auto-fix for MEDIUM risk?
    allow_auto_high_risk: bool = False     # Allow auto-fix for HIGH risk?
    require_tests_always: bool = True      # Always require tests for auto-fix?

    # Strategy preferences
    prefer_ast_over_template: bool = True  # Prefer AST (safer) over template
    allow_template_autofix: bool = True    # Allow template auto-fixes?

    # Escalation settings
    escalate_on_borderline: bool = True    # Escalate if near threshold?
    borderline_margin: float = 0.05        # Margin for borderline detection

    # Department/domain metadata
    department_name: Optional[str] = None
    domain: Optional[str] = None           # "healthcare", "beekeeping", etc.

    # Learning signals (for Phase 4)
    track_outcomes: bool = True            # Track fix outcomes for learning?
    learning_metadata: dict = field(default_factory=dict)

    @classmethod
    def conservative(cls, department_name: str = "Quality Assurance", domain: str = "healthcare"):
        """
        Conservative policy for high-stakes domains (healthcare, finance).

        - Very high confidence required (≥0.95)
        - Only LOW risk auto-fixed
        - Always requires tests
        - Human review for borderline cases
        """
        return cls(
            profile=PolicyProfile.CONSERVATIVE,
            min_confidence_auto=0.95,
            min_confidence_review=0.85,
            min_confidence_manual=0.70,
            allow_auto_medium_risk=False,
            allow_auto_high_risk=False,
            require_tests_always=True,
            prefer_ast_over_template=True,
            allow_template_autofix=False,  # Only AST for healthcare
            escalate_on_borderline=True,
            borderline_margin=0.03,
            department_name=department_name,
            domain=domain,
            track_outcomes=True
        )

    @classmethod
    def balanced(cls, department_name: str = "Quality Assurance", domain: str = "general"):
        """
        Balanced policy (default).

        - High confidence required (≥0.85)
        - LOW risk auto-fixed, MEDIUM with high confidence
        - Requires tests for auto-fix
        - Review for borderline cases
        """
        return cls(
            profile=PolicyProfile.BALANCED,
            min_confidence_auto=0.85,
            min_confidence_review=0.70,
            min_confidence_manual=0.55,
            allow_auto_medium_risk=True,  # With high confidence
            allow_auto_high_risk=False,
            require_tests_always=True,
            prefer_ast_over_template=True,
            allow_template_autofix=True,
            escalate_on_borderline=True,
            borderline_margin=0.05,
            department_name=department_name,
            domain=domain,
            track_outcomes=True
        )

    @classmethod
    def aggressive(cls, department_name: str = "Quality Assurance", domain: str = "beekeeping"):
        """
        Aggressive policy for lower-stakes domains (beekeeping, internal tools).

        - Medium confidence acceptable (≥0.70)
        - LOW and MEDIUM risk auto-fixed
        - Tests not always required
        - Less escalation
        """
        return cls(
            profile=PolicyProfile.AGGRESSIVE,
            min_confidence_auto=0.70,
            min_confidence_review=0.55,
            min_confidence_manual=0.40,
            allow_auto_medium_risk=True,
            allow_auto_high_risk=False,
            require_tests_always=False,  # More relaxed
            prefer_ast_over_template=False,
            allow_template_autofix=True,
            escalate_on_borderline=False,  # Less escalation
            borderline_margin=0.05,
            department_name=department_name,
            domain=domain,
            track_outcomes=True
        )

    def decide(
        self,
        confidence: float,
        risk_level: RiskLevel,
        fix_strategy: FixStrategy,
        has_tests: bool
    ) -> tuple[FixDecision, str]:
        """
        Decide whether to auto-fix, review, or skip.

        Args:
            confidence: Confidence score (0.0-1.0)
            risk_level: Risk level assessment
            fix_strategy: Selected fix strategy
            has_tests: Whether code has test coverage

        Returns:
            (FixDecision, reason_string)
        """
        # Check for borderline confidence (near threshold)
        is_borderline = self._is_borderline(confidence)

        # Decision logic based on confidence tiers
        if confidence < self.min_confidence_manual:
            return FixDecision.SKIP, f"Confidence too low ({confidence:.2f} < {self.min_confidence_manual})"

        elif confidence < self.min_confidence_review:
            return FixDecision.MANUAL, f"Low confidence ({confidence:.2f}), manual fix recommended"

        elif confidence < self.min_confidence_auto:
            # Review tier
            if is_borderline and self.escalate_on_borderline:
                return FixDecision.REVIEW, f"Borderline confidence ({confidence:.2f}), escalating for review"
            else:
                return FixDecision.REVIEW, f"Medium confidence ({confidence:.2f}), needs review"

        else:
            # High confidence tier - check if we can auto-fix
            can_autofix, reason = self._check_autofix_conditions(
                confidence, risk_level, fix_strategy, has_tests
            )

            if can_autofix:
                return FixDecision.AUTO, reason
            else:
                if is_borderline and self.escalate_on_borderline:
                    return FixDecision.REVIEW, f"High confidence but {reason}, escalating"
                else:
                    return FixDecision.REVIEW, reason

    def _check_autofix_conditions(
        self,
        confidence: float,
        risk_level: RiskLevel,
        fix_strategy: FixStrategy,
        has_tests: bool
    ) -> tuple[bool, str]:
        """
        Check if all conditions for auto-fix are met.

        Returns:
            (can_autofix, reason_string)
        """
        # Check strategy
        if fix_strategy == FixStrategy.SKIP:
            return False, "Strategy is SKIP"

        if fix_strategy == FixStrategy.MANUAL:
            return False, "Strategy requires manual fix"

        if fix_strategy == FixStrategy.TEMPLATE and not self.allow_template_autofix:
            return False, "Template auto-fix not allowed in this policy"

        # Check risk level
        if risk_level == RiskLevel.CRITICAL:
            return False, "CRITICAL risk - always requires review"

        if risk_level == RiskLevel.HIGH and not self.allow_auto_high_risk:
            return False, "HIGH risk not allowed for auto-fix"

        if risk_level == RiskLevel.MEDIUM and not self.allow_auto_medium_risk:
            return False, "MEDIUM risk not allowed for auto-fix"

        # Check test coverage
        if self.require_tests_always and not has_tests:
            return False, "Test coverage required but not present"

        # All conditions met!
        return True, f"Confidence {confidence:.2f}, risk {risk_level.value}, tests {'present' if has_tests else 'not required'}"

    def _is_borderline(self, confidence: float) -> bool:
        """Check if confidence is near a threshold"""
        # Near auto threshold
        if abs(confidence - self.min_confidence_auto) <= self.borderline_margin:
            return True

        # Near review threshold
        if abs(confidence - self.min_confidence_review) <= self.borderline_margin:
            return True

        return False

    def get_statistics_thresholds(self) -> dict:
        """Get thresholds for statistics calculation"""
        return {
            'min_auto': self.min_confidence_auto,
            'min_review': self.min_confidence_review,
            'min_manual': self.min_confidence_manual,
            'profile': self.profile.value,
            'domain': self.domain or 'general'
        }

    def __repr__(self) -> str:
        return (
            f"AutofixPolicy(profile={self.profile.value}, "
            f"domain={self.domain or 'general'}, "
            f"min_auto={self.min_confidence_auto:.2f}, "
            f"require_tests={self.require_tests_always})"
        )


# Convenience functions
def get_policy_for_domain(domain: str) -> AutofixPolicy:
    """
    Get recommended policy for a domain.

    Args:
        domain: Domain name (healthcare, finance, beekeeping, etc.)

    Returns:
        AutofixPolicy configured for the domain
    """
    domain_lower = domain.lower()

    if domain_lower in ['healthcare', 'medical', 'finance', 'banking']:
        return AutofixPolicy.conservative(domain=domain_lower)

    elif domain_lower in ['beekeeping', 'internal', 'dev', 'experimental']:
        return AutofixPolicy.aggressive(domain=domain_lower)

    else:
        return AutofixPolicy.balanced(domain=domain_lower)


def should_autofix_with_policy(
    policy: AutofixPolicy,
    confidence: float,
    risk_level: RiskLevel,
    fix_strategy: FixStrategy,
    has_tests: bool
) -> bool:
    """
    Convenience function to check if fix should be auto-applied.

    Returns:
        True if fix should be auto-applied
    """
    decision, _ = policy.decide(confidence, risk_level, fix_strategy, has_tests)
    return decision == FixDecision.AUTO
