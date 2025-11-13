"""
Confidence Scorer
=================

Calculates confidence score (0.0-1.0) for fix proposals.

Confidence factors:
- Detection confidence (from Trough)
- Context clarity (executable > comment)
- Code complexity (simple > complex)
- Pattern matching (known pattern > unknown)
- Test coverage (tested > untested)
- Risk level (low risk > high risk)

Formula:
    confidence = weighted_average([
        detection_confidence × 0.35,
        context_score × 0.25,
        complexity_score × 0.15,
        pattern_score × 0.15,
        coverage_score × 0.10
    ])

Then adjusted:
    - Penalty for high risk (-0.10 to -0.30)
    - Penalty for missing tests (-0.05 to -0.15)
    - Bonus for perfect context (+0.05)
"""

from typing import Dict
from .xterminator_types import RiskLevel, CodeContext, ContextType, FixStrategy


class ConfidenceScorer:
    """Calculates confidence scores for fix proposals"""

    def __init__(self):
        # Context type scores
        self.context_scores = {
            ContextType.EXECUTABLE: 1.0,
            ContextType.DEFINITION: 0.9,
            ContextType.IMPORT: 0.8,
            ContextType.ANNOTATION: 0.7,
            ContextType.DOCSTRING: 0.3,
            ContextType.COMMENT: 0.1,
            ContextType.STRING_LITERAL: 0.2,
            ContextType.UNKNOWN: 0.5,
        }

    def calculate_confidence(
        self,
        detection_confidence: float,
        issue_category: str,
        context: CodeContext,
        risk_level: RiskLevel,
        selected_strategy: FixStrategy,
        code_complexity: int = 1
    ) -> tuple[float, Dict[str, float]]:
        """
        Calculate confidence score for fix proposal.

        Args:
            detection_confidence: Confidence from Trough detection (0.0-1.0)
            issue_category: Issue category
            context: Code context
            risk_level: Assessed risk level
            selected_strategy: Selected fix strategy
            code_complexity: Complexity score (1-10+)

        Returns:
            (confidence_score, breakdown_dict)
        """
        factors = {}

        # Factor 1: Detection confidence (35%)
        factors['detection'] = detection_confidence * 0.35

        # Factor 2: Context clarity (25%)
        context_score = self.context_scores.get(context.context_type, 0.5)
        factors['context'] = context_score * 0.25

        # Factor 3: Code complexity (15%)
        # Simpler code = higher confidence
        complexity_score = max(0.0, 1.0 - (code_complexity / 10.0))
        factors['complexity'] = complexity_score * 0.15

        # Factor 4: Pattern matching (15%)
        # Known patterns = higher confidence
        pattern_score = self._calculate_pattern_score(issue_category, selected_strategy)
        factors['pattern'] = pattern_score * 0.15

        # Factor 5: Test coverage (10%)
        coverage_score = 1.0 if context.has_tests else 0.5
        factors['coverage'] = coverage_score * 0.10

        # Base confidence (sum of factors)
        base_confidence = sum(factors.values())

        # Apply adjustments
        adjustments = {}

        # Adjustment 1: Risk penalty
        risk_penalty = self._calculate_risk_penalty(risk_level)
        adjustments['risk_penalty'] = -risk_penalty

        # Adjustment 2: Missing tests penalty
        if not context.has_tests:
            test_penalty = 0.05 if risk_level == RiskLevel.LOW else 0.15
            adjustments['test_penalty'] = -test_penalty

        # Adjustment 3: Perfect context bonus
        if context_score == 1.0 and context.has_tests:
            adjustments['perfect_context_bonus'] = 0.05

        # Adjustment 4: Scope depth penalty (deeply nested = less confident)
        if context.scope_depth > 3:
            depth_penalty = min(0.10, (context.scope_depth - 3) * 0.03)
            adjustments['depth_penalty'] = -depth_penalty

        # Final confidence
        final_confidence = base_confidence + sum(adjustments.values())

        # Clamp to [0.0, 1.0]
        final_confidence = max(0.0, min(1.0, final_confidence))

        # Combine factors and adjustments for breakdown
        breakdown = {**factors, **adjustments, 'final': final_confidence}

        return final_confidence, breakdown

    def _calculate_pattern_score(
        self,
        issue_category: str,
        selected_strategy: FixStrategy
    ) -> float:
        """
        Calculate pattern matching score.

        Well-known issue patterns that map to specific strategies
        get higher scores.
        """
        # Known patterns with proven fix strategies
        known_patterns = {
            ('copy_paste', FixStrategy.AST): 0.90,
            ('dead_code', FixStrategy.AST): 0.95,
            ('unused_import', FixStrategy.AST): 1.0,
            ('error_handling', FixStrategy.TEMPLATE): 0.85,
            ('hardcoded_values', FixStrategy.TEMPLATE): 0.90,
            ('poor_naming', FixStrategy.AST): 0.75,
            ('resource_management', FixStrategy.TEMPLATE): 0.85,
        }

        # Check if this is a known pattern
        key = (issue_category, selected_strategy)
        if key in known_patterns:
            return known_patterns[key]

        # Unknown pattern or manual strategy
        if selected_strategy == FixStrategy.MANUAL:
            return 0.50  # Manual fixes are inherently less confident

        if selected_strategy == FixStrategy.SKIP:
            return 0.30  # Skip means we're not confident about fixing

        # Default for AST/TEMPLATE with unknown category
        return 0.70

    def _calculate_risk_penalty(self, risk_level: RiskLevel) -> float:
        """Calculate confidence penalty based on risk level"""
        penalties = {
            RiskLevel.LOW: 0.00,      # No penalty
            RiskLevel.MEDIUM: 0.10,   # Small penalty
            RiskLevel.HIGH: 0.20,     # Moderate penalty
            RiskLevel.CRITICAL: 0.30, # Large penalty
        }
        return penalties[risk_level]

    def is_high_confidence(self, confidence: float) -> bool:
        """Check if confidence is high (≥0.80)"""
        return confidence >= 0.80

    def is_medium_confidence(self, confidence: float) -> bool:
        """Check if confidence is medium (0.60-0.80)"""
        return 0.60 <= confidence < 0.80

    def is_low_confidence(self, confidence: float) -> bool:
        """Check if confidence is low (<0.60)"""
        return confidence < 0.60

    def confidence_label(self, confidence: float) -> str:
        """Get human-readable confidence label"""
        if confidence >= 0.85:
            return "Very High"
        elif confidence >= 0.70:
            return "High"
        elif confidence >= 0.55:
            return "Medium"
        elif confidence >= 0.40:
            return "Low"
        else:
            return "Very Low"

    def should_autofix(
        self,
        confidence: float,
        risk_level: RiskLevel,
        has_tests: bool
    ) -> bool:
        """
        Determine if fix should be applied automatically.

        Criteria:
        - Confidence ≥0.85
        - Risk is LOW
        - Has test coverage
        """
        return (
            confidence >= 0.85 and
            risk_level == RiskLevel.LOW and
            has_tests
        )

    def format_breakdown(self, breakdown: Dict[str, float]) -> str:
        """Format confidence breakdown for display"""
        lines = ["Confidence Breakdown:"]

        # Factors (positive)
        lines.append("  Factors:")
        for key in ['detection', 'context', 'complexity', 'pattern', 'coverage']:
            if key in breakdown:
                lines.append(f"    {key}: {breakdown[key]:.3f}")

        # Adjustments (positive or negative)
        adjustments = {k: v for k, v in breakdown.items()
                      if k not in ['detection', 'context', 'complexity', 'pattern', 'coverage', 'final']}

        if adjustments:
            lines.append("  Adjustments:")
            for key, value in adjustments.items():
                sign = "+" if value >= 0 else ""
                lines.append(f"    {key}: {sign}{value:.3f}")

        # Final
        lines.append(f"  Final: {breakdown['final']:.3f} ({self.confidence_label(breakdown['final'])})")

        return "\n".join(lines)
