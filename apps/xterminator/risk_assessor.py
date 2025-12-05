"""
Risk Assessor
=============

Assesses risk level of proposed fixes:
- LOW: Safe to autofix (formatting, dead code)
- MEDIUM: Needs review (error handling, refactoring)
- HIGH: Requires careful review (security-adjacent, data handling)
- CRITICAL: Manual only (actual security, breaking changes)

Risk factors:
- Issue category (security > logic > style)
- Context (production > test > example)
- Code complexity (nested > simple)
- Test coverage (untested > tested)
- Scope of change (widespread > localized)
"""

from typing import List, Dict
from .xterminator_types import RiskLevel, CodeContext, ContextType


class RiskAssessor:
    """Assesses risk level for fix proposals"""

    # Category-based base risk levels
    CATEGORY_RISK: Dict[str, RiskLevel] = {
        # CRITICAL - Never autofix
        'security': RiskLevel.CRITICAL,
        'sql_injection': RiskLevel.CRITICAL,
        'xss': RiskLevel.CRITICAL,
        'path_traversal': RiskLevel.CRITICAL,
        'command_injection': RiskLevel.CRITICAL,
        'unsafe_deserialization': RiskLevel.CRITICAL,

        # HIGH - Requires careful review
        'error_handling': RiskLevel.HIGH,
        'resource_management': RiskLevel.HIGH,
        'state_mutation': RiskLevel.HIGH,
        'async_issues': RiskLevel.HIGH,
        'type_confusion': RiskLevel.HIGH,

        # MEDIUM - Needs review
        'logic_error': RiskLevel.MEDIUM,
        'hardcoded_values': RiskLevel.MEDIUM,
        'poor_naming': RiskLevel.MEDIUM,
        'complexity': RiskLevel.MEDIUM,

        # LOW - Safe to autofix
        'copy_paste': RiskLevel.LOW,
        'dead_code': RiskLevel.LOW,
        'unused_import': RiskLevel.LOW,
        'formatting': RiskLevel.LOW,
        'redundant_code': RiskLevel.LOW,
    }

    def __init__(self):
        pass

    def assess_risk(
        self,
        issue_category: str,
        issue_severity: str,
        context: CodeContext,
        code_lines: List[str]
    ) -> tuple[RiskLevel, List[str]]:
        """
        Assess risk level for fixing this issue.

        Args:
            issue_category: Category from Trough detection
            issue_severity: Severity from Trough detection
            context: Code context information
            code_lines: Full file content as lines

        Returns:
            (RiskLevel, List of risk factors)
        """
        risk_factors = []

        # Start with category-based risk
        base_risk = self.CATEGORY_RISK.get(
            issue_category,
            RiskLevel.MEDIUM  # Default to medium if unknown
        )

        # Adjust based on severity
        if issue_severity == 'critical':
            if base_risk in {RiskLevel.LOW, RiskLevel.MEDIUM}:
                base_risk = RiskLevel.HIGH
            risk_factors.append("Critical severity issue")

        # Adjust based on context
        if not context.is_safe_location():
            # Issue in comment/string - probably false positive
            return RiskLevel.CRITICAL, ["Issue not in executable code"]

        # Check if in production-critical path
        if self._is_production_critical(context):
            base_risk = self._escalate_risk(base_risk)
            risk_factors.append("Production-critical code path")

        # Check test coverage
        if not context.has_tests:
            base_risk = self._escalate_risk(base_risk)
            risk_factors.append("No test coverage detected")

        # Check code complexity
        complexity = self._estimate_complexity(context, code_lines)
        if complexity > 5:
            base_risk = self._escalate_risk(base_risk)
            risk_factors.append(f"High complexity (score: {complexity})")

        # Check scope depth
        if context.scope_depth > 2:
            base_risk = self._escalate_risk(base_risk)
            risk_factors.append(f"Deeply nested scope (depth: {context.scope_depth})")

        # Check if in error handling
        if context.in_try_block:
            # Fixing code in try block is riskier
            base_risk = self._escalate_risk(base_risk)
            risk_factors.append("Inside try/except block")

        return base_risk, risk_factors

    def _is_production_critical(self, context: CodeContext) -> bool:
        """Check if code is in production-critical path"""
        critical_indicators = [
            'production',
            'deploy',
            'config',
            'settings',
            'database',
            'auth',
            'payment',
            'security'
        ]

        file_path_lower = context.file_path.lower()
        return any(indicator in file_path_lower for indicator in critical_indicators)

    def _estimate_complexity(
        self,
        context: CodeContext,
        code_lines: List[str]
    ) -> int:
        """
        Estimate code complexity (simplified McCabe).

        Returns:
            Complexity score (higher = more complex)
        """
        complexity = 1  # Base complexity

        # Get relevant lines (context + surrounding)
        start_line = max(0, context.line_number - 10)
        end_line = min(len(code_lines), context.line_number + 10)
        relevant_lines = code_lines[start_line:end_line]

        # Count decision points
        for line in relevant_lines:
            line = line.strip()

            # Conditionals
            if any(keyword in line for keyword in ['if ', 'elif ', 'else:', 'and ', 'or ']):
                complexity += 1

            # Loops
            if any(keyword in line for keyword in ['for ', 'while ']):
                complexity += 1

            # Exception handling
            if any(keyword in line for keyword in ['try:', 'except ', 'finally:']):
                complexity += 1

            # Boolean operators
            complexity += line.count(' and ')
            complexity += line.count(' or ')

        return complexity

    def _escalate_risk(self, current_risk: RiskLevel) -> RiskLevel:
        """Escalate risk by one level"""
        escalation = {
            RiskLevel.LOW: RiskLevel.MEDIUM,
            RiskLevel.MEDIUM: RiskLevel.HIGH,
            RiskLevel.HIGH: RiskLevel.CRITICAL,
            RiskLevel.CRITICAL: RiskLevel.CRITICAL,  # Already max
        }
        return escalation[current_risk]

    def is_safe_to_autofix(
        self,
        risk_level: RiskLevel,
        confidence: float,
        context: CodeContext
    ) -> bool:
        """
        Determine if fix can be applied automatically.

        Criteria:
        - Risk must be LOW
        - Confidence must be ≥0.85
        - Context must be safe location
        - Must have test coverage (preferred)
        """
        return (
            risk_level == RiskLevel.LOW and
            confidence >= 0.85 and
            context.is_safe_location() and
            context.has_tests  # Require tests for auto-fix
        )

    def requires_human_approval(
        self,
        risk_level: RiskLevel,
        confidence: float
    ) -> bool:
        """
        Determine if fix requires human approval.

        Criteria:
        - Risk is HIGH or CRITICAL
        - Confidence is <0.70
        """
        return (
            risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL} or
            confidence < 0.70
        )
